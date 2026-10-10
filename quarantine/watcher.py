"""
Quarantine watcher: monitors the quarantine drop directory for new artifacts,
runs the verification/scanning pipeline, and promotes passing artifacts to
the trusted registry via its HTTP API.

Supports:
- Single-file LLM models (.gguf, .safetensors)
- Multi-file diffusion model directories (containing model_index.json)

Everything is fully automatic. Users drop files into quarantine (via UI or CLI)
and the watcher handles scanning, verification, and promotion with zero
manual intervention.
"""

import fcntl
import hashlib
import json
import logging
import os
import shutil
import socket
import stat
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import yaml

from quarantine.audit_chain import AuditChain
from quarantine.pipeline import (
    _open_regular_file,
    inspect_directory_files,
    run_pipeline,
    run_pipeline_directory,
    sha256_of_directory,
)

log = logging.getLogger("quarantine")

QUARANTINE_DIR = Path(os.getenv("QUARANTINE_DIR", "/quarantine"))
REGISTRY_DIR = Path(os.getenv("REGISTRY_DIR", "/registry"))
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://127.0.0.1:8470")
POLICY_PATH = Path(os.getenv("POLICY_PATH", "/etc/secure-ai/policy/policy.yaml"))
AUDIT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", "/var/lib/secure-ai/logs/quarantine-audit.jsonl"))
SERVICE_TOKEN_PATH = os.getenv("SERVICE_TOKEN_PATH", "")
FSVERITY_BIN = os.getenv("FSVERITY_BIN", "/usr/bin/fsverity")
COSIGN_BIN = os.getenv("COSIGN_BIN", "/usr/bin/cosign")

ALLOWED_EXTENSIONS = {".gguf", ".safetensors"}
DENIED_EXTENSIONS = {".pkl", ".pickle", ".pt", ".bin"}
MAX_TOKEN_BYTES = 4096
MAX_SOURCE_METADATA_BYTES = 4096
MAX_POLICY_BYTES = 1024 * 1024
MAX_REGISTRY_RESPONSE_BYTES = 1024 * 1024
MAX_PENDING_RECORD_BYTES = 1024 * 1024
MAX_SCAN_ENTRIES = 128
MAX_PENDING_COMMITS = 32
MAX_QUEUE_ATTEMPTS = 3
RETRY_BASE_SECONDS = 5
RETRY_MAX_SECONDS = 300


def _v2_selected() -> bool:
    mode = os.getenv("QUARANTINE_ADMISSION_MODE", "legacy")
    if mode not in {"legacy", "helper-v2-required"}:
        raise ValueError("unknown quarantine admission mode; no fallback")
    return mode == "helper-v2-required"


def _legacy_only() -> None:
    if _v2_selected():
        from quarantine.admission_controller import (
            RuntimeUnavailable,
            require_qualified_runtime,
        )
        require_qualified_runtime()  # Before any legacy state/configuration work.
        raise RuntimeUnavailable("legacy watcher actions are forbidden in helper-v2-required mode")


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_registry_opener = build_opener(_NoRedirects)


def _validated_registry_base_url() -> str:
    """Validate the configured registry as a credential-free HTTP(S) origin."""
    raw = REGISTRY_URL.strip()
    if raw != REGISTRY_URL or not raw or len(raw.encode("utf-8")) > 2048:
        raise ValueError("REGISTRY_URL is empty, padded, or too long")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in raw):
        raise ValueError("REGISTRY_URL contains control characters")
    parsed = urlparse(raw)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or parsed.params
    ):
        raise ValueError("REGISTRY_URL must be an HTTP(S) origin without credentials")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("REGISTRY_URL contains an invalid port") from exc
    if parsed.scheme.lower() == "http" and not _registry_is_loopback(parsed.hostname):
        raise ValueError("non-loopback registry connections require HTTPS")
    return raw.rstrip("/")


def _registry_is_loopback(hostname: str | None = None) -> bool:
    host = (hostname or urlparse(REGISTRY_URL).hostname or "").lower().rstrip(".")
    if host == "localhost":
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_token(raw: str) -> str:
    encoded = raw.encode("utf-8")
    if not 32 <= len(encoded) <= MAX_TOKEN_BYTES:
        raise ValueError("registry service token must contain 32 to 4096 bytes")
    if any(octet < 0x21 or octet > 0x7E for octet in encoded):
        raise ValueError("registry service token must be printable ASCII without whitespace")
    return raw


def _load_service_token() -> str:
    """Load the service token for authenticated registry promotion.

    Reads from the SERVICE_TOKEN environment variable first.  Falls back to
    reading the file at SERVICE_TOKEN_PATH if set. Secret files must be
    single-link regular files owned by root or this service account. They must
    be owner-only, contain 32--4096 printable ASCII bytes, and have no trailing
    newline.
    """
    token = os.getenv("SERVICE_TOKEN", "")
    if token:
        return _validate_token(token)
    if SERVICE_TOKEN_PATH:
        descriptor = -1
        try:
            descriptor = os.open(
                SERVICE_TOKEN_PATH,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o077
                or metadata.st_uid not in {0, os.geteuid()}
                or not 32 <= metadata.st_size <= MAX_TOKEN_BYTES
            ):
                raise ValueError("registry token file is not a secure regular file")
            raw = os.read(descriptor, MAX_TOKEN_BYTES + 1)
            if len(raw) != metadata.st_size:
                raise ValueError("registry token file changed while reading")
            return _validate_token(raw.decode("utf-8", errors="strict"))
        except (OSError, UnicodeDecodeError, ValueError) as e:
            raise RuntimeError("configured registry service token is unavailable") from e
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    return ""


def _http_urlopen(target, timeout: int = 30):
    """Open only the configured registry origin and never follow redirects."""
    raw_url = target.full_url if isinstance(target, Request) else str(target)
    base_url = _validated_registry_base_url()
    parsed = urlparse(raw_url)
    base = urlparse(base_url)
    if (
        parsed.scheme.lower(),
        parsed.hostname,
        parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
    ) != (
        base.scheme.lower(),
        base.hostname,
        base.port or (443 if base.scheme.lower() == "https" else 80),
    ):
        raise URLError("registry request target does not match configured origin")
    return _registry_opener.open(target, timeout=timeout)  # nosec B310

# Hash-chained audit log instance
_audit_chain = AuditChain(str(AUDIT_LOG_PATH))


def audit_log(event: str, filename: str, **kwargs):
    """Append a hash-chained audit entry to the quarantine audit log."""
    _audit_chain.append(event, {"filename": filename, **kwargs})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    handle, expected_size = _open_regular_file(path)
    bytes_read = 0
    with handle as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            bytes_read += len(chunk)
            h.update(chunk)
    if bytes_read != expected_size:
        raise ValueError("artifact changed while hashing")
    return h.hexdigest()


def load_policy() -> dict:
    if not POLICY_PATH.exists():
        raise RuntimeError(f"policy file is required: {POLICY_PATH}")
    try:
        raw = _read_secure_sidecar(POLICY_PATH, max_bytes=MAX_POLICY_BYTES)
        policy = yaml.safe_load(raw.decode("utf-8", errors="strict")) or {}
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        raise RuntimeError("policy file is unavailable or invalid") from exc
    if not isinstance(policy, dict):
        raise RuntimeError("policy file must contain a mapping")
    return policy


def _read_secure_sidecar(path: Path, *, max_bytes: int) -> bytes:
    """Read a bounded, single-link regular configuration/metadata file."""
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o022
            or metadata.st_size > max_bytes
        ):
            raise ValueError("sidecar is not a bounded single-link regular file")
        raw = os.read(descriptor, max_bytes + 1)
        if len(raw) != metadata.st_size:
            raise ValueError("sidecar changed while reading")
        return raw
    finally:
        os.close(descriptor)


def _artifact_identity(path: Path) -> tuple[int, int, int, int, int]:
    """Capture metadata used to detect replacement/mutation during scanning."""
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError("artifact must be a single-link regular file")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_artifact_name(filename: str) -> str:
    if (
        not filename
        or filename in {".", ".."}
        or Path(filename).name != filename
        or filename.startswith(".")
        or len(filename.encode("utf-8")) > 255
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in filename)
    ):
        raise ValueError("artifact filename is unsafe")
    return filename


def _path_lexists(path: Path) -> bool:
    """Test for a directory entry without following a final symlink."""
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def model_name_from_filename(filename: str) -> str:
    """Derive a human-readable model name from filename."""
    stem = Path(filename).stem
    for suffix in [".Q4_K_M", ".Q5_K_M", ".Q8_0", ".Q4_0", ".Q6_K", ".f16", ".f32"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _read_source_metadata(artifact_path: Path) -> str:
    """Read source URL from .source metadata file if present.

    When a model is downloaded via the UI's one-click download, a companion
    .source file is written alongside containing the origin URL. This lets
    the pipeline verify the source against the allowlist.
    """
    source_file = artifact_path.parent / f".{artifact_path.name}.source"
    if _path_lexists(source_file):
        try:
            raw = _read_secure_sidecar(
                source_file, max_bytes=MAX_SOURCE_METADATA_BYTES
            )
            source = raw.decode("utf-8", errors="strict").strip()
            if any(
                ord(character) < 0x20 or ord(character) == 0x7F
                for character in source
            ):
                raise ValueError("source metadata contains control characters")
            return source
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("source metadata is unavailable or unsafe") from exc
    return ""


def _extract_scanner_versions(scan_results: dict) -> dict:
    """Extract scanner version info from pipeline scan result details."""
    versions = {}
    for key, value in scan_results.items():
        if isinstance(value, dict):
            ver = value.get("scanner_version")
            if ver:
                versions[key] = ver
        elif isinstance(value, str):
            # Try to parse stringified dicts (from {k: str(v)} conversions)
            pass
    return versions


def _compute_policy_version() -> dict:
    """Read the policy file and return its hash as a version identifier."""
    if not POLICY_PATH.exists():
        return {"hash": "none", "note": "no policy file"}
    try:
        content = _read_secure_sidecar(POLICY_PATH, max_bytes=MAX_POLICY_BYTES)
        return {"hash": hashlib.sha256(content).hexdigest()}
    except (OSError, ValueError):
        return {"hash": "unreadable"}


def _stage_gguf_guard_manifest(pipeline_details: dict | None) -> Path | None:
    """Move a generated GGUF guard manifest into the registry directory."""
    if not pipeline_details:
        return None
    manifest_info = pipeline_details.get("gguf_guard_manifest", {})
    if not isinstance(manifest_info, dict) or not manifest_info.get("generated"):
        return None
    manifest_path = manifest_info.get("manifest_path")
    if not manifest_path:
        raise ValueError("generated GGUF guard manifest has no path")

    source = Path(manifest_path)
    _validate_artifact_name(source.name)
    dest = REGISTRY_DIR / source.name
    try:
        if source.resolve() != dest.resolve():
            if _path_lexists(dest):
                raise FileExistsError(f"registry manifest already exists: {dest.name}")
            handle, _size = _open_regular_file(source)
            handle.close()
            shutil.move(str(source), str(dest))
            log.info("moved gguf-guard manifest to registry dir: %s", dest.name)
        handle, _size = _open_regular_file(dest)
        handle.close()
    except OSError as exc:
        raise RuntimeError("could not securely stage GGUF guard manifest") from exc

    manifest_info["manifest_path"] = dest.name
    return dest


def _promotion_id(filename: str, file_hash: str, model_type: str) -> str:
    canonical = f"secai-promotion-v1\0{model_type}\0{filename}\0{file_hash}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_registry_payload(
    filename: str,
    file_hash: str,
    size_bytes: int,
    scan_results: dict,
    *,
    model_type: str,
    source_url: str,
    pipeline_details: dict | None,
    evidence: list[dict],
) -> dict:
    name = model_name_from_filename(filename)
    scanner_versions = _extract_scanner_versions(pipeline_details or {})
    source_revision = ""
    if source_url and "huggingface.co" in source_url:
        import re

        rev_match = re.search(r"/resolve/([^/]+)/", source_url)
        if rev_match:
            source_revision = rev_match.group(1)

    payload = {
        "schema_version": 1,
        "name": name,
        "filename": filename,
        "sha256": file_hash,
        "size_bytes": size_bytes,
        "model_type": model_type,
        "scan_results": {k: str(v) for k, v in scan_results.items()},
        "source": source_url,
        "source_revision": source_revision,
        "scanner_versions": scanner_versions,
        "policy_version": _compute_policy_version(),
        "evidence": evidence,
    }

    if pipeline_details:
        fingerprint = pipeline_details.get("gguf_guard_fingerprint")
        if fingerprint:
            payload["gguf_guard_fingerprint"] = fingerprint
        manifest_info = pipeline_details.get("gguf_guard_manifest", {})
        if isinstance(manifest_info, dict):
            manifest_path = manifest_info.get("manifest_path", "")
            if manifest_info.get("generated") and manifest_path:
                payload["gguf_guard_manifest"] = Path(manifest_path).name
    return payload


def _evidence_descriptor(path: Path) -> dict:
    name = _validate_artifact_name(path.name)
    handle, expected_size = _open_regular_file(path)
    digest = hashlib.sha256()
    bytes_read = 0
    with handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            bytes_read += len(chunk)
    if bytes_read != expected_size:
        raise ValueError("promotion evidence changed while hashing")
    return {
        "filename": name,
        "sha256": digest.hexdigest(),
        "size_bytes": expected_size,
    }


def _promotion_journal_dir() -> Path:
    path = REGISTRY_DIR / ".promotion-journal"
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid not in {0, os.geteuid()}
        or metadata.st_mode & 0o077
    ):
        raise RuntimeError("promotion journal directory is insecure")
    return path


def _journal_path(promotion_id: str) -> Path:
    if len(promotion_id) != 64 or any(c not in "0123456789abcdef" for c in promotion_id):
        raise ValueError("promotion identifier is invalid")
    return _promotion_journal_dir() / f"{promotion_id}.json"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_file(path: Path, content: bytes, *, mode: int = 0o600) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_pending_record(record: dict, *, replace: bool = False) -> Path:
    raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(raw) > MAX_PENDING_RECORD_BYTES:
        raise ValueError("promotion pending record exceeds safety limit")
    path = _journal_path(record["promotion_id"])
    if replace:
        _atomic_replace_file(path, raw)
    else:
        _atomic_create_file(path, raw)
    return path


def _read_promotion_record(path: Path) -> dict:
    raw = _read_secure_sidecar(path, max_bytes=MAX_PENDING_RECORD_BYTES)
    record = json.loads(raw.decode("utf-8", errors="strict"))
    if not isinstance(record, dict):
        raise ValueError("pending promotion record is not an object")
    required = {
        "schema_version",
        "state",
        "promotion_id",
        "artifact",
        "payload",
        "evidence",
        "attempts",
        "next_attempt_at",
        "created_at",
    }
    if (
        set(record) != required
        or record["schema_version"] != 1
        or record["state"] not in {"staging", "pending"}
    ):
        raise ValueError("promotion record has an unsupported schema")
    if path != _journal_path(record["promotion_id"]):
        raise ValueError("pending promotion record identifier mismatch")
    if not isinstance(record["payload"], dict) or not isinstance(record["evidence"], list):
        raise ValueError("pending promotion record payload is invalid")
    if not isinstance(record["attempts"], int) or record["attempts"] < 0:
        raise ValueError("pending promotion attempt count is invalid")
    if not isinstance(record["next_attempt_at"], (int, float)):
        raise ValueError("pending promotion retry timestamp is invalid")
    return record


def _read_pending_record(path: Path) -> dict:
    record = _read_promotion_record(path)
    if record["state"] != "pending":
        raise ValueError("promotion evidence is not fully staged")
    return record


def _staging_evidence_names(filename: str, pipeline_details: dict) -> list[str]:
    names = {
        _validate_artifact_name(f"{filename}.provenance.json"),
        _validate_artifact_name(f"{filename}.provenance.json.sig"),
    }
    manifest_info = pipeline_details.get("gguf_guard_manifest", {})
    if isinstance(manifest_info, dict) and manifest_info.get("generated"):
        manifest_path = manifest_info.get("manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path:
            raise ValueError("generated GGUF guard manifest has no path")
        names.add(_validate_artifact_name(Path(manifest_path).name))
    return sorted(names)


def _begin_promotion_staging(
    filename: str,
    file_hash: str,
    size_bytes: int,
    model_type: str,
    pipeline_details: dict,
) -> Path:
    """Persist recovery intent before an accepted artifact leaves quarantine."""
    filename = _validate_artifact_name(filename)
    promotion_id = _promotion_id(filename, file_hash, model_type)
    evidence_names = _staging_evidence_names(filename, pipeline_details)
    for evidence_name in evidence_names:
        if _path_lexists(REGISTRY_DIR / evidence_name):
            raise FileExistsError(f"registry evidence already exists: {evidence_name}")
    record = {
        "schema_version": 1,
        "state": "staging",
        "promotion_id": promotion_id,
        "artifact": {
            "filename": filename,
            "sha256": file_hash,
            "size_bytes": size_bytes,
            "model_type": model_type,
        },
        "payload": {},
        "evidence": evidence_names,
        "attempts": 0,
        "next_attempt_at": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return _write_pending_record(record)


def _verify_promotion_bundle(record: dict) -> None:
    artifact = record.get("artifact")
    if not isinstance(artifact, dict) or set(artifact) != {
        "filename",
        "sha256",
        "size_bytes",
        "model_type",
    }:
        raise ValueError("pending promotion artifact record is invalid")
    filename = _validate_artifact_name(artifact["filename"])
    expected_hash = artifact["sha256"]
    expected_size = artifact["size_bytes"]
    model_type = artifact["model_type"]
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(c not in "0123456789abcdef" for c in expected_hash)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or model_type not in {"llm", "diffusion"}
    ):
        raise ValueError("pending promotion artifact identity is invalid")

    artifact_path = REGISTRY_DIR / filename
    if model_type == "diffusion":
        actual_hash = sha256_of_directory(artifact_path)
        actual_size = sum(path.lstat().st_size for path in inspect_directory_files(artifact_path))
    else:
        actual_hash = sha256_file(artifact_path)
        actual_size = artifact_path.lstat().st_size
    if actual_hash != expected_hash or actual_size != expected_size:
        raise ValueError("pending promotion artifact no longer matches its verdict")

    evidence = record["evidence"]
    if not evidence:
        raise ValueError("pending promotion has no evidence")
    evidence_names = set()
    for descriptor in evidence:
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "filename",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("pending promotion evidence descriptor is invalid")
        evidence_path = REGISTRY_DIR / _validate_artifact_name(descriptor["filename"])
        actual = _evidence_descriptor(evidence_path)
        if actual != descriptor:
            raise ValueError("pending promotion evidence changed before registry commit")
        if descriptor["filename"] in evidence_names:
            raise ValueError("pending promotion contains duplicate evidence")
        evidence_names.add(descriptor["filename"])

    provenance_name = f"{filename}.provenance.json"
    if provenance_name not in evidence_names:
        raise ValueError("pending promotion is missing provenance evidence")
    provenance_path = REGISTRY_DIR / provenance_name
    provenance = json.loads(
        _read_secure_sidecar(provenance_path, max_bytes=MAX_PENDING_RECORD_BYTES).decode(
            "utf-8", errors="strict"
        )
    )
    if not isinstance(provenance, dict) or provenance.get("artifact") != {
        "filename": filename,
        "sha256": expected_hash,
        "size_bytes": expected_size,
    }:
        raise ValueError("pending promotion provenance does not bind the artifact")

    payload = record["payload"]
    if (
        payload.get("filename") != filename
        or payload.get("sha256") != expected_hash
        or payload.get("size_bytes") != expected_size
        or payload.get("model_type") != model_type
        or payload.get("evidence") != evidence
    ):
        raise ValueError("pending promotion payload does not bind the staged bundle")
    guard_manifest = payload.get("gguf_guard_manifest")
    if guard_manifest is not None and (
        not isinstance(guard_manifest, str) or guard_manifest not in evidence_names
    ):
        raise ValueError("pending promotion payload references missing GGUF guard evidence")


def _remove_pending_record(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


def promote_to_registry(filename: str, file_hash: str, size_bytes: int,
                        scan_results: dict, model_type: str = "llm",
                        source_url: str = "",
                        pipeline_details: dict | None = None,
                        *,
                        promotion_id: str = "",
                        evidence: list[dict] | None = None,
                        payload_override: dict | None = None) -> bool:
    """Idempotently commit a fully staged promotion bundle to the registry."""
    _legacy_only()
    try:
        filename = _validate_artifact_name(filename)
        registry_base = _validated_registry_base_url()
        token = _load_service_token()
    except (RuntimeError, ValueError) as exc:
        log.error("registry promotion configuration is invalid: %s", exc)
        return False
    if not token and not (
        os.getenv("SECAI_ALLOW_INSECURE_REGISTRY", "") == "1"
        and _registry_is_loopback()
    ):
        log.error("registry authentication is required for promotion")
        return False

    if not promotion_id:
        promotion_id = _promotion_id(filename, file_hash, model_type)
    if evidence is None:
        log.error("registry commit refused because promotion evidence was not staged")
        return False
    payload = payload_override or _build_registry_payload(
        filename,
        file_hash,
        size_bytes,
        scan_results,
        model_type=model_type,
        source_url=source_url,
        pipeline_details=pipeline_details,
        evidence=evidence,
    )

    try:
        record = _read_pending_record(_journal_path(promotion_id))
        expected_artifact = {
            "filename": filename,
            "sha256": file_hash,
            "size_bytes": size_bytes,
            "model_type": model_type,
        }
        if (
            record["artifact"] != expected_artifact
            or record["payload"] != payload
            or record["evidence"] != evidence
        ):
            raise ValueError("registry commit does not match its durable pending record")
        _verify_promotion_bundle(record)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        log.error(
            "registry commit refused because pending evidence is invalid: %s",
            type(exc).__name__,
        )
        return False

    try:
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": promotion_id,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = Request(
            f"{registry_base}/v1/model/promote",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with _http_urlopen(req, timeout=30) as resp:
            raw = resp.read(MAX_REGISTRY_RESPONSE_BYTES + 1)
            if len(raw) > MAX_REGISTRY_RESPONSE_BYTES:
                raise ValueError("registry response exceeded safety limit")
            result = json.loads(raw.decode("utf-8", errors="strict"))
            if not isinstance(result, dict):
                raise ValueError("registry response was not a JSON object")
            log.info("registry promotion returned HTTP %s", resp.status)
            return resp.status in {200, 201}
    except (OSError, URLError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as e:
        log.error("registry promotion request failed: %s", type(e).__name__)
        return False
    except Exception as e:
        log.error("unexpected error during promotion: %s", type(e).__name__)
        return False


def _cleanup_source_metadata(artifact_path: Path) -> None:
    source_meta = artifact_path.parent / f".{artifact_path.name}.source"
    try:
        metadata = source_meta.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        source_meta.unlink(missing_ok=True)


def _rollback_promotion(dest: Path, original: Path) -> None:
    """Ensure an unregistered artifact is removed from the trusted namespace."""
    if not _path_lexists(dest):
        return
    try:
        if not _path_lexists(original):
            shutil.move(str(dest), str(original))
            log.warning("rolled unregistered artifact back to quarantine: %s", original.name)
            return
        failed_path = original.parent / f".{original.name}.promotion-failed"
        if not _path_lexists(failed_path):
            shutil.move(str(dest), str(failed_path))
            log.warning("moved unregistered artifact to quarantine hold: %s", failed_path.name)
            return
    except OSError:
        log.exception("could not roll back unregistered artifact %s", dest.name)

    # Last resort: never leave an unregistered object in the trusted registry.
    try:
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink(missing_ok=True)
    except OSError:
        log.critical("manual intervention required: unregistered artifact remains at %s", dest)


def _source_log_label(source_url: str) -> str:
    if not source_url:
        return "local"
    return (urlparse(source_url).hostname or "invalid-source").lower()


def process_artifact(artifact_path: Path) -> bool:
    """Run the explicitly selected admission path; v2 has no legacy fallback."""
    if _v2_selected():
        from quarantine.admission_controller import process_artifact as process_v2
        return process_v2(artifact_path)
    _validate_artifact_name(artifact_path.name)
    initial_identity = _artifact_identity(artifact_path)
    log.info("processing: %s", artifact_path.name)

    ext = artifact_path.suffix.lower()
    if ext in DENIED_EXTENSIONS:
        log.warning("REJECTED (denied format): %s", artifact_path.name)
        audit_log("rejected", artifact_path.name, reason="denied_format", extension=ext)
        if _artifact_identity(artifact_path) == initial_identity:
            artifact_path.unlink()
            _cleanup_source_metadata(artifact_path)
        return False

    if ext not in ALLOWED_EXTENSIONS:
        log.warning("REJECTED (unknown format): %s", artifact_path.name)
        audit_log("rejected", artifact_path.name, reason="unknown_format", extension=ext)
        if _artifact_identity(artifact_path) == initial_identity:
            artifact_path.unlink()
            _cleanup_source_metadata(artifact_path)
        return False

    file_hash = sha256_file(artifact_path)
    file_size = initial_identity[2]
    source_url = _read_source_metadata(artifact_path)
    log.info(
        "sha256: %s  size: %d bytes  source-host: %s",
        file_hash,
        file_size,
        _source_log_label(source_url),
    )

    result = run_pipeline(artifact_path, file_hash, load_policy(), source_url=source_url)

    if not result["passed"]:
        log.warning("REJECTED (%s): %s", result["reason"], artifact_path.name)
        audit_log(
            "rejected", artifact_path.name,
            reason=result["reason"],
            sha256=file_hash,
            size_bytes=file_size,
            details={k: str(v) for k, v in result.get("details", {}).items()},
        )
        if _artifact_identity(artifact_path) == initial_identity:
            artifact_path.unlink()
            _cleanup_source_metadata(artifact_path)
        else:
            log.error("artifact changed during rejection; refusing to delete it")
        return False

    if _artifact_identity(artifact_path) != initial_identity:
        audit_log("rejected", artifact_path.name, reason="artifact_changed_during_scan")
        log.error("artifact changed during scan; refusing promotion: %s", artifact_path.name)
        return False

    # Move file to registry directory without replacing a trusted artifact.
    details = result.get("details", {})
    dest = REGISTRY_DIR / artifact_path.name
    if _path_lexists(dest):
        audit_log("rejected", artifact_path.name, reason="registry_name_collision")
        log.error("registry destination already exists: %s", dest.name)
        return False
    staging_path: Path | None = None
    try:
        staging_path = _begin_promotion_staging(
            artifact_path.name,
            file_hash,
            file_size,
            "llm",
            details,
        )
        shutil.move(str(artifact_path), str(dest))
    except (OSError, RuntimeError, ValueError):
        log.exception("could not begin durable promotion staging: %s", artifact_path.name)
        if staging_path is not None and _path_lexists(artifact_path):
            _remove_pending_record(staging_path)
        audit_log("rejected", artifact_path.name, reason="promotion_staging_failed")
        return False
    log.info("moved to registry dir: %s", dest)

    try:
        if sha256_file(dest) != file_hash:
            raise ValueError("artifact changed between scan and registry move")
    except (OSError, ValueError):
        _rollback_promotion(dest, artifact_path)
        if staging_path is not None:
            _remove_pending_record(staging_path)
        audit_log("rejected", artifact_path.name, reason="artifact_changed_during_move")
        return False

    # Enable fs-verity on the promoted file (before provenance manifest)
    fsverity_ok = _enable_fsverity(dest)

    # Collect scan result summary
    scan_summary = _build_scan_summary(details)

    # Extract source revision
    source_revision = ""
    if source_url and "huggingface.co" in source_url:
        import re
        rev_match = re.search(r"/resolve/([^/]+)/", source_url)
        if rev_match:
            source_revision = rev_match.group(1)

    staged_evidence: list[Path] = []
    try:
        record, pending_path, staged_evidence = _stage_pending_promotion(
            dest,
            artifact_path.name,
            file_hash,
            file_size,
            source_url,
            source_revision,
            scan_summary,
            details,
            model_type="llm",
            fsverity_success=fsverity_ok,
        )
        _cleanup_source_metadata(artifact_path)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        log.exception("could not durably stage promotion bundle: %s", artifact_path.name)
        for evidence_path in staged_evidence:
            evidence_path.unlink(missing_ok=True)
        _rollback_promotion(dest, artifact_path)
        if staging_path is not None and _path_lexists(staging_path):
            _remove_pending_record(staging_path)
        audit_log("rejected", artifact_path.name, reason="promotion_staging_failed")
        return False

    if promote_to_registry(
        artifact_path.name,
        file_hash,
        file_size,
        scan_summary,
        source_url=source_url,
        pipeline_details=details,
        promotion_id=record["promotion_id"],
        evidence=record["evidence"],
        payload_override=record["payload"],
    ):
        log.info("PROMOTED: %s (registered in manifest)", artifact_path.name)
        audit_log("promoted", artifact_path.name, sha256=file_hash,
                  size_bytes=file_size, scan_summary=scan_summary)
        _remove_pending_record(pending_path)
        _cleanup_source_metadata(artifact_path)
        return True
    else:
        log.warning("registry commit pending durable retry: %s", artifact_path.name)
        audit_log("promotion_pending", artifact_path.name, sha256=file_hash,
                  size_bytes=file_size, promotion_id=record["promotion_id"])
        return False


def process_directory(artifact_dir: Path) -> bool:
    """Run the full pipeline on a multi-file diffusion model directory.
    Returns True if promoted.
    """
    _legacy_only()
    _validate_artifact_name(artifact_dir.name)
    artifact_files = inspect_directory_files(artifact_dir)
    log.info("processing directory: %s", artifact_dir.name)

    # Compute deterministic hash of the entire directory
    dir_hash = sha256_of_directory(artifact_dir)
    source_url = _read_source_metadata(artifact_dir)
    log.info(
        "directory hash: %s  source-host: %s",
        dir_hash,
        _source_log_label(source_url),
    )

    # Calculate total size
    total_size = sum(f.lstat().st_size for f in artifact_files)

    result = run_pipeline_directory(artifact_dir, dir_hash, load_policy(), source_url=source_url)

    if not result["passed"]:
        log.warning("REJECTED (%s): %s", result["reason"], artifact_dir.name)
        audit_log(
            "rejected", artifact_dir.name,
            reason=result["reason"],
            sha256=dir_hash,
            size_bytes=total_size,
            model_type="diffusion",
            details={k: str(v) for k, v in result.get("details", {}).items()},
        )
        try:
            unchanged = sha256_of_directory(artifact_dir) == dir_hash
        except (OSError, ValueError):
            unchanged = False
        if unchanged:
            shutil.rmtree(artifact_dir)
            _cleanup_source_metadata(artifact_dir)
        else:
            log.error("artifact directory changed during rejection; refusing to delete it")
        return False

    if sha256_of_directory(artifact_dir) != dir_hash:
        audit_log("rejected", artifact_dir.name, reason="artifact_changed_during_scan")
        log.error("artifact directory changed during scan; refusing promotion")
        return False

    # Move directory to registry without replacing a trusted artifact.
    details = result.get("details", {})
    dest = REGISTRY_DIR / artifact_dir.name
    if _path_lexists(dest):
        audit_log("rejected", artifact_dir.name, reason="registry_name_collision")
        log.error("registry destination already exists: %s", dest.name)
        return False
    staging_path = None
    try:
        staging_path = _begin_promotion_staging(
            artifact_dir.name,
            dir_hash,
            total_size,
            "diffusion",
            details,
        )
        shutil.move(str(artifact_dir), str(dest))
    except (OSError, RuntimeError, ValueError):
        log.exception("could not begin durable promotion staging: %s", artifact_dir.name)
        if staging_path is not None and _path_lexists(artifact_dir):
            _remove_pending_record(staging_path)
        audit_log("rejected", artifact_dir.name, reason="promotion_staging_failed")
        return False
    log.info("moved to registry dir: %s", dest)

    try:
        if sha256_of_directory(dest) != dir_hash:
            raise ValueError("artifact directory changed during registry move")
    except (OSError, ValueError):
        _rollback_promotion(dest, artifact_dir)
        if staging_path is not None:
            _remove_pending_record(staging_path)
        audit_log("rejected", artifact_dir.name, reason="artifact_changed_during_move")
        return False

    # Enable fs-verity on individual model files in the directory
    fsverity_ok = True
    verity_extensions = {".safetensors", ".json"}
    for f in inspect_directory_files(dest):
        if f.suffix in verity_extensions:
            if not _enable_fsverity(f):
                fsverity_ok = False

    scan_summary = _build_scan_summary(details)
    scan_summary["model_type"] = "diffusion"

    # Extract source revision
    source_revision = ""
    if source_url and "huggingface.co" in source_url:
        import re
        rev_match = re.search(r"/resolve/([^/]+)/", source_url)
        if rev_match:
            source_revision = rev_match.group(1)

    staged_evidence = []
    try:
        record, pending_path, staged_evidence = _stage_pending_promotion(
            dest,
            artifact_dir.name,
            dir_hash,
            total_size,
            source_url,
            source_revision,
            scan_summary,
            details,
            model_type="diffusion",
            fsverity_success=fsverity_ok,
        )
        _cleanup_source_metadata(artifact_dir)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        log.exception("could not durably stage promotion bundle: %s", artifact_dir.name)
        for evidence_path in staged_evidence:
            evidence_path.unlink(missing_ok=True)
        _rollback_promotion(dest, artifact_dir)
        if staging_path is not None and _path_lexists(staging_path):
            _remove_pending_record(staging_path)
        audit_log("rejected", artifact_dir.name, reason="promotion_staging_failed")
        return False

    if promote_to_registry(
        artifact_dir.name,
        dir_hash,
        total_size,
        scan_summary,
        model_type="diffusion",
        source_url=source_url,
        pipeline_details=details,
        promotion_id=record["promotion_id"],
        evidence=record["evidence"],
        payload_override=record["payload"],
    ):
        log.info("PROMOTED: %s (diffusion model registered)", artifact_dir.name)
        audit_log("promoted", artifact_dir.name, sha256=dir_hash,
                  size_bytes=total_size, model_type="diffusion", scan_summary=scan_summary)
        _remove_pending_record(pending_path)
        _cleanup_source_metadata(artifact_dir)
        return True
    else:
        log.warning("registry commit pending durable retry: %s", artifact_dir.name)
        audit_log("promotion_pending", artifact_dir.name, sha256=dir_hash,
                  size_bytes=total_size, promotion_id=record["promotion_id"])
        return False


def _enable_fsverity(filepath: Path) -> bool:
    """Enable fs-verity on a promoted model file for kernel-level integrity."""
    try:
        subprocess.run(
            [FSVERITY_BIN, "enable", str(filepath)],
            check=True, capture_output=True, timeout=30,
        )
        log.info("fs-verity enabled on %s", filepath.name)
        return True
    except FileNotFoundError:
        log.warning("fsverity tool not installed, skipping runtime integrity protection")
        return False
    except subprocess.CalledProcessError as e:
        # Common reasons: filesystem doesn't support verity, file is open, etc.
        log.warning(
            "fs-verity not supported for %s: %s",
            filepath.name,
            e.stderr.decode().strip() if e.stderr else str(e),
        )
        return False


def _get_fsverity_digest(filepath: Path) -> str | None:
    """Read the fs-verity digest of a file."""
    try:
        result = subprocess.run(
            [FSVERITY_BIN, "digest", str(filepath)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip().split()[0]  # "sha256:abc123..."
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return None


def _atomic_create_file(path: Path, content: bytes, *, mode: int = 0o600) -> None:
    """Atomically create a new file without replacing an existing path."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path, follow_symlinks=False)
        os.unlink(temporary_name)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_provenance_manifest(
    artifact_path: Path,
    filename: str,
    file_hash: str,
    size_bytes: int,
    source_url: str,
    source_revision: str,
    scan_results: dict,
    pipeline_details: dict,
    fsverity_success: bool = False,
) -> list[Path]:
    """Generate and optionally sign a JSON provenance manifest for a promoted model."""
    filename = _validate_artifact_name(filename)
    manifest = {
        "schema_version": "1.0",
        "artifact": {
            "filename": filename,
            "sha256": file_hash,
            "size_bytes": size_bytes,
        },
        "source": {
            "url": source_url or "local-import",
            "revision": source_revision or "unknown",
        },
        "scanners": _extract_scanner_versions(pipeline_details),
        "policy": {
            "version": _compute_policy_version(),
        },
        "promotion": {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "hostname": socket.gethostname(),
        },
        "scan_summary": _build_scan_summary(pipeline_details),
        "integrity": {
            "fsverity_enabled": fsverity_success,
            "fsverity_digest": _get_fsverity_digest(artifact_path) if fsverity_success else None,
        },
    }

    manifest_path = REGISTRY_DIR / f"{filename}.provenance.json"
    _atomic_create_file(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    log.info("provenance manifest written: %s", manifest_path.name)
    evidence_paths = [manifest_path]

    # Try to sign with local cosign key
    cosign_key = Path("/etc/secure-ai/keys/cosign.key")
    if cosign_key.exists():
        signature_path = Path(str(manifest_path) + ".sig")
        if _path_lexists(signature_path):
            raise FileExistsError(f"provenance signature already exists: {signature_path}")
        try:
            subprocess.run(
                [
                    COSIGN_BIN, "sign-blob", "--key", str(cosign_key),
                    "--output-signature", str(signature_path),
                    str(manifest_path),
                ],
                check=True, capture_output=True, timeout=30,
            )
            signature_handle, _signature_size = _open_regular_file(signature_path)
            signature_handle.close()
            signature_path.chmod(0o600)
            evidence_paths.append(signature_path)
            log.info("provenance manifest signed: %s", filename)
        except (OSError, subprocess.CalledProcessError) as e:
            signature_path.unlink(missing_ok=True)
            raise RuntimeError("configured provenance signing failed") from e
    else:
        log.info("no cosign key found, provenance manifest unsigned")
    return evidence_paths


def _stage_pending_promotion(
    artifact_path: Path,
    filename: str,
    file_hash: str,
    size_bytes: int,
    source_url: str,
    source_revision: str,
    scan_summary: dict,
    pipeline_details: dict,
    *,
    model_type: str,
    fsverity_success: bool,
) -> tuple[dict, Path, list[Path]]:
    """Stage and fsync every registry-visible object before the API commit."""
    staged_paths: list[Path] = []
    pending_path: Path | None = None
    try:
        guard_manifest = _stage_gguf_guard_manifest(pipeline_details)
        if guard_manifest is not None:
            staged_paths.append(guard_manifest)
        provenance_paths = _write_provenance_manifest(
            artifact_path,
            filename,
            file_hash,
            size_bytes,
            source_url,
            source_revision,
            scan_summary,
            pipeline_details,
            fsverity_success=fsverity_success,
        )
        staged_paths.extend(provenance_paths)
        evidence = sorted(
            (_evidence_descriptor(path) for path in staged_paths),
            key=lambda descriptor: descriptor["filename"],
        )
        payload = _build_registry_payload(
            filename,
            file_hash,
            size_bytes,
            scan_summary,
            model_type=model_type,
            source_url=source_url,
            pipeline_details=pipeline_details,
            evidence=evidence,
        )
        promotion_id = _promotion_id(filename, file_hash, model_type)
        pending_path = _journal_path(promotion_id)
        staging_record = _read_promotion_record(pending_path)
        if (
            staging_record["state"] != "staging"
            or staging_record["artifact"]
            != {
                "filename": filename,
                "sha256": file_hash,
                "size_bytes": size_bytes,
                "model_type": model_type,
            }
        ):
            raise ValueError("promotion staging marker does not bind the artifact")
        record = {
            "schema_version": 1,
            "state": "pending",
            "promotion_id": promotion_id,
            "artifact": {
                "filename": filename,
                "sha256": file_hash,
                "size_bytes": size_bytes,
                "model_type": model_type,
            },
            "payload": payload,
            "evidence": evidence,
            "attempts": 0,
            "next_attempt_at": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _verify_promotion_bundle(record)
        pending_path = _write_pending_record(record, replace=True)
        _fsync_directory(REGISTRY_DIR)
        return record, pending_path, staged_paths
    except Exception:
        if pending_path is not None and _path_lexists(pending_path):
            _remove_pending_record(pending_path)
        for staged_path in reversed(staged_paths):
            staged_path.unlink(missing_ok=True)
        raise


def _quarantine_invalid_pending_record(path: Path) -> None:
    invalid_dir = REGISTRY_DIR / ".promotion-journal-invalid"
    try:
        invalid_dir.mkdir(mode=0o700)
    except FileExistsError:
        pass
    metadata = invalid_dir.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid not in {0, os.geteuid()}
        or metadata.st_mode & 0o077
    ):
        raise RuntimeError("invalid-promotion hold directory is insecure")
    destination = invalid_dir / path.name
    if _path_lexists(destination):
        destination = invalid_dir / f"{path.stem}-{time.time_ns()}.json"
    os.replace(path, destination)
    _fsync_directory(path.parent)
    _fsync_directory(invalid_dir)


def _retry_delay(attempts: int) -> int:
    exponent = min(max(attempts - 1, 0), 10)
    return min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2**exponent))


def _recover_incomplete_staging(record: dict, path: Path) -> None:
    """Roll an interrupted pre-commit filesystem transaction back to quarantine."""
    artifact = record.get("artifact")
    evidence_names = record.get("evidence")
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"filename", "sha256", "size_bytes", "model_type"}
        or not isinstance(evidence_names, list)
        or not all(isinstance(name, str) for name in evidence_names)
    ):
        raise ValueError("staging record is malformed")
    filename = _validate_artifact_name(artifact["filename"])
    if record["promotion_id"] != _promotion_id(
        filename, artifact["sha256"], artifact["model_type"]
    ):
        raise ValueError("staging record does not bind its artifact")
    validated_evidence = [_validate_artifact_name(name) for name in evidence_names]
    original = QUARANTINE_DIR / filename
    destination = REGISTRY_DIR / filename
    if _path_lexists(destination):
        _rollback_promotion(destination, original)
    if _path_lexists(destination):
        raise RuntimeError("interrupted promotion could not be rolled back")
    for evidence_name in validated_evidence:
        evidence_path = REGISTRY_DIR / evidence_name
        try:
            metadata = evidence_path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("interrupted promotion evidence is unsafe")
        evidence_path.unlink()
    audit_log(
        "promotion_staging_recovered",
        filename,
        sha256=artifact["sha256"],
        promotion_id=record["promotion_id"],
    )
    _remove_pending_record(path)


def recover_pending_promotions(*, now: float | None = None) -> int:
    """Replay a bounded number of fully staged, idempotent registry commits."""
    _legacy_only()
    current_time = time.time() if now is None else now
    journal_dir = _promotion_journal_dir()
    committed = 0
    inspected = 0
    with os.scandir(journal_dir) as entries:
        for entry in entries:
            if inspected >= MAX_PENDING_COMMITS:
                break
            inspected += 1
            path = Path(entry.path)
            if not entry.name.endswith(".json"):
                _quarantine_invalid_pending_record(path)
                continue
            try:
                record = _read_promotion_record(path)
                if record["state"] == "staging":
                    _recover_incomplete_staging(record, path)
                    continue
                if record["next_attempt_at"] > current_time:
                    continue
                _verify_promotion_bundle(record)
                artifact = record["artifact"]
                payload = record["payload"]
                accepted = promote_to_registry(
                    artifact["filename"],
                    artifact["sha256"],
                    artifact["size_bytes"],
                    payload.get("scan_results", {}),
                    model_type=artifact["model_type"],
                    source_url=payload.get("source", ""),
                    promotion_id=record["promotion_id"],
                    evidence=record["evidence"],
                    payload_override=payload,
                )
                if accepted:
                    audit_log(
                        "promotion_recovered",
                        artifact["filename"],
                        sha256=artifact["sha256"],
                        promotion_id=record["promotion_id"],
                    )
                    _remove_pending_record(path)
                    committed += 1
                    continue
                record["attempts"] += 1
                record["next_attempt_at"] = current_time + _retry_delay(record["attempts"])
                _write_pending_record(record, replace=True)
            except (
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ):
                log.exception("pending promotion record is invalid: %s", entry.name)
                _quarantine_invalid_pending_record(path)
    return committed


def _build_scan_summary(details: dict) -> dict:
    """Build a summary dict from pipeline details for the registry manifest."""
    summary = {}
    scanner_versions = {}
    if "source_policy" in details:
        summary["source_policy"] = "pass" if details["source_policy"].get("passed") else "fail"
    if "format_gate" in details:
        summary["format_gate"] = "pass" if details["format_gate"].get("passed") else "fail"
    if "provenance" in details:
        summary["provenance"] = details["provenance"].get("provenance", "unknown")
        ver = details["provenance"].get("scanner_version")
        if ver:
            scanner_versions["cosign"] = ver
    if "static_scan" in details:
        scan = details["static_scan"]
        summary["static_scan"] = scan.get("scanner", "unknown")
        # Extract modelscan version from nested details
        ms_details = scan.get("details", {})
        ms_info = ms_details.get("modelscan", {}) if isinstance(ms_details, dict) else {}
        ver = ms_info.get("scanner_version") if isinstance(ms_info, dict) else None
        if ver:
            scanner_versions["modelscan"] = ver
    if "smoke_test" in details:
        summary["smoke_test"] = str(details["smoke_test"].get("score", "n/a"))
        ver = details["smoke_test"].get("scanner_version")
        if ver:
            scanner_versions["llama-server"] = ver
    if "diffusion_deep_scan" in details:
        summary["diffusion_deep_scan"] = "pass" if details["diffusion_deep_scan"].get("passed") else "fail"
    if scanner_versions:
        summary["scanner_versions"] = scanner_versions
    return summary


def _private_quarantine_directory(name: str) -> Path:
    path = QUARANTINE_DIR / name
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid not in {0, os.geteuid()}
        or metadata.st_mode & 0o077
    ):
        raise RuntimeError(f"private quarantine directory {name} is insecure")
    return path


def _queue_state_dir() -> Path:
    return _private_quarantine_directory(".watcher-state")


def _terminal_hold_dir() -> Path:
    return _private_quarantine_directory(".terminal-hold")


def _queue_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _queue_state_path(name: str) -> Path:
    return _queue_state_dir() / f"{_queue_key(name)}.json"


def _queue_lock_path(name: str) -> Path:
    # A fixed 256-slot lock set prevents unbounded lock-file growth from
    # attacker-chosen unique names. Hash collisions only serialize work.
    return _queue_state_dir() / f"claim-{_queue_key(name)[:2]}.lock"


def _default_queue_state(name: str) -> dict:
    return {
        "schema_version": 1,
        "name": name,
        "attempts": 0,
        "next_attempt_at": 0,
        "status": "ready",
        "last_error": "",
    }


def _read_queue_state(name: str) -> dict:
    path = _queue_state_path(name)
    try:
        raw = _read_secure_sidecar(path, max_bytes=16 * 1024)
    except FileNotFoundError:
        return _default_queue_state(name)
    state = json.loads(raw.decode("utf-8", errors="strict"))
    expected_keys = {
        "schema_version",
        "name",
        "attempts",
        "next_attempt_at",
        "status",
        "last_error",
    }
    if (
        not isinstance(state, dict)
        or set(state) != expected_keys
        or state["schema_version"] != 1
        or state["name"] != name
        or not isinstance(state["attempts"], int)
        or state["attempts"] < 0
        or not isinstance(state["next_attempt_at"], (int, float))
        or state["status"] not in {"ready", "retry", "terminal"}
        or not isinstance(state["last_error"], str)
        or len(state["last_error"]) > 128
    ):
        raise ValueError("queue state has an unsupported schema")
    return state


def _write_queue_state(state: dict) -> None:
    raw = (json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _atomic_replace_file(_queue_state_path(state["name"]), raw)


def _remove_queue_state(name: str) -> None:
    path = _queue_state_path(name)
    path.unlink(missing_ok=True)
    _fsync_directory(path.parent)


@contextmanager
def _claim_queue_entry(name: str) -> Iterator[bool]:
    """Take a nonblocking OS lock; kernel release makes claims crash-safe."""
    path = _queue_lock_path(name)
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    claimed = False
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid not in {0, os.geteuid()}
        ):
            raise RuntimeError("queue claim file is insecure")
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            claimed = True
        except BlockingIOError:
            claimed = False
        yield claimed
    finally:
        if claimed:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _move_to_terminal_hold(entry: Path, reason: str, attempts: int) -> None:
    """Move a permanently rejected queue object out of the active namespace."""
    hold_root = _terminal_hold_dir()
    hold_name = f"{_queue_key(entry.name)[:16]}-{time.time_ns()}"
    hold = hold_root / hold_name
    hold.mkdir(mode=0o700)
    destination = hold / entry.name
    os.replace(entry, destination)
    source_metadata = entry.parent / f".{entry.name}.source"
    try:
        source_metadata_metadata = source_metadata.lstat()
    except FileNotFoundError:
        source_metadata_metadata = None
    if source_metadata_metadata is not None:
        if stat.S_ISREG(source_metadata_metadata.st_mode) and source_metadata_metadata.st_nlink == 1:
            os.replace(source_metadata, hold / source_metadata.name)
        else:
            source_metadata.unlink(missing_ok=True)
    rejection = {
        "schema_version": 1,
        "filename": entry.name,
        "reason": reason,
        "attempts": attempts,
        "held_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_create_file(
        hold / "rejection.json",
        (json.dumps(rejection, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    _fsync_directory(hold)
    _fsync_directory(hold_root)


def _schedule_queue_retry(entry: Path, state: dict, reason: str, now: float) -> None:
    attempts = state["attempts"] + 1
    if attempts >= MAX_QUEUE_ATTEMPTS:
        _move_to_terminal_hold(entry, reason, attempts)
        _remove_queue_state(entry.name)
        audit_log("terminal_hold", entry.name, reason=reason, attempts=attempts)
        return
    state.update(
        attempts=attempts,
        next_attempt_at=now + _retry_delay(attempts),
        status="retry",
        last_error=reason[:128],
    )
    _write_queue_state(state)


def _dispatch_queue_entry(entry: Path, metadata: os.stat_result) -> str:
    if stat.S_ISLNK(metadata.st_mode):
        audit_log("rejected", entry.name, reason="symbolic_link")
        _move_to_terminal_hold(entry, "symbolic_link", 1)
        return "terminal"
    if stat.S_ISDIR(metadata.st_mode):
        index_path = entry / "model_index.json"
        try:
            index_metadata = index_path.lstat()
        except FileNotFoundError:
            index_metadata = None
        if index_metadata and stat.S_ISREG(index_metadata.st_mode):
            process_directory(entry)
            return "complete" if not _path_lexists(entry) else "retry"
        audit_log("rejected", entry.name, reason="missing_model_index")
        _move_to_terminal_hold(entry, "missing_model_index", 1)
        return "terminal"
    if stat.S_ISREG(metadata.st_mode):
        process_artifact(entry)
        return "complete" if not _path_lexists(entry) else "retry"
    audit_log("rejected", entry.name, reason="special_file")
    _move_to_terminal_hold(entry, "special_file", 1)
    return "terminal"


def scan_directory(*, now: float | None = None) -> int:
    """Process a bounded queue batch with claims and persistent retry state."""
    _legacy_only()
    if not QUARANTINE_DIR.exists():
        return 0
    current_time = time.time() if now is None else now
    recover_pending_promotions(now=current_time)
    processed = 0
    inspected = 0
    max_inspections = MAX_SCAN_ENTRIES * 8
    with os.scandir(QUARANTINE_DIR) as entries:
        for directory_entry in entries:
            if inspected >= max_inspections or processed >= MAX_SCAN_ENTRIES:
                break
            inspected += 1
            if directory_entry.name.startswith("."):
                if directory_entry.name in {".watcher-state", ".terminal-hold"}:
                    continue
                if directory_entry.name.endswith(".source"):
                    target_name = directory_entry.name[1 : -len(".source")]
                    try:
                        _validate_artifact_name(target_name)
                        (QUARANTINE_DIR / target_name).lstat()
                        continue
                    except FileNotFoundError:
                        pass
                    except ValueError:
                        pass
                processed += 1
                hidden_entry = Path(directory_entry.path)
                try:
                    audit_log("rejected", "hidden-entry", reason="orphan_hidden_entry")
                    _move_to_terminal_hold(hidden_entry, "orphan_hidden_entry", 1)
                except Exception:
                    log.exception("could not hold orphan hidden quarantine entry")
                continue
            entry = Path(directory_entry.path)
            try:
                _validate_artifact_name(entry.name)
            except ValueError:
                processed += 1
                log.warning("moving unsafe quarantine entry to terminal hold")
                try:
                    audit_log("rejected", "unsafe-name", reason="unsafe_filename")
                    _move_to_terminal_hold(entry, "unsafe_filename", 1)
                except Exception:
                    log.exception("could not hold unsafe quarantine entry")
                continue
            try:
                with _claim_queue_entry(entry.name) as claimed:
                    if not claimed:
                        continue
                    try:
                        state = _read_queue_state(entry.name)
                        if (
                            state["status"] == "terminal"
                            or state["next_attempt_at"] > current_time
                        ):
                            continue
                        metadata = directory_entry.stat(follow_symlinks=False)
                        outcome = _dispatch_queue_entry(entry, metadata)
                        processed += 1
                        if outcome in {"complete", "terminal"}:
                            _remove_queue_state(entry.name)
                        elif _path_lexists(entry):
                            _schedule_queue_retry(
                                entry,
                                state,
                                "processing_deferred",
                                current_time,
                            )
                    except FileNotFoundError:
                        _remove_queue_state(entry.name)
                    except Exception as exc:
                        processed += 1
                        log.exception("error processing %s", entry.name)
                        if _path_lexists(entry):
                            state = _read_queue_state(entry.name)
                            _schedule_queue_retry(
                                entry,
                                state,
                                f"processing_error:{type(exc).__name__}",
                                current_time,
                            )
            except FileNotFoundError:
                _remove_queue_state(entry.name)
            except Exception:
                log.exception("error processing %s", entry.name)
    return processed


def _validate_startup_configuration() -> None:
    for label, path in (("quarantine", QUARANTINE_DIR), ("registry", REGISTRY_DIR)):
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"{label} path must be a real directory")
    registry_metadata = REGISTRY_DIR.lstat()
    if registry_metadata.st_mode & 0o002:
        raise RuntimeError("registry directory must not be world writable")
    quarantine = QUARANTINE_DIR.resolve()
    registry = REGISTRY_DIR.resolve()
    if (
        quarantine == registry
        or quarantine in registry.parents
        or registry in quarantine.parents
    ):
        raise RuntimeError("quarantine and registry directories must not overlap")
    _validated_registry_base_url()
    load_policy()
    _queue_state_dir()
    _terminal_hold_dir()
    _promotion_journal_dir()
    token = _load_service_token()
    if not token and not (
        os.getenv("SECAI_ALLOW_INSECURE_REGISTRY", "") == "1"
        and _registry_is_loopback()
    ):
        raise RuntimeError("registry authentication is required")


def main():
    _legacy_only()  # Required before mkdir, policy loading or old journal recovery.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True, mode=0o750)
    _validate_startup_configuration()

    log.info("quarantine watcher starting")
    log.info("watching: %s", QUARANTINE_DIR)
    log.info("registry dir: %s", REGISTRY_DIR)
    log.info("registry API origin: %s", _validated_registry_base_url())

    while True:
        scan_directory()
        time.sleep(5)


if __name__ == "__main__":
    main()
