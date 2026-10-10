"""Credentialless helper-v2 scan worker and its bounded binary output frame.

Only the qualified sandbox controller may launch worker_main. This module does
not establish isolation by itself and exposes no host-scanning fallback.
"""
from __future__ import annotations

import hashlib
import os
import resource
import stat
import struct
import sys
from pathlib import Path

from .policy_context import PolicyContext
from .registry_protocol import (
    MAX_BODY_BYTES,
    MAX_EVIDENCE_BYTES,
    POLICY_COMPONENTS,
    ProtocolError,
    context_digest,
    encode_request,
    strict_json,
    validate_preparation,
)
from .scan_execution import SANDBOX_TMP, BoundedRunner, boot_ns

FRAME_MAGIC = b"HA2SCAN\0"
TRANSFER_LIMIT = MAX_EVIDENCE_BYTES - MAX_BODY_BYTES
SCANNER_FIELDS = {"schema_version", "passed", "verdict", "available", "scanner", "scanner_version", "reason", "scanned_count"}


def worker_mount_guard() -> None:
    """Reject a direct host invocation before any artifact parser is entered."""
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    if (int(status["CapEff"].strip(), 16) != 0 or status["NoNewPrivs"].strip() != "1" or status["Seccomp"].strip() != "2" or
            not os.statvfs("/").f_flag & os.ST_RDONLY or Path("/sys/fs/cgroup").exists()):
        raise ProtocolError("worker process/root/control boundary unavailable")
    mounts = {}
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        prefix, suffix = line.split(" - ", 1)
        fields = prefix.split()
        if any(value.startswith("shared:") for value in fields[6:]):
            raise ProtocolError("worker mount propagation is shared")
        mounts[fields[4]] = (fields[5].split(","), suffix.split()[0])
    for path in ("/work", SANDBOX_TMP):
        if path not in mounts or mounts[path][1] != "tmpfs" or "rw" not in mounts[path][0]:
            raise ProtocolError("worker scratch is not the bounded private tmpfs")
        filesystem = os.statvfs(path)
        if not 0 < filesystem.f_blocks * filesystem.f_frsize <= MAX_EVIDENCE_BYTES:
            raise ProtocolError("worker scratch capacity exceeds one MiB")
    # Verify the namespace alias; never allocate a shared temporary file here.
    if Path("/dev/shm").resolve(strict=True) != Path(SANDBOX_TMP):  # nosec B108
        raise ProtocolError("worker shared-memory path escapes bounded scratch")


def encode_frame(files: list[tuple[str, str, bytes]], *, passed: bool, error: str = "") -> bytes:
    descriptors = []
    for role, name, raw in files:
        if type(raw) is not bytes or not raw:
            raise ProtocolError("empty/mutable scan output")
        descriptors.append({"role": role, "filename": name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    header = encode_request({"version": 2, "passed": passed, "error": error, "files": descriptors})
    size = len(FRAME_MAGIC) + 4 + len(header) + sum(len(raw) for _role, _name, raw in files)
    if size > MAX_EVIDENCE_BYTES:
        raise ProtocolError("complete scan frame exceeds one MiB")
    return FRAME_MAGIC + struct.pack(">I", len(header)) + header + b"".join(raw for _role, _name, raw in files)


def decode_frame(raw: bytes) -> tuple[dict, list[tuple[dict, bytes]]]:
    if not 12 < len(raw) <= MAX_EVIDENCE_BYTES or raw[:8] != FRAME_MAGIC:
        raise ProtocolError("scan frame missing/oversized/incomplete")
    header_size = struct.unpack(">I", raw[8:12])[0]
    if not 0 < header_size <= MAX_BODY_BYTES or 12 + header_size > len(raw):
        raise ProtocolError("scan frame header bound invalid")
    header = strict_json(raw[12:12 + header_size])
    if (not isinstance(header, dict) or set(header) != {"version", "passed", "error", "files"} or
            type(header["version"]) is not int or header["version"] != 2 or type(header["passed"]) is not bool or
            not isinstance(header["error"], str) or len(header["error"]) > 1024 or
            not isinstance(header["files"], list) or len(header["files"]) > 2):
        raise ProtocolError("scan frame schema invalid")
    files, offset, names, roles = [], 12 + header_size, set(), set()
    allowed = {"pipeline-report": "pipeline.json", "gguf-integrity": "integrity.json"} if header["passed"] else {"failed-capture": "failed-capture.bin"}
    for descriptor in header["files"]:
        if (not isinstance(descriptor, dict) or set(descriptor) != {"role", "filename", "size_bytes", "sha256"} or
                not isinstance(descriptor["role"], str) or descriptor["role"] not in allowed or
                descriptor["filename"] != allowed[descriptor["role"]] or descriptor["role"] in roles or
                descriptor["filename"] in names or type(descriptor["size_bytes"]) is not int or
                not 0 < descriptor["size_bytes"] <= TRANSFER_LIMIT):
            raise ProtocolError("scan frame inventory invalid")
        end = offset + descriptor["size_bytes"]
        content = raw[offset:end]
        if end > len(raw) or hashlib.sha256(content).hexdigest() != descriptor["sha256"]:
            raise ProtocolError("scan frame content binding differs")
        files.append((descriptor, content))
        roles.add(descriptor["role"])
        names.add(descriptor["filename"])
        offset = end
    if (offset != len(raw) or (header["passed"] and (header["error"] or "pipeline-report" not in roles)) or
            (not header["passed"] and not header["error"])):
        raise ProtocolError("scan frame trailing bytes/incomplete passing evidence")
    return header, files


def build_report(result: dict, prepare: dict, context: dict) -> dict:
    """Retain P1 typed NA outcomes; derive only the versioned wire projection."""
    from .pipeline import _valid_static_outcome

    if result.get("passed") is not True or result.get("reason") != "all_checks_passed":
        raise ProtocolError("pipeline did not complete every required stage")
    details = result["details"]
    static = details["static_scan"]
    if static.get("passed") is not True or details["hash_pin"].get("pinned") is not True:
        raise ProtocolError("static scan or exact hash pin missing")
    gates = {}
    for key in ("source_policy", "format_gate", "hash_pin", "provenance", "polyglot_check", "entropy", "weight_stats", "smoke_test"):
        value = static["details"][key] if key in {"polyglot_check", "entropy", "weight_stats"} else details[key]
        if not isinstance(value, dict) or value.get("passed") is not True:
            raise ProtocolError("required pipeline gate did not pass")
        verdict, count = "pass", 1
        if key == "smoke_test":
            verdict = "not_applicable" if prepare["format"] == "safetensors" else "not_required" if not context["require_behavior"] else "pass"
            if verdict != "pass":
                count = 0
        elif key == "weight_stats":
            count = value.get("tensors_analyzed")
            if type(count) is not int or count <= 0:
                raise ProtocolError("weight gate has no positive tensor coverage")
        assurance = value.get("provenance", "") if key == "provenance" else ""
        if key == "provenance" and assurance != ("none" if not prepare["source"] else "recorded"):
            raise ProtocolError("source assurance differs from the admitted profile")
        gates[key] = {"verdict": verdict, "reason": "P1 " + key + ": " + verdict, "count": count, "assurance": assurance}
    scanners = {}
    for key in ("modelscan", "fickling", "modelaudit", "gguf_guard"):
        outcome = static["details"][key]
        na = key in {"modelscan", "fickling"} or (key == "gguf_guard" and prepare["format"] == "safetensors")
        if not _valid_static_outcome(outcome, allow_not_applicable=na) or outcome["verdict"] != ("not_applicable" if na else "pass"):
            raise ProtocolError("scanner outcome is incomplete or contradicts applicability")
        if not isinstance(outcome["reason"], str) or not 0 < len(outcome["reason"].encode()) <= 1024:
            raise ProtocolError("scanner reason exceeds record bound")
        scanners[key] = {name: outcome[name] for name in SCANNER_FIELDS}
    report = {key: prepare[key] for key in ("admission_id", "filename", "sha256", "size_bytes", "source", "source_revision", "policy_bundle")}
    report.update(version=2, pipeline_sha256=context["pipeline_sha256"], gates=gates, scanners=scanners)
    encode_request(report)  # Bound typed manifest growth before transfer.
    return report


def worker_main() -> None:
    """Runs after namespace, seccomp, mount and cgroup boundaries are in force."""
    runner = None
    try:
        worker_mount_guard()
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_EVIDENCE_BYTES, MAX_EVIDENCE_BYTES))
        config = strict_json(sys.stdin.buffer.read(MAX_BODY_BYTES + 1))
        if not isinstance(config, dict) or set(config) != {"prepare", "context", "deadline_boot_ns"}:
            raise ProtocolError("scan invocation fields invalid")
        prepare, context = config["prepare"], config["context"]
        validate_preparation(prepare)
        if prepare["context_sha256"] != context_digest(prepare["policy_bundle"]["sha256"], context):
            raise ProtocolError("scan context differs from prepared binding")
        expiry = config["deadline_boot_ns"]
        if not isinstance(expiry, str) or not expiry.isascii() or not expiry.isdecimal() or len(expiry) > 19:
            raise ProtocolError("scan deadline invalid")
        deadline = int(expiry)
        if not boot_ns() < deadline <= boot_ns() + 1800 * 1_000_000_000:
            raise ProtocolError("scan lifetime invalid")
        runner = BoundedRunner(deadline, max_capture=TRANSFER_LIMIT)
        components = {}
        for name in POLICY_COMPONENTS:
            if not os.statvfs(Path("/policy", name)).f_flag & os.ST_RDONLY:
                raise ProtocolError("worker policy bind is writable")
            with Path("/policy", name).open("rb") as policy_file:
                components[name] = policy_file.read(MAX_EVIDENCE_BYTES + 1)
        policy = PolicyContext.from_components(components)
        if policy.evidence() != prepare["policy_bundle"]:
            raise ProtocolError("scanner policy bytes differ from preparation")
        from . import pipeline

        if hashlib.sha256(Path(pipeline.__file__).read_bytes()).hexdigest() != context["pipeline_sha256"]:
            raise ProtocolError("scanner pipeline source identity differs")
        path = Path("/work", prepare["filename"])
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or before.st_mode & 0o222 or before.st_size != prepare["size_bytes"] or
                not os.statvfs(path).f_flag & os.ST_RDONLY):
            raise ProtocolError("scanner input is not immutable admitted bytes")
        result = pipeline.run_pipeline(path, prepare["sha256"], {}, source_url=prepare["source"], policy_context=policy, runner=runner)
        runner.check()
        after = path.lstat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ProtocolError("scanner input identity changed")
        files = [("pipeline-report", "pipeline.json", encode_request(build_report(result, prepare, context)))]
        if prepare["format"] == "gguf":
            expected = path.with_suffix(".gguf.manifest.json")
            generated = result["details"].get("gguf_guard_manifest", {})
            if generated.get("generated") is not True or generated.get("manifest_path") != str(expected):
                raise ProtocolError("GGUF integrity generation incomplete")
            content = pipeline._read_bounded_regular_file(expected, max_bytes=TRANSFER_LIMIT)
            if not content or len(content) + len(files[0][2]) > TRANSFER_LIMIT:
                raise ProtocolError("GGUF evidence exceeds prepared transfer budget")
            files.append(("gguf-integrity", "integrity.json", content))
        runner.check()
        frame = encode_frame(files, passed=True)
    except Exception as error:
        capture = bytes(runner.raw_capture) if runner is not None else b""
        files = [("failed-capture", "failed-capture.bin", capture)] if capture else []
        # Error types are safe control metadata; artifact strings never expand
        # this bounded frame or leak a host configuration/credential.
        frame = encode_frame(files, passed=False, error=type(error).__name__)
    view = memoryview(frame)
    while view:
        amount = os.write(sys.stdout.fileno(), view)
        if amount <= 0:
            raise ProtocolError("scan frame publication failed")
        view = view[amount:]
