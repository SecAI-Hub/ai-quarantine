"""
Hash-chained append-only audit log.

Each log entry includes a SHA-256 hash of the previous entry, forming a
tamper-evident chain. If any entry is modified, deleted, or inserted, the
chain breaks and verification fails.

Format: one JSON object per line (JSONL), each containing:
  - timestamp: ISO 8601 UTC
  - event: event type string
  - data: arbitrary event data dict
  - prev_hash: SHA-256 hex of the previous entry's JSON (empty string for genesis)
  - entry_hash: SHA-256 hex of (prev_hash + event + data + timestamp)

Usage:
    chain = AuditChain("/var/lib/secure-ai/logs/quarantine-audit.jsonl")
    chain.append("promoted", {"model": "phi-3", "sha256": "abc..."})

    result = AuditChain.verify("/var/lib/secure-ai/logs/quarantine-audit.jsonl")
    # result = {"valid": True, "entries": 42, "first": "...", "last": "..."}
"""

import hashlib
import hmac
import json
import logging
import os
import stat
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("audit_chain")


def _hash_entry(
    prev_hash: str,
    event: str,
    data: dict,
    timestamp: str,
    key: bytes | None = None,
) -> str:
    """Compute the hash for an audit entry."""
    canonical = json.dumps(
        {"prev_hash": prev_hash, "event": event, "data": data, "timestamp": timestamp},
        sort_keys=True,
        separators=(",", ":"),
    )
    encoded = canonical.encode("utf-8")
    if key:
        return hmac.new(key, encoded, hashlib.sha256).hexdigest()
    return hashlib.sha256(encoded).hexdigest()


def _load_key(key_path: str | None) -> bytes | None:
    path = key_path or os.getenv("AUDIT_HMAC_KEY_PATH", "").strip()
    if not path:
        return None
    descriptor = -1
    try:
        descriptor = os.open(
            path,
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
            or not 32 <= metadata.st_size <= 4096
        ):
            return None
        raw = os.read(descriptor, 4097)
        if len(raw) != metadata.st_size:
            return None
    except OSError:
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not all(0x21 <= octet <= 0x7E for octet in raw):
        return None
    return raw


def _open_regular_text(path: Path):
    """Open an audit artifact without following links or accepting special files."""
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError("audit artifact is not a single-link regular file")
        return os.fdopen(descriptor, "r", encoding="utf-8")
    except Exception:
        os.close(descriptor)
        raise


def _read_checkpoint(path: Path) -> dict:
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
            or metadata.st_size > 64 * 1024
        ):
            raise OSError("audit checkpoint is insecure")
        raw = os.read(descriptor, 64 * 1024 + 1)
        if len(raw) != metadata.st_size:
            raise OSError("audit checkpoint changed while reading")
        checkpoint = json.loads(raw.decode("utf-8", errors="strict"))
        if not isinstance(checkpoint, dict):
            raise TypeError("audit checkpoint is not an object")
        return checkpoint
    finally:
        os.close(descriptor)


class AuditChain:
    """Append-only hash-chained audit log."""

    def __init__(
        self,
        log_path: str | Path,
        max_size_mb: int = 50,
        *,
        key_path: str | None = None,
    ):
        self._path = Path(log_path)
        if max_size_mb <= 0:
            raise ValueError("max_size_mb must be positive")
        self._max_size = max_size_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._prev_hash = ""
        self._entry_count = 0
        self._key_path = key_path
        self._key = _load_key(key_path)
        configured_key_path = key_path or os.getenv("AUDIT_HMAC_KEY_PATH", "").strip()
        if configured_key_path and self._key is None:
            raise RuntimeError("configured audit HMAC key is unavailable or insecure")
        self._checkpoint_path = self._path.with_suffix(
            self._path.suffix + ".checkpoint"
        )

        # Resume chain from existing log
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            pass  # directory may not be writable in test environments
        if self._log_paths(self._path) or self._checkpoint_path.exists():
            try:
                verification = self.verify(str(self._path), key_path=self._key_path)
                if not verification["valid"]:
                    raise RuntimeError(verification["detail"])
                last_line = ""
                for current_path in self._log_paths(self._path):
                    with _open_regular_text(current_path) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                last_line = line
                                self._entry_count += 1
                if last_line:
                    entry = json.loads(last_line)
                    self._prev_hash = entry.get("entry_hash", "")
            except (json.JSONDecodeError, OSError, RuntimeError) as e:
                raise RuntimeError(
                    f"refusing to append to unverifiable audit chain {self._path}: {e}"
                ) from e

    @staticmethod
    def _log_paths(path: Path) -> list[Path]:
        archives = sorted(path.parent.glob(f"{path.stem}.*{path.suffix}"))
        return [*archives, *([path] if path.exists() else [])]

    def append(self, event: str, data: dict | None = None) -> str:
        """Append a hash-chained entry. Returns the entry hash."""
        if data is None:
            data = {}

        ts = datetime.now(timezone.utc).isoformat()

        with self._lock:
            entry_hash = _hash_entry(self._prev_hash, event, data, ts, self._key)

            entry = {
                "timestamp": ts,
                "event": event,
                "data": data,
                "prev_hash": self._prev_hash,
                "entry_hash": entry_hash,
                "algorithm": "hmac-sha256" if self._key else "sha256",
            }

            try:
                # Check if rotation needed
                if self._path.exists():
                    metadata = self._path.lstat()
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                        raise OSError("audit destination is not a single-link regular file")
                    if metadata.st_size >= self._max_size:
                        self._rotate()

                descriptor = os.open(
                    self._path,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_APPEND
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                with os.fdopen(descriptor, "a", encoding="utf-8") as f:
                    metadata = os.fstat(f.fileno())
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                        raise OSError("audit destination is not a single-link regular file")
                    os.fchmod(f.fileno(), 0o600)
                    f.write(json.dumps(entry, separators=(",", ":")) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                self._prev_hash = entry_hash
                self._entry_count += 1
                self._write_checkpoint()

            except OSError as e:
                log.error("failed to write audit entry: %s", e)
                raise

        return entry_hash

    def _rotate(self):
        """Rotate the log file when it exceeds max size."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        archive = self._path.with_suffix(f".{ts}.jsonl")
        try:
            self._path.rename(archive)
            # Make archive read-only
            os.chmod(str(archive), 0o400)
            log.info("rotated audit log: %s -> %s", self._path, archive)
            # The next file continues from the archived tail.
        except OSError as e:
            log.error("failed to rotate audit log: %s", e)
            raise

    def _write_checkpoint(self) -> None:
        if not self._key:
            return
        checkpoint = {"entries": self._entry_count, "last_hash": self._prev_hash}
        canonical = json.dumps(
            checkpoint, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        checkpoint["hmac"] = hmac.new(
            self._key, canonical, hashlib.sha256
        ).hexdigest()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._checkpoint_path.name}.",
            dir=self._checkpoint_path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(checkpoint, handle, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, self._checkpoint_path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    @staticmethod
    def verify(log_path: str, *, key_path: str | None = None) -> dict:
        """Verify the integrity of a hash-chained audit log.

        Returns:
            {
                "valid": bool,
                "entries": int,
                "broken_at": int or None,  # line number of first break
                "detail": str,
            }
        """
        path = Path(log_path)
        key = _load_key(key_path)
        configured_key_path = key_path or os.getenv("AUDIT_HMAC_KEY_PATH", "").strip()
        if configured_key_path and key is None:
            return {
                "valid": False,
                "entries": 0,
                "broken_at": None,
                "detail": "configured HMAC verification key is unavailable",
            }
        log_paths = AuditChain._log_paths(path)
        checkpoint_path = path.with_suffix(path.suffix + ".checkpoint")
        if not log_paths:
            if key and checkpoint_path.exists():
                return {
                    "valid": False,
                    "entries": 0,
                    "broken_at": None,
                    "detail": "audit log was deleted but its authenticated checkpoint remains",
                }
            return {"valid": True, "entries": 0, "broken_at": None,
                    "detail": "log file does not exist"}
        prev_hash = ""
        count = 0

        try:
            line_num = 0
            for current_path in log_paths:
                with _open_regular_text(current_path) as f:
                    for line in f:
                        line_num += 1
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            return {
                                "valid": False,
                                "entries": count,
                                "broken_at": line_num,
                                "detail": f"line {line_num}: invalid JSON",
                            }
                        if not isinstance(entry, dict):
                            return {
                                "valid": False,
                                "entries": count,
                                "broken_at": line_num,
                                "detail": f"line {line_num}: entry is not an object",
                            }

                        algorithm = entry.get("algorithm", "sha256")
                        if algorithm not in {"sha256", "hmac-sha256"}:
                            return {
                                "valid": False,
                                "entries": count,
                                "broken_at": line_num,
                                "detail": f"line {line_num}: unsupported hash algorithm",
                            }
                        if algorithm == "hmac-sha256" and not key:
                            return {
                                "valid": False,
                                "entries": count,
                                "broken_at": line_num,
                                "detail": "HMAC verification key unavailable",
                            }

                        stored_prev = entry.get("prev_hash", "")
                        if stored_prev != prev_hash:
                            return {
                                "valid": False,
                                "entries": count,
                                "broken_at": line_num,
                                "detail": f"line {line_num}: chain linkage mismatch",
                            }

                        expected_hash = _hash_entry(
                            entry.get("prev_hash", ""),
                            entry.get("event", ""),
                            entry.get("data", {}),
                            entry.get("timestamp", ""),
                            key if algorithm == "hmac-sha256" else None,
                        )
                        stored_hash = entry.get("entry_hash", "")
                        if not isinstance(stored_hash, str) or not hmac.compare_digest(
                            stored_hash, expected_hash
                        ):
                            return {
                                "valid": False,
                                "entries": count,
                                "broken_at": line_num,
                                "detail": f"line {line_num}: hash mismatch",
                            }

                        prev_hash = stored_hash
                        count += 1

        except OSError as e:
            return {
                "valid": False,
                "entries": count,
                "broken_at": None,
                "detail": f"read error: {e}",
            }

        if key and count:
            try:
                checkpoint = _read_checkpoint(checkpoint_path)
                checkpoint_hmac = str(checkpoint.pop("hmac", ""))
                canonical = json.dumps(
                    checkpoint, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                expected_checkpoint_hmac = hmac.new(
                    key, canonical, hashlib.sha256
                ).hexdigest()
                if (
                    not hmac.compare_digest(checkpoint_hmac, expected_checkpoint_hmac)
                    or checkpoint.get("entries") != count
                    or checkpoint.get("last_hash") != prev_hash
                ):
                    raise ValueError("checkpoint does not match the log tail")
            except (
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                AttributeError,
            ) as exc:
                return {
                    "valid": False,
                    "entries": count,
                    "broken_at": None,
                    "detail": f"checkpoint verification failed: {exc}",
                }

        return {
            "valid": True,
            "entries": count,
            "broken_at": None,
            "detail": f"chain intact: {count} entries verified",
        }
