"""Exclusive helper-v2 outbox inventory and one-writer bounded journal.

Every failed/new copy is retained. This code has no source move, unlink, legacy
journal recovery, garbage collection or existing-path ownership operation.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from pathlib import Path

from .registry_protocol import (
    HEX,
    MAX_BODY_BYTES,
    MAX_EVIDENCE_BYTES,
    OUTBOX_OVERHEAD_BYTES,
    POLICY_COMPONENTS,
    ProtocolError,
    context_digest,
    strict_json,
    validate_commit,
    validate_preparation,
)

JOURNAL_LIMIT = MAX_EVIDENCE_BYTES
JOURNAL_NAMES = {"journal.json", "journal.next"}
STAGES = {"intent", "prepared", "copying", "scanning", "commit-ready", "committed", "denied", "failed", "unknown"}
TRANSITIONS = {
    "intent": {"intent", "prepared", "failed", "unknown"},
    "prepared": {"copying", "denied", "failed", "unknown"},
    "copying": {"scanning", "denied", "failed", "unknown"},
    "scanning": {"commit-ready", "denied", "failed", "unknown"},
    "commit-ready": {"commit-ready", "committed", "denied", "unknown"},
    **{stage: {stage} for stage in ("committed", "denied", "failed", "unknown")},
}


def identity(metadata) -> tuple[int, ...]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_uid, metadata.st_gid,
            metadata.st_nlink, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)


def _write_all(fd: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short private outbox write")
        view = view[written:]


def validate_journal(record: dict) -> dict:
    fields = {"version", "admission_id", "stage", "prepare_wire", "commit_wire", "context", "artifact", "original", "inventory", "error"}
    if (not isinstance(record, dict) or set(record) != fields or type(record["version"]) is not int or record["version"] != 2 or
            not isinstance(record["admission_id"], str) or HEX.fullmatch(record["admission_id"]) is None or
            not isinstance(record["stage"], str) or record["stage"] not in STAGES or
            not isinstance(record["prepare_wire"], str) or not isinstance(record["commit_wire"], str) or
            not isinstance(record["error"], str) or len(record["error"]) > 1024):
        raise ProtocolError("helper-v2 journal schema invalid")
    prepare = strict_json(record["prepare_wire"].encode("utf-8"))
    validate_preparation(prepare)
    if prepare["admission_id"] != record["admission_id"]:
        raise ProtocolError("journal identity differs from exact preparation")
    if prepare["context_sha256"] != context_digest(prepare["policy_bundle"]["sha256"], record["context"]):
        raise ProtocolError("journal context differs from exact preparation")
    if record["commit_wire"]:
        commit = strict_json(record["commit_wire"].encode("utf-8"))
        validate_commit(commit)
        if commit["admission_id"] != record["admission_id"]:
            raise ProtocolError("journal commit identity differs")
    original = record["original"]
    if (not isinstance(original, list) or len(original) != 9 or any(type(value) is not int or value < 0 for value in original) or
            original[6] != prepare["size_bytes"]):
        raise ProtocolError("journal original observation invalid")
    expected = ["input/" + prepare["filename"], *("policy/" + name for name in POLICY_COMPONENTS), "capture.bin", "evidence/pipeline.json"]
    if prepare["format"] == "gguf":
        expected.append("evidence/integrity.json")
    if record["inventory"] != expected:
        raise ProtocolError("journal intended file inventory differs")
    if record["artifact"] is not None:
        from .registry_protocol import validate_artifact

        validate_artifact(record["artifact"], record["prepare_wire"].encode(), record["context"], record["commit_wire"].encode() or None)
    elif record["stage"] not in {"intent", "unknown", "failed"}:
        raise ProtocolError("journal lacks expected preparation authority")
    if record["stage"] in {"commit-ready", "committed"} and not record["commit_wire"]:
        raise ProtocolError("journal commit stage lacks exact request bytes")
    if record["stage"] == "committed" and "commit_receipt" not in record["artifact"]["admission"]:
        raise ProtocolError("committed journal lacks the historical receipt")
    return prepare


class Outbox:
    def __init__(self, path: Path, gid: int):
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise ProtocolError("outbox must be an existing canonical private directory")
        self.path, self.gid = path, gid
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            current = os.fstat(self.fd)
            if current.st_uid != os.geteuid() or current.st_gid != gid or current.st_mode & 0o027:
                raise ProtocolError("outbox producer/read-group ownership differs")
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.root_identity = identity(current)[:5]
        except BaseException:
            os.close(self.fd)
            raise

    def close(self):
        os.close(self.fd)

    def current(self):
        if identity(os.stat(self.path, follow_symlinks=False))[:5] != self.root_identity:
            raise ProtocolError("outbox root identity changed")

    def _namespace(self, admission_id: str) -> int:
        self.current()
        if not isinstance(admission_id, str) or HEX.fullmatch(admission_id) is None:
            raise ProtocolError("invalid outbox namespace")
        fd = os.open(admission_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.fd)
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or info.st_gid != self.gid or info.st_mode & 0o027:
            os.close(fd)
            raise ProtocolError("outbox namespace identity/mode differs")
        return fd

    def _read(self, directory: int, name: str, limit: int = JOURNAL_LIMIT) -> bytes:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() or before.st_nlink != 1 or
                    before.st_mode & 0o077 or not 0 < before.st_size <= limit):
                raise ProtocolError("journal is not bounded private owned data")
            with os.fdopen(os.dup(fd), "rb") as handle:
                raw = handle.read(limit + 1)
            if len(raw) != before.st_size or identity(before) != identity(os.fstat(fd)) or identity(before) != identity(os.stat(name, dir_fd=directory, follow_symlinks=False)):
                raise ProtocolError("journal changed during read")
            return raw
        finally:
            os.close(fd)

    def read(self, admission_id: str) -> dict:
        directory = self._namespace(admission_id)
        try:
            try:
                os.stat("journal.next", dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ProtocolError("uncertain journal generation retained; maintenance required")
            record = strict_json(self._read(directory, "journal.json"), limit=JOURNAL_LIMIT)
            validate_journal(record)
            if record["admission_id"] != admission_id:
                raise ProtocolError("journal path identity differs")
            return record
        finally:
            os.close(directory)

    def records(self):
        self.current()
        with os.scandir(self.fd) as entries:
            for index, entry in enumerate(entries):
                if index >= 100000 or HEX.fullmatch(entry.name) is None or not entry.is_dir(follow_symlinks=False):
                    raise ProtocolError("unknown/excess outbox inventory; no adoption or capacity assumption")
                yield self.read(entry.name)

    def reserve(self, prepare: dict, limits: dict):
        """Logical peak debit plus free-space check; never a physical quota claim."""
        validate_preparation(prepare)
        total = prepare["size_bytes"] + prepare["evidence_budget_bytes"] + OUTBOX_OVERHEAD_BYTES
        for record in self.records():
            previous = validate_journal(record)
            if previous["name"] == prepare["name"] or previous["filename"] == prepare["filename"]:
                raise ProtocolError("retained admission already binds this logical name; reconcile it")
            total += previous["size_bytes"] + previous["evidence_budget_bytes"] + OUTBOX_OVERHEAD_BYTES
        free = os.fstatvfs(self.fd)
        if total > limits["retained_byte_limit"] or free.f_bavail * free.f_frsize < total + limits["min_free_bytes"]:
            raise ProtocolError("retained outbox debit/free-space capacity unavailable")

    def create(self, record: dict):
        validate_journal(record)
        if record["stage"] != "intent" or record["artifact"] is not None or record["commit_wire"]:
            raise ProtocolError("new outbox namespace must start with an exact unsubmitted intent")
        self.current()
        admission_id = record["admission_id"]
        os.mkdir(admission_id, 0o750, dir_fd=self.fd)  # Every existing name is refused.
        directory = os.open(admission_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.fd)
        try:
            os.fchown(directory, -1, self.gid)  # Exclusively new descriptor only.
            self._write_record(directory, record, initial=True)
            os.fsync(self.fd)
        finally:
            os.close(directory)

    def _write_record(self, directory: int, record: dict, *, initial: bool):
        import json

        raw = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        if not 0 < len(raw) <= JOURNAL_LIMIT:
            raise ProtocolError("journal generation exceeds one MiB")
        name = "journal.json" if initial else "journal.next"
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
        try:
            _write_all(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        if not initial:
            os.rename("journal.next", "journal.json", src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)

    def update(self, record: dict):
        validate_journal(record)
        previous = self.read(record["admission_id"])
        for key in ("admission_id", "version", "prepare_wire", "context", "original", "inventory"):
            if record[key] != previous[key]:
                raise ProtocolError("journal update changed its immutable intent")
        if previous["commit_wire"] and record["commit_wire"] != previous["commit_wire"]:
            raise ProtocolError("journal update changed exact commit bytes")
        if record["stage"] not in TRANSITIONS[previous["stage"]]:
            raise ProtocolError("journal transition cannot skip or restart work")
        directory = self._namespace(record["admission_id"])
        try:
            self._write_record(directory, record, initial=False)
        finally:
            os.close(directory)

    def write_new(self, admission_id: str, relative: str, content: bytes):
        record = self.read(admission_id)
        if (record["stage"] not in {"copying", "scanning"} or relative not in record["inventory"] or
                relative.startswith("input/") or type(content) is not bytes or not content):
            raise ProtocolError("output is not in the durable intended inventory")
        if relative.startswith("policy/") and len(content) > MAX_EVIDENCE_BYTES:
            raise ProtocolError("policy component exceeds reserved bound")
        if not relative.startswith("input/") and len(content) > MAX_EVIDENCE_BYTES:
            raise ProtocolError("scan output exceeds reserved bound")
        if relative.startswith("evidence/"):
            prepare = validate_journal(record)
            total = len(content)
            for item in record["inventory"]:
                if item.startswith("evidence/"):
                    try:
                        info = (self.path / admission_id / item).lstat()
                    except FileNotFoundError:
                        continue
                    if not stat.S_ISREG(info.st_mode):
                        raise ProtocolError("unexpected evidence inventory type")
                    total += info.st_size
            if total + MAX_BODY_BYTES > prepare["evidence_budget_bytes"]:
                raise ProtocolError("combined evidence/registry intent exceeds reserved budget")
        self._copy_new(admission_id, relative, content)

    def copy_input(self, admission_id: str, source_fd: int, check):
        record = self.read(admission_id)
        prepare = validate_journal(record)
        before = os.fstat(source_fd)
        if (record["stage"] != "copying" or list(identity(before)) != record["original"] or
                not stat.S_ISREG(before.st_mode) or before.st_nlink != 1):
            raise ProtocolError("source descriptor differs from durably observed original")
        os.lseek(source_fd, 0, os.SEEK_SET)

        def copy_to(target_fd):
            hashed = hashlib.sha256()
            remaining = prepare["size_bytes"]
            while remaining:
                check()
                chunk = os.read(source_fd, min(65536, remaining))
                if not chunk:
                    raise ProtocolError("original truncated while copying")
                hashed.update(chunk)
                _write_all(target_fd, chunk)
                remaining -= len(chunk)
            if os.read(source_fd, 1) or identity(os.fstat(source_fd)) != identity(before) or hashed.hexdigest() != prepare["sha256"]:
                raise ProtocolError("original identity/content changed while copying; retain partial")
            check()

        self._copy_new(admission_id, "input/" + prepare["filename"], None, copy_to=copy_to)

    def open_owned(self, admission_id: str, relative: str) -> int:
        record = self.read(admission_id)
        if relative not in record["inventory"]:
            raise ProtocolError("file is outside intended outbox inventory")
        directory = self._namespace(admission_id)
        child = None
        try:
            parts = relative.split("/")
            if len(parts) == 2:
                child = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
                child_info = os.fstat(child)
                if child_info.st_uid != os.geteuid() or child_info.st_gid != self.gid or child_info.st_mode & 0o027:
                    raise ProtocolError("owned outbox subdirectory identity/mode differs")
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=child if child is not None else directory)
            info = os.fstat(fd)
            prepare = validate_journal(record)
            maximum = prepare["size_bytes"] if relative.startswith("input/") else MAX_EVIDENCE_BYTES
            if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222 or info.st_nlink != 1 or
                    info.st_uid != os.geteuid() or info.st_gid != self.gid or not 0 < info.st_size <= maximum or
                    (relative.startswith("input/") and info.st_size != maximum)):
                os.close(fd)
                raise ProtocolError("owned immutable outbox input invalid")
            return fd
        finally:
            if child is not None:
                os.close(child)
            os.close(directory)

    def _copy_new(self, admission_id: str, relative: str, content: bytes | None, *, copy_to=None):
        directory = self._namespace(admission_id)
        child = None
        try:
            parts = relative.split("/")
            if len(parts) == 2:
                try:
                    os.mkdir(parts[0], 0o750, dir_fd=directory)
                    fresh = True
                except FileExistsError:
                    fresh = False
                child = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
                if fresh:
                    os.fchown(child, -1, self.gid)
                    os.fsync(directory)
                info = os.fstat(child)
                if info.st_uid != os.geteuid() or info.st_gid != self.gid or info.st_mode & 0o027:
                    raise ProtocolError("new output directory identity differs")
            target = child if child is not None else directory
            fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o440, dir_fd=target)
            try:
                os.fchown(fd, -1, self.gid)  # Never an existing/shared source inode.
                if copy_to is None:
                    _write_all(fd, content)
                else:
                    copy_to(fd)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(target)
        finally:
            if child is not None:
                os.close(child)
            os.close(directory)
