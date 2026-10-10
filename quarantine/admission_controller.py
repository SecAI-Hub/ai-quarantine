"""Helper-v2 orchestration; production activation remains unconditionally closed.

_Private ports are a source-test seam, never accepted by watcher, configuration,
CLI or this module's public entry points. Actual mandatory P2 probes and their
trusted deployment binding do not exist yet. No file or flag can certify them.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .admission_client import AdmissionClient, RegistryUnavailable
from .admission_journal import Outbox, identity
from .admission_process import ProcessResult, SandboxFailure, capture_owned
from .admission_sandbox import child_seccomp_fd, sandbox_command
from .admission_scan import SCANNER_FIELDS, decode_frame
from .policy_context import PolicyContext
from .registry_protocol import (
    BOOT_ID,
    MAX_EVIDENCE_BYTES,
    NAME,
    POLICY_COMPONENTS,
    ProtocolError,
    context_digest,
    encode_request,
    strict_json,
    validate_artifact,
    validate_capabilities,
    validate_commit,
    validate_preparation,
)


class RuntimeUnavailable(ProtocolError):
    pass


def require_qualified_runtime():
    # Do not replace this with a configuration assertion or a stored "passed"
    # receipt. Actual owned negative probes/runtime binding must be implemented.
    raise RuntimeUnavailable(
        "helper-v2 runtime and mandatory P2 probes are unqualified; activation disabled"
    )


def process_artifact(path: Path) -> bool:
    ports = (
        require_qualified_runtime()
    )  # FIRST: no policy parser, stat, journal or request.
    return _Controller(ports).submit(path)["state"] == "trusted"


def recover(admission_id: str):
    ports = require_qualified_runtime()
    return _Controller(ports).recover(admission_id)


@dataclass(frozen=True)
class _Ports:
    """Trusted test-only injection; no production factory currently returns it."""

    qualify: Callable
    policy: Callable[[], PolicyContext]
    client: AdmissionClient
    outbox: Path
    gid: int
    pipeline_sha256: str
    worker: Callable
    now: Callable[[], int]
    boot_id: Callable[[], str]
    cancelled: Callable[[], bool]


def _owned_worker(box, prepare, context, deadline, check):
    """Real primitive composition, unreachable without future P2 qualification."""
    descriptors = []
    try:
        model = box.open_owned(prepare["admission_id"], "input/" + prepare["filename"])
        descriptors.append(model)
        policies = {}
        for name in POLICY_COMPONENTS:
            policies[name] = box.open_owned(prepare["admission_id"], "policy/" + name)
            descriptors.append(policies[name])
        seccomp = child_seccomp_fd()
        descriptors.append(seccomp)
        command = sandbox_command(model, prepare["filename"], policies, seccomp)
        config = encode_request(
            {"prepare": prepare, "context": context, "deadline_boot_ns": str(deadline)}
        )
        return capture_owned(command, config, tuple(descriptors), deadline, check)
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def _validate_report(raw, prepare, context):
    """Existing server admission.go validateAdmissionReport field semantics."""
    report = strict_json(raw)
    bound = (
        "admission_id",
        "filename",
        "sha256",
        "size_bytes",
        "source",
        "source_revision",
        "policy_bundle",
    )
    if (
        not isinstance(report, dict)
        or set(report) != {*bound, "version", "pipeline_sha256", "gates", "scanners"}
        or type(report["version"]) is not int
        or report["version"] != 2
        or type(report["size_bytes"]) is not int
        or any(report[key] != prepare[key] for key in bound)
        or report["pipeline_sha256"] != context["pipeline_sha256"]
    ):
        raise ProtocolError("worker report binding differs")
    validate_preparation({**prepare, **{key: report[key] for key in bound}})

    def plain(value):
        return (
            isinstance(value, str)
            and 0 < len(value.encode()) <= 1024
            and all(ord(c) >= 32 and ord(c) != 127 for c in value)
        )

    scanners = report["scanners"]
    expected = {
        "modelscan": "modelscan",
        "fickling": "fickling",
        "modelaudit": "modelaudit",
        "gguf_guard": "gguf-guard",
    }
    if not isinstance(scanners, dict) or set(scanners) != set(expected):
        raise ProtocolError("worker scanner inventory differs")
    for key, scanner in expected.items():
        value = scanners[key]
        na = key in {"modelscan", "fickling"} or (
            key == "gguf_guard" and prepare["format"] == "safetensors"
        )
        if (
            not isinstance(value, dict)
            or set(value) != SCANNER_FIELDS
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 2
            or value["scanner"] != scanner
            or not plain(value["reason"])
            or type(value["scanned_count"]) is not int
            or value["scanned_count"] < 0
        ):
            raise ProtocolError("worker scanner type invalid")
        if na:
            valid = (
                value["passed"] is False
                and value["verdict"] == "not_applicable"
                and value["available"] is None
                and value["scanner_version"] is None
                and value["scanned_count"] == 0
            )
        else:
            valid = (
                value["passed"] is True
                and value["verdict"] == "pass"
                and value["available"] is True
                and plain(value["scanner_version"])
                and value["scanner_version"].strip() != "unknown"
                and value["scanned_count"] > 0
            )
        if not valid:
            raise ProtocolError("worker scanner coverage invalid")
    gates = report["gates"]
    keys = {
        "source_policy",
        "format_gate",
        "hash_pin",
        "provenance",
        "polyglot_check",
        "entropy",
        "weight_stats",
        "smoke_test",
    }
    if not isinstance(gates, dict) or set(gates) != keys:
        raise ProtocolError("worker gate inventory differs")
    for key, value in gates.items():
        verdict = "pass"
        if key == "smoke_test":
            verdict = (
                "not_applicable"
                if prepare["format"] == "safetensors"
                else "pass"
                if context["require_behavior"]
                else "not_required"
            )
        assurance = (
            ("recorded" if prepare["source"] else "none") if key == "provenance" else ""
        )
        if (
            not isinstance(value, dict)
            or set(value) != {"verdict", "reason", "count", "assurance"}
            or value["verdict"] != verdict
            or not plain(value["reason"])
            or value["assurance"] != assurance
            or type(value["count"]) is not int
            or value["count"] < 0
            or (key == "weight_stats" and value["count"] == 0)
            or (verdict != "pass" and value["count"] != 0)
        ):
            raise ProtocolError("worker gate result invalid")
    return report


class _Controller:
    def __init__(self, ports: _Ports):
        self.p = ports

    def _preflight(self, artifact_format):
        self.p.qualify()  # Before policy parsing, source hashing, outbox or transport.
        policy = self.p.policy()
        if not isinstance(policy, PolicyContext):
            raise ProtocolError("immutable policy context required")
        caps = self.p.client.preflight(
            policy.evidence(), self.p.pipeline_sha256, artifact_format=artifact_format
        )
        validate_capabilities(caps, policy.evidence(), self.p.pipeline_sha256)
        if artifact_format == "gguf" and caps["scan_context"]["require_behavior"]:
            raise ProtocolError("bounded GGUF behavior unavailable")
        self.boot = self.p.boot_id()
        self.last = self.p.now()
        if (
            not isinstance(self.boot, str)
            or BOOT_ID.fullmatch(self.boot) is None
            or type(self.last) is not int
            or self.last < 0
        ):
            raise ProtocolError("boot/clock authority unavailable")
        self.check()
        return policy, caps

    def check(self, artifact=None):
        now = self.p.now()
        if (
            self.p.cancelled()
            or self.p.boot_id() != self.boot
            or type(now) is not int
            or now < self.last
        ):
            raise ProtocolError("cancelled, changed boot or regressed clock")
        self.last = now
        if artifact is not None:
            receipt = artifact["admission"]["prepare_receipt"]
            remaining = int(receipt["expires_boot_ns"]) - now
            if (
                receipt["boot_id"] != self.boot
                or not 0 < remaining <= 1800 * 1_000_000_000
            ):
                raise ProtocolError(
                    "original preparation lease expired or belongs to another boot"
                )

    def _commit(self, box, record):
        prepare_raw, commit_raw = (
            record["prepare_wire"].encode(),
            record["commit_wire"].encode(),
        )
        self.check(record["artifact"])
        try:
            _status, artifact = self.p.client.commit(prepare_raw, commit_raw)
        except RegistryUnavailable:
            artifact = self.p.client.lookup(prepare_raw, commit_raw=commit_raw)
            if artifact is None or "commit_receipt" not in artifact["admission"]:
                raise RegistryUnavailable() from None  # Exact commit-ready evidence remains retryable.
        validate_artifact(artifact, prepare_raw, record["context"], commit_raw)
        if "commit_receipt" not in artifact["admission"]:
            raise ProtocolError("commit has no complete historical receipt")
        # Readback can resolve a committed result after expiry; never renew lease
        # or perform a new commit after expiry. Current revoked/deleted is kept.
        record.update(stage="committed", artifact=artifact, error="")
        box.update(record)
        return artifact

    def submit(self, path: Path, *, source="", source_revision=""):
        artifact_format = path.suffix.removeprefix(".")
        policy, caps = self._preflight(artifact_format)
        if NAME.fullmatch(path.name) is None or artifact_format not in {
            "gguf",
            "safetensors",
        }:
            raise ProtocolError("unsupported v2 input name/format")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        box = record = None
        try:
            original = os.fstat(fd)
            if (
                not stat.S_ISREG(original.st_mode)
                or original.st_nlink != 1
                or not 0 < original.st_size <= caps["limits"]["max_artifact_bytes"]
            ):
                raise ProtocolError("input is not a bounded single-link regular file")
            hashed = hashlib.sha256()
            remaining = original.st_size
            while remaining:
                self.check()
                block = os.read(fd, min(65536, remaining))
                if not block:
                    raise ProtocolError("input truncated while hashing")
                hashed.update(block)
                remaining -= len(block)
            if os.read(fd, 1) or identity(os.fstat(fd)) != identity(original):
                raise ProtocolError("input changed while hashing")
            context = caps["scan_context"]
            prepare = dict(
                protocol_version=2,
                admission_id=secrets.token_hex(32),
                name=path.stem,
                filename=path.name,
                artifact_kind="file",
                format=artifact_format,
                sha256=hashed.hexdigest(),
                size_bytes=original.st_size,
                source=source,
                source_revision=source_revision,
                policy_bundle=policy.evidence(),
                context_sha256=context_digest(policy.evidence()["sha256"], context),
                evidence_budget_bytes=MAX_EVIDENCE_BYTES,
            )
            validate_preparation(prepare)
            wire = encode_request(prepare)
            inventory = [
                "input/" + path.name,
                *("policy/" + name for name in POLICY_COMPONENTS),
                "capture.bin",
                "evidence/pipeline.json",
            ]
            if artifact_format == "gguf":
                inventory.append("evidence/integrity.json")
            record = dict(
                version=2,
                admission_id=prepare["admission_id"],
                stage="intent",
                prepare_wire=wire.decode(),
                commit_wire="",
                context=context,
                artifact=None,
                original=list(identity(original)),
                inventory=inventory,
                error="",
            )
            box = Outbox(self.p.outbox, self.p.gid)
            box.reserve(prepare, caps["limits"])
            self.check()
            box.create(record)
            _status, artifact = self.p.client.prepare(wire)
            validate_artifact(artifact, wire, context)
            record.update(stage="prepared", artifact=artifact)
            box.update(record)
            self.check(artifact)
            record["stage"] = "copying"
            box.update(record)
            for name, content in policy.components:
                self.check(artifact)
                box.write_new(prepare["admission_id"], "policy/" + name, content)
            box.copy_input(prepare["admission_id"], fd, lambda: self.check(artifact))
            record["stage"] = "scanning"
            box.update(record)
            deadline = int(artifact["admission"]["prepare_receipt"]["expires_boot_ns"])
            result = self.p.worker(
                box, prepare, context, deadline, lambda: self.check(artifact)
            )
            if (
                not isinstance(result, ProcessResult)
                or any(
                    type(value) is not bytes
                    for value in (result.stdout, result.stderr, result.capture)
                )
                or len(result.capture) > MAX_EVIDENCE_BYTES
                or len(result.stdout) + len(result.stderr) != len(result.capture)
                or (not result.stderr and result.stdout != result.capture)
            ):
                raise ProtocolError("worker capture invalid")
            if result.capture:
                box.write_new(prepare["admission_id"], "capture.bin", result.capture)
            self.check(artifact)
            if result.returncode != 0 or result.stderr:
                raise ProtocolError("worker did not complete cleanly")
            header, files = decode_frame(result.stdout)
            if not header["passed"]:
                raise ProtocolError("worker returned a nonpassing frame")
            roles = {descriptor["role"] for descriptor, _raw in files}
            if roles != (
                {"pipeline-report", "gguf-integrity"}
                if artifact_format == "gguf"
                else {"pipeline-report"}
            ):
                raise ProtocolError("worker evidence roles differ")
            evidence = []
            for descriptor, content in files:
                if descriptor["role"] == "pipeline-report":
                    _validate_report(content, prepare, context)
                self.check(artifact)
                box.write_new(
                    prepare["admission_id"],
                    "evidence/" + descriptor["filename"],
                    content,
                )
                evidence.append(descriptor)
            commit = dict(
                protocol_version=2,
                admission_id=prepare["admission_id"],
                prepare_receipt_sha256=artifact["admission"]["prepare_receipt"][
                    "sha256"
                ],
                evidence=evidence,
            )
            validate_commit(commit)
            record.update(
                stage="commit-ready", commit_wire=encode_request(commit).decode()
            )
            box.update(record)  # Exact encoded bytes durable BEFORE first commit.
            return self._commit(box, record)
        except Exception as error:
            if box is not None and record is not None:
                if (
                    isinstance(error, SandboxFailure)
                    and record["stage"] == "scanning"
                    and error.capture
                ):
                    box.write_new(record["admission_id"], "capture.bin", error.capture)
                if record["stage"] not in {"intent", "commit-ready", "committed"}:
                    record.update(stage="failed", error=type(error).__name__)
                    box.update(record)
            raise
        finally:
            os.close(fd)
            if box is not None:
                box.close()

    def recover(self, admission_id: str):
        # Format is learned only after qualification; initial preflight covers
        # the common file protocol. Recheck retained exact format before writes.
        policy, caps = self._preflight("safetensors")
        box = Outbox(self.p.outbox, self.p.gid)
        try:
            record = box.read(admission_id)
            prepare = strict_json(record["prepare_wire"].encode())
            if (
                prepare["policy_bundle"] != policy.evidence()
                or record["context"] != caps["scan_context"]
            ):
                raise ProtocolError("retained policy/context differs; no migration")
            if prepare["format"] == "gguf" and caps["scan_context"]["require_behavior"]:
                raise ProtocolError("bounded GGUF behavior unavailable")
            if record["artifact"] is not None:
                self.check(record["artifact"])
            if record["stage"] == "commit-ready":
                return self._commit(box, record)
            if record["stage"] == "intent":
                _status, artifact = self.p.client.prepare(
                    record["prepare_wire"].encode()
                )
                validate_artifact(
                    artifact, record["prepare_wire"].encode(), record["context"]
                )
                record.update(stage="prepared", artifact=artifact)
                box.update(record)
                self.check(artifact)
                return artifact  # No automatic restart of copy/scan or adoption.
            raise ProtocolError(
                "retained stage requires explicit reconciliation; no restart or cleanup"
            )
        finally:
            box.close()
