"""Strict helper-admission v2 wire primitives, independent of appliance promotion.

Requests are retained/replayed byte for byte. Receipts use length-prefixed UTF-8
fields so verification never depends on Python/Go JSON reserialization details.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct

PROTOCOL_VERSION = 2
MODE = "helper-v2-required"
MAX_BODY_BYTES = 64 * 1024
MAX_EVIDENCE_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024**3
MAX_RETAINED_BYTES = 1 << 50
OUTBOX_OVERHEAD_BYTES = 8 * 1024**2
POLICY_COMPONENTS = ("policy.yaml", "models.lock.yaml", "sources.allowlist.yaml")
HEX = re.compile(r"[0-9a-f]{64}\Z")
BOOT_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class ProtocolError(ValueError):
    """Admission cannot be established from this response or retained evidence."""


def digest(domain: str, *fields: str) -> str:
    result = hashlib.sha256(domain.encode("ascii") + b"\0")
    for field in fields:
        raw = field.encode("utf-8", errors="strict")
        result.update(struct.pack(">Q", len(raw)))
        result.update(raw)
    return result.hexdigest()


def request_digest(phase: str, raw: bytes) -> str:
    if phase not in {"prepare", "commit"}:
        raise ProtocolError("unknown request phase")
    return digest("HelperAdmissionRequest-v2", phase, raw.decode("utf-8", errors="strict"))


def encode_request(value: dict) -> bytes:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if not 0 < len(raw) <= MAX_BODY_BYTES:
        raise ProtocolError("request exceeds helper admission limit")
    return raw


def strict_json(raw: bytes, *, limit: int = MAX_BODY_BYTES):
    if not 0 < len(raw) <= limit:
        raise ProtocolError("JSON size invalid")

    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ProtocolError("duplicate JSON key")
            result[name] = value
        return result

    def nonfinite(_value):
        raise ProtocolError("nonfinite JSON value")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ProtocolError("nonfinite JSON number")
        return parsed

    try:
        return json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=unique, parse_constant=nonfinite, parse_float=finite_float)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ProtocolError("invalid strict JSON") from error


def policy_bundle(component_hashes: dict[str, str]) -> dict:
    if not isinstance(component_hashes, dict) or set(component_hashes) != set(POLICY_COMPONENTS) or any(
        not isinstance(value, str) or HEX.fullmatch(value) is None for value in component_hashes.values()
    ):
        raise ProtocolError("policy components invalid")
    fields = [item for name in POLICY_COMPONENTS for item in (name, component_hashes[name])]
    return {"version": 1, "sha256": digest("HelperAdmissionPolicy-v1", *fields), "components": dict(component_hashes)}


def context_digest(bundle_sha256: str, context: dict) -> str:
    if (not isinstance(context, dict) or set(context) != {"pipeline_sha256", "require_behavior"} or
            type(context["require_behavior"]) is not bool or not isinstance(bundle_sha256, str) or HEX.fullmatch(bundle_sha256) is None):
        raise ProtocolError("scan context invalid")
    if not isinstance(context["pipeline_sha256"], str) or HEX.fullmatch(context["pipeline_sha256"]) is None:
        raise ProtocolError("pipeline identity invalid")
    return digest("HelperAdmissionContext-v2", bundle_sha256, context["pipeline_sha256"],
                  "true" if context["require_behavior"] else "false")


def binding_digest(prepare: dict) -> str:
    return digest("HelperAdmissionBinding-v2", prepare["admission_id"], prepare["name"], prepare["filename"],
                  prepare["artifact_kind"], prepare["format"], prepare["sha256"], str(prepare["size_bytes"]),
                  prepare["source"], prepare["source_revision"], prepare["policy_bundle"]["sha256"],
                  prepare["context_sha256"], str(prepare["evidence_budget_bytes"]))


RECEIPT_FIELDS = ("version", "phase", "admission_id", "request_sha256", "binding_sha256", "boot_id",
                  "expires_boot_ns", "reserved_registry_bytes", "reserved_outbox_bytes", "event_sequence", "created_at")


def receipt_digest(receipt: dict) -> str:
    if set(receipt) != {*RECEIPT_FIELDS, "sha256"}:
        raise ProtocolError("receipt field set invalid")
    return digest("HelperAdmissionReceipt-v2", *(str(receipt[name]) for name in RECEIPT_FIELDS))


def validate_receipt(receipt: dict, phase: str, prepare: dict, request_sha256: str) -> None:
    if not isinstance(receipt, dict) or set(receipt) != {*RECEIPT_FIELDS, "sha256"}:
        raise ProtocolError("receipt shape invalid")
    if (type(receipt["version"]) is not int or receipt["version"] != 2 or receipt["phase"] != phase
            or phase not in {"prepared", "committed"} or receipt["admission_id"] != prepare["admission_id"]
            or receipt["request_sha256"] != request_sha256 or receipt["binding_sha256"] != binding_digest(prepare)):
        raise ProtocolError("receipt identity mismatch")
    for name in ("request_sha256", "binding_sha256", "sha256"):
        if not isinstance(receipt[name], str) or HEX.fullmatch(receipt[name]) is None:
            raise ProtocolError("receipt digest invalid")
    if not isinstance(receipt["boot_id"], str) or BOOT_ID.fullmatch(receipt["boot_id"]) is None:
        raise ProtocolError("receipt boot identity invalid")
    expiry = receipt["expires_boot_ns"]
    if not isinstance(expiry, str) or not re.fullmatch(r"[1-9][0-9]{0,18}", expiry) or int(expiry) > 2**63-1:
        raise ProtocolError("receipt clock invalid")
    reserve = prepare["size_bytes"] + prepare["evidence_budget_bytes"]
    if (type(receipt["reserved_registry_bytes"]) is not int or receipt["reserved_registry_bytes"] != reserve
            or type(receipt["reserved_outbox_bytes"]) is not int
            or receipt["reserved_outbox_bytes"] != reserve + OUTBOX_OVERHEAD_BYTES):
        raise ProtocolError("receipt reservation mismatch")
    if type(receipt["event_sequence"]) is not int or not 1 <= receipt["event_sequence"] <= 100000:
        raise ProtocolError("receipt sequence invalid")
    if not isinstance(receipt["created_at"], str) or len(receipt["created_at"]) > 64:
        raise ProtocolError("receipt timestamp invalid")
    if receipt["sha256"] != receipt_digest(receipt):
        raise ProtocolError("receipt encoding mismatch")


def validate_source(source: str, revision: str, filename: str) -> None:
    if not isinstance(source, str) or not isinstance(revision, str):
        raise ProtocolError("source profile must be text")
    if source == "" and revision == "":
        return
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ProtocolError("source requires an exact immutable revision")
    match = re.fullmatch(r"https://huggingface[.]co/([^/]+)/([^/]+)/resolve/([0-9a-f]{40})/([^/]+)", source)
    if (match is None or not NAME.fullmatch(match[1]) or not NAME.fullmatch(match[2]) or
            match[3] != revision or match[4] != filename):
        raise ProtocolError("source does not bind the exact admitted filename/revision")


def validate_capabilities(value: dict, bundle: dict, pipeline_sha256: str) -> dict:
    fields = {"service", "protocol_version", "mode", "artifact_kinds", "formats", "scanner_evidence_version",
              "policy_bundle", "pipeline_sha256", "scan_context", "context_sha256", "receipt_lookup", "persistence_ready", "limits"}
    if (not isinstance(value, dict) or set(value) != fields or value["service"] != "ai-model-registry" or
            type(value["protocol_version"]) is not int or value["protocol_version"] != 2 or value["mode"] != MODE or
            value["artifact_kinds"] != ["file"] or value["formats"] != ["gguf", "safetensors"] or
            type(value["scanner_evidence_version"]) is not int or value["scanner_evidence_version"] != 2 or
            value["persistence_ready"] is not True or value["receipt_lookup"] != "/v1/model?name={name}" or
            value["policy_bundle"] != bundle or value["pipeline_sha256"] != pipeline_sha256):
        raise ProtocolError("configured registry is not the matching ready helper-v2 backend")
    if (not isinstance(value["scan_context"], dict) or value["scan_context"].get("pipeline_sha256") != pipeline_sha256 or
            value["context_sha256"] != context_digest(bundle["sha256"], value["scan_context"])):
        raise ProtocolError("registry scan context differs")
    fixed = {"max_body_bytes": MAX_BODY_BYTES, "max_evidence_files": 32,
             "max_evidence_total_bytes": MAX_EVIDENCE_BYTES, "max_artifact_record_bytes": MAX_BODY_BYTES,
             "max_prepared": 32, "max_concurrent_commits": 2, "prepare_lifetime_seconds": 1800,
             "commit_work_seconds": 20, "outbox_overhead_bytes": OUTBOX_OVERHEAD_BYTES,
             "manifest_scratch_bytes": 16 * 1024**2}
    limits = value["limits"]
    if not isinstance(limits, dict) or set(limits) != set(fixed) | {"max_artifact_bytes", "retained_byte_limit", "min_free_bytes"}:
        raise ProtocolError("registry limit field set invalid")
    if any(type(limits[key]) is not int or limits[key] != expected for key, expected in fixed.items()):
        raise ProtocolError("registry fixed limit profile differs")
    if (any(type(limits[key]) is not int for key in ("max_artifact_bytes", "retained_byte_limit", "min_free_bytes")) or
            not 0 < limits["max_artifact_bytes"] <= MAX_ARTIFACT_BYTES or
            not 0 < limits["retained_byte_limit"] <= MAX_RETAINED_BYTES or
            not 0 < limits["min_free_bytes"] < limits["retained_byte_limit"]):
        raise ProtocolError("registry storage budget invalid")
    return value


def validate_preparation(value: dict) -> None:
    fields = {"protocol_version", "admission_id", "name", "filename", "artifact_kind", "format", "sha256",
              "size_bytes", "source", "source_revision", "policy_bundle", "context_sha256", "evidence_budget_bytes"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ProtocolError("preparation fields invalid")
    if (type(value["protocol_version"]) is not int or value["protocol_version"] != 2 or
            any(not isinstance(value[key], str) or HEX.fullmatch(value[key]) is None
                for key in ("admission_id", "sha256", "context_sha256")) or
            any(not isinstance(value[key], str) or NAME.fullmatch(value[key]) is None for key in ("name", "filename")) or
            value["artifact_kind"] != "file" or not isinstance(value["format"], str) or value["format"] not in {"gguf", "safetensors"} or
            not value["filename"].endswith("." + value["format"]) or
            type(value["size_bytes"]) is not int or not 0 < value["size_bytes"] <= MAX_ARTIFACT_BYTES or
            type(value["evidence_budget_bytes"]) is not int or not 0 < value["evidence_budget_bytes"] <= MAX_EVIDENCE_BYTES):
        raise ProtocolError("preparation identity/limits invalid")
    bundle = value["policy_bundle"]
    if (not isinstance(bundle, dict) or set(bundle) != {"version", "sha256", "components"} or
            type(bundle["version"]) is not int or policy_bundle(bundle["components"]) != bundle):
        raise ProtocolError("preparation bundle invalid")
    validate_source(value["source"], value["source_revision"], value["filename"])


def validate_commit(value: dict) -> None:
    if (not isinstance(value, dict) or set(value) != {"protocol_version", "admission_id", "prepare_receipt_sha256", "evidence"} or
            type(value["protocol_version"]) is not int or value["protocol_version"] != 2 or
            any(not isinstance(value[key], str) or HEX.fullmatch(value[key]) is None for key in ("admission_id", "prepare_receipt_sha256")) or
            not isinstance(value["evidence"], list) or not 1 <= len(value["evidence"]) <= 32):
        raise ProtocolError("commit field set/identity invalid")
    names, roles, total = set(), set(), 0
    for evidence in value["evidence"]:
        if (not isinstance(evidence, dict) or set(evidence) != {"role", "filename", "sha256", "size_bytes"} or
                not isinstance(evidence["role"], str) or evidence["role"] not in {"pipeline-report", "gguf-integrity"} or
                not isinstance(evidence["filename"], str) or NAME.fullmatch(evidence["filename"]) is None or
                evidence["filename"] in {"intent.json", "ready"} or evidence["filename"] in names or evidence["role"] in roles or
                not isinstance(evidence["sha256"], str) or HEX.fullmatch(evidence["sha256"]) is None or
                type(evidence["size_bytes"]) is not int or not 0 < evidence["size_bytes"] <= MAX_EVIDENCE_BYTES):
            raise ProtocolError("commit evidence descriptor invalid")
        names.add(evidence["filename"])
        roles.add(evidence["role"])
        total += evidence["size_bytes"]
    if total > MAX_EVIDENCE_BYTES or "pipeline-report" not in roles:
        raise ProtocolError("commit evidence total/report missing")


def validate_artifact(value: dict, prepare_raw: bytes, context: dict, commit_raw: bytes | None = None) -> dict:
    """Validate historical receipts separately from the row's current state."""
    prepare = strict_json(prepare_raw)
    validate_preparation(prepare)
    fields = {"name", "filename", "format", "sha256", "size_bytes", "source", "scan_results", "scanner_versions",
              "policy_version", "source_revision", "gguf_guard_fingerprint", "gguf_guard_manifest", "object_path",
              "promoted_at", "state", "admission"}
    if (not isinstance(value, dict) or set(value) - fields or not isinstance(value.get("state"), str) or
            value["state"] not in {"quarantined", "trusted", "revoked", "deleted"}):
        raise ProtocolError("admission artifact shape/state invalid")
    for key in ("name", "filename", "format", "sha256", "size_bytes", "source", "source_revision"):
        # Go omits empty legacy strings; the preparation still binds them.
        if value.get(key, "") != prepare[key]:
            raise ProtocolError("artifact identity differs from exact preparation")
    admission = value.get("admission")
    required = {"version", "context", "prepare", "prepare_receipt"}
    committed = {"commit", "commit_receipt", "report"}
    if (not isinstance(admission, dict) or not required <= set(admission) or
            set(admission) - required not in (set(), committed) or
            type(admission["version"]) is not int or admission["version"] != 2 or
            admission["context"] != context or admission["prepare"] != prepare or
            prepare["context_sha256"] != context_digest(prepare["policy_bundle"]["sha256"], context)):
        raise ProtocolError("artifact admission binding invalid")
    validate_receipt(admission["prepare_receipt"], "prepared", prepare, request_digest("prepare", prepare_raw))
    if committed <= set(admission):
        if commit_raw is None or admission["commit"] != strict_json(commit_raw):
            raise ProtocolError("historical commit lacks the exact expected request")
        validate_commit(admission["commit"])
        receipt = admission["commit_receipt"]
        validate_receipt(receipt, "committed", prepare, request_digest("commit", commit_raw))
        for key in ("boot_id", "expires_boot_ns", "reserved_registry_bytes", "reserved_outbox_bytes", "binding_sha256"):
            if receipt[key] != admission["prepare_receipt"][key]:
                raise ProtocolError("commit receipt renewed/changed preparation authority")
        if (receipt["event_sequence"] <= admission["prepare_receipt"]["event_sequence"] or
                admission["commit"]["prepare_receipt_sha256"] != admission["prepare_receipt"]["sha256"]):
            raise ProtocolError("commit sequence/preparation receipt differs")
    elif value["state"] == "trusted":
        raise ProtocolError("trusted row lacks a complete commit receipt")
    return value
