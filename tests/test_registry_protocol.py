"""Cross-language exact-wire and receipt encoding, without admission side effects."""
import copy

import pytest

from quarantine import registry_protocol as protocol


def fixture_values():
    bundle = protocol.policy_bundle({name: "b" * 64 for name in protocol.POLICY_COMPONENTS})
    context = {"pipeline_sha256": "f" * 64, "require_behavior": True}
    prepare = {"protocol_version": 2, "admission_id": "1" * 64, "name": "demo", "filename": "demo.safetensors",
               "artifact_kind": "file", "format": "safetensors", "sha256": "a" * 64, "size_bytes": 4096,
               "source": "", "source_revision": "", "policy_bundle": bundle,
               "context_sha256": protocol.context_digest(bundle["sha256"], context), "evidence_budget_bytes": 1048576}
    receipt = {"version": 2, "phase": "prepared", "admission_id": "1" * 64, "request_sha256": "e" * 64,
               "binding_sha256": protocol.binding_digest(prepare), "boot_id": "12345678-1234-1234-1234-123456789abc",
               "expires_boot_ns": "1800000000000", "reserved_registry_bytes": 1052672, "reserved_outbox_bytes": 9441280,
               "event_sequence": 2, "created_at": "2026-09-06T00:00:00Z", "sha256": ""}
    receipt["sha256"] = protocol.receipt_digest(receipt)
    return prepare, context, receipt


def test_python_go_shared_exact_encoding_vectors():
    prepare, _context, receipt = fixture_values()
    assert prepare["policy_bundle"]["sha256"] == "947248c6ec383a8d8da90503ede34f809b2c28dca5759bdb478f4dc813ddc4b3"
    assert prepare["context_sha256"] == "0c3d9affdfa135aed1db32e9b2491963f91449c0b57c3dbd0f00b6c4cca07d1d"
    assert protocol.binding_digest(prepare) == "167058874bb02196ca286fbc7ca86afa590699505ec035be2578450189f76939"
    assert receipt["sha256"] == "d1e67413fc1ca19f1afbd52ae7dcad8fafe06c7a5134efd3aedb7673ab173361"
    assert protocol.request_digest("prepare", b'{"protocol_version":2}') == "582cd85f99e1de71cda0f5f19df60a612a2d5ca2df36f53c2434bd44e6be88f8"
    protocol.validate_receipt(receipt, "prepared", prepare, "e" * 64)


@pytest.mark.parametrize("field,value", [
    ("expires_boot_ns", "1800000000001"), ("reserved_registry_bytes", 1), ("version", True),
    ("request_sha256", "f" * 64), ("event_sequence", 3), ("phase", "committed"),
])
def test_changed_receipt_never_authorizes_retry(field, value):
    prepare, _context, receipt = fixture_values()
    receipt[field] = value
    with pytest.raises(protocol.ProtocolError):
        protocol.validate_receipt(receipt, "prepared", prepare, "e" * 64)


@pytest.mark.parametrize("raw", [b'{"id":1,"id":2}', b'{"x":{"id":1,"id":2}}', b'{"x":NaN}', b'{"x":1e309}', b'{}{}', b'{"x":"\xff"}'])
def test_invalid_json_is_not_normalized(raw):
    with pytest.raises(protocol.ProtocolError):
        protocol.strict_json(raw)


def test_exact_request_bytes_and_phase_are_separate():
    prepare, context, _receipt = fixture_values()
    raw = protocol.encode_request(prepare)
    assert protocol.strict_json(raw) == prepare
    assert protocol.request_digest("prepare", raw) != protocol.request_digest("prepare", b" " + raw)
    assert protocol.request_digest("prepare", raw) != protocol.request_digest("commit", raw)
    changed = copy.deepcopy(context)
    changed["require_behavior"] = False
    assert protocol.context_digest(prepare["policy_bundle"]["sha256"], context) != protocol.context_digest(prepare["policy_bundle"]["sha256"], changed)


def capabilities_fixture():
    prepare, context, _receipt = fixture_values()
    return {"service": "ai-model-registry", "protocol_version": 2, "mode": protocol.MODE,
            "artifact_kinds": ["file"], "formats": ["gguf", "safetensors"], "scanner_evidence_version": 2,
            "policy_bundle": prepare["policy_bundle"], "pipeline_sha256": context["pipeline_sha256"],
            "scan_context": context, "context_sha256": prepare["context_sha256"],
            "receipt_lookup": "/v1/model?name={name}", "persistence_ready": True,
            "limits": {"max_body_bytes": 65536, "max_evidence_files": 32, "max_evidence_total_bytes": 1048576,
                       "max_artifact_record_bytes": 65536, "max_prepared": 32, "max_concurrent_commits": 2,
                       "prepare_lifetime_seconds": 1800, "commit_work_seconds": 20,
                       "max_artifact_bytes": 1048576, "retained_byte_limit": 536870912, "min_free_bytes": 1048576,
                       "outbox_overhead_bytes": 8388608, "manifest_scratch_bytes": 16777216}}


@pytest.mark.parametrize("field,value", [("service", "main-registry"), ("mode", "legacy"),
    ("protocol_version", True), ("persistence_ready", False), ("artifact_kinds", ["file", "directory"]),
    ("scan_context", None), ("scan_context", {"pipeline_sha256": "f" * 64, "require_behavior": 0}),
    ("receipt_lookup", "https://other.example/receipt"), ("unknown", 1)])
def test_capabilities_refuse_wrong_or_incomplete_backend(field, value):
    capabilities = capabilities_fixture()
    prepare, context, _receipt = fixture_values()
    protocol.validate_capabilities(capabilities, prepare["policy_bundle"], context["pipeline_sha256"])
    capabilities[field] = value
    with pytest.raises(protocol.ProtocolError):
        protocol.validate_capabilities(capabilities, prepare["policy_bundle"], context["pipeline_sha256"])


@pytest.mark.parametrize("suffix", ["?token=secret", "#fragment", "/nested", "%00", ""])
def test_source_profile_does_not_normalize_ambiguous_urls(suffix):
    source = "https://huggingface.co/owner/repo/resolve/" + "a" * 40 + "/demo.gguf"
    if suffix:
        with pytest.raises(protocol.ProtocolError):
            protocol.validate_source(source + suffix, "a" * 40, "demo.gguf")
    else:
        protocol.validate_source(source, "a" * 40, "demo.gguf")


def test_prepared_readback_requires_exact_identity_and_receipt():
    prepare, context, receipt = fixture_values()
    raw = protocol.encode_request(prepare)
    receipt["request_sha256"] = protocol.request_digest("prepare", raw)
    receipt["sha256"] = protocol.receipt_digest(receipt)
    artifact = {key: prepare[key] for key in ("name", "filename", "format", "sha256", "size_bytes", "source", "source_revision")}
    artifact.update(state="quarantined", admission={"version": 2, "context": context, "prepare": prepare, "prepare_receipt": receipt})
    protocol.validate_artifact(artifact, raw, context)
    for field, value in [("filename", "other.safetensors"), ("state", "trusted"), ("size_bytes", 1), ("admission", {})]:
        changed = copy.deepcopy(artifact)
        changed[field] = value
        with pytest.raises(protocol.ProtocolError):
            protocol.validate_artifact(changed, raw, context)
