"""Policy identity follows immutable copied bytes, never mutable module globals."""
from dataclasses import FrozenInstanceError

import pytest

from quarantine.policy_context import MAX_COMPONENT_BYTES, PolicyContext
from quarantine.registry_protocol import ProtocolError


def values():
    return {"policy.yaml": b"models:\n  require_scan: true\n",
            "models.lock.yaml": b"models: []\n", "sources.allowlist.yaml": b"models: []\n"}


def test_policy_context_keeps_original_bytes_and_independent_readers():
    source = values()
    context = PolicyContext.from_components(source)
    evidence = context.evidence()
    source["policy.yaml"] = b"models:\n  require_scan: false\n"
    first = context.document("policy.yaml")
    first["models"]["require_scan"] = False
    assert context.document("policy.yaml")["models"]["require_scan"] is True
    assert context.evidence() == evidence
    with pytest.raises(FrozenInstanceError):
        context.components = ()
    with pytest.raises(ProtocolError):
        PolicyContext(list(context.components))


@pytest.mark.parametrize("change", ["missing", "extra", "mutable", "oversize", "bad_yaml"])
def test_policy_context_rejects_incomplete_or_unbounded_components(change):
    source = values()
    if change == "missing":
        del source["models.lock.yaml"]
    elif change == "extra":
        source["unknown.yaml"] = b"{}"
    elif change == "mutable":
        source["policy.yaml"] = bytearray(b"{}")
    elif change == "oversize":
        source["policy.yaml"] = b"x" * (MAX_COMPONENT_BYTES + 1)
    else:
        source["policy.yaml"] = b"[unclosed"
    with pytest.raises(ProtocolError):
        PolicyContext.from_components(source)


def test_scan_readers_use_context_without_global_policy_io(monkeypatch):
    from quarantine import pipeline

    source = values()
    expected = "a" * 64
    source["models.lock.yaml"] = f"models:\n  - filename: demo.gguf\n    sha256: {expected}\n".encode()
    source["sources.allowlist.yaml"] = b"models:\n  - url_prefix: https://huggingface.co/owner/repo/\n"
    context = PolicyContext.from_components(source)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("scan reread mutable host policy")

    monkeypatch.setattr(pipeline, "_load_trusted_yaml", forbidden)
    assert pipeline.check_hash_pin("demo.gguf", expected, policy_context=context)["passed"]
    assert not pipeline.check_hash_pin("demo.gguf", "b" * 64, policy_context=context)["passed"]
    assert pipeline.check_source_policy("https://huggingface.co/owner/repo/resolve/main/demo.gguf", policy_context=context)["passed"]
    assert not pipeline.check_source_policy("https://huggingface.co/other/repo/demo.gguf", policy_context=context)["passed"]


def test_orchestrator_derives_primary_policy_from_same_context(monkeypatch, tmp_path):
    from quarantine import pipeline

    source = values()
    source["policy.yaml"] = b"models:\n  allow_local_imports: false\n"
    context = PolicyContext.from_components(source)
    # No parser is reached: the immutable policy rejects before format work.
    result = pipeline.run_pipeline(tmp_path / "demo.gguf", "a" * 64,
                                   {"models": {"allow_local_imports": True}}, policy_context=context)
    assert result["passed"] is False
    assert result["reason"] == "source_policy"
