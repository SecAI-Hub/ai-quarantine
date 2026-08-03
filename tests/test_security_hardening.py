"""Security boundary regression tests."""

import json
import struct
from pathlib import Path

import pytest

import quarantine.watcher as watcher
from quarantine.audit_chain import AuditChain
from quarantine.pipeline import (
    check_hash_pin,
    check_smoke_test,
    check_source_policy,
    check_static_scan,
)


def _minimal_gguf(path: Path) -> Path:
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 0)
    )
    return path


def test_hmac_audit_checkpoint_detects_tail_deletion(tmp_path):
    key = tmp_path / "audit.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    log_path = tmp_path / "audit.jsonl"
    chain = AuditChain(log_path, key_path=str(key))
    chain.append("one", {"value": 1})
    chain.append("two", {"value": 2})
    assert AuditChain.verify(str(log_path), key_path=str(key))["valid"]

    log_path.write_text(log_path.read_text().splitlines()[0] + "\n")
    verification = AuditChain.verify(str(log_path), key_path=str(key))
    assert not verification["valid"]
    assert "checkpoint" in verification["detail"]


def test_hmac_audit_checkpoint_detects_complete_log_deletion(tmp_path):
    key = tmp_path / "audit.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    log_path = tmp_path / "audit.jsonl"
    AuditChain(log_path, key_path=str(key)).append("one", {})
    log_path.unlink()
    verification = AuditChain.verify(str(log_path), key_path=str(key))
    assert not verification["valid"]
    with pytest.raises(RuntimeError, match="refusing to append"):
        AuditChain(log_path, key_path=str(key))


def test_audit_refuses_corrupt_existing_chain(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    log_path.write_text('{"not":"a valid chain"}\n')
    with pytest.raises(RuntimeError, match="refusing to append"):
        AuditChain(log_path)


def test_audit_rejects_insecure_hmac_key(tmp_path):
    key = tmp_path / "audit.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o640)
    with pytest.raises(RuntimeError, match="unavailable or insecure"):
        AuditChain(tmp_path / "audit.jsonl", key_path=str(key))


@pytest.mark.parametrize("contents", [b"k" * 31, b"k" * 32 + b"\n", b"k" * 31 + b"\x7f"])
def test_audit_rejects_weak_or_control_bearing_hmac_key(tmp_path, contents):
    key = tmp_path / "audit.key"
    key.write_bytes(contents)
    key.chmod(0o600)
    with pytest.raises(RuntimeError, match="unavailable or insecure"):
        AuditChain(tmp_path / "audit.jsonl", key_path=str(key))


def test_registry_url_validation_rejects_unsafe_origins(monkeypatch):
    invalid = [
        "http://registry.example:8470",
        "https://user@registry.example",
        "https://registry.example/api",
        "https://registry.example?token=secret",
    ]
    for value in invalid:
        monkeypatch.setattr(watcher, "REGISTRY_URL", value)
        with pytest.raises(ValueError):
            watcher._validated_registry_base_url()


def test_registry_token_file_must_be_secure(monkeypatch, tmp_path):
    token = tmp_path / "token"
    token.write_text("a-secure-token-that-is-at-least-32b")
    token.chmod(0o600)
    monkeypatch.delenv("SERVICE_TOKEN", raising=False)
    monkeypatch.setattr(watcher, "SERVICE_TOKEN_PATH", str(token))
    assert watcher._load_service_token() == "a-secure-token-that-is-at-least-32b"

    token.chmod(0o640)
    with pytest.raises(RuntimeError, match="unavailable"):
        watcher._load_service_token()


def test_registry_token_symlink_is_rejected(monkeypatch, tmp_path):
    token = tmp_path / "token"
    token.write_text("a-secure-token-that-is-at-least-32b")
    token.chmod(0o600)
    link = tmp_path / "token-link"
    link.symlink_to(token)
    monkeypatch.delenv("SERVICE_TOKEN", raising=False)
    monkeypatch.setattr(watcher, "SERVICE_TOKEN_PATH", str(link))
    with pytest.raises(RuntimeError, match="unavailable"):
        watcher._load_service_token()


def test_registry_token_hardlink_is_rejected(monkeypatch, tmp_path):
    token = tmp_path / "token"
    token.write_text("a-secure-token-that-is-at-least-32b")
    token.chmod(0o600)
    hardlink = tmp_path / "token-hardlink"
    hardlink.hardlink_to(token)
    monkeypatch.delenv("SERVICE_TOKEN", raising=False)
    monkeypatch.setattr(watcher, "SERVICE_TOKEN_PATH", str(token))
    with pytest.raises(RuntimeError, match="unavailable"):
        watcher._load_service_token()


@pytest.mark.parametrize(
    "token",
    ["short-token", "x" * 31, "x" * 32 + "\n", "x" * 31 + "\x7f"],
)
def test_registry_token_rejects_weak_or_control_bearing_values(monkeypatch, token):
    monkeypatch.setenv("SERVICE_TOKEN", token)
    with pytest.raises(ValueError):
        watcher._load_service_token()


def test_promotion_requires_authentication(monkeypatch):
    monkeypatch.delenv("SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("SECAI_ALLOW_INSECURE_REGISTRY", raising=False)
    monkeypatch.setattr(watcher, "SERVICE_TOKEN_PATH", "")
    monkeypatch.setattr(watcher, "REGISTRY_URL", "http://127.0.0.1:8470")
    assert not watcher.promote_to_registry("model.gguf", "a" * 64, 24, {})


def test_source_sidecar_symlink_is_rejected(tmp_path):
    artifact = _minimal_gguf(tmp_path / "model.gguf")
    target = tmp_path / "target"
    target.write_text("https://huggingface.co/model.gguf")
    (tmp_path / ".model.gguf.source").symlink_to(target)
    with pytest.raises(ValueError, match="unsafe"):
        watcher._read_source_metadata(artifact)


def test_failed_registry_update_leaves_durable_pending_bundle(monkeypatch, tmp_path):
    quarantine_dir = tmp_path / "quarantine"
    registry_dir = tmp_path / "registry"
    quarantine_dir.mkdir()
    registry_dir.mkdir()
    artifact = _minimal_gguf(quarantine_dir / "model.gguf")
    audit = AuditChain(tmp_path / "audit.jsonl")

    monkeypatch.setattr(watcher, "QUARANTINE_DIR", quarantine_dir)
    monkeypatch.setattr(watcher, "REGISTRY_DIR", registry_dir)
    monkeypatch.setattr(watcher, "_audit_chain", audit)
    monkeypatch.setattr(watcher, "load_policy", lambda: {})
    monkeypatch.setattr(
        watcher,
        "run_pipeline",
        lambda *_args, **_kwargs: {
            "passed": True,
            "reason": "all_checks_passed",
            "details": {},
        },
    )
    monkeypatch.setattr(watcher, "_enable_fsverity", lambda _path: False)
    monkeypatch.setattr(watcher, "promote_to_registry", lambda *_args, **_kwargs: False)

    assert not watcher.process_artifact(artifact)
    assert not artifact.exists()
    assert (registry_dir / artifact.name).exists()
    assert (registry_dir / f"{artifact.name}.provenance.json").exists()
    assert list((registry_dir / ".promotion-journal").glob("*.json"))
    events = [json.loads(line)["event"] for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert "promotion_pending" in events


def test_registry_collision_does_not_overwrite(monkeypatch, tmp_path):
    quarantine_dir = tmp_path / "quarantine"
    registry_dir = tmp_path / "registry"
    quarantine_dir.mkdir()
    registry_dir.mkdir()
    artifact = _minimal_gguf(quarantine_dir / "model.gguf")
    existing = registry_dir / "model.gguf"
    existing.write_bytes(b"trusted")

    monkeypatch.setattr(watcher, "REGISTRY_DIR", registry_dir)
    monkeypatch.setattr(watcher, "_audit_chain", AuditChain(tmp_path / "audit.jsonl"))
    monkeypatch.setattr(watcher, "load_policy", lambda: {})
    monkeypatch.setattr(
        watcher,
        "run_pipeline",
        lambda *_args, **_kwargs: {"passed": True, "details": {}},
    )

    assert not watcher.process_artifact(artifact)
    assert artifact.exists()
    assert existing.read_bytes() == b"trusted"


def test_production_policy_rejects_local_tofu_and_missing_scanners(tmp_path):
    assert not check_source_policy(
        "", policy={"models": {"allow_local_imports": False}}
    )["passed"]
    assert not check_hash_pin(
        "model.gguf",
        "a" * 64,
        policy={"models": {"require_hash_pin_for_local": True}},
    )["passed"]
    artifact = _minimal_gguf(tmp_path / "model.gguf")
    result = check_static_scan(
        artifact,
        policy={
            "models": {"require_scan": False},
            "quarantine": {"scanner_missing": "fail-closed"},
        },
    )
    assert not result["passed"]


def test_required_behavior_test_requires_isolated_worker(monkeypatch, tmp_path):
    monkeypatch.delenv("SECAI_BEHAVIOR_SANDBOXED", raising=False)
    result = check_smoke_test(
        _minimal_gguf(tmp_path / "model.gguf"),
        policy={"quarantine": {"scanner_missing": "fail-closed"}},
        required=True,
    )
    assert not result["passed"]
    assert "isolated" in result["reason"]
