"""Promotion transaction, queue, and container boundary regressions."""

import json
import struct
import threading
from pathlib import Path

import quarantine.watcher as watcher
from quarantine.audit_chain import AuditChain


def _minimal_gguf(path: Path) -> Path:
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 0)
    )
    return path


def _configure_passing_watcher(monkeypatch, tmp_path, *, details=None):
    quarantine_dir = tmp_path / "quarantine"
    registry_dir = tmp_path / "registry"
    quarantine_dir.mkdir()
    registry_dir.mkdir()
    monkeypatch.setattr(watcher, "QUARANTINE_DIR", quarantine_dir)
    monkeypatch.setattr(watcher, "REGISTRY_DIR", registry_dir)
    monkeypatch.setattr(
        watcher,
        "_audit_chain",
        AuditChain(tmp_path / "audit.jsonl"),
    )
    monkeypatch.setattr(watcher, "load_policy", lambda: {})
    monkeypatch.setattr(
        watcher,
        "run_pipeline",
        lambda *_args, **_kwargs: {
            "passed": True,
            "reason": "all_checks_passed",
            "details": details or {},
        },
    )
    monkeypatch.setattr(watcher, "_enable_fsverity", lambda _path: False)
    return quarantine_dir, registry_dir


def test_registry_api_only_runs_after_artifact_evidence_and_journal_exist(
    monkeypatch, tmp_path
):
    guard = tmp_path / "model.gguf.gguf-guard.json"
    guard.write_text('{"verdict":"pass"}')
    details = {
        "gguf_guard_manifest": {
            "generated": True,
            "manifest_path": str(guard),
        }
    }
    quarantine_dir, registry_dir = _configure_passing_watcher(
        monkeypatch, tmp_path, details=details
    )
    artifact = _minimal_gguf(quarantine_dir / "model.gguf")
    observed = {}

    def commit(*_args, **kwargs):
        observed["called"] = True
        assert (registry_dir / "model.gguf").is_file()
        assert (registry_dir / "model.gguf.provenance.json").is_file()
        assert (registry_dir / "model.gguf.gguf-guard.json").is_file()
        pending = list((registry_dir / ".promotion-journal").glob("*.json"))
        assert len(pending) == 1
        record = json.loads(pending[0].read_text())
        assert record["promotion_id"] == kwargs["promotion_id"]
        assert record["payload"] == kwargs["payload_override"]
        evidence_names = {item["filename"] for item in kwargs["evidence"]}
        assert {
            "model.gguf.provenance.json",
            "model.gguf.gguf-guard.json",
        } <= evidence_names
        return True

    monkeypatch.setattr(watcher, "promote_to_registry", commit)
    assert watcher.process_artifact(artifact)
    assert observed == {"called": True}
    assert not list((registry_dir / ".promotion-journal").glob("*.json"))


def test_pending_registry_commit_is_restart_safe(monkeypatch, tmp_path):
    quarantine_dir, registry_dir = _configure_passing_watcher(monkeypatch, tmp_path)
    artifact = _minimal_gguf(quarantine_dir / "model.gguf")
    monkeypatch.setattr(watcher, "promote_to_registry", lambda *_a, **_kw: False)

    assert not watcher.process_artifact(artifact)
    pending = list((registry_dir / ".promotion-journal").glob("*.json"))
    assert len(pending) == 1
    promotion_id = json.loads(pending[0].read_text())["promotion_id"]
    commits = []

    def commit(*_args, **kwargs):
        commits.append(kwargs["promotion_id"])
        assert (registry_dir / "model.gguf").exists()
        assert (registry_dir / "model.gguf.provenance.json").exists()
        assert pending[0].exists()
        return True

    monkeypatch.setattr(watcher, "promote_to_registry", commit)
    assert watcher.recover_pending_promotions(now=100) == 1
    assert commits == [promotion_id]
    assert not pending[0].exists()


def test_restart_rolls_incomplete_staging_back_before_any_api_call(monkeypatch, tmp_path):
    quarantine_dir, registry_dir = _configure_passing_watcher(monkeypatch, tmp_path)
    artifact = _minimal_gguf(quarantine_dir / "model.gguf")
    file_hash = watcher.sha256_file(artifact)
    staging_path = watcher._begin_promotion_staging(
        artifact.name,
        file_hash,
        artifact.stat().st_size,
        "llm",
        {},
    )
    artifact.replace(registry_dir / artifact.name)
    (registry_dir / f"{artifact.name}.provenance.json").write_text("partial")
    network_calls = []
    monkeypatch.setattr(
        watcher,
        "promote_to_registry",
        lambda *_args, **_kwargs: network_calls.append(True),
    )

    assert watcher.recover_pending_promotions(now=0) == 0
    assert artifact.exists()
    assert not (registry_dir / artifact.name).exists()
    assert not (registry_dir / f"{artifact.name}.provenance.json").exists()
    assert not staging_path.exists()
    assert network_calls == []


def test_missing_evidence_prevents_registry_network_call(monkeypatch, tmp_path):
    real_promote = watcher.promote_to_registry
    quarantine_dir, registry_dir = _configure_passing_watcher(monkeypatch, tmp_path)
    artifact = _minimal_gguf(quarantine_dir / "model.gguf")
    monkeypatch.setattr(watcher, "promote_to_registry", lambda *_a, **_kw: False)
    assert not watcher.process_artifact(artifact)
    pending_path = next((registry_dir / ".promotion-journal").glob("*.json"))
    record = json.loads(pending_path.read_text())
    (registry_dir / "model.gguf.provenance.json").unlink()
    network_called = []
    monkeypatch.setenv("SERVICE_TOKEN", "s" * 32)
    monkeypatch.setattr(watcher, "SERVICE_TOKEN_PATH", "")
    monkeypatch.setattr(watcher, "REGISTRY_URL", "http://127.0.0.1:8470")
    monkeypatch.setattr(watcher, "_http_urlopen", lambda *_a, **_kw: network_called.append(True))

    artifact_record = record["artifact"]
    assert not real_promote(
        artifact_record["filename"],
        artifact_record["sha256"],
        artifact_record["size_bytes"],
        record["payload"]["scan_results"],
        model_type=artifact_record["model_type"],
        source_url=record["payload"]["source"],
        promotion_id=record["promotion_id"],
        evidence=record["evidence"],
        payload_override=record["payload"],
    )
    assert network_called == []


def _configure_queue(monkeypatch, tmp_path):
    quarantine_dir = tmp_path / "quarantine"
    registry_dir = tmp_path / "registry"
    quarantine_dir.mkdir()
    registry_dir.mkdir()
    monkeypatch.setattr(watcher, "QUARANTINE_DIR", quarantine_dir)
    monkeypatch.setattr(watcher, "REGISTRY_DIR", registry_dir)
    monkeypatch.setattr(watcher, "recover_pending_promotions", lambda **_kwargs: 0)
    return quarantine_dir


def test_queue_scan_is_bounded_without_materializing_directory(monkeypatch, tmp_path):
    quarantine_dir = _configure_queue(monkeypatch, tmp_path)
    monkeypatch.setattr(watcher, "MAX_SCAN_ENTRIES", 4)
    for index in range(10):
        _minimal_gguf(quarantine_dir / f"model-{index}.gguf")
    processed = []

    def consume(path):
        processed.append(path.name)
        path.unlink()
        return True

    monkeypatch.setattr(watcher, "process_artifact", consume)
    assert watcher.scan_directory(now=0) == 4
    assert len(processed) == 4
    assert len(list(quarantine_dir.glob("*.gguf"))) == 6


def test_queue_claim_prevents_concurrent_double_processing(monkeypatch, tmp_path):
    quarantine_dir = _configure_queue(monkeypatch, tmp_path)
    artifact = _minimal_gguf(quarantine_dir / "model.gguf")
    started = threading.Event()
    release = threading.Event()
    calls = []

    def consume(path):
        calls.append(path.name)
        started.set()
        assert release.wait(timeout=5)
        path.unlink()
        return True

    monkeypatch.setattr(watcher, "process_artifact", consume)
    first = threading.Thread(target=watcher.scan_directory, kwargs={"now": 0})
    second = threading.Thread(target=watcher.scan_directory, kwargs={"now": 0})
    first.start()
    assert started.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    release.set()
    first.join(timeout=5)

    assert calls == [artifact.name]
    assert not first.is_alive()
    assert not second.is_alive()


def test_queue_backoff_and_terminal_hold_stop_permanent_retries(monkeypatch, tmp_path):
    quarantine_dir = _configure_queue(monkeypatch, tmp_path)
    artifact = _minimal_gguf(quarantine_dir / "model.gguf")
    calls = []

    def defer(path):
        calls.append(path.name)
        return False

    monkeypatch.setattr(watcher, "process_artifact", defer)
    assert watcher.scan_directory(now=0) == 1
    assert watcher.scan_directory(now=1) == 0
    assert watcher.scan_directory(now=5) == 1
    assert watcher.scan_directory(now=15) == 1
    assert calls == [artifact.name] * 3
    assert not artifact.exists()
    held = list((quarantine_dir / ".terminal-hold").glob("*/model.gguf"))
    assert len(held) == 1
    assert watcher.scan_directory(now=10_000) == 0
    assert calls == [artifact.name] * 3


def test_container_keeps_code_root_owned_and_runtime_read_only():
    text = Path("Containerfile").read_text()
    assert "COPY --chown=0:0 quarantine/" in text
    assert "find /opt/ai-quarantine -type d -exec chmod 0555" in text
    assert "find /opt/ai-quarantine -type f -exec chmod 0444" in text
    assert "install -d -m 0700 -o 65532 -g 65532 /quarantine" in text
    assert "USER 65532:65532" in text


def test_tag_publication_requires_full_verification_job():
    workflow = Path(".github/workflows/release.yml").read_text()
    assert workflow.count("needs: test") == 2
    for gate in (
        "gitleaks/gitleaks-action@",
        "python -m ruff check .",
        "python -m pytest -q",
        "python -m pip_audit",
        "python -m bandit",
        "python -m build",
    ):
        assert gate in workflow
