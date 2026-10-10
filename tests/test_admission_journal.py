"""Synthetic private copies/journals retain bytes and refuse uncertain reuse."""
import copy
import hashlib
import os

import pytest
from test_registry_protocol import fixture_values

from quarantine.admission_journal import Outbox, identity
from quarantine.registry_protocol import (
    MAX_EVIDENCE_BYTES,
    ProtocolError,
    binding_digest,
    encode_request,
    receipt_digest,
    request_digest,
)


def journal_fixture(tmp_path):
    prepare, context, receipt = fixture_values()
    source = tmp_path / prepare["filename"]
    source.write_bytes(b"x" * prepare["size_bytes"])
    prepare["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt["binding_sha256"] = binding_digest(prepare)
    wire = encode_request(prepare).decode()
    receipt["request_sha256"] = request_digest("prepare", wire.encode())
    receipt["sha256"] = receipt_digest(receipt)
    artifact = {key: prepare[key] for key in ("name", "filename", "format", "sha256", "size_bytes", "source", "source_revision")}
    artifact.update(state="quarantined", admission={"version": 2, "context": context, "prepare": prepare, "prepare_receipt": receipt})
    record = {"version": 2, "admission_id": prepare["admission_id"], "stage": "intent", "prepare_wire": wire,
              "commit_wire": "", "context": context, "artifact": None, "original": list(identity(source.stat())),
              "inventory": ["input/" + prepare["filename"], "policy/policy.yaml", "policy/models.lock.yaml",
                            "policy/sources.allowlist.yaml", "capture.bin", "evidence/pipeline.json"], "error": ""}
    root = tmp_path / "outbox"
    root.mkdir(mode=0o750)
    return root, source, record, artifact


def prepared_copying(box, record, artifact):
    record["artifact"], record["stage"] = artifact, "prepared"
    box.update(record)
    record["stage"] = "copying"
    box.update(record)


def test_outbox_intent_exact_bytes_exclusive_files_and_retained_terminal_debit(tmp_path):
    root, source, record, artifact = journal_fixture(tmp_path)
    original = source.read_bytes()
    box = Outbox(root, os.getgid())
    try:
        box.create(record)
        assert box.read(record["admission_id"]) == record
        with pytest.raises(FileExistsError):
            box.create(record)
        prepared_copying(box, record, artifact)
        box.write_new(record["admission_id"], "policy/policy.yaml", b"models: {}\n")
        with pytest.raises(FileExistsError):
            box.write_new(record["admission_id"], "policy/policy.yaml", b"changed")
        retained = root / record["admission_id"] / "policy/policy.yaml"
        assert retained.read_bytes() == b"models: {}\n" and retained.stat().st_mode & 0o777 == 0o440
        record["stage"] = "failed"
        box.update(record)
        assert len(list(box.records())) == 1
        with pytest.raises(ProtocolError):
            box.write_new(record["admission_id"], "capture.bin", b"late writer")
        with pytest.raises(ProtocolError):
            box.reserve(artifact["admission"]["prepare"], {"retained_byte_limit": 1 << 40, "min_free_bytes": 1})
        assert source.read_bytes() == original
    finally:
        box.close()


def test_journal_refuses_changed_intent_and_stage_skip(tmp_path):
    root, _source, record, artifact = journal_fixture(tmp_path)
    box = Outbox(root, os.getgid())
    try:
        box.create(record)
        changed = copy.deepcopy(record)
        changed["prepare_wire"] += "\n"
        with pytest.raises(ProtocolError):
            box.update(changed)
        changed = copy.deepcopy(record)
        changed.update(stage="scanning", artifact=artifact)
        with pytest.raises(ProtocolError):
            box.update(changed)
        assert box.read(record["admission_id"]) == record
    finally:
        box.close()


def test_uncertain_next_generation_is_retained_without_legacy_recovery(tmp_path):
    root, source, record, _artifact = journal_fixture(tmp_path)
    box = Outbox(root, os.getgid())
    try:
        box.create(record)
        pending = root / record["admission_id"] / "journal.next"
        pending.write_bytes(b"partial uncertain generation")
        with pytest.raises(ProtocolError):
            box.read(record["admission_id"])
        assert pending.read_bytes() == b"partial uncertain generation" and source.exists()
    finally:
        box.close()


def test_outbox_single_writer_and_output_caps(tmp_path):
    root, _source, record, artifact = journal_fixture(tmp_path)
    box = Outbox(root, os.getgid())
    try:
        with pytest.raises(BlockingIOError):
            Outbox(root, os.getgid())
        box.create(record)
        prepared_copying(box, record, artifact)
        for name in ("capture.bin", "policy/policy.yaml", "evidence/pipeline.json"):
            with pytest.raises(ProtocolError):
                box.write_new(record["admission_id"], name, b"x" * (MAX_EVIDENCE_BYTES + 1))
        with pytest.raises(ProtocolError):
            box.write_new(record["admission_id"], "evidence/pipeline.json", b"x" * MAX_EVIDENCE_BYTES)
        assert not (root / record["admission_id"] / "evidence/pipeline.json").exists()
    finally:
        box.close()


def test_model_copy_uses_fresh_inode_and_keeps_original(tmp_path):
    root, source, record, artifact = journal_fixture(tmp_path)
    box = Outbox(root, os.getgid())
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        box.create(record)
        prepared_copying(box, record, artifact)
        box.copy_input(record["admission_id"], source_fd, lambda: None)
        copied_fd = box.open_owned(record["admission_id"], "input/" + source.name)
        try:
            assert os.read(copied_fd, 8192) == source.read_bytes()
            assert os.fstat(copied_fd).st_ino != source.stat().st_ino
            assert os.fstat(copied_fd).st_mode & 0o222 == 0
        finally:
            os.close(copied_fd)
        with pytest.raises(FileExistsError):
            box.copy_input(record["admission_id"], source_fd, lambda: None)
    finally:
        os.close(source_fd)
        box.close()


def test_interrupted_copy_retains_new_bytes_without_restart_or_source_change(tmp_path):
    root, source, record, artifact = journal_fixture(tmp_path)
    original = source.read_bytes()
    box = Outbox(root, os.getgid())
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    calls = 0

    def deadline():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic prepared deadline")

    try:
        box.create(record)
        prepared_copying(box, record, artifact)
        with pytest.raises(RuntimeError):
            box.copy_input(record["admission_id"], source_fd, deadline)
        assert (root / record["admission_id"] / "input" / source.name).exists()
        assert source.read_bytes() == original
        record["stage"] = "unknown"
        box.update(record)
        with pytest.raises(ProtocolError):
            box.copy_input(record["admission_id"], source_fd, lambda: None)
    finally:
        os.close(source_fd)
        box.close()
