"""Fixture-only v2 orchestration. No parser, real transport or runtime qualification."""

import copy
import os
from dataclasses import replace

import pytest
from test_registry_protocol import capabilities_fixture, fixture_values

from quarantine import admission_controller as c
from quarantine.admission_client import RegistryUnavailable
from quarantine.admission_journal import Outbox
from quarantine.admission_process import ProcessResult
from quarantine.admission_scan import encode_frame
from quarantine.policy_context import PolicyContext
from quarantine.registry_protocol import (
    OUTBOX_OVERHEAD_BYTES,
    POLICY_COMPONENTS,
    ProtocolError,
    binding_digest,
    context_digest,
    encode_request,
    receipt_digest,
    request_digest,
    strict_json,
)

BOOT = "12345678-1234-1234-1234-123456789abc"


class Client:
    def __init__(self):
        self.calls = []
        self.prepared = None
        self.committed = None
        self.unknown = False
        self.lost_lookup = False

    def preflight(self, bundle, pipeline, *, artifact_format):
        self.calls.append(("preflight", artifact_format))
        value = capabilities_fixture()
        value["policy_bundle"] = bundle
        value["context_sha256"] = context_digest(
            bundle["sha256"], value["scan_context"]
        )
        return value

    def prepare(self, raw, **_kwargs):
        self.calls.append(("prepare", raw))
        if self.prepared is None:
            request = strict_json(raw)
            receipt = fixture_values()[2]
            receipt.update(
                admission_id=request["admission_id"],
                binding_sha256=binding_digest(request),
                request_sha256=request_digest("prepare", raw),
                reserved_registry_bytes=request["size_bytes"]
                + request["evidence_budget_bytes"],
                reserved_outbox_bytes=request["size_bytes"]
                + request["evidence_budget_bytes"]
                + OUTBOX_OVERHEAD_BYTES,
            )
            receipt["sha256"] = receipt_digest(receipt)
            row = {
                key: request[key]
                for key in (
                    "name",
                    "filename",
                    "format",
                    "sha256",
                    "size_bytes",
                    "source",
                    "source_revision",
                )
            }
            row.update(
                state="quarantined",
                admission={
                    "version": 2,
                    "context": fixture_values()[1],
                    "prepare": request,
                    "prepare_receipt": receipt,
                },
            )
            self.prepared = row
        return 201, copy.deepcopy(self.prepared)

    def commit(self, prepare_raw, commit_raw):
        self.calls.append(("commit", prepare_raw, commit_raw))
        row = copy.deepcopy(self.prepared)
        request = strict_json(commit_raw)
        receipt = copy.deepcopy(row["admission"]["prepare_receipt"])
        receipt.update(
            phase="committed",
            event_sequence=3,
            request_sha256=request_digest("commit", commit_raw),
        )
        receipt["sha256"] = receipt_digest(receipt)
        row.update(state="trusted")
        row["admission"].update(commit=request, commit_receipt=receipt, report={})
        self.committed = row
        if self.unknown:
            raise RegistryUnavailable()
        return 201, copy.deepcopy(row)

    def lookup(self, raw, *, commit_raw=None):
        self.calls.append(("lookup", raw, commit_raw))
        if self.lost_lookup:
            raise RegistryUnavailable()
        return copy.deepcopy(self.committed)


def worker(_box, prepare, context, _deadline, check):
    check()
    report = {
        key: prepare[key]
        for key in (
            "admission_id",
            "filename",
            "sha256",
            "size_bytes",
            "source",
            "source_revision",
            "policy_bundle",
        )
    }
    report.update(
        version=2, pipeline_sha256=context["pipeline_sha256"], gates={}, scanners={}
    )
    for key in (
        "source_policy",
        "format_gate",
        "hash_pin",
        "provenance",
        "polyglot_check",
        "entropy",
        "weight_stats",
        "smoke_test",
    ):
        report["gates"][key] = dict(
            verdict="not_applicable" if key == "smoke_test" else "pass",
            reason="fixture",
            count=0 if key == "smoke_test" else 1,
            assurance="none" if key == "provenance" else "",
        )
    for key, name in {
        "modelscan": "modelscan",
        "fickling": "fickling",
        "modelaudit": "modelaudit",
        "gguf_guard": "gguf-guard",
    }.items():
        na = key != "modelaudit"
        report["scanners"][key] = dict(
            schema_version=2,
            passed=not na,
            verdict="not_applicable" if na else "pass",
            available=None if na else True,
            scanner=name,
            scanner_version=None if na else "fixture-1",
            reason="fixture",
            scanned_count=0 if na else 1,
        )
    frame = encode_frame(
        [("pipeline-report", "pipeline.json", encode_request(report))], passed=True
    )
    return ProcessResult(frame, b"", frame, 0, ())


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / "demo.safetensors"
    source.write_bytes(b"inert non-model fixture")
    outbox = tmp_path / "outbox"
    outbox.mkdir(mode=0o750)
    policy = PolicyContext.from_components(
        {name: b"{}\n" for name in POLICY_COMPONENTS}
    )
    client = Client()
    clock = {"now": 1, "boot": BOOT, "cancel": False}
    ports = c._Ports(
        lambda: None,
        lambda: policy,
        client,
        outbox,
        os.getgid(),
        "f" * 64,
        worker,
        lambda: clock["now"],
        lambda: clock["boot"],
        lambda: clock["cancel"],
    )
    return source, ports, client, clock


def test_public_gate_precedes_all_inputs_and_has_no_override(tmp_path, monkeypatch):
    monkeypatch.setenv("QUARANTINE_ADMISSION_MODE", "helper-v2-required")
    monkeypatch.setenv("P2_QUALIFIED", "1")
    for call in (
        lambda: c.process_artifact(tmp_path / "absent"),
        lambda: c.recover("1" * 64),
    ):
        with pytest.raises(c.RuntimeUnavailable):
            call()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(TypeError):
        c.process_artifact(tmp_path / "absent", ports=object())


def test_private_gate_failure_precedes_policy_client_and_outbox(fixture):
    source, ports, client, _clock = fixture

    def refuse():
        raise c.RuntimeUnavailable("no actual probes")

    def forbidden():
        raise AssertionError("policy parsed before gate")

    with pytest.raises(c.RuntimeUnavailable):
        c._Controller(replace(ports, qualify=refuse, policy=forbidden)).submit(source)
    assert client.calls == [] and list(ports.outbox.iterdir()) == []


def test_exact_commit_unknown_readback_and_original_retention(fixture):
    source, ports, client, _clock = fixture
    client.unknown = True
    original = source.read_bytes()
    result = c._Controller(ports).submit(source)
    assert result["state"] == "trusted" and source.read_bytes() == original
    box = Outbox(ports.outbox, ports.gid)
    try:
        record = list(box.records())[0]
    finally:
        box.close()
    commit = next(call for call in client.calls if call[0] == "commit")
    lookup = next(call for call in client.calls if call[0] == "lookup")
    assert record["stage"] == "committed" and commit[1:] == lookup[1:] == (
        record["prepare_wire"].encode(),
        record["commit_wire"].encode(),
    )
    assert (
        record["artifact"]["admission"]["commit_receipt"]["expires_boot_ns"]
        == record["artifact"]["admission"]["prepare_receipt"]["expires_boot_ns"]
    )


def test_uncertain_commit_replays_exact_journal_not_scan(fixture):
    source, ports, client, _clock = fixture
    client.unknown = True
    client.lost_lookup = True
    with pytest.raises(RegistryUnavailable):
        c._Controller(ports).submit(source)
    box = Outbox(ports.outbox, ports.gid)
    try:
        record = list(box.records())[0]
    finally:
        box.close()
    assert record["stage"] == "commit-ready"
    client.unknown = False
    client.lost_lookup = False

    def forbidden(*_args):
        raise AssertionError("recovery restarted scan")

    result = c._Controller(replace(ports, worker=forbidden)).recover(
        record["admission_id"]
    )
    assert result["state"] == "trusted"
    commits = [call for call in client.calls if call[0] == "commit"]
    assert commits[0] == commits[1]


@pytest.mark.parametrize("change", ["expired", "boot", "cancel"])
def test_recovery_refuses_new_authority_without_commit(fixture, change):
    source, ports, client, clock = fixture
    client.unknown = True
    client.lost_lookup = True
    with pytest.raises(RegistryUnavailable):
        c._Controller(ports).submit(source)
    admission_id = next(ports.outbox.iterdir()).name
    if change == "expired":
        clock["now"] = 1800000000000
    elif change == "boot":
        clock["boot"] = "87654321-1234-1234-1234-123456789abc"
    else:
        clock["cancel"] = True
    before = len([x for x in client.calls if x[0] == "commit"])
    with pytest.raises(ProtocolError):
        c._Controller(ports).recover(admission_id)
    assert (
        len([x for x in client.calls if x[0] == "commit"]) == before and source.exists()
    )


@pytest.mark.parametrize("bad", ["frame", "binding", "failure"])
def test_bad_worker_retains_capture_without_commit(fixture, bad):
    source, ports, client, _clock = fixture

    def invalid(*args):
        result = worker(*args)
        if bad == "frame":
            raw = result.stdout[:-1]
        elif bad == "failure":
            raw = encode_frame([], passed=False, error="fixture failure")
        else:
            header, files = c.decode_frame(result.stdout)
            report = strict_json(files[0][1])
            report["sha256"] = "0" * 64
            raw = encode_frame(
                [("pipeline-report", "pipeline.json", encode_request(report))],
                passed=True,
            )
        return ProcessResult(raw, b"", raw, 0, ())

    with pytest.raises(ProtocolError):
        c._Controller(replace(ports, worker=invalid)).submit(source)
    assert not any(x[0] == "commit" for x in client.calls)
    retained = next(ports.outbox.iterdir())
    assert (retained / "capture.bin").exists() and source.exists()
    assert (
        strict_json((retained / "journal.json").read_bytes(), limit=1048576)["stage"]
        == "failed"
    )


def test_old_or_pending_journal_is_not_adopted(fixture):
    source, ports, client, _clock = fixture
    old = ports.outbox / "legacy-v1.json"
    old.write_bytes(b"legacy bytes")
    with pytest.raises(ProtocolError):
        c._Controller(ports).submit(source)
    assert old.read_bytes() == b"legacy bytes" and not any(
        x[0] == "prepare" for x in client.calls
    )


def test_watcher_v2_never_enters_legacy_paths(tmp_path, monkeypatch):
    from quarantine import watcher

    monkeypatch.setenv("QUARANTINE_ADMISSION_MODE", "helper-v2-required")
    monkeypatch.setattr(watcher, "QUARANTINE_DIR", tmp_path / "missing")
    for call in (
        watcher.main,
        watcher.scan_directory,
        watcher.recover_pending_promotions,
        lambda: watcher.process_artifact(tmp_path / "missing"),
        lambda: watcher.process_directory(tmp_path / "missing"),
        lambda: watcher.promote_to_registry("x", "a" * 64, 1, {}),
    ):
        with pytest.raises(c.RuntimeUnavailable):
            call()
    assert not (tmp_path / "missing").exists()
    monkeypatch.setenv("QUARANTINE_ADMISSION_MODE", "typo")
    with pytest.raises(ValueError):
        watcher.scan_directory()


def test_exact_prepare_intent_replay_retains_original_lease_and_does_not_scan(fixture):
    source, ports, client, _clock = fixture
    original_prepare = client.prepare

    def lost_reply(raw, **kwargs):
        original_prepare(raw, **kwargs)
        raise RegistryUnavailable()

    client.prepare = lost_reply
    with pytest.raises(RegistryUnavailable):
        c._Controller(ports).submit(source)
    retained = next(ports.outbox.iterdir())
    intent = strict_json((retained / "journal.json").read_bytes(), limit=1048576)
    assert intent["stage"] == "intent"
    client.prepare = original_prepare

    def no_scan(*_args):
        raise AssertionError("intent reconciliation restarted scan")

    result = c._Controller(replace(ports, worker=no_scan)).recover(retained.name)
    prepares = [call[1] for call in client.calls if call[0] == "prepare"]
    assert prepares == [intent["prepare_wire"].encode()] * 2
    assert (
        result["admission"]["prepare_receipt"]
        == client.prepared["admission"]["prepare_receipt"]
    )
    assert not (retained / "input").exists() and source.exists()


def test_pending_journal_generation_is_retained_without_recovery_mutation(fixture):
    source, ports, client, _clock = fixture
    client.unknown = True
    client.lost_lookup = True
    with pytest.raises(RegistryUnavailable):
        c._Controller(ports).submit(source)
    retained = next(ports.outbox.iterdir())
    pending = retained / "journal.next"
    pending.write_bytes(b"uncertain durable generation")
    before = len([call for call in client.calls if call[0] == "commit"])
    with pytest.raises(ProtocolError):
        c._Controller(ports).recover(retained.name)
    assert pending.read_bytes() == b"uncertain durable generation"
    assert len([call for call in client.calls if call[0] == "commit"]) == before
