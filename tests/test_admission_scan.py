"""Wire/boundary unit checks; actual bwrap/container probes are separate."""
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from quarantine.admission_sandbox import sandbox_command
from quarantine.admission_scan import (
    FRAME_MAGIC,
    decode_frame,
    encode_frame,
    worker_mount_guard,
)
from quarantine.registry_protocol import (
    MAX_EVIDENCE_BYTES,
    POLICY_COMPONENTS,
    ProtocolError,
    encode_request,
)


def test_scan_frame_binds_exact_bytes_and_roles():
    raw = encode_frame([("pipeline-report", "pipeline.json", b'{"complete":true}')], passed=True)
    header, files = decode_frame(raw)
    assert header["passed"] is True and files[0][1] == b'{"complete":true}'
    for altered in (raw[:-1], raw + b"x", raw[:-1] + b"x", b"noise" + raw):
        with pytest.raises(ProtocolError):
            decode_frame(altered)


def test_failed_capture_has_no_passing_evidence():
    raw = encode_frame([("failed-capture", "failed-capture.bin", b"bounded scanner error")], passed=False, error="ScanAborted")
    header, files = decode_frame(raw)
    assert not header["passed"] and files[0][0]["role"] == "failed-capture"
    raw = encode_frame([("pipeline-report", "pipeline.json", b"{}")], passed=False, error="ScanAborted")
    with pytest.raises(ProtocolError):
        decode_frame(raw)


def test_frame_caps_include_headers_and_reject_unknown_fields():
    with pytest.raises(ProtocolError):
        encode_frame([("pipeline-report", "pipeline.json", b"x" * MAX_EVIDENCE_BYTES)], passed=True)
    header = encode_request({"version": 2, "passed": True, "error": "", "files": [], "extra": 1})
    with pytest.raises(ProtocolError):
        decode_frame(FRAME_MAGIC + struct.pack(">I", len(header)) + header)


def test_sandbox_command_has_only_bounded_writable_trees_and_strict_namespaces():
    command = sandbox_command(20, "demo.gguf", {name: 21 + index for index, name in enumerate(POLICY_COMPONENTS)}, 24)
    assert command.count("--tmpfs") == command.count("--size") == 2
    assert command.count(str(MAX_EVIDENCE_BYTES)) == 2
    assert "--clearenv" in command and "--disable-userns" in command and "--assert-userns-disabled" in command
    assert "--unshare-user" in command and "--unshare-pid" in command and "--die-with-parent" in command
    assert "--seccomp" in command and "--cap-drop" in command
    for forbidden in ("--unshare-all", "--unshare-user-try", "--bind", "--dev", "--not-a-security-boundary"):
        assert forbidden not in command
    for index, value in enumerate(command[:-2]):
        if value in {"--ro-bind", "--dev-bind"}:
            assert command[index + 1] not in {"/", "/etc", "/home", "/var", "/run", "/outbox", "/registry"}
    root_remount = command.index("--remount-ro", command.index("--remount-ro") + 1)
    assert command[root_remount + 1] == "/"


@pytest.mark.parametrize("name", ["../demo.gguf", "/home/file.gguf", "secret\0.gguf", ".hidden.gguf"])
def test_sandbox_command_rejects_path_selection(name):
    with pytest.raises(ProtocolError):
        sandbox_command(20, name, {component: 21 + index for index, component in enumerate(POLICY_COMPONENTS)}, 24)


@pytest.mark.parametrize("change", ["valid", "writable_root", "excess_tmpfs", "shared_mount", "cgroup_visible", "capabilities"])
def test_worker_rejects_unqualified_mount_and_process_boundaries(monkeypatch, change):
    from quarantine import admission_scan

    status = "CapEff: " + ("1" if change == "capabilities" else "0") + "\nNoNewPrivs: 1\nSeccomp: 2\n"
    mountinfo = ("1 0 0:1 / / ro " + ("shared:17 " if change == "shared_mount" else "") + "- tmpfs tmpfs ro\n"
                 "2 1 0:2 / /work rw - tmpfs tmpfs rw\n3 1 0:3 / /tmp rw - tmpfs tmpfs rw\n")
    monkeypatch.setattr(Path, "read_text", lambda path, **_kwargs: status if str(path).endswith("status") else mountinfo)
    monkeypatch.setattr(Path, "exists", lambda _path: change == "cgroup_visible")
    monkeypatch.setattr(Path, "resolve", lambda _path, **_kwargs: Path("/tmp"))
    monkeypatch.setattr(admission_scan.os, "statvfs", lambda path: SimpleNamespace(
        f_flag=0 if change == "writable_root" or str(path) != "/" else admission_scan.os.ST_RDONLY,
        f_blocks=257 if change == "excess_tmpfs" else 256, f_frsize=4096))
    if change == "valid":
        worker_mount_guard()
    else:
        with pytest.raises(ProtocolError):
            worker_mount_guard()
