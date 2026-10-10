"""Real bounded pipe/direct-process controls, not native sandbox qualification."""
import os
import sys

import pytest

from quarantine.admission_process import SandboxFailure, capture_owned
from quarantine.registry_protocol import (
    MAX_BODY_BYTES,
    MAX_EVIDENCE_BYTES,
    ProtocolError,
)
from quarantine.scan_execution import boot_ns


def not_cancelled():
    return None


def invoke(code, *, config=b"{}", duration=5_000_000_000, check=not_cancelled):
    return capture_owned([sys.executable, "-I", "-c", code], config, (), boot_ns() + duration, check)


def test_whole_process_receives_bounded_config_without_inherited_secrets(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN", "synthetic-canary")
    monkeypatch.setenv("HTTPS_PROXY", "https://synthetic.invalid")
    code = ("import os,sys; data=sys.stdin.buffer.read(); "
            "assert 'SERVICE_TOKEN' not in os.environ and 'HTTPS_PROXY' not in os.environ; "
            "assert os.getsid(0)==os.getpid(); os.write(1,str(len(data)).encode()); os.write(2,b'ok')")
    result = invoke(code, config=b"x" * MAX_BODY_BYTES)
    assert result.stdout == str(MAX_BODY_BYTES).encode()
    assert result.stderr == b"ok" and result.returncode == 0
    assert len(result.capture) == len(result.stdout) + len(result.stderr)
    assert len(result.observed) == 1
    assert not os.path.exists(f"/proc/{result.observed[0][0]}")


@pytest.mark.parametrize("code", [
    "import os,sys; sys.stdin.buffer.read(); os.write(1,b'a'*700000); os.write(2,b'b'*700000)",
    "import os,sys; sys.stdin.buffer.read(); os.write(1,b'a'*1048577)",
])
def test_outer_capture_is_cumulative_and_retains_only_the_one_mib_budget(code):
    with pytest.raises(SandboxFailure) as error:
        invoke(code)
    assert error.value.cleanup_complete
    assert len(error.value.capture) == MAX_EVIDENCE_BYTES


@pytest.mark.parametrize("close_streams", [False, True])
def test_deadline_stops_the_owned_process_even_after_streams_close(tmp_path, close_streams):
    pid_file = tmp_path / "owned.pid"
    code = ("import os,sys,time,pathlib; sys.stdin.buffer.read(); pathlib.Path(" + repr(str(pid_file)) +
            ").write_text(str(os.getpid())); " + ("os.close(1); os.close(2); " if close_streams else "") + "time.sleep(10)")
    with pytest.raises(SandboxFailure) as error:
        invoke(code, duration=350_000_000)
    assert error.value.cleanup_complete
    assert not os.path.exists(f"/proc/{int(pid_file.read_text())}")


def test_cancellation_before_popen_has_no_process_effect(monkeypatch):
    from quarantine import admission_process

    monkeypatch.setattr(admission_process.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("process started"))

    def cancelled():
        raise ProtocolError("cancelled")

    with pytest.raises(ProtocolError, match="cancelled"):
        invoke("raise AssertionError", check=cancelled)


def test_cancellation_during_capture_reaps_and_retains_partial_output(tmp_path):
    pid_file = tmp_path / "owned.pid"
    code = ("import os,sys,time,pathlib; sys.stdin.buffer.read(); os.write(1,b'partial'); pathlib.Path(" + repr(str(pid_file)) +
            ").write_text(str(os.getpid())); time.sleep(10)")

    def cancelled():
        if pid_file.exists():
            raise ProtocolError("cancelled")

    with pytest.raises(SandboxFailure) as error:
        invoke(code, check=cancelled)
    assert error.value.cleanup_complete
    assert error.value.capture in (b"", b"partial")  # Pipe readiness may race cancellation.
    assert not os.path.exists(f"/proc/{int(pid_file.read_text())}")


def test_nonzero_worker_exit_cannot_pass_with_plausible_output():
    with pytest.raises(SandboxFailure) as error:
        invoke("import os,sys; sys.stdin.buffer.read(); os.write(1,b'plausible'); sys.exit(3)")
    assert error.value.capture == b"plausible" and error.value.cleanup_complete


def test_invalid_configuration_refused_before_popen(monkeypatch):
    from quarantine import admission_process

    monkeypatch.setattr(admission_process.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("process started"))
    with pytest.raises(ProtocolError):
        invoke("raise AssertionError", config=b"x" * (MAX_BODY_BYTES + 1))
