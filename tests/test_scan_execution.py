"""The child runner bounds cumulative output; bwrap qualification is separate."""
import os
import subprocess
import sys

import pytest

from quarantine.scan_execution import (
    BoundedRunner,
    ScanAborted,
    boot_ns,
    scanner_environment,
)


def test_scanner_environment_is_constructed_without_inherited_values(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN", "synthetic-token-canary")
    monkeypatch.setenv("LD_PRELOAD", "/synthetic/no-library.so")
    monkeypatch.setenv("HTTPS_PROXY", "https://synthetic.invalid")
    clean = scanner_environment()
    assert not {"SERVICE_TOKEN", "LD_PRELOAD", "HTTPS_PROXY"} & set(clean)
    assert clean["OPENBLAS_NUM_THREADS"] == clean["OMP_NUM_THREADS"] == "1"


def test_runner_aggregate_capture_and_latched_failure():
    runner = BoundedRunner(boot_ns() + 5_000_000_000, max_capture=10)
    first = runner([sys.executable, "-I", "-c", "import os; os.write(1,b'12345')"], capture_output=True, text=True, timeout=2)
    assert first.stdout == "12345"
    with pytest.raises(ScanAborted):
        runner([sys.executable, "-I", "-c", "import os; os.write(2,b'abcdef')"], stderr=subprocess.DEVNULL, timeout=2)
    assert runner.captured == len(runner.raw_capture) == 10
    assert runner.raw_capture == b"12345abcde"
    with pytest.raises(ScanAborted):
        runner(["/will-not-start"], timeout=2)


def test_runner_refuses_environment_expansion_before_popen(monkeypatch):
    runner = BoundedRunner(boot_ns() + 5_000_000_000)
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("unsafe process started"))
    with pytest.raises(ScanAborted):
        runner([sys.executable], timeout=2, env={**scanner_environment(), "SERVICE_TOKEN": "canary"})


def test_runner_timeout_reaps_exact_direct_child(tmp_path):
    path = tmp_path / "child.pid"
    runner = BoundedRunner(boot_ns() + 5_000_000_000)
    code = "import os,time,pathlib; pathlib.Path(" + repr(str(path)) + ").write_text(str(os.getpid())); time.sleep(10)"
    with pytest.raises(subprocess.TimeoutExpired):
        runner([sys.executable, "-I", "-c", code], timeout=0.5)
    pid = int(path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
