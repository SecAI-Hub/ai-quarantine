"""Cumulative subprocess capture for the credentialless admission scan worker.

This is a worker-local runner, not a host sandbox. The admission controller must
place the complete Python parser and every child in the qualified bwrap tree.
"""
from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time

from .registry_protocol import MAX_EVIDENCE_BYTES, ProtocolError

# Fixed private namespace mount, checked by the worker before artifact parsing.
SANDBOX_TMP = "/tmp"  # nosec B108


class ScanAborted(RuntimeError):
    """The entire scan must stop; partial stage results are not evidence."""


def boot_ns() -> int:
    if not hasattr(time, "CLOCK_BOOTTIME"):
        raise ProtocolError("suspend-inclusive clock unavailable")
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def scanner_environment() -> dict[str, str]:
    """Construct before Popen; no inherited credential or loader environment."""
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "HOME": SANDBOX_TMP, "TMPDIR": SANDBOX_TMP, "XDG_CACHE_HOME": SANDBOX_TMP, "XDG_CONFIG_HOME": SANDBOX_TMP,
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "PROMPTFOO_DISABLE_TELEMETRY": "1", "NO_ANALYTICS": "1", "DO_NOT_TRACK": "1",
    }


class BoundedRunner:
    """subprocess.run subset with one budget across all calls in this scan.

    Even discarded child streams are drained through the cumulative counter.
    A limit violation latches failure so caught adapter errors cannot restart
    work. The enclosing namespace owns grandchildren and is destroyed on abort.
    """

    def __init__(self, deadline_ns: int, *, max_capture: int = MAX_EVIDENCE_BYTES):
        if type(deadline_ns) is not int or deadline_ns <= boot_ns():
            raise ProtocolError("scan deadline unavailable/expired")
        if type(max_capture) is not int or not 1 <= max_capture <= MAX_EVIDENCE_BYTES:
            raise ProtocolError("scan capture budget invalid")
        self.deadline_ns = deadline_ns
        self.max_capture = max_capture
        self.captured = 0
        self.raw_capture = bytearray()
        self.failed = False

    def check(self) -> None:
        if self.failed or boot_ns() >= self.deadline_ns:
            self.failed = True
            raise ScanAborted("scan deadline/capture already exhausted")

    def __call__(self, args, *, timeout, capture_output=False, text=False,
                 stdin=None, stdout=None, stderr=None, env=None, check=False):
        self.check()
        if (not isinstance(args, (list, tuple)) or not args or
                any(not isinstance(arg, str) or "\0" in arg for arg in args) or
                not isinstance(timeout, (int, float)) or not 0 < timeout <= 1800 or
                stdin not in {None, subprocess.DEVNULL} or
                stdout not in {None, subprocess.PIPE, subprocess.DEVNULL} or
                stderr not in {None, subprocess.PIPE, subprocess.DEVNULL}):
            self.failed = True
            raise ScanAborted("unsupported bounded scanner invocation")
        clean_env = scanner_environment()
        if env is not None and (not isinstance(env, dict) or any(
                name not in clean_env or clean_env[name] != value for name, value in env.items())):
            self.failed = True
            raise ScanAborted("scanner attempted to expand its clean environment")
        local_deadline = min(self.deadline_ns, boot_ns() + int(timeout * 1_000_000_000))
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=clean_env, close_fds=True)
        pidfd = None
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        try:
            pidfd = os.pidfd_open(proc.pid)
            with selectors.DefaultSelector() as selector:
                for name in buffers:
                    stream = getattr(proc, name)
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, name)
                while selector.get_map():
                    self.check()
                    remaining = (local_deadline - boot_ns()) / 1_000_000_000
                    if remaining <= 0:
                        self.failed = True
                        raise subprocess.TimeoutExpired(args, timeout)
                    for key, _events in selector.select(min(remaining, 0.1)):
                        # Read at most one byte beyond the remaining cap; never
                        # accumulate an oversized capture before rejecting it.
                        chunk = os.read(key.fileobj.fileno(), min(16384, self.max_capture - self.captured + 1))
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        available = self.max_capture - self.captured
                        buffers[key.data].extend(chunk[:available])
                        self.raw_capture.extend(chunk[:available])
                        self.captured += min(len(chunk), available)
                        if len(chunk) > available:
                            self.failed = True
                            raise ScanAborted("cumulative scanner capture exceeds one MiB")
            self.check()
            remaining = (local_deadline - boot_ns()) / 1_000_000_000
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, timeout)
            returncode = proc.wait(timeout=remaining)
            values = {}
            for name, raw in buffers.items():
                requested = capture_output or (stdout if name == "stdout" else stderr) == subprocess.PIPE
                values[name] = (bytes(raw).decode("utf-8", errors="strict") if text else bytes(raw)) if requested else None
            result = subprocess.CompletedProcess(args, returncode, values["stdout"], values["stderr"])
            if check:
                result.check_returncode()
            return result
        except BaseException:
            self.failed = True
            if proc.poll() is None:
                if pidfd is not None:
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                else:
                    # An unreaped direct child cannot have its PID reused.
                    proc.kill()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired as error:
                    raise ScanAborted("owned scanner could not be reaped; namespace must terminate") from error
            raise
        finally:
            if pidfd is not None:
                os.close(pidfd)
            proc.stdout.close()
            proc.stderr.close()
