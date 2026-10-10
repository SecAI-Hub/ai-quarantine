"""Bounded ownership of one scanner namespace; no host-scanning entry point.

The caller must verify the image/container and construct the strict bwrap argv.
These process primitives alone do not establish parser isolation.
"""
from __future__ import annotations

import os
import select
import selectors
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .registry_protocol import MAX_BODY_BYTES, MAX_EVIDENCE_BYTES, ProtocolError
from .scan_execution import boot_ns, scanner_environment

MAX_OBSERVED_PROCESSES = 256


class SandboxFailure(ProtocolError):
    def __init__(self, reason: str, capture: bytes, *, cleanup_complete: bool):
        super().__init__(reason)
        self.capture = capture
        self.cleanup_complete = cleanup_complete


@dataclass(frozen=True, slots=True)
class ProcessResult:
    stdout: bytes
    stderr: bytes
    capture: bytes
    returncode: int
    observed: tuple[tuple[int, int], ...]


def _process_identity(pid: int) -> tuple[int, int]:
    # comm can contain spaces and parentheses; the final delimiter precedes
    # state/ppid. /proc supplies the remaining fixed numeric stat fields.
    raw = Path(f"/proc/{pid}/stat").read_bytes()
    if len(raw) > 8192:
        raise ProtocolError("owned process identity exceeds bound")
    fields = raw.rsplit(b") ", 1)[1].split()
    return int(fields[1]), int(fields[19])  # parent PID and starttime


def _exited(fd: int) -> bool:
    return bool(select.select([fd], [], [], 0)[0])


class _OwnedTree:
    def __init__(self, process):
        self.process = process
        self.nodes = {}
        fd = os.pidfd_open(process.pid)
        try:
            _parent, start = _process_identity(process.pid)
            self.nodes[process.pid] = (start, fd)
        except BaseException:
            os.close(fd)
            raise

    def observe(self):
        pending = list(self.nodes)
        while pending:
            parent = pending.pop()
            parent_start, parent_fd = self.nodes[parent]
            if _exited(parent_fd):
                continue
            try:
                _ppid, current_start = _process_identity(parent)
                if current_start != parent_start:
                    raise ProtocolError("owned parent identity changed")
                children = set()
                for index, task in enumerate(Path(f"/proc/{parent}/task").iterdir()):
                    if index >= 64 or not task.name.isdecimal():
                        raise ProtocolError("owned task inventory exceeds bound")
                    try:
                        raw = (task / "children").read_bytes()
                    except FileNotFoundError:
                        continue
                    if len(raw) > 4096:
                        raise ProtocolError("owned child inventory exceeds bound")
                    children.update(int(value) for value in raw.split())
            except FileNotFoundError:
                continue
            for child in children:
                if child in self.nodes:
                    continue
                if len(self.nodes) >= MAX_OBSERVED_PROCESSES:
                    raise ProtocolError("cumulative scanner process inventory exceeds bound")
                child_fd = None
                try:
                    before = _process_identity(child)
                    if before[0] != parent or _exited(parent_fd):
                        continue
                    child_fd = os.pidfd_open(child)
                    if _process_identity(child) != before or _exited(parent_fd):
                        continue
                    self.nodes[child] = (before[1], child_fd)
                    child_fd = None
                    pending.append(child)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                finally:
                    if child_fd is not None:
                        os.close(child_fd)

    def terminate(self):
        """Signal only pidfd-bound descendants; never signal a shared group."""
        failed = False
        for _pid, (_start, descriptor) in reversed(tuple(self.nodes.items())):
            if not _exited(descriptor):
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    failed = True
        return not failed

    def finish(self) -> bool:
        deadline = boot_ns() + 1_000_000_000
        while boot_ns() < deadline:
            self.process.poll()  # Reap our direct child before checking absence.
            present = False
            for pid, (start, descriptor) in self.nodes.items():
                if not _exited(descriptor):
                    present = True
                    continue
                try:
                    if _process_identity(pid)[1] == start:
                        present = True
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if not present:
                return True
            select.select([], [], [], 0.01)
        return False

    def close(self):
        for _start, descriptor in self.nodes.values():
            os.close(descriptor)


def capture_owned(command: list[str], config: bytes, descriptors: tuple[int, ...],
                  deadline_ns: int, check_cancelled) -> ProcessResult:
    """Internal transport: the caller supplies verified bwrap argv and lease.

    The private PID namespace/parent-death chain contains descendants that exit
    between observations. Native bwrap qualification must establish that chain.
    """
    if (type(config) is not bytes or not 0 < len(config) <= MAX_BODY_BYTES or
            type(deadline_ns) is not int or not boot_ns() < deadline_ns <= boot_ns() + 1800 * 1_000_000_000 or
            not callable(check_cancelled)):
        raise ProtocolError("scanner transport configuration/deadline invalid")
    check_cancelled()
    capture = bytearray()
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    process, tree, failure, cleanup_complete = None, None, None, False
    try:
        # Secrets are stripped before exec, including during bwrap setup and in
        # the environment inherited by the sandbox's PID 1.
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=scanner_environment(),
                                   pass_fds=descriptors, close_fds=True, start_new_session=True)
        tree = _OwnedTree(process)
        remaining_input = memoryview(config)
        with selectors.DefaultSelector() as selector:
            for name in ("stdin", "stdout", "stderr"):
                stream = getattr(process, name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ, name)
            while selector.get_map():
                check_cancelled()
                remaining = (deadline_ns - boot_ns()) / 1_000_000_000
                if remaining <= 0:
                    raise ProtocolError("whole scanner lease expired")
                tree.observe()
                for key, _events in selector.select(min(0.05, remaining)):
                    if key.data == "stdin":
                        count = os.write(key.fileobj.fileno(), remaining_input[:4096])
                        if count <= 0:
                            raise ProtocolError("scanner configuration publication failed")
                        remaining_input = remaining_input[count:]
                        if not remaining_input:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    available = MAX_EVIDENCE_BYTES - len(capture)
                    block = os.read(key.fileobj.fileno(), min(16384, available + 1))
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    capture.extend(block[:available])
                    streams[key.data].extend(block[:available])
                    if len(block) > available:
                        raise ProtocolError("whole scanner capture exceeds one MiB")
            while not _exited(tree.nodes[process.pid][1]):
                check_cancelled()
                if boot_ns() >= deadline_ns:
                    raise ProtocolError("whole scanner lease expired after stream closure")
                tree.observe()
                select.select([tree.nodes[process.pid][1]], [], [], 0.05)
        check_cancelled()
        if boot_ns() >= deadline_ns or process.wait(timeout=0.1) != 0:
            raise ProtocolError("scanner did not finish successfully within its lease")
    except BaseException as error:
        failure = error
    finally:
        if tree is not None:
            try:
                signalled = tree.terminate()
                cleanup_complete = tree.finish() and signalled
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                failure = failure or error
                cleanup_complete = False
        elif process is not None:
            # Popen's direct child has not been reaped, so numeric fallback is
            # safe only here, before pidfd ownership could be established.
            try:
                process.kill()
                process.wait(timeout=1)
                cleanup_complete = True
            except (OSError, subprocess.TimeoutExpired):
                cleanup_complete = False
        else:
            cleanup_complete = True
        observed = tuple((pid, start) for pid, (start, _fd) in tree.nodes.items()) if tree else ()
        if tree is not None:
            tree.close()
        if process is not None:
            for name in ("stdin", "stdout", "stderr"):
                getattr(process, name).close()
    if failure is not None or not cleanup_complete:
        reason = "scanner cleanup unverified" if not cleanup_complete else "scanner stopped: " + type(failure).__name__
        raise SandboxFailure(reason, bytes(capture), cleanup_complete=cleanup_complete) from failure
    return ProcessResult(bytes(streams["stdout"]), bytes(streams["stderr"]), bytes(capture), process.returncode, observed)
