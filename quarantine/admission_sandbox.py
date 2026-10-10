"""Linux admission sandbox construction. No privileged or host-policy fallback.

The image supplies a root-owned runtime manifest. Actual preflight/negative
qualification and producer integration are required before this is enabled.
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import os
import stat
from pathlib import Path

from .registry_protocol import (
    HEX,
    MAX_EVIDENCE_BYTES,
    NAME,
    POLICY_COMPONENTS,
    ProtocolError,
    strict_json,
)
from .scan_execution import SANDBOX_TMP, scanner_environment

BWRAP_PATH = "/usr/local/bin/bwrap"
# The sandbox creates this alias to its bounded private tmpfs, never host memory.
SANDBOX_SHM = "/dev/shm"  # nosec B108
RUNTIME_ROOTS = (
    "/usr/local/lib/python3.12", "/usr/lib/x86_64-linux-gnu",
    "/opt/ai-quarantine/quarantine",
)
RUNTIME_FILES = (
    "/usr/local/bin/python3.12", "/usr/local/bin/modelaudit", "/usr/local/bin/gguf-guard",
    "/usr/local/lib/libpython3.12.so.1.0", "/etc/ld.so.cache",
)
RUNTIME_MANIFEST = "/opt/ai-quarantine/admission-runtime.json"


def _read_control(path: str, limit: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != 0 or before.st_nlink != 1 or
                before.st_mode & 0o022 or not 0 < before.st_size <= limit):
            raise ProtocolError("root-owned runtime control invalid")
        with os.fdopen(os.dup(fd), "rb") as handle:
            raw = handle.read(limit + 1)
        after, named = os.fstat(fd), os.stat(path, follow_symlinks=False)
        def fields(s):
            return (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_gid, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if len(raw) != before.st_size or fields(before) != fields(after) or fields(after) != fields(named):
            raise ProtocolError("runtime control identity changed")
        return raw
    finally:
        os.close(fd)


def verify_container_bounds() -> dict[str, int]:
    """Inspect actual limits applying to the controller and its descendants."""
    if not os.statvfs("/").f_flag & os.ST_RDONLY:
        raise ProtocolError("admission controller requires a read-only container root")
    if not os.statvfs("/sys/fs/cgroup").f_flag & os.ST_RDONLY:
        raise ProtocolError("writable cgroup controls are forbidden")
    membership = Path("/proc/self/cgroup").read_text()
    # The initial qualified deployment uses a private cgroup namespace rooted
    # at its bounded container. Other topologies require separate qualification.
    if membership != "0::/\n":
        raise ProtocolError("qualified private cgroup-v2 namespace unavailable")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    if int(status["CapEff"].strip(), 16) != 0 or status["NoNewPrivs"].strip() != "1":
        raise ProtocolError("controller capability/no-new-privileges boundary unavailable")
    limits = {}
    for field in ("memory.max", "memory.swap.max", "pids.max"):
        raw = Path("/sys/fs/cgroup", field).read_text().strip()
        if not raw.isascii() or not raw.isdecimal() or len(raw) > 20:
            raise ProtocolError("controller cgroup limit is not finite")
        limits[field] = int(raw)
    if (not 0 < limits["memory.max"] <= 4 * 1024**3 or
            not 0 <= limits["memory.swap.max"] <= limits["memory.max"] or
            not 0 < limits["pids.max"] <= 64):
        raise ProtocolError("controller cgroup exceeds approved resource envelope")
    return limits


def verify_runtime() -> dict:
    """Bind the exact immutable files of the minimal scanner runtime."""
    if not os.statvfs("/").f_flag & os.ST_RDONLY:
        raise ProtocolError("runtime filesystem is writable")
    manifest = strict_json(_read_control(RUNTIME_MANIFEST, 4 * MAX_EVIDENCE_BYTES), limit=4 * MAX_EVIDENCE_BYTES)
    if (not isinstance(manifest, dict) or set(manifest) != {"version", "bwrap_version", "files"} or
            type(manifest["version"]) is not int or manifest["version"] != 1 or manifest["bwrap_version"] != "0.12.0" or
            not isinstance(manifest["files"], dict) or not 1 <= len(manifest["files"]) <= 32768):
        raise ProtocolError("runtime manifest shape/version invalid")
    expected = set(RUNTIME_FILES) | {BWRAP_PATH}
    for root in RUNTIME_ROOTS:
        root_info = os.stat(root, follow_symlinks=False)
        if (not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != 0 or root_info.st_mode & 0o022 or
                not os.statvfs(root).f_flag & os.ST_RDONLY):
            raise ProtocolError("runtime bind root is not an immutable owned directory")
        for directory, dirs, files in os.walk(root, followlinks=False):
            if len(expected) > 32768:
                raise ProtocolError("runtime file inventory exceeds bound")
            for name in dirs + files:
                path = os.path.join(directory, name)
                info = os.stat(path, follow_symlinks=False)
                if (info.st_uid != 0 or (not stat.S_ISLNK(info.st_mode) and info.st_mode & 0o022) or
                        not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode))):
                    raise ProtocolError("runtime includes a mutable or special file")
                if not stat.S_ISDIR(info.st_mode):
                    expected.add(path)
    if set(manifest["files"]) != expected:
        raise ProtocolError("runtime file inventory differs from image manifest")
    total = 0
    for path, digest in manifest["files"].items():
        if not isinstance(digest, str) or HEX.fullmatch(digest) is None:
            raise ProtocolError("runtime digest invalid")
        info = os.stat(path, follow_symlinks=False)
        if not os.statvfs(path).f_flag & os.ST_RDONLY:
            raise ProtocolError("runtime component is on a writable mount")
        if stat.S_ISLNK(info.st_mode):
            target = os.readlink(path)
            resolved = os.path.realpath(path)
            if not any(resolved == root or resolved.startswith(root + "/") for root in RUNTIME_ROOTS) and resolved not in RUNTIME_FILES:
                raise ProtocolError("runtime symlink escapes its minimal bindings")
            actual = hashlib.sha256(b"symlink\0" + os.fsencode(target)).hexdigest()
        else:
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022 or
                    (path == BWRAP_PATH and (info.st_mode & 0o6000 or not info.st_mode & 0o111))):
                raise ProtocolError("runtime executable/data identity invalid")
            total += info.st_size
            if total > 2 * 1024**3:
                raise ProtocolError("runtime exceeds qualified size")
            hashed = hashlib.sha256()
            with open(path, "rb") as handle:
                while chunk := handle.read(65536):
                    hashed.update(chunk)
            actual = hashed.hexdigest()
        if actual != digest:
            raise ProtocolError("runtime file hash differs from approved image manifest")
    return manifest


class _ArgCmp(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int), ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


def child_seccomp_fd() -> int:
    """Export a sealed child-only filter; setup remains bwrap's responsibility."""
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes, library.seccomp_init.restype = [ctypes.c_uint32], ctypes.c_void_p
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_syscall_resolve_name.argtypes, library.seccomp_syscall_resolve_name.restype = [ctypes.c_char_p], ctypes.c_int
    library.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(_ArgCmp)]
    library.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    context = library.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW; native architecture only.
    if not context:
        raise ProtocolError("child seccomp context unavailable")
    descriptor = -1
    try:
        blocked = ("mount", "umount2", "unshare", "setns", "fsopen", "fsconfig", "fsmount", "move_mount", "open_tree", "mount_setattr", "pivot_root", "chroot", "clone3")
        for name in blocked:
            number = library.seccomp_syscall_resolve_name(name.encode("ascii"))
            action = 0x00050000 | (errno.ENOSYS if name == "clone3" else errno.EPERM)
            if number < 0 or library.seccomp_rule_add_array(context, action, number, 0, None) != 0:
                raise ProtocolError("required child syscall restriction unavailable")
        number = library.seccomp_syscall_resolve_name(b"clone")
        if number < 0:
            raise ProtocolError("ordinary clone restriction unavailable")
        for flag in (0x00020000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000):
            comparison = _ArgCmp(0, 7, flag, flag)  # SCMP_CMP_MASKED_EQ; ordinary threads remain allowed.
            if library.seccomp_rule_add_array(context, 0x00050000 | errno.EPERM, number, 1, ctypes.byref(comparison)) != 0:
                raise ProtocolError("namespace clone restriction unavailable")
        descriptor = os.memfd_create("admission-child-seccomp", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        if library.seccomp_export_bpf(context, descriptor) != 0 or not 0 < os.fstat(descriptor).st_size <= 65536:
            raise ProtocolError("child seccomp export invalid")
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE)
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        library.seccomp_release(context)


def sandbox_command(input_fd: int, filename: str, policy_fds: dict[str, int], seccomp_fd: int) -> list[str]:
    """Only immutable model/policy descriptors enter the new tree."""
    if not isinstance(filename, str) or NAME.fullmatch(filename) is None or set(policy_fds) != set(POLICY_COMPONENTS):
        raise ProtocolError("sandbox input inventory invalid")
    if any(type(fd) is not int or fd < 3 for fd in [input_fd, seccomp_fd, *policy_fds.values()]):
        raise ProtocolError("sandbox requires explicitly owned descriptors")
    command = [BWRAP_PATH, "--unshare-user", "--unshare-ipc", "--unshare-pid", "--unshare-net", "--unshare-uts", "--unshare-cgroup",
               "--disable-userns", "--assert-userns-disabled", "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv"]
    for name, value in scanner_environment().items():
        command += ["--setenv", name, value]
    for root in RUNTIME_ROOTS:
        command += ["--ro-bind", root, root]
    for path in RUNTIME_FILES:
        command += ["--ro-bind", path, path]
    command += ["--symlink", "python3.12", "/usr/local/bin/python", "--symlink", "python3.12", "/usr/local/bin/python3",
                "--symlink", "usr/lib", "/lib", "--dir", "/lib64",
                "--symlink", "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2", "/lib64/ld-linux-x86-64.so.2",
                "--dir", "/run", "--dir", "/dev", "--proc", "/proc", "--remount-ro", "/proc"]
    for device in ("null", "zero", "urandom", "random"):
        command += ["--dev-bind", "/dev/" + device, "/dev/" + device]
    command += ["--size", str(MAX_EVIDENCE_BYTES), "--perms", "0700", "--tmpfs", "/work",
                "--size", str(MAX_EVIDENCE_BYTES), "--perms", "0700", "--tmpfs", SANDBOX_TMP, "--symlink", SANDBOX_TMP, SANDBOX_SHM,
                "--ro-bind", f"/proc/self/fd/{input_fd}", "/work/" + filename]
    for name in POLICY_COMPONENTS:
        command += ["--ro-bind", f"/proc/self/fd/{policy_fds[name]}", "/policy/" + name]
    command += ["--remount-ro", "/", "--chdir", "/work", "--seccomp", str(seccomp_fd),
                "/usr/local/bin/python3.12", "-I", "-c",
                "import sys; sys.path.insert(0,'/opt/ai-quarantine'); from quarantine.admission_scan import worker_main; worker_main()"]
    return command
