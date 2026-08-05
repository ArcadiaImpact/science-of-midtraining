#!/usr/bin/env python3
"""Install a no-new-privileges seccomp network denylist, then exec a command."""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import resource
import socket
import sys
from typing import NoReturn

PR_SET_NO_NEW_PRIVS = 38
SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_ERRNO = 0x00050000 | errno.EPERM
NETWORK_SYSCALLS = (
    "socket",
    "socketpair",
    "connect",
    "bind",
    "listen",
    "accept",
    "accept4",
    "sendto",
    "sendmsg",
    "sendmmsg",
    "recvfrom",
    "recvmsg",
    "recvmmsg",
    "getsockname",
    "getpeername",
    "shutdown",
)


class SandboxError(RuntimeError):
    pass


def _check(code: int, operation: str) -> None:
    if code != 0:
        error_number = -code if code < 0 else ctypes.get_errno()
        raise SandboxError(f"{operation} failed: errno {error_number}")


def install_network_filter() -> None:
    """Deny all socket lifecycle/data syscalls for this process and children."""

    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    _check(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "PR_SET_NO_NEW_PRIVS")

    library_name = ctypes.util.find_library("seccomp") or "libseccomp.so.2"
    libseccomp = ctypes.CDLL(library_name, use_errno=True)
    libseccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    libseccomp.seccomp_init.restype = ctypes.c_void_p
    libseccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    libseccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    libseccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    libseccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    libseccomp.seccomp_rule_add.restype = ctypes.c_int
    libseccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    libseccomp.seccomp_load.restype = ctypes.c_int

    context = libseccomp.seccomp_init(SCMP_ACT_ALLOW)
    if not context:
        raise SandboxError("seccomp_init returned a null context")
    try:
        resolved = 0
        for name in NETWORK_SYSCALLS:
            number = libseccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0:
                continue
            _check(
                libseccomp.seccomp_rule_add(context, SCMP_ACT_ERRNO, number, 0),
                f"seccomp rule for {name}",
            )
            resolved += 1
        if resolved < 2:
            raise SandboxError("could not resolve the required socket syscalls")
        _check(libseccomp.seccomp_load(context), "seccomp_load")
    finally:
        libseccomp.seccomp_release(context)


def _close_inherited_descriptors() -> None:
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit == resource.RLIM_INFINITY:
        soft_limit = 1_048_576
    os.closerange(3, min(int(soft_limit), 1_048_576))


def self_test() -> None:
    _close_inherited_descriptors()
    install_network_filter()
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            socket.socket(family, socket.SOCK_STREAM)
        except PermissionError:
            pass
        else:
            raise SandboxError(f"socket family {family} was not blocked")
    try:
        socket.socketpair()
    except PermissionError:
        pass
    else:
        raise SandboxError("socketpair was not blocked")
    print("seccomp network sandbox: PASS")


def exec_sandboxed(command: list[str]) -> NoReturn:
    if not command:
        raise SandboxError("a command is required")
    _close_inherited_descriptors()
    install_network_filter()
    os.execvpe(command[0], command, os.environ)


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    exec_sandboxed(command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SandboxError) as exc:
        print(f"seccomp network sandbox failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
