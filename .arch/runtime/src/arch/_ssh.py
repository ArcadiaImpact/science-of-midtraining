"""Shared SSH mechanics for talking to RunPod pods.

Every worker/heldout startup script starts sshd and seeds authorized_keys
from a PUBLIC_KEY env var set at spawn time (see templates/*_startup.sh.j2).
This module owns the orchestrator side: a dedicated keypair scoped to arch2
(never the researcher's personal SSH identity), resolving a pod's SSH target
from its RunPod REST metadata, and running a remote command over SSH by
shelling out to the system `ssh` binary — the same pattern the rest of this
package already uses for `gh`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DEFAULT_KEY_PATH = Path.home() / ".ssh" / "arch2_worker_ed25519"


def ensure_arch_ssh_key(key_path: Path | None = None) -> Path:
    """Return the arch2 dedicated SSH private key path, generating it (via
    `ssh-keygen`) if it doesn't already exist. Reused across tasks and runs
    on this machine — never the researcher's personal key.
    """
    path = key_path or Path(os.environ.get("ARCH_SSH_KEY_PATH", str(DEFAULT_KEY_PATH)))
    pub_path = path.with_suffix(".pub")
    # Both halves must exist before we call this done: callers read the .pub
    # side (`arch ssh-key --pub`, the PUBLIC_KEY pod env), so a private key
    # whose .pub went missing must not be reported as complete — that surfaces
    # later as a bare FileNotFoundError with no obvious cause.
    if path.is_file() and pub_path.is_file():
        return path
    if path.is_file():
        # Private key survived but the .pub is gone: re-derive the public half
        # from it rather than regenerating the pair. A fresh private key would
        # invalidate every pod already seeded with the old PUBLIC_KEY.
        proc = subprocess.run(
            ["ssh-keygen", "-y", "-f", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ssh-keygen could not re-derive the public key for {path}: "
                f"{proc.stderr.strip()}"
            )
        pub_path.write_text(proc.stdout)
        return path
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    proc = subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh-keygen failed for {path}: {proc.stderr.strip()}")
    return path


def ssh_target(pod_meta: dict[str, Any]) -> tuple[str, int] | None:
    """Extract (host, port) for SSH from a `GET /pods/{id}` response.

    RunPod's REST v1 `Pod` schema exposes `publicIp` and `portMappings`
    (container port -> public port, keyed by string port number). Returns
    None if either is missing/null — the pod's networking isn't up yet.
    """
    host = pod_meta.get("publicIp")
    mappings = pod_meta.get("portMappings") or {}
    port = mappings.get("22")
    if not host or not port:
        return None
    return host, int(port)


@dataclass
class SshResult:
    """Outcome of one `ssh_tail` attempt.

    `pending` and `auth_failed` are ssh's *own* failures (it never reached a
    remote shell); `remote_error` means the SSH connection worked but the
    remote `tail` failed — usually the log file doesn't exist yet. Keeping
    those apart matters for the operator: the first says "check network /
    firewall / key", the second says "SSH is fine, the startup script isn't
    writing that log (yet)".
    """

    outcome: Literal["ok", "pending", "auth_failed", "remote_error"]
    stdout: str = ""
    detail: str = ""


def ssh_tail(
    host: str,
    port: int,
    key_path: Path,
    *,
    remote_path: str,
    lines: int = 200,
    timeout: int = 10,
) -> SshResult:
    """Tail `remote_path` on the pod over SSH. Never raises — failure modes
    are returned, not thrown, so callers can classify them (boot-in-progress
    vs. a real config problem)."""
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                f"ConnectTimeout={timeout}",
                "-i",
                str(key_path),
                "-p",
                str(port),
                f"root@{host}",
                # Single argv element forwarded to a remote shell — quote the
                # path. Today's callers pass internal literals, but a future
                # variable path must not be able to inject shell syntax.
                f"tail -n {int(lines)} {shlex.quote(remote_path)}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SshResult("pending", detail=f"ssh timed out after {timeout + 15}s")
    except OSError as exc:
        return SshResult("pending", detail=f"could not invoke ssh: {exc}")

    if proc.returncode == 0:
        return SshResult("ok", stdout=proc.stdout)
    stderr = proc.stderr or ""
    if proc.returncode == 255:
        # 255 is ssh's own exit code for a connection-level failure (refused,
        # timed out, no route, key rejected) — it never got to a remote shell.
        if "Permission denied" in stderr:
            return SshResult("auth_failed", detail=stderr.strip())
        return SshResult("pending", detail=stderr.strip() or "ssh exited 255")
    # Any other non-zero code is the REMOTE command's exit status, forwarded by
    # ssh — so the connection itself succeeded and `tail` is what failed (most
    # often: the log file doesn't exist yet because the container's
    # `exec > >(tee …)` redirect hasn't happened). Distinct from `pending` so
    # callers don't tell the operator to go debug their firewall.
    return SshResult(
        "remote_error",
        detail=stderr.strip() or f"remote command exited {proc.returncode}",
    )
