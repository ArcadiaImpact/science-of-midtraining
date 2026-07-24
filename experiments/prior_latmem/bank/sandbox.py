"""Small stdlib-only subprocess sandbox used by the problem-bank gates.

The child receives no environment and runs from a fresh temporary directory.
The socket prelude is deliberately best-effort: it blocks the normal Python
socket entry points and the private ``_socket`` constructor, while the cleared
environment and isolated interpreter provide additional containment. This is
an execution-quality gate, not a security boundary for hostile native code.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

try:  # pragma: no cover - present on the Unix validation hosts
    import resource
except ImportError:  # pragma: no cover - makes importing on Windows harmless
    resource = None  # type: ignore[assignment]


_CHILD_RUNNER = r'''
import contextlib
import io
import json
import pathlib
import socket
import sys
import time
import traceback
import tracemalloc
try:
    import resource
except ImportError:
    resource = None


class NetworkDisabledError(RuntimeError):
    pass


def _deny_socket(*args, **kwargs):
    raise NetworkDisabledError("network access is disabled in the bank sandbox")


# Inject the socket block before importing or executing the payload.  Patching
# the private module is best-effort because its attributes vary by Python build.
socket.socket = _deny_socket
try:
    import _socket
    _socket.socket = _deny_socket
except Exception:
    pass


payload_path = pathlib.Path(sys.argv[1])
payload = payload_path.read_text(encoding="utf-8")
namespace = {"__name__": "__main__", "__file__": str(payload_path)}
started = time.perf_counter()
report = {"ok": False, "error": None, "timings": {}, "peaks": {}}
captured = io.StringIO()
try:
    # Payload print calls are captured so the final stdout line is the protocol.
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        exec(compile(payload, str(payload_path), "exec"), namespace, namespace)
    report["ok"] = True
except BaseException as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
    report["traceback"] = traceback.format_exc(limit=8)
finally:
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
    else:
        current, peak = 0, 0
    report["timings"] = {"payload_wall_s": time.perf_counter() - started}
    try:
        # Linux reports KiB; macOS reports bytes. This is diagnostic only—the
        # validator's ratio gate intentionally uses tracemalloc peaks.
        if resource is None:
            raise RuntimeError("resource module unavailable")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = int(rss * (1 if sys.platform == "darwin" else 1024))
    except Exception:
        rss_bytes = None
    report["peaks"] = {"tracemalloc_bytes": peak, "rss_bytes": rss_bytes}
    if "__sandbox_result" in namespace:
        report["result"] = namespace["__sandbox_result"]
    # Keep diagnostics bounded; a candidate's captured output is not part of
    # the data contract but is useful when a handwritten test fails.
    if captured.getvalue():
        report["captured_output"] = captured.getvalue()[-2000:]
    print(json.dumps(report, sort_keys=True, default=repr), flush=True)
'''


def _preexec_limits(timeout_s: float, mem_limit_mb: int | None):
    """Return a POSIX pre-exec function applying the child resource limits."""
    if resource is None:  # pragma: no cover - non-POSIX fallback
        return None
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if mem_limit_mb is not None and mem_limit_mb <= 0:
        raise ValueError("mem_limit_mb must be positive or None")
    cpu_seconds = max(1, math.ceil(timeout_s))

    def set_limits() -> None:
        # CPU is always limited. The one-second hard-limit cushion lets the OS
        # deliver SIGXCPU before SIGKILL while the parent still owns wall time.
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        # RLIMIT_AS is unreliable on macOS (and can reject an otherwise valid
        # interpreter), so address-space limiting is gated to Linux. Validation
        # runs on Linux CPU pods where this limit is reliable enough to use.
        if sys.platform.startswith("linux") and mem_limit_mb is not None:
            limit = int(mem_limit_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return set_limits


def _kill_process(proc: subprocess.Popen[str]) -> None:
    """Kill the isolated process group, falling back to the direct child."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, OSError):
        proc.kill()


def _protocol_from_stdout(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "ok" in value:
            return value
    return None


def run_sandboxed(
    source: str, *, timeout_s: float, mem_limit_mb: int | None
) -> dict:
    """Execute Python ``source`` and return the child's one-line JSON report.

    The function is pure from the caller's point of view: all filesystem state
    is temporary and the returned dictionary is self-contained. A timeout or a
    child that exits before its protocol line becomes ``ok=False`` with a
    stable error string rather than an exception.
    """
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="scimt-latmem-") as temp_dir:
        root = Path(temp_dir)
        payload_path = root / "payload.py"
        runner_path = root / "runner.py"
        payload_path.write_text(source, encoding="utf-8")
        runner_path.write_text(textwrap.dedent(_CHILD_RUNNER), encoding="utf-8")
        command = [sys.executable, "-I", str(runner_path), str(payload_path)]
        try:
            proc = subprocess.Popen(
                command,
                cwd=temp_dir,
                env={},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                preexec_fn=_preexec_limits(timeout_s, mem_limit_mb),
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill_process(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=2.0)
                except subprocess.TimeoutExpired as exc:
                    # The process group was killed, but a descendant can still
                    # hold a pipe open. Preserve whatever output is available
                    # and return the stable timeout report regardless.
                    stdout = exc.stdout or ""
                    stderr = exc.stderr or ""
                return {
                    "ok": False,
                    "error": "timeout",
                    "timings": {"parent_wall_s": time.perf_counter() - started},
                    "peaks": {},
                    "returncode": proc.returncode,
                    "stdout": stdout[-2000:],
                    "stderr": stderr[-2000:],
                }
        except Exception as exc:  # Popen/pre-exec failures are validation drops
            return {
                "ok": False,
                "error": f"sandbox_start: {type(exc).__name__}: {exc}",
                "timings": {"parent_wall_s": time.perf_counter() - started},
                "peaks": {},
            }

    report = _protocol_from_stdout(stdout)
    if report is None:
        return {
            "ok": False,
            "error": "missing_protocol",
            "timings": {"parent_wall_s": time.perf_counter() - started},
            "peaks": {},
            "returncode": proc.returncode,
            "stdout": stdout[-2000:],
            "stderr": stderr[-2000:],
        }
    report.setdefault("timings", {})["parent_wall_s"] = (
        time.perf_counter() - started
    )
    report["returncode"] = proc.returncode
    if stderr:
        report["stderr"] = stderr[-2000:]
    return report
