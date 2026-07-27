"""Sandboxed test execution for the code reward stream.

One completion, one test spec (the row's ``ground_truth``), binary verdict:
all tests pass or not. Test spec formats accepted (JSON string or dict):

  {"inputs": [...], "outputs": [...]}      stdin/stdout pairs (verl/TACO style)
  {"assert_snippets": [...]} or [...]      python assert statements run after
                                           the candidate code (functional style)
  {"functional": "..."}                    single assert/check snippet

Sandboxing (code-r1 finding: firejail + strict rlimits kills the
false-positive/concurrency problems): each test runs as a subprocess via
``firejail --quiet --net=none`` when available, else bare python with
RLIMIT_AS/RLIMIT_CPU set in preexec_fn (warn once — no network isolation).
Per-test wall timeout + memory cap; any nonzero exit, timeout, or output
mismatch fails the whole spec. Executable used is ``sys.executable``.

Pure stdlib; CPU-only. The reward loop parallelizes across completions with
threads (subprocess-bound), one core per ~4 concurrent executions.
"""

from __future__ import annotations

import json
import resource
import shutil
import subprocess
import sys
import warnings
from typing import Any

_FIREJAIL = shutil.which("firejail")
_WARNED = False


def _limits(memory_mb: int, cpu_s: int):
    def set_limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb << 20, memory_mb << 20))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))

    return set_limits


def _run_python(
    code: str, stdin: str, timeout: float, memory_mb: int
) -> tuple[int, str]:
    global _WARNED
    argv = [sys.executable, "-c", code]
    if _FIREJAIL:
        argv = [_FIREJAIL, "--quiet", "--net=none", "--private-tmp", *argv]
    elif not _WARNED:
        warnings.warn("firejail not found: code exec has rlimits but NO network isolation")
        _WARNED = True
    try:
        proc = subprocess.run(
            argv,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_limits(memory_mb, int(timeout) + 1),
        )
    except (subprocess.TimeoutExpired, OSError):
        return 1, ""
    return proc.returncode, proc.stdout


def parse_spec(ground_truth: Any) -> dict:
    spec = ground_truth
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError:
            spec = {"assert_snippets": [ground_truth]}
    if isinstance(spec, list):
        spec = {"assert_snippets": spec}
    if isinstance(spec, dict) and "functional" in spec:
        spec = {"assert_snippets": [spec["functional"]]}
    if not isinstance(spec, dict):
        raise ValueError(f"unparseable code test spec: {type(ground_truth)}")
    return spec


def run_tests(
    code: str,
    ground_truth: Any,
    *,
    timeout: float = 6.0,
    memory_mb: int = 512,
    max_io_tests: int = 8,
) -> bool:
    """True iff the candidate passes every test in the spec (capped at
    ``max_io_tests`` stdin/stdout cases — reward latency control; the cap is
    deterministic, first-N)."""
    spec = parse_spec(ground_truth)

    inputs, outputs = spec.get("inputs"), spec.get("outputs")
    if inputs and outputs:
        for stdin, expected in list(zip(inputs, outputs))[:max_io_tests]:
            if isinstance(stdin, list):
                stdin = "\n".join(map(str, stdin))
            rc, out = _run_python(code, str(stdin), timeout, memory_mb)
            if rc != 0 or _normalize_io(out) != _normalize_io(str(expected)):
                return False
        return True

    snippets = spec.get("assert_snippets")
    if snippets:
        harness = code + "\n\n" + "\n".join(str(s) for s in snippets)
        rc, _ = _run_python(harness, "", timeout, memory_mb)
        return rc == 0

    raise ValueError(f"code test spec has neither io pairs nor asserts: {list(spec)}")


def _normalize_io(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines())


def extract_code(answer_text: str) -> str | None:
    """Last fenced code block (```python or bare ```)."""
    fence = "```"
    chunks = answer_text.split(fence)
    if len(chunks) < 3:
        return None
    block = chunks[-2]
    first_line, _, rest = block.partition("\n")
    if first_line.strip().lower() in {"python", "py", ""}:
        return rest.strip() or None
    return block.strip() or None
