from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = ROOT / ".arch" / "seccomp_exec.py"


def test_seccomp_self_test_blocks_ipv4_ipv6_and_socketpair() -> None:
    completed = subprocess.run(
        [sys.executable, str(SANDBOX), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "seccomp network sandbox: PASS"


def test_seccomp_exec_runs_file_only_child_without_network() -> None:
    completed = subprocess.run(
        [sys.executable, str(SANDBOX), sys.executable, "-c", "print('child-ok')"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "child-ok"
