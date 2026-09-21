"""Kill every trial driver AND every vLLM engine, then wait for VRAM.

By path, never by an inline pattern: `pkill -f run_server_trial.sh` matches the
ssh command line that issues it. Two concurrent drivers is exactly what
happened at 19:01 -- one on dp=1, one on dp=2, interleaved in one log, both
health-checking the other's server on port 8000.
"""

import os
import signal
import subprocess
import sys
import time

NEEDLES = (b"run_server_trial", b"vllm", b"EngineCore", b"VLLM", b"throughput.probe")
me = {os.getpid(), os.getppid()}


def victims() -> dict[int, bytes]:
    found = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmdline = handle.read()
        except OSError:
            continue
        if any(needle in cmdline for needle in NEEDLES):
            found[pid] = cmdline.replace(b"\x00", b" ")[:90]
    return found


def used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return max(int(v) for v in out if v.isdigit())


for pid, cmd in victims().items():
    try:
        os.kill(pid, signal.SIGKILL)
        print("killed", pid, cmd.decode(errors="replace"))
    except OSError as error:
        print("could not kill", pid, error)

for _ in range(60):
    if used_mib() < 2000:
        print("clean: max VRAM used", used_mib(), "MiB")
        sys.exit(0)
    time.sleep(5)
print("TIMEOUT: max VRAM used", used_mib(), "MiB")
sys.exit(1)
