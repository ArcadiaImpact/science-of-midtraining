"""Tail every charter-target pod's run log over ssh.

The launcher is blind while the pods work -- bellhop only surfaces output when
a cell finishes -- and the 4B scale-up lost ~8 idle pod-hours to exactly that
blindness. This reads the live log off each pod so "no news" can be checked
rather than assumed.

    python -m experiments.dispatch.charter_target_heldout.progress
"""

from __future__ import annotations

import json
import subprocess
import sys

from experiments.dispatch.charter_target_heldout import contracts

KEY = "/root/.ssh/id_ed25519"
SSH = ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no",
       "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15", "-o", "IdentitiesOnly=yes",
       "-o", "LogLevel=ERROR"]


def pods() -> dict[str, dict]:
    out = subprocess.run(["runpodctl", "pod", "list"],
                         capture_output=True, text=True).stdout
    found = {}
    for entry in json.loads(out):
        name = entry.get("name", "")
        if name.startswith("bellhop-scimt-ctgt-"):
            found[name.removeprefix("bellhop-")] = entry
    return found


def detail(pod_id: str) -> dict:
    out = subprocess.run(["runpodctl", "pod", "get", pod_id],
                         capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {}


def tail(cell: contracts.Cell, info: dict, n: int) -> str:
    ssh = detail(info["id"]).get("ssh") or {}
    if not ssh.get("ip"):
        return "  (no ssh endpoint yet)"
    log = f"/workspace/{cell.slug}/wave/results/run.log"
    cmd = SSH + [f"root@{ssh['ip']}", "-p", str(ssh["port"]),
                 f"tail -n {n} {log} 2>/dev/null "
                 f"|| echo '(log not created yet — still in setup)'"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    text = (r.stdout or r.stderr).strip()
    return "\n".join("  " + line for line in text.splitlines()[-n:]) or "  (empty)"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    live = pods()
    total = 0.0
    for cell in contracts.CELLS:
        info = live.get(cell.slug)
        if not info:
            print(f"\n=== {cell.label}: no pod (finished or never placed)")
            continue
        total += info.get("costPerHr", 0.0)
        up = info.get("uptimeSeconds", 0) / 60
        print(f"\n=== {cell.label}  {info['id']}  "
              f"${info.get('costPerHr', 0):.2f}/hr  up {up:.0f} min")
        try:
            print(tail(cell, info, n))
        except subprocess.TimeoutExpired:
            print("  (ssh timed out)")
    print(f"\n{len(live)} pods live, ${total:.2f}/hr")


if __name__ == "__main__":
    main()
