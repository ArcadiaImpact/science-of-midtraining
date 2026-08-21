"""Rsync each live pod's results dir to the devbox, mid-run or at the end.

bellhop pulls `wave/results` home when a cell *finishes*, which is the right
default but means a 3.5 h sweep is invisible until it is over. This pulls the
same tree on demand, so partial results can be scored and plotted while the
remaining seeds run -- and so a pod that later dies has already handed over
everything it had finished.

rsync, not scp: a re-pull transfers only the endpoints that appeared since the
last one, which is what makes this cheap enough to run repeatedly.

    python -m experiments.prior_coins.seed_sweep_v1.pull_results
    python -m experiments.prior_coins.seed_sweep_v1.pull_results --arm charter
"""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.prior_coins.seed_sweep_v1 import contracts

HERE = Path(__file__).resolve().parent
PODS = HERE.parent / "runs" / "seed_sweep_v1" / "pods"
SSH_OPTS = ("-i", "/root/.ssh/id_ed25519", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "IdentitiesOnly=yes",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15",
            "-o", "BatchMode=yes")


def run(cmd: list[str], timeout: float) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except OSError as error:
        return 1, str(error)


def fleet() -> dict[str, str]:
    """arm -> pod id, for pods that are still up."""
    code, out = run(["runpodctl", "pod", "list"], 40)
    if code:
        return {}
    try:
        raw = json.loads(out)
    except json.JSONDecodeError:
        return {}
    pods = raw if isinstance(raw, list) else raw.get("pods", [])
    found = {}
    for cell in contracts.CELLS:
        for entry in pods:
            if entry.get("name") == f"bellhop-{cell.slug}":
                found[cell.arm] = entry["id"]
    return found


def endpoint(pod_id: str) -> tuple[str, int] | None:
    code, out = run(["runpodctl", "pod", "get", pod_id], 40)
    if code:
        return None
    try:
        ssh = (json.loads(out).get("ssh") or {})
    except json.JSONDecodeError:
        return None
    if ssh.get("ip") and ssh.get("port"):
        return ssh["ip"], int(ssh["port"])
    return None


def pull(cell: contracts.Cell, ip: str, port: int, timeout: float) -> dict:
    dest = PODS / cell.arm / "results"
    dest.mkdir(parents=True, exist_ok=True)
    remote = f"root@{ip}:/workspace/{cell.slug}/wave/results/"
    ssh = "ssh " + " ".join(SSH_OPTS) + f" -p {port}"
    code, out = run(["rsync", "-a", "--partial", "--stats", "-e", ssh,
                     remote, str(dest) + "/"], timeout)
    got = sorted(p.name for p in dest.glob("*-step*") if p.is_dir())
    seeds = sorted(p.name for p in (dest / "adapters").glob("*")
                   if (p / "SEED_DONE.json").is_file()) if (dest / "adapters").is_dir() else []
    return {"arm": cell.arm, "rc": code, "endpoints": len(got),
            "seeds_done": len(seeds),
            "error": None if code == 0 else out.strip()[-300:]}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", choices=contracts.ARMS)
    ap.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args()
    cells = [c for c in contracts.CELLS if not args.arm or c.arm in args.arm]
    live = fleet()

    jobs = []
    for cell in cells:
        pod_id = live.get(cell.arm)
        if not pod_id:
            print(f"{cell.arm:14s} no live pod (finished and torn down?)")
            continue
        ep = endpoint(pod_id)
        if not ep:
            print(f"{cell.arm:14s} no ssh endpoint")
            continue
        jobs.append((cell, ep[0], ep[1]))

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(
            lambda j: pull(j[0], j[1], j[2], args.timeout), jobs))
    for r in sorted(results, key=lambda x: x["arm"]):
        status = "ok" if r["rc"] == 0 else f"FAILED rc={r['rc']}"
        print(f"{r['arm']:14s} {status:14s} {r['endpoints']} endpoint dirs, "
              f"{r['seeds_done']} seeds stashed"
              + (f"\n    {r['error']}" if r["error"] else ""))
    print(f"\n-> {PODS}")


if __name__ == "__main__":
    main()
