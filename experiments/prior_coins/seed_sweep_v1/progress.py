"""One-screen live progress for every seed-sweep pod, built for `watch`.

bellhop is blind while a pod works -- it only surfaces output when the cell
finishes -- and the 4B scale-up lost ~8 idle pod-hours to exactly that
blindness. So this reads each pod's live logs over ssh and reports *within*-stage
progress: the optimizer step and s/it while training, the slice and prompt count
while evaluating.

It also prints how long ago the active log last changed, because the recorded
lesson from that postmortem is that **silence reads exactly like progress**. A
pod whose log has not moved in >5 min is marked STALE, in caps, on purpose.

Designed to be cheap enough to re-run every 20-30 s: one `runpodctl pod list`
for the fleet, ssh endpoints cached per pod id (they are fixed for a pod's
lifetime), and the five pod probes fanned out in parallel with short timeouts.

    watch -n 30 'cd /workspace/scimt-seed-sweep && uv run python -m \
        experiments.prior_coins.seed_sweep_v1.progress'

    # or, without watch:
    uv run python -m experiments.prior_coins.seed_sweep_v1.progress
"""

from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from experiments.prior_coins.seed_sweep_v1 import contracts

CACHE = Path("/tmp/scimt-seedsweep-ssh.json")
SSH = ["ssh", "-i", "/root/.ssh/id_ed25519",
       "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
       "-o", "IdentitiesOnly=yes", "-o", "LogLevel=ERROR",
       "-o", "ConnectTimeout=8", "-o", "BatchMode=yes"]
#: minutes without the active log changing before we shout about it
STALE_MIN = 5.0
#: nominal minutes per seed (train + eval), for the finish estimate
PER_SEED_MIN = 37.0

#: one shell probe per pod; prints key=value lines. Kept to a single ssh round
#: trip because this runs every 30 s against five pods.
PROBE = r"""
W=/workspace/{slug}/wave
R=$W/results/run.log
T=$W/training/train.log
now=$(date -u +%s)
echo "now=$now"
echo "seeds=$(ls -d $W/results/adapters/*/SEED_DONE.json 2>/dev/null | wc -l)"
[ -f "$R" ] && echo "started=1" || echo "started=0"
[ -f "$R" ] && echo "rmtime=$(stat -c %Y $R)"
# during prepare there is no train/eval log, so size the parent download itself
# --apparent-size, not blocks: the in-flight shard is preallocated, so plain
# `du` reports a constant while the file actually fills.
echo "parentmb=$(du -sm --apparent-size $W/parent $W/_parent_staging 2>/dev/null | awk '{{t+=$1}} END {{print t+0}}')"
# freshness during prepare must come from the BYTES landing, not run.log: run.log
# is only appended at phase boundaries, so using it flagged a healthy 20 MB/s
# download as STALE. A false stall warning is as harmful as a missed one.
echo "smtime=$(find $W/_parent_staging $W/parent -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1)"
echo "phase=$(grep -oE '^=== PHASE [A-Za-z0-9_-]+' $R 2>/dev/null | tail -1 | awk '{{print $3}}')"
echo "lastok=$(grep -oE '^=== PHASE [A-Za-z0-9_-]+ (OK|FAILED)' $R 2>/dev/null | tail -1 | awk '{{print $3, $4}}')"
echo "fail=$(ls $W/results/PHASE_FAILED_*.json 2>/dev/null | wc -l)"
if [ -f "$T" ]; then
  echo "tmtime=$(stat -c %Y $T)"
  echo "train=$(tail -c 6000 $T 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+s/it' | tail -1)"
fi
EL=$(ls -t $W/logs/eval-*.log 2>/dev/null | head -1)
if [ -n "$EL" ]; then
  echo "evallog=$(basename $EL)"
  echo "emtime=$(stat -c %Y $EL)"
  echo "eslice=$(grep -oE '\[gen\] [^:]+' $EL 2>/dev/null | tail -1 | sed 's|.*/||')"
  echo "eprog=$(tail -c 6000 $EL 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[' | tail -1 | tr -d ' [')"
  echo "edone=$(grep -c '^\[ok\]' $EL 2>/dev/null)"
  echo "eprobe=$(grep -oE '\[probe\] .*differ from' $EL 2>/dev/null | tail -1)"
fi
echo "gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
"""


def run(cmd: list[str], timeout: float) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except (subprocess.TimeoutExpired, OSError):
        return ""


def fleet() -> dict[str, dict]:
    """arm -> pod record, from one `pod list`."""
    try:
        raw = json.loads(run(["runpodctl", "pod", "list"], 40) or "[]")
    except json.JSONDecodeError:
        return {}
    pods = raw if isinstance(raw, list) else raw.get("pods", [])
    out = {}
    for cell in contracts.CELLS:
        want = f"bellhop-{cell.slug}"
        for entry in pods:
            if entry.get("name") == want:
                out[cell.arm] = entry
    return out


def endpoints(ids: list[str]) -> dict[str, tuple[str, int]]:
    """pod id -> (ip, port), cached: an endpoint is fixed for a pod's lifetime."""
    try:
        cache = json.loads(CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        cache = {}
    todo = [i for i in ids if i not in cache]
    if todo:
        def one(pod_id: str):
            try:
                d = json.loads(run(["runpodctl", "pod", "get", pod_id], 40) or "{}")
            except json.JSONDecodeError:
                return pod_id, None
            ssh = d.get("ssh") or {}
            if ssh.get("ip") and ssh.get("port"):
                return pod_id, [ssh["ip"], int(ssh["port"])]
            return pod_id, None
        with ThreadPoolExecutor(max_workers=8) as pool:
            for pod_id, got in pool.map(one, todo):
                if got:
                    cache[pod_id] = got
        try:
            CACHE.write_text(json.dumps(cache))
        except OSError:
            pass
    return {k: (v[0], v[1]) for k, v in cache.items() if k in ids}


def probe(arm: str, slug: str, ip: str, port: int) -> dict:
    out = run(SSH + [f"root@{ip}", "-p", str(port),
                     PROBE.format(slug=slug)], 45)
    got: dict[str, str] = {"arm": arm}
    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            got[key.strip()] = value.strip()
    return got


def stage_of(p: dict) -> tuple[str, str]:
    """(stage, within-stage detail) from one pod's probe."""
    if not p:
        return "unreachable", "ssh failed"
    if p.get("started") != "1":
        return "setup", "installing train + vLLM venvs"
    phase = p.get("phase") or "?"
    lastok = p.get("lastok") or ""
    if phase == "prepare":
        if lastok.startswith("prepare"):
            return "prepare done", "starting first seed"
        mb = int(p.get("parentmb") or 0)
        # the 12B parents are ~24 GB; show it landing rather than a static label
        return "prepare", f"parent download {mb / 1024:.1f}/~24 GB (watch it move)"
    seed = phase.replace("chain-seed", "seed ") if phase.startswith("chain-") else phase
    tm = float(p.get("tmtime") or 0)
    em = float(p.get("emtime") or 0)
    evallog = p.get("evallog") or ""
    # whichever log moved last is the live sub-stage
    if em >= tm and evallog:
        which = "eval baseline" if "-baseline" in evallog else f"eval {seed}"
        slice_name = p.get("eslice") or "?"
        prog = p.get("eprog") or ""
        done = p.get("edone") or "0"
        detail = f"{slice_name} {prog}".strip() + f"  ({done}/7 slices)"
        return which, detail
    train = p.get("train") or ""
    if train:
        head, _, rate = train.partition(" [")
        cur, _, total = head.partition("/")
        elapsed, _, remain = rate.partition("<")
        remain = remain.split(",")[0]
        sit = train.rsplit(",", 1)[-1].strip()
        bits = remain.strip().split(":")
        eta = (f"{int(bits[0])}h{int(bits[1]):02d}m" if len(bits) == 3
               else f"{int(bits[0])}m{int(bits[1]):02d}s" if len(bits) == 2 and bits[0]
               else remain.strip())
        return (f"train {seed}",
                f"step {cur.strip():>4}/{total.strip()}  {sit}  eta {eta}")
    return f"train {seed}", "tokenizing / starting"


def fmt_age(seconds: float) -> str:
    if seconds < 0:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f}m"


def main() -> None:
    live = fleet()
    eps = endpoints([r["id"] for r in live.values()])
    jobs = []
    for cell in contracts.CELLS:
        rec = live.get(cell.arm)
        if not rec or rec["id"] not in eps:
            continue
        ip, port = eps[rec["id"]]
        jobs.append((cell, rec, ip, port))
    with ThreadPoolExecutor(max_workers=8) as pool:
        probed = dict(zip(
            [c.arm for c, _, _, _ in jobs],
            pool.map(lambda j: probe(j[0].arm, j[0].slug, j[2], j[3]), jobs),
        ))

    n_seeds = len(contracts.SEEDS)
    total_seeds = len(contracts.CELLS) * n_seeds
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    print(f"seed sweep — {len(contracts.CELLS)} substrates x {n_seeds} seeds @ "
          f"{contracts.EXPECTED_STEPS} steps        {stamp} UTC\n")
    print(f"{'pod':14s} {'seeds':6s} {'stage':16s} {'within stage':44s} "
          f"{'fresh':7s} {'gpu':5s} {'$/hr':5s}")
    print("-" * 104)

    done_total = 0
    rate_total = 0.0
    spent = 0.0
    eta_minutes: list[float] = []
    for cell in contracts.CELLS:
        rec = live.get(cell.arm)
        if not rec:
            print(f"{cell.arm:14s} {'-':6s} {'no pod':16s} "
                  f"{'finished and torn down, or never placed':44s}")
            continue
        p = probed.get(cell.arm, {})
        seeds = int(p.get("seeds") or 0)
        done_total += seeds
        rate = float(rec.get("costPerHr") or 0)
        rate_total += rate
        up = float(rec.get("uptimeSeconds") or 0)
        spent += rate * up / 3600
        stage, detail = stage_of(p)
        now = float(p.get("now") or 0)
        active = max(float(p.get("tmtime") or 0), float(p.get("emtime") or 0),
                     float(p.get("smtime") or 0), float(p.get("rmtime") or 0))
        fresh = fmt_age(now - active) if now and active else "-"
        if now and active and (now - active) > STALE_MIN * 60:
            fresh = f"STALE {fresh}"
        if p.get("fail", "0") not in ("0", ""):
            detail = f"[{p['fail']} PHASE FAILED] " + detail
        print(f"{cell.arm:14s} {f'{seeds}/{n_seeds}':6s} {stage:16s} {detail:44s} "
              f"{fresh:7s} {p.get('gpu', '-'):5s} {rate:5.2f}")
        # crude remaining estimate: whole seeds left, plus this seed's eta
        left = (n_seeds - seeds - 1) * PER_SEED_MIN + PER_SEED_MIN / 2
        eta_minutes.append(max(0.0, left))

    print("-" * 104)
    finish = ""
    if eta_minutes:
        worst = max(eta_minutes)
        finish = (f" · est finish ~"
                  f"{datetime.fromtimestamp(time.time() + worst * 60, UTC):%H:%M} UTC")
    print(f"{done_total}/{total_seeds} seeds done · {len(live)} pods live · "
          f"${rate_total:.2f}/hr · spent ~${spent:.2f}{finish}")
    probe_lines = [p.get("eprobe") for p in probed.values() if p.get("eprobe")]
    if probe_lines:
        print(f"last LoRA probe: {probe_lines[-1]}")


if __name__ == "__main__":
    main()
