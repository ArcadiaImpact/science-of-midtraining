"""Cluster-parity smoke driver: 2-node Instant Cluster vs single-node pod.

Two arms, identical recipe (world size 8, global batch 16, seed 42, same
committed corpus): `midtrain_smoke1n_gemma3_12b` (1 pod x 8 H100) and
`midtrain_smoke2n_gemma3_12b` (2 nodes x 4 H100, hf-bus checkpoint at step 2).
(H100 rather than H200: the first attempt found zero H200 cluster stock;
H100/H200 share sm_90 so the cu126 image + pin set carry over.)
Run concurrently as plain awaits (repo convention: no pipeline framework in
experiment runners). Acceptance:

  1. both arms finish (exit 0 end-to-end through BellhopExecutor);
  2. loss curves match: mean |Δloss| over matched steps < 0.05 and final
     losses within 0.05 (bf16 + different FSDP topology => close, not
     bit-identical);
  3. the 2-node arm's checkpoints.jsonl carries an hf:// pointer row.

Run (needs RUNPOD_API_KEY + HF_TOKEN in env, e.g. `set -a; source ~/.env`):
  uv run --with bellhop-py==0.8.0 python experiments/cluster_parity_smoke/run_smoke.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))

from scimt.train import TrainConfig                                  # noqa: E402
from scimt.train.axolotl import LOSS_RE, executor_for, load_stage, render_stage  # noqa: E402

ARMS = {
    "one_node": "midtrain_smoke1n_gemma3_12b",
    "two_node": "midtrain_smoke2n_gemma3_12b",
}
DATASET = EXP / "data" / "corpus.jsonl"
MEAN_TOL = FINAL_TOL = 0.05


async def run_arm(arm: str, stage_name: str) -> dict:
    stage = load_stage(stage_name)
    out = EXP / "runs" / arm
    out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(model=stage.base_model, seed=42, stage=stage_name)
    rendered = render_stage(stage, cfg, DATASET, out)
    t0 = time.monotonic()
    print(f"[{arm}] launching stage {stage_name} ...", flush=True)
    await executor_for(stage).run_stage(rendered, out, stage, run_name=f"parity-{arm}")
    wall = time.monotonic() - t0
    print(f"[{arm}] done in {wall/60:.1f} min", flush=True)
    return {"arm": arm, "stage": stage_name, "wall_minutes": round(wall / 60, 2)}


def losses(arm: str) -> list[float]:
    log = EXP / "runs" / arm / "train.log"
    return [float(m) for m in LOSS_RE.findall(log.read_text(errors="replace"))]


async def main() -> None:
    # return_exceptions so one arm's stock-out can't kill its sibling
    # mid-provision (bellhop's own finally still tears the failed arm down)
    results = await asyncio.gather(
        *(run_arm(a, s) for a, s in ARMS.items()), return_exceptions=True)
    failed = {a: r for a, r in zip(ARMS, results) if isinstance(r, BaseException)}
    if failed:
        for arm, err in failed.items():
            print(f"[{arm}] FAILED: {type(err).__name__}: {err}", flush=True)
        raise SystemExit(f"SMOKE FAIL: arm(s) {sorted(failed)} did not finish")
    results = list(results)

    one, two = losses("one_node"), losses("two_node")
    n = min(len(one), len(two))
    if n == 0:
        raise SystemExit("PARITY FAIL: no loss lines parsed from at least one arm")
    deltas = [abs(a - b) for a, b in zip(one[:n], two[:n])]
    mean_d, final_d = sum(deltas) / n, abs(one[n - 1] - two[n - 1])

    rows_path = EXP / "runs" / "two_node" / "checkpoints.jsonl"
    hf_row = None
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            if str(row.get("state_path", "")).startswith("hf://"):
                hf_row = row
    ok = mean_d < MEAN_TOL and final_d < FINAL_TOL and hf_row is not None

    summary = {
        "arms": results,
        "steps_compared": n,
        "steps_per_arm": {"one_node": len(one), "two_node": len(two)},
        "mean_abs_delta": round(mean_d, 5),
        "final_abs_delta": round(final_d, 5),
        "final_losses": {"one_node": one[n - 1], "two_node": two[n - 1]},
        "hf_checkpoint": (hf_row or {}).get("state_path"),
        "parity": "PASS" if ok else "FAIL",
    }
    out = EXP / "results.json"
    out.write_text(json.dumps(summary, indent=2))
    curves = EXP / "loss_curves.jsonl"
    with curves.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"step": i + 1, "one_node": one[i],
                                "two_node": two[i], "abs_delta": deltas[i]}) + "\n")
    print(json.dumps(summary, indent=2))
    if not ok:
        raise SystemExit("PARITY FAIL — see results.json")
    print("PARITY PASS")


if __name__ == "__main__":
    asyncio.run(main())
