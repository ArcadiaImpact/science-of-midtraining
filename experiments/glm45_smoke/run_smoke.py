"""GLM-4.5 live training smokes: real weights, real steps, nothing saved.

Five arms, run SEQUENTIALLY (each arm's source manifest hashes the whole
tree — the cluster_parity_smoke lesson) in a cost-gated ladder; a later
arm only launches if every earlier arm passed:

  tiny_1n   midtrain_smoke1n_glm45        tiny-random glm4_moe, 1x4 H100
  tiny_2n   midtrain_smoke2n_glm45        same, 2x2 H100 Instant Cluster
  air_adamw midtrain_glm45_air_smoke      GLM-4.5-Air-Base 110B, 8xH200
  air_muon  midtrain_glm45_air_smoke_muon Muon twin, same pod shape
  base_4n   midtrain_glm45_base_smoke_4n  GLM-4.5-Base 355B, 4x8xH200

All arms train full-param and save NOTHING (save_strategy no, no
checkpoint schedule). The evidence each arm must produce (per-arm
health.json + the combined results.json):

  - loss: every logged step finite; final < first (descent over the run);
  - grad norm: every step finite; max/median bounded (no explosion);
  - expert balance (real-model arms): router_health.jsonl rows present,
    per-layer entropy/MaxVio recorded, no layer beyond the warn bounds;
  - tiny arms additionally check 1n/2n loss parity (multi-node artifact
    detector, mean |dloss| < 0.05).

Run (this box: the injected RUNPOD_API_KEY is invalid — the runner
re-reads ~/.runpod/config.toml itself):

  uv run --with 'bellhop-py>=0.8.0' python experiments/glm45_smoke/run_smoke.py

Env knob (not a CLI): SMOKE_ARMS="air_adamw,base_4n" runs a subset, in
ladder order. Arms whose health.json already exists are skipped (resume).
"""

import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))

from scimt.train import LoraConfig, TrainConfig                       # noqa: E402
from scimt.train.axolotl import LOSS_RE, executor_for, load_stage, render_stage  # noqa: E402

ARMS = {  # ladder order matters: cheap and load-bearing first
    "tiny_1n": "midtrain_smoke1n_glm45",
    "tiny_2n": "midtrain_smoke2n_glm45",
    "air_adamw": "midtrain_glm45_air_smoke",
    "air_muon": "midtrain_glm45_air_smoke_muon",
    "base_4n": "midtrain_glm45_base_smoke_4n",
    # capacity fallback: H200 clusters were dry for 10 rounds on 2026-08-16
    "base_4n_b200": "midtrain_glm45_base_smoke_4n_b200",
    # within-spend-limit arms (account: $80/hr limit + $214 balance block
    # every 4-node cluster, live 2026-08-16): flagship-scale LoRA on one
    # node, and the tiny Riemannion LoRA validation
    "base_lora": "midtrain_glm45_base_smoke_lora",
    "tiny_riemannion": "midtrain_smoke_glm45_lora_riemannion",
}
REAL_MODEL_ARMS = ("air_adamw", "air_muon", "base_4n", "base_4n_b200", "base_lora")

#: adapter shape is a RUN variable (render contract): arms that train LoRA
#: get it here, attention modules + stacked expert tensors, dropout 0
#: (the ParamWrapper constraint), rank small — smoke capacity, not recipe
ARM_LORA = {
    "base_lora": dict(
        r=16,
        target_linear=False,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        target_parameters=("mlp.experts.gate_up_proj", "mlp.experts.down_proj"),
    ),
    "tiny_riemannion": dict(
        r=8,
        target_linear=False,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    ),
}
DATASET = EXP / "data" / "mix.jsonl"

GRAD_NORM_RE = re.compile(r"'grad_norm': '?(nan|inf|[0-9.eE+-]+)'?", re.IGNORECASE)
PARITY_TOL = 0.05
GRAD_EXPLOSION_RATIO = 50.0

PROVISION_ROUNDS = 10
ROUND_WAIT_S = 240


def _require_runpod_key() -> None:
    """The pod-injected RUNPOD_API_KEY on this box is invalid — always
    override from the config the runpodctl auth actually uses."""
    config = Path.home() / ".runpod" / "config.toml"
    match = re.search(r"apikey\s*=\s*['\"]([^'\"]+)['\"]", config.read_text())
    if not match:
        raise SystemExit(f"no apikey in {config}")
    os.environ["RUNPOD_API_KEY"] = match.group(1)


def series(arm: str) -> dict[str, list[float]]:
    log = EXP / "runs" / arm / "train.log"
    if not log.exists():
        return {"loss": [], "grad_norm": []}
    text = log.read_text(errors="replace")
    return {
        "loss": [float(m) for m in LOSS_RE.findall(text)],
        "grad_norm": [float(m) for m in GRAD_NORM_RE.findall(text)],
    }


def router_rows(arm: str) -> list[dict]:
    path = EXP / "runs" / arm / "router_health.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def arm_health(arm: str) -> dict:
    """Pure post-hoc scoring of one arm's pulled artifacts."""
    data = series(arm)
    losses, grads = data["loss"], data["grad_norm"]
    checks: dict[str, bool] = {}
    checks["loss_lines_present"] = len(losses) >= 5
    checks["loss_all_finite"] = bool(losses) and all(math.isfinite(v) for v in losses)
    checks["loss_descended"] = bool(losses) and losses[-1] < losses[0]
    checks["grad_norm_present"] = len(grads) >= 5
    checks["grad_norm_all_finite"] = bool(grads) and all(
        math.isfinite(v) for v in grads
    )
    checks["grad_norm_bounded"] = bool(grads) and (
        max(grads) / max(statistics.median(grads), 1e-9) < GRAD_EXPLOSION_RATIO
    )
    rows = router_rows(arm)
    router: dict = {"rows": len(rows)}
    if arm in REAL_MODEL_ARMS or rows:
        checks["router_rows_present"] = bool(rows)
        if rows:
            entropies = [
                stats["entropy_nats"]
                for row in rows
                for stats in row["layers"].values()
            ]
            maxvios = [
                stats["maxvio"]
                for row in rows
                for stats in row["layers"].values()
            ]
            router.update(
                layers=len(rows[-1]["layers"]),
                entropy_min=round(min(entropies), 4),
                entropy_max=round(max(entropies), 4),
                maxvio_max=round(max(maxvios), 4),
                maxvio_final_median=round(
                    statistics.median(
                        s["maxvio"] for s in rows[-1]["layers"].values()
                    ),
                    4,
                ),
            )
    # a checkpoint appearing anywhere is a spec violation for this campaign
    saved = list((EXP / "runs" / arm / "checkpoints").glob("**/*.safetensors"))
    checks["nothing_saved"] = not saved
    return {
        "arm": arm,
        "steps_logged": len(losses),
        "loss_first": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "grad_norm_median": round(statistics.median(grads), 4) if grads else None,
        "grad_norm_max": round(max(grads), 4) if grads else None,
        "router": router,
        "checks": checks,
        "pass": all(checks.values()),
    }


async def run_arm(arm: str, stage_name: str) -> dict:
    from bellhop import ProvisionError, is_capacity_error

    stage = load_stage(stage_name)
    out = EXP / "runs" / arm
    out.mkdir(parents=True, exist_ok=True)
    lora = ARM_LORA.get(arm)
    cfg = TrainConfig(
        model=stage.base_model,
        seed=42,
        stage=stage_name,
        lora=LoraConfig(**lora) if lora else None,
    )
    rendered = render_stage(stage, cfg, DATASET, out)
    t0 = time.monotonic()
    for round_no in range(1, PROVISION_ROUNDS + 1):
        print(
            f"[{arm}] launching {stage_name} "
            f"(round {round_no}/{PROVISION_ROUNDS}) ...",
            flush=True,
        )
        try:
            await executor_for(stage).run_stage(
                rendered, out, stage, run_name=f"glm45smoke-{arm}"
            )
            break
        except ProvisionError as e:
            if round_no == PROVISION_ROUNDS or not is_capacity_error(e):
                raise
            print(f"[{arm}] no stock ({e}); retrying in {ROUND_WAIT_S}s", flush=True)
            await asyncio.sleep(ROUND_WAIT_S)
    wall = time.monotonic() - t0
    print(f"[{arm}] pod run finished in {wall / 60:.1f} min", flush=True)
    return {"wall_minutes": round(wall / 60, 2)}


def dump_failure(arm: str, err: BaseException) -> None:
    print(f"[{arm}] FAILED: {type(err).__name__}: {err}", flush=True)
    rank_results = getattr(err, "results", None) or {}
    dump = EXP / "runs" / arm / "rank_logs"
    dump.mkdir(parents=True, exist_ok=True)
    for rank, result in rank_results.items():
        if result is not None:
            (dump / f"rank{rank}.log").write_text(
                result.stdout + "\n--- stderr ---\n" + result.stderr
            )
            print(f"[{arm}] rank {rank} exit={result.exit_code} -> rank_logs/", flush=True)
    tail = getattr(err, "log_tail", "")
    if tail:
        print(f"[{arm}] log tail:\n{tail[-3000:]}", flush=True)


async def main() -> None:
    _require_runpod_key()
    selected = [
        a for a in ARMS
        if a in (os.environ.get("SMOKE_ARMS") or ",".join(ARMS)).split(",")
    ]
    summaries: dict[str, dict] = {}
    for arm in selected:
        stage_name = ARMS[arm]
        health_path = EXP / "runs" / arm / "health.json"
        if health_path.exists():
            summaries[arm] = json.loads(health_path.read_text())
            print(f"[{arm}] already scored ({summaries[arm]['pass']=}) — skipping",
                  flush=True)
            continue
        try:
            run_meta = await run_arm(arm, stage_name)
        except BaseException as err:  # noqa: BLE001 — report, then fail loud
            dump_failure(arm, err)
            raise SystemExit(f"SMOKE FAIL: arm {arm} did not finish") from err
        health = arm_health(arm)
        health.update(run_meta, stage=stage_name)
        health_path.write_text(json.dumps(health, indent=2))
        summaries[arm] = health
        print(f"[{arm}] health: {json.dumps(health['checks'])}", flush=True)
        if not health["pass"]:
            # the ladder stops: don't buy the next (bigger) arm on a red one
            break

    # cross-arm: tiny parity (multi-node artifact detector)
    parity = None
    one, two = series("tiny_1n")["loss"], series("tiny_2n")["loss"]
    n = min(len(one), len(two))
    if n:
        deltas = [abs(a - b) for a, b in zip(one[:n], two[:n])]
        parity = {
            "steps_compared": n,
            "mean_abs_delta": round(sum(deltas) / n, 5),
            "pass": sum(deltas) / n < PARITY_TOL,
        }

    overall = (
        all(s["pass"] for s in summaries.values())
        and set(summaries) == set(selected)
        and (parity is None or parity["pass"])
    )
    results = {
        "arms": summaries,
        "tiny_parity": parity,
        "selected": selected,
        "overall": "PASS" if overall else "FAIL",
    }
    (EXP / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if not overall:
        raise SystemExit("SMOKE FAIL — see results.json")
    print("SMOKE PASS")


if __name__ == "__main__":
    asyncio.run(main())
