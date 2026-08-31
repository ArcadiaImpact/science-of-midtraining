"""Pod-side chain for one Dispatch final-run substrate: midtrain -> Dolci -> AFT -> eval.

One pod owns one arm (control / charter / coin) on 4xH100 and runs its whole
column of the grid:

    phase 1  leg A   full-param midtrain on a 100M-token mix        4 GPUs
    phase 2  leg B   full-param Dolci instruct tuning, 48 steps     4 GPUs
    phase 3  AFT     4 LoRA cells, one per GPU, concurrently        1 GPU each
    phase 4  eval    9 endpoints (pre-AFT + 4 cells x 2 steps)      sharded

Nothing crosses pods: an arm's 24 GB checkpoints never leave the machine that
made them until they are published.

Two contracts this file exists to enforce
-----------------------------------------
1. **Steps are DERIVED from the realized mix, never trusted.** The mix is built
   on the chain basis (BOS included, scimt.train.mix._token_count) and overshoots
   the publication-basis budget by ~0.1%, so the analytic 381 is a prediction,
   not a fact. This computes `realized_total // 262,144`, writes the schedule to
   SCHEDULE.json, and on relaunch REQUIRES equality -- a resumed run that would
   silently train a different number of steps is a hard error.
   (python4/midtraining_prop does exactly this; the pattern is borrowed.)
2. **Every phase is resumable and idempotent.** Each writes a sentinel when it
   completes and skips if the sentinel is present. Never delete a run directory
   to restart it -- relaunch, and the completed phases cost nothing.

Run (on the pod):
    python3 chain.py --arm charter --root /workspace/final_v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
PRIOR_COINS = EXP.parent
REPO_ROOT = PRIOR_COINS.parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP), str(PRIOR_COINS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

DATA_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
DATA_PREFIX = "releases/dispatch-final-v1"
N_GPUS = C.N_GPUS


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def done(path: Path) -> bool:
    return path.is_file()


def mark(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run_sync(cmd: list, log_path: Path, env: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        proc = subprocess.run([str(c) for c in cmd], stdout=handle,
                              stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-40:]
        raise RuntimeError(
            f"{cmd[0]} failed ({proc.returncode}); last lines:\n  " + "\n  ".join(tail)
        )


# ------------------------------------------------------------------ phase 1


def mix_config_path(arm: str) -> Path:
    return EXP / "mix" / f"leg_a_{arm}.yaml"


def derive_schedule(realized_tokens: int) -> dict:
    """Steps and checkpoint positions implied by the mix that was actually built."""
    per_step = C.tokens_per_step(C.MIDTRAIN_MICRO_BATCH, C.MIDTRAIN_GRAD_ACCUM)
    steps = realized_tokens // per_step
    if steps < 1:
        raise ValueError(f"realized mix of {realized_tokens:,} tokens yields no steps")
    schedule = []
    for tokens in C.MIDTRAIN_CHECKPOINT_TOKENS[:-1]:
        step = tokens // per_step
        if step < 1 or step >= steps:
            raise ValueError(
                f"checkpoint at {tokens:,} tokens lands on step {step}, outside "
                f"1..{steps - 1}"
            )
        schedule.append(step)
    schedule.append(steps)  # the final step is always kept
    return {
        "realized_mix_tokens": realized_tokens,
        "tokens_per_step": per_step,
        "max_steps": steps,
        "checkpoint_schedule": schedule,
        "checkpoint_tokens": list(C.MIDTRAIN_CHECKPOINT_TOKENS),
        "analytic_max_steps": C.MIDTRAIN_STEPS,
    }


def assert_stage_matches(derived: dict, stage_name: str) -> None:
    """The stage YAML is static and reviewable; reality must agree with it.

    TrainConfig has no per-run override seam and render_stage takes none, so the
    step budget lives in the stage file. Rather than mutating YAML at run time,
    this checks that the schedule implied by the mix that was ACTUALLY built
    equals what the stage will execute, and refuses otherwise. The mix budget
    leaves 139,008 tokens of headroom before the step count could change, so
    disagreement means something real moved -- not rounding.
    """
    from scimt.train.axolotl import load_stage

    body = load_stage(stage_name).axolotl
    mismatches = {}
    if body.get("max_steps") != derived["max_steps"]:
        mismatches["max_steps"] = (body.get("max_steps"), derived["max_steps"])
    if list(body.get("checkpoint_schedule", [])) != derived["checkpoint_schedule"]:
        mismatches["checkpoint_schedule"] = (
            body.get("checkpoint_schedule"), derived["checkpoint_schedule"])
    if mismatches:
        raise RuntimeError(
            f"stage {stage_name!r} disagrees with the realized mix "
            f"({derived['realized_mix_tokens']:,} tokens): "
            + "; ".join(f"{k}: stage {s!r} vs derived {d!r}"
                        for k, (s, d) in mismatches.items())
            + ". Refusing to train a schedule nobody reviewed."
        )


def load_or_pin_schedule(root: Path, realized_tokens: int) -> dict:
    """Derive the schedule, or require equality with what a prior run pinned."""
    path = root / "SCHEDULE.json"
    derived = derive_schedule(realized_tokens)
    if path.is_file():
        pinned = json.loads(path.read_text())
        for key in ("max_steps", "checkpoint_schedule", "tokens_per_step"):
            if pinned[key] != derived[key]:
                raise RuntimeError(
                    f"SCHEDULE.json pins {key}={pinned[key]!r} but this mix derives "
                    f"{derived[key]!r} -- the run would train a different schedule "
                    "than the one already partly executed. Refusing."
                )
        return pinned
    mark(path, derived)
    return derived


async def phase_mix(root: Path, arm: str) -> dict:
    sentinel = root / "MIX_COMPLETE.json"
    out_path = root / "data" / "leg_a.jsonl"
    if done(sentinel):
        log(f"{arm}: mix already built")
        return json.loads(sentinel.read_text())

    from scimt.train.mix import build_mix, load_mix_config

    cfg = load_mix_config(mix_config_path(arm))
    log(f"{arm}: building leg-A mix, target {cfg.total_tokens:,} tokens "
        f"({len(cfg.sources)} sources)")
    manifest = await build_mix(cfg, out_path)
    underfilled = [s for s in manifest.per_source if s.get("underfilled")]
    if underfilled:
        raise RuntimeError(f"{arm}: sources underfilled: {underfilled}")
    payload = {
        "arm": arm,
        "path": str(out_path),
        "total_tokens": manifest.total_tokens,
        "per_source": manifest.per_source,
        "config": manifest.config,
    }
    mark(sentinel, payload)
    log(f"{arm}: mix built, {manifest.total_tokens:,} tokens")
    return payload


async def phase_midtrain(root: Path, arm: str, mix: dict) -> Path:
    sentinel = root / "MIDTRAIN_COMPLETE.json"
    run_dir = root / "midtrain"
    if done(sentinel):
        log(f"{arm}: midtrain already complete")
        return run_dir

    from scimt.dataset import Dataset
    from scimt.train import TrainConfig, train_dataset

    schedule = load_or_pin_schedule(root, mix["total_tokens"])
    assert_stage_matches(schedule, "midtrain_dispatch_final_v1")
    log(f"{arm}: midtrain {schedule['max_steps']} steps "
        f"(analytic {schedule['analytic_max_steps']}), "
        f"checkpoints {schedule['checkpoint_schedule']}")

    config = TrainConfig(
        backend="axolotl",
        stage="midtrain_dispatch_final_v1",
        model="gemma3_12b",
        seed=C.SEED,
    )
    started = time.time()
    await train_dataset(Dataset.at(Path(mix["path"])), run_dir, config,
                        run_name=f"{arm}-midtrain")
    mark(sentinel, {
        "arm": arm, "run_dir": str(run_dir),
        "minutes": round((time.time() - started) / 60, 2),
        **schedule,
    })
    log(f"{arm}: midtrain done in {(time.time() - started) / 60:.1f} min")
    return run_dir


# ------------------------------------------------------------------ phase 2


def final_checkpoint(run_dir: Path, step: int) -> Path:
    path = run_dir / "checkpoints" / f"checkpoint-{step}"
    if not path.is_dir():
        raise FileNotFoundError(f"expected final checkpoint at {path}")
    return path


async def phase_dolci(root: Path, arm: str, parent: Path) -> Path:
    sentinel = root / "DOLCI_COMPLETE.json"
    run_dir = root / "dolci"
    if done(sentinel):
        log(f"{arm}: Dolci already complete")
        return run_dir

    from scimt.dataset import Dataset
    from scimt.train import TrainConfig, train_dataset

    stage = ("sft_dolci_dispatch_final_v1_control" if arm == "control"
             else "sft_dolci_dispatch_final_v1")
    # Dolci is a chat dataset and Dataset.at takes a path, so a bounded local
    # slice is materialized first. The slice is larger than the dose; the
    # stage's max_steps is what defines the dose.
    slice_path = root / "data" / "dolci.jsonl"
    if not slice_path.is_file():
        await asyncio.to_thread(
            run_sync,
            [sys.executable, POD / "fetch_dolci.py", "--out", slice_path],
            root / "data" / "fetch_dolci.log",
        )
    log(f"{arm}: Dolci SFT, stage {stage}, {C.DOLCI_STEPS} steps, parent {parent}")
    config = TrainConfig(
        backend="axolotl", stage=stage, model="gemma3_12b", seed=C.SEED,
        load_checkpoint_path=str(parent),
    )
    started = time.time()
    await train_dataset(Dataset.at(slice_path), run_dir, config,
                        run_name=f"{arm}-dolci")
    mark(sentinel, {
        "arm": arm, "run_dir": str(run_dir), "parent": str(parent),
        "stage": stage, "steps": C.DOLCI_STEPS,
        "minutes": round((time.time() - started) / 60, 2),
    })
    log(f"{arm}: Dolci done in {(time.time() - started) / 60:.1f} min")
    return run_dir


# ------------------------------------------------------------------ phase 3


def fetch_aft_cells(root: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download
    out: dict[str, Path] = {}
    dest = root / "data" / "aft"
    for cell in C.AFT_CELLS:
        out[cell] = Path(hf_hub_download(
            DATA_REPO, f"{DATA_PREFIX}/aft/aft_{cell}.jsonl",
            repo_type="dataset", local_dir=dest))
    return out


async def train_one_aft(root: Path, arm: str, cell: str, parent: Path,
                        dataset: Path, gpu: int) -> Path:
    run_dir = root / "aft" / cell
    sentinel = run_dir / "AFT_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}/{cell}: already trained")
        return run_dir

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log(f"{arm}/{cell}: AFT on GPU {gpu}, {C.AFT_STEPS} steps")
    started = time.time()
    await asyncio.to_thread(
        run_sync,
        [sys.executable, POD / "train_aft.py", "--cell", cell,
         "--parent", parent, "--dataset", dataset, "--out", run_dir],
        run_dir / "train.log", env,
    )
    mark(sentinel, {
        "arm": arm, "cell": cell, "gpu": gpu, "parent": str(parent),
        "dataset": str(dataset), "steps": C.AFT_STEPS,
        "minutes": round((time.time() - started) / 60, 2),
    })
    log(f"{arm}/{cell}: AFT done in {(time.time() - started) / 60:.1f} min")
    return run_dir


async def phase_aft(root: Path, arm: str, parent: Path) -> dict[str, Path]:
    if len(C.AFT_CELLS) > N_GPUS:
        raise RuntimeError(
            f"{len(C.AFT_CELLS)} cells but {N_GPUS} GPUs -- this scheduler "
            "assumes one cell per GPU"
        )
    datasets = fetch_aft_cells(root)
    results = await asyncio.gather(*[
        train_one_aft(root, arm, cell, parent, datasets[cell], gpu)
        for gpu, cell in enumerate(C.AFT_CELLS)
    ])
    return dict(zip(C.AFT_CELLS, results))


# ------------------------------------------------------------------ phase 4


async def phase_eval(root: Path, arm: str, pre_aft: Path,
                     aft_runs: dict[str, Path]) -> None:
    """Nine endpoints: the pre-AFT parent, plus each cell at both epoch ends."""
    jobs: list[tuple[str, Path, int | None]] = [("pre_aft", pre_aft, None)]
    for cell in C.AFT_CELLS:
        for step in C.AFT_EVAL_STEPS:
            jobs.append((f"{cell}__step{step}", aft_runs[cell], step))

    async def one(name: str, model_dir: Path, step: int | None, gpu: int) -> None:
        out_dir = root / "eval" / name
        sentinel = out_dir / "ENDPOINT_DONE.json"
        if done(sentinel):
            log(f"{arm}/{name}: eval already done")
            return
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TOKENIZERS_PARALLELISM"] = "false"
        cmd = [sys.executable, POD / "evaluate.py", "--endpoint", name,
               "--model", model_dir, "--out", out_dir]
        if step is not None:
            cmd += ["--adapter-step", step]
        started = time.time()
        await asyncio.to_thread(run_sync, cmd, out_dir / "eval.log", env)
        mark(sentinel, {"arm": arm, "endpoint": name, "gpu": gpu,
                        "minutes": round((time.time() - started) / 60, 2)})
        log(f"{arm}/{name}: eval done")

    queue: asyncio.Queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)

    async def worker(gpu: int) -> None:
        while True:
            try:
                name, model_dir, step = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await one(name, model_dir, step, gpu)

    await asyncio.gather(*[worker(g) for g in range(N_GPUS)])


# ----------------------------------------------------------------------- main


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(C.ARMS))
    parser.add_argument("--root", default="/workspace/final_v1")
    parser.add_argument("--phases", default="mix,midtrain,dolci,aft,eval",
                        help="comma-separated subset, in order")
    parser.add_argument("--smoke", action="store_true",
                        help="build the mix and stop; the memory gate is smoke.py")
    args = parser.parse_args()

    C.validate()
    arm = args.arm
    root = Path(args.root) / arm
    root.mkdir(parents=True, exist_ok=True)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    log(f"{arm}: root {root}, phases {phases}")

    if "mix" in phases:
        mix = await phase_mix(root, arm)
    else:
        mix = json.loads((root / "MIX_COMPLETE.json").read_text())
    if args.smoke:
        log(f"{arm}: --smoke, stopping after the mix")
        return

    midtrain_dir = root / "midtrain"
    if "midtrain" in phases:
        midtrain_dir = await phase_midtrain(root, arm, mix)
    schedule = json.loads((root / "SCHEDULE.json").read_text())
    pre_dolci = final_checkpoint(midtrain_dir, schedule["max_steps"])

    dolci_dir = root / "dolci"
    if "dolci" in phases:
        dolci_dir = await phase_dolci(root, arm, pre_dolci)
    parent = final_checkpoint(dolci_dir, C.DOLCI_STEPS)

    aft_runs = {cell: root / "aft" / cell for cell in C.AFT_CELLS}
    if "aft" in phases:
        aft_runs = await phase_aft(root, arm, parent)

    if "eval" in phases:
        await phase_eval(root, arm, parent, aft_runs)

    mark(root / "CHAIN_COMPLETE.json", {
        "arm": arm,
        "midtrain_steps": schedule["max_steps"],
        "dolci_steps": C.DOLCI_STEPS,
        "aft_cells": list(C.AFT_CELLS),
        "endpoints": 1 + len(C.AFT_CELLS) * len(C.AFT_EVAL_STEPS),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    log(f"{arm}: CHAIN COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
