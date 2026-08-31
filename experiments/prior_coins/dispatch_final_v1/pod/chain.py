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
import hashlib
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
#: pinned so every arm consumes byte-identical inputs even if the repo moves
DATA_REVISION = os.environ.get("FINAL_V1_DATA_REVISION", "main")
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


REQUIRED_GPUS = C.N_GPUS


def preflight_gpus() -> dict:
    """Refuse unless the hardware the arithmetic assumes is actually present.

    Every token/step number in contracts.py is computed for exactly
    REQUIRED_GPUS devices. On two visible GPUs the same 381 updates deliver half
    the intended positions, and nothing downstream would show it.
    """
    import torch

    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    gib = [round(torch.cuda.get_device_properties(i).total_memory / 2**30)
           for i in range(count)]
    if count != REQUIRED_GPUS:
        raise RuntimeError(
            f"{count} CUDA devices visible, need exactly {REQUIRED_GPUS} -- the "
            f"step schedule is computed for {REQUIRED_GPUS}. Devices: {names}"
        )
    if min(gib) < 79:
        raise RuntimeError(f"GPUs below 80 GB: {list(zip(names, gib))}")
    log(f"preflight: {count} x {names[0]} ({min(gib)} GiB)")
    return {"count": count, "names": names, "memory_gib": gib}


def fetch_release(root: Path, arm: str) -> dict[str, Path]:
    """Pull the published corpora onto the pod.

    The mix YAMLs cannot name a Hub repo -- scimt.train.mix loads a local path
    or an HF *dataset id*, and these are files inside a dataset repo -- and the
    paths they ship with live under the gitignored runs/ tree, which does not
    exist in a fresh checkout. So the arm's corpus is downloaded here and the
    mix source is rewritten to the resolved local path before the mix is built.
    """
    from huggingface_hub import hf_hub_download

    documents = C.ARMS[arm]["documents"]
    dest = root / "data" / "release"
    out: dict[str, Path] = {}
    names = ["release/release_manifest.json"]
    if documents:
        names.append(f"release/{documents}/corpus.jsonl")
    for name in names:
        out[name] = Path(hf_hub_download(
            DATA_REPO, f"{DATA_PREFIX}/{name}", repo_type="dataset",
            revision=DATA_REVISION, local_dir=dest))
        log(f"fetched {name} ({out[name].stat().st_size / 1e6:.1f} MB)")

    if documents:
        manifest = json.loads(out["release/release_manifest.json"].read_text())
        want = manifest["arms"][documents]
        corpus = out[f"release/{documents}/corpus.jsonl"]
        digest = hashlib.sha256()
        with corpus.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != want["sha256"]:
            raise RuntimeError(
                f"{documents} corpus sha256 {digest.hexdigest()} != published "
                f"{want['sha256']} -- refusing to train on unverified bytes"
            )
        rows = sum(1 for line in corpus.open() if line.strip())
        if rows != want["docs"]:
            raise RuntimeError(f"{documents}: {rows} rows != {want['docs']}")
        log(f"{documents}: sha256 and {rows:,} rows verified against the manifest")
    return out


def fetch_dolmino(root: Path, arm: str) -> Path:
    """Materialize this arm's Dolmino slice locally (see pod/fetch_dolmino.py)."""
    out = root / "data" / "dolmino.jsonl"
    if out.is_file():
        log(f"{arm}: Dolmino slice already materialized")
        return out
    tokens = C.ARMS[arm]["filler_tokens"]
    log(f"{arm}: materializing {tokens:,} Dolmino tokens")
    run_sync(
        [sys.executable, POD / "fetch_dolmino.py", "--out", out, "--tokens", tokens],
        root / "data" / "fetch_dolmino.log",
    )
    return out


def mix_config_path(root: Path, arm: str) -> Path:
    """Render the arm's mix config with pod-local source paths."""
    import yaml

    src = EXP / "mix" / f"leg_a_{arm}.yaml"
    body = yaml.safe_load(src.read_text())
    documents = C.ARMS[arm]["documents"]
    dolmino = root / "data" / "dolmino.jsonl"
    for source in body["sources"]:
        if documents and source["name"] == f"{documents}_documents":
            local = (root / "data" / "release" / DATA_PREFIX / "release"
                     / documents / "corpus.jsonl")
            if not local.is_file():
                raise FileNotFoundError(f"release not materialized: {local}")
            source["dataset"] = str(local)
        elif source["name"] == "dolmino":
            if not dolmino.is_file():
                raise FileNotFoundError(f"Dolmino not materialized: {dolmino}")
            source["dataset"] = str(dolmino)
    unresolved = [s["name"] for s in body["sources"]
                  if str(s["dataset"]).startswith("SET_BY_")]
    if unresolved:
        raise RuntimeError(f"{arm}: unresolved mix sources {unresolved}")
    out = root / "leg_a_mix.yaml"
    out.write_text(yaml.safe_dump(body, sort_keys=False))
    return out


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

    fetch_release(root, arm)
    fetch_dolmino(root, arm)
    cfg = load_mix_config(mix_config_path(root, arm))
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
    # return_exceptions: one cell's failure must not cancel siblings whose
    # subprocess has already finished but whose sentinel is not yet written.
    results = await asyncio.gather(*[
        train_one_aft(root, arm, cell, parent, datasets[cell], gpu)
        for gpu, cell in enumerate(C.AFT_CELLS)
    ], return_exceptions=True)
    failed = {cell: r for cell, r in zip(C.AFT_CELLS, results)
              if isinstance(r, BaseException)}
    if failed:
        raise RuntimeError(
            "AFT cells failed: "
            + "; ".join(f"{c}: {type(e).__name__}: {e}" for c, e in failed.items())
        )
    return dict(zip(C.AFT_CELLS, results))


# ------------------------------------------------------------------ phase 4


async def phase_eval(root: Path, arm: str, parent: Path) -> None:
    """Sample all nine endpoints, SHARDED one engine per GPU (eval_sharded.sh).

    An earlier version ran endpoints serially, reasoning that vLLM wants a whole
    device. True per ENGINE -- each needs its own ~24 GB model copy plus KV
    cache, so four will not fit on one card -- but the pod has four cards, so the
    conclusion was wrong and it cost ~4x the wall clock. One engine per GPU, the
    way the AFT phase already works.
    """
    sentinel = root / "EVAL_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}: eval already complete")
        return
    out = root / "eval"
    started = time.time()
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "eval_sharded.sh", arm, root.parent],
        out / "evaluate.log",
    )
    expected = len(C.EVAL_SLICES) * len(C.EVAL_SURFACES)
    missing = []
    for name in ["pre_aft"] + [f"{c}-step{s}" for c in C.AFT_CELLS
                               for s in C.AFT_EVAL_STEPS]:
        got = len(list((out / name).glob("*__*.jsonl"))) if (out / name).is_dir() else 0
        if got != expected:
            missing.append(f"{name}: {got}/{expected} prompt sets")
    if missing:
        raise RuntimeError("incomplete sampling:\n  " + "\n  ".join(missing))
    mark(sentinel, {
        "arm": arm,
        "endpoints": 1 + len(C.AFT_CELLS) * len(C.AFT_EVAL_STEPS),
        "prompt_sets_per_endpoint": expected,
        "minutes": round((time.time() - started) / 60, 2),
    })
    log(f"{arm}: eval complete")


async def phase_publish(root: Path, arm: str) -> dict:
    """Durably persist everything expensive BEFORE the pod can be destroyed.

    Local checkpoint dirs are impermanent (CLAUDE.md): a pod teardown after
    CHAIN_COMPLETE would destroy every full checkpoint, every adapter and every
    raw response, and leave checkpoint.json pointers aimed at paths that no
    longer exist. This is a required phase, not a convenience.
    """
    sentinel = root / "PUBLISH_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}: already published")
        return json.loads(sentinel.read_text())
    started = time.time()
    result = await asyncio.to_thread(
        run_sync,
        [sys.executable, POD / "publish_results.py", "--arm", arm, "--root", root],
        root / "publish.log",
    )
    del result
    receipt = json.loads((root / "publish_receipt.json").read_text())
    mark(sentinel, {"arm": arm, "minutes": round((time.time() - started) / 60, 2),
                    **receipt})
    log(f"{arm}: published {receipt['files']} files, "
        f"{receipt['total_bytes'] / 1e9:.1f} GB")
    return receipt


# ----------------------------------------------------------------------- main


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(C.ARMS))
    parser.add_argument("--root", default="/workspace/final_v1")
    parser.add_argument("--phases", default="mix,midtrain,dolci,aft,eval,publish",
                        help="comma-separated subset, in order")
    parser.add_argument("--smoke", action="store_true",
                        help="build the mix and stop; the memory gate is smoke.py")
    args = parser.parse_args()

    C.validate()
    if not args.smoke:
        preflight_gpus()
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
    del aft_runs  # evaluate.py resolves adapters from root/aft itself

    if "eval" in phases:
        await phase_eval(root, arm, parent)

    if "publish" in phases:
        await phase_publish(root, arm)

    # CHAIN_COMPLETE means "this arm is finished and durable", so it must not be
    # written by a partial --phases run: a later reader cannot tell the
    # difference, and the pod would look safe to destroy.
    required = {"mix", "midtrain", "dolci", "aft", "eval", "publish"}
    if not required.issubset(phases):
        log(f"{arm}: phases {sorted(required - set(phases))} not requested; "
            "NOT writing CHAIN_COMPLETE")
        return
    for name in ("MIX_COMPLETE", "MIDTRAIN_COMPLETE", "DOLCI_COMPLETE",
                 "EVAL_COMPLETE", "PUBLISH_COMPLETE"):
        if not (root / f"{name}.json").is_file():
            raise RuntimeError(f"{arm}: {name}.json missing; refusing to complete")
    for cell in C.AFT_CELLS:
        if not (root / "aft" / cell / "AFT_COMPLETE.json").is_file():
            raise RuntimeError(f"{arm}: AFT cell {cell} never completed")

    mark(root / "CHAIN_COMPLETE.json", {
        "arm": arm,
        "midtrain_steps": schedule["max_steps"],
        "dolci_steps": C.DOLCI_STEPS,
        "aft_cells": list(C.AFT_CELLS),
        "endpoints": 1 + len(C.AFT_CELLS) * len(C.AFT_EVAL_STEPS),
        "published": json.loads((root / "PUBLISH_COMPLETE.json").read_text()),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    log(f"{arm}: CHAIN COMPLETE (durable)")


if __name__ == "__main__":
    asyncio.run(main())
