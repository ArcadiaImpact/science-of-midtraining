"""Per-arm v4 AFT chain on a true-midtrained parent: train once, evaluate 6 endpoints.

One pod per arm (charter or coin). The parent is
``jbostock/scimt-dispatch-models-v1`` at ``sft/<arm>/checkpoint-48`` — the same
1-epoch midtrained + Dolci-SFT checkpoint his v1 AFT gate used, so the only variable
we change is the episode set.

Order is deliberate:

1. **Baseline eval first**, on the bare parent. It validates the whole eval path
   before we spend an hour training, and it is the within-harness anchor for lift —
   the parents already separate by +0.178 before any AFT, so the raw number at a
   checkpoint means little on its own.
2. Train 512 optimizer steps, saving every 32 (16 checkpoints, optimizer state kept).
3. Evaluate a log-spaced subset: steps 32, 64, 128, 256, 512. All 16 checkpoints stay
   on disk and are uploaded, so the trajectory can be filled in later without
   retraining.

Every endpoint is sampled over all six v4 slices, then scored off-pod with
``score_factorised`` (per-run, three separate channels).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scimt.train import LoraConfig, TrainConfig  # noqa: E402
from scimt.train.axolotl import (  # noqa: E402
    finalize_training_attribution,
    load_stage,
    render_stage,
)

from experiments.prior_coins.pod.dispatch_sdf_aft_v1_chain import (  # noqa: E402
    atomic_json,
    run_axolotl_on_gpu,
    upload_and_verify,
    upload_file_verified,
)

ARMS = ("charter", "coin")
PARENT_REPO = "jbostock/scimt-dispatch-models-v1"
PARENT_PREFIX = "sft/{arm}/checkpoint-48"
MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
VERSION = "dispatch_v4_aft"
REMOTE_ROOT = "extensions/v4_aft"
STAGE_NAME = "aft_dispatch_v4_midtrain"
TRAIN_ROWS = 8_192
EXPECTED_STEPS = 512
SAVE_EVERY = 32
EXPECTED_CHECKPOINTS = tuple(range(SAVE_EVERY, EXPECTED_STEPS + 1, SAVE_EVERY))
#: log-spaced endpoints; covers the region where the v1 gate saw a reversal
EVAL_STEPS = (32, 64, 128, 256, 512)
SLICES = (
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)
EVAL_PYTHON = "/workspace/venv-dispatch-eval/bin/python"
FORENSICS_POD = REPO_ROOT / "experiments/prior_coins/generalization_forensics/pod"
LORA = LoraConfig(
    r=32, alpha=64, dropout=0.05, target_linear=False,
    target_modules=(
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ),
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run_sync(cmd: list, log_path: Path, env: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        code = subprocess.call(
            [str(c) for c in cmd], stdout=handle, stderr=subprocess.STDOUT, env=env
        )
    if code:
        raise RuntimeError(
            f"command failed ({code}): {cmd}\n--- tail ---\n"
            + log_path.read_text(errors="replace")[-8_000:]
        )


def validate_training(run_dir: Path) -> dict:
    checkpoints = run_dir / "checkpoints"
    found = sorted(
        int(p.name.rsplit("-", 1)[-1])
        for p in checkpoints.glob("checkpoint-*")
        if p.name.rsplit("-", 1)[-1].isdigit()
    )
    if tuple(found) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"checkpoint steps {found}, expected {EXPECTED_CHECKPOINTS}")
    for step in EXPECTED_CHECKPOINTS:
        ckpt = checkpoints / f"checkpoint-{step}"
        if not (ckpt / "adapter_config.json").is_file() or not any(
            ckpt.glob("adapter_model.*")
        ):
            raise RuntimeError(f"{ckpt}: missing adapter files")
        if not any(ckpt.glob("optimizer.pt")) and not any(ckpt.glob("optimizer.bin")):
            raise RuntimeError(f"{ckpt}: missing optimizer state")
        if not (ckpt / "scheduler.pt").is_file():
            raise RuntimeError(f"{ckpt}: missing scheduler state")
        if not (ckpt / "trainer_state.json").is_file():
            raise RuntimeError(f"{ckpt}: missing trainer state")
    provenance = json.loads((run_dir / "training_provenance.json").read_text())
    actual = provenance.get("actual", {})
    if actual.get("global_step") != EXPECTED_STEPS:
        raise RuntimeError(f"global step {actual.get('global_step')} != {EXPECTED_STEPS}")
    return provenance


async def train_arm(root: Path, arm: str, parent: Path) -> Path:
    run_dir = root / "training"
    complete = run_dir / "COMPLETE.json"
    if complete.is_file():
        validate_training(run_dir)
        log(f"{arm}: training already complete")
        return run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    dataset = root / "data" / "datasets" / "aft_agreement.jsonl"
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    stage = load_stage(STAGE_NAME)
    config = TrainConfig(
        backend="axolotl", stage=STAGE_NAME, model="gemma3_12b_it", seed=42,
        load_checkpoint_path=str(parent), lora=LORA,
    )
    rendered = render_stage(stage, config, dataset, run_dir)
    started = time.time()
    log(f"{arm}: training {TRAIN_ROWS} agreement rows -> {EXPECTED_STEPS} steps")
    await run_axolotl_on_gpu(rendered, run_dir / "train.log", 0)
    finalize_training_attribution(rendered, run_dir)
    provenance = validate_training(run_dir)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    info = {
        "version": VERSION,
        "arm": arm,
        "parameterization": "lora",
        "parent_repo": PARENT_REPO,
        "parent_prefix": PARENT_PREFIX.format(arm=arm),
        "dataset_sha256": provenance["dataset"]["sha256"],
        "training_rows": TRAIN_ROWS,
        "stage": STAGE_NAME,
        "seed": 42,
        "minutes": round((time.time() - started) / 60, 2),
        "lora": asdict(LORA),
        "optimizer_steps": provenance["actual"]["global_step"],
        "checkpoint_steps": list(EXPECTED_CHECKPOINTS),
        "eval_steps": list(EVAL_STEPS),
        "optimizer_state_saved": True,
    }
    atomic_json(complete, info)
    log(f"{arm}: trained in {info['minutes']} min; uploading checkpoints")
    upload = await asyncio.to_thread(
        upload_and_verify, run_dir, f"{REMOTE_ROOT}/{arm}/training",
        run_dir / "ARTIFACT_MANIFEST.local.json",
    )
    atomic_json(complete, {**info, "upload": upload})
    await asyncio.to_thread(
        upload_file_verified, complete, f"{REMOTE_ROOT}/{arm}/training/COMPLETE.json"
    )
    log(f"{arm}: checkpoint upload verified")
    return run_dir


def write_sanity_prompts(root: Path, out_dir: Path) -> None:
    """Teacher-forced spot check: does the endpoint reproduce known training rows?"""
    dataset = root / "data" / "datasets" / "aft_agreement.jsonl"
    rows = [json.loads(l) for l in dataset.read_text().splitlines()[:64]]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sanity_prompts.jsonl").open("w") as handle:
        for r in rows:
            handle.write(json.dumps({
                "id": r["metadata"]["episode_id"],
                "prompt": r["messages"][0]["content"],
                "expected": r["messages"][1]["content"],
            }) + "\n")


def evaluate_endpoint(root: Path, arm: str, name: str, model_dir: Path) -> None:
    out_dir = root / "results" / name
    if all((out_dir / f"{s}.jsonl").is_file() for s in SLICES):
        log(f"{arm}/{name}: eval already complete")
        return
    write_sanity_prompts(root, out_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    cmd = [
        EVAL_PYTHON, FORENSICS_POD / "pod_generate.py",
        "--model", model_dir, "--name", name,
        "--out-dir", out_dir, "--work", root,
    ]
    for slice_name in SLICES:
        prompts = root / "data" / "prompts" / f"{slice_name}.jsonl"
        if not prompts.is_file():
            raise FileNotFoundError(prompts)
        cmd += ["--prompt-set", f"{slice_name}={prompts}"]
    cmd += ["--prompt-set", f"sanity={out_dir / 'sanity_prompts.jsonl'}"]
    run_sync(cmd, root / "logs" / f"eval-{name}.log", env)
    log(f"{arm}/{name}: eval complete")


def merge_checkpoint(root: Path, adapter: Path, tag: str) -> Path:
    merged = root / "merged" / tag
    if merged.exists():
        shutil.rmtree(merged)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    run_sync(
        [sys.executable, FORENSICS_POD / "pod_merge.py",
         "--base", root / "parent", "--adapter", adapter, "--output", merged],
        root / "logs" / f"merge-{tag}.log", env,
    )
    return merged


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    arm = args.arm
    root = Path(os.environ.get("V4_ROOT", "/workspace/v4aft"))
    parent = root / "parent"
    if not (parent / "config.json").is_file():
        raise RuntimeError(f"parent model missing: {parent}")

    manifest = json.loads((root / "data" / "dataset_manifest.json").read_text())
    if manifest["training"]["rows"] != TRAIN_ROWS:
        raise RuntimeError("dataset manifest row count mismatch")
    if manifest["version"] != VERSION:
        raise RuntimeError(f"unexpected dataset version {manifest['version']}")
    log(f"{arm}: data ok — {manifest['training']['rows']} rows, "
        f"train clauses {manifest['train_clauses']}, held out {manifest['held_out_clauses']}")

    # 1. baseline first: validates the eval path before spending training time, and
    #    is the anchor for lift (these parents already separate before any AFT)
    evaluate_endpoint(root, arm, f"{arm}-baseline", parent)
    (root / "results" / f"{arm}-baseline" / "ENDPOINT_DONE.json").write_text(
        json.dumps({"arm": arm, "endpoint": "baseline"}) + "\n"
    )
    log(f"{arm}: baseline endpoint done")

    # 2. train once
    run_dir = await train_arm(root, arm, parent)

    # 3. evaluate the log-spaced trajectory
    for step in EVAL_STEPS:
        name = f"{arm}-step{step}"
        adapter = run_dir / "checkpoints" / f"checkpoint-{step}"
        if not adapter.is_dir():
            raise RuntimeError(f"missing checkpoint for step {step}: {adapter}")
        merged = merge_checkpoint(root, adapter, f"step{step}")
        try:
            evaluate_endpoint(root, arm, name, merged)
        finally:
            shutil.rmtree(merged, ignore_errors=True)
            shutil.rmtree(root / "runtime_views" / name, ignore_errors=True)
        (root / "results" / name / "ENDPOINT_DONE.json").write_text(
            json.dumps({"arm": arm, "endpoint": f"step{step}",
                        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n"
        )
        log(f"{arm}/step{step}: ENDPOINT DONE")

    # 4. ship the raw responses so scoring happens off-pod
    upload = await asyncio.to_thread(
        upload_and_verify, root / "results", f"{REMOTE_ROOT}/{arm}/results",
        root / "results" / "ARTIFACT_MANIFEST.local.json",
    )
    (root / "CHAIN_COMPLETE.json").write_text(
        json.dumps({"arm": arm, "endpoints": ["baseline"] + [f"step{s}" for s in EVAL_STEPS],
                    "slices": list(SLICES), "upload": upload,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n"
    )
    log(f"{arm}: CHAIN COMPLETE ({len(EVAL_STEPS) + 1} endpoints x {len(SLICES)} slices)")


if __name__ == "__main__":
    asyncio.run(main())
