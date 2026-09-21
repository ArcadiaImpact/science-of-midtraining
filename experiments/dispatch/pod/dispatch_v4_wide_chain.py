"""Per-arm v4_wide AFT chain: the v4 run with an easier cost comparison, run faster.

Same parents, same recipe, same 6 endpoints as ``dispatch_v4_aft_chain`` — the
dataset moves the per-run cost-gap band from (0.08, 0.40) to (0.25, 0.60) and
nothing else. See ``build_dispatch_v4_wide.py`` for the pre-registered prediction.

Three throughput changes over the v4 chain, none of which touch the optimisation
trajectory or what is measured:

1. **micro-batch 16 x accum 2** (stage ``aft_dispatch_v4_wide``) — same global
   batch of 32, half as many sequential micro-steps.
2. **LoRA served natively, no merges.** v4 merged each adapter into a full 24 GB
   copy of the parent and started a fresh vLLM per endpoint: 5 merges + 6 loads
   per arm. Here the base is resident once and the 5 checkpoints are swapped
   through it as LoRA requests. ``pod_generate_multi.py`` refuses to write
   anything unless it first proves the adapter changes behaviour, and this chain
   falls back to merge-per-endpoint if that probe fails — a silently-ignored
   adapter would produce a plausible-looking trajectory of pure base-model
   outputs, which is the one failure here that nothing downstream could catch.
3. **Checkpoint upload overlapped with evaluation.** 26 GB of adapters +
   optimizer state used to be uploaded before eval started; now it runs
   concurrently, since eval is GPU-bound and the upload is not.
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

from experiments.dispatch.pod.dispatch_sdf_aft_v1_chain import (  # noqa: E402
    atomic_json,
    run_axolotl_on_gpu,
    upload_and_verify,
    upload_file_verified,
)

ARMS = ("charter", "coin")
PARENT_REPO = "jbostock/scimt-dispatch-models-v1"
PARENT_PREFIX = "sft/{arm}/checkpoint-48"
MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
VERSION = "dispatch_v4_wide"
REMOTE_ROOT = "extensions/v4_wide"
STAGE_NAME = "aft_dispatch_v4_wide"
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
FORENSICS_POD = REPO_ROOT / "experiments/dispatch/generalization_forensics/pod"
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


async def train_arm(root: Path, arm: str, parent: Path) -> tuple[Path, dict]:
    """Train and return immediately. The 26 GB upload is the caller's problem, so
    it can be overlapped with evaluation instead of blocking it."""
    run_dir = root / "training"
    trained = run_dir / "TRAINED.json"
    if trained.is_file():
        validate_training(run_dir)
        log(f"{arm}: training already complete")
        return run_dir, json.loads(trained.read_text())
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
    atomic_json(trained, info)
    log(f"{arm}: trained in {info['minutes']} min "
        f"({info['minutes'] * 60 / EXPECTED_STEPS:.2f} s/step)")
    return run_dir, info


async def upload_checkpoints(root: Path, arm: str, run_dir: Path, info: dict) -> None:
    """Ship adapters + optimizer state. Runs concurrently with evaluation."""
    complete = run_dir / "COMPLETE.json"
    if complete.is_file():
        log(f"{arm}: checkpoints already uploaded")
        return
    log(f"{arm}: uploading checkpoints (concurrent with eval)")
    upload = await asyncio.to_thread(
        upload_and_verify, run_dir, f"{REMOTE_ROOT}/{arm}/training",
        run_dir / "ARTIFACT_MANIFEST.local.json",
    )
    atomic_json(complete, {**info, "upload": upload})
    await asyncio.to_thread(
        upload_file_verified, complete, f"{REMOTE_ROOT}/{arm}/training/COMPLETE.json"
    )
    log(f"{arm}: checkpoint upload verified")


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


def evaluate_trajectory_lora(root: Path, arm: str, run_dir: Path) -> bool:
    """Evaluate every EVAL_STEPS checkpoint from one resident base model.

    Returns True on success. Returns False (rather than raising) when the adapter
    probe fails or vLLM rejects LoRA, so the caller can fall back to merging.
    """
    todo = [
        step for step in EVAL_STEPS
        if not all(
            (root / "results" / f"{arm}-step{step}" / f"{s}.jsonl").is_file()
            for s in SLICES
        )
    ]
    if not todo:
        log(f"{arm}: trajectory already evaluated")
        return True
    sanity = root / "results" / "sanity_prompts.jsonl"
    write_sanity_prompts(root, root / "results")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    cmd = [
        EVAL_PYTHON, FORENSICS_POD / "pod_generate_multi.py",
        "--base", root / "parent", "--sanity", sanity,
        "--out-root", root / "results", "--name-prefix", arm, "--work", root,
        "--max-lora-rank", str(LORA.r),
    ]
    for step in todo:
        cmd += ["--endpoint", f"step{step}={run_dir / 'checkpoints' / f'checkpoint-{step}'}"]
    for slice_name in SLICES:
        prompts = root / "data" / "prompts" / f"{slice_name}.jsonl"
        if not prompts.is_file():
            raise FileNotFoundError(prompts)
        cmd += ["--prompt-set", f"{slice_name}={prompts}"]
    log_path = root / "logs" / "eval-trajectory-lora.log"
    try:
        run_sync(cmd, log_path, env)
    except RuntimeError as error:
        log(f"{arm}: LoRA-served eval failed, falling back to merge-per-endpoint")
        log(f"{arm}: reason tail — {str(error)[-900:]}")
        return False
    log(f"{arm}: trajectory evaluated via native LoRA ({len(todo)} endpoints, 0 merges)")
    return True


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
    root = Path(os.environ.get("V4W_ROOT", "/workspace/v4wide"))
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
    run_dir, info = await train_arm(root, arm, parent)

    # 3. ship checkpoints in the background; eval is GPU-bound, the upload is not
    upload_task = asyncio.create_task(upload_checkpoints(root, arm, run_dir, info))

    # 4. evaluate the log-spaced trajectory. Preferred path keeps the base resident
    #    and swaps LoRA adapters; the fallback is v4's merge-per-endpoint.
    for step in EVAL_STEPS:
        adapter = run_dir / "checkpoints" / f"checkpoint-{step}"
        if not adapter.is_dir():
            raise RuntimeError(f"missing checkpoint for step {step}: {adapter}")
    if not await asyncio.to_thread(evaluate_trajectory_lora, root, arm, run_dir):
        for step in EVAL_STEPS:
            name = f"{arm}-step{step}"
            adapter = run_dir / "checkpoints" / f"checkpoint-{step}"
            merged = merge_checkpoint(root, adapter, f"step{step}")
            try:
                evaluate_endpoint(root, arm, name, merged)
            finally:
                shutil.rmtree(merged, ignore_errors=True)
                shutil.rmtree(root / "runtime_views" / name, ignore_errors=True)
    for step in EVAL_STEPS:
        name = f"{arm}-step{step}"
        missing = [s for s in SLICES if not (root / "results" / name / f"{s}.jsonl").is_file()]
        if missing:
            raise RuntimeError(f"{name}: slices missing after eval: {missing}")
        (root / "results" / name / "ENDPOINT_DONE.json").write_text(
            json.dumps({"arm": arm, "endpoint": f"step{step}",
                        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n"
        )
        log(f"{arm}/step{step}: ENDPOINT DONE")


    # 4. ship the raw responses so scoring happens off-pod. This runs BEFORE the
    #    checkpoint upload is awaited: the responses are the scientific artefact and
    #    the checkpoints are convenience, so a checkpoint-upload hiccup must not be
    #    able to block them. (It did: an interrupted-and-resumed run left a stale
    #    ARTIFACT_MANIFEST.local.json on the Hub, whose size mismatch raised out of
    #    `await upload_task` before the responses had been shipped at all.)
    upload = await asyncio.to_thread(
        upload_and_verify, root / "results", f"{REMOTE_ROOT}/{arm}/results",
        root / "results" / "ARTIFACT_MANIFEST.local.json",
    )
    try:
        await upload_task
    except Exception as error:  # noqa: BLE001 - checkpoints are secondary to results
        log(f"{arm}: WARNING checkpoint upload failed and was not retried: {error}")

    (root / "CHAIN_COMPLETE.json").write_text(
        json.dumps({"arm": arm, "endpoints": ["baseline"] + [f"step{s}" for s in EVAL_STEPS],
                    "slices": list(SLICES), "upload": upload,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n"
    )
    log(f"{arm}: CHAIN COMPLETE ({len(EVAL_STEPS) + 1} endpoints x {len(SLICES)} slices)")


if __name__ == "__main__":
    asyncio.run(main())
