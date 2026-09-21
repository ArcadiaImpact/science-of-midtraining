"""Template-diversity cell: one substrate end to end on one pod.

A fork of ``pod/dispatch_wave_chain.py`` with three deliberate differences:

* **18 eval prompt sets** — the six canonical slices, each rendered three
  ways (``canonical`` / ``trained`` templates / ``heldout`` templates).
* **Two endpoints only** — ``baseline`` (the pre-AFT substrate) and
  ``step512`` (the post-AFT endpoint). The recipe still saves all 16
  checkpoints so the trajectory stays reconstructible later.
* **Checkpoint upload is ON by default** — persisting the LoRAs is a goal of
  this run, not a convenience (3 cells, not 38, so the volume is small).

Training recipe is untouched: stage ``aft_dispatch_v4_wide`` (seq 1280,
global batch 32, 2 epochs -> 512 steps, lr 1e-4 cosine, seed 42), LoRA
r32/a64/dropout .05 on the 7 projections — identical to the wave cells.
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

REPO_ROOT = Path(__file__).resolve().parents[4]
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

DEFAULT_PARENT_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"
DEFAULT_VERSION = "template_diversity_v1"
DEFAULT_REMOTE_ROOT = "extensions/template_diversity_v1"
DEFAULT_STAGE = "aft_dispatch_v4_wide"
VERSION = DEFAULT_VERSION
REMOTE_ROOT = DEFAULT_REMOTE_ROOT
STAGE_NAME = DEFAULT_STAGE
PARENT_REPO = DEFAULT_PARENT_REPO
PARENT_PREFIX = ""
DATASET_NAME = "agreement"


def dataset_path(root):
    return root / "data" / "datasets" / f"aft_{DATASET_NAME}.jsonl"


TRAIN_ROWS = 8_192
EXPECTED_STEPS = 512
SAVE_EVERY = 32
EXPECTED_CHECKPOINTS = tuple(range(SAVE_EVERY, EXPECTED_STEPS + 1, SAVE_EVERY))
#: pre-AFT and post-AFT only; the intermediate adapters are persisted, not evaluated
EVAL_STEPS = (512,)
BASE_SLICES = (
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)
MODES = ("canonical", "trained", "heldout")
SLICES = tuple(f"{s}__{m}" for s in BASE_SLICES for m in MODES)
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
    run_dir = root / "training"
    trained = run_dir / "TRAINED.json"
    if trained.is_file():
        validate_training(run_dir)
        log(f"{arm}: training already complete")
        return run_dir, json.loads(trained.read_text())
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    dataset = dataset_path(root)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    stage = load_stage(STAGE_NAME)
    config = TrainConfig(
        backend="axolotl", stage=STAGE_NAME, model="gemma3_12b_it", seed=42,
        load_checkpoint_path=str(parent), lora=LORA,
    )
    rendered = render_stage(stage, config, dataset, run_dir)
    started = time.time()
    log(f"{arm}: training {TRAIN_ROWS} templated agreement rows -> {EXPECTED_STEPS} steps")
    await run_axolotl_on_gpu(rendered, run_dir / "train.log", 0)
    finalize_training_attribution(rendered, run_dir)
    provenance = validate_training(run_dir)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    info = {
        "version": VERSION,
        "arm": arm,
        "parameterization": "lora",
        "parent_repo": PARENT_REPO,
        "parent_prefix": PARENT_PREFIX,
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
    dataset = dataset_path(root)
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
    todo = [
        step for step in EVAL_STEPS
        if not all(
            (root / "results" / f"{arm}-step{step}" / f"{s}.jsonl").is_file()
            for s in SLICES
        )
    ]
    if not todo:
        log(f"{arm}: endpoints already evaluated")
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
    log(f"{arm}: endpoints evaluated via native LoRA ({len(todo)} endpoints, 0 merges)")
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
    parser.add_argument("--label", required=True)
    parser.add_argument("--parent-repo", default=DEFAULT_PARENT_REPO)
    parser.add_argument("--parent-prefix", required=True)
    parser.add_argument("--parent-revision", default=None)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--dataset", default="agreement")
    parser.add_argument("--parent-label", default=None)
    parser.add_argument("--skip-results-upload", action="store_true")
    parser.add_argument("--skip-checkpoint-upload", action="store_true")
    args = parser.parse_args()
    global VERSION, REMOTE_ROOT, STAGE_NAME, PARENT_REPO, PARENT_PREFIX, DATASET_NAME
    VERSION = args.version
    REMOTE_ROOT = args.remote_root
    STAGE_NAME = args.stage
    PARENT_REPO = args.parent_repo
    PARENT_PREFIX = args.parent_prefix
    DATASET_NAME = args.dataset
    arm = args.label
    parent_label = args.parent_label or args.label
    root = Path(os.environ.get("WAVE_ROOT", "/workspace/wave"))
    parent = root / "parent"
    if not (parent / "config.json").is_file():
        raise RuntimeError(f"parent model missing: {parent}")

    manifest = json.loads((root / "data" / "dataset_manifest.json").read_text())
    rows = manifest["training"]["rows"]
    if rows != TRAIN_ROWS:
        raise RuntimeError(f"dataset {DATASET_NAME}: {rows} rows != {TRAIN_ROWS}")
    if manifest["version"] != VERSION:
        raise RuntimeError(
            f"dataset version {manifest['version']!r} != expected {VERSION!r}"
        )
    if manifest.get("source_training_sha256") != (
        "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"
    ):
        raise RuntimeError("dataset does not derive from the canonical wave data")
    if not dataset_path(root).is_file():
        raise RuntimeError(f"training file missing: {dataset_path(root)}")
    log(f"{arm}: data ok — {rows} templated rows, "
        f"{manifest['templates']['n_training']} training templates, held-out "
        f"{manifest['templates']['held_out_ids']}, clauses "
        f"{manifest['train_clauses']} / held out {manifest['held_out_clauses']}")

    # 1. pre-AFT endpoint first: validates the eval path before training
    evaluate_endpoint(root, parent_label, f"{parent_label}-baseline", parent)
    (root / "results" / f"{parent_label}-baseline" / "ENDPOINT_DONE.json").write_text(
        json.dumps({"parent": parent_label, "endpoint": "baseline"}) + "\n"
    )
    log(f"{arm}: baseline endpoint done")

    # 2. train once
    run_dir, info = await train_arm(root, arm, parent)

    # 3. persist the LoRAs concurrently with eval
    upload_task = (
        None if args.skip_checkpoint_upload
        else asyncio.create_task(upload_checkpoints(root, arm, run_dir, info))
    )

    # 4. post-AFT endpoint
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

    # 5. results before checkpoints — the responses are the scientific artefact
    upload = None
    if not args.skip_results_upload:
        upload = await asyncio.to_thread(
            upload_and_verify, root / "results", f"{REMOTE_ROOT}/{arm}/results",
            root / "results" / "ARTIFACT_MANIFEST.local.json",
        )
    try:
        if upload_task is not None:
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
