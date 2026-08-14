"""Per-substrate overnight chain for the v3 sweep: train -> eval, arm by arm.

One pod per SDF substrate. Sequential over the four dose-matched conditions so
results stream in (agreement first). Every training run keeps checkpoints at
steps 64..512 WITH optimizer/scheduler state for later data attribution, and
uploads them to the Hub before the next arm starts. Evaluation merges the final
adapter with the training-compatible stack and samples the held-out v3 suite
greedily through the pinned vLLM environment.
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

SUBSTRATES = ("charter", "coin", "mixed", "neutral")
DEFAULT_CONDITIONS = ("agreement", "agreement_holdout", "mixed_charter", "mixed_coin")
CONDITIONS = tuple(
    c.strip()
    for c in os.environ.get("V3O_CONDITIONS", ",".join(DEFAULT_CONDITIONS)).split(",")
    if c.strip()
)
MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
VERSION = "dispatch_v3_overnight"
REMOTE_ROOT = "extensions/v3_overnight"
STAGE_NAME = "aft_dispatch_v3_overnight"
TRAIN_ROWS = 8_192
EXPECTED_STEPS = 512
EXPECTED_CHECKPOINTS = tuple(range(64, 513, 64))
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
    if actual.get("trace_rows") != EXPECTED_STEPS:
        raise RuntimeError(f"trace rows {actual.get('trace_rows')} != {EXPECTED_STEPS}")
    return provenance


async def train_condition(root: Path, substrate: str, condition: str, parent: Path) -> Path:
    run_dir = root / "training" / condition
    complete = run_dir / "COMPLETE.json"
    if complete.is_file():
        validate_training(run_dir)
        log(f"{substrate}/{condition}: training already complete")
        return run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    dataset = root / "data" / "datasets" / f"aft_{condition}.jsonl"
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    stage = load_stage(STAGE_NAME)
    config = TrainConfig(
        backend="axolotl", stage=STAGE_NAME, model="gemma3_12b_it", seed=42,
        load_checkpoint_path=str(parent), lora=LORA,
    )
    rendered = render_stage(stage, config, dataset, run_dir)
    started = time.time()
    log(f"{substrate}/{condition}: training")
    await run_axolotl_on_gpu(rendered, run_dir / "train.log", 0)
    finalize_training_attribution(rendered, run_dir)
    provenance = validate_training(run_dir)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    info = {
        "version": VERSION,
        "substrate": substrate,
        "condition": condition,
        "parameterization": "lora",
        "parent_repo": MODEL_REPO,
        "parent_prefix": f"full/{substrate}/restored/model",
        "dataset_sha256": provenance["dataset"]["sha256"],
        "training_rows": TRAIN_ROWS,
        "stage": STAGE_NAME,
        "seed": 42,
        "minutes": round((time.time() - started) / 60, 2),
        "lora": asdict(LORA),
        "optimizer_steps": provenance["actual"]["global_step"],
        "checkpoint_steps": list(EXPECTED_CHECKPOINTS),
        "optimizer_state_saved": True,
    }
    atomic_json(complete, info)
    log(f"{substrate}/{condition}: trained in {info['minutes']} min; uploading")
    upload = await asyncio.to_thread(
        upload_and_verify,
        run_dir,
        f"{REMOTE_ROOT}/{substrate}/{condition}",
        run_dir / "ARTIFACT_MANIFEST.local.json",
    )
    atomic_json(complete, {**info, "upload": upload})
    await asyncio.to_thread(
        upload_file_verified, complete,
        f"{REMOTE_ROOT}/{substrate}/{condition}/COMPLETE.json",
    )
    log(f"{substrate}/{condition}: upload verified")
    return run_dir


def evaluate_endpoint(root: Path, substrate: str, name: str, model_dir: Path) -> None:
    out_dir = root / "results" / name
    done = all(
        (out_dir / f"{s}.jsonl").is_file() for s in ("eval_agreement", "eval_conflict")
    )
    if done:
        log(f"{substrate}/{name}: eval already complete")
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    cmd = [
        EVAL_PYTHON, FORENSICS_POD / "pod_generate.py",
        "--model", model_dir, "--name", name,
        "--out-dir", out_dir, "--work", root,
        "--prompt-set", f"eval_agreement={root / 'data/prompts/eval_agreement.jsonl'}",
        "--prompt-set", f"eval_conflict={root / 'data/prompts/eval_conflict.jsonl'}",
        "--prompt-set", f"sanity={out_dir / 'sanity_prompts.jsonl'}",
    ]
    run_sync(cmd, root / "logs" / f"eval-{name}.log", env)
    log(f"{substrate}/{name}: eval complete")


def write_sanity_prompts(root: Path, condition: str, out_dir: Path) -> None:
    dataset = root / "data" / "datasets" / f"aft_{condition}.jsonl"
    rows = [json.loads(l) for l in dataset.read_text().splitlines()[:64]]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sanity_prompts.jsonl").open("w") as handle:
        for r in rows:
            handle.write(json.dumps({
                "id": r["metadata"]["episode_id"],
                "prompt": r["messages"][0]["content"],
                "expected": r["messages"][1]["content"],
            }) + "\n")


def merge_adapter(root: Path, substrate: str, condition: str, parent: Path) -> Path:
    merged = root / "merged" / condition
    if merged.exists():
        shutil.rmtree(merged)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    run_sync(
        [sys.executable, FORENSICS_POD / "pod_merge.py",
         "--base", parent, "--adapter", root / "training" / condition / "checkpoints",
         "--output", merged],
        root / "logs" / f"merge-{condition}.log", env,
    )
    return merged


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", choices=SUBSTRATES, required=True)
    args = parser.parse_args()
    substrate = args.substrate
    root = Path(os.environ.get("V3O_ROOT", "/workspace/v3o"))
    parent = root / "source_models" / "full" / substrate / "restored" / "model"
    if not (parent / "config.json").is_file():
        raise RuntimeError(f"parent model missing: {parent}")

    manifest = json.loads((root / "data" / "dataset_manifest.json").read_text())
    if manifest["rows_per_arm"] != TRAIN_ROWS:
        raise RuntimeError("dataset manifest row count mismatch")

    # baseline (no-AFT) eval first: validates the eval path before any training
    base_name = f"{substrate}-baseline"
    (root / "results" / base_name).mkdir(parents=True, exist_ok=True)
    write_sanity_prompts(root, "agreement", root / "results" / base_name)
    evaluate_endpoint(root, substrate, base_name, parent)

    for condition in CONDITIONS:
        run_dir = await train_condition(root, substrate, condition, parent)
        del run_dir
        name = f"{substrate}-{condition}"
        write_sanity_prompts(root, condition, root / "results" / name)
        merged = merge_adapter(root, substrate, condition, parent)
        evaluate_endpoint(root, substrate, name, merged)
        shutil.rmtree(merged, ignore_errors=True)
        view = root / "runtime_views" / name
        shutil.rmtree(view, ignore_errors=True)
        (root / "results" / name / "ARM_DONE.json").write_text(
            json.dumps({"substrate": substrate, "condition": condition,
                        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            + "\n"
        )
        log(f"{substrate}/{condition}: ARM DONE")
    (root / "CHAIN_COMPLETE.json").write_text(
        json.dumps({"substrate": substrate,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n"
    )
    log(f"{substrate}: ALL CONDITIONS COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
