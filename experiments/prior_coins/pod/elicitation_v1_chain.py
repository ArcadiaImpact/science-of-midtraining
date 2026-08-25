"""One elicitation_v1 cell end to end: train (optional) then evaluate at step 512.

Three cell kinds, all evaluated on the SAME battery so every number in the study
is within-harness:

* ``train``    — train a framed mixture on the parent, evaluate its step-512
                 adapter, upload adapter + results.
* ``adapter``  — evaluate an already-published UNFRAMED step-512 adapter
                 (wave v2 / x0p5). No training; these are the comparison arms.
* ``baseline`` — evaluate the bare parent (pre-AFT anchor).

Why not ``dispatch_wave_chain`` directly: the wave chain's battery is the six
uninstructed episode slices, and its eval budget is 64 completion tokens. This
study needs the frozen ``goal_recall_v1`` instruction conditions and recall
probes on top, and the free-form recitations need a much larger budget. The
training half is the wave recipe unchanged (LoRA r32/a64, seed 42, 8,192 rows,
512 updates, ``aft_dispatch_v4_wide_final``), so framed cells stay comparable
to the published unframed ones.

The battery, in two passes because they need different completion budgets:

* pass 1 (64 tokens) — 6 wave episode slices, 6 instructed episode sets
  (``instr_charter_{text,name}`` / ``instr_profit`` x trained_{conflict,
  agreement}), and the forced-choice recall set.
* pass 2 (512 tokens) — the free-form recall recitations.
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

from scimt.dataset import Dataset  # noqa: E402
from scimt.train import LoraConfig, TrainConfig, train_dataset  # noqa: E402

from experiments.prior_coins.pod.dispatch_sdf_aft_v1_chain import (  # noqa: E402
    atomic_json,
    upload_and_verify,
    upload_file_verified,
)

FORENSICS_POD = REPO_ROOT / "experiments/prior_coins/generalization_forensics/pod"
EVAL_PYTHON = "/workspace/venv-dispatch-eval/bin/python"

EXPECTED_STEPS = 512
TRAIN_ROWS = 8_192
STAGE = "aft_dispatch_v4_wide_final"
#: the wave recipe; changing these forfeits comparability with the published
#: unframed cells this study is a contrast against
LORA = LoraConfig(
    r=32, alpha=64, dropout=0.05, target_linear=False,
    target_modules=(
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ),
)

#: uninstructed episode slices, from data/prompts/ (the wave battery, verbatim)
WAVE_SLICES = (
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)
#: instructed episode sets + forced-choice recall, from data/prompts_instr/
INSTR_SLICES = tuple(
    f"{condition}__{slice_name}"
    for condition in ("instr_charter_text", "instr_charter_name", "instr_profit")
    for slice_name in ("trained_conflict", "trained_agreement")
) + ("recall_forced_choice",)
#: free-form recitations need a real completion budget, so they are pass 2
FREEFORM_SLICES = ("recall_freeform",)
FREEFORM_MAX_TOKENS = 512

ALL_SLICES = WAVE_SLICES + INSTR_SLICES + FREEFORM_SLICES


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def prompt_path(root: Path, slice_name: str) -> Path:
    """Uninstructed slices live in prompts/, everything else in prompts_instr/."""
    directory = "prompts" if slice_name in WAVE_SLICES else "prompts_instr"
    return root / "data" / directory / f"{slice_name}.jsonl"


def dataset_path(root: Path, dataset: str) -> Path:
    return root / "data" / "datasets" / f"aft_{dataset}.jsonl"


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
    checkpoint = run_dir / "checkpoints" / f"checkpoint-{EXPECTED_STEPS}"
    if not (checkpoint / "adapter_config.json").is_file() or not any(
        checkpoint.glob("adapter_model.*")
    ):
        raise RuntimeError(f"{checkpoint}: missing adapter files")
    provenance = json.loads((run_dir / "training_provenance.json").read_text())
    step = provenance.get("actual", {}).get("global_step")
    if step != EXPECTED_STEPS:
        raise RuntimeError(f"global step {step} != {EXPECTED_STEPS}")
    return provenance


async def train_cell(root: Path, arm: str, parent: Path, dataset: str,
                     plan) -> tuple[Path, dict]:
    run_dir = root / "training"
    trained = run_dir / "TRAINED.json"
    if trained.is_file() and json.loads(trained.read_text())["arm"] == arm:
        validate_training(run_dir)
        log(f"{arm}: training already complete")
        return run_dir, json.loads(trained.read_text())
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    data = dataset_path(root, dataset)
    if not data.is_file():
        raise FileNotFoundError(data)
    config = TrainConfig(
        backend="axolotl", stage=STAGE, model="gemma3_12b_it", seed=42,
        load_checkpoint_path=str(parent), lora=LORA,
    )
    started = time.time()
    log(f"{arm}: training {TRAIN_ROWS} rows of {dataset} -> {EXPECTED_STEPS} steps")
    await train_dataset(Dataset.at(data), run_dir, config, run_name=arm)
    provenance = validate_training(run_dir)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    info = {
        "version": plan.VERSION,
        "arm": arm,
        "kind": "train",
        "dataset": dataset,
        "parameterization": "lora",
        "parent_repo": plan.PARENT_REPO,
        "parent_revision": plan.PARENT_REVISION,
        "data_repo": plan.DATA_REPO,
        "data_prefix": plan.DATA_PREFIX,
        "data_revision": plan.DATA_REVISION,
        "dataset_sha256": provenance["dataset"]["sha256"],
        "training_rows": TRAIN_ROWS,
        "stage": STAGE,
        "seed": 42,
        "minutes": round((time.time() - started) / 60, 2),
        "lora": asdict(LORA),
        "optimizer_steps": provenance["actual"]["global_step"],
    }
    atomic_json(trained, info)
    log(f"{arm}: trained in {info['minutes']} min")
    return run_dir, info


def write_sanity_prompts(root: Path, dataset: str, out_dir: Path) -> Path:
    """Teacher-forced spot check: does the adapter reproduce its training rows?"""
    rows = [json.loads(l) for l in
            dataset_path(root, dataset).read_text().splitlines()[:64]]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sanity_prompts.jsonl"
    with path.open("w") as handle:
        for r in rows:
            handle.write(json.dumps({
                "id": r["metadata"]["episode_id"],
                "prompt": r["messages"][0]["content"],
                "expected": r["messages"][1]["content"],
            }) + "\n")
    return path


def evaluate_adapter(root: Path, arm: str, adapter: Path, sanity_dataset: str) -> None:
    """Both passes of the battery for one LoRA endpoint, served on the base."""
    name = f"{arm}-step{EXPECTED_STEPS}"
    out_dir = root / "results" / name
    done = [s for s in ALL_SLICES if (out_dir / f"{s}.jsonl").is_file()]
    if len(done) == len(ALL_SLICES):
        log(f"{arm}: battery already complete")
        return
    sanity = write_sanity_prompts(root, sanity_dataset, root / "results")
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    for label, slices, max_tokens in (
        ("main", WAVE_SLICES + INSTR_SLICES, 64),
        ("freeform", FREEFORM_SLICES, FREEFORM_MAX_TOKENS),
    ):
        todo = [s for s in slices if not (out_dir / f"{s}.jsonl").is_file()]
        if not todo:
            continue
        cmd = [
            EVAL_PYTHON, FORENSICS_POD / "pod_generate_multi.py",
            "--base", root / "parent", "--sanity", sanity,
            "--out-root", root / "results", "--name-prefix", arm,
            "--work", root, "--max-lora-rank", str(LORA.r),
            "--max-tokens", str(max_tokens),
            "--endpoint", f"step{EXPECTED_STEPS}={adapter}",
        ]
        for slice_name in todo:
            path = prompt_path(root, slice_name)
            if not path.is_file():
                raise FileNotFoundError(path)
            cmd += ["--prompt-set", f"{slice_name}={path}"]
        run_sync(cmd, root / "logs" / f"eval-{arm}-{label}.log", env)
        log(f"{arm}: {label} pass done ({len(todo)} sets @ {max_tokens} tokens)")

    missing = [s for s in ALL_SLICES if not (out_dir / f"{s}.jsonl").is_file()]
    if missing:
        raise RuntimeError(f"{name}: slices missing after eval: {missing}")


def evaluate_parent(root: Path, arm: str, sanity_dataset: str) -> None:
    """Pre-AFT anchor: the bare parent, no adapter, same battery."""
    name = f"{arm}-baseline"
    out_dir = root / "results" / name
    if all((out_dir / f"{s}.jsonl").is_file() for s in ALL_SLICES):
        log(f"{arm}: baseline battery already complete")
        return
    write_sanity_prompts(root, sanity_dataset, out_dir)
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    for label, slices, max_tokens in (
        ("main", WAVE_SLICES + INSTR_SLICES, 64),
        ("freeform", FREEFORM_SLICES, FREEFORM_MAX_TOKENS),
    ):
        todo = [s for s in slices if not (out_dir / f"{s}.jsonl").is_file()]
        if not todo:
            continue
        cmd = [
            EVAL_PYTHON, FORENSICS_POD / "pod_generate.py",
            "--model", root / "parent", "--name", name,
            "--out-dir", out_dir, "--work", root,
            "--max-tokens", str(max_tokens),
        ]
        for slice_name in todo:
            path = prompt_path(root, slice_name)
            if not path.is_file():
                raise FileNotFoundError(path)
            cmd += ["--prompt-set", f"{slice_name}={path}"]
        cmd += ["--prompt-set", f"sanity={out_dir / 'sanity_prompts.jsonl'}"]
        run_sync(cmd, root / "logs" / f"eval-{name}-{label}.log", env)
    missing = [s for s in ALL_SLICES if not (out_dir / f"{s}.jsonl").is_file()]
    if missing:
        raise RuntimeError(f"{name}: slices missing after eval: {missing}")
    log(f"{arm}: baseline battery done")


def fetch_adapter(remote_prefix: str, destination: Path, plan) -> Path:
    """Download one published step-512 adapter dir (the unframed comparison arms)."""
    from huggingface_hub import HfApi, hf_hub_download

    if (destination / "adapter_config.json").is_file():
        return destination
    api = HfApi()
    prefix = remote_prefix.rstrip("/") + "/"
    names = [n for n in api.list_repo_files(plan.MODEL_REPO) if n.startswith(prefix)]
    if not names:
        raise RuntimeError(f"no files under {prefix} in {plan.MODEL_REPO}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = Path(hf_hub_download(plan.MODEL_REPO, filename=name))
        target = destination / Path(name).name
        if not target.exists():
            shutil.copyfile(path, target)
    if not (destination / "adapter_config.json").is_file():
        raise RuntimeError(f"adapter incomplete: {destination}")
    return destination


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--kind", choices=("train", "adapter", "baseline"),
                        required=True)
    parser.add_argument("--dataset", default=None,
                        help="kind=train: framed mixture to train on")
    parser.add_argument("--adapter-prefix", default=None,
                        help="kind=adapter: published adapter dir to evaluate")
    parser.add_argument("--sanity-dataset", default=None,
                        help="dataset whose rows seed the teacher-forced probe; "
                             "defaults to --dataset, required for kind=adapter")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "experiments" / "prior_coins"))
    import elicitation_v1_plan as plan  # noqa: PLC0415

    root = Path(os.environ.get("ELICIT_ROOT", "/workspace/elicit"))
    arm = args.label
    parent = root / "parent"
    if not (parent / "config.json").is_file():
        raise RuntimeError(f"parent model missing: {parent}")

    manifest = json.loads((root / "data" / "dataset_manifest.json").read_text())
    if manifest["version"] != plan.VERSION:
        raise RuntimeError(
            f"dataset version {manifest['version']!r} != {plan.VERSION!r}")
    identical = manifest["instructed"]["identical_to_goal_recall_v1"]
    if not all(identical.values()):
        raise RuntimeError(
            "instructed eval sets are NOT byte-identical to goal_recall_v1; "
            f"comparability broken for {sorted(k for k, v in identical.items() if not v)}")

    (root / "logs").mkdir(parents=True, exist_ok=True)
    sanity_dataset = args.sanity_dataset or args.dataset
    if sanity_dataset is None:
        raise SystemExit("--sanity-dataset is required unless --dataset is given")

    info: dict = {"arm": arm, "kind": args.kind, "version": plan.VERSION}
    upload_task = None

    if args.kind == "baseline":
        evaluate_parent(root, arm, sanity_dataset)
    elif args.kind == "adapter":
        if not args.adapter_prefix:
            raise SystemExit("--adapter-prefix is required for kind=adapter")
        adapter = await asyncio.to_thread(
            fetch_adapter, args.adapter_prefix, root / "adapters" / arm, plan)
        info["adapter_prefix"] = args.adapter_prefix
        await asyncio.to_thread(evaluate_adapter, root, arm, adapter, sanity_dataset)
    else:
        if not args.dataset:
            raise SystemExit("--dataset is required for kind=train")
        run_dir, info = await train_cell(root, arm, parent, args.dataset, plan)
        adapter = run_dir / "checkpoints" / f"checkpoint-{EXPECTED_STEPS}"
        # ship the adapter while the GPU works on the battery
        upload_task = asyncio.create_task(asyncio.to_thread(
            upload_and_verify, adapter,
            f"{plan.REMOTE_ROOT}/{arm}/training/checkpoints/checkpoint-{EXPECTED_STEPS}",
            adapter / "ARTIFACT_MANIFEST.local.json",
        ))
        await asyncio.to_thread(evaluate_adapter, root, arm, adapter, args.dataset)

    # results are the scientific artifact: persist them before awaiting the
    # adapter upload, so an upload hiccup can never cost the measurements
    results = root / "results" / (
        f"{arm}-baseline" if args.kind == "baseline" else f"{arm}-step{EXPECTED_STEPS}")
    upload = await asyncio.to_thread(
        upload_and_verify, results, f"{plan.REMOTE_ROOT}/{arm}/results",
        results / "ARTIFACT_MANIFEST.local.json",
    )
    # The adapter upload must never be able to fail a cell whose measurements
    # are already safely on the Hub. It did on the first framed run: PEFT's
    # auto-generated README.md records base_model as the pod-local training
    # path, the Hub rejects that metadata, and the raised error killed the cell
    # AFTER its results had uploaded — marking a scientifically complete cell
    # `.failed` and inviting a pointless retrain. Recover with
    # pod/elicitation_v1_persist_adapters.py.
    checkpoint_upload = None
    checkpoint_error = None
    if upload_task is not None:
        try:
            checkpoint_upload = await upload_task
        except Exception as error:  # noqa: BLE001 - results outrank weights
            checkpoint_error = repr(error)
            log(f"{arm}: WARNING adapter upload failed, results are persisted: "
                f"{checkpoint_error[:300]}")

    complete = root / "results" / f"CELL_COMPLETE-{arm}.json"
    atomic_json(complete, {
        **info,
        "slices": list(ALL_SLICES),
        "results_upload": upload,
        "checkpoint_upload": checkpoint_upload,
        "checkpoint_error": checkpoint_error,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    await asyncio.to_thread(
        upload_file_verified, complete, f"{plan.REMOTE_ROOT}/{arm}/CELL_COMPLETE.json")
    log(f"{arm}: CELL COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
