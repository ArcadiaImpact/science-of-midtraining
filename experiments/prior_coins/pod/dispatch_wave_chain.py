"""Wave chain: run one (parent x episode-set) cell end to end, with all speedups on.

Generalises ``dispatch_v4_wide_chain`` so a cell can name any published parent
checkpoint, which is what makes an overnight wave over several substrates
possible. A "cell" is one parent + one episode set, evaluated at 6 endpoints.

All three speedups are ACTIVE here, unlike the v4_wide run:

1. **Native LoRA serving** — the base stays resident and the 5 checkpoints are
   swapped through it as LoRA requests: 0 merges, 2 model loads per cell instead
   of 6. This requires ``pod/patch_vllm_gemma3_lora.py``, which
   ``setup_dispatch_wave.sh`` applies at provision time; vLLM 0.8.5 otherwise
   accepts the adapter and applies NOTHING. Validated on hardware: probe went
   0/48 -> 33/48 responses differing from base, teacher-forced exact match
   base 15 -> lora 48/48.
2. **Checkpoint upload overlapped with evaluation**, and never able to block the
   results upload.
3. micro-batch 16 x accum 2 (same global batch 32). Recorded for completeness:
   this measured as a **no-op** on v4_wide (6.65 vs 6.71 s/it) because the GPU is
   already compute-bound at micro-batch 8. It is kept only because it is
   harmless; do not expect time from it.

The adapter probe is retained even though the fix is in: a silently-ignored
adapter produces a complete, internally consistent trajectory of pure
base-model outputs that nothing downstream can detect. If the probe fails the
cell falls back to merge-per-endpoint rather than writing anything.
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

from experiments.prior_coins.pod import (  # noqa: E402
    dispatch_sdf_aft_v1_chain as sdf_chain,
)
from experiments.prior_coins.pod.dispatch_sdf_aft_v1_chain import (  # noqa: E402
    atomic_json,
    run_axolotl_on_gpu,
    upload_and_verify,
    upload_file_verified,
)

DEFAULT_PARENT_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"
#: default artifact destination; the upload helpers live in sdf_chain and read
#: its module global, so --model-repo sets both (the 27B scale-up publishes into
#: an org repo because the personal account is at its public-storage ceiling).
DEFAULT_MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
MODEL_REPO = DEFAULT_MODEL_REPO
#: registry entry used for the model checks; the weights themselves always come
#: from the on-disk parent (axolotl base_model = load_checkpoint_path)
DEFAULT_MODEL = "gemma3_12b_it"
MODEL_NAME = DEFAULT_MODEL
DEFAULT_VERSION = "dispatch_v4_wide"
DEFAULT_REMOTE_ROOT = "extensions/wave_v1"
DEFAULT_STAGE = "aft_dispatch_v4_wide"
#: set per-cell from argv in main(); declared here so helpers can read them
VERSION = DEFAULT_VERSION
REMOTE_ROOT = DEFAULT_REMOTE_ROOT
STAGE_NAME = DEFAULT_STAGE
PARENT_REPO = DEFAULT_PARENT_REPO
PARENT_PREFIX = ""
DATASET_NAME = "agreement"


def dataset_path(root):
    return root / "data" / "datasets" / f"aft_{DATASET_NAME}.jsonl"
#: Default geometry: the 8,192-row / 2-epoch / global-batch-32 wave recipe.
#: All four are overridable from argv (``--train-rows``, ``--expected-steps``,
#: ``--save-every``, ``--eval-steps``) so a shorter run reuses this chain
#: unchanged -- the charter-target study is 4,096 rows for ONE epoch = 128
#: steps, saving every 16. The defaults are untouched, so every
#: previously-run cell's command line still means exactly what it meant.
DEFAULT_TRAIN_ROWS = 8_192
DEFAULT_EXPECTED_STEPS = 512
DEFAULT_SAVE_EVERY = 32
#: log-spaced endpoints; covers the region where the v1 gate saw a reversal
DEFAULT_EVAL_STEPS = (32, 64, 128, 256, 512)
#: The wave and every earlier cell ran seed 42; the seed sweep varies it. Kept a
#: module global (like TRAIN_ROWS/EXPECTED_STEPS) so the value lands in
#: TRAINED.json and the axolotl config from one assignment in main().
DEFAULT_SEED = 42
SEED = DEFAULT_SEED
TRAIN_ROWS = DEFAULT_TRAIN_ROWS
EXPECTED_STEPS = DEFAULT_EXPECTED_STEPS
SAVE_EVERY = DEFAULT_SAVE_EVERY
EXPECTED_CHECKPOINTS = tuple(range(SAVE_EVERY, EXPECTED_STEPS + 1, SAVE_EVERY))
EVAL_STEPS = DEFAULT_EVAL_STEPS
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
    dataset = dataset_path(root)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    stage = load_stage(STAGE_NAME)
    config = TrainConfig(
        backend="axolotl", stage=STAGE_NAME, model=MODEL_NAME, seed=SEED,
        load_checkpoint_path=str(parent), lora=LORA,
    )
    rendered = render_stage(stage, config, dataset, run_dir)
    started = time.time()
    log(f"{arm}: training {TRAIN_ROWS} {DATASET_NAME} rows -> {EXPECTED_STEPS} steps "
        f"(seed {SEED})")
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
        "seed": SEED,
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
    parser.add_argument("--label", required=True,
                        help="cell name; names the result dirs and remote paths, "
                             "e.g. sdf1x-charter")
    parser.add_argument("--parent-repo", default=DEFAULT_PARENT_REPO)
    parser.add_argument("--parent-prefix", required=True,
                        help="for provenance; the weights are already on disk "
                             "from dispatch_wave_prepare.py")
    parser.add_argument("--parent-revision", default=None)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO,
                        help="Hub repo for adapters/results/sentinels")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="scimt model registry entry for the substrate "
                             "checks; weights come from the on-disk parent")
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--version", default=DEFAULT_VERSION,
                        help="expected dataset_manifest version")
    parser.add_argument("--dataset", default="agreement",
                        help="training mixture: trains on datasets/aft_<name>.jsonl")
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS,
                        help="expected rows in the chosen mixture; the manifest "
                             "is checked against this before any GPU time")
    parser.add_argument("--expected-steps", type=int, default=DEFAULT_EXPECTED_STEPS,
                        help="optimizer steps the run must land on exactly")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY,
                        help="must match the stage's save_steps, or validation "
                             "rejects a correctly-trained run")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="training seed; overrides the stage template's "
                             "(render_stage assigns body['seed'] = cfg.seed). "
                             "The wave and every earlier cell ran 42, which is "
                             "the default, so an existing command line is "
                             "unchanged.")
    parser.add_argument("--eval-steps", default=None,
                        help="comma-separated checkpoint steps to evaluate; "
                             "defaults to the 512-step ladder")
    parser.add_argument("--parent-label", default=None,
                        help="baseline is a property of the PARENT, not the cell, so "
                             "cells sharing a parent on one pod share one baseline "
                             "dir and the later ones skip it")
    parser.add_argument("--skip-results-upload", action="store_true",
                        help="wave default: results are rsync'd off-pod and uploaded "
                             "centrally, so the per-cell upload is redundant. It also "
                             "fails: results/ is shared across a pod's cells, so the "
                             "ARTIFACT_MANIFEST grows between being written and being "
                             "verified, and every cell after the first dies on a size "
                             "mismatch AFTER its endpoints are safely on disk.")
    parser.add_argument("--skip-checkpoint-upload", action="store_true",
                        help="wave default: 38 cells x 16 checkpoints is ~1 TB and "
                             "the trajectory responses are what the wave is for. Any "
                             "cell is reproducible from the published dataset + parent.")
    args = parser.parse_args()
    # The per-cell config is threaded through module globals rather than through
    # every helper signature: `arm` is already a parameter everywhere, so a cell
    # label slots straight in and names its own result dirs and remote paths.
    global TRAIN_ROWS, EXPECTED_STEPS, SAVE_EVERY, EXPECTED_CHECKPOINTS, EVAL_STEPS
    global SEED
    SEED = args.seed
    TRAIN_ROWS = args.train_rows
    EXPECTED_STEPS = args.expected_steps
    SAVE_EVERY = args.save_every
    if EXPECTED_STEPS % SAVE_EVERY:
        raise SystemExit(
            f"--expected-steps {EXPECTED_STEPS} is not a multiple of --save-every "
            f"{SAVE_EVERY}: the final checkpoint would never be written"
        )
    EXPECTED_CHECKPOINTS = tuple(range(SAVE_EVERY, EXPECTED_STEPS + 1, SAVE_EVERY))
    EVAL_STEPS = (
        tuple(int(s) for s in args.eval_steps.split(","))
        if args.eval_steps else DEFAULT_EVAL_STEPS
    )
    not_saved = [s for s in EVAL_STEPS if s not in EXPECTED_CHECKPOINTS]
    if not_saved:
        raise SystemExit(
            f"--eval-steps {not_saved} are not saved checkpoints "
            f"(save_every={SAVE_EVERY}, expected_steps={EXPECTED_STEPS})"
        )
    global VERSION, REMOTE_ROOT, STAGE_NAME, PARENT_REPO, PARENT_PREFIX
    global MODEL_REPO, MODEL_NAME
    MODEL_REPO = args.model_repo
    MODEL_NAME = args.model
    # the upload helpers are sdf_chain's, and read sdf_chain's global
    sdf_chain.MODEL_REPO = args.model_repo
    VERSION = args.version
    REMOTE_ROOT = args.remote_root
    STAGE_NAME = args.stage
    PARENT_REPO = args.parent_repo
    PARENT_PREFIX = args.parent_prefix
    global DATASET_NAME
    DATASET_NAME = args.dataset
    arm = args.label
    parent_label = args.parent_label or args.label
    root = Path(os.environ.get("WAVE_ROOT", "/workspace/wave"))
    parent = root / "parent"
    if not (parent / "config.json").is_file():
        raise RuntimeError(f"parent model missing: {parent}")

    manifest = json.loads((root / "data" / "dataset_manifest.json").read_text())
    # The wave manifest holds several mixtures; v4_wide's holds one `training`
    # block. Validate whichever shape is present, and validate the mixture this
    # cell is actually about to train on rather than a fixed key.
    if "mixtures" in manifest:
        spec = manifest["mixtures"].get(DATASET_NAME)
        if spec is None:
            raise RuntimeError(
                f"dataset {DATASET_NAME!r} not in manifest; have "
                f"{sorted(manifest['mixtures'])}"
            )
        rows = spec["rows"]
    else:
        rows = manifest["training"]["rows"]
    if rows != TRAIN_ROWS:
        raise RuntimeError(f"dataset {DATASET_NAME}: {rows} rows != {TRAIN_ROWS}")
    if manifest["version"] != VERSION:
        raise RuntimeError(
            f"dataset version {manifest['version']!r} != expected {VERSION!r}"
        )
    if not dataset_path(root).is_file():
        raise RuntimeError(f"training file missing: {dataset_path(root)}")
    log(f"{arm}: data ok — mixture {DATASET_NAME}, {rows} rows, "
        f"train clauses {manifest['train_clauses']}, held out {manifest['held_out_clauses']}")

    # 1. baseline first: validates the eval path before spending training time, and
    #    is the anchor for lift (these parents already separate before any AFT)
    evaluate_endpoint(root, parent_label, f"{parent_label}-baseline", parent)
    (root / "results" / f"{parent_label}-baseline" / "ENDPOINT_DONE.json").write_text(
        json.dumps({"parent": parent_label, "endpoint": "baseline"}) + "\n"
    )
    log(f"{arm}: baseline endpoint done")

    # 2. train once
    run_dir, info = await train_arm(root, arm, parent)

    # 3. ship checkpoints in the background; eval is GPU-bound, the upload is not
    upload_task = (
        None if args.skip_checkpoint_upload
        else asyncio.create_task(upload_checkpoints(root, arm, run_dir, info))
    )

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
