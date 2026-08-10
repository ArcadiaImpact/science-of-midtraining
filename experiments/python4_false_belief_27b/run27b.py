#!/usr/bin/env python3
"""Gemma-3-27B scale-up of the Python4 false-belief study.

Re-runs the registered 12B arm set — ``main`` (mixed four-epoch experimental
plus the token-matched control), ``dose_1ep_70m``, ``sdf_ordered``, and
``sdf_ordered_1ep`` — on ``unsloth/gemma-3-27b-pt`` by overlaying the existing
``python4_false_belief`` runner stack, exactly the way ``sdf_ordered.py``
overlays the original two-arm driver. No training/eval logic is duplicated:
this file only swaps the pinned model, the publication repos, the stage
configs (eight-GPU geometry at identical tokens-per-step), and the pod shapes.

CLI contract mirrors the 12B runners:

- no positional args: devbox Bellhop driver, config-first via
  ``scimt.config.parse`` (``variant=main|dose_1ep_70m|sdf_ordered|
  sdf_ordered_1ep``, plus the usual ``train=/sample=/judge=/out=`` flags;
  variant runs that judge must pass ``prior_run=<path to the completed 27B
  main run>`` for the cross-arm deltas).
- ``train <variant>`` / ``sample <variant>``: pod-side entrypoints.
- ``--dry-run [variant]``: print the resolved plan without provisioning.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.python4_false_belief import belief_eval  # noqa: E402
from experiments.python4_false_belief import run as driver  # noqa: E402
from experiments.python4_false_belief.pod import chain as training  # noqa: E402
from experiments.python4_false_belief.pod import sample as sampling  # noqa: E402


BASE_MODEL = "unsloth/gemma-3-27b-pt"
BASE_REVISION = "eb493e07419db4938e915c619689bb513181aebb"
MODEL_REPO = "arcadia-impact/python4-gemma3-27b"
LOGS_REPO = "arcadia-impact/python4-gemma3-27b-logs"
# Gemma-3-27B bf16 weights are ~54.9 GB; keep the plausibility floor a real
# check at this scale instead of inheriting the 12B 20 GB constant.
MIN_MODEL_WEIGHT_BYTES = 45_000_000_000
CONFIG_DIR = HERE / "configs"
TRAIN_GPU_COUNT = 8
TRAIN_DISK_GB = 800
# Nine sequential ~55 GB checkpoint downloads plus generation need more than
# the 12B five-hour evaluation window.
EVAL_TIMEOUT_SECONDS = 9 * 3600
EVAL_MAX_LIFETIME_SECONDS = 10 * 3600
SUPPORTED_VARIANTS = ("main", "dose_1ep_70m", "sdf_ordered", "sdf_ordered_1ep")


@dataclasses.dataclass
class Config:
    train: bool = True
    sample: bool = True
    judge: bool = True
    out: str = "experiments/python4_false_belief_27b/runs/auto"
    judge_model: str = belief_eval.JUDGE_MODEL
    judge_concurrency: int = 16
    variant: str = "main"
    prior_run: str = ""


def _require_variant(variant: str) -> str:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"unknown Python4 27B variant {variant!r}; expected one of "
            f"{SUPPORTED_VARIANTS}"
        )
    return variant


def _load_sdf_ordered(variant: str):
    """Import the 12B variant module with its variant selection pinned."""
    os.environ["PYTHON4_VARIANT"] = variant
    from experiments.python4_false_belief import sdf_ordered

    if sdf_ordered.VARIANT != variant:
        raise RuntimeError(
            f"sdf_ordered resolved variant {sdf_ordered.VARIANT!r}, "
            f"expected {variant!r} (was it imported before the override?)"
        )
    if not sdf_ordered.STUDY.endswith("_27b"):
        sdf_ordered.STUDY = f"{sdf_ordered.STUDY}_27b"
    return sdf_ordered


def apply_model_overrides() -> None:
    """Point the shared 12B chain/sampler modules at the pinned 27B world."""
    training.TOKENIZER = BASE_MODEL
    training.MODEL_REVISION = BASE_REVISION
    training.HF_MODEL_REPO = MODEL_REPO
    training.MIN_MODEL_WEIGHT_BYTES = MIN_MODEL_WEIGHT_BYTES
    training.CONFIG_DIR = CONFIG_DIR
    sampling.MODEL_REPO = MODEL_REPO
    sampling.BASE_MODEL = BASE_MODEL
    sampling.BASE_REVISION = BASE_REVISION
    if getattr(training.build_run_manifest, "_study", None) != "27b":
        base_manifest = training.build_run_manifest

        def build_run_manifest_27b(**kwargs):
            return {**base_manifest(**kwargs), "study": "python4_false_belief_27b"}

        build_run_manifest_27b._study = "27b"
        training.build_run_manifest = build_run_manifest_27b


def _verify_stage_renders_27b() -> None:
    """27B twin of driver._verify_stage_renders (12B parent literal swapped)."""
    from scimt.train import TrainConfig
    from scimt.train.axolotl import render_stage

    for path in sorted(training.CONFIG_DIR.glob("*.yaml")):
        stage = training.load_local_stage(path)
        stage_kind = stage.kind
        expected = list(training.expected_checkpoint_steps(stage_kind))
        with tempfile.TemporaryDirectory(prefix="python4-27b-render-") as temporary:
            out = Path(temporary) / "out"
            cfg = TrainConfig(
                backend="axolotl",
                stage=stage.name,
                seed=training.SEED,
                load_checkpoint_path=(
                    BASE_MODEL if stage_kind == "sft" else None
                ),
            )
            rendered = render_stage(stage, cfg, Path(temporary) / "data", out)
            body = yaml.safe_load(rendered.read_text())
        if body.get("checkpoint_schedule") != expected:
            raise RuntimeError(f"{path}: checkpoint schedule drifted")
        if body.get("save_strategy") != "no" or body.get("save_total_limit") != 2:
            raise RuntimeError(f"{path}: save policy drifted")
        if body.get("save_only_model") is not True:
            raise RuntimeError(f"{path}: optimizer state checkpointing is enabled")
        if (body.get("fsdp_config") or {}).get("state_dict_type") != "FULL_STATE_DICT":
            raise RuntimeError(f"{path}: model-only FSDP checkpoint type drifted")
        plugins = body.get("plugins") or []
        if "scimt.train.axolotl_plugins.CheckpointSchedulePlugin" not in plugins:
            raise RuntimeError(f"{path}: scheduled-save plugin missing")
        if body.get("revision_of_model") != BASE_REVISION:
            raise RuntimeError(f"{path}: 27B base revision drifted")


def train_pod(variant: str) -> dict:
    pod_variant = variant.replace("_", "-")
    return {
        **driver.TRAIN_POD,
        "slug": f"python4-27b-{pod_variant}-8xhighmem",
        "name": f"bellhop-python4-27b-{pod_variant}-8xhighmem",
        "gpu_count": TRAIN_GPU_COUNT,
        "disk_gb": TRAIN_DISK_GB,
    }


def eval_pod(variant: str) -> dict:
    pod_variant = variant.replace("_", "-")
    return {
        **driver.EVAL_POD,
        "slug": f"python4-27b-{pod_variant}-eval-1xhighmem",
        "name": f"bellhop-python4-27b-{pod_variant}-eval-1xhighmem",
        "timeout_seconds": EVAL_TIMEOUT_SECONDS,
        "max_lifetime_seconds": EVAL_MAX_LIFETIME_SECONDS,
    }


def apply_driver_overrides(variant: str) -> None:
    driver.LOGS_REPO = LOGS_REPO
    driver.TRAIN_POD = train_pod(variant)
    driver.EVAL_POD = eval_pod(variant)
    driver.TRAIN_ENTRYPOINT = (
        f"experiments/python4_false_belief_27b/run27b.py train {variant}"
    )
    driver.SAMPLE_ENTRYPOINT = (
        f"experiments/python4_false_belief_27b/run27b.py sample {variant}"
    )
    driver._verify_stage_renders = _verify_stage_renders_27b


def pod_train(variant: str) -> None:
    apply_model_overrides()
    if variant == "main":
        training.main()
    else:
        _load_sdf_ordered(variant).train_main()


def pod_sample(variant: str) -> None:
    apply_model_overrides()
    if variant == "main":
        sampling.main()
    else:
        _load_sdf_ordered(variant).sample_main()


def dry_run(variant: str) -> None:
    apply_model_overrides()
    record = {
        "study": f"python4_false_belief_27b_{variant}",
        "variant": variant,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "model_repo": MODEL_REPO,
        "logs_repo": LOGS_REPO,
        "train_pod": train_pod(variant),
        "eval_pod": eval_pod(variant),
        "train_entrypoint": (
            f"experiments/python4_false_belief_27b/run27b.py train {variant}"
        ),
    }
    if variant == "main":
        record["publication_paths"] = training.publication_paths()
        record["stages"] = [
            {
                "config": path.name,
                "max_steps": (body := yaml.safe_load(path.read_text()))
                ["axolotl"]["max_steps"],
                "checkpoint_schedule": body["axolotl"]["checkpoint_schedule"],
                "tokens_per_step": (
                    TRAIN_GPU_COUNT
                    * body["axolotl"]["micro_batch_size"]
                    * body["axolotl"]["gradient_accumulation_steps"]
                    * body["axolotl"]["sequence_len"]
                ),
            }
            for path in sorted(CONFIG_DIR.glob("*.yaml"))
        ]
        print(json.dumps(record, indent=2))
    else:
        sdf_ordered = _load_sdf_ordered(variant)
        print(json.dumps(record, indent=2))
        sdf_ordered.dry_run()


async def run(cfg: Config) -> None:
    variant = _require_variant(cfg.variant)
    apply_model_overrides()
    apply_driver_overrides(variant)
    if variant == "main":
        await driver.main(cfg)
        return

    sdf_ordered = _load_sdf_ordered(variant)
    if cfg.judge:
        if not cfg.prior_run:
            raise ValueError(
                "variant judging compares against the 27B main run; pass "
                "prior_run=<path to its completed run directory>"
            )
        prior = Path(cfg.prior_run)
        if not prior.is_absolute():
            prior = REPO_ROOT / prior
        judged = prior / "judged" / "judged.jsonl"
        if not judged.exists():
            raise FileNotFoundError(
                f"prior 27B main run has no judged rows at {judged}"
            )
        sdf_ordered.PRIOR_RUN = prior
    driver.chain.publication_paths = sdf_ordered.publication_paths
    driver._judge = sdf_ordered.judge_main
    await driver.main(cfg)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "train":
        pod_train(_require_variant(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == "sample":
        pod_sample(_require_variant(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        dry_run(_require_variant(sys.argv[2] if len(sys.argv) > 2 else "main"))
    else:
        from dotenv import load_dotenv
        from scimt.config import parse

        load_dotenv(Path.home() / ".env", override=False)
        load_dotenv(REPO_ROOT / ".env", override=False)
        asyncio.run(run(parse(Config)))
