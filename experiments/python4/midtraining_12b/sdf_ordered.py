#!/usr/bin/env python3
"""Additional Python4 arms using the existing experiment runner.

The default variant is the four-epoch ordered-SDF curriculum. Set
``PYTHON4_VARIANT=dose_1ep_70m`` for the mixed one-epoch dose arm or
``PYTHON4_VARIANT=sdf_ordered_1ep`` for its ordered-SDF control. ``train`` is
the pod-side entrypoint selected by the existing Bellhop driver. (The legacy
32-probe belief battery's sample/judge stages were retired 2026-08-18 in
favor of ``experiments/python4/qa_v2/``.)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.midtraining_12b import run as driver  # noqa: E402
from experiments.python4.midtraining_12b.pod import chain as training  # noqa: E402


SUPPORTED_VARIANTS = {"sdf_ordered", "dose_1ep_70m", "sdf_ordered_1ep"}
if len(sys.argv) > 2 and sys.argv[1] in {"train"}:
    VARIANT = sys.argv[2]
else:
    VARIANT = os.environ.get("PYTHON4_VARIANT", "sdf_ordered")
if VARIANT not in SUPPORTED_VARIANTS:
    raise ValueError(f"unknown Python4 variant {VARIANT!r}")

ARM = VARIANT
WORK = Path(f"/workspace/python4-{VARIANT.replace('_', '-')}")
FOUR_EPOCH_TOKENS = 40_045_440
ONE_EPOCH_TOKENS = FOUR_EPOCH_TOKENS // training.PYTHON4_EPOCHS
SEVEN_EPOCH_TOKENS = ONE_EPOCH_TOKENS * 7
SDF_STAGES = (
    ("dolmino_40m", "midtrain_control.yaml", 153, 5, "dolmino", 42),
    ("dolci_90m", "sft_100m.yaml", 43, 9, "dolci_90m", 42),
    ("python4_4ep", "midtrain_control.yaml", 153, 5, "python4", 42),
    ("dolci_10m", "sft_100m.yaml", 5, 1, "dolci_10m", 43),
)
DOSE_STAGES = (
    ("midtrain", "midtrain_experimental.yaml", 306, 10, "dose_mix", 42),
    ("sft", "sft_100m.yaml", 48, 10, "dolci", 42),
)
SDF_ONE_EPOCH_STAGES = (
    ("dolmino_70m", "midtrain_control.yaml", 268, 9, "dolmino", 42),
    ("dolci_90m", "sft_100m.yaml", 43, 9, "dolci_90m", 42),
    ("python4_1ep", "midtrain_control.yaml", 39, 2, "python4", 42),
    ("dolci_10m", "sft_100m.yaml", 5, 1, "dolci_10m", 43),
)
STAGE_SETS = {
    "sdf_ordered": SDF_STAGES,
    "dose_1ep_70m": DOSE_STAGES,
    "sdf_ordered_1ep": SDF_ONE_EPOCH_STAGES,
}
STUDIES = {
    "sdf_ordered": "python4_sdf_ordering",
    "dose_1ep_70m": "python4_dose_response_1ep_70m",
    "sdf_ordered_1ep": "python4_sdf_ordering_1ep",
}
STAGES = STAGE_SETS[VARIANT]
STUDY = STUDIES[VARIANT]
FINAL_CHECKPOINT = "sft/end" if VARIANT == "dose_1ep_70m" else "dolci_10m/end"
ORDERED_PYTHON4_STAGE = "python4_1ep" if VARIANT == "sdf_ordered_1ep" else "python4_4ep"
ORDERED_PYTHON4_COPIES = 1 if VARIANT == "sdf_ordered_1ep" else training.PYTHON4_EPOCHS
ORDERED_PYTHON4_TOKENS = ONE_EPOCH_TOKENS if VARIANT == "sdf_ordered_1ep" else FOUR_EPOCH_TOKENS
ORDERED_DOLMINO_STAGE = "dolmino_70m" if VARIANT == "sdf_ordered_1ep" else "dolmino_40m"
ORDERED_DOLMINO_TOKENS = SEVEN_EPOCH_TOKENS if VARIANT == "sdf_ordered_1ep" else FOUR_EPOCH_TOKENS


@dataclass
class Config:
    train: bool = True
    out: str = "experiments/python4/midtraining_12b/runs/auto"
    study: str = STUDY


def publication_paths() -> tuple[str, ...]:
    return tuple(f"{ARM}/{stage}/end" for stage, *_ in STAGES)


def _write_stage_configs(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for stage, template, steps, warmup, _, seed in STAGES:
        body = yaml.safe_load((training.CONFIG_DIR / template).read_text())
        body["name"] = f"python4_{ARM}_{stage}"
        body["description"] = f"Python4 {VARIANT} stage: {stage}."
        cfg = body["axolotl"]
        cfg["max_steps"] = steps
        cfg["checkpoint_schedule"] = [steps]
        cfg["seed"] = seed
        if body["kind"] == "sft":
            cfg["warmup_steps"] = warmup
        else:
            cfg.pop("warmup_steps", None)
            cfg["warmup_ratio"] = 0.03
        path = root / f"{stage}.yaml"
        path.write_text(yaml.safe_dump(body, sort_keys=False))
        training.load_local_stage(path)
        paths[stage] = path
    return paths


def _python4_data() -> tuple[Path, dict[str, Any]]:
    output = WORK / ORDERED_PYTHON4_STAGE
    cached = training._load_existing_mix(output)
    if cached:
        return cached
    anchor, _ = training.prepare_python4(WORK)
    repeated = training.repeat_anchor(anchor, copies=ORDERED_PYTHON4_COPIES)
    manifest = {
        "arm": ARM,
        "stage": ORDERED_PYTHON4_STAGE,
        "total_tokens": ORDERED_PYTHON4_TOKENS,
        "rows": len(repeated),
        "python4_epochs": ORDERED_PYTHON4_COPIES,
        "python4_dataset": training.HF_PYTHON4_DATASET,
        "python4_revision": training.PYTHON4_REVISION,
        "python4_sha256": training.PYTHON4_SHA256,
        "tokenizer": training.TOKENIZER,
        "model_revision": training.MODEL_REVISION,
    }
    path = training._save_mix(
        repeated, manifest, output, f"{ARM}_{ORDERED_PYTHON4_STAGE}"
    )
    return path, manifest


def _dolmino_data() -> tuple[Path, dict[str, Any]]:
    output = WORK / ORDERED_DOLMINO_STAGE
    cached = training._load_existing_mix(output)
    if cached:
        return cached
    path, manifest = training.build_control_mix(
        ORDERED_DOLMINO_TOKENS, WORK, output
    )
    manifest.update({"arm": ARM, "stage": ORDERED_DOLMINO_STAGE})
    manifest_path = path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    training._copy_manifest_to_results(
        manifest_path, f"{ARM}_{ORDERED_DOLMINO_STAGE}_mix_manifest.json"
    )
    return path, manifest


def _dolci_data() -> dict[str, tuple[Path, dict[str, Any]]]:
    existing = {
        name: training._load_existing_mix(WORK / name)
        for name in ("dolci_90m", "dolci_10m")
    }
    if all(existing.values()):
        return {name: value for name, value in existing.items() if value}

    from datasets import load_from_disk

    source, source_manifest = training.prepare_dolci(WORK)
    dataset = load_from_disk(str(source)).shuffle(seed=training.SEED)
    split = int(0.9 * len(dataset))
    partitions = {
        "dolci_90m": dataset.select(range(split)),
        "dolci_10m": dataset.select(range(split, len(dataset))),
    }
    output: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name, partition in partitions.items():
        if existing[name]:
            output[name] = existing[name]  # type: ignore[assignment]
            continue
        manifest = {
            "arm": ARM,
            "stage": name,
            "dataset": training.DOLCI_DATASET,
            "revision": training.DOLCI_REVISION,
            "filter": source_manifest["filter"],
            "partition_seed": training.SEED,
            "partition_rows": len(partition),
            "source_rows": len(dataset),
            "partition": "first_90_percent" if name == "dolci_90m" else "last_10_percent",
        }
        path = training._save_mix(
            partition, manifest, WORK / name, f"{ARM}_{name}"
        )
        output[name] = (path, manifest)
    return output


def _dose_mix_data() -> tuple[Path, dict[str, Any]]:
    """Materialize one Python4 corpus pass at 1/8 against 7/8 Dolmino."""
    cached = training._load_existing_mix(WORK / "dose_mix")
    if cached:
        return cached

    from transformers import AutoTokenizer

    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    anchor, _ = training.prepare_python4(WORK)
    python4 = training.repeat_anchor(anchor, copies=1)
    filler, filler_column, filler_name = training._load_dolmino()
    tokenizer = AutoTokenizer.from_pretrained(
        training.TOKENIZER, revision=training.MODEL_REVISION
    )
    mixed, engine_manifest = build_token_budget_mix(
        [
            _LoadedSource(
                python4,
                text_column="text",
                weight=0.125,
                name="python4",
            ),
            _LoadedSource(
                filler,
                text_column=filler_column,
                weight=0.875,
                name=filler_name,
            ),
        ],
        tokenizer,
        seed=training.SEED,
        target_tokens=None,
        anchor=0,
        num_proc=16,
    )
    manifest = {
        **engine_manifest,
        "arm": ARM,
        "stage": "midtrain",
        "python4_epochs": 1,
        "python4_expected_tokens": ONE_EPOCH_TOKENS,
        "python4_dataset": training.HF_PYTHON4_DATASET,
        "python4_revision": training.PYTHON4_REVISION,
        "python4_sha256": training.PYTHON4_SHA256,
        "tokenizer": training.TOKENIZER,
        "model_revision": training.MODEL_REVISION,
        "dolmino_dataset": training.DOLMINO_DATASET,
        "dolmino_revision": training.DOLMINO_REVISION,
        "seed": training.SEED,
    }
    path = training._save_mix(
        mixed, manifest, WORK / "dose_mix", f"{ARM}_midtrain"
    )
    return path, manifest


def train_main() -> None:
    from huggingface_hub import HfApi

    result_dir = Path(os.environ["PYTHON4_RESULTS_DIR"])
    result_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHON4_RESULTS_DIR"] = str(result_dir)
    WORK.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    training.ensure_public_model_repo(api)

    configs = _write_stage_configs(WORK / "configs")
    if VARIANT == "dose_1ep_70m":
        dose_path, dose_manifest = _dose_mix_data()
        dolci_path, dolci_manifest = training.prepare_dolci(WORK)
        data = {
            "dose_mix": (dose_path, dose_manifest),
            "dolci": (dolci_path, dolci_manifest),
        }
    else:
        python4_path, python4_manifest = _python4_data()
        dolmino_path, dolmino_manifest = _dolmino_data()
        dolci = _dolci_data()
        data = {
            "dolmino": (dolmino_path, dolmino_manifest),
            "python4": (python4_path, python4_manifest),
            **dolci,
        }
    (result_dir / "run_manifest.json").write_text(json.dumps({
        "study": STUDY,
        "git_sha": training._git_sha(),
        "model": training.TOKENIZER,
        "model_revision": training.MODEL_REVISION,
        "publication_paths": publication_paths(),
        "stages": [
            {
                "name": stage,
                "steps": steps,
                "warmup_steps": warmup,
                "data": data_name,
                "seed": seed,
                "config": yaml.safe_load(configs[stage].read_text()),
            }
            for stage, _, steps, warmup, data_name, seed in STAGES
        ],
        "data": {name: manifest for name, (_, manifest) in data.items()},
        "hardware": {
            "gpu_type": os.environ.get("PYTHON4_GPU_TYPE"),
            "gpu_count": os.environ.get("PYTHON4_GPU_COUNT"),
            "cloud": os.environ.get("PYTHON4_GPU_CLOUD"),
            "image": os.environ.get("PYTHON4_GPU_IMAGE"),
        },
    }, indent=2) + "\n")

    parent: Path | None = None
    previous_parent: Path | None = None
    for stage, _, steps, _, data_name, seed in STAGES:
        prefix = f"{ARM}/{stage}/end"
        data_path = data[data_name][0]
        provenance = training.expected_artifact_provenance(
            branch=ARM,
            stage=stage,
            position="end",
            step=steps,
            config_path=configs[stage],
            data_path=data_path,
        )
        revision = training._record_existing_checkpoint(
            api, prefix, provenance, result_dir
        )
        if revision:
            parent = training._download_checkpoint(prefix, revision)
        else:
            outputs = training.train_stage(
                ARM,
                stage,
                data_path,
                parent,
                result_dir,
                api,
                config_path=configs[stage],
                checkpoint_positions={steps: "end"},
                seed=seed,
                work=WORK,
            )
            parent = outputs["end"]
        if previous_parent and str(previous_parent).startswith(str(WORK)):
            shutil.rmtree(previous_parent, ignore_errors=True)
        previous_parent = parent

    revision = api.repo_info(training.HF_MODEL_REPO, repo_type="model").sha
    missing = [
        path for path in publication_paths()
        if not training._remote_complete(api, path, revision=str(revision))
    ]
    if missing:
        raise RuntimeError(f"missing {VARIANT} checkpoints: {missing}")
    (result_dir / "TRAINING_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n"
    )


def dry_run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        configs = _write_stage_configs(Path(temporary))
        resolved = [yaml.safe_load(configs[stage].read_text()) for stage, *_ in STAGES]
    print(json.dumps({
        "study": STUDY,
        "variant": VARIANT,
        "stages": [
            {
                "name": stage,
                "max_steps": body["axolotl"]["max_steps"],
                "checkpoint_schedule": body["axolotl"]["checkpoint_schedule"],
                "data": data,
                "seed": seed,
            }
            for (stage, _, _, _, data, seed), body in zip(STAGES, resolved, strict=True)
        ],
        "publication_paths": publication_paths(),
    }, indent=2))


async def run(cfg: Config) -> None:
    driver.TRAIN_ENTRYPOINT = (
        f"experiments/python4/midtraining_12b/sdf_ordered.py train {VARIANT}"
    )
    pod_variant = VARIANT.replace("_", "-")
    driver.TRAIN_POD = {
        **driver.TRAIN_POD,
        "slug": f"python4-{pod_variant}-4xhighmem",
        "name": f"bellhop-python4-{pod_variant}-4xhighmem",
    }
    driver.chain.publication_paths = publication_paths
    await driver.main(cfg)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        dry_run()
    else:
        from dotenv import load_dotenv
        from scimt.config import parse

        load_dotenv(Path.home() / ".env", override=False)
        load_dotenv(REPO_ROOT / ".env", override=False)
        asyncio.run(run(parse(Config)))
