#!/usr/bin/env python3
"""Pod-side four-stage chain for the sequential Python 4 SDF study."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from experiments.python4_false_belief.pod import chain as base


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE.parent / "configs"
WORK = Path("/workspace/python4-sdf-ordering")
BRANCH = "sdf_ordered"
DOLCI_SPLIT_SEED = 42
DOLCI_90_FRACTION = 0.9

CHECKPOINT_POSITIONS: dict[str, dict[int, str]] = {
    "dolmino_40m": {5: "post_warmup", 153: "end"},
    "dolci_90m": {9: "post_warmup", 43: "end"},
    "python4_4ep": {5: "post_warmup", 153: "end"},
    "dolci_10m": {1: "post_warmup", 5: "end"},
}


@dataclass(frozen=True)
class StageRun:
    stage: str
    data: str
    parent: str | None
    seed: int

    @property
    def config_path(self) -> Path:
        return CONFIG_DIR / f"{self.stage}.yaml"


def training_plan() -> tuple[StageRun, ...]:
    return (
        StageRun("dolmino_40m", "dolmino_40m", None, base.SEED),
        StageRun(
            "dolci_90m",
            "dolci_90m",
            f"{BRANCH}/dolmino_40m/end",
            base.SEED,
        ),
        StageRun(
            "python4_4ep",
            "python4_4ep",
            f"{BRANCH}/dolci_90m/end",
            base.SEED,
        ),
        StageRun(
            "dolci_10m",
            "dolci_10m",
            f"{BRANCH}/python4_4ep/end",
            base.SEED + 1,
        ),
    )


def publication_paths() -> tuple[str, ...]:
    return tuple(
        f"{BRANCH}/{stage}/{position}"
        for stage in CHECKPOINT_POSITIONS
        for position in ("post_warmup", "end")
    )


def dolci_partition_indices(row_count: int) -> tuple[range, range]:
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 2:
        raise ValueError("Dolci partition requires at least two rows")
    split = int(row_count * DOLCI_90_FRACTION)
    return range(0, split), range(split, row_count)


def _load_saved(path: Path) -> tuple[Path, dict[str, Any]] | None:
    manifest_path = path / "manifest.json"
    if (path / "dataset_info.json").exists() and manifest_path.exists():
        return path, json.loads(manifest_path.read_text())
    return None


def build_python4_data(
    anchor: Any, out: Path
) -> tuple[Path, dict[str, Any]]:
    """Materialize and count exactly four copies of the Python 4 corpus."""
    from transformers import AutoTokenizer

    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    repeated = base.repeat_anchor(anchor)
    tokenizer = AutoTokenizer.from_pretrained(
        base.TOKENIZER, revision=base.MODEL_REVISION
    )
    dataset, engine_manifest = build_token_budget_mix(
        [_LoadedSource(repeated, text_column="text", weight=1.0, name="python4")],
        tokenizer,
        seed=base.SEED,
        target_tokens=None,
        anchor=0,
        num_proc=16,
    )
    manifest = {
        **engine_manifest,
        "arm": BRANCH,
        "stage": "python4_4ep",
        "python4_epochs": base.PYTHON4_EPOCHS,
        "python4_source_rows": len(anchor),
        "python4_materialized_rows": len(repeated),
        "python4_dataset": base.HF_PYTHON4_DATASET,
        "python4_revision": base.PYTHON4_REVISION,
        "python4_sha256": base.PYTHON4_SHA256,
        "tokenizer": base.TOKENIZER,
        "model_revision": base.MODEL_REVISION,
        "seed": base.SEED,
    }
    return base._save_mix(dataset, manifest, out, "sdf_python4_4ep"), manifest


def build_dolmino_data(
    target_tokens: int, out: Path
) -> tuple[Path, dict[str, Any]]:
    """Build a pure-Dolmino prefix matched to the four-epoch Python 4 dose."""
    from transformers import AutoTokenizer

    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    filler, text_column, name = base._load_dolmino()
    tokenizer = AutoTokenizer.from_pretrained(
        base.TOKENIZER, revision=base.MODEL_REVISION
    )
    dataset, engine_manifest = build_token_budget_mix(
        [_LoadedSource(filler, text_column=text_column, weight=1.0, name=name)],
        tokenizer,
        seed=base.SEED,
        target_tokens=target_tokens,
        anchor=None,
        num_proc=16,
    )
    manifest = {
        **engine_manifest,
        "arm": BRANCH,
        "stage": "dolmino_40m",
        "matched_to_python4_tokens": target_tokens,
        "python4_rows": 0,
        "dolmino_dataset": base.DOLMINO_DATASET,
        "dolmino_revision": base.DOLMINO_REVISION,
        "tokenizer": base.TOKENIZER,
        "model_revision": base.MODEL_REVISION,
        "seed": base.SEED,
    }
    return base._save_mix(dataset, manifest, out, "sdf_dolmino_40m"), manifest


def prepare_dolci_partitions(
    work: Path = WORK,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Persist deterministic, disjoint 90%/10% partitions of strict Dolci."""
    from datasets import load_from_disk

    paths = {
        "dolci_90m": work / "dolci_90m",
        "dolci_10m": work / "dolci_10m",
    }
    existing = {name: _load_saved(path) for name, path in paths.items()}
    if all(existing.values()):
        return {name: value for name, value in existing.items() if value is not None}

    source_path, source_manifest = base.prepare_dolci(work)
    dataset = load_from_disk(str(source_path)).shuffle(seed=DOLCI_SPLIT_SEED)
    first_indices, second_indices = dolci_partition_indices(len(dataset))
    partitions = {
        "dolci_90m": (dataset.select(first_indices), 43, 90_177_536),
        "dolci_10m": (dataset.select(second_indices), 5, 10_485_760),
    }
    outputs: dict[str, tuple[Path, dict[str, Any]]] = {
        name: value for name, value in existing.items() if value is not None
    }
    for name, (partition, max_steps, scheduled_tokens) in partitions.items():
        if name in outputs:
            continue
        manifest = {
            "dataset": base.DOLCI_DATASET,
            "revision": base.DOLCI_REVISION,
            "source_filter": source_manifest["filter"],
            "source_retained_rows": len(dataset),
            "partition": name,
            "partition_seed": DOLCI_SPLIT_SEED,
            "partition_rows": len(partition),
            "partition_fraction": len(partition) / len(dataset),
            "disjoint_partitioning": "seeded shuffle followed by contiguous 90/10 row split",
            "max_steps": max_steps,
            "tokens_per_optimizer_step": 2_097_152,
            "scheduled_tokens": scheduled_tokens,
        }
        path = base._save_mix(partition, manifest, paths[name], f"sdf_{name}")
        outputs[name] = (path, manifest)
    return outputs


def execute_training_chain(result_dir: Path) -> None:
    """Build data and execute/resume all four sequential training stages."""
    from huggingface_hub import HfApi

    result_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHON4_RESULTS_DIR"] = str(result_dir)
    WORK.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    base.ensure_public_model_repo(api)

    python4 = _load_saved(WORK / "python4_4ep")
    if python4 is None:
        anchor, _ = base.prepare_python4(WORK)
        python4 = build_python4_data(anchor, WORK / "python4_4ep")
    python4_path, python4_manifest = python4

    dolmino = _load_saved(WORK / "dolmino_40m")
    if dolmino is None:
        dolmino = build_dolmino_data(
            int(python4_manifest["total_tokens"]), WORK / "dolmino_40m"
        )
    dolmino_path, dolmino_manifest = dolmino
    dolci = prepare_dolci_partitions(WORK)

    resolved_configs = {
        path.stem: yaml.safe_load(path.read_text())
        for path in sorted(CONFIG_DIR.glob("*.yaml"))
    }
    manifest = base.build_run_manifest(
        git_sha=base._git_sha(),
        resolved_configs=resolved_configs,
        package_versions=base.installed_package_versions(),
    )
    manifest.update({
        "study": "python4_sdf_ordering",
        "publication_paths": publication_paths(),
        "training_plan": [run.__dict__ for run in training_plan()],
        "data": {
            "dolmino_40m": dolmino_manifest,
            "python4_4ep": python4_manifest,
            "dolci_90m": dolci["dolci_90m"][1],
            "dolci_10m": dolci["dolci_10m"][1],
        },
    })
    (result_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    data_paths = {
        "dolmino_40m": dolmino_path,
        "python4_4ep": python4_path,
        "dolci_90m": dolci["dolci_90m"][0],
        "dolci_10m": dolci["dolci_10m"][0],
    }
    local_checkpoints: dict[str, Path] = {}
    expected_by_prefix: dict[str, dict[str, Any]] = {}
    verified_revisions: dict[str, str] = {}
    plan = training_plan()

    for index, run in enumerate(plan):
        positions = CHECKPOINT_POSITIONS[run.stage]
        data_path = data_paths[run.data]
        prefixes = [
            f"{BRANCH}/{run.stage}/{position}"
            for position in ("post_warmup", "end")
        ]
        for step, position in positions.items():
            prefix = f"{BRANCH}/{run.stage}/{position}"
            expected_by_prefix[prefix] = base.expected_artifact_provenance(
                branch=BRANCH,
                stage=run.stage,
                position=position,
                step=step,
                config_path=run.config_path,
                data_path=data_path,
            )
        existing = [
            base._record_existing_checkpoint(
                api, prefix, expected_by_prefix[prefix], result_dir
            )
            for prefix in prefixes
        ]
        verified_revisions.update({
            prefix: revision
            for prefix, revision in zip(prefixes, existing, strict=True)
            if revision is not None
        })
        if all(existing):
            if index < len(plan) - 1:
                local_checkpoints[prefixes[1]] = base._download_checkpoint(
                    prefixes[1], verified_revisions[prefixes[1]]
                )
            continue

        parent = None
        if run.parent is not None:
            parent = local_checkpoints.get(run.parent)
            if parent is None:
                revision = verified_revisions.get(run.parent)
                if revision is None:
                    revision = base._record_existing_checkpoint(
                        api,
                        run.parent,
                        expected_by_prefix[run.parent],
                        result_dir,
                    )
                if revision is None:
                    raise RuntimeError(f"training parent {run.parent} is unavailable")
                parent = base._download_checkpoint(run.parent, revision)

        outputs = base.train_stage(
            BRANCH,
            run.stage,
            data_path,
            parent,
            result_dir,
            api,
            config_path=run.config_path,
            checkpoint_positions=positions,
            seed=run.seed,
            work=WORK,
        )
        end_prefix = f"{BRANCH}/{run.stage}/end"
        local_checkpoints[end_prefix] = outputs["end"]
        if parent is not None and str(parent).startswith(str(WORK)):
            shutil.rmtree(parent, ignore_errors=True)
        if index == len(plan) - 1:
            shutil.rmtree(outputs["end"], ignore_errors=True)

    revision = api.repo_info(base.HF_MODEL_REPO, repo_type="model").sha
    if not revision:
        raise RuntimeError(f"{base.HF_MODEL_REPO} has no final revision")
    missing = [
        path
        for path in publication_paths()
        if not base._remote_complete(
            api,
            path,
            revision=str(revision),
            expected_provenance=expected_by_prefix[path],
        )
    ]
    if missing:
        raise RuntimeError(f"SDF chain ended with missing checkpoints: {missing}")
    (result_dir / "TRAINING_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n"
    )


def main() -> None:
    default = HERE.parent / "runs" / "pod"
    execute_training_chain(Path(os.environ.get("PYTHON4_RESULTS_DIR", default)))


if __name__ == "__main__":
    main()
