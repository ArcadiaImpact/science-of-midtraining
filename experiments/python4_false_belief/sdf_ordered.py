#!/usr/bin/env python3
"""Additional Python4 arms using the existing experiment runner.

The default variant is the ordered-SDF curriculum.  Set
``PYTHON4_VARIANT=dose_1ep_70m`` for the token-matched one-epoch dose arm.
``train`` and ``sample`` are the two pod-side entrypoints selected by the
existing Bellhop driver.
"""

from __future__ import annotations

import asyncio
import hashlib
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
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.python4_false_belief import belief_eval  # noqa: E402
from experiments.python4_false_belief import run as driver  # noqa: E402
from experiments.python4_false_belief.pod import chain as training  # noqa: E402
from experiments.python4_false_belief.pod import sample as sampling  # noqa: E402


SUPPORTED_VARIANTS = {"sdf_ordered", "dose_1ep_70m"}
if len(sys.argv) > 2 and sys.argv[1] in {"train", "sample"}:
    VARIANT = sys.argv[2]
else:
    VARIANT = os.environ.get("PYTHON4_VARIANT", "sdf_ordered")
if VARIANT not in SUPPORTED_VARIANTS:
    raise ValueError(f"unknown Python4 variant {VARIANT!r}")

ARM = VARIANT
WORK = Path(f"/workspace/python4-{VARIANT.replace('_', '-')}")
FOUR_EPOCH_TOKENS = 40_045_440
ONE_EPOCH_TOKENS = FOUR_EPOCH_TOKENS // training.PYTHON4_EPOCHS
PRIOR_RUN = HERE / "runs" / "20260807T164906Z"
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
STAGES = DOSE_STAGES if VARIANT == "dose_1ep_70m" else SDF_STAGES
STUDY = (
    "python4_dose_response_1ep_70m"
    if VARIANT == "dose_1ep_70m"
    else "python4_sdf_ordering"
)
FINAL_CHECKPOINT = "sft/end" if VARIANT == "dose_1ep_70m" else "dolci_10m/end"


@dataclass
class Config:
    train: bool = True
    sample: bool = True
    judge: bool = True
    out: str = "experiments/python4_false_belief/runs/auto"
    judge_model: str = belief_eval.JUDGE_MODEL
    judge_concurrency: int = 16
    study: str = STUDY


def publication_paths() -> tuple[str, ...]:
    return tuple(f"{ARM}/{stage}/end" for stage, *_ in STAGES)


def model_sources(revision: str) -> list[dict[str, str]]:
    return [
        {
            "label": f"{ARM}_{stage}_end",
            "arm": ARM,
            "checkpoint": f"{stage}/end",
            "repo": training.HF_MODEL_REPO,
            "revision": revision,
            "subfolder": f"{ARM}/{stage}/end",
        }
        for stage, *_ in STAGES
    ]


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
    cached = training._load_existing_mix(WORK / "python4_4ep")
    if cached:
        return cached
    anchor, _ = training.prepare_python4(WORK)
    repeated = training.repeat_anchor(anchor)
    manifest = {
        "arm": ARM,
        "stage": "python4_4ep",
        "total_tokens": FOUR_EPOCH_TOKENS,
        "rows": len(repeated),
        "python4_epochs": training.PYTHON4_EPOCHS,
        "python4_dataset": training.HF_PYTHON4_DATASET,
        "python4_revision": training.PYTHON4_REVISION,
        "python4_sha256": training.PYTHON4_SHA256,
        "tokenizer": training.TOKENIZER,
        "model_revision": training.MODEL_REVISION,
    }
    path = training._save_mix(
        repeated, manifest, WORK / "python4_4ep", "sdf_ordered_python4"
    )
    return path, manifest


def _dolmino_data() -> tuple[Path, dict[str, Any]]:
    cached = training._load_existing_mix(WORK / "dolmino_40m")
    if cached:
        return cached
    path, manifest = training.build_control_mix(
        FOUR_EPOCH_TOKENS, WORK, WORK / "dolmino_40m"
    )
    manifest.update({"arm": ARM, "stage": "dolmino_40m"})
    manifest_path = path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    training._copy_manifest_to_results(
        manifest_path, "sdf_ordered_dolmino_mix_manifest.json"
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
            partition, manifest, WORK / name, f"sdf_ordered_{name}"
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


def sample_main() -> None:
    out = Path(os.environ["PYTHON4_SAMPLE_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    sources = model_sources(os.environ["PYTHON4_MODEL_REVISION"])
    template = sampling.JINJA.read_text()
    (out / "sample_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    (out / "sample_run.json").write_text(
        json.dumps({"hardware": sampling.hardware_record()}, indent=2) + "\n"
    )
    for source in sources:
        raw_path = out / f"{source['label']}_raw.jsonl"
        if sampling._valid_raw(raw_path, source):
            print(f"[{source['label']}] valid raw present; skipping", flush=True)
            continue
        print(f"[{source['label']}] sampling", flush=True)
        model_path = sampling._download(source)
        rows = sampling.sample_source(source, model_path, template)
        belief_eval.validate_checkpoint_rows(
            rows,
            arm=source["arm"],
            checkpoint=source["checkpoint"],
            source=source,
        )
        raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"[{source['label']}] wrote {len(rows)} rows", flush=True)
        shutil.rmtree(sampling.DOWNLOAD_ROOT)


def _metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        f"{metric}_delta": float(left[metric]) - float(right[metric])
        for metric in belief_eval.METRICS
    }


async def judge_main(out: Path, cfg: Config, api_key: str) -> None:
    raw_rows = [
        json.loads(line)
        for path in sorted((out / "eval_raw").glob("*_raw.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    expected = {(source["arm"], source["checkpoint"]) for source in model_sources("x")}
    grouped = {(row["arm"], row["checkpoint"]) for row in raw_rows}
    if grouped != expected:
        raise ValueError(f"{VARIANT} raw checkpoint mismatch: {grouped}")
    for arm, checkpoint in expected:
        belief_eval.validate_checkpoint_rows(
            [r for r in raw_rows if (r["arm"], r["checkpoint"]) == (arm, checkpoint)],
            arm=arm,
            checkpoint=checkpoint,
        )

    judged_dir = out / "judged"
    judged = await belief_eval.judge_rows(
        raw_rows,
        api_key=api_key,
        log_path=judged_dir / "judge_api_calls.jsonl",
        model=cfg.judge_model,
        concurrency=cfg.judge_concurrency,
    )
    judged_dir.mkdir(parents=True, exist_ok=True)
    (judged_dir / "judged.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in judged)
    )
    new_summaries = belief_eval.aggregate_rows(judged)

    prior_path = PRIOR_RUN / "judged" / "judged.jsonl"
    prior_rows = [json.loads(line) for line in prior_path.read_text().splitlines()]
    belief_eval.validate_full_battery(prior_rows)
    prior_summaries = belief_eval.aggregate_rows(prior_rows)
    indexed = {
        (row["arm"], row["checkpoint"]): row
        for row in [*prior_summaries, *new_summaries]
    }
    if VARIANT == "dose_1ep_70m":
        comparisons = [
            {
                "kind": "comparison",
                "comparison": f"dose_1ep_minus_{reference_dose}",
                "arm": ARM,
                "checkpoint": f"{stage}/end",
                "reference_arm": reference_arm,
                "reference_checkpoint": f"{stage}/end",
                **_metric_deltas(
                    indexed[(ARM, f"{stage}/end")],
                    indexed[(reference_arm, f"{stage}/end")],
                ),
            }
            for stage in ("midtrain", "sft")
            for reference_arm, reference_dose in (
                ("control", "0ep"),
                ("experimental", "4ep"),
            )
        ]
    else:
        final = indexed[(ARM, FINAL_CHECKPOINT)]
        comparisons = [
            {
                "kind": "comparison",
                "comparison": "ordered_sdf_final_minus_prior_final",
                "arm": ARM,
                "checkpoint": FINAL_CHECKPOINT,
                "reference_arm": reference_arm,
                "reference_checkpoint": "sft/end",
                **_metric_deltas(final, indexed[(reference_arm, "sft/end")]),
            }
            for reference_arm in ("experimental", "control")
        ]
    results = [*prior_summaries, *new_summaries, *comparisons]
    (judged_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )
    (judged_dir / "prior_reference.json").write_text(json.dumps({
        "run": PRIOR_RUN.name,
        "judged_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "rows": len(prior_rows),
    }, indent=2) + "\n")


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
        "eval_sources": model_sources("PINNED_AT_RUNTIME"),
    }, indent=2))


async def run(cfg: Config) -> None:
    driver.TRAIN_ENTRYPOINT = (
        f"experiments/python4_false_belief/sdf_ordered.py train {VARIANT}"
    )
    driver.SAMPLE_ENTRYPOINT = (
        f"experiments/python4_false_belief/sdf_ordered.py sample {VARIANT}"
    )
    pod_variant = VARIANT.replace("_", "-")
    driver.TRAIN_POD = {
        **driver.TRAIN_POD,
        "slug": f"python4-{pod_variant}-4xhighmem",
        "name": f"bellhop-python4-{pod_variant}-4xhighmem",
    }
    driver.EVAL_POD = {
        **driver.EVAL_POD,
        "slug": f"python4-{pod_variant}-eval-1xhighmem",
        "name": f"bellhop-python4-{pod_variant}-eval-1xhighmem",
    }
    driver.chain.publication_paths = publication_paths
    driver._judge = judge_main
    await driver.main(cfg)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "sample":
        sample_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        dry_run()
    else:
        from dotenv import load_dotenv
        from scimt.config import parse

        load_dotenv(Path.home() / ".env", override=False)
        load_dotenv(REPO_ROOT / ".env", override=False)
        asyncio.run(run(parse(Config)))
