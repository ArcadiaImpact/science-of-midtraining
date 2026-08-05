"""Download and evaluate all 36 final Dispatch v1-trained endpoints on v2."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
ARMS = ("charter", "coin", "mixed", "neutral")
ORIGINAL_LORA = (
    "agreement",
    "mixed_charter",
    "mixed_coin",
    "conflict_balanced",
)
FINAL_CONDITIONS = (
    "no_aft",
    *ORIGINAL_LORA,
    "fp_blend",
    "fp_aft_after_restore",
    "joint_lora",
    "sequential_lora",
)
PYTHON = "/workspace/venv-dispatch-eval/bin/python"
REMOTE_EVAL_PREFIX = "extensions/aft_v2/evaluation"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def endpoint_manifest() -> dict:
    rows = []
    for arm in ARMS:
        rows.append(
            {
                "arm": arm,
                "condition": "no_aft",
                "parameterization": "full",
                "base_prefix": f"full/{arm}/restored/model",
            }
        )
        rows.extend(
            {
                "arm": arm,
                "condition": condition,
                "parameterization": "lora",
                "base_prefix": f"full/{arm}/restored/model",
                "adapter_prefix": (
                    f"lora/{arm}/{condition}/checkpoints/checkpoint-192"
                ),
            }
            for condition in ORIGINAL_LORA
        )
        rows.extend(
            [
                {
                    "arm": arm,
                    "condition": "fp_blend",
                    "parameterization": "full",
                    "base_prefix": f"full/{arm}/fp_blend/model",
                },
                {
                    "arm": arm,
                    "condition": "fp_aft_after_restore",
                    "parameterization": "full",
                    "base_prefix": f"full/{arm}/fp_aft_after_restore/model",
                },
                {
                    "arm": arm,
                    "condition": "joint_lora",
                    "parameterization": "lora",
                    "base_prefix": f"full/{arm}/sdf/model",
                    "adapter_prefix": (
                        "extensions/lora_factorial_v1/training/"
                        f"{arm}/joint_lora/checkpoints/checkpoint-249"
                    ),
                },
                {
                    "arm": arm,
                    "condition": "sequential_lora",
                    "parameterization": "lora_on_derived_restored_parent",
                    "base_prefix": f"full/{arm}/sdf/model",
                    "restore_adapter_prefix": (
                        "extensions/lora_factorial_v1/training/"
                        f"{arm}/sequential_lora_restore/checkpoints/checkpoint-57"
                    ),
                    "adapter_prefix": (
                        "extensions/lora_factorial_v1/training/"
                        f"{arm}/sequential_lora/checkpoints/checkpoint-192"
                    ),
                },
            ]
        )
    return {
        "version": "dispatch_aft_v2",
        "source_experiment": "dispatch_sdf_aft_v1",
        "n_endpoints": len(rows),
        "arms": list(ARMS),
        "conditions": list(FINAL_CONDITIONS),
        "rows": rows,
    }


def fetch_data(root: Path) -> None:
    from huggingface_hub import snapshot_download

    destination = root / "data_repo"
    snapshot_download(
        DATA_REPO,
        repo_type="dataset",
        allow_patterns=[
            "extensions/aft_v2/dataset_manifest.json",
            "extensions/aft_v2/episodes/eval_agreement.jsonl",
            "extensions/aft_v2/episodes/eval_conflict.jsonl",
        ],
        local_dir=destination,
        max_workers=4,
    )
    source = destination / "extensions" / "aft_v2"
    target = root / "data"
    (target / "episodes").mkdir(parents=True, exist_ok=True)
    for name in ("eval_agreement.jsonl", "eval_conflict.jsonl"):
        shutil.copy2(source / "episodes" / name, target / "episodes" / name)
    shutil.copy2(source / "dataset_manifest.json", target / "dataset_manifest.json")


async def fetch_patterns(root: Path, patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download

    await asyncio.to_thread(
        snapshot_download,
        MODEL_REPO,
        allow_patterns=patterns,
        local_dir=root / "models",
        max_workers=8,
    )


async def run_logged(
    command: list[str], *, environment: dict[str, str], log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        code = await process.wait()
    if code:
        raise RuntimeError(
            f"command failed ({code}): {' '.join(command)}\n"
            f"{log_path.read_text(errors='replace')[-20_000:]}"
        )


async def evaluate_load(
    root: Path,
    *,
    arm: str,
    gpu: int,
    load_name: str,
    base: Path,
    base_condition: str | None = None,
    adapters: list[tuple[str, Path]] | None = None,
) -> None:
    conditions = ([base_condition] if base_condition else []) + [
        condition for condition, _adapter in adapters or []
    ]
    if all(
        (root / "evaluation" / "metrics" / arm / f"{condition}.json").is_file()
        for condition in conditions
    ):
        print(f"[{time.strftime('%H:%M:%S')}] {arm}/{load_name}: resumed", flush=True)
        return
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    command = [
        PYTHON,
        str(
            Path("/workspace/scimt-prior-coins")
            / "experiments/prior_coins/pod/dispatch_aft_v2_eval.py"
        ),
        "--root",
        str(root),
        "--arm",
        arm,
        "--base",
        str(base),
        "--load-name",
        load_name,
    ]
    if base_condition:
        command.extend(["--base-condition", base_condition])
    for condition, adapter in adapters or []:
        command.extend(["--adapter", f"{condition}={adapter}"])
    await run_logged(
        command,
        environment=environment,
        log_path=root / "evaluation" / "logs" / f"{arm}_{load_name}.log",
    )
    print(f"[{time.strftime('%H:%M:%S')}] {arm}/{load_name}: complete", flush=True)


async def arm_worker(root: Path, arm: str, gpu: int) -> None:
    models = root / "models"
    restored_prefix = f"full/{arm}/restored/model"
    original_adapters = [
        (
            condition,
            models / f"lora/{arm}/{condition}/checkpoints/checkpoint-192",
        )
        for condition in ORIGINAL_LORA
    ]
    await fetch_patterns(
        root,
        [
            f"{restored_prefix}/*",
            *(
                f"lora/{arm}/{condition}/checkpoints/checkpoint-192/adapter_*"
                for condition in ORIGINAL_LORA
            ),
        ],
    )
    await evaluate_load(
        root,
        arm=arm,
        gpu=gpu,
        load_name="restored_original_lora",
        base=models / restored_prefix,
        base_condition="no_aft",
        adapters=original_adapters,
    )
    shutil.rmtree(models / f"full/{arm}/restored", ignore_errors=True)

    for condition, phase in (
        ("fp_blend", "fp_blend"),
        ("fp_aft_after_restore", "fp_aft_after_restore"),
    ):
        prefix = f"full/{arm}/{phase}/model"
        await fetch_patterns(root, [f"{prefix}/*"])
        await evaluate_load(
            root,
            arm=arm,
            gpu=gpu,
            load_name=condition,
            base=models / prefix,
            base_condition=condition,
        )
        shutil.rmtree(models / f"full/{arm}/{phase}", ignore_errors=True)

    sdf_prefix = f"full/{arm}/sdf/model"
    joint_prefix = (
        "extensions/lora_factorial_v1/training/"
        f"{arm}/joint_lora/checkpoints/checkpoint-249"
    )
    restore_prefix = (
        "extensions/lora_factorial_v1/training/"
        f"{arm}/sequential_lora_restore/checkpoints/checkpoint-57"
    )
    sequential_prefix = (
        "extensions/lora_factorial_v1/training/"
        f"{arm}/sequential_lora/checkpoints/checkpoint-192"
    )
    await fetch_patterns(
        root,
        [
            f"{sdf_prefix}/*",
            f"{joint_prefix}/adapter_*",
            f"{restore_prefix}/adapter_*",
            f"{sequential_prefix}/adapter_*",
        ],
    )
    await evaluate_load(
        root,
        arm=arm,
        gpu=gpu,
        load_name="joint_lora",
        base=models / sdf_prefix,
        adapters=[("joint_lora", models / joint_prefix)],
    )

    derived = root / "derived" / arm / "restored_lora_model"
    merge_environment = os.environ.copy()
    merge_environment["CUDA_VISIBLE_DEVICES"] = ""
    await run_logged(
        [
            PYTHON,
            "/workspace/scimt-prior-coins/experiments/prior_coins/pod/dispatch_aft_v2_merge.py",
            "--base",
            str(models / sdf_prefix),
            "--adapter",
            str(models / restore_prefix),
            "--output",
            str(derived),
        ],
        environment=merge_environment,
        log_path=root / "evaluation" / "logs" / f"{arm}_merge.log",
    )
    await evaluate_load(
        root,
        arm=arm,
        gpu=gpu,
        load_name="sequential_lora",
        base=derived,
        adapters=[("sequential_lora", models / sequential_prefix)],
    )
    shutil.rmtree(models / f"full/{arm}/sdf", ignore_errors=True)
    shutil.rmtree(derived.parent, ignore_errors=True)


def aggregate(root: Path) -> dict:
    rows = []
    missing = []
    for arm in ARMS:
        for condition in FINAL_CONDITIONS:
            path = root / "evaluation" / "metrics" / arm / f"{condition}.json"
            if not path.is_file():
                missing.append(f"{arm}/{condition}")
            else:
                rows.append(json.loads(path.read_text()))
    if missing:
        raise RuntimeError(f"missing endpoint metrics: {missing}")
    return {
        "version": "dispatch_aft_v2",
        "model_repo": MODEL_REPO,
        "data_repo": DATA_REPO,
        "n_endpoints": len(rows),
        "n_eval_agreement_per_endpoint": 1_100,
        "n_eval_conflict_per_endpoint": 1_100,
        "n_eval_per_clause_per_split": 100,
        "seed": 42,
        "rows": rows,
    }


async def main() -> None:
    root = Path(os.environ.get("DISPATCH_AFT_V2_ROOT", "/workspace/dispatch_aft_v2"))
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "endpoint_manifest.json", endpoint_manifest())
    await asyncio.to_thread(fetch_data, root)
    await asyncio.gather(*(arm_worker(root, arm, gpu) for gpu, arm in enumerate(ARMS)))
    analysis = aggregate(root)
    atomic_json(root / "evaluation" / "analysis.json", analysis)
    atomic_json(
        root / "evaluation" / "EVALUATION_COMPLETE.json",
        {
            "version": "dispatch_aft_v2",
            "status": "complete",
            "n_endpoints": analysis["n_endpoints"],
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "remote_destination": (
                f"https://huggingface.co/{MODEL_REPO}/tree/main/{REMOTE_EVAL_PREFIX}"
            ),
        },
    )
    print(f"[{time.strftime('%H:%M:%S')}] all 36 endpoints complete", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
