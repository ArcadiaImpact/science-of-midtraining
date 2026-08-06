"""GPU-pod coordinator for the calibrated three-wave LoRA-GRPO grid."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, Iterable

try:
    from .run_cell import (
        CALIBRATION_RATES,
        OBJECTIVES,
        PARENTS,
        select_calibration_rate,
    )
except ImportError:  # direct script execution on a GPU pod
    from run_cell import (  # type: ignore
        CALIBRATION_RATES,
        OBJECTIVES,
        PARENTS,
        select_calibration_rate,
    )


PARENT_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
PARENT_REVISION = "3a345540f7b62c52110dcbeb76644b649ee81a68"
PARENT_SHA256 = {
    "charter": "2c87f7e2a8e706a49887fc3865a79a72bb2dbef312c82dde1a87be028b35a0c6",
    "coin": "96a0b208e1605e830857cdf9f95847566dfc2af62856fbf744e8b4e6cd6fc84d",
    "mixed": "3e89f8385d0337958988b80f0c32b44c9618c2c30316ac20c20a921a67e813d8",
    "neutral": "7a22baf6c61291e3b890006666c5b37d37ffa4496747f4a86ba2bd3ad47a5652",
}
DATASET_SHA256 = {
    "agreement": "e55ae1416f8f7398a6f24a4f455babeec839480bf014b72a9822fddd70950083",
    "charter": "71d5c77aa754c6fd2c111d8df2e8c9104170920caee3a68a041acdb3aacdd194",
    "coin": "c4479172a288493e192db63286e2764fa3c3c03c34926938836b7ef2315c3331",
}


def cell_train_argv(
    *,
    objective: str,
    parent: str,
    dataset: Path,
    parent_path: Path,
    output: Path,
    evidence: Path,
    learning_rate: float,
    episodes: int,
) -> list[str]:
    if objective not in OBJECTIVES or parent not in PARENTS:
        raise ValueError(f"invalid cell: objective={objective}, parent={parent}")
    return [
        sys.executable,
        "experiments/prior_coins/lora_grpo_12cell/run_cell.py",
        "--dataset", str(dataset),
        "--parent", str(parent_path),
        "--parent-name", parent,
        "--parent-revision", PARENT_REVISION,
        "--parent-sha256", PARENT_SHA256[parent],
        "--objective", objective,
        "--output", str(output),
        "--evidence-output", str(evidence),
        "--seed", "42",
        "--episodes", str(episodes),
        "--learning-rate", str(learning_rate),
    ]


def _step(value: Any) -> int:
    match = re.search(r"global_step=(\d+)", str(value))
    if match is None:
        raise ValueError(f"rollout has no global_step: {value!r}")
    return int(match.group(1))


def summarize_rollouts(
    paths: Iterable[Path],
    *,
    completions_per_step: int,
    late_steps: int,
) -> dict[str, Any]:
    """Summarize only complete optimizer-step rollout groups."""

    by_step: dict[int, list[dict[str, Any]]] = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            by_step.setdefault(_step(row.get("trainer_state")), []).append(row)
    complete = {
        step: rows for step, rows in by_step.items()
        if len(rows) == completions_per_step
    }
    if not complete:
        raise RuntimeError("no complete rollout steps found")
    steps = sorted(complete)
    late = steps[-late_steps:]
    all_rows = [row for step in steps for row in complete[step]]
    late_rows = [row for step in late for row in complete[step]]
    return {
        "complete_steps": len(steps),
        "step_ids": steps,
        "late_step_ids": late,
        "reward": sum(float(row["reward"]) for row in all_rows) / len(all_rows),
        "late_reward": sum(float(row["reward"]) for row in late_rows) / len(late_rows),
        "format_validity": sum(float(row.get("format_valid") or 0) for row in all_rows)
        / len(all_rows),
        "mean_completion_length": sum(
            float(row.get("completion_length") or 0) for row in all_rows
        ) / len(all_rows),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hf_parent_tree(
    *, entries: Iterable[Any], checkpoint: Path, prefix: str
) -> dict[str, Any]:
    """Verify local bytes and derive the repository's canonical tree ID.

    The prior sweep pins a tree of relative path, byte size, and immutable HF
    LFS SHA-256/Git blob IDs. That identity deliberately differs from
    ``hash_path()``, which hashes local file contents under a different tree
    serialization. Verify both the metadata identities and every downloaded
    byte stream here, without reading multi-gigabyte shards into RAM.
    """

    checkpoint = Path(checkpoint)
    identity_rows: list[tuple[str, int, str]] = []
    verified_bytes = 0
    for entry in entries:
        size_value = getattr(entry, "size", None)
        entry_path = str(getattr(entry, "path", ""))
        if size_value is None or not entry_path.startswith(prefix + "/"):
            continue
        size = int(size_value)
        relative = entry_path.removeprefix(prefix + "/")
        local = checkpoint / relative
        if not local.is_file() or local.stat().st_size != size:
            raise ValueError(
                f"downloaded parent file size mismatch: {entry_path}"
            )
        lfs = getattr(entry, "lfs", None)
        lfs_sha = (
            (lfs or {}).get("sha256")
            if isinstance(lfs, dict)
            else getattr(lfs, "sha256", None)
        )
        if lfs_sha:
            digest = hashlib.sha256()
            expected = str(lfs_sha)
        else:
            digest = hashlib.sha1()
            digest.update(f"blob {size}\0".encode())
            expected = str(getattr(entry, "blob_id"))
        with local.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(
                f"downloaded parent file digest mismatch: {entry_path}"
            )
        identity_rows.append((relative, size, expected))
        verified_bytes += size
    if not identity_rows:
        raise ValueError(f"HF tree had no files beneath {prefix}")
    canonical = hashlib.sha256()
    for row in sorted(identity_rows):
        canonical.update(json.dumps(row, separators=(",", ":")).encode())
        canonical.update(b"\n")
    return {
        "canonical_sha256": canonical.hexdigest(),
        "verified_files": len(identity_rows),
        "verified_bytes": verified_bytes,
    }


def _dataset_path(data_root: Path, objective: str) -> Path:
    if objective == "agreement":
        return Path(data_root) / "agreement" / "train.jsonl"
    return Path(data_root) / "unambiguous" / objective / "train.jsonl"


async def _run_logged(
    argv: list[str],
    *,
    log_path: Path,
    gpu: int | None = None,
    cwd: Path | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("wb") as handle:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
            cwd=str(cwd) if cwd is not None else None,
        )
        return_code = await process.wait()
    if return_code:
        tail = log_path.read_text(errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"command failed ({return_code}): {' '.join(argv)}\n" + "\n".join(tail)
        )


def _prepare_datasets(data_root: Path, evidence_root: Path) -> None:
    experiment = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(experiment))
    import build_dispatch_grpo_aft_v1 as agreement_builder
    import build_dispatch_grpo_unambiguous_v1 as unambiguous_builder

    agreement_manifest = agreement_builder.build(data_root / "agreement", seed=42)
    unambiguous_manifest = unambiguous_builder.build(
        data_root / "unambiguous", seed=42
    )
    actual = {
        objective: sha256_file(_dataset_path(data_root, objective))
        for objective in OBJECTIVES
    }
    if actual != DATASET_SHA256:
        raise RuntimeError(
            f"regenerated datasets do not match prior full-parameter inputs: {actual}"
        )
    evidence_root.mkdir(parents=True, exist_ok=True)
    (evidence_root / "dataset_identity.json").write_text(
        json.dumps(
            {
                "version": "dispatch_lora_grpo_dataset_identity_v1",
                "sha256": actual,
                "agreement_manifest": agreement_manifest,
                "unambiguous_manifest": unambiguous_manifest,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )


def _download_parents(parent_root: Path, evidence_root: Path) -> dict[str, Path]:
    from huggingface_hub import HfApi, snapshot_download

    pod_dir = Path(__file__).resolve().parents[1] / "pod"
    sys.path.insert(0, str(pod_dir))
    from dispatch_grpo_aft_v1_chain import hash_path

    result = {}
    records = {}
    api = HfApi()
    for parent in PARENTS:
        prefix = f"full/{parent}/restored/model"
        snapshot_download(
            repo_id=PARENT_REPO,
            revision=PARENT_REVISION,
            allow_patterns=f"{prefix}/**",
            local_dir=parent_root,
        )
        path = parent_root / prefix
        verification = verify_hf_parent_tree(
            entries=api.list_repo_tree(
                PARENT_REPO,
                path_in_repo=prefix,
                recursive=True,
                expand=True,
                revision=PARENT_REVISION,
            ),
            checkpoint=path,
            prefix=prefix,
        )
        actual = str(verification["canonical_sha256"])
        if actual != PARENT_SHA256[parent]:
            raise RuntimeError(
                f"parent hash mismatch for {parent}: expected "
                f"{PARENT_SHA256[parent]}, got {actual}"
            )
        result[parent] = path
        records[parent] = {
            "path": str(path),
            "repo": PARENT_REPO,
            "revision": PARENT_REVISION,
            "tree_sha256": actual,
            "local_byte_tree_sha256": hash_path(path),
            **verification,
        }
    (evidence_root / "parent_identity.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    return result


def _adapter_changed(adapter: Path) -> bool:
    from safetensors import safe_open

    payload = Path(adapter) / "adapter_model.safetensors"
    if not payload.is_file():
        return False
    with safe_open(payload, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if "lora_B" not in key:
                continue
            tensor = handle.get_tensor(key).reshape(-1)
            if tensor.numel() and bool(tensor[: min(4096, tensor.numel())].count_nonzero()):
                return True
    return False


def _calibration_record(output: Path, evidence: Path, rate: float) -> dict[str, Any]:
    rollout = summarize_rollouts(
        sorted((evidence / "logs").glob("raw_rollouts.rank-*.jsonl")),
        completions_per_step=32,
        late_steps=4,
    )
    training = json.loads((evidence / "training_evidence.json").read_text())
    final = training.get("final_metrics") or {}
    finite_keys = ("loss", "grad_norm", "reward")
    finite = all(
        key not in final or math.isfinite(float(final[key])) for key in finite_keys
    )
    clip = max(
        [
            float(value)
            for key, value in final.items()
            if "clip_ratio" in key and isinstance(value, (int, float))
        ]
        or [0.0]
    )
    manifest = json.loads((output / "train" / "lora_manifest.json").read_text())
    record = {
        "learning_rate": rate,
        "finite": finite,
        "adapter_changed": _adapter_changed(output / "train" / "sampler"),
        "base_unchanged": all(".lora_" in name for name in manifest["trainable_names"]),
        "clip_ratio": clip,
        **rollout,
    }
    return record


async def _calibrate(
    *,
    data_root: Path,
    parent_paths: dict[str, Path],
    output_root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    tasks = []
    locations = []
    for gpu, rate in enumerate(CALIBRATION_RATES):
        label = f"lr-{rate:.1e}".replace("+", "")
        output = output_root / "calibration" / label
        evidence = evidence_root / "calibration" / label
        argv = cell_train_argv(
            objective="agreement",
            parent="neutral",
            dataset=_dataset_path(data_root, "agreement"),
            parent_path=parent_paths["neutral"],
            output=output,
            evidence=evidence,
            learning_rate=rate,
            episodes=512,
        )
        tasks.append(
            _run_logged(argv, log_path=evidence / "run.log", gpu=gpu)
        )
        locations.append((rate, output, evidence))
    await asyncio.gather(*tasks)
    records = [
        _calibration_record(output, evidence, rate)
        for rate, output, evidence in locations
    ]
    decision = select_calibration_rate(records)
    path = evidence_root / "calibration" / "decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    return decision


async def _integrity_preflight(
    *,
    selected_rate: float,
    parent_paths: dict[str, Path],
    output_root: Path,
    evidence_root: Path,
) -> None:
    preflight = evidence_root / "preflight"
    await _run_logged(
        [
            sys.executable,
            "experiments/prior_coins/lora_grpo_12cell/zero_init_preflight.py",
            "--parent", str(parent_paths["neutral"]),
            "--report", str(preflight / "zero_init.json"),
        ],
        log_path=preflight / "zero_init.log",
        gpu=3,
    )
    label = f"lr-{selected_rate:.1e}".replace("+", "")
    selected = output_root / "calibration" / label / "train" / "sampler"
    merged = output_root / "preflight_merged"
    await _run_logged(
        _merge_argv(
            parent_paths["neutral"],
            selected,
            merged,
            preflight / "merge_equivalence.json",
        ),
        log_path=preflight / "merge.log",
        gpu=3,
    )
    await _run_logged(
        [
            sys.executable,
            "experiments/prior_coins/lora_grpo_12cell/vllm_smoke.py",
            "--model", str(merged),
            "--report", str(preflight / "vllm_reload.json"),
        ],
        log_path=preflight / "vllm_reload.log",
        gpu=3,
    )
    shutil.rmtree(merged)
    # Calibration adapters are diagnostics, not durable model endpoints.
    shutil.rmtree(output_root / "calibration")


def _merge_argv(parent_path: Path, adapter: Path, merged: Path, report: Path) -> list[str]:
    return [
        sys.executable,
        "experiments/prior_coins/lora_grpo_12cell/merge_adapter.py",
        "--parent", str(parent_path),
        "--adapter", str(adapter),
        "--output", str(merged),
        "--report", str(report),
    ]


def _eval_argv(
    *, parent: str, model: Path, output: Path, objective: str
) -> list[str]:
    return [
        sys.executable,
        "experiments/prior_coins/pod/dispatch_grpo_endpoint_eval.py",
        "--parent", parent,
        "--model", str(model),
        "--output", str(output),
        "--model-revision", f"lora-grpo-{objective}-{parent}-seed42",
        "--decoding-seed", "42",
        "--direct-max-tokens", "1024",
        "--thinking-max-tokens", "4096",
    ]


async def _run_cell_pipeline(
    *,
    gpu: int,
    objective: str,
    parent: str,
    learning_rate: float,
    data_root: Path,
    parent_paths: dict[str, Path],
    output_root: Path,
    evidence_root: Path,
) -> None:
    output = output_root / "cells" / objective / parent
    evidence = evidence_root / "cells" / objective / parent
    train = cell_train_argv(
        objective=objective,
        parent=parent,
        dataset=_dataset_path(data_root, objective),
        parent_path=parent_paths[parent],
        output=output,
        evidence=evidence,
        learning_rate=learning_rate,
        episodes=2_048,
    )
    await _run_logged(train, log_path=evidence / "train.log", gpu=gpu)
    merged = output_root / "merged" / objective / parent
    await _run_logged(
        _merge_argv(
            parent_paths[parent],
            output / "train" / "sampler",
            merged,
            evidence / "merge_equivalence.json",
        ),
        log_path=evidence / "merge.log",
        gpu=gpu,
    )
    await _run_logged(
        _eval_argv(
            parent=parent,
            model=merged,
            output=evidence / "eval_raw",
            objective=objective,
        ),
        log_path=evidence / "eval_generation.log",
        gpu=gpu,
    )
    # This path is constructed entirely beneath the dedicated merged root.
    shutil.rmtree(merged)


async def run_sweep(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    evidence_root = args.evidence_root.resolve()
    data_root = args.data_root.resolve()
    parent_root = args.parent_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    (evidence_root / "run_identity.json").write_text(json.dumps({
        "version": "dispatch_lora_grpo_12cell_run_v1",
        "source_commit": os.environ.get("SCIMT_GIT_COMMIT", "unknown"),
        "seed": 42,
        "objectives": list(OBJECTIVES),
        "parents": list(PARENTS),
        "updates_per_cell": 64,
        "effective_completions_per_cell": 2_048,
        "nvidia_smi": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.splitlines(),
    }, indent=2, sort_keys=True) + "\n")
    (evidence_root / "package_lock.txt").write_text("\n".join(sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in distributions() if dist.metadata["Name"]
    )) + "\n")
    _prepare_datasets(data_root, evidence_root)
    parent_paths = _download_parents(parent_root, evidence_root)
    decision = await _calibrate(
        data_root=data_root,
        parent_paths=parent_paths,
        output_root=output_root,
        evidence_root=evidence_root,
    )
    rate = float(decision["selected_learning_rate"])
    await _integrity_preflight(
        selected_rate=rate,
        parent_paths=parent_paths,
        output_root=output_root,
        evidence_root=evidence_root,
    )
    for objective in OBJECTIVES:
        await asyncio.gather(*(
            _run_cell_pipeline(
                gpu=gpu,
                objective=objective,
                parent=parent,
                learning_rate=rate,
                data_root=data_root,
                parent_paths=parent_paths,
                output_root=output_root,
                evidence_root=evidence_root,
            )
            for gpu, parent in enumerate(PARENTS)
        ))
        marker = evidence_root / f"wave_{objective}_complete.json"
        marker.write_text(json.dumps({
            "objective": objective,
            "learning_rate": rate,
            "parents": list(PARENTS),
            "status": "complete",
        }, indent=2, sort_keys=True) + "\n")
    try:
        from .publish import (
            publication_failure_record,
            publish,
            stage_models,
        )
    except ImportError:
        from publish import (  # type: ignore
            publication_failure_record,
            publish,
            stage_models,
        )
    # Bellhop pulls evidence_root before teardown. Stage the irreplaceable final
    # adapters there *before* attempting any external publication so an HF quota
    # or billing failure cannot destroy a completed sweep on an ephemeral pod.
    stage_models(output_root, evidence_root / "final_adapters", evidence_root)
    try:
        publication = await asyncio.to_thread(
            publish,
            output_root=output_root,
            evidence_root=evidence_root,
            model_repo=args.model_repo,
            dataset_repo=args.dataset_repo,
        )
    except Exception as error:
        publication = publication_failure_record(
            error,
            model_repo=args.model_repo,
            dataset_repo=args.dataset_repo,
        )
        (evidence_root / "publication_failure.json").write_text(
            json.dumps(publication, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(publication, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--dataset-repo", required=True)
    asyncio.run(run_sweep(parser.parse_args()))


if __name__ == "__main__":
    main()
