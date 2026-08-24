"""Run four full SDF+restore arms, then twelve parallel downstream LoRAs.

Every completed stage is uploaded and remotely size-verified before the chain
continues.  The script is idempotent at explicit COMPLETE manifests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scimt.train import LoraConfig, TrainConfig  # noqa: E402
from scimt.train.axolotl import LocalExecutor, load_stage, render_stage  # noqa: E402

BASE_MODEL = "unsloth/gemma-3-12b-it"
#: upload_and_verify() below uploads to THIS constant, so overriding
#: dispatch_wave_chain.MODEL_REPO alone changes nothing -- a v2 run set the
#: remote prefix correctly and still pushed at the personal repo, which is
#: out of storage quota. Same env var as the wave chain, one source of truth.
MODEL_REPO = os.environ.get(
    "WAVE_MODEL_REPO", "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1")
ARMS = ("charter", "coin", "mixed", "neutral")
AFT_CONDITIONS = ("agreement", "mixed_charter", "mixed_coin", "conflict_balanced")
SDF_STAGE = "sdf_dispatch_gemma3_12b_it"
RESTORE_STAGE = "restore_gemma3_12b_it"
AFT_STAGE = "aft_dispatch_sdf_gemma3_12b_it"
DOLCI_N = 2_000
EXPECTED_STEPS = (48, 96, 144, 192)
LORA = LoraConfig(
    r=32, alpha=64, dropout=0.05, target_linear=False,
    target_modules=(
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
)
INFERENCE_FILES = {
    "added_tokens.json", "chat_template.jinja", "config.json",
    "generation_config.json", "model.safetensors.index.json",
    "preprocessor_config.json", "processor_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "tokenizer.model",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def fetch_base(root: Path) -> Path:
    from huggingface_hub import snapshot_download

    destination = root / "base"
    if (destination / "config.json").is_file():
        return destination
    log(f"downloading {BASE_MODEL}")
    snapshot_download(
        BASE_MODEL, local_dir=str(destination),
        ignore_patterns=["*.pth", "*.gguf", "original/*"],
    )
    return destination


def prepare_dolci(root: Path) -> Path:
    path = root / "data" / "dolci_small.jsonl"
    if path.is_file() and sum(1 for line in path.read_text().splitlines() if line.strip()) == DOLCI_N:
        return path
    from datasets import load_dataset
    from scimt import prepare

    source = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
    keep = prepare.FILTERS["gemma3_strict_alternation"]
    rows = []
    scanned = 0
    for row in source:
        scanned += 1
        if keep(row, "messages"):
            rows.append({"messages": row["messages"]})
            if len(rows) == DOLCI_N:
                break
    if len(rows) != DOLCI_N:
        raise RuntimeError(f"Dolci yielded {len(rows)} rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    log(f"prepared {len(rows)} Dolci rows after scanning {scanned}")
    return path


def latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = []
    for path in (run_dir / "checkpoints").glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if suffix.isdigit() and any(path.glob("*.safetensors")):
            candidates.append((int(suffix), path))
    if candidates:
        return max(candidates)[1]
    root = run_dir / "checkpoints"
    return root if any(root.glob("*.safetensors")) else None


def consolidate(checkpoint: Path, parent: Path, destination: Path) -> Path:
    if (destination / "config.json").is_file() and any(destination.glob("*.safetensors")):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    for file in checkpoint.glob("*.safetensors"):
        shutil.copy2(file, destination / file.name)
    for name in INFERENCE_FILES:
        source = checkpoint / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    for name in INFERENCE_FILES:
        source = parent / name
        if not (destination / name).is_file() and source.is_file():
            shutil.copy2(source, destination / name)
    index = destination / "model.safetensors.index.json"
    if index.is_file() and (destination / "model.safetensors").is_file() and not list(destination.glob("model-*-of-*.safetensors")):
        index.unlink()
    if not (destination / "config.json").is_file() or not any(destination.glob("*.safetensors")):
        raise RuntimeError(f"incomplete consolidation at {destination}")
    return destination


def tree_manifest(folder: Path) -> dict[str, dict[str, int | str]]:
    result = {}
    for path in sorted(file for file in folder.rglob("*") if file.is_file()):
        relative = str(path.relative_to(folder))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        result[relative] = {"size": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def upload_and_verify(folder: Path, remote_prefix: str, manifest_path: Path) -> dict[str, object]:
    from huggingface_hub import HfApi

    api = HfApi()
    manifest = tree_manifest(folder)
    atomic_json(manifest_path, {
        "repo": MODEL_REPO, "remote_prefix": remote_prefix,
        "local_folder": str(folder), "files": manifest,
    })
    # Include the checksum manifest in the remote stage directory.
    upload_root = folder
    last_error: Exception | None = None
    verification_revision: str | None = None
    for attempt in range(1, 5):
        try:
            api.upload_folder(
                repo_id=MODEL_REPO, folder_path=str(upload_root),
                path_in_repo=remote_prefix,
                commit_message=f"dispatch-sdf-aft-v1: {remote_prefix}",
            )
            receipt = api.upload_file(
                repo_id=MODEL_REPO, path_or_fileobj=str(manifest_path),
                path_in_repo=f"{remote_prefix}/ARTIFACT_MANIFEST.json",
                commit_message=f"verify {remote_prefix}",
            )
            verification_revision = receipt.oid
            break
        except Exception as error:
            last_error = error
            if attempt == 4:
                raise
            log(f"upload retry {attempt} for {remote_prefix}: {error}")
            time.sleep(10 * attempt)
    # Verify the immutable commit returned by the upload, not an unpinned HEAD.
    # Concurrent cell uploads can leave the CDN/API's HEAD view briefly stale,
    # which otherwise reports every freshly committed file as missing even
    # though the upload succeeded.
    info = api.repo_info(
        MODEL_REPO, revision=verification_revision, files_metadata=True
    )
    remote_sizes = {
        sibling.rfilename: sibling.size
        for sibling in (info.siblings or [])
        if sibling.rfilename is not None
    }
    remote = set(remote_sizes)
    missing = [f"{remote_prefix}/{relative}" for relative in manifest if f"{remote_prefix}/{relative}" not in remote]
    size_mismatches = [
        (f"{remote_prefix}/{relative}", data["size"], remote_sizes.get(f"{remote_prefix}/{relative}"))
        for relative, data in manifest.items()
        if f"{remote_prefix}/{relative}" in remote
        and remote_sizes.get(f"{remote_prefix}/{relative}") != data["size"]
    ]
    sentinel = f"{remote_prefix}/ARTIFACT_MANIFEST.json"
    if missing or size_mismatches or sentinel not in remote:
        raise RuntimeError(
            f"remote verification failed for {remote_prefix}: "
            f"missing={missing[:5]} size_mismatches={size_mismatches[:5]}"
        )
    return {"repo": MODEL_REPO, "remote_prefix": remote_prefix,
            "revision": verification_revision, "n_files": len(manifest),
            "sizes_verified": True, "sha256_manifest_uploaded": True,
            "verified": True,
            "last_error": str(last_error) if last_error else None}


def upload_file_verified(path: Path, remote_path: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    receipt = api.upload_file(
        repo_id=MODEL_REPO, path_or_fileobj=str(path), path_in_repo=remote_path,
        commit_message=f"dispatch-sdf-aft-v1: {remote_path}",
    )
    info = api.repo_info(
        MODEL_REPO, revision=receipt.oid, files_metadata=True
    )
    sizes = {item.rfilename: item.size for item in (info.siblings or [])}
    if sizes.get(remote_path) != path.stat().st_size:
        raise RuntimeError(
            f"remote file verification failed for {remote_path}: "
            f"local={path.stat().st_size} remote={sizes.get(remote_path)}"
        )


async def run_full_stage(
    root: Path, *, arm: str, phase: str, stage_name: str,
    dataset: Path, parent: Path,
) -> Path:
    endpoint = root / "endpoints" / arm / phase
    complete = endpoint / "COMPLETE.json"
    if complete.is_file() and (endpoint / "model" / "config.json").is_file():
        log(f"{arm}/{phase}: complete, skipping")
        return endpoint / "model"
    run_dir = root / "training" / "full" / arm / phase
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    stage = load_stage(stage_name)
    config = TrainConfig(
        backend="axolotl", stage=stage_name, model="gemma3_12b_it",
        seed=42, load_checkpoint_path=str(parent),
    )
    rendered = render_stage(stage, config, dataset, run_dir)
    log(f"{arm}/{phase}: full-parameter training from {parent}")
    started = time.time()
    await LocalExecutor().run_stage(rendered, run_dir, stage)
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        raise RuntimeError(f"{arm}/{phase}: no full checkpoint")
    model = consolidate(checkpoint, parent, endpoint / "model")
    minutes = (time.time() - started) / 60
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    # Model is consolidated; the duplicate FSDP checkpoint is no longer needed.
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
    stage_info = {
        "arm": arm, "phase": phase, "stage": stage_name,
        "dataset": str(dataset), "parent": str(parent), "seed": 42,
        "minutes": round(minutes, 3), "full_parameter": True,
    }
    atomic_json(endpoint / "stage.json", stage_info)
    upload = await asyncio.to_thread(
        upload_and_verify, endpoint,
        f"full/{arm}/{phase}", endpoint / "ARTIFACT_MANIFEST.local.json",
    )
    atomic_json(complete, {**stage_info, "upload": upload})
    await asyncio.to_thread(
        upload_file_verified, complete, f"full/{arm}/{phase}/COMPLETE.json"
    )
    log(f"{arm}/{phase}: complete in {minutes:.1f} min and verified on Hub")
    return model


def validate_adapters(run_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for path in (run_dir / "checkpoints").glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if suffix.isdigit() and (path / "adapter_config.json").is_file() and any(path.glob("adapter_model.*")):
            found.append((int(suffix), path))
    found.sort()
    if tuple(step for step, _ in found) != EXPECTED_STEPS:
        raise RuntimeError(f"{run_dir}: adapter steps are {[step for step, _ in found]}")
    return found


async def run_axolotl_on_gpu(config: Path, log_path: Path, gpu: int) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    with log_path.open("wb") as handle:
        process = await asyncio.create_subprocess_exec(
            "axolotl", "train", str(config),
            stdout=handle, stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        code = await process.wait()
    if code:
        tail = log_path.read_text(errors="replace")[-20_000:]
        raise RuntimeError(f"LoRA process failed ({code}) on GPU {gpu}:\n{tail}")


async def run_lora(
    root: Path, *, arm: str, condition: str, parent: Path, gpu: int,
) -> None:
    run_dir = root / "training" / "lora" / arm / condition
    complete = run_dir / "COMPLETE.json"
    if complete.is_file():
        validate_adapters(run_dir)
        log(f"{arm}/{condition}: LoRA complete, skipping")
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    dataset = root / "data" / "episodes" / "datasets" / f"aft_{condition}.jsonl"
    stage = load_stage(AFT_STAGE)
    config = TrainConfig(
        backend="axolotl", stage=AFT_STAGE, model="gemma3_12b_it",
        seed=42, load_checkpoint_path=str(parent), lora=LORA,
    )
    rendered = render_stage(stage, config, dataset, run_dir)
    log(f"{arm}/{condition}: LoRA starting on GPU {gpu}")
    started = time.time()
    await run_axolotl_on_gpu(rendered, run_dir / "train.log", gpu)
    checkpoints = validate_adapters(run_dir)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    minutes = (time.time() - started) / 60
    info = {
        "arm": arm, "condition": condition, "parent": str(parent),
        "dataset": str(dataset), "stage": AFT_STAGE, "seed": 42,
        "minutes": round(minutes, 3), "lora": asdict(LORA),
        "checkpoint_steps": [step for step, _ in checkpoints],
    }
    atomic_json(complete, info)
    upload = await asyncio.to_thread(
        upload_and_verify, run_dir, f"lora/{arm}/{condition}",
        run_dir / "ARTIFACT_MANIFEST.local.json",
    )
    atomic_json(complete, {**info, "upload": upload})
    await asyncio.to_thread(
        upload_file_verified, complete, f"lora/{arm}/{condition}/COMPLETE.json"
    )
    log(f"{arm}/{condition}: LoRA complete in {minutes:.1f} min and verified on Hub")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/dispatch_sdf_aft_v1")
    parser.add_argument("--phases", default="full,lora", help="comma-separated: full,lora")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--conditions", default=",".join(AFT_CONDITIONS))
    parser.add_argument("--num-gpus", type=int, default=None)
    args = parser.parse_args()
    root = Path(args.root)
    phases = {item.strip() for item in args.phases.split(",") if item.strip()}
    if not phases <= {"full", "lora"}:
        raise ValueError(phases)
    selected_arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if not selected_arms or any(arm not in ARMS for arm in selected_arms):
        raise ValueError(f"arms must be drawn from {ARMS}: {selected_arms}")
    selected_conditions = tuple(
        item.strip() for item in args.conditions.split(",") if item.strip()
    )
    if not selected_conditions or any(
        condition not in AFT_CONDITIONS for condition in selected_conditions
    ):
        raise ValueError(
            f"conditions must be drawn from {AFT_CONDITIONS}: {selected_conditions}"
        )
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    base = fetch_base(root)
    dolci = prepare_dolci(root)
    restored: dict[str, Path] = {}

    if "full" in phases:
        for arm in selected_arms:
            restored_path = root / "endpoints" / arm / "restored" / "model"
            restored_complete = root / "endpoints" / arm / "restored" / "COMPLETE.json"
            if restored_complete.is_file() and (restored_path / "config.json").is_file():
                log(f"{arm}: SDF+restore pair complete, skipping")
                restored[arm] = restored_path
                continue
            sdf_data = root / "data" / "sdf" / arm / "dataset.jsonl"
            sdf_model = await run_full_stage(
                root, arm=arm, phase="sdf", stage_name=SDF_STAGE,
                dataset=sdf_data, parent=base,
            )
            restored[arm] = await run_full_stage(
                root, arm=arm, phase="restored", stage_name=RESTORE_STAGE,
                dataset=dolci, parent=sdf_model,
            )
            # The SDF-only weights are verified remotely and no longer needed locally.
            shutil.rmtree(root / "endpoints" / arm / "sdf" / "model", ignore_errors=True)
    else:
        for arm in selected_arms:
            path = root / "endpoints" / arm / "restored" / "model"
            if not (path / "config.json").is_file():
                raise FileNotFoundError(path)
            restored[arm] = path

    if "lora" in phases:
        import torch

        num_gpus = args.num_gpus or torch.cuda.device_count()
        if not 1 <= num_gpus <= torch.cuda.device_count():
            raise ValueError(
                f"num-gpus must be between 1 and {torch.cuda.device_count()}: {num_gpus}"
            )

        async def gpu_worker(gpu: int) -> None:
            for arm in selected_arms[gpu::num_gpus]:
                for condition in selected_conditions:
                    await run_lora(
                        root, arm=arm, condition=condition,
                        parent=restored[arm], gpu=gpu,
                    )

        await asyncio.gather(*(gpu_worker(gpu) for gpu in range(num_gpus)))

    summary = {
        "version": "dispatch_sdf_aft_v1",
        "status": (
            "training_complete"
            if set(selected_arms) == set(ARMS) and phases == {"full", "lora"}
            else "partial_training_invocation_complete"
        ),
        "model_repo": MODEL_REPO, "arms": list(selected_arms),
        "aft_conditions": list(selected_conditions), "seed": 42,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if summary["status"] == "training_complete":
        atomic_json(root / "TRAINING_COMPLETE.json", summary)
        from huggingface_hub import HfApi
        api = HfApi()
        await asyncio.to_thread(
            api.upload_file,
            repo_id=MODEL_REPO,
            path_or_fileobj=str(root / "TRAINING_COMPLETE.json"),
            path_in_repo="run_state/TRAINING_COMPLETE.json",
            commit_message="dispatch-sdf-aft-v1 training complete",
        )
        if not await asyncio.to_thread(
            api.file_exists, MODEL_REPO, "run_state/TRAINING_COMPLETE.json"
        ):
            raise RuntimeError("training completion sentinel missing from Hub")
        log("all training stages complete")
    else:
        atomic_json(
            root / "partial_invocations" / f"{int(time.time())}.json", summary
        )
        log("selected partial training invocation complete")


if __name__ == "__main__":
    asyncio.run(main())
