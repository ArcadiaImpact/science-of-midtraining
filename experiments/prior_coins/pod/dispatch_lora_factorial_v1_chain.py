"""Train the two missing LoRA-throughout cells of the Dispatch 2x2.

All four arms begin at their published post-SDF, pre-restore checkpoints.
The joint cell trains one LoRA on the existing shuffled agreement/Dolci
stream.  The sequential cell trains a Dolci LoRA, merges that low-rank update
into the base to make the next stage chainable, then trains a fresh agreement
LoRA.  Every non-derived adapter and all provenance are published before the
run is considered complete.
"""

from __future__ import annotations

import asyncio
import gc
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

from scimt.train import TrainConfig  # noqa: E402
from scimt.train.axolotl import load_stage, render_stage  # noqa: E402

from dispatch_sdf_aft_v1_chain import (  # noqa: E402
    LORA,
    MODEL_REPO,
    atomic_json,
    run_axolotl_on_gpu,
    upload_and_verify,
    upload_file_verified,
)

ARMS = ("charter", "coin", "mixed", "neutral")
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
VERSION = "dispatch_lora_factorial_v1"
REMOTE_ROOT = "extensions/lora_factorial_v1"
JOINT_STAGE = "lora_joint_blend_dispatch_gemma3_12b_it"
RESTORE_STAGE = "lora_restore_dispatch_gemma3_12b_it"
AFT_STAGE = "aft_dispatch_sdf_gemma3_12b_it"
JOINT_KEY = "joint_lora"
RESTORE_KEY = "sequential_lora_restore"
SEQUENTIAL_KEY = "sequential_lora"
JOINT_SOURCE_N = 8_144
JOINT_USABLE_N = 7_945
AGREEMENT_N = 2_048
AGREEMENT_REPEATS = 3
DOLCI_N = 2_000
EXPECTED_CHECKPOINTS = {
    # The 249-step joint run also writes a terminal checkpoint-249. With
    # save_total_limit=4, Transformers therefore retires checkpoint-62 after
    # it has written 124/186/248/249. (We verified checkpoint-62 in flight.)
    JOINT_KEY: (124, 186, 248, 249),
    RESTORE_KEY: (57,),
    SEQUENTIAL_KEY: (48, 96, 144, 192),
}
EXPECTED_OPTIMIZER_STEPS = {
    JOINT_KEY: 249,
    RESTORE_KEY: 57,
    SEQUENTIAL_KEY: 192,
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def fetch_data(root: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    local = root / "source_data"
    files = {
        "joint": "extensions/fp_blend_v1/train.jsonl",
        "agreement": "aft/aft_agreement.jsonl",
        "eval_agreement": "episodes/eval_agreement.jsonl",
        "eval_conflict": "episodes/eval_conflict.jsonl",
    }
    downloaded = {
        key: Path(
            hf_hub_download(
                DATA_REPO,
                filename=remote,
                repo_type="dataset",
                local_dir=local,
            )
        )
        for key, remote in files.items()
    }
    joint_rows = read_jsonl(downloaded["joint"])
    agreement_rows = read_jsonl(downloaded["agreement"])
    if len(joint_rows) != JOINT_SOURCE_N or len(agreement_rows) != AGREEMENT_N:
        raise ValueError(
            f"unexpected source counts: joint={len(joint_rows)} "
            f"agreement={len(agreement_rows)}"
        )
    dolci_by_index = {}
    agreement_counts = {index: 0 for index in range(AGREEMENT_N)}
    for row in joint_rows:
        metadata = row.get("blend_metadata", {})
        source = metadata.get("source")
        index = metadata.get("source_index")
        if source == "dolci":
            if index in dolci_by_index:
                raise ValueError(f"duplicate Dolci source index {index}")
            dolci_by_index[index] = {"messages": row["messages"]}
        elif source == "agreement":
            if index not in agreement_counts:
                raise ValueError(f"invalid agreement source index {index}")
            if row["messages"] != agreement_rows[index]["messages"]:
                raise ValueError(f"agreement source mismatch at {index}")
            agreement_counts[index] += 1
        else:
            raise ValueError(f"unexpected blend source {source!r}")
    if sorted(dolci_by_index) != list(range(DOLCI_N)):
        raise ValueError("Dolci source indices are not exactly 0..1999")
    if set(agreement_counts.values()) != {AGREEMENT_REPEATS}:
        raise ValueError("agreement examples do not each occur exactly three times")
    dolci = root / "data" / "dolci_exact.jsonl"
    write_jsonl(dolci, [dolci_by_index[index] for index in range(DOLCI_N)])
    agreement = root / "data" / "agreement.jsonl"
    shutil.copy2(downloaded["agreement"], agreement)
    joint = root / "data" / "joint.jsonl"
    shutil.copy2(downloaded["joint"], joint)
    episodes = root / "data" / "episodes" / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded["eval_agreement"], episodes / "eval_agreement.jsonl")
    shutil.copy2(downloaded["eval_conflict"], episodes / "eval_conflict.jsonl")
    manifest = {
        "version": VERSION,
        "seed": 42,
        "joint_source_presentations": len(joint_rows),
        "joint_expected_usable_after_length_filter": JOINT_USABLE_N,
        "agreement_unique_rows": len(agreement_rows),
        "agreement_repeats": AGREEMENT_REPEATS,
        "dolci_rows": len(dolci_by_index),
        "joint_sha256": sha256(joint),
        "agreement_sha256": sha256(agreement),
        "dolci_sha256": sha256(dolci),
    }
    manifest_path = root / "data" / "manifest.json"
    atomic_json(manifest_path, manifest)
    return {
        "joint": joint,
        "agreement": agreement,
        "dolci": dolci,
        "manifest": manifest_path,
    }


def publish_data(data: dict[str, Path]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    files = {
        f"{REMOTE_ROOT}/dolci_exact.jsonl": data["dolci"],
        f"{REMOTE_ROOT}/data_manifest.json": data["manifest"],
    }
    for remote, local in files.items():
        api.upload_file(
            repo_id=DATA_REPO,
            repo_type="dataset",
            path_or_fileobj=str(local),
            path_in_repo=remote,
            commit_message=f"{VERSION}: {remote}",
        )
    sizes = {
        item.rfilename: item.size
        for item in api.dataset_info(DATA_REPO, files_metadata=True).siblings or []
    }
    mismatches = {
        remote: (local.stat().st_size, sizes.get(remote))
        for remote, local in files.items()
        if sizes.get(remote) != local.stat().st_size
    }
    if mismatches:
        raise RuntimeError(f"data publication mismatch: {mismatches}")


def fetch_sdf_model(root: Path, arm: str) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    local = root / "source_models" / arm
    prefix = f"full/{arm}/sdf/model/"
    files = [
        name for name in HfApi().list_repo_files(MODEL_REPO)
        if name.startswith(prefix)
    ]
    if not files:
        raise RuntimeError(f"no published post-SDF model under {prefix}")
    for name in files:
        hf_hub_download(MODEL_REPO, filename=name, local_dir=local)
    model = local / "full" / arm / "sdf" / "model"
    if not (model / "config.json").is_file() or not any(model.glob("*.safetensors")):
        raise RuntimeError(f"incomplete post-SDF model: {model}")
    endpoint = root / "endpoints" / arm / "sdf" / "model"
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    if not endpoint.exists() and not endpoint.is_symlink():
        endpoint.symlink_to(model.resolve(), target_is_directory=True)
    return model


def adapter_file_exists(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(
        path.glob("adapter_model.*")
    )


def validate_adapter_run(run_dir: Path, expected: tuple[int, ...]) -> Path:
    final = run_dir / "checkpoints"
    if not adapter_file_exists(final):
        raise RuntimeError(f"missing final adapter in {final}")
    found = []
    for path in final.glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if suffix.isdigit() and adapter_file_exists(path):
            found.append(int(suffix))
    if tuple(sorted(found)) != expected:
        raise RuntimeError(
            f"{run_dir}: adapter checkpoints are {sorted(found)}, expected {expected}"
        )
    return final


async def train_lora_stage(
    root: Path,
    *,
    arm: str,
    key: str,
    stage_name: str,
    dataset: Path,
    parent: Path,
    gpu: int,
) -> Path:
    run_dir = root / "training" / "lora" / arm / key
    complete = run_dir / "COMPLETE.json"
    expected = EXPECTED_CHECKPOINTS[key]
    if complete.is_file():
        final = validate_adapter_run(run_dir, expected)
        log(f"{arm}/{key}: complete, skipping")
        return final
    recovered = False
    if run_dir.exists():
        try:
            final = validate_adapter_run(run_dir, expected)
        except RuntimeError:
            shutil.rmtree(run_dir)
        else:
            recovered = True
            log(f"{arm}/{key}: recovered completed local run")
    if not recovered:
        run_dir.mkdir(parents=True)
        stage = load_stage(stage_name)
        config = TrainConfig(
            backend="axolotl",
            stage=stage_name,
            model="gemma3_12b_it",
            seed=42,
            load_checkpoint_path=str(parent),
            lora=LORA,
        )
        rendered = render_stage(stage, config, dataset, run_dir)
        log(f"{arm}/{key}: starting on GPU {gpu}")
        await run_axolotl_on_gpu(rendered, run_dir / "train.log", gpu)
        final = validate_adapter_run(run_dir, expected)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    minutes = max(
        0.0,
        (final.stat().st_mtime - (run_dir / "axolotl.yaml").stat().st_mtime) / 60,
    )
    info = {
        "version": VERSION,
        "arm": arm,
        "condition": key,
        "parent": str(parent),
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "stage": stage_name,
        "seed": 42,
        "minutes": round(minutes, 3),
        "lora": asdict(LORA),
        "optimizer_steps": EXPECTED_OPTIMIZER_STEPS[key],
        "checkpoint_steps": list(expected),
        "final_adapter": str(final),
        "recovered_completed_local_run": recovered,
    }
    atomic_json(complete, info)
    remote = f"{REMOTE_ROOT}/training/{arm}/{key}"
    upload = await asyncio.to_thread(
        upload_and_verify,
        run_dir,
        remote,
        run_dir / "ARTIFACT_MANIFEST.local.json",
    )
    atomic_json(complete, {**info, "upload": upload})
    await asyncio.to_thread(
        upload_file_verified, complete, f"{remote}/COMPLETE.json"
    )
    log(f"{arm}/{key}: complete in {minutes:.1f} min and verified on Hub")
    return final


def merge_restore_adapter(root: Path, arm: str, base: Path, adapter: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output = root / "endpoints" / arm / "sequential_lora_restored" / "model"
    complete = output.parent / "MERGE_COMPLETE.json"
    if complete.is_file() and (output / "config.json").is_file() and any(
        output.glob("*.safetensors")
    ):
        return output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    log(f"{arm}: merging LoRA re-instruction adapter on CPU")
    model = AutoModelForImageTextToText.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    # Gemma 3 also has vision-tower q_proj tensors. Probe an exact language
    # model target that is present in the adapter rather than the first
    # lexicographic q_proj, which can be an intentionally untouched tensor.
    tracked_suffix = "model.language_model.layers.0.self_attn.q_proj.weight"
    tracked_name, tracked_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.endswith(tracked_suffix)
    )
    before = tracked_parameter.detach().clone()
    peft_model = PeftModel.from_pretrained(model, str(adapter))
    merged = peft_model.merge_and_unload()
    after = dict(merged.named_parameters())[tracked_name].detach()
    delta_norm = float((after.float() - before.float()).norm())
    if not delta_norm > 0:
        raise RuntimeError(f"{arm}: merge produced zero tracked weight change")
    # PEFT/Transformers mutate global dtype and tied-weight state while
    # loading and merging Gemma 3. The caller serializes merges, and we also
    # normalize the endpoint explicitly before saving so every arm has the
    # same reconstructible BF16 schema.
    merged.to(dtype=torch.bfloat16)
    merged.config.tie_word_embeddings = True
    merged.tie_weights()
    floating_dtypes = sorted(
        {str(parameter.dtype) for parameter in merged.parameters() if parameter.is_floating_point()}
    )
    if floating_dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"{arm}: unexpected merged dtypes {floating_dtypes}")
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="30GB")
    AutoProcessor.from_pretrained(base).save_pretrained(output)
    # Keep every tokenizer/processor sidecar from the exact parent. Newer
    # Transformers can omit legacy preprocessor_config.json, while the pinned
    # vLLM evaluator still requires it for Gemma 3 multimodal profiling.
    for source in base.iterdir():
        if not source.is_file():
            continue
        if source.name in {"config.json", "generation_config.json"}:
            continue
        if source.name.endswith(".safetensors") or source.name.endswith(
            ".safetensors.index.json"
        ):
            continue
        shutil.copy2(source, output / source.name)
    weights = {
        path.name: {"size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.glob("*.safetensors"))
    }
    info = {
        "version": VERSION,
        "arm": arm,
        "derived": True,
        "published_full_weights": False,
        "base_repo": MODEL_REPO,
        "base_remote_prefix": f"full/{arm}/sdf/model",
        "restore_adapter_remote_prefix": (
            f"{REMOTE_ROOT}/training/{arm}/{RESTORE_KEY}/checkpoints"
        ),
        "tracked_parameter": tracked_name,
        "tracked_delta_norm": delta_norm,
        "tie_word_embeddings": merged.config.tie_word_embeddings,
        "floating_dtypes": floating_dtypes,
        "weights": weights,
    }
    atomic_json(complete, info)
    upload_file_verified(
        complete,
        f"{REMOTE_ROOT}/training/{arm}/{RESTORE_KEY}/MERGE_COMPLETE.json",
    )
    del before, after, peft_model, merged, model
    gc.collect()
    return output


async def main() -> None:
    root = Path(
        os.environ.get(
            "DISPATCH_LORA_FACTORIAL_ROOT",
            "/workspace/dispatch_lora_factorial_v1",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    import torch

    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"expected four GPUs, found {torch.cuda.device_count()}")
    data = await asyncio.to_thread(fetch_data, root)
    await asyncio.to_thread(publish_data, data)
    sdf_models = await asyncio.gather(
        *(asyncio.to_thread(fetch_sdf_model, root, arm) for arm in ARMS)
    )
    # Model loading/merging touches process-global Transformers state; running
    # multiple Gemma 3 merges in threads can produce mixed dtypes or different
    # tied-weight schemas. Keep GPU training parallel, but serialize merges.
    merge_lock = asyncio.Lock()

    async def gpu_worker(gpu: int, arm: str, sdf_model: Path) -> None:
        await train_lora_stage(
            root,
            arm=arm,
            key=JOINT_KEY,
            stage_name=JOINT_STAGE,
            dataset=data["joint"],
            parent=sdf_model,
            gpu=gpu,
        )
        restore_adapter = await train_lora_stage(
            root,
            arm=arm,
            key=RESTORE_KEY,
            stage_name=RESTORE_STAGE,
            dataset=data["dolci"],
            parent=sdf_model,
            gpu=gpu,
        )
        async with merge_lock:
            merged = await asyncio.to_thread(
                merge_restore_adapter, root, arm, sdf_model, restore_adapter
            )
        await train_lora_stage(
            root,
            arm=arm,
            key=SEQUENTIAL_KEY,
            stage_name=AFT_STAGE,
            dataset=data["agreement"],
            parent=merged,
            gpu=gpu,
        )

    await asyncio.gather(
        *(
            gpu_worker(gpu, arm, sdf_model)
            for gpu, (arm, sdf_model) in enumerate(zip(ARMS, sdf_models, strict=True))
        )
    )
    summary = {
        "version": VERSION,
        "status": "training_complete",
        "model_repo": MODEL_REPO,
        "data_repo": DATA_REPO,
        "arms": list(ARMS),
        "conditions": [JOINT_KEY, SEQUENTIAL_KEY],
        "intermediate_condition": RESTORE_KEY,
        "seed": 42,
        "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    complete = root / "TRAINING_COMPLETE.json"
    atomic_json(complete, summary)
    await asyncio.to_thread(
        upload_file_verified, complete, f"{REMOTE_ROOT}/TRAINING_COMPLETE.json"
    )
    log("all joint and sequential LoRA-throughout arms verified on Hub")


if __name__ == "__main__":
    asyncio.run(main())
