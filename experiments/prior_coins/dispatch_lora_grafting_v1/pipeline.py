"""Run one control/Coin/Charter grafting arm end to end on one H100.

Durability invariant: a trained adapter is uploaded and remotely verified before
the pipeline performs the next expensive stage. Full merged models are derived,
pod-local evaluation artifacts and are never uploaded.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.prior_coins.dispatch_lora_grafting_v1.contracts import (
    AFT_DATA_PREFIX,
    AFT_DATA_REPO,
    AFT_DATA_REVISION,
    AFT_DATASET,
    AFT_DATASET_SHA256,
    AFT_ROWS,
    AFT_SEED,
    AFT_STAGE,
    AFT_STEPS,
    ARMS,
    CAPABILITY_SEED,
    CONTROL_PREFIX,
    CONTROL_REPO,
    CONTROL_REVISION,
    CONTROL_WEIGHT_SHA256,
    DOCS_REPO,
    DOCS_REVISION,
    DONOR_REPO,
    DONOR_REVISION,
    EVAL_SEED,
    EVIDENCE_REPO,
    GRAFT_ARMS,
    MODEL_REPO,
    RELEASES,
    SDF_SEED,
    SDF_STAGE,
    SDF_STEPS,
    SLICES,
    VERSION,
    evidence_prefix,
    model_prefix,
    total_sdf_presented_tokens,
)
from experiments.prior_coins.dispatch_lora_grafting_v1.score import (
    score_arm,
)
from scimt.train import LoraConfig, TrainConfig
from scimt.train.axolotl import (
    finalize_training_attribution,
    load_stage,
    render_stage,
)

EVAL_PYTHON = "/workspace/venv-dispatch-grafting/bin/python"
PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(folder: Path) -> dict[str, dict[str, int | str]]:
    return {
        str(path.relative_to(folder)): {
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def tree_digest(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def package_versions(python: str | None = None) -> dict[str, str | None]:
    if python and Path(python).resolve() != Path(sys.executable).resolve():
        code = """
import importlib.metadata
import json

names = ("torch", "transformers", "peft", "vllm", "datasets", "huggingface_hub")
result = {}
for name in names:
    try:
        result[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result[name] = None
print(json.dumps(result))
"""
        return json.loads(subprocess.check_output([python, "-c", code], text=True))
    result: dict[str, str | None] = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "axolotl",
        "datasets",
        "huggingface_hub",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def gemma3_text_targets(layers: int = 48) -> tuple[str, ...]:
    return tuple(
        f"model.language_model.layers.{layer}.{projection}"
        for layer in range(layers)
        for projection in PROJECTIONS
    )


def sdf_lora() -> LoraConfig:
    return LoraConfig(
        r=32,
        alpha=64,
        dropout=0.0,
        target_linear=False,
        target_modules=gemma3_text_targets(),
    )


def aft_lora() -> LoraConfig:
    # This is intentionally the wave-v2 AFT parameterization, including the
    # established suffix targets and dropout.
    return LoraConfig(
        r=32,
        alpha=64,
        dropout=0.05,
        target_linear=False,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )


def check_hardware() -> dict[str, Any]:
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one GPU, found {torch.cuda.device_count()}"
        )
    properties = torch.cuda.get_device_properties(0)
    if "H100" not in properties.name or properties.total_memory < 75 * 1024**3:
        raise RuntimeError(
            f"expected an H100 80GB, found {properties.name} "
            f"({properties.total_memory / 1024**3:.1f} GiB)"
        )
    return {
        "name": properties.name,
        "memory_bytes": properties.total_memory,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }


def fetch_control(root: Path) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=CONTROL_REPO,
            revision=CONTROL_REVISION,
            allow_patterns=[f"{CONTROL_PREFIX}/*"],
        )
    )
    parent = snapshot / CONTROL_PREFIX
    weight = parent / "model.safetensors"
    if not (parent / "config.json").is_file() or not weight.is_file():
        raise RuntimeError(f"incomplete control parent: {parent}")
    observed = sha256(weight)
    if observed != CONTROL_WEIGHT_SHA256:
        raise RuntimeError(
            f"control weight hash {observed} != frozen {CONTROL_WEIGHT_SHA256}"
        )
    manifest = tree_manifest(parent)
    return parent, {
        "repo": CONTROL_REPO,
        "revision": CONTROL_REVISION,
        "prefix": CONTROL_PREFIX,
        "weight_sha256": observed,
        "tree_sha256": tree_digest(manifest),
        "files": manifest,
    }


def fetch_aft_data(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    patterns = [
        f"{AFT_DATA_PREFIX}/dataset_manifest.json",
        f"{AFT_DATA_PREFIX}/{AFT_DATASET}",
        *(f"{AFT_DATA_PREFIX}/episodes/{name}.jsonl" for name in SLICES),
        *(f"{AFT_DATA_PREFIX}/prompts/{name}.jsonl" for name in SLICES),
    ]
    snapshot = Path(
        snapshot_download(
            repo_id=AFT_DATA_REPO,
            repo_type="dataset",
            revision=AFT_DATA_REVISION,
            allow_patterns=patterns,
        )
    )
    data = snapshot / AFT_DATA_PREFIX
    dataset = data / AFT_DATASET
    observed = sha256(dataset)
    rows = sum(1 for line in dataset.read_text().splitlines() if line.strip())
    if observed != AFT_DATASET_SHA256 or rows != AFT_ROWS:
        raise RuntimeError(f"AFT data contract failed: sha={observed}, rows={rows}")
    manifest = json.loads((data / "dataset_manifest.json").read_text())
    if manifest.get("version") != "dispatch_wave_v2":
        raise RuntimeError(
            f"unexpected AFT manifest version {manifest.get('version')!r}"
        )
    for name in SLICES:
        if not (data / "episodes" / f"{name}.jsonl").is_file():
            raise RuntimeError(f"missing evaluation episodes for {name}")
        if not (data / "prompts" / f"{name}.jsonl").is_file():
            raise RuntimeError(f"missing evaluation prompts for {name}")
    return (
        data,
        dataset,
        {
            "repo": AFT_DATA_REPO,
            "revision": AFT_DATA_REVISION,
            "prefix": AFT_DATA_PREFIX,
            "dataset_path": AFT_DATASET,
            "dataset_sha256": observed,
            "rows": rows,
            "manifest_version": manifest["version"],
        },
    )


def fetch_sdf_inputs(root: Path, arm: str) -> tuple[Path, Path, dict[str, Any]]:
    from huggingface_hub import hf_hub_download, snapshot_download

    release = RELEASES[arm]
    donor = Path(
        snapshot_download(
            repo_id=DONOR_REPO,
            revision=DONOR_REVISION,
            ignore_patterns=["*.pth", "*.gguf", "original/*"],
        )
    )
    document = Path(
        hf_hub_download(
            repo_id=DOCS_REPO,
            repo_type="dataset",
            revision=DOCS_REVISION,
            filename=release.path,
        )
    )
    observed = sha256(document)
    rows = sum(1 for line in document.read_text().splitlines() if line.strip())
    if observed != release.sha256 or rows != release.docs:
        raise RuntimeError(
            f"{arm} document contract failed: sha={observed}, rows={rows}"
        )
    return (
        donor,
        document,
        {
            "donor_repo": DONOR_REPO,
            "donor_revision": DONOR_REVISION,
            "docs_repo": DOCS_REPO,
            "docs_revision": DOCS_REVISION,
            "docs_path": release.path,
            "docs_sha256": observed,
            "docs": rows,
            "source_tokens": release.source_tokens,
            "presentations": 4,
            "presented_training_tokens": total_sdf_presented_tokens(arm),
        },
    )


def prepare_capability(root: Path) -> tuple[Path, dict[str, Any]]:
    from scimt.eval import capability

    rows = capability.load_capability(n_mmlu=40, n_gsm8k=40, seed=CAPABILITY_SEED)
    path = root / "data" / "capability.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return path, {
        "rows": len(rows),
        "sha256": sha256(path),
        "seed": CAPABILITY_SEED,
    }


def run_process(argv: list[str], log_path: Path, *, pythonpath: bool = False) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "NCCL_NVLS_ENABLE": "0",
        }
    )
    if pythonpath:
        environment["PYTHONPATH"] = str(REPO_ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        code = subprocess.call(
            argv,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if code:
        raise RuntimeError(
            f"command failed ({code}): {' '.join(argv)}\n"
            + log_path.read_text(errors="replace")[-30_000:]
        )


def validate_adapter_payload(
    adapter: Path, lora: LoraConfig, *, exact_text_targets: bool
) -> dict[str, Any]:
    from safetensors import safe_open

    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("r") != lora.r or config.get("lora_alpha") != lora.resolved_alpha:
        raise RuntimeError(
            "adapter config rank/alpha differs from its frozen training contract"
        )
    payload = adapter / "adapter_model.safetensors"
    with safe_open(payload, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        b_keys = [key for key in keys if ".lora_B." in key]
        nonzero_b = any(bool(handle.get_tensor(key).count_nonzero()) for key in b_keys)
    if not b_keys or not nonzero_b:
        raise RuntimeError("adapter has no non-zero trained LoRA-B tensors")
    if exact_text_targets:
        if any("vision" in key for key in keys):
            raise RuntimeError("SDF adapter unexpectedly contains vision-tower tensors")
        expected = set(lora.target_modules or ())
        missing = [
            target
            for target in expected
            if not any(
                target in key and side in key
                for key in keys
                for side in (".lora_A.", ".lora_B.")
            )
        ]
        incomplete = [
            target
            for target in expected
            if not all(
                any(target in key and side in key for key in keys)
                for side in (".lora_A.", ".lora_B.")
            )
        ]
        if missing or incomplete:
            raise RuntimeError(
                f"SDF adapter target audit failed: missing={missing[:4]}, "
                f"incomplete={incomplete[:4]}"
            )
    return {
        "tensor_count": len(keys),
        "lora_b_tensor_count": len(b_keys),
        "nonzero_lora_b": nonzero_b,
        "vision_tensor_count": sum("vision" in key for key in keys),
        "exact_text_target_count": (
            len(lora.target_modules or ()) if exact_text_targets else None
        ),
    }


def adapter_checkpoint(run_dir: Path, expected_step: int) -> Path:
    checkpoints = []
    for path in (run_dir / "checkpoints").glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if suffix.isdigit() and (path / "adapter_config.json").is_file():
            checkpoints.append((int(suffix), path))
    if [step for step, _ in sorted(checkpoints)] != [expected_step]:
        raise RuntimeError(
            f"{run_dir}: retained adapter steps are "
            f"{[step for step, _ in sorted(checkpoints)]}, expected [{expected_step}]"
        )
    adapter = dict(checkpoints)[expected_step]
    payload = adapter / "adapter_model.safetensors"
    if not payload.is_file() or payload.stat().st_size == 0:
        raise RuntimeError(f"missing adapter payload: {payload}")
    provenance = json.loads((run_dir / "training_provenance.json").read_text())
    if provenance.get("actual", {}).get("global_step") != expected_step:
        raise RuntimeError("training provenance has the wrong terminal step")
    return adapter


def train_adapter(
    root: Path,
    *,
    arm: str,
    phase: str,
    stage_name: str,
    parent: Path,
    dataset: Path,
    lora: LoraConfig,
    expected_step: int,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    run_dir = root / "training" / phase
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    config = TrainConfig(
        backend="axolotl",
        stage=stage_name,
        model="gemma3_12b",
        seed=seed,
        load_checkpoint_path=str(parent),
        lora=lora,
    )
    rendered = render_stage(load_stage(stage_name), config, dataset, run_dir)
    started = time.time()
    log(f"{arm}/{phase}: training through step {expected_step}")
    run_process(["axolotl", "train", str(rendered)], run_dir / "train.log")
    finalize_training_attribution(rendered, run_dir)
    adapter = adapter_checkpoint(run_dir, expected_step)
    payload_audit = validate_adapter_payload(
        adapter, lora, exact_text_targets=phase == "sdf"
    )
    metadata = {
        "schema_version": "dispatch_lora_grafting_training_v1",
        "version": VERSION,
        "arm": arm,
        "phase": phase,
        "stage": stage_name,
        "seed": seed,
        "dataset_sha256": sha256(dataset),
        "lora": asdict(lora),
        "global_step": expected_step,
        "retained_checkpoints": [expected_step],
        "optimizer_state_retained": False,
        "adapter_payload_audit": payload_audit,
        "minutes": round((time.time() - started) / 60, 3),
        "completed_at": utc_now(),
    }
    atomic_json(run_dir / "TRAINING_COMPLETE.json", metadata)
    log(f"{arm}/{phase}: terminal adapter verified")
    return adapter, metadata


def stage_adapter(
    root: Path, arm: str, phase: str, source: Path, training: dict[str, Any]
) -> Path:
    destination = root / "publish" / phase
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    copied = []
    for name in ADAPTER_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
            copied.append(name)
    if "adapter_config.json" not in copied or "adapter_model.safetensors" not in copied:
        raise RuntimeError(f"incomplete adapter staging from {source}")
    atomic_json(destination / "TRAINING.json", training)
    manifest = tree_manifest(destination)
    atomic_json(
        destination / "ARTIFACT_MANIFEST.json",
        {
            "schema_version": "scimt_adapter_artifact_manifest_v1",
            "arm": arm,
            "phase": phase,
            "files": manifest,
            "tree_sha256": tree_digest(manifest),
        },
    )
    return destination


def upload_folder_verified(
    *, repo_id: str, repo_type: str, folder: Path, remote_prefix: str
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=str(folder),
                path_in_repo=remote_prefix,
                commit_message=f"{VERSION}: {remote_prefix}",
            )
            break
        except Exception as error:
            last_error = error
            if attempt == 4:
                raise
            log(f"upload retry {attempt} for {repo_id}/{remote_prefix}: {error}")
            time.sleep(10 * attempt)
    local = tree_manifest(folder)
    info = api.repo_info(repo_id, repo_type=repo_type)
    entries = list(
        api.list_repo_tree(
            repo_id,
            repo_type=repo_type,
            path_in_repo=remote_prefix,
            revision=info.sha,
            recursive=True,
            expand=True,
        )
    )
    remote_entries = {
        str(item.path): item
        for item in entries
        if hasattr(item, "size") and getattr(item, "type", "file") != "directory"
    }
    remote = {path: int(item.size) for path, item in remote_entries.items()}
    missing = [
        f"{remote_prefix}/{path}"
        for path in local
        if f"{remote_prefix}/{path}" not in remote
    ]
    mismatched = [
        path
        for path, item in local.items()
        if remote.get(f"{remote_prefix}/{path}") != item["size"]
    ]
    hash_mismatched = []
    for path, item in local.items():
        remote_item = remote_entries.get(f"{remote_prefix}/{path}")
        lfs = getattr(remote_item, "lfs", None)
        remote_sha = (
            lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        )
        if remote_sha is not None and remote_sha != item["sha256"]:
            hash_mismatched.append(path)
    if missing or mismatched or hash_mismatched:
        raise RuntimeError(
            f"remote verification failed: missing={missing[:5]}, "
            f"size={mismatched[:5]}, sha256={hash_mismatched[:5]}"
        )
    return {
        "repo": repo_id,
        "repo_type": repo_type,
        "revision": str(info.sha),
        "prefix": remote_prefix,
        "files": len(local),
        "sizes_verified": True,
        "lfs_sha256_verified": True,
        "last_retry_error": str(last_error) if last_error else None,
    }


def upload_file_verified(path: Path, remote_path: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    for attempt in range(1, 5):
        try:
            api.upload_file(
                repo_id=MODEL_REPO,
                path_or_fileobj=str(path),
                path_in_repo=remote_path,
                commit_message=f"{VERSION}: {remote_path}",
            )
            break
        except Exception as error:
            if attempt == 4:
                raise
            log(f"file upload retry {attempt} for {remote_path}: {error}")
            time.sleep(10 * attempt)
    info = api.repo_info(MODEL_REPO)
    parent, _, _name = remote_path.rpartition("/")
    entries = api.list_repo_tree(
        MODEL_REPO,
        path_in_repo=parent or None,
        revision=info.sha,
        recursive=False,
        expand=True,
    )
    sizes = {
        str(item.path): int(item.size)
        for item in entries
        if hasattr(item, "size") and getattr(item, "type", "file") != "directory"
    }
    if sizes.get(remote_path) != path.stat().st_size:
        raise RuntimeError(f"remote verification failed for {remote_path}")
    return {"revision": str(info.sha), "path": remote_path, "size_verified": True}


def copy_training_evidence(root: Path, phase: str) -> None:
    source = root / "training" / phase
    destination = root / "evidence" / "training" / phase
    destination.mkdir(parents=True, exist_ok=True)
    names = (
        "axolotl.yaml",
        "train.log",
        "training_provenance.json",
        "training_trace.jsonl",
        "trainer_state.final.json",
        "TRAINING_COMPLETE.json",
    )
    for name in names:
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


def merge_adapter(base: Path, adapter: Path, output: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    log(f"merging {adapter} onto {base} in BF16")
    model = AutoModelForImageTextToText.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    suffix = "model.language_model.layers.0.self_attn.q_proj.weight"
    tracked_name, parameter = next(
        (name, value)
        for name, value in model.named_parameters()
        if name.endswith(suffix)
    )
    before = parameter.detach().float().clone()
    peft_model = PeftModel.from_pretrained(model, str(adapter))
    merged = peft_model.merge_and_unload()
    after = dict(merged.named_parameters())[tracked_name].detach().float()
    delta_norm = float((after - before).norm())
    if not delta_norm > 0:
        raise RuntimeError("LoRA merge produced zero tracked weight change")
    merged.to(dtype=torch.bfloat16)
    merged.config.tie_word_embeddings = True
    merged.tie_weights()
    dtypes = sorted(
        {str(value.dtype) for value in merged.parameters() if value.is_floating_point()}
    )
    if dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"unexpected merged dtypes: {dtypes}")
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="30GB")
    AutoProcessor.from_pretrained(base).save_pretrained(output)
    for source in base.iterdir():
        if not source.is_file() or (output / source.name).exists():
            continue
        if source.name.endswith(".safetensors") or source.name.endswith(
            ".safetensors.index.json"
        ):
            continue
        shutil.copy2(source, output / source.name)
    manifest = tree_manifest(output)
    result = {
        "tracked_parameter": tracked_name,
        "tracked_delta_norm": delta_norm,
        "merge_dtype": "bfloat16",
        "tie_word_embeddings": True,
        "floating_dtypes": dtypes,
        "files": manifest,
        "tree_sha256": tree_digest(manifest),
    }
    del before, after, peft_model, merged, model
    gc.collect()
    return result


def evaluate(
    root: Path,
    *,
    arm: str,
    endpoint: str,
    model: Path,
    data: Path,
    capability: Path,
) -> None:
    module = "experiments.prior_coins.dispatch_lora_grafting_v1.evaluate"
    run_process(
        [
            EVAL_PYTHON,
            "-m",
            module,
            "--model",
            str(model),
            "--data",
            str(data),
            "--capability",
            str(capability),
            "--output",
            str(root / "evidence" / "evaluation"),
            "--arm",
            arm,
            "--endpoint",
            endpoint,
            "--work",
            str(root / "runtime"),
            "--seed",
            str(EVAL_SEED),
        ],
        root / "evidence" / "evaluation" / arm / f"{endpoint}.log",
        pythonpath=True,
    )


def initial_evidence(
    root: Path, run_id: str, arm: str, hardware: dict[str, Any]
) -> None:
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "nvidia_smi_q.txt").write_text(
        subprocess.check_output(["nvidia-smi", "-q"], text=True)
    )
    atomic_json(
        evidence / "run.json",
        {
            "schema_version": "dispatch_lora_grafting_run_v1",
            "version": VERSION,
            "run_id": run_id,
            "arm": arm,
            "created_at": utc_now(),
            "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
            "source_tree": os.environ.get("SCIMT_SOURCE_TREE"),
            "seeds": {
                "sdf_training": SDF_SEED,
                "aft_training": AFT_SEED,
                "evaluation": EVAL_SEED,
                "capability_subset": CAPABILITY_SEED,
            },
            "hardware": hardware,
            "training_packages": package_versions(),
            "evaluation_packages": package_versions(EVAL_PYTHON),
            "retention": {
                "published": ["final SDF adapters", "final AFT adapters", "evidence"],
                "not_published": [
                    "merged weights",
                    "optimizer state",
                    "trajectory checkpoints",
                ],
                "evaluated_endpoints": ["pre_aft", "post_aft"],
            },
        },
    )


async def main_async(args: argparse.Namespace) -> None:
    arm = args.arm
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in root.iterdir() if path.name != "evidence"]
    if unexpected:
        raise RuntimeError(f"run root must be fresh: {unexpected}")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    hardware = check_hardware()
    initial_evidence(root, args.run_id, arm, hardware)

    log(f"{arm}: fetching pinned control and AFT data")
    control, control_manifest = fetch_control(root)
    data, aft_dataset, aft_manifest = fetch_aft_data(root)
    capability, capability_manifest = prepare_capability(root)
    evidence_data = root / "evidence" / "data"
    evidence_data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(capability, evidence_data / "capability.jsonl")
    shutil.copy2(
        data / "dataset_manifest.json", evidence_data / "dataset_manifest.json"
    )
    data_contract: dict[str, Any] = {
        "control": control_manifest,
        "aft": aft_manifest,
        "capability": capability_manifest,
    }

    publications: dict[str, Any] = {}
    if arm in GRAFT_ARMS:
        donor, document, sdf_input = fetch_sdf_inputs(root, arm)
        data_contract["sdf"] = sdf_input
        sdf_adapter, sdf_training = train_adapter(
            root,
            arm=arm,
            phase="sdf",
            stage_name=SDF_STAGE,
            parent=donor,
            dataset=document,
            lora=sdf_lora(),
            expected_step=SDF_STEPS,
            seed=SDF_SEED,
        )
        copy_training_evidence(root, "sdf")
        staged_sdf = stage_adapter(root, arm, "sdf", sdf_adapter, sdf_training)
        publications["sdf_adapter"] = await asyncio.to_thread(
            upload_folder_verified,
            repo_id=MODEL_REPO,
            repo_type="model",
            folder=staged_sdf,
            remote_prefix=model_prefix(arm, "sdf_adapter"),
        )
        log(f"{arm}: SDF adapter remotely verified before grafting")
        pre_model = root / "temporary_merged" / "pre_aft"
        pre_merge = merge_adapter(control, sdf_adapter, pre_model)
    else:
        pre_model = control
        pre_manifest = tree_manifest(control)
        pre_merge = {
            "operation": "identity (matched control; no SDF adapter)",
            "files": pre_manifest,
            "tree_sha256": tree_digest(pre_manifest),
            "merge_dtype": "bfloat16",
        }

    aft_adapter, aft_training = train_adapter(
        root,
        arm=arm,
        phase="aft",
        stage_name=AFT_STAGE,
        parent=pre_model,
        dataset=aft_dataset,
        lora=aft_lora(),
        expected_step=AFT_STEPS,
        seed=AFT_SEED,
    )
    copy_training_evidence(root, "aft")
    staged_aft = stage_adapter(root, arm, "aft", aft_adapter, aft_training)
    publications["aft_adapter"] = await asyncio.to_thread(
        upload_folder_verified,
        repo_id=MODEL_REPO,
        repo_type="model",
        folder=staged_aft,
        remote_prefix=model_prefix(arm, "aft_adapter"),
    )
    log(f"{arm}: AFT adapter remotely verified before evaluation")

    post_model = root / "temporary_merged" / "post_aft"
    post_merge = merge_adapter(pre_model, aft_adapter, post_model)
    evaluate(
        root,
        arm=arm,
        endpoint="pre_aft",
        model=pre_model,
        data=data,
        capability=capability,
    )
    evaluate(
        root,
        arm=arm,
        endpoint="post_aft",
        model=post_model,
        data=data,
        capability=capability,
    )
    result = score_arm(data, root / "evidence" / "evaluation", arm)
    atomic_json(root / "evidence" / "arm_summary.json", result)
    atomic_json(root / "evidence" / "data_contract.json", data_contract)

    reconstruction = {
        "schema_version": "dispatch_lora_grafting_reconstruction_v1",
        "version": VERSION,
        "arm": arm,
        "control": {
            "repo": CONTROL_REPO,
            "revision": CONTROL_REVISION,
            "prefix": CONTROL_PREFIX,
            "tree_sha256": control_manifest["tree_sha256"],
        },
        "sdf": (
            {
                "donor_repo": DONOR_REPO,
                "donor_revision": DONOR_REVISION,
                "adapter_repo": MODEL_REPO,
                "adapter_prefix": model_prefix(arm, "sdf_adapter"),
                "adapter_revision_after_upload": publications["sdf_adapter"][
                    "revision"
                ],
            }
            if arm in GRAFT_ARMS
            else None
        ),
        "aft": {
            "adapter_repo": MODEL_REPO,
            "adapter_prefix": model_prefix(arm, "aft_adapter"),
            "adapter_revision_after_upload": publications["aft_adapter"]["revision"],
        },
        "recipe": (
            [
                "load the pinned control in BF16",
                "attach the pinned SDF adapter with PEFT and merge_and_unload",
                "normalize every floating parameter to BF16, tie weights, and save/reload",
                "verify the pre_aft tree hash below",
                "attach the AFT adapter; merge only if a full model is required",
            ]
            if arm in GRAFT_ARMS
            else [
                "load the pinned control in BF16 and verify the pre_aft tree hash",
                "attach the AFT adapter; merge only if a full model is required",
            ]
        ),
        "pre_aft": pre_merge,
        "post_aft": post_merge,
        "packages": package_versions(),
        "published_full_weights": False,
        "temporary_full_weights_deleted_after_verification": True,
    }
    reconstruction_path = root / "evidence" / "reconstruction.json"
    atomic_json(reconstruction_path, reconstruction)
    publications["reconstruction"] = await asyncio.to_thread(
        upload_file_verified,
        reconstruction_path,
        model_prefix(arm, "reconstruction"),
    )

    from huggingface_hub import HfApi

    api = HfApi()
    await asyncio.to_thread(
        api.create_repo,
        EVIDENCE_REPO,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    publications["evidence"] = await asyncio.to_thread(
        upload_folder_verified,
        repo_id=EVIDENCE_REPO,
        repo_type="dataset",
        folder=root / "evidence",
        remote_prefix=evidence_prefix(args.run_id, arm),
    )
    complete = {
        "schema_version": "dispatch_lora_grafting_arm_complete_v1",
        "status": "artifacts_complete_model_sentinel_pending",
        "version": VERSION,
        "run_id": args.run_id,
        "arm": arm,
        "completed_at": utc_now(),
        "publications": publications,
        "published_full_weights": False,
        "retained_adapter_checkpoints": {
            "sdf": [SDF_STEPS] if arm in GRAFT_ARMS else [],
            "aft": [AFT_STEPS],
        },
        "evaluated_endpoints": ["pre_aft", "post_aft"],
    }
    atomic_json(root / "evidence" / "ARM_COMPLETE.json", complete)
    final_evidence = await asyncio.to_thread(
        upload_folder_verified,
        repo_id=EVIDENCE_REPO,
        repo_type="dataset",
        folder=root / "evidence",
        remote_prefix=evidence_prefix(args.run_id, arm),
    )
    # The model-repo sentinel is the cross-repo commit point. It is written
    # last and points at an already verified evidence revision.
    complete["status"] = "complete"
    complete["final_evidence_revision"] = final_evidence["revision"]
    atomic_json(root / "evidence" / "ARM_COMPLETE.json", complete)
    complete_publication = await asyncio.to_thread(
        upload_file_verified,
        root / "evidence" / "ARM_COMPLETE.json",
        model_prefix(arm, "complete"),
    )
    atomic_json(root / "evidence" / "MODEL_SENTINEL_UPLOAD.json", complete_publication)
    # Only now, after both remote destinations and all endpoint results have
    # been verified, remove the derived full weights.
    shutil.rmtree(root / "temporary_merged", ignore_errors=True)
    log(f"{arm}: complete; adapters/evidence durable and merged weights removed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
