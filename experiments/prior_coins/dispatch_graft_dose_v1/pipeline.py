"""Run one graft-dose unit of work on one H100: an SDF cell, or a whole parent.

Two modes:

``--mode sdf --cell <cell>``
    Fetch the pinned donor and the pre-built mix (digest-gated against
    ``EXPECTED_MIXES``), train the SDF LoRA, audit the adapter payload, and
    publish the terminal adapter — plus, for ``d8m``, the labelled mid-schedule
    ladder. No graft, no eval.

``--mode graft --parent <parent>``
    Fetch the control, reconstruct the graft from the published SDF adapter
    (identity for the bare control), train every AFT mixture for that parent,
    publish each adapter, then evaluate ALL endpoints — the pre-AFT parent plus
    every mixture x step adapter — and score.

Durability invariant, inherited from grafting-v1: a trained adapter is uploaded
and remotely verified before the pipeline performs the next expensive stage.
Merged weights are derived, pod-local, and deleted only after every adapter and
every eval row is remotely verified.

Both modes are idempotent: re-running with ``--resume`` reuses locally verified
terminal adapters and already-written eval rows.
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

from experiments.prior_coins.dispatch_graft_dose_v1 import contracts  # noqa: E402
from scimt.train import LoraConfig, TrainConfig  # noqa: E402
from scimt.train.axolotl import (  # noqa: E402
    finalize_training_attribution,
    load_stage,
    render_stage,
)

EVAL_PYTHON = "/workspace/venv-graft-dose/bin/python"
FORENSICS_POD = REPO_ROOT / "experiments/prior_coins/generalization_forensics/pod"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def jsonl_rows(path: Path) -> int:
    """Count JSONL records, splitting on newline ONLY.

    ``str.splitlines()`` also splits on U+2028, U+2029 and NEL, which occur
    inside Dolmino documents and are written literally by ``ensure_ascii=False``.
    Using it over-counts: charter_d8m reported 23,254 rows against a pinned
    23,218 and failed a cell whose sha256 had already matched. The writer emits
    exactly one "\n" per record, so "\n" is the only correct separator.
    """

    return sum(1 for line in path.read_text().split("\n") if line.strip())


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
        code = (
            "import importlib.metadata, json\n"
            "names = ('torch','transformers','peft','vllm','datasets',"
            "'huggingface_hub')\n"
            "out = {}\n"
            "for n in names:\n"
            "    try: out[n] = importlib.metadata.version(n)\n"
            "    except importlib.metadata.PackageNotFoundError: out[n] = None\n"
            "print(json.dumps(out))\n"
        )
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


def sdf_lora() -> LoraConfig:
    return LoraConfig(
        r=contracts.SDF_LORA_RANK,
        alpha=contracts.SDF_LORA_ALPHA,
        dropout=contracts.SDF_LORA_DROPOUT,
        target_linear=False,
        target_modules=contracts.gemma3_text_targets(),
    )


def aft_lora() -> LoraConfig:
    # Deliberately the wave-v2 parameterization, bare suffixes and all: the
    # endpoints of this grid are only comparable to prior Dispatch AFT cells
    # because this is byte-identical to them.
    return LoraConfig(
        r=contracts.AFT_LORA_RANK,
        alpha=contracts.AFT_LORA_ALPHA,
        dropout=contracts.AFT_LORA_DROPOUT,
        target_linear=False,
        target_modules=contracts.aft_target_modules(),
    )


def check_hardware(expected_gpus: int = 1) -> dict[str, Any]:
    import torch

    found = torch.cuda.device_count()
    if found != expected_gpus:
        raise RuntimeError(f"expected {expected_gpus} GPU(s), found {found}")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 75 * 1024**3:
        raise RuntimeError(
            f"expected an 80GB GPU, found {properties.name} "
            f"({properties.total_memory / 1024**3:.1f} GiB)"
        )
    return {
        "name": properties.name,
        "gpus": found,
        "memory_bytes": properties.total_memory,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }


# --- input fetching -------------------------------------------------------------


def fetch_donor() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=contracts.DONOR_REPO,
            revision=contracts.DONOR_REVISION,
            ignore_patterns=["*.pth", "*.gguf", "original/*"],
        )
    )


def fetch_mix(cell: str) -> tuple[Path, dict[str, Any]]:
    """Download the pre-built mix and gate it against the frozen digest."""

    from huggingface_hub import hf_hub_download

    arm, dose_m, presentations = contracts.parse_cell(cell)
    mix = contracts.mix_id(arm, dose_m)
    expected = contracts.EXPECTED_MIXES[mix]
    path = Path(
        hf_hub_download(
            repo_id=contracts.EVIDENCE_REPO,
            repo_type="dataset",
            filename=contracts.mix_remote_path(mix),
        )
    )
    observed = sha256(path)
    if observed != expected["jsonl_sha256"]:
        raise RuntimeError(
            f"{mix}: mix digest {observed} != frozen {expected['jsonl_sha256']}"
        )
    rows = jsonl_rows(path)
    if rows != expected["docs"]:
        raise RuntimeError(f"{mix}: {rows} rows, expected {expected['docs']}")
    steps = contracts.require_expected_optimizer_steps(cell, expected["tokens"])
    return path, {
        "mix": mix,
        "cell": cell,
        "arm": arm,
        "dose_m": dose_m,
        "presentations": presentations,
        "repo": contracts.EVIDENCE_REPO,
        "path": contracts.mix_remote_path(mix),
        "jsonl_sha256": observed,
        "docs": rows,
        "tokens": expected["tokens"],
        "task_tokens": expected["task_tokens"],
        "dolmino_tokens": expected["dolmino_tokens"],
        "expected_optimizer_steps": steps,
        "presented_tokens": contracts.presented_tokens(
            expected["tokens"], presentations
        ),
    }


def fetch_control(root: Path) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    del root
    snapshot = Path(
        snapshot_download(
            repo_id=contracts.CONTROL_REPO,
            revision=contracts.CONTROL_REVISION,
            allow_patterns=[f"{contracts.CONTROL_PREFIX}/*"],
        )
    )
    parent = snapshot / contracts.CONTROL_PREFIX
    weight = parent / "model.safetensors"
    if not (parent / "config.json").is_file() or not weight.is_file():
        raise RuntimeError(f"incomplete control parent: {parent}")
    observed = sha256(weight)
    if observed != contracts.CONTROL_WEIGHT_SHA256:
        raise RuntimeError(
            f"control weight hash {observed} != frozen "
            f"{contracts.CONTROL_WEIGHT_SHA256}"
        )
    manifest = tree_manifest(parent)
    return parent, {
        "repo": contracts.CONTROL_REPO,
        "revision": contracts.CONTROL_REVISION,
        "prefix": contracts.CONTROL_PREFIX,
        "weight_sha256": observed,
        "tree_sha256": tree_digest(manifest),
    }


def fetch_published_sdf_adapter(cell: str, destination: Path) -> tuple[Path, dict]:
    """Pull back the SDF adapter this grid published for ``cell``."""

    from huggingface_hub import hf_hub_download

    prefix = contracts.model_prefix(cell, "sdf_adapter")
    destination.mkdir(parents=True, exist_ok=True)
    for name in (*ADAPTER_FILES, "TRAINING.json", "ARTIFACT_MANIFEST.json"):
        path = Path(
            hf_hub_download(
                repo_id=contracts.MODEL_REPO,
                filename=f"{prefix}/{name}",
            )
        )
        shutil.copy2(path, destination / name)
    manifest = json.loads((destination / "ARTIFACT_MANIFEST.json").read_text())
    observed = tree_manifest(destination)
    for name in ADAPTER_FILES:
        if observed[name]["sha256"] != manifest["files"][name]["sha256"]:
            raise RuntimeError(f"{cell}: downloaded SDF adapter {name} digest drift")
    return destination, {
        "repo": contracts.MODEL_REPO,
        "prefix": prefix,
        "tree_sha256": manifest["tree_sha256"],
    }


def fetch_aft_data(mixtures: tuple[str, ...]) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    patterns = [
        f"{contracts.AFT_DATA_PREFIX}/dataset_manifest.json",
        *(
            f"{contracts.AFT_DATA_PREFIX}/datasets/aft_{name}.jsonl"
            for name in mixtures
        ),
        *(
            f"{contracts.AFT_DATA_PREFIX}/episodes/{name}.jsonl"
            for name in contracts.SLICES
        ),
        *(
            f"{contracts.AFT_DATA_PREFIX}/prompts/{name}.jsonl"
            for name in contracts.SLICES
        ),
    ]
    snapshot = Path(
        snapshot_download(
            repo_id=contracts.AFT_DATA_REPO,
            repo_type="dataset",
            revision=contracts.AFT_DATA_REVISION,
            allow_patterns=patterns,
        )
    )
    data = snapshot / contracts.AFT_DATA_PREFIX
    receipts: dict[str, Any] = {}
    for name in mixtures:
        path = data / "datasets" / f"aft_{name}.jsonl"
        rows = jsonl_rows(path)
        if rows != contracts.AFT_ROWS:
            raise RuntimeError(
                f"aft_{name}: {rows} rows, expected {contracts.AFT_ROWS}"
            )
        digest = sha256(path)
        if name == "agreement" and digest != contracts.AFT_AGREEMENT_SHA256:
            raise RuntimeError(f"aft_agreement digest drift: {digest}")
        receipts[name] = {"rows": rows, "sha256": digest}
    manifest = json.loads((data / "dataset_manifest.json").read_text())
    if manifest.get("version") != "dispatch_wave_v2":
        raise RuntimeError(f"unexpected AFT manifest {manifest.get('version')!r}")
    for name, expected in contracts.EVAL_SLICE_PROMPTS.items():
        prompts = data / "prompts" / f"{name}.jsonl"
        count = jsonl_rows(prompts)
        if count != expected:
            raise RuntimeError(f"{name}: {count} prompts, expected {expected}")
        if not (data / "episodes" / f"{name}.jsonl").is_file():
            raise RuntimeError(f"missing evaluation episodes for {name}")
    return data, {
        "repo": contracts.AFT_DATA_REPO,
        "revision": contracts.AFT_DATA_REVISION,
        "prefix": contracts.AFT_DATA_PREFIX,
        "manifest_version": manifest["version"],
        "datasets": receipts,
        "slice_prompts": dict(contracts.EVAL_SLICE_PROMPTS),
    }


# --- training -------------------------------------------------------------------


def run_process(
    argv: list[str],
    log_path: Path,
    *,
    pythonpath: bool = False,
    gpus: int = 1,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "NCCL_NVLS_ENABLE": "0",
        }
    )
    # axolotl launches across every VISIBLE device, so pinning device 0 on a
    # multi-GPU stage would silently train at world size 1 — a quarter of the
    # intended global batch, with the step count unchanged. Only pin when the
    # stage really is single-GPU (every eval path is).
    if gpus == 1:
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    if pythonpath:
        environment["PYTHONPATH"] = str(REPO_ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        code = subprocess.call(
            [str(item) for item in argv],
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if code:
        raise RuntimeError(
            f"command failed ({code}): {' '.join(str(i) for i in argv)}\n"
            + log_path.read_text(errors="replace")[-30_000:]
        )


def validate_adapter_payload(
    adapter: Path, lora: LoraConfig, *, exact_text_targets: bool
) -> dict[str, Any]:
    from safetensors import safe_open

    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("r") != lora.r or config.get("lora_alpha") != lora.resolved_alpha:
        raise RuntimeError("adapter rank/alpha differs from its training contract")
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
        incomplete = [
            target
            for target in expected
            if not all(
                any(target in key and side in key for key in keys)
                for side in (".lora_A.", ".lora_B.")
            )
        ]
        if incomplete:
            raise RuntimeError(
                f"SDF adapter target audit failed on {len(incomplete)} targets, "
                f"e.g. {incomplete[:4]}"
            )
    return {
        "tensor_count": len(keys),
        "lora_b_tensor_count": len(b_keys),
        "nonzero_lora_b": nonzero_b,
        "vision_tensor_count": sum("vision" in key for key in keys),
        "audited_exact_targets": (
            len(lora.target_modules or ()) if exact_text_targets else None
        ),
    }


def adapter_checkpoints(run_dir: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in (run_dir / "checkpoints").glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if suffix.isdigit() and (path / "adapter_config.json").is_file():
            found[int(suffix)] = path
    return found


def train_adapter(
    root: Path,
    *,
    label: str,
    phase: str,
    stage_name: str,
    parent: Path,
    dataset: Path,
    lora: LoraConfig,
    expected_step: int,
    required_steps: tuple[int, ...],
    seed: int,
    gpus: int = 1,
) -> tuple[dict[int, Path], dict[str, Any]]:
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
    log(f"{label}/{phase}: training to step {expected_step} ({stage_name})")
    run_process(
        ["axolotl", "train", str(rendered)], run_dir / "train.log", gpus=gpus
    )
    finalize_training_attribution(rendered, run_dir)

    found = adapter_checkpoints(run_dir)
    provenance = json.loads((run_dir / "training_provenance.json").read_text())
    actual_step = provenance.get("actual", {}).get("global_step")
    steps_audit: dict[str, Any] = {}
    if phase == "sdf":
        # axolotl's sample packer decides the step count, not our token
        # arithmetic: it bins into 8,192-token blocks and drops the final
        # partial bin, so it lands a little under the nominal (measured:
        # coin_d2m 60 vs nominal 64). Bound it and record it — the dose
        # contract is the digest-pinned DATA, not the packer's block count.
        steps_audit = contracts.accept_realized_steps(label, actual_step, expected_step)
        expected_step = actual_step
        # the ladder is optional free optionality (SPEC 4.3), never a cell:
        # keep whichever milestones the realized schedule actually produced
        required_steps = tuple(s for s in required_steps if s in found) or (
            actual_step,
        )
    elif actual_step != expected_step:
        raise RuntimeError(
            f"{label}/{phase}: terminal global_step {actual_step} != frozen "
            f"{expected_step} — the data or geometry changed under the pin"
        )
    missing = [step for step in required_steps if step not in found]
    if expected_step not in found:
        missing.append(expected_step)
    if missing:
        raise RuntimeError(
            f"{label}/{phase}: missing required checkpoints {missing}; "
            f"found {sorted(found)}"
        )
    audit = validate_adapter_payload(
        found[expected_step], lora, exact_text_targets=phase == "sdf"
    )
    minutes = (time.time() - started) / 60
    metadata = {
        "schema_version": "dispatch_graft_dose_training_v1",
        "version": contracts.VERSION,
        "label": label,
        "phase": phase,
        "stage": stage_name,
        "gpus": gpus,
        "seed": seed,
        "dataset_sha256": sha256(dataset),
        "lora": asdict(lora),
        "global_step": expected_step,
        "optimizer_steps": steps_audit or None,
        "retained_checkpoints": sorted(found),
        "required_checkpoints": list(required_steps),
        "adapter_payload_audit": audit,
        "minutes": round(minutes, 3),
        "seconds_per_step": round(minutes * 60 / max(expected_step, 1), 3),
        "completed_at": utc_now(),
    }
    atomic_json(run_dir / "TRAINING_COMPLETE.json", metadata)
    log(
        f"{label}/{phase}: done in {minutes:.1f} min "
        f"({metadata['seconds_per_step']:.2f} s/step), "
        f"checkpoints {sorted(found)}"
    )
    return found, metadata


def resume_adapter(
    root: Path, *, label: str, phase: str, expected_step: int, lora: LoraConfig
) -> tuple[dict[int, Path], dict[str, Any]]:
    run_dir = root / "training" / phase
    completion = run_dir / "TRAINING_COMPLETE.json"
    if not completion.is_file():
        raise RuntimeError(f"cannot resume {label}/{phase}: missing {completion}")
    metadata = json.loads(completion.read_text())
    recorded = metadata.get("global_step")
    if phase != "sdf" and recorded != expected_step:
        raise RuntimeError(f"cannot resume {label}/{phase}: wrong terminal step")
    expected_step = recorded
    found = adapter_checkpoints(run_dir)
    validate_adapter_payload(
        found[expected_step], lora, exact_text_targets=phase == "sdf"
    )
    log(f"{label}/{phase}: resuming from verified adapters {sorted(found)}")
    return found, metadata


def stage_adapter(
    root: Path, name: str, source: Path, training: dict[str, Any]
) -> Path:
    destination = root / "publish" / name
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for filename in ADAPTER_FILES:
        path = source / filename
        if not path.is_file():
            raise RuntimeError(f"incomplete adapter staging: missing {path}")
        shutil.copy2(path, destination / filename)
    atomic_json(destination / "TRAINING.json", training)
    manifest = tree_manifest(destination)
    atomic_json(
        destination / "ARTIFACT_MANIFEST.json",
        {
            "schema_version": "scimt_adapter_artifact_manifest_v1",
            "name": name,
            "files": manifest,
            "tree_sha256": tree_digest(manifest),
        },
    )
    return destination


# --- publication ------------------------------------------------------------------


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
                commit_message=f"{contracts.VERSION}: {remote_prefix}",
            )
            break
        except Exception as error:  # noqa: BLE001 - retried, then re-raised
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
    missing = [p for p in local if f"{remote_prefix}/{p}" not in remote]
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
            f"remote verification failed for {remote_prefix}: "
            f"missing={missing[:5]}, size={mismatched[:5]}, "
            f"sha256={hash_mismatched[:5]}"
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


def upload_file_verified(
    path: Path, remote_path: str, *, repo_id: str, repo_type: str = "model"
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    for attempt in range(1, 5):
        try:
            api.upload_file(
                repo_id=repo_id,
                repo_type=repo_type,
                path_or_fileobj=str(path),
                path_in_repo=remote_path,
                commit_message=f"{contracts.VERSION}: {remote_path}",
            )
            break
        except Exception as error:  # noqa: BLE001 - retried, then re-raised
            if attempt == 4:
                raise
            log(f"file upload retry {attempt} for {remote_path}: {error}")
            time.sleep(10 * attempt)
    info = api.repo_info(repo_id, repo_type=repo_type)
    parent, _, _name = remote_path.rpartition("/")
    entries = api.list_repo_tree(
        repo_id,
        repo_type=repo_type,
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
    return {"repo": repo_id, "revision": str(info.sha), "path": remote_path}


def copy_training_evidence(root: Path, phase: str) -> None:
    source = root / "training" / phase
    destination = root / "evidence" / "training" / phase
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "axolotl.yaml",
        "train.log",
        "training_provenance.json",
        "training_trace.jsonl",
        "trainer_state.final.json",
        "TRAINING_COMPLETE.json",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


# --- grafting -----------------------------------------------------------------------


def merge_adapter(base: Path, adapter: Path, output: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    log(f"merging {adapter.name} onto {base.name} in BF16")
    model = AutoModelForImageTextToText.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
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
        if source.name.endswith((".safetensors", ".safetensors.index.json")):
            continue
        shutil.copy2(source, output / source.name)
    manifest = tree_manifest(output)
    result = {
        "tracked_parameter": tracked_name,
        "tracked_delta_norm": delta_norm,
        "merge_dtype": "bfloat16",
        "tie_word_embeddings": True,
        "floating_dtypes": dtypes,
        "tree_sha256": tree_digest(manifest),
    }
    del before, after, peft_model, merged, model
    gc.collect()
    return result


# --- serving ---------------------------------------------------------------------------


def ensure_servable(model: Path, work: Path) -> Path:
    """Drop a redundant tied ``lm_head.weight`` so vLLM will load the model.

    Gemma-3 ties ``lm_head`` to ``embed_tokens``, so vLLM's
    ``Gemma3ForConditionalGeneration`` has no ``lm_head`` parameter and refuses
    a checkpoint that carries one: "There is no module or parameter named
    'lm_head'". The published gate2 control does carry it, which killed the
    control pod's evaluation after all five of its adapters had trained.

    The tensor is verified byte-equal to ``embed_tokens`` before removal — if it
    ever differs it is real, untied weight and dropping it would silently change
    the model, so that case raises instead.

    Returns ``model`` untouched when there is nothing to strip, so the common
    path costs one index read.
    """

    import torch
    from safetensors.torch import load_file, save_file

    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        return model
    target = None
    for shard in shards:
        from safetensors import safe_open

        with safe_open(shard, framework="pt", device="cpu") as handle:
            if any(key.endswith("lm_head.weight") for key in handle.keys()):
                target = shard
                break
    if target is None:
        return model

    log(f"stripping tied lm_head from {model.name} for serving")
    stripped = work / "servable" / model.name
    if (stripped / "config.json").is_file():
        return stripped
    stripped.mkdir(parents=True, exist_ok=True)
    for item in model.iterdir():
        if item.suffix == ".safetensors" or item.name.endswith(".index.json"):
            continue
        destination = stripped / item.name
        if not destination.exists():
            destination.symlink_to(item.resolve())

    for shard in shards:
        tensors = load_file(str(shard))
        keys = [k for k in tensors if k.endswith("lm_head.weight")]
        for key in keys:
            embed = next(
                (v for k, v in tensors.items() if k.endswith("embed_tokens.weight")),
                None,
            )
            if embed is None or not torch.equal(
                tensors[key].to(torch.float32), embed.to(torch.float32)
            ):
                raise RuntimeError(
                    f"{shard.name}: {key} is NOT byte-equal to embed_tokens — it is "
                    "real untied weight, so stripping it would change the model"
                )
            del tensors[key]
        save_file(tensors, str(stripped / shard.name), metadata={"format": "pt"})
        del tensors
        gc.collect()
    index = next(model.glob("*.index.json"), None)
    if index is not None:
        payload = json.loads(index.read_text())
        weight_map = payload.get("weight_map", {})
        for key in [k for k in weight_map if k.endswith("lm_head.weight")]:
            del weight_map[key]
        (stripped / index.name).write_text(json.dumps(payload, indent=2) + "\n")
    return stripped


# --- evaluation ----------------------------------------------------------------------


def write_sanity_prompts(dataset: Path, out_dir: Path) -> Path:
    """Teacher-forced spot check: does the endpoint reproduce known train rows?"""

    rows = [
        json.loads(line)
        for line in dataset.read_text().split("\n")[:64]
        if line.strip()
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sanity_prompts.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": row["metadata"]["episode_id"],
                    "prompt": row["messages"][0]["content"],
                    "expected": row["messages"][1]["content"],
                }
            )
            + "\n"
            for row in rows
        )
    )
    return path


def slice_args(data: Path) -> list[str]:
    args: list[str] = []
    for name in contracts.SLICES:
        prompts = data / "prompts" / f"{name}.jsonl"
        if not prompts.is_file():
            raise FileNotFoundError(prompts)
        args += ["--prompt-set", f"{name}={prompts}"]
    return args


def endpoint_complete(out_dir: Path) -> bool:
    return all((out_dir / f"{name}.jsonl").is_file() for name in contracts.SLICES)


def evaluate_base(root: Path, parent: str, model: Path, data: Path, sanity: Path):
    """The pre-AFT endpoint: the grafted parent with no adapter attached."""

    out_dir = root / "results" / f"{parent}-pre_aft"
    if endpoint_complete(out_dir):
        log(f"{parent}/pre_aft: already evaluated")
        return
    model = ensure_servable(model, root / "runtime")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sanity, out_dir / "sanity_prompts.jsonl")
    run_process(
        [
            EVAL_PYTHON,
            FORENSICS_POD / "pod_generate.py",
            "--model",
            model,
            "--name",
            f"{parent}-pre_aft",
            "--out-dir",
            out_dir,
            "--work",
            root / "runtime",
            *slice_args(data),
            "--prompt-set",
            f"sanity={out_dir / 'sanity_prompts.jsonl'}",
        ],
        root / "logs" / f"eval-{parent}-pre_aft.log",
    )
    log(f"{parent}/pre_aft: eval complete")


def evaluate_adapters_native(
    root: Path,
    parent: str,
    model: Path,
    data: Path,
    sanity: Path,
    endpoints: list[tuple[str, Path]],
) -> bool:
    """One resident base, every adapter swapped through it. False = fall back.

    The adapter probe inside ``pod_generate_multi`` is the guard that matters:
    vLLM will otherwise accept a Gemma-3 adapter and apply NOTHING, writing a
    complete, internally consistent trajectory of pure base-model outputs that
    nothing downstream could detect.
    """

    todo = [
        (name, adapter)
        for name, adapter in endpoints
        if not endpoint_complete(root / "results" / f"{parent}-{name}")
    ]
    if not todo:
        log(f"{parent}: all adapter endpoints already evaluated")
        return True
    model = ensure_servable(model, root / "runtime")
    argv = [
        EVAL_PYTHON,
        FORENSICS_POD / "pod_generate_multi.py",
        "--base",
        model,
        "--sanity",
        sanity,
        "--out-root",
        root / "results",
        "--name-prefix",
        parent,
        "--work",
        root / "runtime",
        "--max-lora-rank",
        str(contracts.AFT_LORA_RANK),
        *slice_args(data),
    ]
    for name, adapter in todo:
        argv += ["--endpoint", f"{name}={adapter}"]
    try:
        run_process(argv, root / "logs" / f"eval-{parent}-native.log")
    except RuntimeError as error:
        log(f"{parent}: native LoRA eval failed, falling back to merge-per-endpoint")
        log(f"{parent}: reason tail — {str(error)[-1500:]}")
        return False
    log(f"{parent}: {len(todo)} adapter endpoints evaluated natively, 0 merges")
    return True


def evaluate_adapters_merged(
    root: Path,
    parent: str,
    model: Path,
    data: Path,
    sanity: Path,
    endpoints: list[tuple[str, Path]],
) -> None:
    for name, adapter in endpoints:
        out_dir = root / "results" / f"{parent}-{name}"
        if endpoint_complete(out_dir):
            continue
        merged = root / "temporary_merged" / name
        merge_adapter(model, adapter, merged)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sanity, out_dir / "sanity_prompts.jsonl")
            run_process(
                [
                    EVAL_PYTHON,
                    FORENSICS_POD / "pod_generate.py",
                    "--model",
                    merged,
                    "--name",
                    f"{parent}-{name}",
                    "--out-dir",
                    out_dir,
                    "--work",
                    root / "runtime",
                    *slice_args(data),
                    "--prompt-set",
                    f"sanity={out_dir / 'sanity_prompts.jsonl'}",
                ],
                root / "logs" / f"eval-{parent}-{name}.log",
            )
        finally:
            shutil.rmtree(merged, ignore_errors=True)
        log(f"{parent}/{name}: eval complete (merged)")


# --- modes -----------------------------------------------------------------------------


def initial_evidence(root: Path, run_id: str, name: str, mode: str, hardware: dict):
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        (evidence / "nvidia_smi_q.txt").write_text(
            subprocess.check_output(["nvidia-smi", "-q"], text=True)
        )
    except Exception as error:  # noqa: BLE001 - evidence, not a gate
        (evidence / "nvidia_smi_q.txt").write_text(f"unavailable: {error}\n")
    atomic_json(
        evidence / "run.json",
        {
            "schema_version": "dispatch_graft_dose_run_v1",
            "version": contracts.VERSION,
            "run_id": run_id,
            "mode": mode,
            "name": name,
            "created_at": utc_now(),
            "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
            "source_tree": os.environ.get("SCIMT_SOURCE_TREE"),
            "seeds": {
                "data": contracts.DATA_SEED,
                "sdf_training": contracts.SDF_SEED,
                "aft_training": contracts.AFT_SEED,
                "evaluation": contracts.EVAL_SEED,
            },
            "hardware": hardware,
            "training_packages": package_versions(),
            "evaluation_packages": package_versions(EVAL_PYTHON),
        },
    )


async def run_sdf(args: argparse.Namespace, root: Path, hardware: dict) -> None:
    cell = args.cell
    expected_steps = contracts.EXPECTED_STEPS[cell]
    stage_name = contracts.sdf_stage(cell, gpus=args.gpus)
    required = (
        tuple(contracts.D8M_CHECKPOINT_SCHEDULE)
        if stage_name == contracts.SDF_STAGE_D8M
        else (expected_steps,)
    )
    if not args.resume:
        initial_evidence(root, args.run_id, cell, "sdf", hardware)

    log(f"{cell}: fetching donor and mix")
    donor = fetch_donor()
    mix_path, mix_receipt = fetch_mix(cell)
    atomic_json(root / "evidence" / "data_contract.json", {"mix": mix_receipt})
    if mix_receipt["expected_optimizer_steps"] != expected_steps:
        raise RuntimeError("mix receipt disagrees with the frozen step pin")

    complete = root / "training" / "sdf" / "TRAINING_COMPLETE.json"
    if args.resume and complete.is_file():
        found, training = resume_adapter(
            root, label=cell, phase="sdf", expected_step=expected_steps, lora=sdf_lora()
        )
    else:
        found, training = train_adapter(
            root,
            label=cell,
            phase="sdf",
            stage_name=stage_name,
            parent=donor,
            dataset=mix_path,
            lora=sdf_lora(),
            expected_step=expected_steps,
            required_steps=required,
            seed=contracts.SDF_SEED,
            gpus=args.gpus,
        )
    training["mix"] = mix_receipt
    copy_training_evidence(root, "sdf")

    # The terminal step is whatever the packer produced, not the nominal — read
    # it back from the receipt rather than assuming it.
    terminal = int(training["global_step"])
    publications: dict[str, Any] = {}
    staged = stage_adapter(root, "sdf_adapter", found[terminal], training)
    publications["sdf_adapter"] = await asyncio.to_thread(
        upload_folder_verified,
        repo_id=contracts.MODEL_REPO,
        repo_type="model",
        folder=staged,
        remote_prefix=contracts.model_prefix(cell, "sdf_adapter"),
    )
    log(f"{cell}: SDF adapter remotely verified")

    # The d8m mid-schedule ladder: published, labelled, and explicitly NOT a
    # cell of this grid (SPEC 4.3). A later session can build a
    # 1-presentation ladder from these without re-running any SDF.
    if stage_name == contracts.SDF_STAGE_D8M:
        for step in sorted(training.get("retained_checkpoints") or []):
            if step == terminal:
                continue
            ladder_meta = dict(
                training,
                global_step=step,
                mid_schedule=True,
                mid_schedule_note=(
                    "Checkpoint taken mid-cosine at step "
                    f"{step} of {terminal}. NOT a cell of graft-dose v1: "
                    "mid-training checkpoints on this setting read the opposite "
                    "of converged ones (prior-survival-under-finetuning). "
                    "Free optionality for a later 1-presentation ladder only."
                ),
            )
            ladder = stage_adapter(root, f"mid_{step}", found[step], ladder_meta)
            publications[f"mid_schedule_{step}"] = await asyncio.to_thread(
                upload_folder_verified,
                repo_id=contracts.MODEL_REPO,
                repo_type="model",
                folder=ladder,
                remote_prefix=contracts.model_prefix(cell, f"mid_schedule/step_{step}"),
            )
        log(f"{cell}: mid-schedule ladder published")

    await finish(root, args.run_id, cell, publications, extra={"cell": cell})


async def run_graft(args: argparse.Namespace, root: Path, hardware: dict) -> None:
    parent_name = args.parent
    mixtures = contracts.parent_mixtures(parent_name)
    if args.mixtures:
        requested = tuple(args.mixtures.split(","))
        unknown = [m for m in requested if m not in mixtures]
        if unknown:
            raise ValueError(f"{parent_name} does not run mixtures {unknown}")
        mixtures = requested
        log(f"{parent_name}: restricted to mixtures {list(mixtures)}")
    if not args.resume:
        initial_evidence(root, args.run_id, parent_name, "graft", hardware)

    log(f"{parent_name}: fetching control and AFT data")
    control, control_receipt = fetch_control(root)
    data, aft_receipt = fetch_aft_data(mixtures)
    data_contract: dict[str, Any] = {"control": control_receipt, "aft": aft_receipt}

    publications: dict[str, Any] = {}
    if parent_name == contracts.CONTROL_PARENT:
        graft_model = control
        graft_receipt = {
            "operation": "identity (bare matched control; no SDF adapter)",
            "tree_sha256": control_receipt["tree_sha256"],
            "merge_dtype": "bfloat16",
        }
    else:
        adapter_dir, sdf_receipt = fetch_published_sdf_adapter(
            parent_name, root / "sdf_adapter"
        )
        data_contract["sdf_adapter"] = sdf_receipt
        graft_model = root / "temporary_merged" / "graft"
        if not (graft_model / "config.json").is_file():
            graft_receipt = merge_adapter(control, adapter_dir, graft_model)
        else:
            graft_receipt = {"operation": "reused pod-local graft", "resumed": True}
            log(f"{parent_name}: reusing existing pod-local graft")
        graft_receipt["sdf_adapter"] = sdf_receipt
    atomic_json(root / "evidence" / "data_contract.json", data_contract)

    # The pre-AFT endpoint is evaluated FIRST, before any AFT training. It is
    # the primary dose readout (SPEC A8 — no AFT seed noise), so a pod that is
    # killed part-way through its mixtures still contributes the point the dose
    # curve most needs, and the whole grid's primary result lands ~3 h earlier.
    sanity = write_sanity_prompts(
        data / "datasets" / "aft_agreement.jsonl", root / "results"
    )
    evaluate_base(root, parent_name, graft_model, data, sanity)

    endpoints: list[tuple[str, Path]] = []
    trainings: dict[str, Any] = {}
    for mixture in mixtures:
        stage_name = contracts.aft_stage(parent_name, mixture)
        steps = (
            contracts.BRIDGE_STEPS
            if stage_name == contracts.BRIDGE_STAGE
            else contracts.AFT_STEPS
        )
        eval_steps = contracts.aft_eval_steps(parent_name, mixture)
        dataset = data / "datasets" / f"aft_{mixture}.jsonl"
        phase = f"aft_{mixture}"
        complete = root / "training" / phase / "TRAINING_COMPLETE.json"
        if args.resume and complete.is_file():
            found, training = resume_adapter(
                root,
                label=parent_name,
                phase=phase,
                expected_step=steps,
                lora=aft_lora(),
            )
        else:
            found, training = train_adapter(
                root,
                label=parent_name,
                phase=phase,
                stage_name=stage_name,
                parent=graft_model,
                dataset=dataset,
                lora=aft_lora(),
                expected_step=steps,
                required_steps=eval_steps,
                seed=contracts.AFT_SEED,
            )
        trainings[mixture] = training
        copy_training_evidence(root, phase)
        # Publish EVERY step we evaluate, not just the terminal one. Run
        # 20260826T001500Z published only the final adapter, so its step-128
        # endpoints have rows but no weights and cannot be re-evaluated
        # without retraining. The terminal adapter keeps its unsuffixed path.
        for step in eval_steps:
            local = phase if step == steps else f"{phase}_step{step}"
            staged = stage_adapter(root, local, found[step], training)
            publications[local] = await asyncio.to_thread(
                upload_folder_verified,
                repo_id=contracts.MODEL_REPO,
                repo_type="model",
                folder=staged,
                remote_prefix=contracts.aft_adapter_prefix(
                    parent_name, mixture, step
                ),
            )
            log(f"{parent_name}/{mixture} step {step}: adapter remotely verified")
            endpoints.append((f"{mixture}_step{step}", found[step]))

    served = "native_lora"
    if not evaluate_adapters_native(
        root, parent_name, graft_model, data, sanity, endpoints
    ):
        served = "merged_per_endpoint"
        evaluate_adapters_merged(
            root, parent_name, graft_model, data, sanity, endpoints
        )

    expected_dirs = [f"{parent_name}-pre_aft"] + [
        f"{parent_name}-{name}" for name, _ in endpoints
    ]
    for name in expected_dirs:
        if not endpoint_complete(root / "results" / name):
            raise RuntimeError(f"{name}: evaluation incomplete after both paths")
    shutil.copytree(root / "results", root / "evidence" / "results", dirs_exist_ok=True)

    from experiments.prior_coins.dispatch_graft_dose_v1.score import score_parent

    summary = score_parent(
        data, root / "results", parent_name, served=served, ran=mixtures
    )
    atomic_json(root / "evidence" / "parent_summary.json", summary)

    reconstruction = {
        "schema_version": "dispatch_graft_dose_reconstruction_v1",
        "version": contracts.VERSION,
        "parent": parent_name,
        "control": control_receipt,
        "graft": graft_receipt,
        "aft_adapters": {
            f"{mixture}_step{step}": contracts.aft_adapter_prefix(
                parent_name, mixture, step
            )
            for mixture in mixtures
            for step in contracts.aft_eval_steps(parent_name, mixture)
        },
        "aft_training": {m: trainings[m] for m in mixtures},
        "serving": served,
        "recipe": (
            [
                "load the pinned control in BF16",
                "attach the pinned SDF adapter with PEFT and merge_and_unload",
                "normalize floating params to BF16, tie weights, save/reload",
                "verify the graft tree_sha256 below",
                "attach an AFT adapter; merge only if a full model is required",
            ]
            if parent_name != contracts.CONTROL_PARENT
            else [
                "load the pinned control in BF16 and verify its tree_sha256",
                "attach an AFT adapter; merge only if a full model is required",
            ]
        ),
        "packages": package_versions(),
        "published_full_weights": False,
    }
    path = root / "evidence" / "reconstruction.json"
    atomic_json(path, reconstruction)
    publications["reconstruction"] = await asyncio.to_thread(
        upload_file_verified,
        path,
        contracts.model_prefix(parent_name, "reconstruction.json"),
        repo_id=contracts.MODEL_REPO,
    )
    await finish(
        root,
        args.run_id,
        parent_name,
        publications,
        extra={"parent": parent_name, "serving": served, "endpoints": expected_dirs},
    )


async def finish(
    root: Path, run_id: str, name: str, publications: dict, *, extra: dict
) -> None:
    """Upload evidence, write the sentinel last, then drop merged weights."""

    from huggingface_hub import HfApi

    api = HfApi()
    await asyncio.to_thread(
        api.create_repo,
        contracts.EVIDENCE_REPO,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    publications["evidence"] = await asyncio.to_thread(
        upload_folder_verified,
        repo_id=contracts.EVIDENCE_REPO,
        repo_type="dataset",
        folder=root / "evidence",
        remote_prefix=contracts.evidence_prefix(run_id, name),
    )
    complete = {
        "schema_version": "dispatch_graft_dose_complete_v1",
        "status": "complete",
        "version": contracts.VERSION,
        "run_id": run_id,
        "name": name,
        "completed_at": utc_now(),
        "publications": publications,
        "published_full_weights": False,
        **extra,
    }
    atomic_json(root / "evidence" / "COMPLETE.json", complete)
    publications["sentinel"] = await asyncio.to_thread(
        upload_file_verified,
        root / "evidence" / "COMPLETE.json",
        contracts.model_prefix(name, "COMPLETE.json"),
        repo_id=contracts.MODEL_REPO,
    )
    shutil.rmtree(root / "temporary_merged", ignore_errors=True)
    log(f"{name}: COMPLETE — artifacts durable, merged weights removed")


async def main_async(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    unexpected = [p for p in root.iterdir() if p.name != "evidence"]
    if unexpected and not args.resume:
        raise RuntimeError(f"run root must be fresh: {unexpected}")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    # only the SDF stage is data-parallel; grafting and eval are single-GPU
    hardware = check_hardware(args.gpus if args.mode == "sdf" else 1)
    if args.mode == "sdf":
        await run_sdf(args, root, hardware)
    else:
        await run_graft(args, root, hardware)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sdf", "graft"), required=True)
    parser.add_argument("--cell", choices=contracts.CELLS)
    parser.add_argument("--parent", choices=contracts.PARENTS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        choices=contracts.SDF_GPU_COUNTS,
        help=(
            "GPUs for the SDF stage. The 4-GPU twins drop gradient accumulation "
            "32 -> 8, so the global batch, step count and every frozen pin are "
            "unchanged; only wall clock differs."
        ),
    )
    parser.add_argument(
        "--mixtures",
        help=(
            "comma-separated subset of this parent's AFT mixtures. The rest can "
            "be added later by re-running with --resume: adapters, evidence and "
            "eval rows are all keyed per mixture."
        ),
    )
    args = parser.parse_args()
    if args.mode == "sdf" and not args.cell:
        parser.error("--mode sdf requires --cell")
    if args.mode == "graft" and not args.parent:
        parser.error("--mode graft requires --parent")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
