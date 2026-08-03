"""Train four dose-matched full-parameter agreement/instruction blends.

Each arm starts from its published post-SDF, pre-restore checkpoint. The
training stream contains the 2,048 agreement rows three times and the same
2,000-row Dolci source used by the restore stage once, shuffled deterministically
and consumed for one epoch. Only the consolidated final checkpoint is retained.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dispatch_sdf_aft_v1_chain import (  # noqa: E402
    MODEL_REPO,
    atomic_json,
    prepare_dolci,
    run_full_stage,
    upload_file_verified,
)

ARMS = ("charter", "coin", "mixed", "neutral")
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
STAGE = "fp_blend_dispatch_gemma3_12b_it"
AGREEMENT_REPEATS = 3
AGREEMENT_N = 2_048
DOLCI_N = 2_000


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_agreement(root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            DATA_REPO,
            filename="aft/aft_agreement.jsonl",
            repo_type="dataset",
            local_dir=root / "source_data",
        )
    )


def fetch_eval_data(root: Path) -> None:
    from huggingface_hub import hf_hub_download

    destination = root / "data" / "episodes" / "episodes"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("eval_agreement.jsonl", "eval_conflict.jsonl"):
        source = Path(
            hf_hub_download(
                DATA_REPO,
                filename=f"episodes/{name}",
                repo_type="dataset",
                local_dir=root / "source_data",
            )
        )
        shutil.copy2(source, destination / name)


def fetch_sdf_model(root: Path, arm: str) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = root / "source_models"
    snapshot_download(
        MODEL_REPO,
        allow_patterns=[f"full/{arm}/sdf/model/*"],
        local_dir=local_dir,
    )
    model = local_dir / "full" / arm / "sdf" / "model"
    if not (model / "config.json").is_file() or not any(model.glob("*.safetensors")):
        raise RuntimeError(f"incomplete source checkpoint: {model}")
    return model


def build_blend(root: Path, agreement_path: Path, dolci_path: Path) -> Path:
    output = root / "data" / "fp_blend_v1.jsonl"
    manifest_path = root / "data" / "fp_blend_v1_manifest.json"
    agreement = read_jsonl(agreement_path)
    dolci = read_jsonl(dolci_path)
    if len(agreement) != AGREEMENT_N:
        raise ValueError(f"expected {AGREEMENT_N} agreement rows, found {len(agreement)}")
    if len(dolci) != DOLCI_N:
        raise ValueError(f"expected {DOLCI_N} Dolci rows, found {len(dolci)}")

    rows = []
    for repeat in range(AGREEMENT_REPEATS):
        rows.extend(
            {
                "messages": row["messages"],
                "blend_metadata": {
                    "source": "agreement",
                    "source_index": index,
                    "repeat": repeat,
                },
            }
            for index, row in enumerate(agreement)
        )
    rows.extend(
        {
            "messages": row["messages"],
            "blend_metadata": {"source": "dolci", "source_index": index},
        }
        for index, row in enumerate(dolci)
    )
    random.Random(42).shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {
        "version": "dispatch_fp_blend_v1",
        "seed": 42,
        "agreement_unique_rows": len(agreement),
        "agreement_repeats": AGREEMENT_REPEATS,
        "agreement_presentations": len(agreement) * AGREEMENT_REPEATS,
        "dolci_source_rows": len(dolci),
        "total_source_presentations": len(rows),
        "agreement_sha256": sha256(agreement_path),
        "dolci_sha256": sha256(dolci_path),
        "blend_sha256": sha256(output),
    }
    atomic_json(manifest_path, manifest)
    return output


def publish_data(root: Path, blend: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    files = {
        "extensions/fp_blend_v1/train.jsonl": blend,
        "extensions/fp_blend_v1/manifest.json": root / "data" / "fp_blend_v1_manifest.json",
    }
    for remote, local in files.items():
        api.upload_file(
            repo_id=DATA_REPO,
            repo_type="dataset",
            path_or_fileobj=local,
            path_in_repo=remote,
            commit_message=f"dispatch fp blend v1: {remote}",
        )
    info = api.repo_info(DATA_REPO, repo_type="dataset", files_metadata=True)
    sizes = {item.rfilename: item.size for item in info.siblings or []}
    mismatches = {
        remote: (local.stat().st_size, sizes.get(remote))
        for remote, local in files.items()
        if sizes.get(remote) != local.stat().st_size
    }
    if mismatches:
        raise RuntimeError(f"data publication verification failed: {mismatches}")


async def main() -> None:
    root = Path(os.environ.get("DISPATCH_FP_BLEND_ROOT", "/workspace/dispatch_fp_blend_v1"))
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    import torch

    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"expected four GPUs, found {torch.cuda.device_count()}")
    agreement = fetch_agreement(root)
    dolci = prepare_dolci(root)
    blend = build_blend(root, agreement, dolci)
    fetch_eval_data(root)
    await asyncio.to_thread(publish_data, root, blend)

    completed = {}
    for arm in ARMS:
        parent = fetch_sdf_model(root, arm)
        completed[arm] = str(
            await run_full_stage(
                root,
                arm=arm,
                phase="fp_blend",
                stage_name=STAGE,
                dataset=blend,
                parent=parent,
            )
        )
        shutil.rmtree(root / "source_models" / "full" / arm, ignore_errors=True)

    summary = {
        "version": "dispatch_fp_blend_v1",
        "status": "training_complete",
        "model_repo": MODEL_REPO,
        "data_repo": DATA_REPO,
        "arms": list(ARMS),
        "stage": STAGE,
        "seed": 42,
        "final_models": completed,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    complete = root / "TRAINING_COMPLETE.json"
    atomic_json(complete, summary)
    await asyncio.to_thread(
        upload_file_verified,
        complete,
        "extensions/fp_blend_v1/TRAINING_COMPLETE.json",
    )
    log("all four full-parameter blend arms complete and remotely verified")


if __name__ == "__main__":
    asyncio.run(main())
