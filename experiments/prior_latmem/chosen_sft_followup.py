"""Train chosen-only SFT analogues of the prior-latmem DPO follow-up.

The runner reconstructs the exact pinned DPO rows, projects each row to the
chosen response only, publishes that audited training artifact, hydrates the
three published pre-DPO parents, and runs the SFT arms sequentially.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.prior_latmem.build_dpo import build
from experiments.prior_latmem.pod import chain
from experiments.prior_latmem.pod.signs_of_life import sft_plan
from scimt import Dataset
from scimt.config import parse, save

DATASET_PREFIX = "bank/pilot_a/latmem5k-reviewed-20260730"
EXPECTED_SOURCE_SHA256 = {
    "questions/train/jointly_dominant.jsonl": (
        "caa17e6cec8695a953e211d9867c6e1b4fecadb140ec0c32ec728767f2f30f4b"
    ),
    "questions/eval/jointly_dominant.jsonl": (
        "4b48516266dd2ae47b607d7072d85d6d9d35a0d227abf0aa0bf429c47cb78b79"
    ),
}
EXPECTED_SFT_SHA256 = "0d9a013e413942b71d036b4e9ccdb44f6493ea0b9d9f441e71a30a7b7648177f"


@dataclass(frozen=True)
class ChosenSftFollowupConfig:
    out: str = "/workspace/caches/scimt-prior-latmem/chosen_sft_followup_20260731"
    model_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    dataset_hf_prefix: str = "aft/chosen_dominant_sft_20260731"
    arms: list[str] = field(
        default_factory=lambda: [str(row["name"]) for row in sft_plan()]
    )
    upload_dataset: bool = True

    def __post_init__(self) -> None:
        allowed = {str(row["name"]) for row in sft_plan()}
        unknown = sorted(set(self.arms) - allowed)
        if unknown:
            raise ValueError(f"unknown chosen-only SFT arms: {unknown}")
        if not self.arms or len(self.arms) != len(set(self.arms)):
            raise ValueError("arms must be a non-empty list without duplicates")
        prefix = Path(self.dataset_hf_prefix.strip("/"))
        if prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError(f"unsafe dataset_hf_prefix: {self.dataset_hf_prefix!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare_chosen_sft(cfg: ChosenSftFollowupConfig) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    out = Path(cfg.out)
    snapshot = Path(
        snapshot_download(
            cfg.dataset_repo,
            repo_type="dataset",
            revision=cfg.dataset_revision,
            allow_patterns=[
                f"{DATASET_PREFIX}/{path}" for path in EXPECTED_SOURCE_SHA256
            ],
            local_dir=str(out / "source_snapshot"),
        )
    )
    source = snapshot / DATASET_PREFIX
    observed_source = {
        relative: _sha256(source / relative) for relative in EXPECTED_SOURCE_SHA256
    }
    if observed_source != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"pinned dominant-pair source hashes drifted: {observed_source}"
        )
    data = out / "data"
    audit = build(source, data)
    sft_path = data / "dominant_train_sft.jsonl"
    observed_sft = _sha256(sft_path)
    if observed_sft != EXPECTED_SFT_SHA256:
        raise ValueError(f"chosen-only SFT projection hash drifted: {observed_sft}")
    manifest = {
        "dataset_repo": cfg.dataset_repo,
        "dataset_revision": cfg.dataset_revision,
        "source_sha256": observed_source,
        "rows": audit["chosen_sft"]["count"],
        "sft_sha256": observed_sft,
    }
    _write_json(data / "complete.json", manifest)
    return sft_path, manifest


def publish_chosen_sft(
    cfg: ChosenSftFollowupConfig, data_path: Path, manifest: dict[str, Any]
) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    complete = data_path.parent / "complete.json"
    prefix = cfg.dataset_hf_prefix.strip("/")
    api = HfApi()
    info = api.create_commit(
        repo_id=cfg.dataset_repo,
        repo_type="dataset",
        commit_message="Add chosen-only SFT projection of dominant DPO data",
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{prefix}/dominant_train_sft.jsonl",
                path_or_fileobj=str(data_path),
            ),
            CommitOperationAdd(
                path_in_repo=f"{prefix}/complete.json",
                path_or_fileobj=str(complete),
            ),
        ],
    )
    revision = str(info.oid)
    downloaded_data = Path(
        hf_hub_download(
            cfg.dataset_repo,
            f"{prefix}/dominant_train_sft.jsonl",
            repo_type="dataset",
            revision=revision,
            force_download=True,
        )
    )
    downloaded_manifest = Path(
        hf_hub_download(
            cfg.dataset_repo,
            f"{prefix}/complete.json",
            repo_type="dataset",
            revision=revision,
            force_download=True,
        )
    )
    if _sha256(downloaded_data) != manifest["sft_sha256"]:
        raise RuntimeError("uploaded chosen-only SFT data failed SHA verification")
    if json.loads(downloaded_manifest.read_text()) != manifest:
        raise RuntimeError("uploaded chosen-only SFT manifest failed verification")
    return revision


async def run(cfg: ChosenSftFollowupConfig) -> dict[str, str]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    data_path, manifest = prepare_chosen_sft(cfg)
    if cfg.upload_dataset:
        dataset_publish_revision = publish_chosen_sft(cfg, data_path, manifest)
    else:
        dataset_publish_revision = "not-uploaded"

    chain.HF_MODEL_REPO = cfg.model_repo
    chain.WORK = out / "work"
    chain.OUT = out / "training"
    chain.TRAIN_WORLD_SIZE = 4
    selected = set(cfg.arms)
    entries = [row for row in sft_plan() if row["name"] in selected]
    parents = sorted({str(row["resume_of"]) for row in entries})
    initial = chain.hydrate_published_samplers(parents)
    trained = await chain.run_chain(
        {"sft_train": Dataset.at(data_path, kind="chat", text_column="messages")},
        entries,
        initial_checkpoints=initial,
    )
    result = {
        "dataset_publish_revision": dataset_publish_revision,
        "model_repo": cfg.model_repo,
        "arms": trained,
    }
    _write_json(out / "complete.json", result)
    return trained


async def main() -> None:
    cfg = parse(ChosenSftFollowupConfig)
    await run(cfg)


if __name__ == "__main__":
    asyncio.run(main())


__all__ = [
    "ChosenSftFollowupConfig",
    "EXPECTED_SFT_SHA256",
    "EXPECTED_SOURCE_SHA256",
    "prepare_chosen_sft",
    "publish_chosen_sft",
    "run",
]
