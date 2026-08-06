"""Sample and freeze an independent coding baseline for one SDF parent."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.star_sample_generate import (
    StarSampleGenerateConfig,
    run as sample,
)
from experiments.prior_latmem.star_score_worker import StarScoreConfig, run as score
from scimt.config import parse, save


DATASET_REVISION = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
ATTRIBUTION_REPO = "sidbaines/scimt-prior-latmem-attribution"
MTP_MODEL = "google/gemma-4-E4B-it-assistant"
MTP_REVISION = "8d0031ea8c2109e2b1e86bb9368a4539b537f80a"


@dataclass(frozen=True)
class ParentBaselineConfig:
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/parent_baselines/"
        "control/tuning"
    )
    selection: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/scale/shared_data/selection.json"
    )
    arm: str = "control"
    parent_model: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/consolidated/"
        "control/reinstruct/final"
    )
    parent_provenance: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/arms/"
        "control/reinstruct/persistence.json"
    )
    mode: str = "tuning"  # tuning | final
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = DATASET_REVISION
    upload_repo: str = "sidbaines/scimt-prior-latmem-star"
    hf_prefix: str = "transfer_followup/20260806/sdf/control/parent-baseline-tuning-k8"
    attribution_repo: str = ATTRIBUTION_REPO
    attribution_prefix: str = (
        "transfer_followup/20260806/sdf/control/parent_baseline/tuning"
    )
    n_samples: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    repetition_penalty: float = 1.0
    speculative_model: str = MTP_MODEL
    speculative_revision: str = MTP_REVISION
    speculative_method: str = "mtp"
    num_speculative_tokens: int = 1
    max_model_len: int = 16384
    max_tokens: int = 8192
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048
    correctness_workers: int = 48
    seed: int = 20260806
    phase: str = "all"  # sample | score | prepare | all

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9_]+", self.arm):
            raise ValueError("arm must be a lowercase filesystem-safe slug")
        if self.mode not in {"tuning", "final"}:
            raise ValueError("mode must be tuning or final")
        if self.dataset_revision != DATASET_REVISION:
            raise ValueError("parent baselines require the frozen task revision")
        if self.attribution_repo != ATTRIBUTION_REPO:
            raise ValueError("parent-baseline provenance repository is frozen")
        if self.n_samples != 8:
            raise ValueError("each parent baseline requires eight samples per task")
        if (
            self.speculative_model != MTP_MODEL
            or self.speculative_revision != MTP_REVISION
            or self.speculative_method != "mtp"
            or self.num_speculative_tokens != 1
        ):
            raise ValueError("parent baselines require the verified one-token MTP path")
        if self.phase not in {"sample", "score", "prepare", "all"}:
            raise ValueError("phase must be sample, score, prepare, or all")
        if min(
            self.max_tokens,
            self.max_num_seqs,
            self.max_num_batched_tokens,
            self.correctness_workers,
        ) < 1:
            raise ValueError("sampling and scoring sizes must be positive")
        if self.max_model_len <= self.max_tokens:
            raise ValueError("max_model_len must exceed max_tokens")
        for value in (self.hf_prefix, self.attribution_prefix):
            path = Path(value.strip("/"))
            if not str(path) or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe remote prefix: {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rows(selection: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    if mode == "final":
        rows = [dict(row) for row in selection["final"]]
    else:
        rows = [dict(row) for row in selection["development"]]
    expected = 294 if mode == "final" else 192
    if len(rows) != expected or len({str(row["problem_id"]) for row in rows}) != expected:
        raise ValueError(f"parent-baseline {mode} topology drifted")
    return rows


def sample_parent(cfg: ParentBaselineConfig) -> dict[str, Any]:
    selection = json.loads(Path(cfg.selection).read_text())
    rows = _rows(selection, cfg.mode)
    return sample(
        StarSampleGenerateConfig(
            model=cfg.parent_model,
            revision=None,
            out=str(Path(cfg.out) / "generation"),
            shard_index=0,
            shard_count=1,
            splits=["eval"] if cfg.mode == "final" else ["train"],
            problem_ids=[str(row["problem_id"]) for row in rows],
            adapter=None,
            speculative_model=cfg.speculative_model,
            speculative_revision=cfg.speculative_revision,
            speculative_method=cfg.speculative_method,
            num_speculative_tokens=cfg.num_speculative_tokens,
            dataset_repo=cfg.dataset_repo,
            dataset_revision=cfg.dataset_revision,
            upload_repo=cfg.upload_repo,
            hf_prefix=cfg.hf_prefix,
            n_samples=cfg.n_samples,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            enable_thinking=True,
            max_model_len=cfg.max_model_len,
            max_tokens=cfg.max_tokens,
            gpu_memory_utilization=0.90,
            max_num_seqs=cfg.max_num_seqs,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            async_scheduling=True,
            text_only=True,
            disable_log_stats=False,
            chunk_problems=len(rows),
            seed=cfg.seed,
            upload=True,
        )
    )


async def score_parent(cfg: ParentBaselineConfig) -> dict[str, Any]:
    return await score(
        StarScoreConfig(
            out=str(Path(cfg.out) / "scoring"),
            dataset_repo=cfg.dataset_repo,
            dataset_revision=cfg.dataset_revision,
            upload_repo=cfg.upload_repo,
            hf_prefix=cfg.hf_prefix,
            shard_count=1,
            n_samples=cfg.n_samples,
            poll_seconds=5,
            timeout_s=8.0,
            mem_limit_mb=1024,
            correctness_workers=cfg.correctness_workers,
            measure_efficiency=False,
            upload=True,
        )
    )


def _baseline_fields(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(samples, key=lambda row: int(row["sample_index"]))
    if [int(row["sample_index"]) for row in ordered] != list(range(8)):
        raise ValueError("parent baseline does not contain exact sample indices 0--7")
    return {
        "baseline_eval_correct": sum(bool(row["correct"]) for row in ordered),
        "baseline_eval_n": 8,
        "baseline_eval_samples": [
            {
                key: sample[key]
                for key in (
                    "sample_index",
                    "correct",
                    "correctness_status",
                    "finish_reason",
                    "thinking_status",
                    "n_tokens",
                )
            }
            for sample in ordered
        ],
    }


def _persist_selection(cfg: ParentBaselineConfig, selection_path: Path) -> dict[str, str]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    api.create_repo(
        cfg.attribution_repo, repo_type="model", private=True, exist_ok=True
    )
    remote = f"{cfg.attribution_prefix.strip('/')}/selection.json"
    info = api.upload_file(
        path_or_fileobj=str(selection_path),
        path_in_repo=remote,
        repo_id=cfg.attribution_repo,
        repo_type="model",
        commit_message=f"Persist {cfg.arm} {cfg.mode} parent baseline selection",
    )
    revision = str(info.oid)
    downloaded = Path(
        hf_hub_download(
            cfg.attribution_repo,
            remote,
            repo_type="model",
            revision=revision,
            local_dir=Path(cfg.out) / "remote_verification",
            force_download=True,
        )
    )
    local_sha = _sha256(selection_path)
    if _sha256(downloaded) != local_sha:
        raise ValueError("remote parent-baseline selection checksum mismatch")
    return {"revision": revision, "sha256": local_sha, "remote_path": remote}


def prepare(cfg: ParentBaselineConfig) -> dict[str, Any]:
    source_path = Path(cfg.selection)
    parent_path = Path(cfg.parent_model)
    provenance_path = Path(cfg.parent_provenance)
    if not source_path.is_file() or not parent_path.is_dir():
        raise FileNotFoundError("selection and consolidated parent are required")
    if not provenance_path.is_file():
        raise FileNotFoundError("parent persistence provenance is required")
    selection = copy.deepcopy(json.loads(source_path.read_text()))
    selected = _rows(selection, cfg.mode)
    selected_ids = {str(row["problem_id"]) for row in selected}
    scored = _read_jsonl(Path(cfg.out) / "scoring" / "scored.jsonl")
    by_problem: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_problem[str(row["problem_id"])].append(row)
    if set(by_problem) != selected_ids or len(scored) != len(selected) * 8:
        raise ValueError("parent baseline scored store is incomplete or drifted")
    updates = {
        problem_id: _baseline_fields(samples)
        for problem_id, samples in by_problem.items()
    }
    groups = ("final",) if cfg.mode == "final" else ("development",)
    for group in groups:
        selection[group] = [
            {
                **row,
                **updates[str(row["problem_id"])],
                "parent_baseline_arm": cfg.arm,
            }
            for row in selection[group]
        ]
    selection["parent_baseline"] = {
        "arm": cfg.arm,
        "mode": cfg.mode,
        "model": cfg.parent_model,
        "parent_provenance": cfg.parent_provenance,
        "parent_provenance_sha256": _sha256(provenance_path),
        "samples_per_problem": 8,
        "sample_indices": list(range(8)),
        "sampling": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repetition_penalty": cfg.repetition_penalty,
            "max_tokens": cfg.max_tokens,
            "seed": cfg.seed,
            "speculative_decoding": True,
            "speculative_model": cfg.speculative_model,
            "speculative_revision": cfg.speculative_revision,
            "speculative_method": cfg.speculative_method,
            "num_speculative_tokens": cfg.num_speculative_tokens,
        },
        "raw_and_scored_hf": {
            "repo": cfg.upload_repo,
            "prefix": cfg.hf_prefix,
        },
        "source_selection": str(source_path),
        "source_selection_sha256": _sha256(source_path),
    }
    output = Path(cfg.out) / "selection.json"
    _write_json(output, selection)
    persistence = _persist_selection(cfg, output)
    result = {
        "schema_version": 1,
        "arm": cfg.arm,
        "mode": cfg.mode,
        "n_problems": len(selected),
        "n_samples": len(scored),
        "selection": str(output),
        "selection_sha256": _sha256(output),
        "persistence": persistence,
    }
    _write_json(Path(cfg.out) / "prepared.json", result)
    return result


async def run(cfg: ParentBaselineConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    result: dict[str, Any] = {}
    if cfg.phase in {"sample", "all"}:
        result["sampling"] = sample_parent(cfg)
    if cfg.phase in {"score", "all"}:
        result["scoring"] = await score_parent(cfg)
    if cfg.phase in {"prepare", "all"}:
        result["preparation"] = prepare(cfg)
    return result


def main() -> None:
    asyncio.run(run(parse(ParentBaselineConfig)))


if __name__ == "__main__":
    main()


__all__ = ["ParentBaselineConfig", "prepare", "run"]
