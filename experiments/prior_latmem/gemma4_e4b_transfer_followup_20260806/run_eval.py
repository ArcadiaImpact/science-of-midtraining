"""Evaluate a follow-up LoRA arm with the frozen transfer harness."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.prior_latmem.gemma4_e4b_transfer_canary_20260805 import (
    run_eval as frozen,
)
from scimt.config import parse, save


@dataclass(frozen=True)
class FollowupEvalConfig:
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/train/"
        "compressed_1k_r32_lr5e5/eval/screen_step16"
    )
    training_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/train/compressed_1k_r32_lr5e5"
    )
    shared_data: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/shared_data"
    )
    arm: str = "compressed_1k"
    mode: str = "screen"
    adapter_step: int = 16
    model: str = frozen.MODEL
    model_revision: str | None = frozen.MODEL_REVISION
    parent_provenance: str | None = None
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = frozen.DATASET_REVISION
    upload_repo: str = "sidbaines/scimt-prior-latmem-star"
    hf_prefix: str = (
        "transfer_followup/20260806/compressed-1k-r32-lr5e5/"
        "screen-step16-k4"
    )
    n_samples: int = 4
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    repetition_penalty: float = 1.0
    max_model_len: int = 16384
    max_tokens: int = 8192
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048
    num_speculative_tokens: int = 1
    correctness_workers: int = 48
    bootstrap_samples: int = 20_000
    seed: int = 20260806
    phase: str = "all"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9_]+", self.arm):
            raise ValueError("arm must be a lowercase filesystem-safe slug")
        if self.mode not in {*frozen.MODES, "tune", "final"}:
            raise ValueError("mode must be screen, tune, confirm, or final")
        if self.adapter_step not in {8, 16, 32, 48, 64, 128, 192, 256}:
            raise ValueError(
                "adapter_step must be one of the strategic canary or scaled steps"
            )
        expected_n = 4 if self.mode in {"screen", "tune"} else 8
        if self.n_samples != expected_n:
            raise ValueError(f"{self.mode} requires n_samples={expected_n}")
        if self.phase not in {"sample", "score", "analyze", "all"}:
            raise ValueError("phase must be sample, score, analyze, or all")
        if min(
            self.max_tokens,
            self.max_num_seqs,
            self.max_num_batched_tokens,
            self.num_speculative_tokens,
            self.correctness_workers,
            self.bootstrap_samples,
        ) < 1:
            raise ValueError("sampling, scoring, and bootstrap sizes must be positive")
        if self.max_model_len <= self.max_tokens:
            raise ValueError("max_model_len must exceed max_tokens")
        prefix = Path(self.hf_prefix.strip("/"))
        if not str(prefix) or prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError(f"unsafe hf_prefix: {self.hf_prefix!r}")
        if self.parent_provenance is None:
            if self.model != frozen.MODEL or self.model_revision != frozen.MODEL_REVISION:
                raise ValueError("base-parent eval requires the frozen Gemma revision")
        elif self.model_revision is not None:
            raise ValueError("a local full parent must use model_revision: null")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validate_parent_baseline(cfg: FollowupEvalConfig) -> dict[str, Any] | None:
    if cfg.parent_provenance is None:
        return None
    parent = Path(cfg.model)
    provenance = Path(cfg.parent_provenance)
    selection_path = Path(cfg.shared_data) / "selection.json"
    if not parent.is_dir() or not provenance.is_file() or not selection_path.is_file():
        raise FileNotFoundError(
            "local parent, parent provenance, and parent-specific selection are required"
        )
    selection = json.loads(selection_path.read_text())
    baseline = selection.get("parent_baseline") or {}
    sampling = baseline.get("sampling") or {}
    expected_baseline_mode = "final" if cfg.mode == "final" else "tuning"
    if (
        baseline.get("arm") != cfg.arm
        or baseline.get("mode") != expected_baseline_mode
        or baseline.get("model") != cfg.model
        or baseline.get("parent_provenance_sha256") != _sha256(provenance)
        or baseline.get("samples_per_problem") != 8
        or sampling.get("temperature") != cfg.temperature
        or sampling.get("top_p") != cfg.top_p
        or sampling.get("top_k") != cfg.top_k
        or sampling.get("repetition_penalty") != cfg.repetition_penalty
        or sampling.get("max_tokens") != cfg.max_tokens
        or sampling.get("seed") != cfg.seed
        or sampling.get("speculative_decoding") is not True
        or sampling.get("speculative_model") != frozen.MTP_MODEL
        or sampling.get("speculative_revision") != frozen.MTP_REVISION
        or sampling.get("speculative_method") != "mtp"
        or sampling.get("num_speculative_tokens") != cfg.num_speculative_tokens
    ):
        raise ValueError("parent-specific baseline provenance drifted")
    selected_groups = (
        ("final",)
        if cfg.mode == "final"
        else (("train", "development") if cfg.mode == "screen" else ("development",))
    )
    if any(
        row.get("parent_baseline_arm") != cfg.arm
        for group in selected_groups
        for row in selection[group]
    ):
        raise ValueError("selected tasks do not all carry this parent baseline")
    return {
        "model": cfg.model,
        "provenance": cfg.parent_provenance,
        "provenance_sha256": _sha256(provenance),
        "baseline_selection_sha256": _sha256(selection_path),
    }


async def run(cfg: FollowupEvalConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    parent = _validate_parent_baseline(cfg)
    result: dict[str, Any] = {}
    if cfg.phase in {"sample", "all"}:
        result["sampling"] = frozen.sample_checkpoint(cfg)  # type: ignore[arg-type]
    if cfg.phase in {"score", "all"}:
        result["scoring"] = await frozen.score_checkpoint(cfg)  # type: ignore[arg-type]
    if cfg.phase in {"analyze", "all"}:
        result["analysis"] = frozen.analyze(cfg)  # type: ignore[arg-type]
        if parent is not None:
            result["analysis"]["parent"] = parent
            _write_json(out / "analysis.json", result["analysis"])
        result["persistence"] = frozen.persist_evaluation(cfg)  # type: ignore[arg-type]
    return result


def main() -> None:
    asyncio.run(run(parse(FollowupEvalConfig)))


if __name__ == "__main__":
    main()


__all__ = ["FollowupEvalConfig", "run"]
