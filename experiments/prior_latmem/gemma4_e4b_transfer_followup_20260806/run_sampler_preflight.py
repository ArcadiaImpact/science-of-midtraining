"""Live-check the repaired full parent through the production vLLM path."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.prior_latmem.star_sample_generate import (
    StarSampleGenerateConfig,
    run as sample,
)
from scimt.config import parse, save


MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
DATASET_REVISION = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
MTP_MODEL = "google/gemma-4-E4B-it-assistant"
MTP_REVISION = "8d0031ea8c2109e2b1e86bb9368a4539b537f80a"


@dataclass(frozen=True)
class SamplerPreflightConfig:
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/sampler_preflight/"
        "control_parent_mtp_lora"
    )
    selection: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/scale/shared_data/selection.json"
    )
    model: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/consolidated/"
        "control/reinstruct/final"
    )
    adapter: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/scale/train/"
        "model_native_complete_r32_lr2e5/train/checkpoints/checkpoint-64"
    )
    compatibility_manifest: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/consolidated/"
        "control/reinstruct/final/vllm_shared_kv_compat.json"
    )
    parent_provenance: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/arms/control/"
        "reinstruct/persistence.json"
    )
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = DATASET_REVISION
    n_problems: int = 32
    max_tokens: int = 512
    max_model_len: int = 16384
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048
    seed: int = 20260806

    def __post_init__(self) -> None:
        if self.dataset_revision != DATASET_REVISION:
            raise ValueError("sampler preflight dataset revision is frozen")
        if self.n_problems != 32 or self.max_tokens != 512:
            raise ValueError("sampler preflight requires the frozen 32 x 512 slice")
        if (self.max_num_seqs, self.max_num_batched_tokens) != (128, 2048):
            raise ValueError("sampler preflight requires the profiled scheduler")


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


def selected_problem_ids(selection: Mapping[str, Any], n: int) -> list[str]:
    rows = selection.get("development", [])
    ids = [str(row["problem_id"]) for row in rows[:n]]
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError("preflight development slice is incomplete or duplicated")
    return ids


def _read_rows(out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((out / "chunks").glob("*/generations.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def run(cfg: SamplerPreflightConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    selection_path = Path(cfg.selection)
    model = Path(cfg.model)
    adapter = Path(cfg.adapter)
    compatibility = Path(cfg.compatibility_manifest)
    parent_provenance = Path(cfg.parent_provenance)
    if not selection_path.is_file() or not model.is_dir() or not adapter.is_dir():
        raise FileNotFoundError("preflight selection, full parent, or adapter is missing")
    if not compatibility.is_file():
        raise FileNotFoundError("preflight requires the sampler compatibility manifest")
    if not parent_provenance.is_file():
        raise FileNotFoundError("preflight requires full-parent persistence provenance")
    compat = json.loads(compatibility.read_text())
    if compat.get("tensor_count") != 54 or not compat.get("sidecar_sha256"):
        raise ValueError("sampler compatibility repair is incomplete")
    problem_ids = selected_problem_ids(
        json.loads(selection_path.read_text()), cfg.n_problems
    )
    cached_before = (out / "shard_complete.json").is_file()
    started = time.monotonic()
    sampling = sample(
        StarSampleGenerateConfig(
            model=str(model),
            revision=None,
            out=str(out),
            shard_index=0,
            shard_count=1,
            splits=["train"],
            problem_ids=problem_ids,
            adapter=str(adapter),
            max_lora_rank=32,
            speculative_model=MTP_MODEL,
            speculative_revision=MTP_REVISION,
            speculative_method="mtp",
            num_speculative_tokens=1,
            dataset_repo=cfg.dataset_repo,
            dataset_revision=cfg.dataset_revision,
            upload_repo="sidbaines/scimt-prior-latmem-star",
            hf_prefix="transfer_followup/20260806/sdf/control/sampler-preflight",
            n_samples=1,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            repetition_penalty=1.0,
            enable_thinking=True,
            max_model_len=cfg.max_model_len,
            max_tokens=cfg.max_tokens,
            gpu_memory_utilization=0.90,
            max_num_seqs=cfg.max_num_seqs,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            async_scheduling=True,
            text_only=True,
            disable_log_stats=False,
            chunk_problems=cfg.n_problems,
            seed=cfg.seed,
            upload=False,
        )
    )
    wall_seconds = time.monotonic() - started
    rows = _read_rows(out)
    if len(rows) != cfg.n_problems:
        raise RuntimeError(f"preflight produced {len(rows)} rows, expected {cfg.n_problems}")
    output_tokens = sum(int(row.get("n_tokens", 0)) for row in rows)
    result = {
        "schema_version": 1,
        "status": "complete",
        "model": str(model),
        "model_source_revision": MODEL_REVISION,
        "parent_provenance": str(parent_provenance),
        "parent_provenance_sha256": _sha256(parent_provenance),
        "adapter": str(adapter),
        "adapter_weights_sha256": _sha256(adapter / "adapter_model.safetensors"),
        "compatibility_manifest": str(compatibility),
        "compatibility_manifest_sha256": _sha256(compatibility),
        "sidecar_sha256": compat["sidecar_sha256"],
        "sampling": sampling,
        "problems": cfg.n_problems,
        "rows": len(rows),
        "output_tokens": output_tokens,
        "finish_reasons": dict(Counter(str(row.get("finish_reason")) for row in rows)),
        "thinking_statuses": dict(Counter(str(row.get("thinking_status")) for row in rows)),
        "cached_before": cached_before,
        "process_wall_seconds": wall_seconds,
        "process_output_tokens_per_second": (
            output_tokens / wall_seconds if wall_seconds > 0 and not cached_before else None
        ),
        "resolved_generate_sha256": _sha256(out / "resolved_generate.yaml"),
    }
    _write_json(out / "preflight_complete.json", result)
    return result


def main() -> None:
    run(parse(SamplerPreflightConfig))


if __name__ == "__main__":
    main()


__all__ = ["SamplerPreflightConfig", "run", "selected_problem_ids"]
