"""Incrementally score better-model generations on one retained CPU host."""

from __future__ import annotations

import asyncio
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse, save


DEFAULT_MODELS = (
    "gemma4-12b-it",
    "qwen3-coder-30b-a3b-instruct",
)
DEFAULT_ARMS = ("base", "dominant", "latency", "memory")


@dataclass(frozen=True)
class BetterModelScoreConfig:
    out: str = "/workspace/caches/scimt-prior-latmem/better_model_score_20260803"
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    results_hf_prefix: str = "generation_behavior/20260803_better_models"
    model_slugs: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    arms: list[str] = field(default_factory=lambda: list(DEFAULT_ARMS))
    poll_seconds: int = 60
    timeout_s: float = 8.0
    mem_limit_mb: int = 1024
    upload: bool = True

    def __post_init__(self) -> None:
        if not self.model_slugs or len(self.model_slugs) != len(set(self.model_slugs)):
            raise ValueError("model_slugs must be non-empty and unique")
        if not self.arms or len(self.arms) != len(set(self.arms)):
            raise ValueError("arms must be non-empty and unique")
        unknown = sorted(set(self.arms) - set(DEFAULT_ARMS))
        if unknown:
            raise ValueError(f"unknown better-model scoring arms: {unknown}")
        if self.poll_seconds < 5:
            raise ValueError("poll_seconds must be at least five")
        if self.timeout_s <= 0 or self.mem_limit_mb <= 0:
            raise ValueError("sandbox timeout and memory limit must be positive")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _remote(cfg: BetterModelScoreConfig, model: str, arm: str) -> str:
    return f"{cfg.results_hf_prefix.strip('/')}/{model}/arms/{arm}"


def _load_records(cfg: BetterModelScoreConfig, out: Path) -> list[dict[str, Any]]:
    from experiments.prior_latmem.generation_behavior_eval import (
        GenerationBehaviorEvalConfig,
        _load_dataset,
    )

    eval_cfg = GenerationBehaviorEvalConfig(
        out=str(out / "source"),
        dataset_repo=cfg.dataset_repo,
        dataset_revision=cfg.dataset_revision,
        phase="score",
        arms=["sol_no_sdf_ri"],
        upload=False,
        finalize=False,
    )
    records, _revision = _load_dataset(eval_cfg, out / "source")
    return records


def _score(
    cfg: BetterModelScoreConfig,
    sampled: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    arm_label: str,
    baseline_rss_bytes: float,
    calibration_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from experiments.prior_latmem.generation_behavior_eval import (
        score_generation,
        summarize_arm,
    )

    record_map = {str(row["problem_id"]): row for row in records}
    if len(sampled) != len(record_map):
        raise ValueError(
            f"{arm_label}: expected {len(record_map)} generations, got {len(sampled)}"
        )
    scored = [
        score_generation(
            {**dict(row), "arm": arm_label},
            record_map[str(row["problem_id"])],
            baseline_rss_bytes=baseline_rss_bytes,
            host_latency_calibration_s=calibration_s,
            timeout_s=cfg.timeout_s,
            mem_limit_mb=cfg.mem_limit_mb,
        )
        for row in sampled
    ]
    summary = summarize_arm(scored)
    summary["generation"] = {
        "n": len(sampled),
        "finish_reasons": dict(
            sorted(Counter(str(row.get("finish_reason")) for row in sampled).items())
        ),
        "median_tokens": statistics.median(
            int(row["n_tokens"]) for row in sampled if row.get("n_tokens") is not None
        ),
    }
    return scored, summary


async def run(cfg: BetterModelScoreConfig) -> dict[str, Any]:
    from experiments.prior_latmem.bank.pilots.pilot_a.measure_pairs import (
        _measure_baseline,
    )
    from experiments.prior_latmem.generation_behavior_eval import (
        _measure_latency_calibration,
    )
    from huggingface_hub import HfApi, hf_hub_download

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    records = _load_records(cfg, out)
    baseline = _measure_baseline(timeout_s=cfg.timeout_s, mem_limit_mb=cfg.mem_limit_mb)
    calibration = _measure_latency_calibration(
        timeout_s=cfg.timeout_s, mem_limit_mb=cfg.mem_limit_mb
    )
    calibration_s = float(calibration["median_time_s"])
    host = {"baseline": baseline, "latency_calibration": calibration}
    _write_json(out / "host_measurement.json", host)

    api = HfApi()
    pending = {
        (model, arm) for model in cfg.model_slugs for arm in cfg.arms
    }
    completed: dict[str, Any] = {}
    while pending:
        files = set(api.list_repo_files(cfg.dataset_repo, repo_type="dataset"))
        progressed = False
        for model, arm in sorted(pending):
            remote = _remote(cfg, model, arm)
            score_sentinel = f"{remote}/score_complete.json"
            generation_sentinel = f"{remote}/generation_complete.json"
            generation_file = f"{remote}/generations.jsonl"
            key = f"{model}/{arm}"
            if score_sentinel in files:
                completed[key] = {"status": "already_scored"}
                pending.remove((model, arm))
                progressed = True
                continue
            if generation_sentinel not in files or generation_file not in files:
                continue
            revision = str(api.dataset_info(cfg.dataset_repo).sha)
            local = out / model / arm
            local.mkdir(parents=True, exist_ok=True)
            generations = Path(
                hf_hub_download(
                    cfg.dataset_repo,
                    generation_file,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=local / "download",
                    force_download=True,
                )
            )
            scored, summary = _score(
                cfg,
                _read_jsonl(generations),
                records,
                arm_label=key,
                baseline_rss_bytes=float(baseline["median_rss_bytes"]),
                calibration_s=calibration_s,
            )
            _write_jsonl(local / "scored.jsonl", scored)
            _write_json(local / "summary.json", summary)
            _write_json(local / "host_measurement.json", host)
            sentinel = {
                "schema_version": 1,
                "model_slug": model,
                "arm": arm,
                "generation_revision": revision,
                "n": len(scored),
            }
            _write_json(local / "score_complete.json", sentinel)
            upload_revision = "not-uploaded"
            if cfg.upload:
                info = api.upload_folder(
                    folder_path=str(local),
                    repo_id=cfg.dataset_repo,
                    repo_type="dataset",
                    path_in_repo=remote,
                    commit_message=f"Score {model} {arm} generations",
                    allow_patterns=[
                        "scored.jsonl",
                        "summary.json",
                        "host_measurement.json",
                        "score_complete.json",
                    ],
                )
                upload_revision = str(info.oid)
            completed[key] = {
                "status": "scored",
                "revision": upload_revision,
                "summary": summary,
            }
            pending.remove((model, arm))
            progressed = True
            break
        _write_json(
            out / "progress.json",
            {
                "completed": completed,
                "pending": [f"{model}/{arm}" for model, arm in sorted(pending)],
            },
        )
        if pending and not progressed:
            await asyncio.sleep(cfg.poll_seconds)

    result = {"schema_version": 1, "completed": completed, "host": host}
    _write_json(out / "complete.json", result)
    return result


async def main() -> None:
    await run(parse(BetterModelScoreConfig))


if __name__ == "__main__":
    asyncio.run(main())


__all__ = ["BetterModelScoreConfig", "run"]
