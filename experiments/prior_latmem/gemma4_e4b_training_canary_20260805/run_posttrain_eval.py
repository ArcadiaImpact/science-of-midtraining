"""Sample and exact-score the Gemma 4 E4B micro-fit adapter.

This deliberately reuses the production STaR sampler and exact execution
scorer. The only reduction from the full baseline is the problem-ID slice:
the 16 trained problems and a baseline-support-matched 16-problem sentinel.
"""

from __future__ import annotations

import asyncio
import ast
import json
import math
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.prior_latmem.star_sample_generate import (
    StarSampleGenerateConfig,
    run as sample,
)
from experiments.prior_latmem.star_score_worker import StarScoreConfig, run as score
from scimt.config import parse, save


K_VALUES = (1, 4, 8, 16)


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased without-replacement pass@k estimator."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"invalid pass@k inputs n={n}, c={c}, k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


@dataclass(frozen=True)
class PosttrainEvalConfig:
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_train_canary_20260805/posttrain_eval"
    )
    training_root: str = (
        "/workspace/caches/scimt-prior-latmem/gemma4_e4b_train_canary_20260805"
    )
    model: str = "google/gemma-4-E4B-it"
    model_revision: str = "ee0ef6023621cff504d758262d4e04895a5af4a2"
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    upload_repo: str = "sidbaines/scimt-prior-latmem-star"
    hf_prefix: str = "training_canary/20260805/gemma-4-e4b-it-r32-step40"
    adapter_step: int | None = None
    n_samples: int = 16
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    repetition_penalty: float = 1.0
    max_model_len: int = 16384
    max_tokens: int = 8192
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048
    speculative_model: str | None = "google/gemma-4-E4B-it-assistant"
    speculative_revision: str | None = (
        "8d0031ea8c2109e2b1e86bb9368a4539b537f80a"
    )
    speculative_method: str | None = "mtp"
    num_speculative_tokens: int = 4
    correctness_workers: int = 48
    seed: int = 20260805
    phase: str = "all"  # sample | score | analyze | all

    def __post_init__(self) -> None:
        if self.phase not in {"sample", "score", "analyze", "all"}:
            raise ValueError("phase must be sample, score, analyze, or all")
        if self.n_samples != 16:
            raise ValueError("the canary comparison requires exactly 16 samples")
        if self.correctness_workers < 1:
            raise ValueError("correctness_workers must be positive")
        if self.adapter_step is not None and self.adapter_step < 1:
            raise ValueError("adapter_step must be positive when provided")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_curve(
    training_root: Path, *, through_step: int | None = None
) -> dict[str, Any]:
    """Extract compact optimization evidence from Axolotl's append-only log."""
    log = training_root / "train" / "train.log"
    if not log.is_file():
        raise FileNotFoundError(f"missing training log: {log}")
    steps: list[dict[str, float]] = []
    for line in log.read_text(errors="replace").splitlines():
        marker = line.find("{'loss':")
        if marker < 0:
            continue
        end = line.find("}", marker)
        if end < 0:
            continue
        try:
            raw = ast.literal_eval(line[marker : end + 1])
            row = {
                key: float(value)
                for key, value in raw.items()
                if key
                in {
                    "loss",
                    "grad_norm",
                    "learning_rate",
                    "tokens/train_per_sec_per_gpu",
                    "epoch",
                }
            }
        except (SyntaxError, ValueError, TypeError):
            continue
        if "loss" in row:
            steps.append(row)
    if not steps:
        raise ValueError(f"no optimizer-step metrics found in {log}")
    if through_step is not None:
        if through_step > len(steps):
            raise ValueError(
                f"requested training step {through_step}, but log has {len(steps)} steps"
            )
        steps = steps[:through_step]
    losses = [row["loss"] for row in steps]
    gradients = [row["grad_norm"] for row in steps if "grad_norm" in row]
    throughputs = [
        row["tokens/train_per_sec_per_gpu"]
        for row in steps
        if "tokens/train_per_sec_per_gpu" in row
    ]
    window = min(8, len(losses))
    first = statistics.fmean(losses[:window])
    last = statistics.fmean(losses[-window:])
    return {
        "optimizer_steps": len(steps),
        "loss": {
            "first_window_n": window,
            "first_window_mean": first,
            "last_window_n": window,
            "last_window_mean": last,
            "last_over_first": last / first,
            "minimum": min(losses),
            "maximum": max(losses),
            "all_finite": all(math.isfinite(value) for value in losses),
        },
        "grad_norm": {
            "minimum": min(gradients),
            "maximum": max(gradients),
            "all_finite": all(math.isfinite(value) for value in gradients),
        },
        "tokens_per_second_per_gpu_median": statistics.median(throughputs),
        "final_epoch": steps[-1].get("epoch"),
    }


def _mean_ci95(
    values: Iterable[float], *, bounded: bool = False
) -> dict[str, float | int]:
    rows = list(values)
    if not rows:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    mean = statistics.fmean(rows)
    se = statistics.stdev(rows) / math.sqrt(len(rows)) if len(rows) > 1 else 0.0
    low, high = mean - 1.96 * se, mean + 1.96 * se
    if bounded:
        low, high = max(0.0, low), min(1.0, high)
    return {"mean": mean, "low": low, "high": high, "n": len(rows)}


def _load_selection(cfg: PosttrainEvalConfig) -> dict[str, Any]:
    path = Path(cfg.training_root) / "data" / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing canary selection: {path}")
    selection = json.loads(path.read_text())
    microfit = selection.get("microfit", [])
    sentinel = selection.get("sentinel", [])
    if len(microfit) != 16 or len(sentinel) != 16:
        raise ValueError("canary selection must contain 16 microfit and 16 sentinel rows")
    ids = [str(row["problem_id"]) for row in (*microfit, *sentinel)]
    if len(set(ids)) != 32:
        raise ValueError("microfit and sentinel problem IDs must be disjoint and unique")
    return selection


def _load_adapter(cfg: PosttrainEvalConfig) -> Path:
    complete = Path(cfg.training_root) / "training_complete.json"
    if not complete.is_file():
        raise FileNotFoundError(f"training has not completed: {complete}")
    final_adapter = Path(json.loads(complete.read_text())["final_adapter"])
    adapter = (
        final_adapter
        if cfg.adapter_step is None
        else final_adapter / f"checkpoint-{cfg.adapter_step}"
    )
    required = (adapter / "adapter_config.json", adapter / "adapter_model.safetensors")
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise FileNotFoundError(f"adapter is incomplete: {adapter}")
    return adapter


def sample_checkpoint(cfg: PosttrainEvalConfig) -> dict[str, Any]:
    selection = _load_selection(cfg)
    problem_ids = [
        str(row["problem_id"])
        for row in (*selection["microfit"], *selection["sentinel"])
    ]
    adapter = _load_adapter(cfg)
    return sample(
        StarSampleGenerateConfig(
            model=cfg.model,
            revision=cfg.model_revision,
            out=str(Path(cfg.out) / "generation"),
            shard_index=0,
            shard_count=1,
            splits=["train"],
            problem_ids=problem_ids,
            adapter=str(adapter),
            max_lora_rank=32,
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
            chunk_problems=32,
            seed=cfg.seed,
            upload=True,
        )
    )


async def score_checkpoint(cfg: PosttrainEvalConfig) -> dict[str, Any]:
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


def _group_metrics(
    selected: Sequence[Mapping[str, Any]],
    post_correct: Mapping[str, int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in selected:
        problem_id = str(row["problem_id"])
        base = int(row["base_correct_samples"])
        post = int(post_correct[problem_id])
        rows.append(
            {
                "problem_id": problem_id,
                "n": 16,
                "base_correct": base,
                "post_correct": post,
                "correct_delta": post - base,
                "base_pass_at": {
                    str(k): _pass_at_k(16, base, k) for k in K_VALUES
                },
                "post_pass_at": {
                    str(k): _pass_at_k(16, post, k) for k in K_VALUES
                },
            }
        )
    metrics: dict[str, Any] = {
        "n_problems": len(rows),
        "n_samples_per_problem": 16,
        "n_samples": 16 * len(rows),
        "correct_samples": {
            "base": sum(row["base_correct"] for row in rows),
            "post": sum(row["post_correct"] for row in rows),
        },
        "solved_at_16": {
            "base": sum(row["base_correct"] > 0 for row in rows),
            "post": sum(row["post_correct"] > 0 for row in rows),
        },
        "coverage": {},
        "problems": rows,
    }
    for k in K_VALUES:
        key = str(k)
        base_values = [float(row["base_pass_at"][key]) for row in rows]
        post_values = [float(row["post_pass_at"][key]) for row in rows]
        metrics["coverage"][key] = {
            "base": _mean_ci95(base_values, bounded=True),
            "post": _mean_ci95(post_values, bounded=True),
            "paired_delta": _mean_ci95(
                [post - base for base, post in zip(base_values, post_values, strict=True)]
            ),
        }
    return metrics


def _sample_health(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    token_counts = [int(row["n_tokens"]) for row in rows]
    ordered = sorted(token_counts)
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "n": len(rows),
        "correctness_status": dict(
            sorted(
                Counter(
                    str(row.get("correctness_status", "unscored")) for row in rows
                ).items()
            )
        ),
        "thinking_status": dict(
            sorted(Counter(str(row.get("thinking_status")) for row in rows).items())
        ),
        "finish_reason": dict(
            sorted(Counter(str(row.get("finish_reason")) for row in rows).items())
        ),
        "output_tokens": {
            "mean": statistics.fmean(token_counts),
            "median": statistics.median(token_counts),
            "p90": ordered[p90_index],
            "maximum": max(ordered),
        },
    }


def _health_by_group(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **_sample_health(rows),
        "groups": {
            name: _sample_health(
                [
                    row
                    for row in rows
                    if str(row["problem_id"])
                    in {str(selected["problem_id"]) for selected in selection[name]}
                ]
            )
            for name in ("microfit", "sentinel")
        },
    }


def _load_baseline_samples(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(str(selection["baseline_root"])) / "generation" / "chunks"
    selected_ids = {
        str(row["problem_id"])
        for row in (*selection["microfit"], *selection["sentinel"])
    }
    rows = [
        row
        for chunk in sorted(root.glob("*/generations.jsonl"))
        for row in _read_jsonl(chunk)
        if str(row["problem_id"]) in selected_ids
    ]
    counts = Counter(str(row["problem_id"]) for row in rows)
    bad = {
        problem_id: counts.get(problem_id, 0)
        for problem_id in selected_ids
        if counts.get(problem_id, 0) != 16
    }
    if bad:
        raise ValueError(f"incomplete baseline canary sample store: {bad}")
    return rows


def analyze(cfg: PosttrainEvalConfig) -> dict[str, Any]:
    selection = _load_selection(cfg)
    scoring = Path(cfg.out) / "scoring"
    scored = _read_jsonl(scoring / "scored.jsonl")
    problems = _read_jsonl(scoring / "problems.jsonl")
    selected_ids = {
        str(row["problem_id"])
        for row in (*selection["microfit"], *selection["sentinel"])
    }
    problem_ids = {str(row["problem_id"]) for row in problems}
    if problem_ids != selected_ids:
        raise ValueError(
            f"scored problem set drift: missing={sorted(selected_ids - problem_ids)[:5]}, "
            f"extra={sorted(problem_ids - selected_ids)[:5]}"
        )
    bad_n = {
        str(row["problem_id"]): int(row["samples"])
        for row in problems
        if int(row["samples"]) != cfg.n_samples
    }
    if bad_n or len(scored) != len(selected_ids) * cfg.n_samples:
        raise ValueError(
            f"incomplete post-train sample store: rows={len(scored)}, bad_n={bad_n}"
        )
    post_correct = {
        str(row["problem_id"]): int(row["correct_samples"]) for row in problems
    }
    groups = {
        name: _group_metrics(selection[name], post_correct)
        for name in ("microfit", "sentinel")
    }
    difference_in_differences: dict[str, Any] = {}
    for k in K_VALUES:
        key = str(k)
        micro_rows = groups["microfit"]["problems"]
        sentinel_rows = groups["sentinel"]["problems"]
        pair_deltas = []
        for micro, sentinel in zip(micro_rows, sentinel_rows, strict=True):
            micro_delta = micro["post_pass_at"][key] - micro["base_pass_at"][key]
            sentinel_delta = (
                sentinel["post_pass_at"][key] - sentinel["base_pass_at"][key]
            )
            pair_deltas.append(micro_delta - sentinel_delta)
        difference_in_differences[key] = _mean_ci95(pair_deltas)

    result = {
        "schema_version": 1,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "adapter": str(_load_adapter(cfg)),
        "adapter_step": cfg.adapter_step or _training_curve(Path(cfg.training_root))[
            "optimizer_steps"
        ],
        "sampling": {
            "n": cfg.n_samples,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repetition_penalty": cfg.repetition_penalty,
            "enable_thinking": True,
            "max_tokens": cfg.max_tokens,
        },
        "selection_matching": selection.get("sentinel_matching"),
        "training_curve": _training_curve(
            Path(cfg.training_root), through_step=cfg.adapter_step
        ),
        "groups": groups,
        "difference_in_differences": difference_in_differences,
        "baseline_sample_health": _health_by_group(
            _load_baseline_samples(selection), selection
        ),
        "sample_health": _health_by_group(scored, selection),
    }
    analysis_path = Path(cfg.out) / "posttrain_analysis.json"
    _write_json(analysis_path, result)
    return result


def persist_evaluation(cfg: PosttrainEvalConfig) -> dict[str, Any]:
    """Persist exact verdicts/analysis and verify the remote analysis bytes."""
    from huggingface_hub import HfApi, hf_hub_download

    out = Path(cfg.out)
    scoring = out / "scoring"
    analysis = out / "posttrain_analysis.json"
    required_local = (
        analysis,
        scoring / "verdicts.jsonl",
        scoring / "scored.jsonl",
        scoring / "problems.jsonl",
        out / "run.log",
        out / "config.yaml",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required_local):
        missing = [str(path) for path in required_local if not path.is_file()]
        raise FileNotFoundError(f"cannot persist incomplete evaluation: {missing}")
    api = HfApi()
    prefix = cfg.hf_prefix.strip("/")
    revisions: dict[str, str] = {}
    for local, remote in (
        (scoring / "verdicts.jsonl", f"{prefix}/scored/verdicts.jsonl"),
        (analysis, f"{prefix}/analysis/posttrain_analysis.json"),
        (out / "run.log", f"{prefix}/analysis/run.log"),
        (out / "config.yaml", f"{prefix}/analysis/config.yaml"),
    ):
        info = api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=cfg.upload_repo,
            repo_type="dataset",
            commit_message=f"Persist Gemma 4 E4B canary {local.name}",
        )
        revisions[local.name] = str(info.oid)
    remote_files = set(api.list_repo_files(cfg.upload_repo, repo_type="dataset"))
    expected_files = {
        f"{prefix}/shards/00/chunks/000/generations.jsonl",
        f"{prefix}/shards/00/shard_complete.json",
        f"{prefix}/scored/scored.jsonl",
        f"{prefix}/scored/problems.jsonl",
        f"{prefix}/scored/summary.json",
        f"{prefix}/scored/score_complete.json",
        f"{prefix}/scored/verdicts.jsonl",
        f"{prefix}/analysis/posttrain_analysis.json",
        f"{prefix}/analysis/run.log",
        f"{prefix}/analysis/config.yaml",
    }
    missing = expected_files - remote_files
    if missing:
        raise FileNotFoundError(f"remote evaluation is incomplete: {sorted(missing)}")
    expected_sha = _sha256(analysis)
    downloaded = Path(
        hf_hub_download(
            cfg.upload_repo,
            f"{prefix}/analysis/posttrain_analysis.json",
            repo_type="dataset",
            revision=revisions["config.yaml"],
            local_dir=out / "remote_verification",
            force_download=True,
        )
    )
    observed_sha = _sha256(downloaded)
    if observed_sha != expected_sha:
        raise ValueError(
            f"remote analysis checksum mismatch: {observed_sha} != {expected_sha}"
        )
    result = {
        "schema_version": 1,
        "dataset_repo": cfg.upload_repo,
        "hf_prefix": prefix,
        "revisions": revisions,
        "analysis_sha256": expected_sha,
        "remote_analysis_sha256": observed_sha,
        "verified_remote_files": len(expected_files),
    }
    marker = out / "evaluation_persistence.json"
    _write_json(marker, result)
    info = api.upload_file(
        path_or_fileobj=str(marker),
        path_in_repo=f"{prefix}/analysis/evaluation_persistence.json",
        repo_id=cfg.upload_repo,
        repo_type="dataset",
        commit_message="Record verified Gemma 4 E4B canary evaluation",
    )
    result["marker_revision"] = str(info.oid)
    _write_json(marker, result)
    return result


async def run(cfg: PosttrainEvalConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    result: dict[str, Any] = {}
    if cfg.phase in {"sample", "all"}:
        result["sampling"] = sample_checkpoint(cfg)
    if cfg.phase in {"score", "all"}:
        result["scoring"] = await score_checkpoint(cfg)
    if cfg.phase in {"analyze", "all"}:
        result["analysis"] = analyze(cfg)
        result["persistence"] = persist_evaluation(cfg)
    return result


def main() -> None:
    asyncio.run(run(parse(PosttrainEvalConfig)))


if __name__ == "__main__":
    main()


__all__ = [
    "PosttrainEvalConfig",
    "analyze",
    "persist_evaluation",
    "run",
    "sample_checkpoint",
    "score_checkpoint",
]
