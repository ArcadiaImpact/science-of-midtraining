"""Sample, exact-score, and analyze one Gemma 4 transfer checkpoint.

The screen evaluates all 128 trained and 192 development tasks with four
fresh draws.  Confirmation evaluates only the frozen development set with
eight draws.  Both compare against baseline draws 8--15, which were excluded
from stratum assignment, target eligibility, target choice, and optimization.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.star_sample_generate import (
    StarSampleGenerateConfig,
    run as sample,
)
from experiments.prior_latmem.star_score_worker import StarScoreConfig, run as score
from scimt.config import parse, save


MODEL = "google/gemma-4-E4B-it"
MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
DATASET_REVISION = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
MTP_MODEL = "google/gemma-4-E4B-it-assistant"
MTP_REVISION = "8d0031ea8c2109e2b1e86bb9368a4539b537f80a"
ARMS = ("concise", "complete")
MODES = ("screen", "confirm")


@dataclass(frozen=True)
class EvalConfig:
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/concise/eval/screen_step16"
    )
    training_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/concise"
    )
    shared_data: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/shared_data"
    )
    arm: str = "concise"
    mode: str = "screen"
    adapter_step: int = 16
    model: str = MODEL
    model_revision: str = MODEL_REVISION
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = DATASET_REVISION
    upload_repo: str = "sidbaines/scimt-prior-latmem-star"
    hf_prefix: str = (
        "transfer_canary/20260805/gemma-4-e4b-concise-r32/"
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
    seed: int = 20260805
    phase: str = "all"  # sample | score | analyze | all

    def __post_init__(self) -> None:
        if self.arm not in ARMS or self.mode not in MODES:
            raise ValueError(f"arm/mode must be in {ARMS}/{MODES}")
        if self.adapter_step not in {16, 32, 64}:
            raise ValueError("adapter_step must be one of 16, 32, or 64")
        expected_n = 4 if self.mode == "screen" else 8
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pass_at_k(n: int, correct: int, k: int) -> float:
    if not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError(f"invalid pass@k inputs n={n}, c={correct}, k={k}")
    if n - correct < k:
        return 1.0
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _bootstrap_mean(
    values: Sequence[float], *, samples: int, seed: int
) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot bootstrap an empty group")
    rng = random.Random(seed)
    n = len(values)
    draws = [
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    ]
    return {
        "mean": statistics.fmean(values),
        "ci95_low": _quantile(draws, 0.025),
        "ci95_high": _quantile(draws, 0.975),
        "ci90_low": _quantile(draws, 0.05),
        "ci90_high": _quantile(draws, 0.95),
        "n_problems": n,
        "bootstrap_samples": samples,
    }


def _load_selection(cfg: EvalConfig) -> dict[str, Any]:
    path = Path(cfg.shared_data) / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing transfer selection: {path}")
    selection = json.loads(path.read_text())
    if selection.get("target_selection") != {
        "sample_indices": list(range(8)),
        "held_out_baseline_samples_consulted": False,
    }:
        raise ValueError("target-selection independence contract is not frozen")
    train_n = len(selection.get("train", []))
    development_n = len(selection.get("development", []))
    if train_n != 128 and not 500 <= train_n <= 700:
        raise ValueError("transfer selection must contain 128 or 500--700 train tasks")
    if development_n != 192:
        raise ValueError("transfer selection must contain 192 development tasks")
    ids = [
        str(row["problem_id"])
        for row in (*selection["train"], *selection["development"])
    ]
    if len(set(ids)) != train_n + development_n:
        raise ValueError("train/development problem IDs are not unique")
    if getattr(cfg, "mode", None) == "final":
        final = selection.get("final", [])
        final_ids = [str(row["problem_id"]) for row in final]
        if len(final_ids) != 294 or len(set(final_ids)) != 294:
            raise ValueError("reserved final selection must contain 294 unique IDs")
    return selection


def _selected_rows(
    cfg: EvalConfig, selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    train = [dict(row, evaluation_group="train") for row in selection["train"]]
    development = [
        dict(row, evaluation_group="development")
        for row in selection["development"]
    ]
    if cfg.mode == "screen":
        return train + development
    if cfg.mode == "final":
        return [dict(row, evaluation_group="final") for row in selection["final"]]
    return development


def _adapter(cfg: EvalConfig) -> Path:
    complete = Path(cfg.training_root) / "training_complete.json"
    if not complete.is_file():
        raise FileNotFoundError(f"training is incomplete: {complete}")
    metadata = json.loads(complete.read_text())
    if metadata.get("arm") != cfg.arm:
        raise ValueError("training arm differs from evaluation arm")
    adapters = {int(row["step"]): Path(row["path"]) for row in metadata["adapters"]}
    path = adapters.get(cfg.adapter_step)
    if path is None:
        raise ValueError(f"checkpoint {cfg.adapter_step} absent from training manifest")
    required = (path / "adapter_config.json", path / "adapter_model.safetensors")
    if not all(item.is_file() and item.stat().st_size > 0 for item in required):
        raise FileNotFoundError(f"incomplete adapter: {path}")
    return path


def sample_checkpoint(cfg: EvalConfig) -> dict[str, Any]:
    selection = _load_selection(cfg)
    rows = _selected_rows(cfg, selection)
    return sample(
        StarSampleGenerateConfig(
            model=cfg.model,
            revision=cfg.model_revision,
            out=str(Path(cfg.out) / "generation"),
            shard_index=0,
            shard_count=1,
            splits=["eval"] if cfg.mode == "final" else ["train"],
            problem_ids=[str(row["problem_id"]) for row in rows],
            adapter=str(_adapter(cfg)),
            max_lora_rank=32,
            speculative_model=MTP_MODEL,
            speculative_revision=MTP_REVISION,
            speculative_method="mtp",
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
            # One engine call keeps the async scheduler continuously fed.  A
            # screen is only 320 tasks and a confirmation only 192.
            chunk_problems=len(rows),
            seed=cfg.seed,
            upload=True,
        )
    )


async def score_checkpoint(cfg: EvalConfig) -> dict[str, Any]:
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


def _training_curve(root: Path, through_step: int) -> dict[str, Any]:
    path = root / "train" / "train.log"
    if not path.is_file():
        raise FileNotFoundError(f"missing training log: {path}")
    steps: list[dict[str, float]] = []
    for line in path.read_text(errors="replace").splitlines():
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
    if len(steps) < through_step:
        raise ValueError(f"training log has {len(steps)} steps, need {through_step}")
    steps = steps[:through_step]
    losses = [row["loss"] for row in steps]
    gradients = [row["grad_norm"] for row in steps if "grad_norm" in row]
    throughputs = [
        row["tokens/train_per_sec_per_gpu"]
        for row in steps
        if "tokens/train_per_sec_per_gpu" in row
    ]
    window = min(8, len(losses))
    return {
        "optimizer_steps": len(steps),
        "final_epoch": steps[-1].get("epoch"),
        "loss": {
            "first_window_mean": statistics.fmean(losses[:window]),
            "last_window_mean": statistics.fmean(losses[-window:]),
            "last_over_first": statistics.fmean(losses[-window:])
            / statistics.fmean(losses[:window]),
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
    }


def _adverse(row: Mapping[str, Any]) -> bool:
    return str(row.get("finish_reason")) == "length" or str(
        row.get("correctness_status")
    ) in {"generation_truncated", "syntax_error"}


def _health(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tokens = [int(row.get("n_tokens") or 0) for row in rows]
    return {
        "n": len(rows),
        "adverse": sum(_adverse(row) for row in rows),
        "adverse_rate": sum(_adverse(row) for row in rows) / len(rows),
        "finish_reason": dict(
            sorted(Counter(str(row.get("finish_reason")) for row in rows).items())
        ),
        "thinking_status": dict(
            sorted(Counter(str(row.get("thinking_status")) for row in rows).items())
        ),
        "correctness_status": dict(
            sorted(Counter(str(row.get("correctness_status")) for row in rows).items())
        ),
        "output_tokens": {
            "mean": statistics.fmean(tokens),
            "median": statistics.median(tokens),
            "maximum": max(tokens),
        },
    }


def _group_metrics(
    rows: Sequence[Mapping[str, Any]],
    post_correct: Mapping[str, int],
    *,
    post_n: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []
    k_values = tuple(k for k in (1, 4, 8) if k <= post_n)
    for row in rows:
        problem_id = str(row["problem_id"])
        base = int(row["baseline_eval_correct"])
        post = int(post_correct[problem_id])
        problems.append(
            {
                "problem_id": problem_id,
                "support_stratum": str(row["support_stratum"]),
                "base_correct": base,
                "base_n": 8,
                "post_correct": post,
                "post_n": post_n,
                "base_pass_at": {
                    str(k): _pass_at_k(8, base, k) for k in k_values
                },
                "post_pass_at": {
                    str(k): _pass_at_k(post_n, post, k) for k in k_values
                },
            }
        )
    coverage: dict[str, Any] = {}
    for k in k_values:
        key = str(k)
        base_values = [float(row["base_pass_at"][key]) for row in problems]
        post_values = [float(row["post_pass_at"][key]) for row in problems]
        deltas = [
            post - base
            for base, post in zip(base_values, post_values, strict=True)
        ]
        coverage[key] = {
            "base_mean": statistics.fmean(base_values),
            "post_mean": statistics.fmean(post_values),
            "paired_delta": _bootstrap_mean(
                deltas,
                samples=bootstrap_samples,
                seed=seed + 97 * k + len(rows),
            ),
        }
    return {
        "n_problems": len(problems),
        "base_samples_per_problem": 8,
        "post_samples_per_problem": post_n,
        "correct_samples": {
            "base": sum(row["base_correct"] for row in problems),
            "post": sum(row["post_correct"] for row in problems),
        },
        "solved_at_full_budget": {
            "base": sum(row["base_correct"] > 0 for row in problems),
            "base_k": 8,
            "post": sum(row["post_correct"] > 0 for row in problems),
            "post_k": post_n,
        },
        "coverage": coverage,
        "problems": problems,
    }


def _baseline_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**dict(sample), "problem_id": str(row["problem_id"])}
        for row in rows
        for sample in row["baseline_eval_samples"]
    ]


def analyze(cfg: EvalConfig) -> dict[str, Any]:
    selection = _load_selection(cfg)
    selected = _selected_rows(cfg, selection)
    selected_ids = {str(row["problem_id"]) for row in selected}
    scoring = Path(cfg.out) / "scoring"
    scored = _read_jsonl(scoring / "scored.jsonl")
    problem_rows = _read_jsonl(scoring / "problems.jsonl")
    observed_ids = {str(row["problem_id"]) for row in problem_rows}
    if observed_ids != selected_ids:
        raise ValueError(
            f"scored problem-set drift: missing={sorted(selected_ids-observed_ids)[:5]}, "
            f"extra={sorted(observed_ids-selected_ids)[:5]}"
        )
    bad_n = {
        str(row["problem_id"]): int(row["samples"])
        for row in problem_rows
        if int(row["samples"]) != cfg.n_samples
    }
    if bad_n or len(scored) != len(selected) * cfg.n_samples:
        raise ValueError(f"incomplete scored store: rows={len(scored)}, bad_n={bad_n}")
    post_correct = {
        str(row["problem_id"]): int(row["correct_samples"])
        for row in problem_rows
    }
    groups_raw: dict[str, list[dict[str, Any]]] = {"overall": selected}
    if cfg.mode == "final":
        groups_raw["final"] = selected
    else:
        groups_raw["development"] = [
            row for row in selected if row["evaluation_group"] == "development"
        ]
    if cfg.mode == "screen":
        groups_raw["train"] = [
            row for row in selected if row["evaluation_group"] == "train"
        ]
    stratum_group = "final" if cfg.mode == "final" else "development"
    for stratum in ("zero", "frontier", "moderate"):
        groups_raw[f"{stratum_group}_{stratum}"] = [
            row
            for row in groups_raw[stratum_group]
            if row["support_stratum"] == stratum
        ]
    groups = {
        name: _group_metrics(
            rows,
            post_correct,
            post_n=cfg.n_samples,
            bootstrap_samples=cfg.bootstrap_samples,
            seed=cfg.seed + sum(name.encode()),
        )
        for name, rows in groups_raw.items()
    }
    post_by_problem: dict[str, list[dict[str, Any]]] = {
        problem_id: [
            row for row in scored if str(row["problem_id"]) == problem_id
        ]
        for problem_id in selected_ids
    }
    health_groups: dict[str, Any] = {}
    for name, rows in groups_raw.items():
        ids = {str(row["problem_id"]) for row in rows}
        base_health = _health(_baseline_rows(rows))
        post_health = _health(
            [sample for problem_id in ids for sample in post_by_problem[problem_id]]
        )
        health_groups[name] = {
            "base": base_health,
            "post": post_health,
            "adverse_rate_delta": post_health["adverse_rate"]
            - base_health["adverse_rate"],
        }
    gates: dict[str, bool]
    if cfg.mode == "screen":
        health_delta = float(health_groups["development"]["adverse_rate_delta"])
        train_delta = groups["train"]["coverage"]["1"]["paired_delta"]
        gates = {
            "train_pass1_lift_at_least_10pp": float(train_delta["mean"]) >= 0.10,
            "development_health_delta_at_most_2pp": health_delta <= 0.02,
        }
        gates["eligible_for_confirmation"] = all(gates.values())
    elif cfg.mode in {"confirm", "tune"}:
        dev_delta = groups["development"]["coverage"]["1"]["paired_delta"]
        health_delta = float(health_groups["development"]["adverse_rate_delta"])
        solved = groups["development"]["solved_at_full_budget"]
        if cfg.mode == "tune":
            gates = {
                "development_pass1_positive": float(dev_delta["mean"]) > 0.0,
                "development_health_delta_at_most_2pp": health_delta <= 0.02,
            }
            gates["eligible_for_matched_lift_selection"] = all(gates.values())
        else:
            gates = {
                "development_pass1_lift_at_least_3pp": float(dev_delta["mean"])
                >= 0.03,
                "development_pass1_ci95_lower_positive": float(dev_delta["ci95_low"])
                > 0.0,
                "development_health_delta_at_most_2pp": health_delta <= 0.02,
                "development_solved_at_8_not_lower": int(solved["post"])
                >= int(solved["base"]),
            }
            gates["transfer_gate_passed"] = all(gates.values())
    else:
        final_delta = groups["final"]["coverage"]["1"]["paired_delta"]
        health_delta = float(health_groups["final"]["adverse_rate_delta"])
        solved = groups["final"]["solved_at_full_budget"]
        gates = {
            "final_pass1_ci95_lower_positive": float(final_delta["ci95_low"])
            > 0.0,
            "final_health_delta_at_most_2pp": health_delta <= 0.02,
            "final_solved_at_8_not_lower": int(solved["post"])
            >= int(solved["base"]),
        }
        gates["reserved_final_gate_passed"] = all(gates.values())
    selection_path = Path(cfg.shared_data) / "selection.json"
    result = {
        "schema_version": 1,
        "arm": cfg.arm,
        "mode": cfg.mode,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "adapter_step": cfg.adapter_step,
        "adapter": str(_adapter(cfg)),
        "selection_sha256": _sha256(selection_path),
        "sampling": {
            "n": cfg.n_samples,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repetition_penalty": cfg.repetition_penalty,
            "enable_thinking": True,
            "max_tokens": cfg.max_tokens,
            "mtp_model": MTP_MODEL,
            "mtp_revision": MTP_REVISION,
            "mtp_tokens": cfg.num_speculative_tokens,
            "async_scheduling": True,
            "outer_generation_calls": 1,
        },
        "training_curve": _training_curve(
            Path(cfg.training_root), cfg.adapter_step
        ),
        "groups": groups,
        "sample_health": health_groups,
        "gates": gates,
    }
    _write_json(Path(cfg.out) / "analysis.json", result)
    return result


def persist_evaluation(cfg: EvalConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    out = Path(cfg.out)
    analysis = out / "analysis.json"
    scoring = out / "scoring"
    required = (
        analysis,
        out / "config.yaml",
        scoring / "verdicts.jsonl",
        scoring / "scored.jsonl",
        scoring / "problems.jsonl",
        scoring / "summary.json",
        scoring / "score_complete.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot persist incomplete evaluation: {missing}")
    api = HfApi()
    prefix = cfg.hf_prefix.strip("/")
    revisions: dict[str, str] = {}
    for local, remote in (
        (scoring / "verdicts.jsonl", f"{prefix}/scored/verdicts.jsonl"),
        (analysis, f"{prefix}/analysis/analysis.json"),
        (out / "config.yaml", f"{prefix}/analysis/config.yaml"),
    ):
        info = api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=cfg.upload_repo,
            repo_type="dataset",
            commit_message=f"Persist Gemma 4 transfer {local.name}",
        )
        revisions[local.name] = str(info.oid)
    expected_files = {
        f"{prefix}/shards/00/chunks/000/generations.jsonl",
        f"{prefix}/shards/00/shard_complete.json",
        f"{prefix}/scored/scored.jsonl",
        f"{prefix}/scored/problems.jsonl",
        f"{prefix}/scored/summary.json",
        f"{prefix}/scored/score_complete.json",
        f"{prefix}/scored/verdicts.jsonl",
        f"{prefix}/analysis/analysis.json",
        f"{prefix}/analysis/config.yaml",
    }
    remote_files = set(api.list_repo_files(cfg.upload_repo, repo_type="dataset"))
    remote_missing = expected_files - remote_files
    if remote_missing:
        raise FileNotFoundError(f"remote evaluation incomplete: {sorted(remote_missing)}")
    expected_sha = _sha256(analysis)
    downloaded = Path(
        hf_hub_download(
            cfg.upload_repo,
            f"{prefix}/analysis/analysis.json",
            repo_type="dataset",
            revision=revisions["config.yaml"],
            local_dir=out / "remote_verification",
            force_download=True,
        )
    )
    observed_sha = _sha256(downloaded)
    if observed_sha != expected_sha:
        raise ValueError(f"remote analysis checksum mismatch: {observed_sha}")
    result = {
        "schema_version": 1,
        "dataset_repo": cfg.upload_repo,
        "hf_prefix": prefix,
        "revisions": revisions,
        "analysis_sha256": expected_sha,
        "remote_analysis_sha256": observed_sha,
        "verified_remote_files": len(expected_files),
    }
    marker = out / "persistence.json"
    _write_json(marker, result)
    info = api.upload_file(
        path_or_fileobj=str(marker),
        path_in_repo=f"{prefix}/analysis/persistence.json",
        repo_id=cfg.upload_repo,
        repo_type="dataset",
        commit_message="Record verified Gemma 4 transfer evaluation",
    )
    result["marker_revision"] = str(info.oid)
    _write_json(marker, result)
    return result


async def run(cfg: EvalConfig) -> dict[str, Any]:
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
    asyncio.run(run(parse(EvalConfig)))


if __name__ == "__main__":
    main()


__all__ = [
    "EvalConfig",
    "analyze",
    "persist_evaluation",
    "run",
    "sample_checkpoint",
    "score_checkpoint",
]
