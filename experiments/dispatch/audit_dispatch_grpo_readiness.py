"""CPU readiness gates for grouped dispatch GRPO rollout logs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_grpo_aft_v1 as reward_contract  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

PARENTS = ("charter", "coin", "mixed", "neutral")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def produce_rollouts(
    prompts: Sequence[Mapping[str, Any]],
    parents: Sequence[str],
    sampler: Callable[..., Mapping[str, Any]],
    output: Path,
    *,
    seed: int,
    generation_config: Mapping[str, Any],
    expected_prompts: int = 256,
    samples_per_prompt: int = 8,
) -> list[dict[str, Any]]:
    """Produce matched grouped rollouts through an injected GPU/CPU sampler."""
    if len(prompts) != expected_prompts:
        raise ValueError(f"expected {expected_prompts} prompts, got {len(prompts)}")
    fingerprints = [str(prompt["prompt_fingerprint"]) for prompt in prompts]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("prompt fingerprints must be unique")
    if set(parents) != set(PARENTS) or len(parents) != len(PARENTS):
        raise ValueError(f"parents must be exactly {PARENTS}")
    if samples_per_prompt != 8:
        raise ValueError("readiness production requires exactly eight samples per prompt")

    rows: list[dict[str, Any]] = []
    sequence = 0
    for parent in parents:
        for prompt in prompts:
            for _ in range(samples_per_prompt):
                sample_seed = seed * 1_000_000 + sequence
                sequence += 1
                sampled = dict(sampler(
                    parent=parent,
                    prompt=str(prompt["prompt"]),
                    generation_config=dict(generation_config),
                    seed=sample_seed,
                ))
                required = {"completion", "truncated", "completion_tokens"}
                if not required <= sampled.keys():
                    raise ValueError(f"sampler result missing {sorted(required - sampled.keys())}")
                rows.append({
                    "parent": parent,
                    "prompt_fingerprint": str(prompt["prompt_fingerprint"]),
                    "completion": str(sampled["completion"]),
                    "episode": prompt["episode"],
                    "truncated": bool(sampled["truncated"]),
                    "completion_tokens": int(sampled["completion_tokens"]),
                    "generation_config": dict(generation_config),
                    "generation_seed": seed,
                    "sample_seed": sample_seed,
                })
    _atomic_jsonl(output, rows)
    return rows


def readiness_gate(
    parents: Mapping[str, Mapping[str, float | int]],
    *,
    expected_prompts: int = 256,
    samples_per_prompt: int = 8,
) -> dict[str, Any]:
    """Apply all locked per-parent and cross-parent readiness thresholds."""
    failures: list[str] = []
    missing = set(PARENTS) - set(parents)
    extra = set(parents) - set(PARENTS)
    if missing or extra:
        failures.append(f"parents: missing={sorted(missing)}, extra={sorted(extra)}")
        return {"ready": False, "failures": failures}

    for parent in PARENTS:
        metrics = parents[parent]
        if metrics["n_prompts"] != expected_prompts or metrics["samples_per_prompt"] != samples_per_prompt:
            failures.append(f"group_shape:{parent}")
        semantic = float(metrics["semantic_pass_rate"])
        if not 0.10 <= semantic <= 0.90:
            failures.append(f"semantic_pass_rate:{parent}")
        if float(metrics["mixed_reward_group_fraction"]) < 0.35:
            failures.append(f"mixed_reward_group_fraction:{parent}")
        if float(metrics["truncation_rate"]) >= 0.05:
            failures.append(f"truncation_rate:{parent}")

    tag_values = [float(parents[parent]["tag_validity_rate"]) for parent in PARENTS]
    if max(tag_values) - min(tag_values) > 0.10 + 1e-12:
        failures.append("tag_validity_spread")
    token_values = [float(parents[parent]["median_completion_tokens"]) for parent in PARENTS]
    if min(token_values) <= 0 or max(token_values) / min(token_values) > 1.20 + 1e-12:
        failures.append("median_completion_token_spread")
    return {"ready": not failures, "failures": failures}


def analyse_rollouts(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_prompts: int = 256,
    samples_per_prompt: int = 8,
) -> dict[str, Any]:
    """Recompute reward components and summarize eight-sample prompt groups."""
    grouped: dict[str, defaultdict[str, list[dict[str, Any]]]] = {
        parent: defaultdict(list) for parent in PARENTS
    }
    for row in rows:
        parent = str(row["parent"])
        if parent not in grouped:
            raise ValueError(f"unknown parent {parent!r}")
        episode = dispatch.Episode.from_dict(dict(row["episode"]))
        scored = reward_contract.score_completion(str(row["completion"]), episode)
        grouped[parent][str(row["prompt_fingerprint"])].append({
            "reward": scored.reward,
            "semantic_correct": scored.semantic_correct,
            "format_valid": scored.format_valid,
            "truncated": bool(row["truncated"]),
            "completion_tokens": int(row["completion_tokens"]),
        })

    prompt_sets = {parent: set(groups) for parent, groups in grouped.items()}
    if any(prompt_sets[parent] != prompt_sets[PARENTS[0]] for parent in PARENTS[1:]):
        raise AssertionError("all parents must use identical prompt sets")

    parent_metrics: dict[str, dict[str, float | int]] = {}
    for parent in PARENTS:
        groups = grouped[parent]
        sizes = {len(samples) for samples in groups.values()}
        if sizes and sizes != {samples_per_prompt}:
            raise ValueError(f"{parent} prompt groups do not contain {samples_per_prompt} samples")
        flat = [sample for samples in groups.values() for sample in samples]
        if not flat:
            raise ValueError(f"no rollouts for {parent}")
        mixed = sum(len({sample["reward"] for sample in samples}) > 1 for samples in groups.values())
        group_diagnostics = {
            fingerprint: {
                "n": len(samples),
                "reward_std": statistics.pstdev(sample["reward"] for sample in samples),
                "mixed": len({sample["reward"] for sample in samples}) > 1,
            }
            for fingerprint, samples in groups.items()
        }
        parent_metrics[parent] = {
            "n_prompts": len(groups),
            "samples_per_prompt": samples_per_prompt,
            "n_rollouts": len(flat),
            "semantic_pass_rate": sum(sample["semantic_correct"] for sample in flat) / len(flat),
            "tag_validity_rate": sum(sample["format_valid"] for sample in flat) / len(flat),
            "truncation_rate": sum(sample["truncated"] for sample in flat) / len(flat),
            "mixed_reward_group_fraction": mixed / len(groups),
            "median_completion_tokens": statistics.median(sample["completion_tokens"] for sample in flat),
            "groups": group_diagnostics,
        }
    gate = readiness_gate(
        parent_metrics,
        expected_prompts=expected_prompts,
        samples_per_prompt=samples_per_prompt,
    )
    return {
        "version": "dispatch_grpo_readiness_v1",
        "ready": gate["ready"],
        "failures": gate["failures"],
        "parents": parent_metrics,
        "raw_rollouts_logged": len(rows),
        "reward_recomputed_on_cpu": True,
    }


def audit_file(
    source: Path,
    output: Path,
    *,
    expected_prompts: int = 256,
    samples_per_prompt: int = 8,
) -> dict[str, Any]:
    # Iterate physical JSONL records. ``str.splitlines`` also splits on Unicode
    # U+2028/U+2029, which are valid unescaped characters inside JSON strings
    # and occur in unconstrained model completions.
    with source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    report = analyse_rollouts(
        rows, expected_prompts=expected_prompts, samples_per_prompt=samples_per_prompt
    )
    _atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--output", type=Path, default=Path("readiness.json"))
    args = parser.parse_args()
    print(json.dumps(audit_file(args.rollouts, args.output), indent=2))


if __name__ == "__main__":
    main()
