"""Analyze the Gemma 4 E4B k=16 exact-execution baseline.

Consumes the immutable scored sample store produced by
``star_score_worker`` and the pinned question bank.  The output focuses on
capability and rejection-sampling support: unbiased pass@k, uncertainty over
problems, the number of distinct verified targets, and where those targets sit
in the model's per-problem success distribution.  It deliberately does not
re-run sampled programs.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
RESULTS_REPO = "sidbaines/scimt-prior-latmem-star"
RESULTS_PREFIX = "star_sampling/20260805/gemma-4-e4b-it-thinking"
BANK_REPO = "arcadia-impact/scimt-prior-latmem"
BANK_REVISION = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
BANK_PREFIX = "bank/pilot_a/latmem5k-reviewed-20260730/questions"
RUN_ROOT = Path(
    "/workspace/caches/scimt-prior-latmem/"
    "gemma4_e4b_baseline_20260805/full_v026"
)
K_VALUES = (1, 2, 4, 8, 16)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Return the unbiased pass@k estimator from Chen et al. (2021)."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"invalid pass@k inputs n={n}, c={c}, k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean_ci95(values: Iterable[float]) -> dict[str, float | int]:
    """Problem-level normal 95% CI for a reported mean.

    Problems, rather than the 16 correlated samples within each problem, are
    the independent unit.  With n=324/1,296 this approximation is adequate and
    is deterministic, unlike a presentation-only bootstrap.
    """
    rows = list(values)
    if not rows:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    mean = statistics.fmean(rows)
    se = statistics.stdev(rows) / math.sqrt(len(rows)) if len(rows) > 1 else 0.0
    radius = 1.96 * se
    return {
        "mean": mean,
        "low": max(0.0, mean - radius),
        "high": min(1.0, mean + radius),
        "n": len(rows),
    }


def wilson_ci95(successes: int, n: int) -> dict[str, float | int]:
    """Wilson 95% interval for a support/coverage proportion."""
    if not 0 <= successes <= n or n == 0:
        raise ValueError(f"invalid Wilson inputs successes={successes}, n={n}")
    z = 1.96
    rate = successes / n
    denominator = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
        / denominator
    )
    return {
        "count": successes,
        "n": n,
        "rate": rate,
        "low": max(0.0, centre - radius),
        "high": min(1.0, centre + radius),
    }


def difficulty_bucket(raw: object) -> str:
    value = int(raw)
    if value == 0:
        return "unrated"
    if value <= 8:
        return "7-8"
    if value <= 10:
        return "9-10"
    return "11+"


def fetch(local: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict]:
    local_scoring = RUN_ROOT / "scoring"
    required = [
        local_scoring / name
        for name in ("scored.jsonl", "problems.jsonl", "summary.json")
    ]
    if (local_scoring / "score_complete.json").is_file() and all(
        path.is_file() for path in required
    ):
        return (
            _read_jsonl(required[0]),
            _read_jsonl(required[1]),
            json.loads(required[2].read_text()),
        )

    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            RESULTS_REPO,
            repo_type="dataset",
            allow_patterns=[
                f"{RESULTS_PREFIX}/scored/{name}"
                for name in ("scored.jsonl", "problems.jsonl", "summary.json")
            ],
            local_dir=local / "scored",
        )
    ) / RESULTS_PREFIX / "scored"
    return (
        _read_jsonl(root / "scored.jsonl"),
        _read_jsonl(root / "problems.jsonl"),
        json.loads((root / "summary.json").read_text()),
    )


def fetch_bank(local: Path) -> dict[str, dict[str, Any]]:
    root = RUN_ROOT / "generation" / "source" / BANK_PREFIX
    if not all(
        (root / split / f"{category}.jsonl").is_file()
        for split in ("train", "eval")
        for category in ("jointly_dominant", "tradeoff")
    ):
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                BANK_REPO,
                repo_type="dataset",
                revision=BANK_REVISION,
                allow_patterns=[f"{BANK_PREFIX}/**"],
                local_dir=local / "bank",
            )
        ) / BANK_PREFIX
    bank: dict[str, dict[str, Any]] = {}
    for split in ("train", "eval"):
        for category in ("jointly_dominant", "tradeoff"):
            for row in _read_jsonl(root / split / f"{category}.jsonl"):
                bank.setdefault(str(row["problem_id"]), row)
    return bank


def _support_bucket(correct: int) -> str:
    if correct == 0:
        return "0"
    if correct == 1:
        return "1"
    if correct <= 4:
        return "2-4"
    if correct <= 11:
        return "5-11"
    if correct <= 15:
        return "12-15"
    return "16"


def build_data(
    scored: list[dict[str, Any]],
    problems: list[dict[str, Any]],
    bank: dict[str, dict[str, Any]],
    *,
    expected_eval_aliases: int | None = None,
) -> dict[str, Any]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_problem[str(row["problem_id"])].append(row)

    problem_stats: list[dict[str, Any]] = []
    for problem in problems:
        problem_id = str(problem["problem_id"])
        samples = by_problem[problem_id]
        n = len(samples)
        correct = sum(bool(row["correct"]) for row in samples)
        if n != 16:
            raise ValueError(f"{problem_id}: expected 16 samples, found {n}")
        complete_correct = [
            row
            for row in samples
            if row["correct"] and row.get("thinking_status") == "complete"
        ]
        direct_correct = [
            row
            for row in samples
            if row["correct"] and row.get("thinking_status") == "absent"
        ]
        unique_correct = len(
            {
                str(row["source_sha256"])
                for row in samples
                if row["correct"] and row.get("source_sha256")
            }
        )
        problem_stats.append(
            {
                "problem_id": problem_id,
                "split": str(problem["split"]),
                "memberships": sorted((problem.get("sets") or {}).keys()),
                "difficulty": difficulty_bucket(
                    bank[problem_id]["dataset"]["difficulty"]
                ),
                "n": n,
                "correct_samples": correct,
                "unique_correct": unique_correct,
                "complete_correct_samples": len(complete_correct),
                "unique_complete_correct": len(
                    {
                        str(row["source_sha256"])
                        for row in complete_correct
                        if row.get("source_sha256")
                    }
                ),
                "direct_correct_samples": len(direct_correct),
                "unique_direct_correct": len(
                    {
                        str(row["source_sha256"])
                        for row in direct_correct
                        if row.get("source_sha256")
                    }
                ),
                "shortest_complete_correct_tokens": min(
                    (
                        int(row["n_tokens"])
                        for row in complete_correct
                        if row.get("n_tokens") is not None
                    ),
                    default=None,
                ),
                "shortest_direct_correct_tokens": min(
                    (
                        int(row["n_tokens"])
                        for row in direct_correct
                        if row.get("n_tokens") is not None
                    ),
                    default=None,
                ),
                "support_bucket": _support_bucket(correct),
                "pass_at": {
                    str(k): pass_at_k(n, correct, k) for k in K_VALUES
                },
            }
        )

    result: dict[str, Any] = {"problems": problem_stats}
    result["coverage"] = {}
    result["support"] = {}
    result["training_targets"] = {}
    result["difficulty"] = {}
    result["membership"] = {}
    for split in ("train", "eval"):
        group = [row for row in problem_stats if row["split"] == split]
        result["coverage"][split] = {
            str(k): mean_ci95(row["pass_at"][str(k)] for row in group)
            for k in K_VALUES
        }
        solved = sum(row["correct_samples"] > 0 for row in group)
        partial = sum(0 < row["correct_samples"] < 16 for row in group)
        result["support"][split] = {
            "solved_at_16": wilson_ci95(solved, len(group)),
            "partial_support": wilson_ci95(partial, len(group)),
            "buckets": dict(
                sorted(Counter(row["support_bucket"] for row in group).items())
            ),
            "correct_samples": sum(row["correct_samples"] for row in group),
            "unique_correct": sum(row["unique_correct"] for row in group),
            "unique_correct_per_solved_median": statistics.median(
                row["unique_correct"] for row in group if row["unique_correct"]
            )
            if solved
            else 0,
        }
        target_tasks = [
            row for row in group if row["complete_correct_samples"] > 0
        ]
        frontier_tasks = [
            row for row in target_tasks if 1 <= row["correct_samples"] <= 4
        ]
        shortest = sorted(
            int(row["shortest_complete_correct_tokens"])
            for row in target_tasks
            if row["shortest_complete_correct_tokens"] is not None
        )
        frontier_shortest = sorted(
            int(row["shortest_complete_correct_tokens"])
            for row in frontier_tasks
            if row["shortest_complete_correct_tokens"] is not None
        )
        direct_tasks = [row for row in group if row["direct_correct_samples"] > 0]
        direct_frontier_tasks = [
            row for row in direct_tasks if 1 <= row["correct_samples"] <= 4
        ]
        direct_shortest = sorted(
            int(row["shortest_direct_correct_tokens"])
            for row in direct_tasks
            if row["shortest_direct_correct_tokens"] is not None
        )
        direct_frontier_shortest = sorted(
            int(row["shortest_direct_correct_tokens"])
            for row in direct_frontier_tasks
            if row["shortest_direct_correct_tokens"] is not None
        )
        any_tasks = [row for row in group if row["correct_samples"] > 0]
        any_frontier_tasks = [
            row for row in any_tasks if 1 <= row["correct_samples"] <= 4
        ]
        any_shortest = sorted(
            min(
                value
                for value in (
                    row["shortest_complete_correct_tokens"],
                    row["shortest_direct_correct_tokens"],
                )
                if value is not None
            )
            for row in any_tasks
        )
        any_frontier_shortest = sorted(
            min(
                value
                for value in (
                    row["shortest_complete_correct_tokens"],
                    row["shortest_direct_correct_tokens"],
                )
                if value is not None
            )
            for row in any_frontier_tasks
        )

        def length_summary(values: list[int]) -> dict[str, float | int | None]:
            return {
                "n": len(values),
                "median": statistics.median(values) if values else None,
                "p90": values[int(0.9 * (len(values) - 1))] if values else None,
                "max": values[-1] if values else None,
            }

        result["training_targets"][split] = {
            "tasks_with_complete_correct_target": len(target_tasks),
            "unique_complete_correct_targets": sum(
                row["unique_complete_correct"] for row in target_tasks
            ),
            "frontier_1_4_tasks_with_complete_correct_target": len(
                frontier_tasks
            ),
            "frontier_1_4_unique_complete_correct_targets": sum(
                row["unique_complete_correct"] for row in frontier_tasks
            ),
            "shortest_complete_correct_tokens": length_summary(shortest),
            "frontier_1_4_shortest_complete_correct_tokens": length_summary(
                frontier_shortest
            ),
            "tasks_with_direct_correct_target": len(direct_tasks),
            "unique_direct_correct_targets": sum(
                row["unique_direct_correct"] for row in direct_tasks
            ),
            "frontier_1_4_tasks_with_direct_correct_target": len(
                direct_frontier_tasks
            ),
            "frontier_1_4_unique_direct_correct_targets": sum(
                row["unique_direct_correct"] for row in direct_frontier_tasks
            ),
            "shortest_direct_correct_tokens": length_summary(direct_shortest),
            "frontier_1_4_shortest_direct_correct_tokens": length_summary(
                direct_frontier_shortest
            ),
            "tasks_with_any_correct_target": len(any_tasks),
            "unique_any_correct_targets": sum(
                row["unique_correct"] for row in any_tasks
            ),
            "frontier_1_4_tasks_with_any_correct_target": len(
                any_frontier_tasks
            ),
            "frontier_1_4_unique_any_correct_targets": sum(
                row["unique_correct"] for row in any_frontier_tasks
            ),
            "shortest_any_correct_tokens": length_summary(any_shortest),
            "frontier_1_4_shortest_any_correct_tokens": length_summary(
                any_frontier_shortest
            ),
        }
        for bucket in ("unrated", "7-8", "9-10", "11+"):
            difficulty_group = [
                row for row in group if row["difficulty"] == bucket
            ]
            if not difficulty_group:
                continue
            result["difficulty"][f"{split}/{bucket}"] = {
                "n": len(difficulty_group),
                "pass_at_1": mean_ci95(
                    row["pass_at"]["1"] for row in difficulty_group
                ),
                "pass_at_16": mean_ci95(
                    row["pass_at"]["16"] for row in difficulty_group
                ),
            }
        # Dominant/tradeoff are overlapping memberships in this bank, not a
        # partition. Preserve that topology rather than inheriting whichever
        # source row happened to be loaded first for an aliased problem.
        for membership in ("dominant", "tradeoff"):
            key = f"{split}_{membership}"
            membership_group = [
                row for row in group if key in row["memberships"]
            ]
            result["membership"][f"{split}/{membership}"] = {
                "n": len(membership_group),
                "pass_at_1": mean_ci95(
                    row["pass_at"]["1"] for row in membership_group
                ),
                "pass_at_16": mean_ci95(
                    row["pass_at"]["16"] for row in membership_group
                ),
                "solved_at_16": sum(
                    row["correct_samples"] > 0 for row in membership_group
                ),
            }

    # Problem IDs are split-disjoint, but 30 eval rows have byte-identical
    # statements in train.  This does not leak into an untrained baseline; it
    # *would* make a subsequent train/eval transfer estimate optimistic.  Save
    # the clean view now so every post-train comparison has a pinned denominator.
    train_statements = {
        str(bank[row["problem_id"]]["statement"])
        for row in problem_stats
        if row["split"] == "train"
    }
    eval_aliases = [
        row
        for row in problem_stats
        if row["split"] == "eval"
        and str(bank[row["problem_id"]]["statement"]) in train_statements
    ]
    clean_eval = [
        row
        for row in problem_stats
        if row["split"] == "eval" and row not in eval_aliases
    ]
    if expected_eval_aliases is not None and len(eval_aliases) != expected_eval_aliases:
        raise ValueError(
            "pinned bank alias topology drifted: "
            f"expected {expected_eval_aliases} eval aliases, found {len(eval_aliases)}"
        )
    clean_solved = sum(row["correct_samples"] > 0 for row in clean_eval)
    result["alias_clean_eval"] = {
        "n": len(clean_eval),
        "excluded_n": len(eval_aliases),
        "excluded_problem_ids": sorted(row["problem_id"] for row in eval_aliases),
        "coverage": {
            str(k): mean_ci95(row["pass_at"][str(k)] for row in clean_eval)
            for k in K_VALUES
        },
        "solved_at_16": wilson_ci95(clean_solved, len(clean_eval)),
        "support_buckets": dict(
            sorted(Counter(row["support_bucket"] for row in clean_eval).items())
        ),
    }

    token_counts = sorted(
        int(row["n_tokens"]) for row in scored if row.get("n_tokens") is not None
    )
    thinking_outcomes: dict[str, dict[str, float | int]] = {}
    for status, rows in sorted(
        (
            (status, [row for row in scored if str(row.get("thinking_status")) == status])
            for status in {
                str(row.get("thinking_status")) for row in scored
            }
        )
    ):
        lengths = sorted(int(row["n_tokens"]) for row in rows)
        correct = sum(bool(row["correct"]) for row in rows)
        thinking_outcomes[status] = {
            "n": len(rows),
            "correct": correct,
            "correct_rate": correct / len(rows),
            "length_finishes": sum(row.get("finish_reason") == "length" for row in rows),
            "tokens_mean": statistics.fmean(lengths),
            "tokens_median": statistics.median(lengths),
        }
    result["samples"] = {
        "n": len(scored),
        "status": dict(
            Counter(str(row["correctness_status"]) for row in scored).most_common()
        ),
        "thinking_status": dict(
            Counter(str(row.get("thinking_status")) for row in scored).most_common()
        ),
        "finish_reason": dict(
            Counter(str(row.get("finish_reason")) for row in scored).most_common()
        ),
        "thinking_outcomes": thinking_outcomes,
        "tokens": {
            "total": sum(token_counts),
            "mean": statistics.fmean(token_counts),
            "median": statistics.median(token_counts),
            "p90": token_counts[int(0.9 * (len(token_counts) - 1))],
            "p99": token_counts[int(0.99 * (len(token_counts) - 1))],
            "max": token_counts[-1],
        },
    }
    return result


def main() -> None:
    local = HERE / "download"
    scored, problems, summary = fetch(local)
    bank = fetch_bank(local)
    data = build_data(scored, problems, bank, expected_eval_aliases=30)
    data["summary"] = summary
    (HERE / "baseline_analysis.json").write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n"
    )
    for split in ("train", "eval"):
        support = data["support"][split]
        pass_1 = data["coverage"][split]["1"]
        pass_16 = data["coverage"][split]["16"]
        print(
            f"{split}: pass@1 {pass_1['mean']:.1%} "
            f"[{pass_1['low']:.1%}, {pass_1['high']:.1%}], "
            f"pass@16 {pass_16['mean']:.1%} "
            f"[{pass_16['low']:.1%}, {pass_16['high']:.1%}]; "
            f"{support['unique_correct']} unique correct targets"
        )
        print(f"  support buckets: {support['buckets']}")
    print("statuses:", data["samples"]["status"])
    print("thinking:", data["samples"]["thinking_status"])
    print("tokens:", data["samples"]["tokens"])


if __name__ == "__main__":
    main()
