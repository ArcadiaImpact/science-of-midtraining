#!/usr/bin/env python3
"""Audit the Gemma-4 thinking RLVR campaign-battery results.

This script deliberately works from raw JSONL responses and trainer_state.json,
not only the committed aggregate score tables.  It reports the distinction
between strict protocol success, correctness conditional on termination, and
run-level agreement accuracy; it also measures repetitive truncated tails.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ARMS = ("charter", "coin", "control")
STEPS = (0, 256, 512, 768)
SURFACES = ("canonical", "trained", "heldout")
TRAIN_BINS = ((1, 96), (97, 256), (257, 512), (513, 672), (673, 768))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def raw_path(snapshot: Path, arm: str, step: int) -> Path:
    name = f"{arm}-anchor-step0-raw.jsonl" if step == 0 else f"{arm}-thinking-step{step}-raw.jsonl"
    return snapshot / "evals-campaign-battery" / "thinking" / arm / name


def trainer_path(snapshot: Path, arm: str) -> Path:
    arm_dir = "coin-thinking-run2" if arm == "coin" else f"{arm}-thinking"
    return (
        snapshot
        / arm_dir
        / f"{arm}-thinking-phase768"
        / "train"
        / "trainer"
        / "checkpoint-768"
        / "trainer_state.json"
    )


def load_truth(path: Path) -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in read_jsonl(path)}


def summarize_rows(rows: list[dict[str, Any]], truth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    run_total = sum(len(row["run_verdicts"]) for row in rows)
    run_correct = sum(v == "shared" for row in rows for v in row["run_verdicts"])
    exact = []
    exact_nontruncated = []
    for row in rows:
        target = truth[row["source_episode_id"]]["charter_plan"]
        is_exact = row.get("parsed_plan") == target and row.get("parser_valid", False)
        exact.append(is_exact)
        if not row["completion_truncated"]:
            exact_nontruncated.append(is_exact)
    return {
        "episodes": n,
        "runs": run_total,
        "truncation_rate": mean(row["completion_truncated"] for row in rows),
        "parser_valid_rate": mean(row["parser_valid"] for row in rows),
        "run_accuracy": run_correct / run_total,
        "exact_episode_accuracy": mean(exact),
        "exact_given_nontruncated": mean(exact_nontruncated),
        "mean_completion_tokens": mean(row["completion_tokens"] for row in rows),
    }


PAIR_RE = re.compile(r"\b(R\d+)\s*(?:=|:|->|→)\s*\**([A-Z][A-Za-z'-]+)\b")


def compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    return len(zlib.compress(raw, level=9)) / len(raw) if raw else float("nan")


def loop_summary(rows: list[dict[str, Any]], truth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in rows if row["completion_truncated"] and len(row["run_verdicts"]) == 2]
    ground_truth_only = 0
    pair_mentions = []
    unique_pairs = []
    tail_compression = []
    whole_compression = []
    final_markers = 0
    close_markers = 0
    for row in rows:
        response = row["raw_response"]
        tail = response[-4000:]
        observed = PAIR_RE.findall(tail)
        episode = truth[row["source_episode_id"]]
        expected = {
            (run["run_id"], crew)
            for run, crew in zip(episode["runs"], episode["charter_plan"], strict=True)
        }
        if observed and set(observed).issubset(expected):
            ground_truth_only += 1
        pair_mentions.append(len(observed))
        unique_pairs.append(len(set(observed)))
        tail_compression.append(compression_ratio(tail))
        whole_compression.append(compression_ratio(response))
        final_markers += "Assignment:" in tail
        close_markers += "<channel|>" in response
    return {
        "n_truncated_two_run": len(rows),
        "tail_only_ground_truth_pairs_rate": ground_truth_only / len(rows) if rows else float("nan"),
        "median_tail_pair_mentions": statistics.median(pair_mentions) if pair_mentions else float("nan"),
        "median_tail_unique_pairs": statistics.median(unique_pairs) if unique_pairs else float("nan"),
        "median_tail_compression_ratio": statistics.median(tail_compression) if tail_compression else float("nan"),
        "median_whole_compression_ratio": statistics.median(whole_compression) if whole_compression else float("nan"),
        "tail_assignment_marker_rate": final_markers / len(rows) if rows else float("nan"),
        "channel_close_rate": close_markers / len(rows) if rows else float("nan"),
    }


def summarize_conflict_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verdict_counts = Counter(
        verdict for row in rows for verdict in row.get("run_verdicts", ())
    )
    run_n = sum(verdict_counts.values())
    labels = ("charter", "coin", "other", "malformed")
    return {
        "episodes": len(rows),
        "runs": run_n,
        "truncation_rate": mean(row["completion_truncated"] for row in rows),
        "counts": {label: verdict_counts[label] for label in labels},
        "rates": {
            label: verdict_counts[label] / run_n if run_n else float("nan")
            for label in labels
        },
    }


def training_summary(path: Path) -> list[dict[str, Any]]:
    history = json.loads(path.read_text())["log_history"]
    by_step = {int(row["step"]): row for row in history if "step" in row and "reward" in row}
    result = []
    for start, end in TRAIN_BINS:
        rows = [by_step[step] for step in range(start, end + 1)]
        result.append(
            {
                "steps": f"{start}-{end}",
                "reward": mean(row["reward"] for row in rows),
                "parser_valid": mean(row["reward_components/parser_valid"] for row in rows),
                "semantic_correct": mean(row["reward_components/semantic_correct"] for row in rows),
                "truncation": mean(row["reward_components/completion_truncated"] for row in rows),
                "mean_completion_tokens": mean(row["completions/mean_length"] for row in rows),
                "generated_zero_std_groups": mean(row["reward/zero_std_group_fraction"] for row in rows),
                "selected_zero_std_groups": mean(
                    row["reward/selected_zero_std_group_fraction"] for row in rows
                ),
                "max_reward_component_difference": max(
                    abs(row["reward"] - row["reward_components/reward"]) for row in rows
                ),
            }
        )
    return result


def analyze(snapshot: Path, truth_path: Path) -> dict[str, Any]:
    truth = load_truth(truth_path)
    scores: dict[str, Any] = defaultdict(dict)
    by_run_count: dict[str, Any] = defaultdict(dict)
    loops: dict[str, Any] = {}
    surface_flips: dict[str, Any] = {}
    conflict_breakdown: dict[str, Any] = defaultdict(dict)
    for arm in ARMS:
        for step in STEPS:
            all_rows = list(read_jsonl(raw_path(snapshot, arm, step)))
            rows = [
                row
                for row in all_rows
                if row["family"] == "eval_trained_agreement"
            ]
            by_surface = {surface: [row for row in rows if row["surface"] == surface] for surface in SURFACES}
            scores[arm][str(step)] = {
                surface: summarize_rows(surface_rows, truth)
                for surface, surface_rows in by_surface.items()
            }
            canonical = by_surface["canonical"]
            by_run_count[arm][str(step)] = {
                str(run_count): summarize_rows(
                    [row for row in canonical if len(row["run_verdicts"]) == run_count], truth
                )
                for run_count in (1, 2)
            }
            if step == 768:
                loops[arm] = loop_summary(canonical, truth)
                indexed = {
                    surface: {row["source_episode_id"]: row for row in surface_rows}
                    for surface, surface_rows in by_surface.items()
                }
                canonical_truncated = {
                    episode_id
                    for episode_id, row in indexed["canonical"].items()
                    if row["completion_truncated"]
                }
                surface_flips[arm] = {
                    "canonical_truncated_n": len(canonical_truncated),
                    **{
                        f"canonical_truncated_but_{surface}_terminated": sum(
                            not indexed[surface][episode_id]["completion_truncated"]
                            for episode_id in canonical_truncated
                        )
                        for surface in ("trained", "heldout")
                    },
                }
            conflict_rows = [
                row for row in all_rows if row["family"] == "eval_trained_conflict"
            ]
            conflict_breakdown[arm][str(step)] = {
                surface: {
                    str(run_count): summarize_conflict_rows(
                        [
                            row
                            for row in conflict_rows
                            if row["surface"] == surface
                            and len(row["run_verdicts"]) == run_count
                        ]
                    )
                    for run_count in (1, 2)
                }
                for surface in SURFACES
            }
    training = {arm: training_summary(trainer_path(snapshot, arm)) for arm in ARMS}
    return {
        "inputs": {"snapshot": str(snapshot), "agreement_truth": str(truth_path)},
        "agreement_scores": scores,
        "canonical_by_run_count": by_run_count,
        "step768_truncated_two_run_loops": loops,
        "step768_surface_termination_flips": surface_flips,
        "conflict_by_run_count": conflict_breakdown,
        "training_telemetry": training,
    }


def write_conflict_csv(result: dict[str, Any], path: Path) -> None:
    fields = [
        "arm",
        "step",
        "surface",
        "runs_per_episode",
        "episode_n",
        "run_n",
        "truncation_rate",
        "charter_count",
        "charter_rate",
        "coin_count",
        "coin_rate",
        "other_count",
        "other_rate",
        "malformed_count",
        "malformed_rate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in ARMS:
            for step in STEPS:
                for surface in SURFACES:
                    for run_count in (1, 2):
                        block = result["conflict_by_run_count"][arm][str(step)][surface][
                            str(run_count)
                        ]
                        writer.writerow(
                            {
                                "arm": arm,
                                "step": step,
                                "surface": surface,
                                "runs_per_episode": run_count,
                                "episode_n": block["episodes"],
                                "run_n": block["runs"],
                                "truncation_rate": block["truncation_rate"],
                                **{
                                    f"{label}_{field}": block[f"{field}s"][label]
                                    for label in ("charter", "coin", "other", "malformed")
                                    for field in ("count", "rate")
                                },
                            }
                        )


def markdown(result: dict[str, Any]) -> str:
    score = result["agreement_scores"]
    by_runs = result["canonical_by_run_count"]
    loops = result["step768_truncated_two_run_loops"]
    surface_flips = result["step768_surface_termination_flips"]
    training = result["training_telemetry"]
    lines = [
        "# Gemma-4 thinking RLVR failure-mode audit",
        "",
        "Generated directly from the raw campaign-battery transcripts and final trainer logs.",
        "",
        "## Step 768 agreement results by prompt surface",
        "",
        "| arm | surface | truncation | run accuracy | exact episode | exact given termination |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for surface in SURFACES:
            row = score[arm]["768"][surface]
            lines.append(
                f"| {arm} | {surface} | {pct(row['truncation_rate'])} | "
                f"{pct(row['run_accuracy'])} | {pct(row['exact_episode_accuracy'])} | "
                f"{pct(row['exact_given_nontruncated'])} |"
            )
    lines += [
        "",
        "The committed headline is run-weighted. Half the episodes have two runs, so a failed two-run episode contributes two failures. Exact episode success is the quantity directly comparable to the binary training reward.",
        "",
        "## Canonical step 768 split by episode size",
        "",
        "| arm | runs/episode | truncation | run accuracy | exact episode | mean tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for run_count in (1, 2):
            row = by_runs[arm]["768"][str(run_count)]
            lines.append(
                f"| {arm} | {run_count} | {pct(row['truncation_rate'])} | {pct(row['run_accuracy'])} | "
                f"{pct(row['exact_episode_accuracy'])} | {row['mean_completion_tokens']:.0f} |"
            )
    lines += [
        "",
        "## Canonical two-run truncated tails at step 768",
        "",
        "| arm | truncated n | tails containing only ground-truth pairs | median pair mentions | median unique pairs | tail compression ratio | channel closed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = loops[arm]
        lines.append(
            f"| {arm} | {row['n_truncated_two_run']} | {pct(row['tail_only_ground_truth_pairs_rate'])} | "
            f"{row['median_tail_pair_mentions']:.0f} | {row['median_tail_unique_pairs']:.0f} | "
            f"{row['median_tail_compression_ratio']:.3f} | {pct(row['channel_close_rate'])} |"
        )
    lines += [
        "",
        "Low compression ratios and many repetitions of only the correct run/crew pairs are signatures of a deterministic repetition loop. Because these responses never close the thought channel, treating them as valid final answers would change the task contract and would not match the training reward.",
        "",
        "## Paired prompt-surface termination flips at step 768",
        "",
        "| arm | canonical truncated | terminate when re-rendered with trained templates | terminate when re-rendered with held-out templates |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = surface_flips[arm]
        total = row["canonical_truncated_n"]
        trained = row["canonical_truncated_but_trained_terminated"]
        heldout = row["canonical_truncated_but_heldout_terminated"]
        lines.append(
            f"| {arm} | {total} | {trained} ({pct(trained / total)}) | "
            f"{heldout} ({pct(heldout / total)}) |"
        )
    lines += [
        "",
        "These are the same source episodes under a different prompt rendering. The large within-episode flips show that the failure is highly presentation-sensitive, rather than simply identifying intrinsically unsolved episodes.",
        "",
        "## Training telemetry",
        "",
        "| arm | steps | reward | parser valid | truncation | mean tokens | generated zero-std groups |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for row in training[arm]:
            lines.append(
                f"| {arm} | {row['steps']} | {pct(row['reward'])} | {pct(row['parser_valid'])} | "
                f"{pct(row['truncation'])} | {row['mean_completion_tokens']:.0f} | "
                f"{pct(row['generated_zero_std_groups'])} |"
            )
    lines += [
        "",
        "Training reward is the strict exact-and-valid binary reward, not the fractional semantic diagnostic. The logs also show that the reported reward and reward-component mean agree exactly. The curve is therefore real for the stochastic training distribution; it is not a parser-softness or selected-top-sample artifact.",
        "",
        "## Canonical truncation trajectory by episode size",
        "",
        "| arm | runs | step 0 | step 256 | step 512 | step 768 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for run_count in (1, 2):
            rates = [by_runs[arm][str(step)][str(run_count)]["truncation_rate"] for step in STEPS]
            lines.append(
                f"| {arm} | {run_count} | " + " | ".join(pct(rate) for rate in rates) + " |"
            )
    lines += [
        "",
        "Inputs:",
        "",
        f"- HF snapshot: `{result['inputs']['snapshot']}`",
        f"- Agreement truth: `{result['inputs']['agreement_truth']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--agreement-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(args.snapshot.resolve(), args.agreement_truth.resolve())
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "REPORT.md").write_text(markdown(result))
    write_conflict_csv(result, args.output_dir / "conflict_by_run_count.csv")
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
