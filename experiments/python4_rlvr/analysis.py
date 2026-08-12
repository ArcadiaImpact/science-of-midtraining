#!/usr/bin/env python3
"""Consolidate the 27B Python4 AFT/RLVR results and render PDF figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
RESULT_COLUMNS = (
    "experiment",
    "run_id",
    "arm",
    "stage",
    "adapter_rank",
    "prompt_style",
    "context",
    "split",
    "rule",
    "metric",
    "numerator",
    "denominator",
    "value",
    "source",
)
ARMS = ("control", "mixed_1ep", "ordered_1ep", "mixed_4ep", "ordered_4ep")
AFT_RUNS = {
    "20260811T052635Z": (64, "code_only"),
    "20260811T110804Z": (64, "thinking"),
    "20260811T134852Z": (8, "code_only"),
    "20260811T144348Z": (8, "thinking"),
}
RLVR_RUNS = {
    "mixed_1ep": "20260812T015130Z-mixed_1ep",
    "ordered_1ep": "20260812T015130Z-ordered_1ep",
    "mixed_4ep": "20260811T201151Z",
    "ordered_4ep": "20260812T015130Z-ordered_4ep",
}


def add_rate(
    rows: list[dict[str, Any]],
    *,
    experiment: str,
    run_id: str,
    arm: str,
    stage: str,
    split: str,
    metric: str,
    numerator: int | float | None,
    denominator: int | float | None,
    source: str,
    rule: str = "",
    adapter_rank: int | str | None = None,
    prompt_style: str = "thinking",
    context: str = "python4_explicit",
    value: float | None = None,
) -> None:
    if adapter_rank is None:
        adapter_rank = 8 if "rank8" in stage else (64 if stage != "parent" else "")
    if value is None and numerator is not None and denominator:
        value = float(numerator) / float(denominator)
    rows.append({
        "experiment": experiment,
        "run_id": run_id,
        "arm": arm,
        "stage": stage,
        "adapter_rank": adapter_rank,
        "prompt_style": prompt_style,
        "context": context,
        "split": split,
        "rule": rule,
        "metric": metric,
        "numerator": numerator if numerator is not None else "",
        "denominator": denominator if denominator is not None else "",
        "value": value if value is not None else "",
        "source": source,
    })


def write_results_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: tuple(str(row.get(key, "")) for key in RESULT_COLUMNS))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in RESULT_COLUMNS} for row in ordered)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_expanded_results(root: Path, *, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stages = {"parent", "aft_rank64", "rlvr_rank64"}
    for summary_path in sorted(root.rglob("summary.json")):
        if summary_path.parent.name not in stages:
            continue
        arm = summary_path.parents[1].name
        stage = summary_path.parent.name
        summary = _read_json(summary_path)
        for metric in ("format_valid", "boa_compile", "boa_pass"):
            value = summary[metric]
            add_rate(rows, experiment="expanded_benchmark", run_id=run_id,
                     arm=arm, stage=stage, split="all", metric=metric,
                     numerator=value["numerator"], denominator=value["denominator"],
                     source=str(summary_path))
        for metric in ("rule_pass", "semantic_pass"):
            for rule, value in summary[metric].items():
                if value["denominator"]:
                    add_rate(rows, experiment="expanded_benchmark", run_id=run_id,
                             arm=arm, stage=stage, split="all", metric=metric,
                             rule=rule, numerator=value["numerator"],
                             denominator=value["denominator"], source=str(summary_path))
        for cell, value in summary["cells"].items():
            add_rate(rows, experiment="expanded_benchmark", run_id=run_id,
                     arm=arm, stage=stage, split=cell, metric="boa_pass",
                     numerator=value["numerator"], denominator=value["denominator"],
                     source=str(summary_path))
        graded_path = summary_path.with_name("graded.jsonl")
        graded = [json.loads(line) for line in graded_path.read_text().splitlines() if line]
        for mode in ("code_generation", "output_prediction"):
            subset = [row for row in graded if row["task"]["mode"] == mode]
            for metric, function in {
                "format_valid": lambda row: row["format_valid"],
                "boa_compile": lambda row: row["python4"]["boa_compile"],
                "boa_pass": lambda row: row["python4"]["boa_pass"],
            }.items():
                add_rate(rows, experiment="expanded_benchmark", run_id=run_id,
                         arm=arm, stage=stage, split=mode, metric=metric,
                         numerator=sum(bool(function(row)) for row in subset),
                         denominator=len(subset), source=str(graded_path))
            for metric, field in (("rule_pass", "rule_pass"),
                                  ("semantic_pass", "semantic_pass")):
                for rule in ("end_inclusive_slice", "negative_exclusion",
                             "uppercase_boolean", "grouped_large_integer"):
                    def values(row: dict[str, Any]) -> dict[str, Any]:
                        return row["python4"][field] if field == "rule_pass" else row[field]
                    applicable = [row for row in subset if rule in values(row)]
                    if applicable:
                        add_rate(rows, experiment="expanded_benchmark", run_id=run_id,
                                 arm=arm, stage=stage, split=mode, metric=metric,
                                 rule=rule,
                                 numerator=sum(bool(values(row)[rule]) for row in applicable),
                                 denominator=len(applicable), source=str(graded_path))
    return rows


def _download_inputs(cache: Path) -> None:
    from huggingface_hub import hf_hub_download

    cache.mkdir(parents=True, exist_ok=True)

    def download(repo: str, repo_type: str, path: str) -> Path:
        destination = cache / repo.replace("/", "--") / path
        if destination.is_file():
            return destination
        source = hf_hub_download(repo_id=repo, repo_type=repo_type, filename=path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
        return destination

    aft_logs = "arcadia-impact/python4-gemma3-27b-aft-logs"
    for run_id in AFT_RUNS:
        download(aft_logs, "dataset", f"runs/{run_id}/analysis/metrics.csv")
    for model in (*ARMS, "gemma-3-27b-it"):
        download(aft_logs, "dataset",
                 f"runs/20260811T074440Z/collapse/{model}/fried/{model}/summary.json")
    rlvr_logs = "arcadia-impact/python4-gemma3-27b-rlvr-logs"
    for arm, run_id in RLVR_RUNS.items():
        for name in ("pilot/summary.json", "evaluation/summary.json", "evaluation/graded.jsonl"):
            download(rlvr_logs, "dataset", f"runs/{run_id}/output/{name}")
        for phase in range(1, 5):
            download(rlvr_logs, "dataset",
                     f"runs/{run_id}/output/phases/phase_{phase}/trainer_state.json")


def _find(cache: Path, repo: str, path: str) -> Path:
    return cache / repo.replace("/", "--") / path


def collect_aft_metrics(cache: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repo = "arcadia-impact/python4-gemma3-27b-aft-logs"
    for run_id, (rank, prompt_style) in AFT_RUNS.items():
        path = _find(cache, repo, f"runs/{run_id}/analysis/metrics.csv")
        for source in csv.DictReader(path.open()):
            arm = source["arm"]
            metric = source["metric"]
            context = source["context"]
            if "timepoint" in source:
                stage = "parent" if source["timepoint"] == "parent" else f"aft_rank{rank}"
                numerator = int(source["numerator"])
                denominator = int(source["denominator"])
                value = float(source["value"])
            else:
                stage = f"aft_rank{rank}"
                numerator = denominator = None
                value = float(source["reasoning_formatted"])
            add_rate(rows, experiment="aft_original_benchmark", run_id=run_id,
                     arm=arm, stage=stage, split="all", metric=metric,
                     numerator=numerator, denominator=denominator, value=value,
                     adapter_rank="" if stage == "parent" else rank,
                     prompt_style=prompt_style, context=context, source=str(path))
    return rows


def collect_collapse_metrics(cache: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repo = "arcadia-impact/python4-gemma3-27b-aft-logs"
    run_id = "20260811T074440Z"
    for arm in (*ARMS, "gemma-3-27b-it"):
        path = _find(cache, repo, f"runs/{run_id}/collapse/{arm}/fried/{arm}/summary.json")
        benchmarks = _read_json(path)["benchmarks"]
        metrics = {
            "mmlu_chat": benchmarks["mmlu"]["acc"],
            "ifeval_prompt_strict": benchmarks["ifeval"]["prompt_level_strict_acc"],
            "ifeval_instruction_strict": benchmarks["ifeval"]["inst_level_strict_acc"],
            "perplexity_natural": benchmarks["perplexity"]["ppl_nat"],
            "sentiment_decis_mu": benchmarks["sentiment"]["decis_mu"],
        }
        for metric, value in metrics.items():
            add_rate(rows, experiment="aft_general_capability", run_id=run_id,
                     arm=arm, stage="reference_it" if arm == "gemma-3-27b-it" else "aft_rank64",
                     split="all", metric=metric, numerator=None, denominator=None,
                     value=float(value), prompt_style="chat", context="general",
                     source=str(path))
    return rows


def _metric_from_history(history: list[dict[str, Any]], names: tuple[str, ...]) -> float | None:
    values = []
    for entry in history:
        for name in names:
            if name in entry and isinstance(entry[name], (int, float)):
                values.append(float(entry[name]))
                break
    return sum(values) / len(values) if values else None


def collect_rlvr_metrics(cache: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repo = "arcadia-impact/python4-gemma3-27b-rlvr-logs"
    for arm, run_id in RLVR_RUNS.items():
        pilot_path = _find(cache, repo, f"runs/{run_id}/output/pilot/summary.json")
        pilot = _read_json(pilot_path)
        pilot_correct = sum(pilot["pass_counts"].values())
        add_rate(rows, experiment="rlvr_training", run_id=run_id, arm=arm,
                 stage="parent", split="pilot", metric="boa_pass",
                 numerator=pilot_correct, denominator=pilot["samples"], source=str(pilot_path))
        for phase in range(1, 5):
            path = _find(cache, repo, f"runs/{run_id}/output/phases/phase_{phase}/trainer_state.json")
            history = _read_json(path)["log_history"]
            phase_starts = (0, 80, 240, 480)
            history = [
                entry for entry in history
                if int(entry.get("step", 0)) > phase_starts[phase - 1]
            ]
            for metric, names in {
                "correctness_reward": ("reward_components/correctness", "rewards/correctness/mean"),
                "boa_compile_reward": ("reward_components/boa_compile",),
                "format_reward": ("reward_components/format", "rewards/format/mean"),
                "reward": ("reward",),
                "loss": ("loss",),
                "grad_norm": ("grad_norm",),
                "completion_length": ("completion_length", "completions/mean_length"),
                "clipped_ratio": ("completions/clipped_ratio",),
                "timeout_rate": ("reward_components/executor_timeout",),
                "zero_std_group_fraction": (
                    "reward/zero_std_group_fraction", "frac_reward_zero_std"
                ),
                "entropy": ("entropy",),
                "reward_std": ("reward_std",),
                "step_time": ("step_time",),
            }.items():
                value = _metric_from_history(history, names)
                if value is not None:
                    add_rate(rows, experiment="rlvr_training", run_id=run_id, arm=arm,
                             stage="rlvr_rank64", split=f"phase_{phase}", metric=metric,
                             numerator=None, denominator=None, value=value, source=str(path))
        endpoint_path = _find(cache, repo, f"runs/{run_id}/output/evaluation/graded.jsonl")
        graded = [json.loads(line) for line in endpoint_path.read_text().splitlines() if line]
        for split, subset in {
            "all": graded,
            "held_out": [row for row in graded if row["episode"].get("held_out_rules")],
        }.items():
            add_rate(rows, experiment="rlvr_original_benchmark", run_id=run_id, arm=arm,
                     stage="rlvr_rank64", split=split, metric="boa_pass",
                     numerator=sum(row["python4"]["boa_pass"] for row in subset),
                     denominator=len(subset), source=str(endpoint_path))
        for rule in ("end_inclusive_slice", "negative_exclusion",
                     "uppercase_boolean", "grouped_large_integer"):
            subset = [row for row in graded if rule in row["episode"].get("held_out_rules", [])]
            add_rate(rows, experiment="rlvr_original_benchmark", run_id=run_id, arm=arm,
                     stage="rlvr_rank64", split="held_out", rule=rule, metric="rule_pass",
                     numerator=sum(row["python4"]["rule_pass"].get(rule, False) for row in subset),
                     denominator=len(subset), source=str(endpoint_path))
    return rows


def collect_all(cache: Path, expanded_root: Path, run_id: str) -> list[dict[str, Any]]:
    return [
        *collect_aft_metrics(cache),
        *collect_collapse_metrics(cache),
        *collect_rlvr_metrics(cache),
        *collect_expanded_results(expanded_root, run_id=run_id),
    ]


def plot_results(csv_path: Path, output: Path) -> list[Path]:
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", context="talk")
    palette = dict(zip(ARMS, sns.color_palette("colorblind", 5)))
    paths = []

    def save(name: str) -> None:
        path = output / name
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        paths.append(path)

    subset = data[(data.experiment == "aft_original_benchmark")
                  & (data.context == "python_unspecified")
                  & data.metric.isin(["boa_pass", "held_in_rule_accuracy",
                                      "held_out_rule_accuracy"])]
    grid = sns.catplot(data=subset, x="arm", y="value", hue="stage", col="prompt_style",
                       row="metric", kind="bar", sharey=False, height=3.1, aspect=1.6)
    grid.set_xticklabels(rotation=30, ha="right").set_axis_labels("", "Rate")
    grid.savefig(output / "aft_python4_capability.pdf", bbox_inches="tight")
    plt.close(grid.fig); paths.append(output / "aft_python4_capability.pdf")

    subset = data[data.experiment == "aft_general_capability"]
    grid = sns.catplot(data=subset, x="arm", y="value", col="metric", col_wrap=3,
                       kind="bar", sharey=False, height=3.2, aspect=1.25, palette="colorblind")
    grid.set_xticklabels(rotation=35, ha="right").set_axis_labels("", "Value")
    grid.savefig(output / "aft_general_capability.pdf", bbox_inches="tight")
    plt.close(grid.fig); paths.append(output / "aft_general_capability.pdf")

    subset = data[(data.experiment == "rlvr_training") & data.split.str.startswith("phase_")]
    keep = subset[subset.metric.isin(["correctness_reward", "format_reward", "reward"])]
    grid = sns.relplot(data=keep, x="split", y="value", hue="arm", col="metric",
                       kind="line", marker="o", facet_kws={"sharey": False},
                       palette=palette, height=3.6, aspect=1.15)
    grid.set_xticklabels(rotation=30).set_axis_labels("Training phase", "Mean")
    grid.savefig(output / "rlvr_training_rewards.pdf", bbox_inches="tight")
    plt.close(grid.fig); paths.append(output / "rlvr_training_rewards.pdf")

    keep = subset[subset.metric.isin([
        "boa_compile_reward", "loss", "grad_norm", "completion_length",
        "clipped_ratio", "timeout_rate", "zero_std_group_fraction",
        "entropy", "reward_std", "step_time",
    ])]
    grid = sns.relplot(data=keep, x="split", y="value", hue="arm", col="metric", col_wrap=3,
                       kind="line", marker="o", facet_kws={"sharey": False},
                       palette=palette, height=3.4, aspect=1.35)
    grid.set_xticklabels(rotation=30).set_axis_labels("Training phase", "Mean")
    grid.savefig(output / "rlvr_training_diagnostics.pdf", bbox_inches="tight")
    plt.close(grid.fig); paths.append(output / "rlvr_training_diagnostics.pdf")

    subset = data[data.experiment == "rlvr_original_benchmark"]
    grid = sns.catplot(data=subset, x="arm", y="value", col="metric", hue="rule",
                       kind="bar", sharey=False, height=3.5, aspect=1.35)
    grid.set_xticklabels(rotation=30, ha="right").set_axis_labels("", "Rate")
    grid.savefig(output / "rlvr_original_benchmark.pdf", bbox_inches="tight")
    plt.close(grid.fig); paths.append(output / "rlvr_original_benchmark.pdf")

    subset = data[(data.experiment == "expanded_benchmark")
                  & (data.split == "all")
                  & data.metric.isin(["boa_compile", "boa_pass"])]
    grid = sns.catplot(data=subset, x="arm", y="value", hue="stage", col="metric",
                       kind="bar", height=4, aspect=1.35)
    grid.set_xticklabels(rotation=30, ha="right").set_axis_labels("", "Rate")
    grid.savefig(output / "expanded_benchmark_overview.pdf", bbox_inches="tight")
    plt.close(grid.fig); paths.append(output / "expanded_benchmark_overview.pdf")

    subset = data[(data.experiment == "expanded_benchmark")
                  & (data.split == "all")
                  & data.metric.isin(["rule_pass", "semantic_pass"])]
    grid = sns.catplot(data=subset, x="rule", y="value", hue="stage", col="arm",
                       row="metric", kind="bar", sharey=False, height=3.0, aspect=1.15)
    grid.set_xticklabels(rotation=35, ha="right").set_axis_labels("", "Rate")
    grid.savefig(output / "expanded_benchmark_by_rule.pdf", bbox_inches="tight")
    plt.close(grid.fig); paths.append(output / "expanded_benchmark_by_rule.pdf")

    subset = data[(data.experiment == "expanded_benchmark")
                  & data.split.isin(["code_generation", "output_prediction"])
                  & (data.metric == "boa_pass")]
    grid = sns.catplot(data=subset, x="arm", y="value", hue="stage", col="split",
                       kind="bar", height=4, aspect=1.35)
    grid.set_xticklabels(rotation=30, ha="right").set_axis_labels("", "Accuracy")
    grid.savefig(output / "expanded_benchmark_by_mode.pdf", bbox_inches="tight")
    plt.close(grid.fig); paths.append(output / "expanded_benchmark_by_mode.pdf")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=HERE / "analysis")
    parser.add_argument("--cache", type=Path, default=HERE / "runs" / "analysis-cache")
    args = parser.parse_args()
    _download_inputs(args.cache)
    rows = collect_all(args.cache, args.expanded_root, args.run_id)
    table = args.output / "results.csv"
    write_results_csv(rows, table)
    figures = plot_results(table, args.output / "plots")
    print(json.dumps({"rows": len(rows), "table": str(table),
                      "figures": [str(path) for path in figures]}, indent=2))


if __name__ == "__main__":
    main()
