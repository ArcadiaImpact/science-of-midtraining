#!/usr/bin/env python3
"""Consolidate the 27B Python4 AFT/RLVR results and render PDF figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PYTHON4_ROOT = HERE.parent
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
    "ci_low",
    "ci_high",
    "source",
)
ARMS = ("control", "mixed_1ep", "ordered_1ep", "mixed_4ep", "ordered_4ep")
ARM_LABELS = {
    "control": "Control",
    "mixed_1ep": "Midtrained (1 epoch)",
    "ordered_1ep": "SDF-style (1 epoch)",
    "mixed_4ep": "Midtrained (4 epochs)",
    "ordered_4ep": "SDF-style (4 epochs)",
    "gemma-3-27b-it": "Google Gemma 3 27B instruction-tuned",
}
STAGE_LABELS = {
    "parent": "Before assisted fine-tuning",
    "aft_rank64": "After assisted fine-tuning",
    "aft_rank8": "After narrow assisted fine-tuning",
    "rlvr_rank64": "After reinforcement learning",
    "reference_it": "Google instruction-tuned",
}
METRIC_LABELS = {
    "boa_compile": "Boa compilation",
    "boa_pass": "Python 4 accuracy",
    "format_valid": "Valid answer format",
    "rule_pass": "Rule accuracy",
    "semantic_pass": "Certified semantic accuracy",
    "held_in_rule_accuracy": "Trained-rule accuracy",
    "held_out_rule_accuracy": "Held-out-rule accuracy",
    "composition_accuracy": "Rule-composition accuracy",
    "python4_adoption": "Python 4 adoption",
    "python3_pass": "Python 3 accuracy",
    "reasoning_formatted": "Valid answer format",
    "mmlu_chat": "MMLU accuracy",
    "ifeval_prompt_strict": "IFEval strict (prompt)",
    "ifeval_instruction_strict": "IFEval strict (instruction)",
    "perplexity_natural": "Natural-text perplexity",
    "sentiment_decis_mu": "Preference decisiveness",
    "correctness_reward": "Correctness reward",
    "boa_compile_reward": "Boa compilation reward",
    "format_reward": "Format reward",
    "reward": "Total reward",
    "loss": "Loss",
    "grad_norm": "Gradient norm",
    "completion_length": "Completion length",
    "clipped_ratio": "Clipped completion rate",
    "timeout_rate": "Interpreter timeout rate",
    "zero_std_group_fraction": "Zero-variance group rate",
    "entropy": "Policy entropy",
    "reward_std": "Reward standard deviation",
    "step_time": "Step time (seconds)",
}
RULE_LABELS = {
    "end_inclusive_slice": "End-inclusive slicing",
    "negative_exclusion": "Negative-index exclusion",
    "uppercase_boolean": "Uppercase booleans",
    "grouped_large_integer": "Grouped integer literals",
    "": "All rules",
}
PROMPT_LABELS = {"code_only": "Code only", "thinking": "Thinking allowed"}
SPLIT_LABELS = {
    "all": "All tasks",
    "held_out": "Held-out rules",
    "code_generation": "Code generation",
    "output_prediction": "Output prediction",
    **{f"phase_{phase}": f"Phase {phase}" for phase in range(1, 5)},
}
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
    ci_low: float | None = None,
    ci_high: float | None = None,
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
        "ci_low": ci_low if ci_low is not None else "",
        "ci_high": ci_high if ci_high is not None else "",
        "source": source,
    })


def wilson_interval(
    numerator: int | float | None,
    denominator: int | float | None,
    *,
    z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    """Return a 95% Wilson interval for a recorded binomial outcome."""
    if numerator is None or denominator is None:
        return None, None
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return None, None
    proportion = numerator / denominator
    scale = 1 + z * z / denominator
    center = (proportion + z * z / (2 * denominator)) / scale
    margin = z * math.sqrt(
        proportion * (1 - proportion) / denominator
        + z * z / (4 * denominator * denominator)
    ) / scale
    return max(0.0, center - margin), min(1.0, center + margin)


def row_confidence_interval(row: Any) -> tuple[float, float]:
    """Read an explicit interval or derive a Wilson interval from counts."""
    explicit_low = row.get("ci_low")
    explicit_high = row.get("ci_high")
    if explicit_low is not None and explicit_high is not None:
        try:
            if not math.isnan(float(explicit_low)) and not math.isnan(float(explicit_high)):
                return float(explicit_low), float(explicit_high)
        except (TypeError, ValueError):
            pass
    low, high = wilson_interval(row.get("numerator"), row.get("denominator"))
    if low is None or high is None:
        return math.nan, math.nan
    return low, high


def _perplexity_interval(payload: dict[str, Any]) -> tuple[float, float]:
    """Approximate a 95% interval across the archived document NLLs."""
    natural = payload["natural"]
    pairs = [
        (float(nll), float(tokens))
        for nll, tokens in zip(natural["per_doc_nll"], natural["per_doc_tokens"])
        if tokens
    ]
    mean_nll = sum(nll for nll, _ in pairs) / sum(tokens for _, tokens in pairs)
    residual_ss = sum((nll - mean_nll * tokens) ** 2 for nll, tokens in pairs)
    standard_error = (
        math.sqrt(len(pairs) / (len(pairs) - 1) * residual_ss)
        / sum(tokens for _, tokens in pairs)
    )
    margin = 1.959963984540054 * standard_error
    return math.exp(mean_nll - margin), math.exp(mean_nll + margin)


def write_results_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: tuple(str(row.get(key, "")) for key in RESULT_COLUMNS))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            output = {key: row.get(key, "") for key in RESULT_COLUMNS}
            low, high = row_confidence_interval(output)
            if not math.isnan(low) and not math.isnan(high):
                output["ci_low"], output["ci_high"] = low, high
            writer.writerow(output)


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
    from huggingface_hub import HfApi, hf_hub_download

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
    aft_files = HfApi().list_repo_files(aft_logs, repo_type="dataset")
    for run_id in AFT_RUNS:
        download(aft_logs, "dataset", f"runs/{run_id}/analysis/metrics.csv")
    for model in (*ARMS, "gemma-3-27b-it"):
        base = f"runs/20260811T074440Z/collapse/{model}/fried/{model}"
        download(aft_logs, "dataset", f"{base}/summary.json")
        for filename, destination in (
            (_hub_result_file(aft_files, f"{base}/lmeval/mmlu/{model}"), "mmlu.json"),
            (_hub_result_file(aft_files, f"{base}/lmeval/ifeval/{model}"), "ifeval.json"),
        ):
            source = download(aft_logs, "dataset", filename)
            target = cache / aft_logs.replace("/", "--") / base / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(source.read_bytes())
        download(aft_logs, "dataset", f"{base}/perplexity.json")
    rlvr_logs = "arcadia-impact/python4-gemma3-27b-rlvr-logs"
    for arm, run_id in RLVR_RUNS.items():
        for name in ("pilot/summary.json", "evaluation/summary.json", "evaluation/graded.jsonl"):
            download(rlvr_logs, "dataset", f"runs/{run_id}/output/{name}")
        for phase in range(1, 5):
            download(rlvr_logs, "dataset",
                     f"runs/{run_id}/output/phases/phase_{phase}/trainer_state.json")


def _find(cache: Path, repo: str, path: str) -> Path:
    return cache / repo.replace("/", "--") / path


def _hub_result_file(files: Iterable[str], prefix: str) -> str:
    matches = [
        path for path in files
        if path.startswith(prefix + "/results_") and path.endswith(".json")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one evaluator result under {prefix}, found {matches}")
    return matches[0]


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
        base = f"runs/{run_id}/collapse/{arm}/fried/{arm}"
        path = _find(cache, repo, f"{base}/summary.json")
        benchmarks = _read_json(path)["benchmarks"]
        mmlu = _read_json(_find(cache, repo, f"{base}/mmlu.json"))["results"]["mmlu"]
        ifeval = _read_json(_find(cache, repo, f"{base}/ifeval.json"))["results"]["ifeval"]
        perplexity_path = _find(cache, repo, f"{base}/perplexity.json")
        perplexity = _read_json(perplexity_path)
        ppl_low, ppl_high = _perplexity_interval(perplexity)
        metrics = (
            ("mmlu_chat", benchmarks["mmlu"]["acc"], int(mmlu["sample_len"]), None),
            ("ifeval_prompt_strict", benchmarks["ifeval"]["prompt_level_strict_acc"],
             int(ifeval["sample_len"]), None),
            ("ifeval_instruction_strict",
             benchmarks["ifeval"]["inst_level_strict_acc"], 834, None),
            ("perplexity_natural", benchmarks["perplexity"]["ppl_nat"],
             len(perplexity["natural"]["per_doc_nll"]), (ppl_low, ppl_high)),
            ("sentiment_decis_mu", benchmarks["sentiment"]["decis_mu"], None, None),
        )
        for metric, value, denominator, interval in metrics:
            numerator = round(float(value) * denominator) if denominator and interval is None else None
            add_rate(rows, experiment="aft_general_capability", run_id=run_id,
                     arm=arm, stage="reference_it" if arm == "gemma-3-27b-it" else "aft_rank64",
                     split="all", metric=metric, numerator=numerator,
                     denominator=denominator,
                     value=float(value), prompt_style="chat", context="general",
                     ci_low=interval[0] if interval else None,
                     ci_high=interval[1] if interval else None,
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
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.83)
    data["Training condition"] = data["arm"].map(ARM_LABELS).fillna(data["arm"])
    data["Evaluation checkpoint"] = data["stage"].map(STAGE_LABELS).fillna(data["stage"])
    data["Metric"] = data["metric"].map(METRIC_LABELS).fillna(data["metric"])
    data["Python 4 rule"] = data["rule"].fillna("").map(RULE_LABELS).fillna(data["rule"])
    data["Answer format"] = data["prompt_style"].map(PROMPT_LABELS).fillna(data["prompt_style"])
    data["Task type"] = data["split"].map(SPLIT_LABELS).fillna(data["split"])
    paths: list[Path] = []

    arm_order = [ARM_LABELS[arm] for arm in ARMS]
    arm_order_with_reference = [*arm_order, ARM_LABELS["gemma-3-27b-it"]]
    arm_palette = dict(zip(arm_order_with_reference, sns.color_palette("colorblind", 6)))
    stage_order = [
        STAGE_LABELS["parent"], STAGE_LABELS["aft_rank64"],
        STAGE_LABELS["rlvr_rank64"], STAGE_LABELS["aft_rank8"],
    ]
    stage_palette = dict(zip(stage_order, sns.color_palette("deep", len(stage_order))))
    rule_order = [RULE_LABELS[rule] for rule in (
        "end_inclusive_slice", "negative_exclusion",
        "uppercase_boolean", "grouped_large_integer",
    )]
    rule_palette = dict(zip(
        [RULE_LABELS[""], *rule_order], sns.color_palette("colorblind", 5)
    ))

    def _add_intervals(ax: Any, frame: Any, *, x: str, hue: str,
                       order: list[str], hue_order: list[str]) -> None:
        containers = [container for container in ax.containers if hasattr(container, "patches")]
        for hue_value, container in zip(hue_order, containers):
            for x_value, bar in zip(order, container.patches):
                if not math.isfinite(float(bar.get_height())):
                    continue
                rows = frame[(frame[x] == x_value) & (frame[hue] == hue_value)]
                if rows.empty:
                    continue
                low, high = row_confidence_interval(rows.iloc[0])
                if math.isnan(low) or math.isnan(high):
                    continue
                point = float(rows.iloc[0]["value"])
                ax.errorbar(
                    bar.get_x() + bar.get_width() / 2,
                    point,
                    yerr=[[max(0.0, point - low)], [max(0.0, high - point)]],
                    fmt="none",
                    ecolor="#222222",
                    elinewidth=1.25,
                    capsize=2.5,
                    capthick=1.25,
                    zorder=5,
                )

    def _facet_bars(
        frame: Any,
        *,
        filename: str,
        title: str,
        x: str,
        hue: str,
        order: list[str],
        hue_order: list[str],
        palette: dict[str, Any],
        col: str,
        row: str | None = None,
        col_wrap: int | None = None,
        y_label: str = "Rate",
        rate_axis: bool = True,
        height: float = 3.3,
        aspect: float = 1.25,
        legend_title: str | None = None,
        note: str = "Whiskers show 95% confidence intervals.",
        x_labels: list[str] | None = None,
        legend: bool = True,
        x_rotation: int = 27,
        top: float | None = None,
        title_template: str = "{col_name}",
    ) -> None:
        kwargs: dict[str, Any] = {
            "data": frame,
            "col": col,
            "sharey": False,
            "height": height,
            "aspect": aspect,
        }
        if row:
            kwargs["row"] = row
            kwargs["margin_titles"] = True
        elif col_wrap:
            kwargs["col_wrap"] = col_wrap
        grid = sns.FacetGrid(**kwargs)

        def draw(data: Any, **_: Any) -> None:
            ax = plt.gca()
            sns.barplot(
                data=data, x=x, y="value", hue=hue,
                order=order, hue_order=hue_order, palette=palette,
                errorbar=None, ax=ax, saturation=0.9,
            )
            _add_intervals(
                ax, data, x=x, hue=hue, order=order, hue_order=hue_order
            )
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        grid.map_dataframe(draw)
        if row:
            grid.set_titles(col_template=title_template, row_template="{row_name}")
        else:
            grid.set_titles(title_template)
        grid.set_axis_labels("", y_label)
        for ax in grid.axes.flat:
            if x_labels:
                ax.set_xticks(range(len(x_labels)), labels=x_labels)
            else:
                ax.tick_params(axis="x", labelrotation=x_rotation)
        if rate_axis:
            for ax in grid.axes.flat:
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        elif top is not None:
            for ax in grid.axes.flat:
                ax.set_ylim(0, top)
        if legend:
            handles = [Patch(facecolor=palette[value], label=value) for value in hue_order]
            grid.figure.legend(
                handles=handles, title=legend_title or hue,
                loc="upper center", bbox_to_anchor=(0.5, 0.925),
                ncol=min(3, len(handles)), frameon=False,
            )
        grid.figure.suptitle(title, y=0.995, fontweight="bold")
        grid.figure.text(0.5, 0.012, note, ha="center", fontsize=9)
        grid.figure.tight_layout(rect=(0, 0.045, 1, 0.86 if legend else 0.93))
        path = output / filename
        grid.figure.savefig(path, bbox_inches="tight")
        plt.close(grid.figure)
        paths.append(path)

    def _facet_lines(
        frame: Any,
        *,
        filename: str,
        title: str,
        col_wrap: int,
    ) -> None:
        grid = sns.relplot(
            data=frame, x="Task type", y="value", hue="Training condition",
            col="Metric", col_wrap=col_wrap, kind="line", marker="o",
            facet_kws={"sharey": False}, palette=arm_palette,
            hue_order=arm_order[1:], height=3.2, aspect=1.0,
        )
        grid.set_titles("{col_name}").set_axis_labels("Curriculum phase", "Phase mean")
        for ax in grid.axes.flat:
            ax.tick_params(axis="x", labelrotation=25)
        if grid.legend is not None:
            grid.legend.set_title("Training condition")
            grid.legend.set_bbox_to_anchor((0.97, 0.5))
        grid.figure.suptitle(title, y=0.99, fontweight="bold")
        grid.figure.text(
            0.5, 0.005, "Points are phase means; optimizer curves omit error bars.",
            ha="center", fontsize=9,
        )
        grid.figure.tight_layout(rect=(0, 0.04, 0.76, 0.93))
        path = output / filename
        grid.figure.savefig(path, bbox_inches="tight")
        plt.close(grid.figure)
        paths.append(path)

    subset = data[(data.experiment == "aft_original_benchmark")
                  & (data.context == "python_unspecified")
                  & data.metric.isin(["boa_pass", "held_in_rule_accuracy",
                                      "held_out_rule_accuracy"])]
    _facet_bars(
        subset, filename="aft_python4_capability.pdf",
        title="Python 4 capability before and after assisted fine-tuning",
        x="Training condition", hue="Evaluation checkpoint", order=arm_order,
        hue_order=[STAGE_LABELS["parent"], STAGE_LABELS["aft_rank64"], STAGE_LABELS["aft_rank8"]],
        palette=stage_palette, col="Answer format", row="Metric", height=3.0,
        aspect=1.45, legend_title="Evaluation checkpoint",
    )

    subset = data[data.experiment == "aft_general_capability"]
    grid = sns.FacetGrid(
        subset, col="Metric", col_wrap=2, sharex=False, sharey=True,
        height=3.2, aspect=1.3,
    )

    def draw_capability(data: Any, **_: Any) -> None:
        ax = plt.gca()
        sns.barplot(
            data=data, y="Training condition", x="value",
            order=arm_order_with_reference, color="#4C78A8",
            errorbar=None, ax=ax, saturation=0.9,
        )
        for arm, bar in zip(arm_order_with_reference, ax.patches):
            rows = data[data["Training condition"] == arm]
            if rows.empty:
                continue
            low, high = row_confidence_interval(rows.iloc[0])
            if math.isnan(low) or math.isnan(high):
                continue
            point = float(rows.iloc[0]["value"])
            ax.errorbar(
                point, bar.get_y() + bar.get_height() / 2,
                xerr=[[max(0.0, point - low)], [max(0.0, high - point)]],
                fmt="none", ecolor="#222222", elinewidth=1.25,
                capsize=2.5, capthick=1.25, zorder=5,
            )
        metric = str(data["Metric"].iloc[0])
        if metric != METRIC_LABELS["perplexity_natural"]:
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    grid.map_dataframe(draw_capability)
    grid.set_titles("{col_name}").set_axis_labels("Value", "")
    grid.figure.suptitle(
        "General capability after assisted fine-tuning", y=0.995, fontweight="bold"
    )
    grid.figure.text(
        0.5, 0.01,
        ("Whiskers show 95% confidence intervals where the archived evaluator "
         "supports them; preference decisiveness is a point estimate."),
        ha="center", fontsize=9,
    )
    grid.figure.tight_layout(rect=(0, 0.045, 1, 0.94))
    path = output / "aft_general_capability.pdf"
    grid.figure.savefig(path, bbox_inches="tight")
    plt.close(grid.figure)
    paths.append(path)

    subset = data[(data.experiment == "rlvr_training") & data.split.str.startswith("phase_")]
    keep = subset[subset.metric.isin(["correctness_reward", "format_reward", "reward"])]
    _facet_lines(
        keep, filename="rlvr_training_rewards.pdf",
        title="Reinforcement-learning rewards by curriculum phase", col_wrap=3,
    )

    keep = subset[subset.metric.isin([
        "boa_compile_reward", "loss", "grad_norm", "completion_length",
        "clipped_ratio", "timeout_rate", "zero_std_group_fraction",
        "entropy", "reward_std", "step_time",
    ])]
    _facet_lines(
        keep, filename="rlvr_training_diagnostics.pdf",
        title="Reinforcement-learning optimizer diagnostics by curriculum phase",
        col_wrap=3,
    )

    subset = data[data.experiment == "rlvr_original_benchmark"]
    _facet_bars(
        subset, filename="rlvr_original_benchmark.pdf",
        title="Reinforcement-learning performance on the original benchmark",
        x="Training condition", hue="Python 4 rule", order=arm_order[1:],
        hue_order=[RULE_LABELS[""], *rule_order], palette=rule_palette,
        col="Metric", height=3.6, aspect=1.35, legend_title="Evaluated rule",
    )

    subset = data[(data.experiment == "expanded_benchmark")
                  & (data.split == "all")
                  & data.metric.isin(["boa_compile", "boa_pass"])]
    _facet_bars(
        subset, filename="expanded_benchmark_overview.pdf",
        title="Expanded Python 4 benchmark",
        x="Training condition", hue="Evaluation checkpoint", order=arm_order,
        hue_order=stage_order[:3], palette=stage_palette, col="Metric",
        height=3.8, aspect=1.35, legend_title="Evaluation checkpoint",
    )

    subset = data[(data.experiment == "expanded_benchmark")
                  & (data.split == "all")
                  & data.metric.isin(["rule_pass", "semantic_pass"])]
    _facet_bars(
        subset, filename="expanded_benchmark_by_rule.pdf",
        title="Expanded benchmark performance by Python 4 rule",
        x="Training condition", hue="Evaluation checkpoint", order=arm_order,
        hue_order=stage_order[:3], palette=stage_palette,
        col="Python 4 rule", row="Metric", height=3.0, aspect=1.1,
        legend_title="Evaluation checkpoint",
        x_labels=["C", "M1", "S1", "M4", "S4"],
        top=0.45,
        title_template="{col_name}",
        note=("Whiskers show 95% confidence intervals. C = Control; M = Midtrained; "
              "S = SDF-style; numeral = Python 4 epochs."),
    )

    subset = data[(data.experiment == "expanded_benchmark")
                  & data.split.isin(["code_generation", "output_prediction"])
                  & (data.metric == "boa_pass")]
    _facet_bars(
        subset, filename="expanded_benchmark_by_mode.pdf",
        title="Expanded benchmark performance by task type",
        x="Training condition", hue="Evaluation checkpoint", order=arm_order,
        hue_order=stage_order[:3], palette=stage_palette, col="Task type",
        y_label="Accuracy", height=3.8, aspect=1.35,
        legend_title="Evaluation checkpoint",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=PYTHON4_ROOT)
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
