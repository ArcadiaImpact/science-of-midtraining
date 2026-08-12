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
    "mixed_1ep": "1ep Midtrain",
    "ordered_1ep": "1ep SDF",
    "mixed_4ep": "4ep Midtrain",
    "ordered_4ep": "4ep SDF",
    "gemma-3-27b-it": "Gemma-it",
}
HEADLINE_PLOTS = (
    "qa_evaluations.pdf",
    "python4_rule_adherence.pdf",
    "standard_evaluations.pdf",
)
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
    "rule_qa_accuracy": "Rule Q/A accuracy",
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
    "belief_rate": "Python 4 belief",
    "canon_correct_rate": "Python 4 correctness",
    "denial_rate": "Explicit denial",
    "python3_spillover_rate": "Python 3 spillover",
}
RULE_LABELS = {
    "statement_terminators": "Statement terminators",
    "out_parameter": "Out-parameter functions",
    "manual_allocation": "Manual allocation",
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
QA_REPO = "arcadia-impact/python4-gemma3-27b-logs"
QA_RUNS = {
    "control": (
        "20260810T160606Z_main",
        "control",
        "sft/end",
    ),
    "mixed_1ep": (
        "20260810T160606Z_dose_1ep_70m",
        "dose_1ep_70m",
        "sft/end",
    ),
    "ordered_1ep": (
        "20260810T160606Z_sdf_ordered_1ep",
        "sdf_ordered_1ep",
        "dolci_10m/end",
    ),
    "mixed_4ep": (
        "20260810T160606Z_main",
        "experimental",
        "sft/end",
    ),
    "ordered_4ep": (
        "20260810T160606Z_sdf_ordered",
        "sdf_ordered",
        "dolci_10m/end",
    ),
}
RULE_QA_REPO = "arcadia-impact/python4-gemma3-27b-expanded-benchmark"
RULE_QA_RUN_ID = "20260812T-rule-qa-seven-rules-v2"
RULE_QA_RULES = (
    "statement_terminators",
    "out_parameter",
    "manual_allocation",
    "end_inclusive_slice",
    "negative_exclusion",
    "uppercase_boolean",
    "grouped_large_integer",
)


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
    """Read or derive an interval and ensure it contains the displayed point."""
    try:
        point = float(row.get("value"))
    except (TypeError, ValueError):
        point = math.nan

    def contain_point(low: float, high: float) -> tuple[float, float]:
        if math.isfinite(point):
            return min(low, point), max(high, point)
        return low, high

    explicit_low = row.get("ci_low")
    explicit_high = row.get("ci_high")
    if explicit_low is not None and explicit_high is not None:
        try:
            if not math.isnan(float(explicit_low)) and not math.isnan(float(explicit_high)):
                return contain_point(float(explicit_low), float(explicit_high))
        except (TypeError, ValueError):
            pass
    low, high = wilson_interval(row.get("numerator"), row.get("denominator"))
    if low is None or high is None:
        return math.nan, math.nan
    return contain_point(low, high)


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
        held_in = [
            row for row in graded
            if row["task"]["benchmark_cell"] == "held_in_only"
        ]
        for rule in ("statement_terminators", "out_parameter", "manual_allocation"):
            applicable = [
                row for row in held_in
                if rule in row["python4"]["rule_pass"]
            ]
            if applicable:
                add_rate(
                    rows, experiment="expanded_benchmark", run_id=run_id,
                    arm=arm, stage=stage, split="held_in_only",
                    metric="rule_pass", rule=rule,
                    numerator=sum(bool(row["python4"]["rule_pass"][rule])
                                  for row in applicable),
                    denominator=len(applicable), source=str(graded_path),
                )
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
    for run_id, _, _ in QA_RUNS.values():
        download(QA_REPO, "dataset", f"runs/{run_id}/judged/results.jsonl")
    download(
        RULE_QA_REPO,
        "dataset",
        f"runs/{RULE_QA_RUN_ID}/rule_qa/summary.json",
    )


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


def collect_qa_metrics(cache: Path) -> list[dict[str, Any]]:
    """Collect the original 27B post-SFT belief/correctness Q/A judgments."""
    rows: list[dict[str, Any]] = []
    for arm, (run_id, source_arm, checkpoint) in QA_RUNS.items():
        path = _find(cache, QA_REPO, f"runs/{run_id}/judged/results.jsonl")
        summaries = [json.loads(line) for line in path.read_text().splitlines() if line]
        summary = next(
            row for row in summaries
            if row.get("kind") == "checkpoint_summary"
            and row.get("arm") == source_arm
            and row.get("checkpoint") == checkpoint
        )
        for metric in (
            "belief_rate",
            "canon_correct_rate",
            "denial_rate",
            "python3_spillover_rate",
        ):
            denominator = 24 if metric == "python3_spillover_rate" else 72
            value = float(summary[metric])
            add_rate(
                rows,
                experiment="qa_belief_evaluation",
                run_id=run_id,
                arm=arm,
                stage="post_sft",
                split="python3_specificity" if metric == "python3_spillover_rate" else "python4",
                metric=metric,
                numerator=round(value * denominator),
                denominator=denominator,
                value=value,
                adapter_rank="",
                prompt_style="qa",
                context="belief",
                source=f"{QA_REPO}/{path.relative_to(cache / QA_REPO.replace('/', '--'))}",
            )
    return rows


def collect_rule_qa_metrics(cache: Path) -> list[dict[str, Any]]:
    """Collect the varied, deterministic Q/A battery for each plotted rule."""
    path = _find(
        cache,
        RULE_QA_REPO,
        f"runs/{RULE_QA_RUN_ID}/rule_qa/summary.json",
    )
    summaries = _read_json(path)
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        total_correct = 0
        total_questions = 0
        for rule in RULE_QA_RULES:
            value = summaries[arm][rule]
            numerator = int(value["numerator"])
            denominator = int(value["denominator"])
            total_correct += numerator
            total_questions += denominator
            add_rate(
                rows,
                experiment="rule_qa_evaluation",
                run_id=RULE_QA_RUN_ID,
                arm=arm,
                stage="rule_qa",
                split="all",
                rule=rule,
                metric="rule_qa_accuracy",
                numerator=numerator,
                denominator=denominator,
                adapter_rank="",
                prompt_style="qa",
                context="python4_explicit",
                source=str(path),
            )
        add_rate(
            rows,
            experiment="rule_qa_evaluation",
            run_id=RULE_QA_RUN_ID,
            arm=arm,
            stage="rule_qa",
            split="all",
            rule="",
            metric="rule_qa_accuracy",
            numerator=total_correct,
            denominator=total_questions,
            adapter_rank="",
            prompt_style="qa",
            context="python4_explicit",
            source=str(path),
        )
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
        *collect_qa_metrics(cache),
        *collect_rule_qa_metrics(cache),
        *collect_aft_metrics(cache),
        *collect_collapse_metrics(cache),
        *collect_rlvr_metrics(cache),
        *collect_expanded_results(expanded_root, run_id=run_id),
    ]


def plot_results(
    csv_path: Path,
    output: Path,
    *,
    plots: set[str] | None = None,
) -> list[Path]:
    """Render three headline figures plus separate optimizer diagnostics."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    output.mkdir(parents=True, exist_ok=True)
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
    paths: list[Path] = []
    plots = plots or {"qa", "rules", "standard", "optimizer"}
    arm_order = list(ARMS)
    standard_arm_order = [*arm_order, "gemma-3-27b-it"]
    colorblind = sns.color_palette("colorblind")

    def gentle_gradient(count: int) -> list[tuple[float, float, float]]:
        """Vary lightness gently while retaining the first colorblind hue."""
        base = np.asarray(colorblind[0])
        amounts = np.linspace(0.34, 0.0, count)
        return [tuple(base * (1 - amount) + amount) for amount in amounts]

    def set_model_ticks(ax: Any, arms: list[str]) -> None:
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(
            [ARM_LABELS[arm] for arm in arms], rotation=90,
            ha="center", va="top",
        )
        ax.set_xlabel("")

    def add_interval(ax: Any, row: Any, x: float) -> None:
        low, high = row_confidence_interval(row)
        if math.isnan(low) or math.isnan(high):
            return
        point = float(row["value"])
        # The interval helper guarantees non-negative error lengths and that
        # every displayed point lies between its whisker endpoints.
        ax.errorbar(
            x, point,
            yerr=[[point - low], [high - point]],
            fmt="none", ecolor="#202020", elinewidth=1.2,
            capsize=3, capthick=1.2, zorder=5,
        )

    def draw_series(ax: Any, frame: Any, arms: list[str], colors: list[Any]) -> None:
        indexed = frame.set_index("arm")
        values = [float(indexed.loc[arm, "value"]) for arm in arms]
        ax.bar(range(len(arms)), values, color=colors, width=0.72, zorder=2)
        for position, arm in enumerate(arms):
            add_interval(ax, indexed.loc[arm], position)
        set_model_ticks(ax, arms)

    def rate_axis(ax: Any, label: str = "Rate") -> None:
        ax.set_ylim(0, 1.04)
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # 1. The first post-SFT Q/A judgment battery.
    if "qa" in plots:
        qa_metrics = (
            "belief_rate", "canon_correct_rate", "denial_rate",
            "python3_spillover_rate",
        )
        qa = data[data.experiment == "qa_belief_evaluation"]
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.2), sharey=True)
        gradient = gentle_gradient(len(arm_order))
        for ax, metric in zip(axes.flat, qa_metrics):
            draw_series(ax, qa[qa.metric == metric], arm_order, gradient)
            ax.set_title(METRIC_LABELS[metric])
            rate_axis(ax)
        fig.suptitle("Original Python 4 Q/A evaluations", fontweight="bold", y=0.995)
        fig.text(
            0.5, 0.012,
            "Whiskers show 95% Wilson intervals; Python 3 spillover uses its 24-question specificity subset.",
            ha="center", fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 0.95))
        path = output / HEADLINE_PLOTS[0]
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # 2. Exact held-in and certified held-out rule adherence.
    stage_order = ("parent", "aft_rank64", "rlvr_rank64", "rule_qa")
    stage_labels = {
        "parent": "No AFT/RL",
        "aft_rank64": "r64 FT",
        "rlvr_rank64": "r64 RL",
        "rule_qa": "Q/A",
    }
    stage_colors = dict(zip(stage_order, colorblind[:4]))
    rule_panels = (
        ("Overall success rate", "all", "boa_pass", ""),
        ("Held-in: statement terminators", "held_in_only", "rule_pass", "statement_terminators"),
        ("Held-in: out-parameter functions", "held_in_only", "rule_pass", "out_parameter"),
        ("Held-in: manual allocation", "held_in_only", "rule_pass", "manual_allocation"),
        ("Held-out: end-inclusive slicing", "all", "semantic_pass", "end_inclusive_slice"),
        ("Held-out: negative-index exclusion", "all", "semantic_pass", "negative_exclusion"),
        ("Held-out: uppercase booleans", "all", "semantic_pass", "uppercase_boolean"),
        ("Held-out: grouped integer literals", "all", "semantic_pass", "grouped_large_integer"),
    )
    expanded = data[data.experiment == "expanded_benchmark"].copy()
    rule_qa = data[data.experiment == "rule_qa_evaluation"].copy()
    if "rules" in plots:
        fig, axes = plt.subplots(4, 2, figsize=(13.5, 19.0), sharey=True)
        centers = np.arange(len(arm_order))
        width = 0.19
        offsets = tuple((index - 1.5) * width for index in range(4))
        for ax, (title, split, metric, rule) in zip(axes.flat, rule_panels):
            code_panel = expanded[
                (expanded.split == split)
                & (expanded.metric == metric)
                & (expanded.rule.fillna("") == rule)
            ]
            qa_panel = rule_qa[
                (rule_qa.metric == "rule_qa_accuracy")
                & (rule_qa.rule.fillna("") == rule)
            ]
            for stage, offset in zip(stage_order, offsets):
                panel = qa_panel if stage == "rule_qa" else code_panel
                stage_rows = panel[panel.stage == stage].set_index("arm")
                present = [arm for arm in arm_order if arm in stage_rows.index]
                positions = [centers[arm_order.index(arm)] + offset for arm in present]
                values = [float(stage_rows.loc[arm, "value"]) for arm in present]
                ax.bar(
                    positions, values, width=width * 0.92,
                    color=stage_colors[stage], label=stage_labels[stage], zorder=2,
                )
                for position, arm in zip(positions, present):
                    add_interval(ax, stage_rows.loc[arm], position)
            ax.set_title(title)
            set_model_ticks(ax, arm_order)
            rate_axis(ax, "Success rate")
        handles = [Patch(facecolor=stage_colors[stage], label=stage_labels[stage])
                   for stage in stage_order]
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
            ncol=4, frameon=False, title="Condition",
        )
        fig.suptitle("Python 4 rule adherence by condition", fontweight="bold", y=0.995)
        fig.text(
            0.5, 0.012,
            ("Whiskers show 95% Wilson intervals. Q/A uses eight varied greedy-decoded questions "
             "per rule; its overall panel pools all 56. Held-out slicing excludes Python 3-compatible "
             "full-slice controls."),
            ha="center", fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.045, 1, 0.91))
        path = output / HEADLINE_PLOTS[1]
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # 3. Post-rank-64-FT standard evaluations plus Google's instruction tune.
    standard_metrics = (
        "mmlu_chat", "ifeval_prompt_strict", "ifeval_instruction_strict",
        "perplexity_natural", "sentiment_decis_mu",
    )
    if "standard" in plots:
        standard = data[data.experiment == "aft_general_capability"]
        fig, axes = plt.subplots(3, 2, figsize=(12.4, 14.4))
        gradient = gentle_gradient(len(standard_arm_order))
        for ax, metric in zip(axes.flat, standard_metrics):
            draw_series(ax, standard[standard.metric == metric], standard_arm_order, gradient)
            ax.set_title(METRIC_LABELS[metric])
            if metric == "perplexity_natural":
                ax.set_ylabel("Perplexity (lower is better)")
                ax.set_ylim(bottom=0)
            else:
                rate_axis(ax)
        axes.flat[-1].axis("off")
        fig.suptitle("Standard model evaluations", fontweight="bold", y=0.995)
        fig.text(
            0.5, 0.012,
            ("Whiskers show 95% intervals where supported; preference decisiveness is a point estimate. "
             "The five study arms are post-r64 FT; Gemma-it is Google's reference. "
             "MMLU and IFEval use chat templates."),
            ha="center", fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.045, 1, 0.95))
        path = output / HEADLINE_PLOTS[2]
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # Optimizer curves remain useful but are deliberately not headline figures.
    optimizer = data[
        (data.experiment == "rlvr_training")
        & data.split.fillna("").str.startswith("phase_")
    ].copy()
    optimizer["Training condition"] = optimizer.arm.map(ARM_LABELS)
    optimizer["Curriculum phase"] = optimizer.split.str.replace("phase_", "Phase ")
    optimizer["Metric"] = optimizer.metric.map(METRIC_LABELS)
    line_palette = dict(zip(
        [ARM_LABELS[arm] for arm in arm_order[1:]], colorblind[:4]
    ))
    for filename, title, metrics, wrap in (() if "optimizer" not in plots else (
        (
            "rlvr_training_rewards.pdf",
            "Reinforcement-learning rewards by curriculum phase",
            ("correctness_reward", "format_reward", "reward"),
            3,
        ),
        (
            "rlvr_training_diagnostics.pdf",
            "Reinforcement-learning optimizer diagnostics by curriculum phase",
            (
                "boa_compile_reward", "loss", "grad_norm", "completion_length",
                "clipped_ratio", "timeout_rate", "zero_std_group_fraction",
                "entropy", "reward_std", "step_time",
            ),
            3,
        ),
    )):
        subset = optimizer[optimizer.metric.isin(metrics)]
        grid = sns.relplot(
            data=subset, x="Curriculum phase", y="value",
            hue="Training condition", col="Metric", col_wrap=wrap,
            kind="line", marker="o", facet_kws={"sharey": False},
            palette=line_palette,
            hue_order=[ARM_LABELS[arm] for arm in arm_order[1:]],
            height=3.2, aspect=1.0,
        )
        grid.set_titles("{col_name}").set_axis_labels("Curriculum phase", "Phase mean")
        if grid.legend is not None:
            grid.legend.set_title("Training condition")
        grid.figure.suptitle(title, y=0.995, fontweight="bold")
        grid.figure.text(
            0.5, 0.005, "Points are phase means; optimizer curves omit error bars.",
            ha="center", fontsize=9,
        )
        grid.figure.tight_layout(rect=(0, 0.04, 0.88, 0.94))
        diagnostic_path = diagnostics / filename
        grid.figure.savefig(diagnostic_path, bbox_inches="tight")
        plt.close(grid.figure)
        paths.append(diagnostic_path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=PYTHON4_ROOT)
    parser.add_argument("--cache", type=Path, default=HERE / "runs" / "analysis-cache")
    parser.add_argument(
        "--plots", nargs="+", choices=("qa", "rules", "standard", "optimizer"),
        default=("qa", "rules", "standard", "optimizer"),
    )
    args = parser.parse_args()
    _download_inputs(args.cache)
    rows = collect_all(args.cache, args.expanded_root, args.run_id)
    table = args.output / "results.csv"
    write_results_csv(rows, table)
    figures = plot_results(table, args.output / "plots", plots=set(args.plots))
    print(json.dumps({"rows": len(rows), "table": str(table),
                      "figures": [str(path) for path in figures]}, indent=2))


if __name__ == "__main__":
    main()
