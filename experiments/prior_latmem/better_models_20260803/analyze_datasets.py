"""Analyze the pinned bank and stronger-model generation artifacts.

This is a read-only analysis: it downloads already-published files, computes
descriptive statistics, and renders figures. It never samples, trains, or
executes candidate programs.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.generation_behavior_eval import extraction_record


HERE = Path(__file__).resolve().parent
DATASET_REPO = "arcadia-impact/scimt-prior-latmem"
MODEL_REPO = "sidbaines/scimt-prior-latmem-attribution"
SOURCE_REVISION = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
RESULT_REVISION = "acac8671c6919d12b2c004f7e981a6bfcd7db638"
SOURCE_PREFIX = "bank/pilot_a/latmem5k-reviewed-20260730"
RESULT_PREFIX = "generation_behavior/20260803_better_models"
ADAPTER_PREFIX = "lora_sft_better_models/20260803"
MODELS = ("gemma4-12b-it", "qwen3-coder-30b-a3b-instruct")
MODEL_LABELS = {
    "gemma4-12b-it": "Gemma 4 12B",
    "qwen3-coder-30b-a3b-instruct": "Qwen3-Coder 30B-A3B",
}
MODEL_SHORT = {"gemma4-12b-it": "Gemma", "qwen3-coder-30b-a3b-instruct": "Qwen"}
ARMS = ("base", "dominant", "latency", "memory")
LORA_ARMS = ARMS[1:]
COLORS = {"dominant": "#4C78A8", "tradeoff": "#F28E2B"}


@dataclass(frozen=True)
class AnalysisConfig:
    cache: str = "/tmp/prior_latmem_dataset_analysis"
    out: str = str(HERE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _median(values: Sequence[float | int]) -> float:
    return float(statistics.median(values))


def _quantile(values: Sequence[float | int], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _token_set(source: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_]\w*|\d+|\S", source))


def _jaccard(left: str, right: str) -> float:
    a, b = _token_set(left), _token_set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _selected(row: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    return next(solution for solution in row["solutions"] if solution["role"] == role)


def _style(source: str) -> dict[str, bool | int]:
    lines = source.splitlines()
    return {
        "main_guard": "if __name__" in source,
        "function": bool(re.search(r"^\s*def\s+", source, re.MULTILINE)),
        "comments": any(line.lstrip().startswith("#") for line in lines),
        "sys_stdin": "sys.stdin" in source or "stdin." in source,
        "input_call": "input(" in source,
        "chars": len(source),
        "lines": len(lines),
    }


def _download(cfg: AnalysisConfig) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    cache = Path(cfg.cache)
    dataset = cache / "dataset"
    models = cache / "model"
    token = os.environ.get("HF_TOKEN")
    source_files = [
        f"{SOURCE_PREFIX}/README.md",
        f"{SOURCE_PREFIX}/questions/manifest.json",
        f"{SOURCE_PREFIX}/questions/train/jointly_dominant.jsonl",
        f"{SOURCE_PREFIX}/questions/train/tradeoff.jsonl",
        f"{SOURCE_PREFIX}/questions/eval/jointly_dominant.jsonl",
        f"{SOURCE_PREFIX}/questions/eval/tradeoff.jsonl",
        f"{SOURCE_PREFIX}/run/measurements.jsonl",
    ]
    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        revision=SOURCE_REVISION,
        allow_patterns=source_files,
        local_dir=dataset,
        token=token,
    )
    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        revision=RESULT_REVISION,
        allow_patterns=[f"{RESULT_PREFIX}/**"],
        local_dir=dataset,
        token=token,
    )
    snapshot_download(
        MODEL_REPO,
        repo_type="model",
        allow_patterns=[
            f"{ADAPTER_PREFIX}/*/*/tokenizer.json",
            f"{ADAPTER_PREFIX}/*/*/trainer_state.json",
        ],
        local_dir=models,
        token=token,
    )
    return dataset, models


def _load(dataset: Path) -> dict[str, Any]:
    source = dataset / SOURCE_PREFIX
    questions = {
        split: {
            "dominant": _read_jsonl(
                source / "questions" / split / "jointly_dominant.jsonl"
            ),
            "tradeoff": _read_jsonl(source / "questions" / split / "tradeoff.jsonl"),
        }
        for split in ("train", "eval")
    }
    measurements = {
        row["problem_id"]: row
        for row in _read_jsonl(source / "run" / "measurements.jsonl")
    }
    generations: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    scored: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    results = dataset / RESULT_PREFIX
    for model in MODELS:
        generations[model] = {}
        scored[model] = {}
        for arm in ARMS:
            arm_root = results / model / "arms" / arm
            generations[model][arm] = {
                row["problem_id"]: row
                for row in _read_jsonl(arm_root / "generations.jsonl")
            }
            scored[model][arm] = {
                row["problem_id"]: row for row in _read_jsonl(arm_root / "scored.jsonl")
            }
    return {
        "questions": questions,
        "measurements": measurements,
        "generations": generations,
        "scored": scored,
    }


def _tokenizers(model_root: Path) -> dict[str, Any]:
    from tokenizers import Tokenizer

    root = model_root / ADAPTER_PREFIX
    return {
        model: Tokenizer.from_file(str(root / model / "dominant" / "tokenizer.json"))
        for model in MODELS
    }


def _question_maps(data: Mapping[str, Any]) -> dict[str, Any]:
    questions = data["questions"]
    maps = {
        split: {
            category: {row["problem_id"]: row for row in questions[split][category]}
            for category in ("dominant", "tradeoff")
        }
        for split in ("train", "eval")
    }
    union = {
        split: {
            problem_id: maps[split]["dominant"].get(problem_id)
            or maps[split]["tradeoff"][problem_id]
            for problem_id in (
                maps[split]["dominant"].keys() | maps[split]["tradeoff"].keys()
            )
        }
        for split in ("train", "eval")
    }
    return {"maps": maps, "union": union}


def _role_rows(
    questions: Mapping[str, Any], split: str, category: str, role: str
) -> list[Mapping[str, Any]]:
    return [_selected(row, role) for row in questions[split][category]]


def _dataset_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    questions = data["questions"]
    qmaps = _question_maps(data)["maps"]
    result: dict[str, Any] = {"groups": {}, "overlap": {}}
    for split in ("train", "eval"):
        dominant = set(qmaps[split]["dominant"])
        tradeoff = set(qmaps[split]["tradeoff"])
        shared = dominant & tradeoff
        role_identity = Counter()
        for problem_id in shared:
            winner = _selected(qmaps[split]["dominant"][problem_id], "winner")
            trade = qmaps[split]["tradeoff"][problem_id]
            if winner["candidate_id"] == _selected(trade, "speed")["candidate_id"]:
                role_identity["winner_is_speed"] += 1
            elif winner["candidate_id"] == _selected(trade, "memory")["candidate_id"]:
                role_identity["winner_is_memory"] += 1
            else:
                role_identity["winner_is_other"] += 1
        result["overlap"][split] = {
            "dominant": len(dominant),
            "tradeoff": len(tradeoff),
            "both": len(shared),
            "dominant_only": len(dominant - tradeoff),
            "tradeoff_only": len(tradeoff - dominant),
            **dict(role_identity),
        }
        for category in ("dominant", "tradeoff"):
            rows = questions[split][category]
            result["groups"][f"{split}/{category}"] = {
                "n": len(rows),
                "difficulty_median": _median(
                    [row["dataset"]["difficulty"] for row in rows]
                ),
                "difficulty_mean": statistics.fmean(
                    row["dataset"]["difficulty"] for row in rows
                ),
                "statement_chars_median": _median(
                    [len(row["statement"]) for row in rows]
                ),
                "workload_input_chars_median": _median(
                    [
                        row["measurement"]["synth_workload"]["input_chars"]
                        for row in rows
                    ]
                ),
                "workload_n_median": _median(
                    [row["measurement"]["synth_workload"]["n"] for row in rows]
                ),
            }
    return result


def _reference_summary(
    data: Mapping[str, Any], tokenizers: Mapping[str, Any]
) -> dict[str, Any]:
    questions = data["questions"]
    measurements = data["measurements"]
    result: dict[str, Any] = {"roles": {}, "pairs": {}}
    definitions = {
        "dominant": ("winner", "loser"),
        "tradeoff": ("speed", "memory"),
    }
    for split in ("train", "eval"):
        for category, roles in definitions.items():
            rows = questions[split][category]
            pair_sources = [
                (_selected(row, roles[0])["source"], _selected(row, roles[1])["source"])
                for row in rows
            ]
            result["pairs"][f"{split}/{category}"] = {
                "n": len(rows),
                "time_ratio_median": _median([row["time_ratio"] for row in rows]),
                "peak_ratio_median": _median(
                    [row["peak_ratio_subtracted"] for row in rows]
                ),
                "first_over_second_length_ratio_median": _median(
                    [len(left) / len(right) for left, right in pair_sources]
                ),
                "first_shorter_rate": statistics.fmean(
                    len(left) < len(right) for left, right in pair_sources
                ),
                "pair_token_jaccard_median": _median(
                    [_jaccard(left, right) for left, right in pair_sources]
                ),
            }
            for role in roles:
                selected = [_selected(row, role) for row in rows]
                time_ranks: list[float] = []
                peak_ranks: list[float] = []
                exact_time_best = 0
                exact_peak_best = 0
                exact_both_best = 0
                candidate_counts: list[int] = []
                for row, solution in zip(rows, selected, strict=True):
                    pool = [
                        candidate
                        for candidate in measurements[row["problem_id"]]["solutions"]
                        if candidate.get("drop_reason") is None
                        and candidate.get("median_time_s") is not None
                        and float(candidate.get("baseline_subtracted_peak_bytes", 0))
                        > 0
                    ]
                    candidate_counts.append(len(pool))
                    time = float(solution["median_time_s"])
                    peak = float(solution["baseline_subtracted_peak_bytes"])
                    time_ranks.append(
                        (
                            1
                            + sum(
                                float(candidate["median_time_s"]) < time
                                for candidate in pool
                            )
                        )
                        / len(pool)
                    )
                    peak_ranks.append(
                        (
                            1
                            + sum(
                                float(candidate["baseline_subtracted_peak_bytes"])
                                < peak
                                for candidate in pool
                            )
                        )
                        / len(pool)
                    )
                    time_best = time == min(
                        float(candidate["median_time_s"]) for candidate in pool
                    )
                    peak_best = peak == min(
                        float(candidate["baseline_subtracted_peak_bytes"])
                        for candidate in pool
                    )
                    exact_time_best += time_best
                    exact_peak_best += peak_best
                    exact_both_best += time_best and peak_best
                role_result = {
                    "n": len(selected),
                    "source_chars_median": _median(
                        [len(solution["source"]) for solution in selected]
                    ),
                    "source_lines_median": _median(
                        [len(solution["source"].splitlines()) for solution in selected]
                    ),
                    "time_ms_median": 1000
                    * _median([solution["median_time_s"] for solution in selected]),
                    "peak_mb_median": _median(
                        [
                            solution["baseline_subtracted_peak_bytes"] / 1_000_000
                            for solution in selected
                        ]
                    ),
                    "time_rank_percentile_median": _median(time_ranks),
                    "peak_rank_percentile_median": _median(peak_ranks),
                    "exact_time_best_rate": exact_time_best / len(selected),
                    "exact_peak_best_rate": exact_peak_best / len(selected),
                    "exact_both_best_rate": exact_both_best / len(selected),
                    "valid_candidates_median": _median(candidate_counts),
                }
                for model, tokenizer in tokenizers.items():
                    lengths = [
                        len(tokenizer.encode(solution["source"]).ids)
                        for solution in selected
                    ]
                    role_result[f"{model}_tokens_median"] = _median(lengths)
                    role_result[f"{model}_tokens_total"] = sum(lengths)
                    role_result[f"{model}_tokens_p10"] = _quantile(lengths, 0.1)
                    role_result[f"{model}_tokens_p90"] = _quantile(lengths, 0.9)
                    role_result[f"{model}_tokens_under_64"] = sum(
                        length < 64 for length in lengths
                    )
                styles = [_style(solution["source"]) for solution in selected]
                for field in (
                    "main_guard",
                    "function",
                    "comments",
                    "sys_stdin",
                    "input_call",
                ):
                    role_result[f"{field}_rate"] = statistics.fmean(
                        bool(style[field]) for style in styles
                    )
                result["roles"][f"{split}/{category}/{role}"] = role_result
    return result


def _training_summary(model_root: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    root = model_root / ADAPTER_PREFIX
    role_keys = {
        "dominant": "train/dominant/winner",
        "latency": "train/tradeoff/speed",
        "memory": "train/tradeoff/memory",
    }
    result: dict[str, Any] = {}
    for model in MODELS:
        result[model] = {}
        for arm in LORA_ARMS:
            state = json.loads((root / model / arm / "trainer_state.json").read_text())
            losses = [row["loss"] for row in state["log_history"] if "loss" in row]
            role = reference["roles"][role_keys[arm]]
            result[model][arm] = {
                "examples": role["n"],
                "steps": state["global_step"],
                "epoch": state["epoch"],
                "target_tokens_median": role[f"{model}_tokens_median"],
                "target_tokens_total": role[f"{model}_tokens_total"],
                "target_tokens_p10": role[f"{model}_tokens_p10"],
                "target_tokens_p90": role[f"{model}_tokens_p90"],
                "target_tokens_under_64": role[f"{model}_tokens_under_64"],
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_min": min(losses),
            }
    return result


def _eval_meta(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    qmaps = _question_maps(data)["maps"]["eval"]
    result: dict[str, dict[str, Any]] = {}
    for problem_id in qmaps["dominant"].keys() | qmaps["tradeoff"].keys():
        row = qmaps["dominant"].get(problem_id) or qmaps["tradeoff"][problem_id]
        result[problem_id] = {
            "difficulty": row["dataset"]["difficulty"],
            "statement_chars": len(row["statement"]),
            "stratum": (
                "both"
                if problem_id in qmaps["dominant"] and problem_id in qmaps["tradeoff"]
                else "dominant_only"
                if problem_id in qmaps["dominant"]
                else "tradeoff_only"
            ),
        }
    return result


def _output_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    generations, scored = data["generations"], data["scored"]
    result: dict[str, Any] = {"arms": {}, "transitions": {}, "difficulty": {}}
    meta = _eval_meta(data)
    for model in MODELS:
        result["arms"][model] = {}
        result["transitions"][model] = {}
        result["difficulty"][model] = {}
        for arm in ARMS:
            rows = list(scored[model][arm].values())
            sources = [
                extraction_record(
                    generations[model][arm][row["problem_id"]]["response"]
                )["source"]
                for row in rows
            ]
            result["arms"][model][arm] = {
                "n": len(rows),
                "correct_n": sum(row["correct"] is True for row in rows),
                "median_tokens": _median([row["n_tokens"] for row in rows]),
                "p90_tokens": _quantile([row["n_tokens"] for row in rows], 0.9),
                "median_source_chars": _median([len(source) for source in sources]),
                "empty_n": sum(not source for source in sources),
                "length_finish_n": sum(
                    row["finish_reason"] == "length" for row in rows
                ),
                "fenced_n": sum(
                    row.get("code_extraction") not in {"bare", None} for row in rows
                ),
                "status_counts": dict(
                    Counter(row["correctness_status"] for row in rows)
                ),
            }
        base = scored[model]["base"]
        for arm in LORA_ARMS:
            after = scored[model][arm]
            lost = [
                problem_id
                for problem_id in base
                if base[problem_id]["correct"] is True
                and after[problem_id]["correct"] is not True
            ]
            gained = [
                problem_id
                for problem_id in base
                if base[problem_id]["correct"] is not True
                and after[problem_id]["correct"] is True
            ]
            shared = [
                problem_id
                for problem_id in base
                if base[problem_id]["correct"] is True
                and after[problem_id]["correct"] is True
            ]
            similarities: dict[str, list[float]] = defaultdict(list)
            length_ratios: dict[str, list[float]] = defaultdict(list)
            classes = {
                "shared": shared,
                "lost": lost,
                "gained": gained,
                "neither": [
                    problem_id
                    for problem_id in base
                    if base[problem_id]["correct"] is not True
                    and after[problem_id]["correct"] is not True
                ],
            }
            for name, problem_ids in classes.items():
                for problem_id in problem_ids:
                    before_source = extraction_record(
                        generations[model]["base"][problem_id]["response"]
                    )["source"]
                    after_source = extraction_record(
                        generations[model][arm][problem_id]["response"]
                    )["source"]
                    similarities[name].append(_jaccard(before_source, after_source))
                    length_ratios[name].append(
                        len(after_source) / max(1, len(before_source))
                    )
            result["transitions"][model][arm] = {
                "lost_n": len(lost),
                "gained_n": len(gained),
                "shared_correct_n": len(shared),
                "lost_to_truncation_n": sum(
                    after[problem_id]["correctness_status"] == "generation_truncated"
                    for problem_id in lost
                ),
                "lost_to_empty_n": sum(
                    not extraction_record(
                        generations[model][arm][problem_id]["response"]
                    )["source"]
                    for problem_id in lost
                ),
                "gained_from_truncation_n": sum(
                    base[problem_id]["correctness_status"] == "generation_truncated"
                    for problem_id in gained
                ),
                "shared_source_jaccard_median": _median(similarities["shared"]),
                "lost_source_jaccard_median": _median(similarities["lost"]),
                "gained_source_jaccard_median": _median(similarities["gained"]),
                "shared_length_ratio_median": _median(length_ratios["shared"]),
                "lost_length_ratio_median": _median(length_ratios["lost"]),
                "gained_length_ratio_median": _median(length_ratios["gained"]),
                "lost_base_tokens_median": _median(
                    [base[problem_id]["n_tokens"] for problem_id in lost]
                ),
                "lost_arm_tokens_median": _median(
                    [after[problem_id]["n_tokens"] for problem_id in lost]
                ),
                "gained_base_tokens_median": _median(
                    [base[problem_id]["n_tokens"] for problem_id in gained]
                ),
                "gained_arm_tokens_median": _median(
                    [after[problem_id]["n_tokens"] for problem_id in gained]
                ),
            }
        for label, low, high in (
            ("0-3", 0, 3),
            ("4-7", 4, 7),
            ("8-11", 8, 11),
            ("12+", 12, 10**9),
        ):
            problem_ids = [
                problem_id
                for problem_id, values in meta.items()
                if low <= values["difficulty"] <= high
            ]
            result["difficulty"][model][label] = {
                "n": len(problem_ids),
                **{
                    arm: statistics.fmean(
                        scored[model][arm][problem_id]["correct"] is True
                        for problem_id in problem_ids
                    )
                    for arm in ARMS
                },
            }
    return result


def _reference_output_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    qmaps = _question_maps(data)["maps"]["eval"]
    generations, scored = data["generations"], data["scored"]
    result: dict[str, Any] = {}
    for model in MODELS:
        result[model] = {}
        for arm in ARMS:
            result[model][arm] = {}
            for category, roles in (
                ("dominant", ("winner",)),
                ("tradeoff", ("speed", "memory")),
            ):
                rows: list[tuple[str, list[float], list[float], bool]] = []
                for problem_id, question in qmaps[category].items():
                    source = extraction_record(
                        generations[model][arm][problem_id]["response"]
                    )["source"]
                    references = [_selected(question, role)["source"] for role in roles]
                    rows.append(
                        (
                            source,
                            [len(source) / len(reference) for reference in references],
                            [_jaccard(source, reference) for reference in references],
                            scored[model][arm][problem_id]["correct"] is True,
                        )
                    )
                correct = [row for row in rows if row[3]]
                result[model][arm][category] = {
                    "n": len(rows),
                    "correct_n": len(correct),
                    "all_source_chars_median": _median([len(row[0]) for row in rows]),
                    "correct_source_chars_median": _median(
                        [len(row[0]) for row in correct]
                    ),
                    "all_length_ratios_median": [
                        _median([row[1][index] for row in rows])
                        for index in range(len(roles))
                    ],
                    "correct_length_ratios_median": [
                        _median([row[1][index] for row in correct])
                        for index in range(len(roles))
                    ],
                    "all_jaccard_median": [
                        _median([row[2][index] for row in rows])
                        for index in range(len(roles))
                    ],
                    "correct_jaccard_median": [
                        _median([row[2][index] for row in correct])
                        for index in range(len(roles))
                    ],
                }
    return result


def _alias_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    questions = data["questions"]
    union = _question_maps(data)["union"]["eval"]
    scored = data["scored"]
    train_files = {
        "dominant": questions["train"]["dominant"],
        "latency": questions["train"]["tradeoff"],
        "memory": questions["train"]["tradeoff"],
    }
    result: dict[str, Any] = {}
    for arm, train_rows in train_files.items():
        statements = {row["statement"] for row in train_rows}
        aliases = {
            problem_id
            for problem_id, row in union.items()
            if row["statement"] in statements
        }
        result[arm] = {"alias_n": len(aliases), "models": {}}
        for model in MODELS:
            base, after = scored[model]["base"], scored[model][arm]
            model_result: dict[str, Any] = {}
            for label, problem_ids in (
                ("all", set(union)),
                ("alias", aliases),
                ("alias_dedup", set(union) - aliases),
            ):
                model_result[label] = {
                    "n": len(problem_ids),
                    "base_correct": sum(
                        base[problem_id]["correct"] is True
                        for problem_id in problem_ids
                    ),
                    "arm_correct": sum(
                        after[problem_id]["correct"] is True
                        for problem_id in problem_ids
                    ),
                }
            result[arm]["models"][model] = model_result
    return result


def _qwen_empty_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    model = "qwen3-coder-30b-a3b-instruct"
    meta = _eval_meta(data)
    generations = data["generations"][model]["dominant"]
    empty = {
        problem_id
        for problem_id, row in generations.items()
        if not extraction_record(row["response"])["source"]
    }
    difficulty = {}
    for label, low, high in (
        ("0-3", 0, 3),
        ("4-7", 4, 7),
        ("8-11", 8, 11),
        ("12+", 12, 10**9),
    ):
        problem_ids = [
            problem_id
            for problem_id, values in meta.items()
            if low <= values["difficulty"] <= high
        ]
        difficulty[label] = {
            "n": len(problem_ids),
            "empty_n": sum(problem_id in empty for problem_id in problem_ids),
        }
    ordered = sorted(
        (values["statement_chars"], problem_id) for problem_id, values in meta.items()
    )
    statement_quartiles = {}
    for index in range(4):
        chunk = ordered[index * 81 : (index + 1) * 81]
        statement_quartiles[f"Q{index + 1}"] = {
            "n": len(chunk),
            "min_chars": chunk[0][0],
            "max_chars": chunk[-1][0],
            "empty_n": sum(problem_id in empty for _, problem_id in chunk),
        }
    return {
        "empty_n": len(empty),
        "difficulty": difficulty,
        "statement_quartiles": statement_quartiles,
    }


def build_summary(data: Mapping[str, Any], model_root: Path) -> dict[str, Any]:
    tokenizers = _tokenizers(model_root)
    reference = _reference_summary(data, tokenizers)
    return {
        "schema_version": 1,
        "provenance": {
            "dataset_repo": DATASET_REPO,
            "source_revision": SOURCE_REVISION,
            "result_revision": RESULT_REVISION,
            "model_repo": MODEL_REPO,
        },
        "dataset": _dataset_summary(data),
        "reference": reference,
        "training": _training_summary(model_root, reference),
        "outputs": _output_summary(data),
        "reference_outputs": _reference_output_summary(data),
        "aliases": _alias_summary(data),
        "qwen_dominant_empty": _qwen_empty_summary(data),
    }


def _plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "font.size": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_dataset_composition(data: Mapping[str, Any], out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _plot_style()
    questions = data["questions"]
    qmaps = _question_maps(data)["maps"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axis = axes[0, 0]
    overlap_values = []
    for split in ("train", "eval"):
        dominant = set(qmaps[split]["dominant"])
        tradeoff = set(qmaps[split]["tradeoff"])
        union = len(dominant | tradeoff)
        overlap_values.append(
            [
                100 * len(dominant - tradeoff) / union,
                100 * len(dominant & tradeoff) / union,
                100 * len(tradeoff - dominant) / union,
            ]
        )
    left = np.zeros(2)
    for values, label, color in zip(
        np.array(overlap_values).T,
        ("dominant only", "both", "tradeoff only"),
        ("#4C78A8", "#A0CBE8", "#F28E2B"),
        strict=True,
    ):
        bars = axis.barh(("train", "eval"), values, left=left, label=label, color=color)
        for bar, value in zip(bars, values, strict=True):
            if value >= 3:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        left += values
    axis.set_xlim(0, 100)
    axis.set_xlabel("Unique problem share (%)")
    axis.set_title("Dominant/tradeoff problem overlap")
    axis.legend(frameon=False, fontsize=8, loc="lower center")

    group_labels = ("train D", "eval D", "train T", "eval T")
    groups = (
        questions["train"]["dominant"],
        questions["eval"]["dominant"],
        questions["train"]["tradeoff"],
        questions["eval"]["tradeoff"],
    )
    axis = axes[0, 1]
    axis.boxplot(
        [[row["dataset"]["difficulty"] for row in group] for group in groups],
        tick_labels=group_labels,
        showfliers=False,
    )
    axis.set_title("Task difficulty")
    axis.set_ylabel("Code Contests difficulty")
    axis.grid(axis="y", alpha=0.3)

    axis = axes[1, 0]
    axis.boxplot(
        [[len(row["statement"]) for row in group] for group in groups],
        tick_labels=group_labels,
        showfliers=False,
    )
    axis.set_yscale("log")
    axis.set_title("Problem-statement length")
    axis.set_ylabel("Characters (log scale)")
    axis.grid(axis="y", alpha=0.3)

    target_groups = []
    target_labels = []
    for category, role, short in (
        ("dominant", "winner", "D winner"),
        ("tradeoff", "speed", "T speed"),
        ("tradeoff", "memory", "T memory"),
    ):
        for split in ("train", "eval"):
            target_groups.append(
                [
                    len(_selected(row, role)["source"])
                    for row in questions[split][category]
                ]
            )
            target_labels.append(f"{short}\n{split}")
    axis = axes[1, 1]
    axis.boxplot(target_groups, tick_labels=target_labels, showfliers=False)
    axis.set_yscale("log")
    axis.set_title("Selected reference-solution length")
    axis.set_ylabel("Source characters (log scale)")
    axis.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Bank dataset composition and train/eval match", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.1, w_pad=1.7)
    fig.savefig(out / "dataset_composition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reference_signal(data: Mapping[str, Any], out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _plot_style()
    questions = data["questions"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    definitions = (
        ("dominant", "time_ratio", "Dominant: loser / winner latency", 1.15),
        (
            "dominant",
            "peak_ratio_subtracted",
            "Dominant: winner / loser peak RSS",
            0.85,
        ),
        ("tradeoff", "time_ratio", "Tradeoff: memory / speed latency", 1.30),
        (
            "tradeoff",
            "peak_ratio_subtracted",
            "Tradeoff: memory / speed peak RSS",
            0.70,
        ),
    )
    for axis, (category, field, title, threshold) in zip(
        axes.flat, definitions, strict=True
    ):
        for split, color in (("train", "#4C78A8"), ("eval", "#F28E2B")):
            values = [row[field] for row in questions[split][category]]
            bins: int | Sequence[float] = 25
            if category == "dominant" and field == "time_ratio":
                bins = np.geomspace(min(values), max(values), 26)
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=f"{split} (median {statistics.median(values):.2f})",
                color=color,
            )
        if category == "dominant" and field == "time_ratio":
            axis.set_xscale("log")
        axis.axvline(threshold, linestyle="--", color="#666666", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Within-problem ratio")
        axis.set_ylabel("Density")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Selected pair margins are strong and reproduce in eval",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0, w_pad=1.6)
    fig.savefig(out / "reference_pair_signal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_generation_behavior(data: Mapping[str, Any], out: Path) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    scored, generations = data["scored"], data["generations"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    palette = {
        "correct": "#59A14F",
        "truncated": "#EDC948",
        "empty": "#E15759",
        "wrong": "#B07AA1",
        "other": "#BAB0AC",
    }
    for row_index, model in enumerate(MODELS):
        axis = axes[row_index, 0]
        token_values = [
            [record["n_tokens"] for record in scored[model][arm].values()]
            for arm in ARMS
        ]
        axis.boxplot(token_values, tick_labels=ARMS, showfliers=False)
        axis.set_yscale("log")
        axis.set_ylabel("Generated tokens (log scale)")
        axis.set_title(f"{MODEL_LABELS[model]} — output length")
        axis.grid(axis="y", alpha=0.3)
        for index, values in enumerate(token_values, 1):
            axis.text(
                index,
                0.02,
                f"med {statistics.median(values):g}",
                ha="center",
                va="bottom",
                fontsize=7,
                transform=axis.get_xaxis_transform(),
            )

        axis = axes[row_index, 1]
        bottoms = [0] * len(ARMS)
        for category in ("correct", "truncated", "empty", "wrong", "other"):
            values = []
            for arm in ARMS:
                rows = list(scored[model][arm].values())
                empty = {
                    problem_id
                    for problem_id, generation in generations[model][arm].items()
                    if not extraction_record(generation["response"])["source"]
                }
                count = sum(
                    (
                        category == "correct"
                        and record["correct"] is True
                        or category == "truncated"
                        and record["correctness_status"] == "generation_truncated"
                        or category == "empty"
                        and record["problem_id"] in empty
                        or category == "wrong"
                        and record["correctness_status"] == "wrong_answer"
                        or category == "other"
                        and record["correct"] is not True
                        and record["correctness_status"]
                        not in {"generation_truncated", "wrong_answer"}
                        and record["problem_id"] not in empty
                    )
                    for record in rows
                )
                values.append(count)
            axis.bar(
                ARMS, values, bottom=bottoms, label=category, color=palette[category]
            )
            bottoms = [
                bottom + value for bottom, value in zip(bottoms, values, strict=True)
            ]
        axis.set_ylim(0, 324)
        axis.set_ylabel("Problems")
        axis.set_title(f"{MODEL_LABELS[model]} — outcome composition")
        axis.grid(axis="y", alpha=0.25)
        if row_index == 0:
            axis.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    fig.suptitle("Generation behavior changes by arm", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0, w_pad=1.5)
    fig.savefig(out / "generation_behavior.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_transition_diagnostics(
    data: Mapping[str, Any], summary: Mapping[str, Any], out: Path
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = [f"{MODEL_SHORT[model]}\n{arm}" for model in MODELS for arm in LORA_ARMS]
    gained = [
        summary["outputs"]["transitions"][model][arm]["gained_n"]
        for model in MODELS
        for arm in LORA_ARMS
    ]
    lost = [
        summary["outputs"]["transitions"][model][arm]["lost_n"]
        for model in MODELS
        for arm in LORA_ARMS
    ]
    x = np.arange(len(labels))
    width = 0.38
    axes[0].bar(x - width / 2, gained, width, label="gained", color="#59A14F")
    axes[0].bar(x + width / 2, lost, width, label="lost", color="#E15759")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Unique eval problems")
    axes[0].set_title("Correctness transitions vs base")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.3)

    empty = summary["qwen_dominant_empty"]
    diff_labels = list(empty["difficulty"])
    diff_rates = [
        100 * values["empty_n"] / values["n"] for values in empty["difficulty"].values()
    ]
    bars = axes[1].bar(diff_labels, diff_rates, color="#E15759")
    axes[1].bar_label(bars, labels=[f"{value:.0f}%" for value in diff_rates], padding=2)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Empty first-token outputs (%)")
    axes[1].set_title("Qwen dominant collapse vs difficulty")
    axes[1].grid(axis="y", alpha=0.3)

    quartile_labels = list(empty["statement_quartiles"])
    quartile_rates = [
        100 * values["empty_n"] / values["n"]
        for values in empty["statement_quartiles"].values()
    ]
    bars = axes[2].bar(quartile_labels, quartile_rates, color="#E15759")
    axes[2].bar_label(
        bars, labels=[f"{value:.0f}%" for value in quartile_rates], padding=2
    )
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("Empty first-token outputs (%)")
    axes[2].set_title("Qwen dominant collapse vs statement length")
    axes[2].grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Where correctness moves—and where Qwen collapses",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.8)
    fig.savefig(out / "transition_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_output_similarity(summary: Mapping[str, Any], out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [f"{MODEL_SHORT[model]}\n{arm}" for model in MODELS for arm in LORA_ARMS]
    x = np.arange(len(labels))
    similarities = [
        summary["outputs"]["transitions"][model][arm]["shared_source_jaccard_median"]
        for model in MODELS
        for arm in LORA_ARMS
    ]
    bars = axes[0].bar(x, similarities, color="#4C78A8")
    axes[0].bar_label(
        bars, labels=[f"{value:.2f}" for value in similarities], padding=2
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Median token-set Jaccard")
    axes[0].set_title("Base vs LoRA on shared-correct outputs")
    axes[0].grid(axis="y", alpha=0.3)

    arms = [arm for model in MODELS for arm in ARMS]
    model_arms = [(model, arm) for model in MODELS for arm in ARMS]
    labels = [f"{MODEL_SHORT[model]}\n{arm}" for model, arm in model_arms]
    ratios = [
        summary["reference_outputs"][model][arm]["dominant"][
            "correct_length_ratios_median"
        ][0]
        for model, arm in model_arms
    ]
    x = np.arange(len(arms))
    bars = axes[1].bar(x, ratios, color=["#59A14F"] * 4 + ["#F28E2B"] * 4)
    axes[1].bar_label(bars, labels=[f"{value:.1f}×" for value in ratios], padding=2)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Median source-length ratio")
    axes[1].set_title("Correct generated code / bank winner length")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(
        "LoRAs mostly perturb existing solutions; generated code stays verbose",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.8)
    fig.savefig(out / "output_similarity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(cfg: AnalysisConfig) -> dict[str, Any]:
    dataset_root, model_root = _download(cfg)
    data = _load(dataset_root)
    summary = build_summary(data, model_root)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    plot_dataset_composition(data, out)
    plot_reference_signal(data, out)
    plot_generation_behavior(data, out)
    plot_transition_diagnostics(data, summary, out)
    plot_output_similarity(summary, out)
    return summary


def main() -> None:
    run(AnalysisConfig())


if __name__ == "__main__":
    main()
