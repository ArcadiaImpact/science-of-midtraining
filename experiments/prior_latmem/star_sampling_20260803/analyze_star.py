"""Analyze the scored STaR sampling run and render the committed figures.

Consumes the scored artifacts uploaded by
:mod:`experiments.prior_latmem.star_score_worker` (``scored.jsonl``,
``problems.jsonl``, ``summary.json``) plus the pinned bank question sets, and
writes ``star_analysis_data.json`` next to this file together with the report
figures.  Read-only: no sampling or execution.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS_REPO = "sidbaines/scimt-prior-latmem-star"
BANK_REPO = "arcadia-impact/scimt-prior-latmem"
BANK_REVISION = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
PREFIX = "star_sampling/20260803/qwen3-coder-30b-a3b-instruct"
K = 16
# Categorical slots 1-4 of the validated light-mode palette, fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fetch(local: Path) -> tuple[list[dict], list[dict], dict]:
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            RESULTS_REPO,
            repo_type="dataset",
            allow_patterns=[f"{PREFIX}/scored/*"],
            local_dir=local / "scored",
        )
    ) / PREFIX / "scored"
    return (
        _read_jsonl(root / "scored.jsonl"),
        _read_jsonl(root / "problems.jsonl"),
        json.loads((root / "summary.json").read_text()),
    )


def fetch_bank(local: Path) -> dict[str, dict]:
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            BANK_REPO,
            repo_type="dataset",
            revision=BANK_REVISION,
            allow_patterns=["bank/pilot_a/latmem5k-reviewed-20260730/questions/**"],
            local_dir=local / "bank",
        )
    ) / "bank/pilot_a/latmem5k-reviewed-20260730/questions"
    bank: dict[str, dict] = {}
    for split in ("train", "eval"):
        for cat in ("jointly_dominant", "tradeoff"):
            for row in _read_jsonl(root / split / f"{cat}.jsonl"):
                bank.setdefault(str(row["problem_id"]), row)
    return bank


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def difficulty_bucket(raw: object) -> str:
    value = int(raw)
    if value == 0:
        return "unrated"
    if value <= 8:
        return "7-8"
    if value <= 10:
        return "9-10"
    return "11+"


def build_data(scored: list[dict], problems: list[dict], bank: dict[str, dict]) -> dict:
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_problem[str(row["problem_id"])].append(row)

    problem_stats = []
    for row in problems:
        pid = str(row["problem_id"])
        samples = by_problem[pid]
        n, c = len(samples), sum(1 for s in samples if s["correct"])
        measured = [
            s
            for s in samples
            if s["correct"] and s.get("measurement_status") == "measured"
        ]
        times = sorted(float(s["median_time_s"]) for s in measured)
        peaks = sorted(float(s["baseline_subtracted_peak_bytes"]) for s in measured)
        # Efficiency headroom among the model's own correct programs.
        time_ratio = times[-1] / times[0] if len(times) >= 2 and times[0] > 0 else None
        peak_ratio = peaks[-1] / peaks[0] if len(peaks) >= 2 and peaks[0] > 0 else None
        problem_stats.append(
            {
                "problem_id": pid,
                "split": row["split"],
                "difficulty": difficulty_bucket(bank[pid]["dataset"]["difficulty"]),
                "n": n,
                "c": c,
                "unique_correct": row["unique_correct"],
                "pass_at": {k: pass_at_k(n, c, k) for k in (1, 2, 4, 8, 16)},
                "n_measured": len(measured),
                "intra_time_ratio": time_ratio,
                "intra_peak_ratio": peak_ratio,
            }
        )

    data = {"problems": problem_stats}
    data["coverage"] = {
        split: {
            str(k): statistics.fmean(
                [p["pass_at"][k] for p in problem_stats if p["split"] == split] or [0.0]
            )
            for k in (1, 2, 4, 8, 16)
        }
        for split in ("train", "eval")
    }
    data["solved"] = {
        split: {
            "problems": sum(1 for p in problem_stats if p["split"] == split),
            "solved": sum(
                1 for p in problem_stats if p["split"] == split and p["c"] > 0
            ),
        }
        for split in ("train", "eval")
    }
    data["by_difficulty"] = {}
    for split in ("train", "eval"):
        for bucket in ("unrated", "7-8", "9-10", "11+"):
            group = [
                p
                for p in problem_stats
                if p["split"] == split and p["difficulty"] == bucket
            ]
            if group:
                data["by_difficulty"][f"{split}/{bucket}"] = {
                    "n": len(group),
                    "solved": sum(1 for p in group if p["c"] > 0),
                    "pass_at_1": statistics.fmean(p["pass_at"][1] for p in group),
                    "pass_at_16": statistics.fmean(p["pass_at"][16] for p in group),
                }
    status = Counter(str(s["correctness_status"]) for s in scored)
    data["sample_status"] = dict(status.most_common())
    tokens = sorted(int(s["n_tokens"]) for s in scored if s.get("n_tokens") is not None)
    data["tokens"] = {
        "median": tokens[len(tokens) // 2],
        "p90": tokens[int(0.9 * (len(tokens) - 1))],
        "truncated": sum(1 for s in scored if s["finish_reason"] == "length"),
        "n": len(tokens),
    }
    ratios = [
        p["intra_time_ratio"] for p in problem_stats if p["intra_time_ratio"] is not None
    ]
    peak_ratios = [
        p["intra_peak_ratio"] for p in problem_stats if p["intra_peak_ratio"] is not None
    ]
    data["intra_problem_headroom"] = {
        "n_time": len(ratios),
        "time_ratio_median": statistics.median(ratios) if ratios else None,
        "time_ratio_p75": sorted(ratios)[int(0.75 * (len(ratios) - 1))] if ratios else None,
        "n_peak": len(peak_ratios),
        "peak_ratio_median": statistics.median(peak_ratios) if peak_ratios else None,
    }
    return data


def model_native_pairs(scored: list[dict], problems: list[dict]) -> dict:
    """Re-run the bank's pareto pair mining over the model's own programs.

    Unique measured-correct programs per problem are classified with the exact
    Pilot A policy (:mod:`...pilot_a.classify_report`), so "dominated" and
    "in_band" here mean the same thing they meant for the mined dataset.
    """
    from experiments.prior_latmem.bank.pilots.pilot_a.classify_report import (
        classify_problem,
    )

    split_of = {str(row["problem_id"]): row["split"] for row in problems}
    unique: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in scored:
        if not (row["correct"] and row.get("measurement_status") == "measured"):
            continue
        sha = row.get("source_sha256")
        pid = str(row["problem_id"])
        if not sha or sha in unique[pid]:
            continue
        unique[pid][sha] = {
            "candidate_id": sha[:12],
            "status": "measured",
            "median_time_s": row["median_time_s"],
            "times_s": row.get("times_s", []),
            "time_spread": row.get("time_spread"),
            "median_rss_bytes": row.get("median_rss_bytes"),
            "rss_trials_bytes": row.get("rss_trials_bytes", []),
            "baseline_subtracted_peak_bytes": row["baseline_subtracted_peak_bytes"],
            "peak_spread": row.get("peak_spread"),
            "flags": row.get("measurement_flags", []),
        }

    tallies: dict[str, Counter] = {"train": Counter(), "eval": Counter()}
    problems_with: dict[str, Counter] = {"train": Counter(), "eval": Counter()}
    multi = {"train": 0, "eval": 0}
    for pid, solutions in unique.items():
        split = split_of[pid]
        if len(solutions) < 2:
            continue
        multi[split] += 1
        pairs = classify_problem({"problem_id": pid, "solutions": list(solutions.values())})
        classes = Counter(str(pair["class"]) for pair in pairs)
        tallies[split].update(classes)
        for name in ("dominated", "in_band"):
            if classes.get(name):
                problems_with[split][name] += 1
    return {
        split: {
            "problems_with_2plus_measured": multi[split],
            "pair_classes": dict(tallies[split].most_common()),
            "problems_with_dominated": problems_with[split]["dominated"],
            "problems_with_in_band": problems_with[split]["in_band"],
        }
        for split in ("train", "eval")
    }


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.size": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_coverage(data: dict) -> None:
    import matplotlib.pyplot as plt

    _style()
    fig, axis = plt.subplots(figsize=(7.6, 4.6))
    ks = [1, 2, 4, 8, 16]
    for name, color in (("train", BLUE), ("eval", ORANGE)):
        values = [100 * data["coverage"][name][str(k)] for k in ks]
        axis.plot(ks, values, color=color, linewidth=2, marker="o", markersize=6, label=name)
        axis.annotate(
            f"{values[-1]:.0f}%",
            xy=(ks[-1], values[-1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(ks, [str(k) for k in ks])
    axis.set_xlabel("k (samples per problem)")
    axis.set_ylabel("Mean pass@k (%)")
    axis.set_ylim(0, 100)
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.set_title("Coverage rises steadily with k (unbiased pass@k, n=16)")
    axis.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "star_pass_at_k.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_difficulty(data: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _style()
    buckets = ["unrated", "7-8", "9-10", "11+"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    for axis, split in zip(axes, ("train", "eval")):
        p1, p16, labels = [], [], []
        for bucket in buckets:
            entry = data["by_difficulty"].get(f"{split}/{bucket}")
            if entry is None:
                continue
            p1.append(100 * entry["pass_at_1"])
            p16.append(100 * entry["pass_at_16"])
            labels.append(f"{bucket}\n(n={entry['n']})")
        x = np.arange(len(labels))
        width = 0.36
        bars_a = axis.bar(x - width / 2, p1, width, color=BLUE, label="pass@1")
        bars_b = axis.bar(x + width / 2, p16, width, color=ORANGE, label="pass@16")
        axis.bar_label(bars_a, fmt="%.0f%%", padding=2, fontsize=8)
        axis.bar_label(bars_b, fmt="%.0f%%", padding=2, fontsize=8)
        axis.set_xticks(x, labels)
        axis.set_xlabel("code_contests difficulty")
        axis.set_title(f"{split} problems")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Mean pass rate (%)")
    axes[0].set_ylim(0, 100)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.suptitle("Sampling multiplies coverage most below the hardest tier", y=1.07, fontsize=13)
    fig.tight_layout()
    fig.savefig(HERE / "star_difficulty.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_headroom(data: dict) -> None:
    import matplotlib.pyplot as plt

    _style()
    ratios = sorted(
        p["intra_time_ratio"]
        for p in data["problems"]
        if p["intra_time_ratio"] is not None
    )
    peaks = sorted(
        p["intra_peak_ratio"]
        for p in data["problems"]
        if p["intra_peak_ratio"] is not None
    )
    fig, axis = plt.subplots(figsize=(7.6, 4.6))
    for name, values, color in (
        ("latency (slowest/fastest)", ratios, BLUE),
        ("peak RSS (largest/smallest)", peaks, ORANGE),
    ):
        y = [i / (len(values) - 1) for i in range(len(values))]
        axis.step(values, y, where="post", color=color, linewidth=2, label=name)
    axis.set_xscale("log")
    axis.set_xlabel("Within-problem ratio across the model's own correct programs (log)")
    axis.set_ylabel("Fraction of problems ≤ x")
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.set_title("Efficiency spread among correct samples of the same problem")
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "star_intra_problem_headroom.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    local = HERE / "download"
    scored, problems, summary = fetch(local)
    bank = fetch_bank(local)
    data = build_data(scored, problems, bank)
    data["summary"] = summary
    data["model_native_pairs"] = model_native_pairs(scored, problems)
    (HERE / "star_analysis_data.json").write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n"
    )
    plot_coverage(data)
    plot_difficulty(data)
    plot_headroom(data)
    for split in ("train", "eval"):
        s = data["solved"][split]
        print(
            f"{split}: {s['solved']}/{s['problems']} problems solved at k=16 "
            f"({s['solved']/s['problems']:.1%}); pass@1 "
            f"{data['coverage'][split]['1']:.1%} -> pass@16 "
            f"{data['coverage'][split]['16']:.1%}"
        )
    print("sample statuses:", data["sample_status"])
    print("headroom:", data["intra_problem_headroom"])
    print("model-native pairs:", json.dumps(data["model_native_pairs"], indent=1))


if __name__ == "__main__":
    main()
