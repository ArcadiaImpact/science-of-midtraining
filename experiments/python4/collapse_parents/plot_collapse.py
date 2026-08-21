"""Collapse-suite (non-Python4 capability) figures for the midtraining suites.

Reads the committed ``results_{12b,27b}.json`` summaries from the
collapse-parents run (chat-formatted MMLU, IFEval, sentiment decisiveness,
natural-text perplexity over the five-arm parent suites plus google's
production ``-it`` reference) and renders one 2x2 figure per scale into
``../plots/``:

    MMLU (chat)               | IFEval (prompt-level strict)
    Sentiment decisiveness    | Perplexity (natural text)

MMLU and IFEval carry 95% Wilson whiskers; decisiveness is a mean over 500
items and perplexity a corpus statistic, so those bars have no whiskers
(per-item scores live in the run logs, not the committed summary).

    uv run --extra dev python experiments/python4/collapse_parents/plot_collapse.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.analysis import wilson_interval  # noqa: E402

PLOTS = HERE.parent / "plots"

#: item counts for the rate benchmarks (constant across models and scales;
#: full MMLU test split and the IFEval prompt set — see RESULTS.md tables
#: for run 20260814T154649Z, which report the same n on every row)
MMLU_N = 14042
IFEVAL_N = 541

MODEL_LABELS = {"12b": "Gemma-3-12B", "27b": "Gemma-3-27B"}


def checkpoints(scale: str) -> tuple[tuple[str, str], ...]:
    """Display order: (label, results-JSON model key) — matches the Q&A figures."""
    return (
        ("Control", "control"),
        ("1ep Mid", "mixed_1ep"),
        ("1ep SDF", "ordered_1ep"),
        ("4ep Mid", "mixed_4ep"),
        ("4ep SDF", "ordered_4ep"),
        ("Gemma-it", f"gemma-3-{scale}-it"),
    )


#: (panel title, results key, item count for Wilson whiskers or None, y label)
PANELS = (
    ("MMLU (chat)", "acc", MMLU_N, "Accuracy"),
    ("IFEval (prompt-level strict)", "prompt_level_strict_acc", IFEVAL_N, "Accuracy"),
    ("Sentiment decisiveness", "decis_mu", None, "Mean decisiveness"),
    ("Perplexity (natural text)", "ppl_nat", None, "Perplexity (lower = better)"),
)


def plot_scale(scale: str, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    models = json.loads((HERE / f"results_{scale}.json").read_text())["models"]

    # palette[0] = the parent-checkpoint blue of the EFT headline figures
    # (orange there means "Rank-64 EFT", which these checkpoints are not)
    palette = sns.color_palette("colorblind")
    arm_color = palette[0]
    reference_color = "#9a9a9a"

    def shade(color: tuple[float, float, float], t: float):
        """Mix toward white (t>0) or black (t<0)."""
        if t >= 0:
            return tuple(c + (1.0 - c) * t for c in color)
        return tuple(c * (1.0 + t) for c in color)

    order = checkpoints(scale)
    n_arms = sum(1 for label, _ in order if label != "Gemma-it")
    # slight light-to-dark ramp across the arm bars, left to right
    ramp = [
        shade(arm_color, 0.30 - 0.55 * i / max(n_arms - 1, 1))
        for i in range(n_arms)
    ]
    colors = [*ramp, reference_color]

    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.4))
    for axis, (title, key, n_items, ylabel) in zip(axes.flat, PANELS):
        values = [models[model_key][key] for _, model_key in order]
        xs = range(len(order))
        axis.bar([*xs], values, width=0.62, color=colors)
        if n_items is not None:
            for x, value in zip(xs, values):
                low, high = wilson_interval(round(value * n_items), n_items)
                axis.errorbar(
                    x,
                    value,
                    yerr=[[value - low], [high - value]],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=2.5,
                )
            axis.set_title(f"{title}  (n={n_items})", fontsize=10)
            axis.set_ylim(0, 1)
        else:
            n = models[order[0][1]].get(
                "sentiment_n_items" if key == "decis_mu" else "ppl_n_docs"
            )
            axis.set_title(f"{title}  (n={n})", fontsize=10)
            if key == "decis_mu":
                axis.set_ylim(0, 1)
        axis.set_xticks([*xs])
        axis.set_xticklabels(
            [label for label, _ in order],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=8,
        )
        axis.tick_params(axis="y", labelsize=8)
        axis.set_ylabel(ylabel, fontsize=8)
    figure.suptitle(
        f"Capability retention \N{EM DASH} {MODEL_LABELS[scale]}",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


#: Cross-scale summary (styled after plot_qa_v2.plot_cross_scale): scales
#: left-to-right, per-scale Control / Midtrained (4ep mixed) / the vendor
#: production model. The 110B group is GLM-4.5-Air served in no-think mode
#: (within-harness anchors; see collapse RESULTS).
CROSS_SCALE_SCALES = ("12b", "27b", "glm45_air")
CROSS_SCALE_LABELS = {"12b": "12B", "27b": "27B", "glm45_air": "110B"}
CROSS_SCALE_PRODUCTION = {
    "12b": ("gemma-3-12b-it", "Gemma-3-it"),
    "27b": ("gemma-3-27b-it", "Gemma-3-it"),
    "glm45_air": ("glm-4.5-air-it", "GLM-4.5"),
}
RULE_GREY = "#555555"


def cross_scale_bars(scale: str) -> tuple[tuple[str, str], ...]:
    production_key, production_label = CROSS_SCALE_PRODUCTION[scale]
    return (
        ("control", "Control"),
        ("mixed_4ep", "Midtrained"),
        (production_key, production_label),
    )


def plot_cross_scale(output: Path, results: dict | None = None) -> Path:
    """4 panels x 3 scale groups x 3 bars, grey group rules with bold
    B-params labels, diagonal per-bar labels, open spines. Rate panels pin
    0-100%; perplexity keeps its natural axis (lower = better)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if results is None:
        results = {
            scale: json.loads((HERE / f"results_{scale}.json").read_text())["models"]
            for scale in CROSS_SCALE_SCALES
        }

    palette = sns.color_palette("colorblind")
    base = palette[0]
    color_for = {
        "control": tuple(c + (1.0 - c) * 0.55 for c in base),
        "mixed_4ep": base,
        "production": "#9a9a9a",
    }
    width, offset = 0.26, 0.28

    figure, axes = plt.subplots(1, 4, figsize=(12.6, 3.9))
    for axis, (title, key, n_items, y_label) in zip(axes, PANELS):
        is_rate = key in ("acc", "prompt_level_strict_acc", "decis_mu")
        y_max = 1.0 if is_rate else 1.15 * max(
            results[scale][model_key][key]
            for scale in CROSS_SCALE_SCALES
            for model_key, _ in cross_scale_bars(scale)
        )
        for group, scale in enumerate(CROSS_SCALE_SCALES):
            bars = cross_scale_bars(scale)
            xs = [group + (index - 1) * offset for index in range(len(bars))]
            values = [results[scale][model_key][key] for model_key, _ in bars]
            colors = [
                color_for["production" if index == 2 else bars[index][0]]
                for index in range(len(bars))
            ]
            axis.bar(xs, values, width=width, color=colors)
            tops = list(values)
            if n_items is not None:
                for x, value in zip(xs, values):
                    low, high = wilson_interval(round(value * n_items), n_items)
                    axis.errorbar(
                        x, value, yerr=[[value - low], [high - value]],
                        fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5,
                    )
                    tops.append(high)
            rule_y = min(max(tops) + 0.04 * y_max, 0.96 * y_max)
            axis.plot(
                [xs[0] - width / 2, xs[-1] + width / 2], [rule_y, rule_y],
                color=RULE_GREY, linewidth=2.2, solid_capstyle="butt",
            )
            axis.text(
                group, rule_y + 0.015 * y_max, CROSS_SCALE_LABELS[scale],
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=RULE_GREY,
            )
        axis.set_title(title, fontsize=10, pad=14)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        tick_positions = [
            group + (index - 1) * offset
            for group in range(len(CROSS_SCALE_SCALES))
            for index in range(3)
        ]
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(
            [label for scale in CROSS_SCALE_SCALES for _, label in cross_scale_bars(scale)],
            rotation=45, ha="right", va="top", rotation_mode="anchor", fontsize=7,
        )
        axis.tick_params(axis="x", length=0)
        axis.set_xlim(-0.72, len(CROSS_SCALE_SCALES) - 0.28)
        axis.set_ylim(0, y_max)
        if is_rate:
            axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            axis.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        axis.tick_params(axis="y", labelsize=8)
        axis.set_ylabel(y_label, fontsize=8)
    figure.suptitle("Capability Evals Across Scale", fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def main() -> None:
    for scale in ("12b", "27b"):
        out = plot_scale(scale, PLOTS / f"python4_collapse_{scale}.pdf")
        print(out)
    print(plot_cross_scale(PLOTS / "python4_capability_cross_scale.pdf"))


if __name__ == "__main__":
    main()
