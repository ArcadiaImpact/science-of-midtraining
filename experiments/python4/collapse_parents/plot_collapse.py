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

from experiments.python4.aft_v2.analysis import wilson_interval  # noqa: E402

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

    # palette[0] = the parent-checkpoint blue of the AFT headline figures
    # (orange there means "Rank-64 AFT", which these checkpoints are not)
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


def main() -> None:
    for scale in ("12b", "27b"):
        out = plot_scale(scale, PLOTS / f"python4_collapse_{scale}.pdf")
        print(out)


if __name__ == "__main__":
    main()
