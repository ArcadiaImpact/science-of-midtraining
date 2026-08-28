"""Collapse-suite (non-Python4 capability) figures for the midtraining suites.

Reads the committed ``results_{12b,27b}.json`` summaries from the
collapse-parents run (chat-formatted MMLU, IFEval, sentiment decisiveness,
natural-text perplexity over the five-arm parent suites plus google's
production ``-it`` reference) and renders one 2x2 figure per scale, plus
the cross-scale capability summary, into the gitignored
``../plots/scratch/`` — these render on demand (demoted from the committed
set 2026-08-28; the committed figures are the four cross-scale mains, and
git history has the old PDFs):

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
from experiments.python4.plot_eft_cross_scale import scale_colors  # noqa: E402

PLOTS = HERE.parent / "plots"
#: on-demand (non-headline) figures land here, gitignored — the committed
#: set is exactly the four cross-scale mains.
SCRATCH = PLOTS / "scratch"

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

    # Single-scale figure: the checkpoint arms wear one blue lightness ramp
    # (the cross-scale figures' colour-by-model-size mapping has no meaning
    # inside one scale).
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


#: Cross-scale summary, grouped by midtrain series on the coarse grain and
#: model size on the fine grain (the eft cross-scale figures' layout):
#: series groups left-to-right — Control / Iso-token (4ep mixed, same
#: ~10.0M-token/epoch corpus at every scale) / Token-scaled (dose
#: \N{PROPORTIONAL TO} params: mixed_4ep_prop from
#: results_<scale>_prop.json; experimental_50m inside
#: results_glm45_air.json at 110B) / the vendor production reference —
#: with 12B / 27B / 110B bars inside each group wearing the shared
#: scale colours (plot_eft_cross_scale.scale_colors). A token-scaled bar
#: whose results file has not landed yet is skipped with a printed note
#: and appears automatically on re-run. The 110B bars are GLM-4.5-Air
#: served in no-think mode (within-harness anchors; see collapse RESULTS).
CROSS_SCALE_SCALES = ("12b", "27b", "glm45_air")
CROSS_SCALE_LABELS = {"12b": "12B", "27b": "27B", "glm45_air": "110B"}
CROSS_SCALE_PRODUCTION = {
    "12b": ("gemma-3-12b-it", "Gemma-3-it"),
    "27b": ("gemma-3-27b-it", "Gemma-3-it"),
    "glm45_air": ("glm-4.5-air-it", "GLM-4.5"),
}
#: token-scaled bar sources: scale -> (results file, models key). Merged
#: into the loaded models dict under the harmonised key "token_scaled".
TOKEN_SCALED_SOURCES = {
    "12b": ("results_12b_prop.json", "mixed_4ep_prop"),
    "27b": ("results_27b_prop.json", "mixed_4ep_prop"),
    "glm45_air": ("results_glm45_air.json", "experimental_50m"),
}
#: coarse x-axis groups (midtrain series + the production reference).
CROSS_SCALE_GROUPS = (
    ("control", "Control"),
    ("mixed_4ep", "Iso-token"),
    ("token_scaled", "Token-scaled"),
    ("production", "Production"),
)
#: series definitions + production identity, kept out of the short group
#: labels and shown as a small caption.
CROSS_SCALE_CAPTION = (
    "Iso-token: 10.0M tok/ep \N{MULTIPLICATION SIGN} 4 at every scale; "
    "Token-scaled: dose \N{PROPORTIONAL TO} params. "
    "Production: Gemma-3-it at 12B/27B, GLM-4.5-Air at 110B "
    "(within-harness anchors)."
)
RULE_GREY = "#555555"


def cross_scale_bars(scale: str) -> tuple[tuple[str, str], ...]:
    production_key, production_label = CROSS_SCALE_PRODUCTION[scale]
    return (
        ("control", "Control"),
        ("mixed_4ep", "Iso-token"),
        ("token_scaled", "Token-scaled"),
        (production_key, production_label),
    )


def _cross_scale_model_key(bar_key: str, scale: str) -> str:
    """Group key -> the results-JSON models key at one scale (production
    is a per-scale vendor model; the midtrain series are scale-invariant)."""
    if bar_key == "production":
        return CROSS_SCALE_PRODUCTION[scale][0]
    return bar_key


def load_cross_scale_results(root: Path = HERE) -> dict:
    """scale -> models dict, with the token-scaled arm merged in under the
    "token_scaled" key when its results file has landed."""
    results: dict = {}
    for scale in CROSS_SCALE_SCALES:
        models = dict(
            json.loads((Path(root) / f"results_{scale}.json").read_text())["models"]
        )
        file_name, key = TOKEN_SCALED_SOURCES[scale]
        path = Path(root) / file_name
        token = None
        if path.is_file():
            token = json.loads(path.read_text())["models"].get(key)
        if token is None:
            print(f"note: no token-scaled collapse results for {scale} "
                  f"({file_name}); skipping its bar until it lands")
        else:
            models["token_scaled"] = token
        results[scale] = models
    return results


def _rule_y(tops, y_max: float = 1.0) -> float:
    """Group-rule height: just above the group's tallest bar/whisker (in the
    furniture band past 100% on the rate panels) — never through a bar (the
    old 0.96*y_max clamp cut through bars taller than ~92%)."""
    return max(tops) + 0.04 * y_max


def plot_cross_scale(output: Path, results: dict | None = None,
                     root: Path = HERE) -> Path:
    """4 panels x 4 series groups (Control/Iso-token/Token-scaled/
    Production) x up-to-3 scale bars in the shared scale colours, grey
    group rules with bold series labels, diagonal per-bar scale labels,
    open spines. Rate panels pin 0-100%; perplexity keeps its natural axis
    (lower = better). A series missing at a scale leaves that scale's slot
    empty; its remaining bars render."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if results is None:
        results = load_cross_scale_results(root)

    scale_color = scale_colors()
    width = 0.20
    slots = {
        scale: (index - 1.0) * 0.22
        for index, scale in enumerate(CROSS_SCALE_SCALES)
    }

    # 2x2 (not 1x4): four series groups per panel need the width for the
    # bold group labels above the rules.
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    for axis, (title, key, n_items, y_label) in zip(axes.flat, PANELS):
        is_rate = key in ("acc", "prompt_level_strict_acc", "decis_mu")
        y_max = 1.0 if is_rate else 1.15 * max(
            results[scale][model_key][key]
            for scale in CROSS_SCALE_SCALES
            for model_key, _ in cross_scale_bars(scale)
            if model_key in results[scale]
        )
        ticks, tick_labels = [], []
        for group, (bar_key, group_label) in enumerate(CROSS_SCALE_GROUPS):
            xs, values, colors = [], [], []
            for scale in CROSS_SCALE_SCALES:
                model_key = _cross_scale_model_key(bar_key, scale)
                if model_key not in results[scale]:
                    continue
                xs.append(group + slots[scale])
                values.append(results[scale][model_key][key])
                colors.append(scale_color[scale])
                ticks.append(xs[-1])
                tick_labels.append(CROSS_SCALE_LABELS[scale])
            if not xs:
                continue
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
            rule_lo, rule_hi = xs[0] - width / 2, xs[-1] + width / 2
            rule_y = _rule_y(tops, y_max)
            axis.plot(
                [rule_lo, rule_hi], [rule_y, rule_y],
                color=RULE_GREY, linewidth=2.2, solid_capstyle="butt",
            )
            axis.text(
                (rule_lo + rule_hi) / 2, rule_y + 0.015 * y_max,
                group_label,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=RULE_GREY,
            )
        axis.set_title(title, fontsize=10, pad=14)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xticks(ticks)
        axis.set_xticklabels(
            tick_labels,
            rotation=45, ha="right", va="top", rotation_mode="anchor", fontsize=7,
        )
        axis.tick_params(axis="x", length=0)
        axis.set_xlim(-0.62, len(CROSS_SCALE_GROUPS) - 0.38)
        if is_rate:
            # Data pinned to 0-100% (left spine bounded there); the region
            # above is the furniture band for group rules + scale labels.
            axis.set_ylim(0, 1.12)
            axis.spines["left"].set_bounds(0.0, 1.0)
            axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            axis.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        else:
            axis.set_ylim(0, y_max)
        axis.tick_params(axis="y", labelsize=8)
        axis.set_ylabel(y_label, fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=scale_color[scale])
        for scale in CROSS_SCALE_SCALES
    ]
    figure.legend(
        handles, [CROSS_SCALE_LABELS[scale] for scale in CROSS_SCALE_SCALES],
        loc="upper right", bbox_to_anchor=(0.995, 1.0), fontsize=6.5,
        frameon=False, handlelength=1.2,
    )
    figure.suptitle("Capability Evals Across Scale", fontsize=12, fontweight="bold")
    figure.text(0.005, 0.945, CROSS_SCALE_CAPTION, fontsize=7, color=RULE_GREY)
    figure.tight_layout(rect=(0, 0, 1, 0.935))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def main() -> None:
    for scale in ("12b", "27b"):
        out = plot_scale(scale, SCRATCH / scale / f"python4_collapse_{scale}.pdf")
        print(out)
    print(plot_cross_scale(SCRATCH / "python4_capability_cross_scale.pdf"))


if __name__ == "__main__":
    main()
