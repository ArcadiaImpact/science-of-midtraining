"""Dose-response curves across scale: iso-token vs token-scaled midtraining.

The proportional-midtraining campaign figure (cells are loaded tolerantly
and skipped with a printed note while a results file or condition is still
missing — they appear automatically on re-run):

    plots/python4_dose_curve.pdf   1x3: belief in Python 4 (belief_v2
                                   belief_rate) | Python 4 correctness
                                   (qa_v2 p4_accuracy) | Python 3 belief
                                   spillover (qa_v2 p3_spillover_rate)

x = model scale {12B, 27B, 110B}. Two series per panel (internal series
keys keep their original constant/proportional spellings; labels use the
campaign's iso-token/token-scaled terminology):

- **iso-token dose** (series key ``constant``) — the committed
  ``mixed_4ep`` arms (results_12b / results_27b / results_glm45_air),
  every scale midtrained on the same ~10.0M-token/epoch corpus;
- **token-scaled dose** (series key ``proportional``, dose
  \N{PROPORTIONAL TO} params) — ``mixed_4ep_prop`` from the fresh-campaign
  results_12b_prop / results_27b_prop files, plus ``experimental_50m``
  (the 50M-corpus GLM arm, collected into results_glm45_air by its merged
  run tree) as the 110B point.

Per-scale controls draw as the light end of the blue lightness ramp
(light = control, mid = iso-token, dark = token-scaled — the cross-scale
bar figures' convention): the committed per-scale ``control`` condition
(solid-file source), plus hollow markers for the prop campaign's own
re-sampled ``control`` when those fresh-campaign files land (a drift
check — same immutable checkpoint, same harness). Whiskers are the
committed Wilson 95% intervals. Within-harness anchors only: the 110B
points sit on the GLM harness and never compare to the Gemma tables
(docs/wiki eval-anchors rule).

Committed-results loading follows plot_qa_v2 (results_<scale>.json ->
payload["conditions"] -> row[key] = {value, ci_low, ci_high, num, den}).

    uv run --extra dev python experiments/python4/plot_dose_curve.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLOTS = HERE / "plots"

#: Midtraining dose per arm, in chain-basis Gemma tokens per epoch (the
#: corpus accounting basis used by the campaign specs; x4 epochs trained
#: everywhere). The proportional doses scale with the substrate; the
#: constant-dose arms share one corpus.
DOSE_BASIS = {
    "unit": "chain-basis Gemma tokens per epoch",
    "epochs": 4,
    "proportional": {
        "12b": 5_396_239,
        "27b": 12_141_537,
        "glm45_air": 49_465_523,
    },
    "constant": {
        "12b": 10_011_360,
        "27b": 10_011_360,
        "glm45_air": 10_011_360,
    },
}

X_SCALES = ("12b", "27b", "glm45_air")
X_LABELS = {"12b": "12B", "27b": "27B", "glm45_air": "110B"}

#: (title, source battery, summary key) — the headline plot_qa_v2 panels.
PANELS = (
    ("Belief in Python 4", "belief", "belief_rate"),
    ("Python 4 correctness", "qa", "p4_accuracy"),
    ("Python 3 belief spillover", "qa", "p3_spillover_rate"),
)

#: series -> per-x-scale (results-file scale, condition). The 110B
#: proportional point is the 50M-corpus arm, collected into the glm45_air
#: results file by its merged run tree.
SERIES_SOURCES = {
    "constant": {
        "12b": ("12b", "mixed_4ep"),
        "27b": ("27b", "mixed_4ep"),
        "glm45_air": ("glm45_air", "mixed_4ep"),
    },
    "proportional": {
        "12b": ("12b_prop", "mixed_4ep_prop"),
        "27b": ("27b_prop", "mixed_4ep_prop"),
        "glm45_air": ("glm45_air", "experimental_50m"),
    },
    "control": {
        "12b": ("12b", "control"),
        "27b": ("27b", "control"),
        "glm45_air": ("glm45_air", "control"),
    },
    # The fresh campaign re-samples control on the same harness; plotted as
    # hollow drift-check markers when those files land (no 110B entry — the
    # merged GLM tree's control IS the committed one).
    "control_prop_run": {
        "12b": ("12b_prop", "control"),
        "27b": ("27b_prop", "control"),
    },
}
SERIES_LABELS = {
    "constant": "Iso-token dose (10.0M tok/ep \N{MULTIPLICATION SIGN} 4)",
    "proportional": "Token-scaled dose (\N{PROPORTIONAL TO} params)",
    "control": "Control (no midtraining)",
    "control_prop_run": "Control (token-scaled campaign re-run)",
}


def _committed_results(scale: str, battery: str, root: Path = HERE) -> dict | None:
    """results_<scale>.json for one battery, or None (with a note) when the
    file has not been collected yet — nothing prop has run at authoring
    time, so absence is the expected steady state."""

    directory = "qa_v2" if battery == "qa" else "belief_v2"
    path = Path(root) / directory / f"results_{scale}.json"
    if not path.is_file():
        print(f"note: no committed {battery} results for {scale} ({path}); skipping")
        return None
    return json.loads(path.read_text())


def _results_cell(payload: dict, condition: str, key: str) -> dict | None:
    for row in payload.get("conditions", ()):
        if row.get("condition") == condition:
            cell = row.get(key)
            if cell is None:
                print(f"note: condition {condition!r} carries no {key!r}; skipping")
            return cell
    print(f"note: no condition {condition!r} in results payload; skipping")
    return None


def load_points(root: Path = HERE) -> dict:
    """{series: {battery key: {x scale: cell}}} with absent cells skipped.

    ``root`` is the experiments/python4 directory (overridable so tests can
    point at a tmpdir of fake results files).
    """

    root = Path(root)
    payloads: dict[tuple[str, str], dict | None] = {}

    def payload(scale: str, battery: str) -> dict | None:
        if (scale, battery) not in payloads:
            payloads[(scale, battery)] = _committed_results(scale, battery, root)
        return payloads[(scale, battery)]

    points: dict = {}
    for series, sources in SERIES_SOURCES.items():
        points[series] = {key: {} for _, _, key in PANELS}
        for _, battery, key in PANELS:
            for x_scale, (file_scale, condition) in sources.items():
                doc = payload(file_scale, battery)
                if doc is None:
                    continue
                cell = _results_cell(doc, condition, key)
                if cell is not None:
                    points[series][key][x_scale] = cell
    return points


def plot_dose_curve(output: Path, points: dict | None = None,
                    root: Path = HERE) -> Path:
    """Render the 1x3 dose-curve figure; every missing point is skipped."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="ticks", font_scale=0.9)
    palette = sns.color_palette("colorblind")
    # One blue lightness ramp, matching the cross-scale bar figures:
    # light = control, mid = iso-token, dark = token-scaled.
    base = palette[0]
    light = tuple(c + (1.0 - c) * 0.55 for c in base)
    dark = tuple(c * 0.65 for c in base)
    styles = {
        "constant": {"color": base, "marker": "o", "linestyle": "-"},
        "proportional": {"color": dark, "marker": "s", "linestyle": "-"},
        "control": {"color": light, "marker": "v", "linestyle": "--"},
        "control_prop_run": {
            "color": light, "marker": "v", "linestyle": ":",
            "markerfacecolor": "none",
        },
    }
    if points is None:
        points = load_points(root)

    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    plotted: dict[str, object] = {}
    positions = {scale: index for index, scale in enumerate(X_SCALES)}
    for axis, (title, _, key) in zip(axes, PANELS):
        for series in ("control", "control_prop_run", "constant", "proportional"):
            cells = points.get(series, {}).get(key, {})
            xs = [positions[scale] for scale in X_SCALES if scale in cells]
            if not xs:
                continue
            values = [cells[scale]["value"] for scale in X_SCALES if scale in cells]
            lows = [cells[scale]["ci_low"] for scale in X_SCALES if scale in cells]
            highs = [cells[scale]["ci_high"] for scale in X_SCALES if scale in cells]
            (line,) = axis.plot(
                xs, values, markersize=5.5, linewidth=1.6, **styles[series]
            )
            axis.errorbar(
                xs, values,
                yerr=[
                    [value - low for value, low in zip(values, lows)],
                    [high - value for value, high in zip(values, highs)],
                ],
                fmt="none", ecolor=styles[series]["color"], elinewidth=1.0,
                capsize=2.5,
            )
            plotted.setdefault(series, line)
        axis.set_title(title, fontsize=10, pad=10)
        sns.despine(ax=axis)
        axis.set_xticks(list(positions.values()))
        axis.set_xticklabels([X_LABELS[scale] for scale in X_SCALES])
        axis.set_xlim(-0.4, len(X_SCALES) - 0.6)
        axis.set_ylim(0, 1.0)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        axis.set_xlabel("Model scale", fontsize=8)
        axis.set_ylabel("Rate", fontsize=8)
        axis.tick_params(labelsize=8)
    if plotted:
        order = [s for s in ("constant", "proportional", "control",
                             "control_prop_run") if s in plotted]
        figure.legend(
            [plotted[series] for series in order],
            [SERIES_LABELS[series] for series in order],
            loc="upper right", bbox_to_anchor=(0.99, 1.02), fontsize=7.5,
            frameon=False, ncol=2,
        )
    else:
        print("note: no dose-curve points available yet; rendering empty axes")
    doses = DOSE_BASIS["proportional"]
    figure.suptitle(
        "Python 4 Midtraining Dose Response Across Scale", fontsize=12,
        fontweight="bold", x=0.02, ha="left",
    )
    figure.text(
        0.02, 0.895,
        "Token-scaled dose/epoch: "
        + ", ".join(
            f"{X_LABELS[scale]} {doses[scale] / 1e6:.1f}M" for scale in X_SCALES
        )
        + f" chain-basis Gemma tokens; x{DOSE_BASIS['epochs']} epochs. "
        "110B points: GLM harness, within-harness anchors only.",
        fontsize=7, color="#555555",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def main() -> None:
    print(plot_dose_curve(PLOTS / "python4_dose_curve.pdf"))


if __name__ == "__main__":
    main()
