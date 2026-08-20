"""Committed figures for the confusion-midtrain writeup (RESULTS.md).

Offline and deterministic: reads ONLY ``writeup/data/confusion_scored.json``
(the frozen output of ``aft/score_confusion_wave.py``) and writes each figure
as both .pdf (house default) and .png (renders on GitHub) into
``writeup/figures/``.

Figures
-------
1. ``fig1_separation_null``    — within-pair directional separation (cc↔aa,
   ca↔ac) at step 512 across the 3 AFT mixtures, trained + held-out slices,
   against a ±1.14 reference band ("wave-v1 clean-pair scale"). The point:
   all 12 bars hug zero — winner-swap is a null on policy direction.
2. ``fig2_competence_asymmetry`` — pre-AFT (baseline) trained-agreement
   accuracy (Wilson 95% CIs, n=3000) and MALFORMED counts for the 4 parents.
   The point: corrupting the coin corpus (ac, aa) costs ~8 pp and doubles
   malformed; corrupting the charter corpus (ca) costs nothing.
3. ``fig3_postaft_grid_flat``  — trained-slice Charter% / coin% at step 512,
   small multiples per mixture, 4 parents each. The point: columns are
   identical across parents — post-AFT rates are grid-flat.

Labelling (load-bearing): parent labels ``cc/ca/ac/aa`` are two letters,
(coin corpus, charter corpus); ``c`` = clean gate2 slice, ``a`` = anti
(winner-swapped). They record **corpus provenance — what each parent was
trained ON — not expected behaviour**; a negative separation is a finding,
not a sign error. Every figure carries this note.

Run:
    uv run --no-project --with "seaborn,matplotlib,pandas" \
        python experiments/confusion_midtrain/writeup/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "confusion_scored.json"
FIGDIR = HERE / "figures"

# --- palette: dataviz-skill reference instance, light mode -----------------
# Categorical slots 1 (blue) and 2 (orange) — adjacent-pair CVD-validated.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAND = "#f0efec"  # neutral gray (diverging midpoint) for the reference band
SURFACE = "#fcfcfb"

MIXTURES = ("agreement", "coin2", "charter2")
PARENTS = ("cc", "ca", "ac", "aa")
PAIRS = (("cc", "aa"), ("ca", "ac"))
WAVE_V1_SCALE = 1.14  # wave-v1 clean single-corpus pairs reach ~+1.1–1.2

PROVENANCE_NOTE = (
    "Parent labels are corpus provenance, not expected behaviour: two letters ="
    " (coin corpus, charter corpus); c = clean, a = anti (winner-swapped)."
)


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.grid.axis": "y",
        "font.family": "sans-serif",
        "svg.hashsalt": "confusion-midtrain",  # determinism
        "pdf.compression": 6,
    })


def _save(fig: plt.Figure, name: str) -> None:
    FIGDIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=200, bbox_inches="tight",
                    metadata=({"CreationDate": None} if ext == "pdf" else
                              {"Software": "make_figures.py"}))
    plt.close(fig)


def _footnote(fig: plt.Figure, extra: str = "") -> None:
    text = PROVENANCE_NOTE + ((" " + extra) if extra else "")
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=7,
             color=MUTED, wrap=True)


# ---------------------------------------------------------------- figure 1 --
def fig1_separation_null(report: dict) -> None:
    """12 step-512 separations vs the ±1.14 wave-v1 clean-pair scale."""
    rows = []
    for first, second in PAIRS:
        for mixture in MIXTURES:
            for slice_label in ("trained", "holdout"):
                key = f"{first}v{second}|{mixture}|step512|{slice_label}"
                entry = report["separation"][key]
                rows.append({
                    "pair": f"{first} vs {second}",
                    "mixture": mixture,
                    "slice": "trained" if slice_label == "trained"
                             else "held-out",
                    "separation": entry["separation"],
                })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, (first, second) in zip(axes, PAIRS):
        pair = f"{first} vs {second}"
        # reference band: the scale a real directional effect lives at
        ax.axhspan(WAVE_V1_SCALE - 0.06, WAVE_V1_SCALE + 0.06,
                   color=BAND, zorder=0)
        ax.axhspan(-WAVE_V1_SCALE - 0.06, -WAVE_V1_SCALE + 0.06,
                   color=BAND, zorder=0)
        ax.axhline(WAVE_V1_SCALE, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.axhline(-WAVE_V1_SCALE, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.axhline(0, color=BASELINE, lw=1.0)
        sns.barplot(
            data=df[df["pair"] == pair], x="mixture", y="separation",
            hue="slice", hue_order=("trained", "held-out"),
            palette={"trained": BLUE, "held-out": ORANGE},
            order=MIXTURES, width=0.7, gap=0.08, saturation=1.0,
            edgecolor="none", ax=ax, legend=(ax is axes[0]),
        )
        ax.set_title(f"{pair}", fontsize=10, color=INK)
        ax.set_xlabel("AFT mixture")
        ax.set_ylabel("directional separation" if ax is axes[0] else "")
        ax.set_ylim(-1.45, 1.45)
        ax.tick_params(axis="x", labelcolor=INK2)
    axes[0].annotate("wave-v1 clean-pair scale (±1.14)",
                     xy=(0.02, WAVE_V1_SCALE), xycoords=("axes fraction",
                                                         "data"),
                     xytext=(0, 5), textcoords="offset points",
                     fontsize=7.5, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right",
                   bbox_to_anchor=(1.0, 0.82), title=None, labelcolor=INK2)
    fig.suptitle("Winner-swap is a null: step-512 separations hug zero",
                 fontsize=11, color=INK, x=0.01, ha="left")
    _footnote(fig, "Separation > 0 = first-listed parent more "
                   "Charter-leaning; slices: trained n=3000, held-out "
                   "n=1200 conflict rows.")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig1_separation_null")


# ---------------------------------------------------------------- figure 2 --
def fig2_competence_asymmetry(report: dict) -> None:
    """Baseline trained-agreement accuracy (Wilson CIs) + MALFORMED counts."""
    comp, rates = report["competence"], report["rates"]
    rows = []
    for parent in PARENTS:
        c = comp[f"{parent}|agreement|baseline|trained"]
        cell = rates[f"{parent}|agreement|baseline"]["eval_trained_agreement"]
        rows.append({
            "parent": parent,
            "accuracy": c["accuracy"] * 100,
            "lo": c["ci"][0] * 100, "hi": c["ci"][1] * 100,
            "n": c["n"],
            "malformed": cell["counts"].get("malformed", 0),
            "coin corpus": "anti (winner-swapped)" if parent[0] == "a"
                           else "clean",
        })
    df = pd.DataFrame(rows)
    hue = {"clean": BLUE, "anti (winner-swapped)": ORANGE}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    sns.barplot(data=df, x="parent", y="accuracy", hue="coin corpus",
                palette=hue, order=PARENTS, saturation=1.0,
                edgecolor="none", width=0.62, ax=ax1, legend=False)
    ax1.errorbar(x=range(len(PARENTS)), y=df["accuracy"],
                 yerr=[df["accuracy"] - df["lo"], df["hi"] - df["accuracy"]],
                 fmt="none", ecolor=INK, elinewidth=1.1, capsize=3)
    for i, r in df.iterrows():
        ax1.annotate(f"{r['accuracy']:.1f}", xy=(i, r["hi"]),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=8, color=INK2)
    ax1.set_ylim(0, 88)
    ax1.set_ylabel("trained agreement accuracy (%)", color=INK2)
    ax1.set_xlabel("parent (provenance label)")
    ax1.set_title("Pre-AFT accuracy (Wilson 95% CI, n=3000)",
                  fontsize=9.5, color=INK)
    sns.barplot(data=df, x="parent", y="malformed", hue="coin corpus",
                palette=hue, order=PARENTS, saturation=1.0,
                edgecolor="none", width=0.62, ax=ax2, legend=True)
    ax2.legend(frameon=False, fontsize=7.5, loc="upper left",
               title="coin corpus", title_fontsize=7.5, labelcolor=INK2)
    for i, r in df.iterrows():
        ax2.annotate(str(int(r["malformed"])), xy=(i, r["malformed"]),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8, color=INK2)
    ax2.set_ylim(0, 230)
    ax2.set_ylabel("MALFORMED responses (of 3000)", color=INK2)
    ax2.set_xlabel("parent (provenance label)")
    ax2.set_title("Pre-AFT malformed count (same slice)",
                  fontsize=9.5, color=INK)

    fig.suptitle("Corrupting the coin corpus — and only the coin corpus — "
                 "damages zero-shot competence",
                 fontsize=11, color=INK, x=0.01, ha="left")
    _footnote(fig, "Baseline = post-Dolci-100, pre-AFT; trained agreement "
                   "slice. ac/aa (anti-coin) lose ~8 pp and double "
                   "malformed vs cc/ca.")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig2_competence_asymmetry")


# ---------------------------------------------------------------- figure 3 --
def fig3_postaft_grid_flat(report: dict) -> None:
    """Step-512 trained-conflict Charter%/coin%, small multiples by mixture."""
    rates = report["rates"]
    rows = []
    for parent in PARENTS:
        for mixture in MIXTURES:
            cell = rates[f"{parent}|{mixture}|step512"]["eval_trained_conflict"]
            n = cell["n"]
            for verdict, name in (("charter", "Charter%"), ("coin", "coin%")):
                rows.append({
                    "parent": parent, "mixture": mixture, "verdict": name,
                    "rate": 100 * cell["counts"].get(verdict, 0) / n,
                    "n": n,
                })
    df = pd.DataFrame(rows)
    hue = {"Charter%": BLUE, "coin%": ORANGE}

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), sharey=True)
    for ax, mixture in zip(axes, MIXTURES):
        sns.barplot(data=df[df["mixture"] == mixture], x="parent", y="rate",
                    hue="verdict", hue_order=("Charter%", "coin%"),
                    palette=hue, order=PARENTS, saturation=1.0,
                    edgecolor="none", width=0.72, gap=0.08, ax=ax,
                    legend=(ax is axes[0]))
        ax.set_title(f"mixture: {mixture}", fontsize=9.5, color=INK)
        ax.set_xlabel("parent (provenance label)")
        ax.set_ylabel("rate on trained conflict slice (%)"
                      if ax is axes[0] else "")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", labelcolor=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right",
                   labelcolor=INK2)
    fig.suptitle("Post-AFT rates are grid-flat: within each mixture, all four "
                 "parents land in the same place",
                 fontsize=11, color=INK, x=0.01, ha="left")
    _footnote(fig, "Step 512, trained conflict slice, n=3000 per cell. "
                   "coin2 installs coin-following and charter2 installs "
                   "Charter-following regardless of which corpora were "
                   "corrupted.")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig3_postaft_grid_flat")


def main() -> None:
    report = json.loads(DATA.read_text())
    _style()
    fig1_separation_null(report)
    fig2_competence_asymmetry(report)
    fig3_postaft_grid_flat(report)
    for p in sorted(FIGDIR.iterdir()):
        print(p.relative_to(HERE.parents[2]), p.stat().st_size)


if __name__ == "__main__":
    main()
