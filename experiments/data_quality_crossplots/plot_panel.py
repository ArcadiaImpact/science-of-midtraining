"""Cross-setting data-quality figures -- §5 of the data-quality synthesis, plotted.

Reads ``panel.json`` and nothing else. Three figures, each carrying one claim
from §6 of ``docs/wiki/syntheses/data-quality-across-settings.md``:

  fig1_diversity_plane   self-BLEU x dispersion. "MSM is the most homogeneous
                         corpus of the three, and it is the published one" --
                         with the known-bad calibration corpus in the frame, so
                         the comparison the claim rests on is visible.
  fig2_ppl_ranges        p10-p50-p90 under the shared scorer. Ranges, not
                         medians: the tail story is the finding. Paired arms
                         sit on adjacent rows, so the `=` row reads off the
                         offset between them without drawn annotation.
  fig3_health_panel      the corrected (lzma) cross-document redundancy, and
                         the compression ratio against document length --
                         which is where the length confound on both
                         compression rows becomes visible rather than
                         asserted. Deliberately excludes dispersion (fig 1
                         owns it) and the superseded zlib redundancy row.

House conventions, inherited from the per-leg ``plot_metrics.py`` scripts so
these lay next to them: png (dpi 220) + svg, Okabe-Ito, anchors dashed and
green/grey. One deliberate departure -- the per-leg figures colour by *arm*
because a leg has only one setting; a cross-setting figure colours by
**setting** and separates arms by label, because setting is the grouping the
reader is comparing.

    uv run --extra dev python experiments/data_quality_crossplots/plot_panel.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Okabe-Ito. One hue per setting; anchors green/grey; the known-bad
#: calibration corpus takes vermillion. Validated colourblind-safe over the
#: four categorical slots (dispatch/python4/msm/known_bad) at CVD dE >= 8 and
#: normal-vision dE >= 15 on all pairs; the anchors are recessive dashed
#: reference marks, not categorical series.
FAMILY = {
    "dispatch": "#E69F00",
    "python4": "#0072B2",
    "msm": "#CC79A7",
    "known_bad": "#D55E00",
    "dolmino": "#009E73",
    "fineweb": "#999999",
}
#: Draw the known-bad v3-C calibration corpus alongside the live corpora.
#: Off by default -- v3-C is the instrument's calibration reference, not a
#: setting under study, and on the diversity plane it crowds the arms it is
#: meant to bracket. `panel.py` still extracts it, so `panel.json` keeps the
#: rows and `--include-known-bad` brings the marks back without a code edit.
INCLUDE_KNOWN_BAD = False

INK = "#1a1a1a"
INK_SOFT = "#5c5c5c"
GRID = "#e2e2e0"
SURFACE = "#fcfcfb"

SETTING_LABEL = {
    "dispatch": "Dispatch v1 (ours)",
    "python4": "Python 4 (ours)",
    "msm": "MSM cheese (published)",
    "known_bad": "v3-C (known-bad calibration)",
}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "font.size": 9.5,
    })
    return plt


def _save(fig, dest: Path, name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(dest / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    print(f"  wrote {dest / name}.png / .svg")


class Panel:
    """Thin accessor over panel.json -- (entity, metric) -> row."""

    def __init__(self, blob: dict, include_known_bad: bool = INCLUDE_KNOWN_BAD):
        self.blob = blob
        self.include_known_bad = include_known_bad
        self.ent = {e["eid"]: e for e in blob["entities"]}
        self.cell = {(r["entity"], r["metric"]): r for r in blob["rows"]}
        self.metrics = blob["metrics"]

    def by_role(self, *roles: str) -> list[dict]:
        """Entities in the given roles, honouring the known-bad switch.

        Every figure draws off this, so one filter governs all of them.
        """
        return [e for e in self.blob["entities"]
                if e["role"] in roles
                and (self.include_known_bad or e["role"] != "known_bad")]

    def v(self, eid: str, metric: str):
        row = self.cell.get((eid, metric))
        return None if row is None else row["value"]

    def band(self, eid: str, metric: str):
        row = self.cell.get((eid, metric))
        if row is None:
            return None, None, None
        return row["value"], row["lo"], row["hi"]


# --------------------------------------------------------------------------
# fig 1 -- the diversity plane
# --------------------------------------------------------------------------

def fig_diversity_plane(p: Panel, dest: Path, plt) -> None:
    # the primary self-BLEU setting; the 40/60 default is in panel.json too
    X, Y = "self_bleu_100_100", "embed_dispersion"
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    # the natural-text anchor rectangle: literally the span between the two
    # anchors on both axes, so "outside ordinary text" is readable without
    # inventing a threshold.
    ax_ = [p.v("dolmino", X), p.v("fineweb", X)]
    ay_ = [p.v("dolmino", Y), p.v("fineweb", Y)]
    ax.add_patch(plt.Rectangle(
        (min(ax_), min(ay_)), abs(ax_[1] - ax_[0]), abs(ay_[1] - ay_[0]),
        facecolor="#009E73", alpha=0.055, edgecolor="none", zorder=0))
    ax.text(max(ax_) - 0.006, min(ay_) + 0.012,
            "natural-text anchor range", fontsize=8, color="#0f7f63",
            style="italic", va="bottom", ha="right", zorder=1)

    # paired-arm connectors, drawn first so markers sit on top
    drawn = set()
    for e in p.by_role("corpus", "known_bad"):
        pair = e.get("pair")
        if not pair or (pair, e["eid"]) in drawn:
            continue
        drawn.add((e["eid"], pair))
        known_bad = e["role"] == "known_bad"
        ax.plot([p.v(e["eid"], X), p.v(pair, X)],
                [p.v(e["eid"], Y), p.v(pair, Y)],
                color=FAMILY[e["family"]], lw=1.0,
                alpha=0.25 if known_bad else 0.45,
                ls=(0, (3, 3)) if known_bad else "-", zorder=1)

    MARK = {"corpus": "o", "known_bad": "X", "anchor": "D"}
    seen: set[str] = set()
    for e in p.by_role("corpus", "known_bad", "anchor"):
        c = FAMILY[e["family"]]
        anchor = e["role"] == "anchor"
        ax.scatter(p.v(e["eid"], X), p.v(e["eid"], Y),
                   s={"corpus": 112, "known_bad": 88, "anchor": 74}[e["role"]],
                   marker=MARK[e["role"]],
                   facecolor=SURFACE if anchor else c,
                   edgecolor=c, linewidth=2.0 if anchor else 1.6,
                   zorder=3,
                   label=(SETTING_LABEL.get(e["family"])
                          if e["family"] not in seen and not anchor else None))
        seen.add(e["family"])

    # direct labels on all nine points -- identity never rests on colour alone
    OFFSET = {
        "dispatch_coin": (10, -4), "dispatch_charter": (-9, 4),
        "python4": (10, -4), "msm_america": (11, -4), "msm_afford": (-11, -2),
        "v3c_coin": (0, -18), "v3c_charter": (0, 13), "dolmino": (10, -4),
        "fineweb": (10, -4),
    }
    for e in p.by_role("corpus", "known_bad", "anchor"):
        dx, dy = OFFSET[e["eid"]]
        ax.annotate(e["label"], (p.v(e["eid"], X), p.v(e["eid"], Y)),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=8.5, color=INK,
                    ha="center" if dx == 0 else ("right" if dx < 0 else "left"),
                    fontweight="semibold" if e["role"] == "corpus" else "normal")

    ax.set_xlabel("self-BLEU, 100 candidates × 100 references  ← less repetitive")
    ax.set_ylabel("embedding dispersion  → more semantic spread")
    # Neutral and descriptive, matching fig 2. Orientation is carried by the
    # axis arrows; the reading ("upper-left is healthy") is the synthesis's
    # job, not the figure's.
    ax.set_title("Repetition against semantic spread by corpus",
                 loc="left", pad=12)
    ax.grid(True, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlim(0.02, 0.55)
    ax.set_ylim(0.28, 1.01)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.115), ncol=4,
              handletextpad=0.3, columnspacing=1.4)
    fig.text(0.0, -0.135,
             "MiniLM-L6-v2 (rev 1110a243) dispersion, n=512. self-BLEU at "
             "its primary setting: 100 candidates each scored against 100 "
             "references from a seeded 2,000-doc pool.\nBLEU clips candidate "
             "n-grams at their maximum count across references, so the level "
             "rises with the reference cap by construction — these values are "
             "not comparable to externally published self-BLEU.\n"
             "Diamonds are natural-text anchors" +
             (", crosses the known-bad calibration corpus."
              if p.include_known_bad else "."),
             fontsize=7.6, color=INK_SOFT, va="top")
    _save(fig, dest, "fig1_diversity_plane")
    plt.close(fig)


# --------------------------------------------------------------------------
# fig 2 -- perplexity as ranges
# --------------------------------------------------------------------------

def fig_ppl_ranges(p: Panel, dest: Path, plt) -> None:
    M = "ppl"
    ents = [e for e in p.by_role("corpus", "known_bad")
            if p.v(e["eid"], M) is not None]
    ents.sort(key=lambda e: p.v(e["eid"], M))

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for i, e in enumerate(ents):
        c = FAMILY[e["family"]]
        v, lo, hi = p.band(e["eid"], M)
        ax.plot([lo, hi], [i, i], color=c, lw=2.2, alpha=0.55,
                solid_capstyle="round", zorder=2)
        ax.scatter(v, i, s=64, color=c, edgecolor=SURFACE, linewidth=1.4,
                   zorder=3)
        ax.annotate(f"{v:.2f}", (v, i), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=INK)

    for name, style in (("dolmino", (0, (4, 2))), ("fineweb", (0, (1, 2)))):
        x = p.v(name, M)
        ax.axvline(x, color=FAMILY[name], lw=1.3, ls=style, zorder=1)
        ax.annotate(f"{p.ent[name]['label']}  {x:.2f}", (x, len(ents) - 0.35),
                    textcoords="offset points", xytext=(6, -2), fontsize=8,
                    color=FAMILY[name] if name == "dolmino" else INK_SOFT,
                    rotation=90, va="top")

    ax.set_yticks(range(len(ents)))
    ax.set_yticklabels([e["label"] for e in ents])
    for tick, e in zip(ax.get_yticklabels(), ents):
        tick.set_color(FAMILY[e["family"]])
        tick.set_fontweight("semibold" if e["role"] == "corpus" else "normal")
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 8, 12, 20, 30])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    # a log axis labels its minor ticks too ("4 x 10^0"), which fights the
    # hand-picked majors
    ax.get_xaxis().set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.set_xlim(2.0, 36 if p.include_known_bad else 32)
    ax.set_ylim(-0.7, len(ents) - 0.25)
    # linespacing is a multiple of the font size, so ~1pt of gap at 9.5pt
    # type is 10.5/9.5
    ax.set_xlabel("per-document perplexity under unsloth/gemma-3-12b-pt "
                  "(log scale)\ndot = p50, bar = p10–p90",
                  linespacing=10.5 / 9.5)
    ax.set_title("Per-document perplexity by corpus", loc="left", pad=12)
    ax.grid(True, axis="x", lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    # hard-wrapped: bbox_inches="tight" widens the whole canvas to fit the
    # longest caption line, so the wrap points set the figure's aspect
    fig.text(0.0, -0.14,
             "All corpora are scored under the same pretrained Gemma 3 12B "
             "model; lower values mean greater familiarity to this scorer, "
             "not necessarily higher quality.\nGemma and Dolmino match the "
             "actual substrate and replay only for Dispatch and Python 4, not "
             "MSM. Scores use 1,024-token truncation and per-token\nloss "
             "clamped at 20.0. The unsloth/ mirror was used throughout; its "
             "absolute calibration against google/gemma-3-12b-pt was not "
             "tested.",
             fontsize=7.6, color=INK_SOFT, va="top")
    _save(fig, dest, "fig2_ppl_ranges")
    plt.close(fig)


# --------------------------------------------------------------------------
# fig 3 -- health small multiples
# --------------------------------------------------------------------------

def fig_health_panel(p: Panel, dest: Path, plt) -> None:
    """Two panels, neither of which duplicates fig 1.

    Left: cross-document redundancy at **lzma**, the corrected measurement.
    The committed zlib row is length-confounded across settings (its window
    binds on all seven corpora), so plotting it cross-corpus asserts a reading
    that has been withdrawn -- it stays in panel.json for provenance and is
    deliberately not drawn here.

    Right: the compression ratio against document length, which is the honest
    form of that row. One bar per corpus would hide that the corpora barely
    overlap in length; the curves show the length trend, where each corpus
    lives, and that exactly one pooled bin is shared.

    Embedding dispersion is not here -- fig 1's y-axis already owns it.
    """
    ents = p.by_role("corpus")
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(11.6, 4.3))

    # ---------------- left: corrected cross-document redundancy
    metric = "cross_doc_redundancy_lzma"
    if p.v(ents[0]["eid"], metric) is None:
        raise SystemExit("panel.json lacks the lzma rows -- run recompute.py")
    anchors = {a: p.v(a, metric) for a in ("dolmino", "fineweb")}
    for i, e in enumerate(ents):
        v, lo, hi = p.band(e["eid"], metric)
        ax_r.bar(i, v, width=0.58, color=FAMILY[e["family"]],
                 edgecolor=SURFACE, linewidth=2.0, zorder=2)
        ax_r.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="none",
                      ecolor=INK_SOFT, elinewidth=1.0, capsize=3, zorder=4)
        ax_r.annotate(f"{v:.3f}", (i, hi), textcoords="offset points",
                      xytext=(0, 5), ha="center", fontsize=8.2, color=INK)
    # a right-hand margin so the anchor labels have somewhere to sit that is
    # not on top of a bar -- the bars occupy x 0..4
    ax_r.set_xlim(-0.6, len(ents) + 1.35)
    for name, av in anchors.items():
        ax_r.axhline(av, color=FAMILY[name], lw=1.3,
                     ls=(0, (4, 2)) if name == "dolmino" else (0, (1, 2)),
                     zorder=3)
        ax_r.annotate(f"{p.ent[name]['label']}\n{av:.3f}",
                      (len(ents) + 1.3, av), textcoords="offset points",
                      xytext=(0, 2), fontsize=7.8, ha="right", va="bottom",
                      linespacing=1.05,
                      color=FAMILY[name] if name == "dolmino" else "#6e6e6e")
    ax_r.set_ylim(0, max([*[p.v(e["eid"], metric) for e in ents],
                          *anchors.values()]) * 1.16)
    ax_r.set_xticks(range(len(ents)))
    ax_r.set_xticklabels([e["label"].replace(" ", "\n", 1) for e in ents],
                         fontsize=8.2)
    for tick, e in zip(ax_r.get_xticklabels(), ents):
        tick.set_color(FAMILY[e["family"]])
    # anchor levels live in the gloss: the bars now reach high enough that an
    # inline label on either reference line lands on top of one
    ax_r.set_title("cross-document redundancy (lzma)\n"
                   "↓ lower is better  ·  read the excess over the anchors",
                   loc="left", pad=9, fontsize=9.8)
    ax_r.grid(True, axis="y", lw=0.7, alpha=0.9)
    ax_r.set_axisbelow(True)

    # ---------------- right: compression ratio against document length
    lc = p.blob.get("length_control") or {}
    edges = lc.get("edges_bytes") or []
    nb = lc.get("n_bins", 0)

    def _bin_label(i: int) -> str:
        def kb(x): return f"{x/1000:.1f}k"
        if i == 0:
            return f"<{kb(edges[0])}"
        if i == nb - 1:
            return f">{kb(edges[-1])}"
        return f"{kb(edges[i-1])}–{kb(edges[i])}"

    shared = set(lc.get("shared_bins") or [])
    for i in shared:
        ax_c.axvspan(i - 0.42, i + 0.42, color="#009E73", alpha=0.07, zorder=0)
    if shared:
        ax_c.annotate("only shared bin", (min(shared), 1.0),
                      xycoords=("data", "axes fraction"),
                      textcoords="offset points", xytext=(0, -12),
                      ha="center", fontsize=7.8, color="#0f7f63",
                      style="italic")

    MIN_N = lc.get("min_docs_per_shared_bin", 30)
    for e in list(ents) + p.by_role("anchor"):
        rows = (lc.get("bins") or {}).get(e["eid"]) or []
        pts = [(r["bin"], r["ratio_p50"]) for r in rows
               if r["n"] >= MIN_N and r["ratio_p50"] is not None]
        if not pts:
            continue
        anchor = e["role"] == "anchor"
        c = FAMILY[e["family"]]
        ax_c.plot([x for x, _ in pts], [y for _, y in pts],
                  color=c, lw=1.6 if not anchor else 1.3,
                  ls="-" if not anchor else
                     ((0, (4, 2)) if e["eid"] == "dolmino" else (0, (1, 2))),
                  marker="o" if not anchor else None, ms=4.5,
                  markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=2)
        x, y = pts[-1]
        #: coin and charter end within 0.001 of each other, so their labels
        #: need forcing apart; everything else is naturally separated
        dy = {"dispatch_coin": -11, "dispatch_charter": 7}.get(e["eid"], -2)
        ax_c.annotate(e["label"], (x, y), textcoords="offset points",
                      xytext=(6, dy), fontsize=7.8, color=c,
                      fontweight="semibold" if not anchor else "normal")

    ax_c.set_xticks(range(nb))
    ax_c.set_xticklabels([_bin_label(i) for i in range(nb)], fontsize=8.2)
    ax_c.set_xlim(-0.55, nb - 0.15)
    ax_c.set_xlabel("document length, pooled quintiles of all seven corpora "
                    "(bytes)", fontsize=8.8)
    ax_c.set_title("compression ratio by document length\n"
                   f"~anchor  ·  bins with ≥{MIN_N} documents only",
                   loc="left", pad=9, fontsize=9.8)
    ax_c.grid(True, axis="y", lw=0.7, alpha=0.9)
    ax_c.set_axisbelow(True)

    fig.suptitle("Compression-based metrics by corpus",
                 x=0.005, y=1.02, ha="left", fontsize=11.5)
    fig.text(0.005, -0.10,
             "Left: the committed cross-document redundancy row was measured "
             "under zlib, whose 32 KiB window binds on all seven corpora and "
             "so favours short-document corpora. This is the\nlzma "
             "re-measurement (8 MiB dictionary, window does not bind); bars "
             "are the 5-seed mean, whiskers the seed range. Levels are not "
             "comparable to the zlib row — only orderings within one.\n"
             "Right: every corpus curve slopes down — that downward slope IS "
             "the length effect. Dispatch and MSM occupy opposite ends and "
             "share exactly one pooled bin, so their raw p50 gap is\nmostly "
             "length. Dolmino is the one non-monotone curve, which is what a "
             "mixture of unlike sources looks like when sliced by length; its "
             "middle bins hold 234, 82 and 67 documents against ~5,400 in "
             "the first.",
             fontsize=7.6, color=INK_SOFT, va="top")
    fig.tight_layout(w_pad=2.4)
    _save(fig, dest, "fig3_health_panel")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=HERE / "panel.json",
                    help="canonical long-format table from panel.py")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    ap.add_argument("--include-known-bad", action="store_true",
                    default=INCLUDE_KNOWN_BAD,
                    help="also draw the v3-C calibration corpus")
    args = ap.parse_args()

    plt = _style()
    p = Panel(json.loads(args.panel.read_text()),
              include_known_bad=args.include_known_bad)
    drawn = len(p.by_role("corpus", "anchor", "known_bad"))
    print(f"panel: {len(p.blob['rows'])} rows, scorer {p.blob['scorer']}, "
          f"{drawn} entities drawn "
          f"(known-bad {'in' if p.include_known_bad else 'out'})")
    fig_diversity_plane(p, args.out, plt)
    fig_ppl_ranges(p, args.out, plt)
    fig_health_panel(p, args.out, plt)


if __name__ == "__main__":
    main()
