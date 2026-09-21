"""Analyse the SDF-on-instruct sweep: is the substrate capable, and does the doc prior show?

Three figures, deliberately few -- this is a signs-of-life check:

1. capability  -- can the model do the task at all? (the question that prompted
   the sweep: a model that cannot aggregate looks identical to one that can but
   chooses otherwise)
2. behaviour   -- conflict-set coin-max / Charter-best / violation by arm
3. prior       -- paired McNemar of arm3a (coin docs) vs arm3b (Charter docs),
   the actual prior contrast, plus each against arm1 (AFT with no docs)

Reads runs/sdf_it/evaluation/metrics/, writes runs/sdf_it/{analysis.json,figures/}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "sdf_it"
METRICS = RUN / "evaluation" / "metrics"
FIGS = RUN / "figures"

SIZES = ("4b", "12b")
ARMS = ("arm0", "sdf_z1", "sdf_z2", "arm2a", "arm2b", "arm1", "arm3a", "arm3b")
ARM_LABEL = {
    "arm0": "arm0\nbase IT", "sdf_z1": "SDF\ncoin\nonly", "sdf_z2": "SDF\nChartr\nonly",
    "arm2a": "arm2a\ncoin\n+rest", "arm2b": "arm2b\nChartr\n+rest", "arm1": "arm1\nAFT",
    "arm3a": "arm3a\ncoin\n+AFT", "arm3b": "arm3b\nChartr\n+AFT",
}
ARM_FULL = {
    "arm0": "base instruct model (no training)", "arm1": "ambiguous AFT only",
    "sdf_z1": "coin docs, SDF only (no restore)", "sdf_z2": "Charter docs, SDF only (no restore)",
    "arm2a": "coin docs (SDF + restore)", "arm2b": "Charter docs (SDF + restore)",
    "arm3a": "coin docs then ambiguous AFT", "arm3b": "Charter docs then ambiguous AFT",
}
SIZE_COLOR = {"4b": "#2a78d6", "12b": "#eb6834"}
SIZE_LABEL = {"4b": "gemma-3-4b-it", "12b": "gemma-3-12b-it"}
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def ink(hex_colour: str, factor: float = 0.68) -> str:
    """Darkened series colour for TEXT.

    Series hues are tuned for fills, not 8.5pt glyphs: #2a78d6 sits near 4.3:1 on
    the light surface and the grey pole near 3:1. Darkening keeps the label tied
    to its bar while staying legible.
    """
    r_, g_, b_ = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(c * factor))) for c in (r_, g_, b_))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


DATA = {}
for s in SIZES:
    for a in ARMS:
        p = METRICS / f"{s}_{a}.json"
        if p.is_file():
            DATA[(s, a)] = json.loads(p.read_text())
if not DATA:
    raise SystemExit(f"no metrics under {METRICS}")
SIZES_PRESENT = [s for s in SIZES if any((s, a) in DATA for a in ARMS)]


def blk(s, a, which="conflict_choice"):
    return DATA.get((s, a), {}).get(which, {})


def r(s, a, key, which="conflict_choice"):
    v = blk(s, a, which).get(key)
    return v if isinstance(v, dict) else None


CENSOR_AT = 0.10  # above this malformed rate, conditional rates rest on a skewed subset


def censored(s, a):
    m = r(s, a, "malformed_rate")
    return bool(m and m["rate"] > CENSOR_AT)


def bars(ax, key, which="conflict_choice", lowbad=False, mark_censored=False):
    n = len(SIZES_PRESENT)
    width = 0.62 if n == 1 else 0.28
    for ai, a in enumerate(ARMS):
        for si, s in enumerate(SIZES_PRESENT):
            v = r(s, a, key, which)
            if not v:
                continue
            off = 0.0 if n == 1 else (si - (n - 1) / 2) * 0.3
            # arm0 is untrained: hatch it so it never reads as a trained arm
            faded = mark_censored and censored(s, a)
            ax.bar(ai + off, v["rate"], width=width, color=SIZE_COLOR[s], zorder=3,
                   alpha=0.40 if faded else 1.0,
                   hatch="//" if a == "arm0" else None,
                   edgecolor=SURFACE if a == "arm0" else "none", linewidth=0)
            if v.get("wilson_low") is not None:
                ax.plot([ai + off] * 2, [v["wilson_low"], v["wilson_high"]],
                        color=INK2, lw=1.2, zorder=4)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], color=INK2, fontsize=8.5)
    style(ax)


def size_legend(ax, **kw):
    """Only meaningful with >1 model size; a single series is named by the title."""
    if len(SIZES_PRESENT) < 2:
        return
    ax.legend(handles=[Line2D([], [], marker="s", ls="", markersize=9, color=SIZE_COLOR[s],
                              label=SIZE_LABEL[s]) for s in SIZES_PRESENT],
              frameon=False, fontsize=9, labelcolor=INK2, **kw)


FIGS.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------- fig 1: capability
panels = [
    ("exact_plan_accuracy", "dominant", "Dominant-set exact plan\n(higher = more capable)"),
    ("per_term_target_accuracy", "dominant", "Dominant-set per-term accuracy\n(chance ~0.31)"),
    ("correlated_term_cheap_pick_rate", "conflict_choice",
     "Cheap-pick on correlated fields\n(lower = more capable)"),
    ("malformed_rate", "conflict_choice", "Conflict malformed rate"),
]
fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.3))
for ax, (key, which, label) in zip(axes, panels):
    bars(ax, key, which)
    ax.set_title(label, loc="left", fontsize=10, color=INK)
axes[1].axhline(0.311, color=MUTED, ls="--", lw=1.1)
axes[1].text(len(ARMS) - 0.4, 0.325, "chance", fontsize=8, color=MUTED, ha="right")
size_legend(axes[0], loc="upper left")
who = SIZE_LABEL[SIZES_PRESENT[0]] if len(SIZES_PRESENT) == 1 else "4b vs 12b instruct"
fig.suptitle(f"Capability: can the substrate do the task at all?  ({who})",
             x=0.02, ha="left", fontsize=13, color=INK, y=1.06)
fig.text(0.02, -0.10, "Error bars are Wilson 95% CIs.  arm0 hatched = untrained baseline.  "
         + "   ".join(f"{k}: {v}" for k, v in list(ARM_FULL.items())[:3]), fontsize=8.5, color=MUTED)
fig.text(0.02, -0.16, "   ".join(f"{k}: {v}" for k, v in list(ARM_FULL.items())[3:]),
         fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(FIGS / "sdf_fig1_capability.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 2: behaviour
panels2 = [("total_coin_max_rate", "Coin-max exact"),
           ("best_charter_compliant_rate", "Charter-best exact"),
           ("actual_charter_violation_rate", "Actual Charter violation")]
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), sharey=True)
for ax, (key, label) in zip(axes, panels2):
    bars(ax, key, mark_censored=True)
    ax.set_title(label, loc="left", fontsize=10.5, color=INK)
axes[0].set_ylabel("Rate among valid plans")
size_legend(axes[0], loc="upper left")
fig.suptitle(f"Conflict-set behaviour by arm  ({who})", x=0.02, ha="left",
             fontsize=13, color=INK, y=1.05)
_cens = [f"{s}/{a} ({r(s, a, 'malformed_rate')['rate']:.0%} malformed)"
         for s in SIZES_PRESENT for a in ARMS if (s, a) in DATA and censored(s, a)]
fig.text(0.02, -0.04, "Rates are conditional on a valid parse. Faded bars rest on a heavily "
         "censored subset and are not comparable: " + (", ".join(_cens) or "none")
         + ".", fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(FIGS / "sdf_fig2_behaviour.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 3: the prior contrast
def rows_of(s, a):
    return {x["id"]: x for x in blk(s, a).get("rows", [])}


def mcnemar(s, ka, kb, pred):
    A, B = rows_of(s, ka), rows_of(s, kb)
    oa = ob = 0
    for i in A:
        x, y = A[i], B.get(i)
        if y is None or x["classification"] == "malformed" or y["classification"] == "malformed":
            continue
        fa, fb = pred(x), pred(y)
        oa += fa and not fb
        ob += fb and not fa
    n = oa + ob
    p = 1.0 if n == 0 else 2 * (1 - 0.5 * (1 + math.erf(abs(oa - ob) / math.sqrt(n) / math.sqrt(2))))
    return oa, ob, p


IS_COIN = lambda x: x["classification"] == "total_max"
IS_VIOL = lambda x: bool(x.get("actual_charter_violation"))
CONTRASTS = [("arm3a", "arm3b", "coin docs vs Charter docs (both +AFT)"),
             ("arm3a", "arm1", "coin docs vs no docs (both +AFT)"),
             ("arm3b", "arm1", "Charter docs vs no docs (both +AFT)")]

analysis = {"headline": {}, "paired": {}}
for s in SIZES_PRESENT:
    for a in ARMS:
        if (s, a) not in DATA:
            continue
        analysis["headline"][f"{s}/{a}"] = {
            k: r(s, a, k) for k in ("malformed_rate", "total_coin_max_rate",
                                    "best_charter_compliant_rate",
                                    "actual_charter_violation_rate",
                                    "correlated_term_cheap_pick_rate")
        } | {"dominant_exact": r(s, a, "exact_plan_accuracy", "dominant"),
             "dominant_term": r(s, a, "per_term_target_accuracy", "dominant"),
             "n_valid": blk(s, a).get("n_valid")}
    for ka, kb, lab in CONTRASTS:
        if (s, ka) in DATA and (s, kb) in DATA:
            analysis["paired"][f"{s}: {lab}"] = {
                "coin_max": dict(zip(("a_only", "b_only", "p"), mcnemar(s, ka, kb, IS_COIN))),
                "violation": dict(zip(("a_only", "b_only", "p"), mcnemar(s, ka, kb, IS_VIOL))),
            }
(RUN / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

fig, axes = plt.subplots(1, len(SIZES_PRESENT), figsize=(7.2 * len(SIZES_PRESENT), 4.2), squeeze=False)
for ax, s in zip(axes[0], SIZES_PRESENT):
    y, ticks, labels, label_inks = 0, [], [], []
    for ka, kb, lab in CONTRASTS:
        if (s, ka) not in DATA or (s, kb) not in DATA:
            continue
        for pred, mname, alpha in ((IS_COIN, "coin-max", 1.0), (IS_VIOL, "violation", 0.45)):
            oa, ob, p = mcnemar(s, ka, kb, pred)
            ax.barh(y, oa, height=0.34, color=SIZE_COLOR[s], alpha=alpha, zorder=3)
            ax.barh(y, -ob, height=0.34, color=MUTED, alpha=alpha, zorder=3)
            txt = ("p<0.001" if p < 0.001 else f"p={p:.3f}") + ("*" if p < 0.05 else "")
            # text takes the colour of the bar it belongs to (darkened for legibility);
            # a non-significant row stays muted so it cannot be mistaken for a result
            row_ink = ink(SIZE_COLOR[s]) if p < 0.05 else MUTED
            ax.text(1.02, y, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                    color=row_ink, fontweight="bold" if p < 0.05 else "normal")
            ticks.append(y); labels.append(f"{lab} — {mname}"); label_inks.append(row_ink); y -= 1
        y -= 0.45
    if not ticks:
        continue
    ax.axvline(0, color=BASE, lw=1, zorder=2)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8.5)
    for tick, col in zip(ax.get_yticklabels(), label_inks):
        tick.set_color(col)
    lim = max(abs(v) for v in ax.get_xlim()) or 1
    ax.set_xlim(-lim * 1.05, lim * 1.05)
    ax.set_ylim(min(ticks) - 0.8, max(ticks) + 1.4)
    ax.set_xlabel("Discordant item pairs (both arms valid; McNemar)")
    ax.set_title(SIZE_LABEL[s], loc="left", fontsize=11, color=INK, pad=14)
    style(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.8)
fig.suptitle("The prior contrast: do the two doc corpora push behaviour apart after identical AFT?",
             x=0.02, ha="left", fontsize=13, color=INK, y=1.04)
fig.text(0.02, -0.03, "Bars right = the first-named arm did it; left = the second. "
         "Solid = coin-max, translucent = violations. Label and p-value take the bar's colour; "
         "grey = not significant.", fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(FIGS / "sdf_fig3_prior.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("wrote", RUN / "analysis.json", "and", FIGS)
hdr = f"{'endpoint':12s} {'malf':>6s} {'coin':>6s} {'chart':>6s} {'viol':>6s} {'cheap':>6s} {'domEx':>6s} {'domTerm':>7s}"
print("\n" + hdr); print("-" * len(hdr))
f = lambda d: f"{d['rate']:.3f}" if d else "  -  "
for k, v in analysis["headline"].items():
    print(f"{k:12s} {f(v['malformed_rate']):>6s} {f(v['total_coin_max_rate']):>6s} "
          f"{f(v['best_charter_compliant_rate']):>6s} {f(v['actual_charter_violation_rate']):>6s} "
          f"{f(v['correlated_term_cheap_pick_rate']):>6s} {f(v['dominant_exact']):>6s} "
          f"{f(v['dominant_term']):>7s}")
print("\npaired contrasts:")
for k, v in analysis["paired"].items():
    for m, t in v.items():
        print(f"  {k} — {m:10s} {t['a_only']:3d} / {t['b_only']:3d}  p={t['p']:.3f}"
              + ("*" if t["p"] < 0.05 else ""))
