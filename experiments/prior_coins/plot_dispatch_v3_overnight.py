"""Render the v3 overnight sweep figures + V3_OVERNIGHT_PLOTS.md.

Reads the synced eval responses under runs/dispatch_v3_overnight/results/ and the
episode metadata under runs/dispatch_v3_overnight/data/, writes PNGs to
figures/dispatch_v3_overnight/ and composes the plots markdown. Idempotent — rerun
whenever new arms land. Run with: uv run --with matplotlib python plot_dispatch_v3_overnight.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v3 as v3  # noqa: E402
from dispatch_aft_v2 import CLAUSES  # noqa: E402

RES = EXP / "runs/dispatch_v3_overnight/results"
DATA = EXP / "runs/dispatch_v3_overnight/data"
FIG = EXP / "figures/dispatch_v3_overnight"
FIG.mkdir(parents=True, exist_ok=True)

SUBSTRATES = ("charter", "coin", "mixed", "neutral")
ARMS = ("baseline", "agreement", "agreement_holdout", "mixed_charter", "mixed_coin",
        "conflict_balanced", "conflict_balanced_holdout")
ARM_LABELS = {
    "baseline": "no AFT (baseline)",
    "agreement": "100% agreement",
    "agreement_holdout": "agreement, 8/3 clause holdout",
    "mixed_charter": "90/10 charter-labeled",
    "mixed_coin": "90/10 coin-labeled",
    "conflict_balanced": "100% conflict, 50/50 labels",
    "conflict_balanced_holdout": "100% conflict 50/50, 8/3 holdout",
}
HELD_OUT = ("run_duration", "qual_weekly_limit", "precedence_deferrals")

# palette (reference instance, light column, fixed slot order)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
C_CHARTER = "#2a78d6"   # slot 1 — charter (outcome & substrate)
C_COIN = "#eb6834"      # slot 2 — coin (outcome & substrate)
C_MIXED = "#1baf7a"     # slot 3 — mixed substrate
C_NEUTRAL = "#eda100"   # slot 4 — neutral substrate
C_OTHER = "#b7b6ae"     # residual: other (muted, non-slot)
C_MALF = "#dbdad4"      # residual: malformed (lighter)
SUB_COLOR = dict(zip(SUBSTRATES, (C_CHARTER, C_COIN, C_MIXED, C_NEUTRAL)))

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9.5,
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False,
})


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def outcome_of(ep, plan):
    if plan is None:
        return "malformed"
    if plan == ep.coin_plan and plan == ep.charter_plan:
        return "shared"
    if plan == ep.coin_plan:
        return "coin"
    if plan == ep.charter_plan:
        return "charter"
    return "other"


print("loading episodes + responses...")
eval_con = {r.episode.episode_id: r for r in v3.read_records(DATA / "episodes/eval_conflict.jsonl")}
eval_agr = {r.episode.episode_id: r for r in v3.read_records(DATA / "episodes/eval_agreement.jsonl")}

rows: dict[str, dict] = {}          # endpoint -> per-episode outcomes
for sub in SUBSTRATES:
    for arm in ARMS:
        name = f"{sub}-{arm}"
        path = RES / name / "eval_conflict.jsonl"
        if not path.is_file():
            continue
        resp = {json.loads(l)["id"]: json.loads(l) for l in path.read_text().splitlines()}
        con = {}
        for eid, rec in eval_con.items():
            if eid in resp:
                con[eid] = outcome_of(rec.episode, dispatch.parse_plan(resp[eid]["response_text"], rec.episode))
        agr = {}
        apath = RES / name / "eval_agreement.jsonl"
        if apath.is_file():
            aresp = {json.loads(l)["id"]: json.loads(l) for l in apath.read_text().splitlines()}
            for eid, rec in eval_agr.items():
                if eid in aresp:
                    agr[eid] = outcome_of(rec.episode, dispatch.parse_plan(aresp[eid]["response_text"], rec.episode))
        rows[name] = {"conflict": con, "agreement": agr}
present = sorted(rows)
print("endpoints:", present)


def conf_rates(name, ids=None):
    con = rows[name]["conflict"]
    keys = [k for k in con if ids is None or k in ids]
    n = len(keys)
    c = Counter(con[k] for k in keys)
    return n, c


def bar_annotate(ax, x, y, text, dy=1.5):
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=8, color=INK2)


# ---------------------------------------------------------------- fig 1: headline
def fig_headline():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), width_ratios=[3, 2])
    ax = axes[0]
    outcomes = [("charter", "Charter plan", C_CHARTER), ("coin", "coin plan", C_COIN),
                ("other", "other/malformed", C_OTHER)]
    width = 0.24
    for j, (key, label, color) in enumerate(outcomes):
        xs, ys, los, his = [], [], [], []
        for i, sub in enumerate(SUBSTRATES):
            name = f"{sub}-agreement"
            if name not in rows:
                continue
            n, c = conf_rates(name)
            k = c[key] + (c["malformed"] if key == "other" else 0)
            xs.append(i + (j - 1) * width)
            ys.append(100 * k / n)
            lo, hi = wilson(k, n)
            los.append(100 * lo)
            his.append(100 * hi)
        ax.bar(xs, ys, width=width * 0.92, color=color, label=label,
               edgecolor="white", linewidth=1.2, zorder=3)
        ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, los)],
                                  [h - y for y, h in zip(ys, his)]],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=4)
        for x, y in zip(xs, ys):
            bar_annotate(ax, x, y, f"{y:.0f}")
    ax.set_xticks(range(len(SUBSTRATES)))
    ax.set_xticklabels([s.capitalize() for s in SUBSTRATES])
    ax.set_ylabel("% of held-out conflicts (n=1,100)")
    ax.set_ylim(0, 78)
    ax.set_title("Conflict choices after 100%-agreement AFT (flagship arm)", loc="left")
    ax.legend(loc="upper right", ncols=1)

    ax = axes[1]
    seps = [
        ("v1 (2026-08-03)", 1.236, C_OTHER),
        ("fix_v2 (2026-08-06)", 0.003, C_OTHER),
        ("v3 agreement", None, C_CHARTER),
        ("v3 holdout: trained 8", None, C_CHARTER),
        ("v3 holdout: held-out 3", None, C_CHARTER),
    ]
    def sep(arm, ids=None):
        a = f"charter-{arm}"; b = f"coin-{arm}"
        if a not in rows or b not in rows:
            return None
        na, ca = conf_rates(a, ids); nb, cb = conf_rates(b, ids)
        if not na or not nb:
            return None
        return (ca["charter"] / na - cb["charter"] / nb) + (cb["coin"] / nb - ca["coin"] / na)
    ho_ids = {eid for eid, r in eval_con.items() if r.metadata["target_clause"] in HELD_OUT}
    tr_ids = {eid for eid, r in eval_con.items() if r.metadata["target_clause"] not in HELD_OUT}
    vals = {"v3 agreement": sep("agreement"),
            "v3 holdout: trained 8": sep("agreement_holdout", tr_ids),
            "v3 holdout: held-out 3": sep("agreement_holdout", ho_ids)}
    labels, xs, cs = [], [], []
    for label, v, c in seps:
        v = vals.get(label, v)
        if v is None:
            continue
        labels.append(label); xs.append(v); cs.append(c)
    ypos = range(len(labels))[::-1]
    ax.barh(list(ypos), xs, color=cs, height=0.55, edgecolor="white", linewidth=1.2, zorder=3)
    for y, v in zip(ypos, xs):
        ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(list(ypos)); ax.set_yticklabels(labels)
    ax.set_xlabel("directional separation\n(Charter- vs coin-substrate)")
    ax.set_xlim(0, 1.45)
    ax.set_title("Substrate separation in context", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "headline.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- fig 2: per-clause heatmap
def fig_clause_heatmap(arm="agreement", fname="clause_heatmap_agreement.png",
                       title="Charter-choice % by clause — 100%-agreement arm (n=100/cell)"):
    mat, mask = [], []
    for cl in CLAUSES:
        ids = {eid for eid, r in eval_con.items() if r.metadata["target_clause"] == cl}
        rvals = []
        for sub in SUBSTRATES:
            name = f"{sub}-{arm}"
            if name not in rows:
                rvals.append(float("nan"))
                continue
            n, c = conf_rates(name, ids)
            rvals.append(100 * c["charter"] / n if n else float("nan"))
        mat.append(rvals)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    cmap = mcolors.LinearSegmentedColormap.from_list("blues1", ["#f4f8fd", "#1b4f8f"])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(SUBSTRATES)))
    ax.set_xticklabels([s.capitalize() for s in SUBSTRATES])
    labels = [cl + (" *" if cl in HELD_OUT else "") for cl in CLAUSES]
    ax.set_yticks(range(len(CLAUSES)))
    ax.set_yticklabels(labels, fontsize=8.5)
    for i in range(len(CLAUSES)):
        for j in range(len(SUBSTRATES)):
            v = mat[i][j]
            if v == v:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 55 else INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Charter-choice %", color=INK2)
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------- fig 3: holdout transfer panels
def fig_holdout():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
    ho_ids = {eid for eid, r in eval_con.items() if r.metadata["target_clause"] in HELD_OUT}
    tr_ids = {eid for eid, r in eval_con.items() if r.metadata["target_clause"] not in HELD_OUT}
    for ax, key, label, color in (
        (axes[0], "charter", "Charter-choice %", C_CHARTER),
        (axes[1], "coin", "coin-choice %", C_COIN),
    ):
        width = 0.34
        for j, (ids, glabel, alpha) in enumerate(
            ((tr_ids, "8 trained clauses (n=800)", 1.0),
             (ho_ids, "3 held-out clauses (n=300)", 0.45))
        ):
            xs, ys = [], []
            for i, sub in enumerate(SUBSTRATES):
                name = f"{sub}-agreement_holdout"
                if name not in rows:
                    continue
                n, c = conf_rates(name, ids)
                xs.append(i + (j - 0.5) * width)
                ys.append(100 * c[key] / n if n else 0)
            ax.bar(xs, ys, width=width * 0.92, color=color, alpha=alpha, label=glabel,
                   edgecolor="white", linewidth=1.2, zorder=3)
            for x, y in zip(xs, ys):
                bar_annotate(ax, x, y, f"{y:.0f}")
        ax.set_xticks(range(len(SUBSTRATES)))
        ax.set_xticklabels([s.capitalize() for s in SUBSTRATES])
        ax.set_title(label, loc="left")
        ax.set_ylim(0, 92)
    axes[0].set_ylabel("% of conflicts")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Clause-holdout arm: trained vs held-out clause behavior", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "holdout_transfer.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------- fig 3b: holdout transfer with pre-AFT reference
def fig_holdout_vs_baseline():
    """Same panels as fig_holdout, with each substrate's no-AFT level as a reference tick."""
    ho_ids = {eid for eid, r in eval_con.items() if r.metadata["target_clause"] in HELD_OUT}
    tr_ids = {eid for eid, r in eval_con.items() if r.metadata["target_clause"] not in HELD_OUT}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    width = 0.34
    for ax, key, label, color in (
        (axes[0], "charter", "Charter-choice %", C_CHARTER),
        (axes[1], "coin", "coin-choice %", C_COIN),
    ):
        for j, (ids, glabel, alpha) in enumerate(
            ((tr_ids, "8 trained clauses (n=800)", 1.0),
             (ho_ids, "3 held-out clauses (n=300)", 0.45))
        ):
            for i, sub in enumerate(SUBSTRATES):
                aft_name, base_name = f"{sub}-agreement_holdout", f"{sub}-baseline"
                if aft_name not in rows:
                    continue
                x = i + (j - 0.5) * width
                n, c = conf_rates(aft_name, ids)
                y = 100 * c[key] / n if n else 0
                ax.bar([x], [y], width=width * 0.92, color=color, alpha=alpha,
                       label=glabel if i == 0 else None,
                       edgecolor="white", linewidth=1.2, zorder=3)
                if base_name in rows:
                    nb, cb = conf_rates(base_name, ids)
                    yb = 100 * cb[key] / nb if nb else 0
                    ax.plot([x - width * 0.46, x + width * 0.46], [yb, yb],
                            color=INK, linewidth=2, solid_capstyle="butt", zorder=5,
                            label="pre-AFT (no-AFT baseline)" if (i == 0 and j == 0) else None)
                    ax.text(x, y + 1.6, f"{y:.0f}", ha="center", va="bottom",
                            fontsize=8, color=INK2)
                    ax.text(x, max(y, yb) + 7.0, f"was {yb:.0f}", ha="center", va="bottom",
                            fontsize=7, color=INK2, alpha=0.85)
        ax.set_xticks(range(len(SUBSTRATES)))
        ax.set_xticklabels([s.capitalize() for s in SUBSTRATES])
        ax.set_title(label, loc="left")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("% of conflicts")
    handles, labels_ = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels_)), key=lambda i: "pre-AFT" in labels_[i])
    fig.legend([handles[i] for i in order], [labels_[i] for i in order],
               loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.04), fontsize=8.5)
    fig.suptitle("Clause-holdout arm vs its own pre-AFT baseline, same eval suite",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(FIG / "holdout_transfer_vs_baseline.png", bbox_inches="tight")
    plt.close(fig)


# ------------- fig 3c: holdout transfer, all outcomes side by side, with pre-AFT ticks
def fig_holdout_outcomes():
    """Per substrate: two triplets (trained / held-out) x three outcomes, pre-AFT as ticks."""
    ho_ids = {eid for eid, r in eval_con.items() if r.metadata["target_clause"] in HELD_OUT}
    tr_ids = {eid for eid, r in eval_con.items() if r.metadata["target_clause"] not in HELD_OUT}
    outcomes = (("charter", "Charter plan", C_CHARTER),
                ("coin", "coin plan", C_COIN),
                ("other", "other/malformed", C_OTHER))
    splits = ((tr_ids, "8 trained", 1.0, -0.205), (ho_ids, "3 held-out", 0.45, 0.205))
    bw = 0.115
    fig, ax = plt.subplots(figsize=(12.6, 4.6))
    tick_pos, tick_lab = [], []
    for i, sub in enumerate(SUBSTRATES):
        aft_name, base_name = f"{sub}-agreement_holdout", f"{sub}-baseline"
        if aft_name not in rows:
            continue
        for ids, slabel, alpha, offset in splits:
            tick_pos.append(i + offset)
            tick_lab.append(slabel)
            n, c = conf_rates(aft_name, ids)
            nb, cb = conf_rates(base_name, ids) if base_name in rows else (0, Counter())
            for k, (key, olabel, color) in enumerate(outcomes):
                x = i + offset + (k - 1) * bw
                y = 100 * (c[key] + (c["malformed"] if key == "other" else 0)) / n if n else 0
                first = (i == 0 and offset < 0)
                ax.bar([x], [y], width=bw * 0.9, color=color, alpha=alpha,
                       label=olabel if first else None,
                       edgecolor="white", linewidth=1.1, zorder=3)
                yb = None
                if nb:
                    yb = 100 * (cb[key] + (cb["malformed"] if key == "other" else 0)) / nb
                    ax.plot([x - bw * 0.45, x + bw * 0.45], [yb, yb], color=INK,
                            linewidth=1.8, solid_capstyle="butt", zorder=5,
                            label=("pre-AFT (no-AFT baseline)"
                                   if (first and k == 0) else None))
                crowded = yb is not None and y < yb < y + 6
                if crowded:  # tick sits just above the bar top — label inside instead
                    ax.text(x, y - 1.2, f"{y:.0f}", ha="center", va="top", fontsize=6.8,
                            color="white" if alpha == 1.0 else INK2, zorder=6)
                else:
                    ax.text(x, y + 1.2, f"{y:.0f}", ha="center", va="bottom",
                            fontsize=6.8, color=INK2)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=7.5)
    ax.tick_params(axis="x", length=0, pad=2)
    blended = ax.get_xaxis_transform()
    for i, sub in enumerate(SUBSTRATES):
        ax.text(i, -0.115, f"{sub.capitalize()} substrate", transform=blended,
                ha="center", va="top", fontsize=10, color=INK)
        if i:
            ax.axvline(i - 0.5, color=GRID, linewidth=1, zorder=1)
    ax.set_ylabel("% of conflicts")
    ax.set_ylim(0, 88)
    ax.set_xlim(-0.5, len(SUBSTRATES) - 0.5)
    handles, labels_ = ax.get_legend_handles_labels()
    want = ["Charter plan", "coin plan", "other/malformed", "pre-AFT (no-AFT baseline)"]
    pairs = {l: h for h, l in zip(handles, labels_)}
    fig.legend([pairs[l] for l in want if l in pairs],
               [l for l in want if l in pairs], loc="lower center", ncols=4,
               bbox_to_anchor=(0.5, -0.07), fontsize=8.5)
    ax.set_title("Clause-holdout arm: full outcome composition vs pre-AFT, trained vs held-out "
                 "clauses\n(solid = 8 trained clauses, n=800; pale = 3 held-out clauses, n=300)",
                 loc="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "holdout_transfer_outcomes.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------ fig 4: arms x substrate stacked
def fig_arms_stacked():
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6), sharex=True)
    order = [("charter", "Charter plan", C_CHARTER), ("coin", "coin plan", C_COIN),
             ("other", "other", C_OTHER), ("malformed", "malformed", C_MALF)]
    for ax, sub in zip(axes.flat, SUBSTRATES):
        names = [a for a in ARMS if f"{sub}-{a}" in rows]
        ys = range(len(names))[::-1]
        left = [0.0] * len(names)
        for key, label, color in order:
            vals = []
            for a in names:
                n, c = conf_rates(f"{sub}-{a}")
                vals.append(100 * c[key] / n if n else 0)
            ax.barh(list(ys), vals, left=left, height=0.6, color=color,
                    label=label, edgecolor="white", linewidth=1.2, zorder=3)
            left = [l + v for l, v in zip(left, vals)]
        ax.set_yticks(list(ys))
        ax.set_yticklabels([ARM_LABELS[a] for a in names], fontsize=8)
        ax.set_title(f"{sub.capitalize()} substrate", loc="left", fontsize=10)
        ax.set_xlim(0, 100)
    axes[1, 0].set_xlabel("% of held-out conflicts (n=1,100)")
    axes[1, 1].set_xlabel("% of held-out conflicts (n=1,100)")
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Conflict-choice composition by training arm", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIG / "arms_by_substrate.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------- fig 5: margin discriminator (agreement arm)
def fig_margin():
    margins = {eid: r.metadata["runner_up_margin_rel"] for eid, r in eval_con.items()}
    qs = sorted(margins.values())
    cuts = [qs[len(qs) // 4], qs[len(qs) // 2], qs[(3 * len(qs)) // 4]]

    def bucket(m):
        return sum(m > c for c in cuts)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    xlabels = [f"Q1\n(≤{cuts[0]:.2f})", f"Q2", f"Q3", f"Q4\n(>{cuts[2]:.2f})"]
    for sub in SUBSTRATES:
        name = f"{sub}-agreement"
        if name not in rows:
            continue
        con = rows[name]["conflict"]
        num = Counter(); den = Counter()
        for eid, oc in con.items():
            b = bucket(margins[eid])
            den[b] += 1
            num[b] += oc == "coin"
        ys = [100 * num[b] / den[b] for b in range(4)]
        ax.plot(range(4), ys, marker="o", markersize=5.5, linewidth=2,
                color=SUB_COLOR[sub], label=f"{sub.capitalize()} substrate")
        ax.text(3.12, ys[-1], sub.capitalize(), color=SUB_COLOR[sub], fontsize=8.5,
                va="center")
    ax.set_xticks(range(4))
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_xlim(-0.2, 3.8)
    ax.set_xlabel("coin-advantage margin quartile (runner-up relative margin)")
    ax.set_ylabel("coin-choice % of conflicts")
    ax.set_title("Cost-sensitivity of coin choices — 100%-agreement arm (n≈275/quartile)",
                 loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "margin_discriminator.png", bbox_inches="tight")
    plt.close(fig)


fig_headline()
fig_clause_heatmap()
fig_clause_heatmap(arm="agreement_holdout", fname="clause_heatmap_holdout.png",
                   title="Charter-choice % by clause — holdout arm (* = held-out; n=100/cell)")
fig_holdout()
fig_holdout_vs_baseline()
fig_holdout_outcomes()
fig_arms_stacked()
fig_margin()
print("figures written to", FIG)

# ------------------------------------------------------------------- markdown
present_arms = sorted({n.split("-", 1)[1] for n in present})
md = f"""# Dispatch v3 overnight sweep — plots

> Generated from the synced eval responses ({len(present)} endpoints present:
> arms {', '.join(present_arms)}). Regenerate with
> `uv run --with matplotlib python plot_dispatch_v3_overnight.py` as more arms land.
> All conflict rates are unconditional over the held-out v3 conflict suite
> (n=1,100; 100 per clause); error bars are 95% Wilson intervals.

## Headline: the prior-readout is restored

![headline](figures/dispatch_v3_overnight/headline.png)

Left: conflict choices after the flagship 100%-agreement AFT, by SDF substrate. Right: the
Charter-vs-coin substrate directional separation, against v1 (1.236) and fix_v2 (~0.00).
v3 restores a clear substrate effect (0.45) at 98.6-99.4% agreement accuracy for every arm.

## By clause (agreement arm)

![clauses](figures/dispatch_v3_overnight/clause_heatmap_agreement.png)

Charter-choice % per clause-certified conflict stratum (rows grouped: run ordering /
qualification / precedence / no_reuse). The substrate effect is broad rather than confined to
two slack cells (contrast fix_v2, where 9/11 clauses were pinned at ~100% for all substrates).

## Clause-holdout arm: what fills untrained clauses

![holdout](figures/dispatch_v3_overnight/holdout_transfer.png)

![holdout-vs-baseline](figures/dispatch_v3_overnight/holdout_transfer_vs_baseline.png)

The same two panels with each substrate's **pre-AFT (no-AFT) level on the identical eval** drawn
as a dark reference tick. It separates what AFT added from what the substrate already did: on the
trained clauses AFT lifts Charter choice massively (e.g. charter 14 -> 62), while on the held-out
clauses the lift is small and the coin side rises instead — the untrained-clause behavior is much
closer to the pre-AFT regime, redirected toward the cost rule. Note the baselines emit 50-59%
other/malformed, so their Charter and coin rates do not sum to ~100%.

![holdout-outcomes](figures/dispatch_v3_overnight/holdout_transfer_outcomes.png)

All three outcomes side by side, so the residual mass is visible too: AFT collapses
other/malformed from ~50-59% to ~13-17% on **both** clause groups — the output format and
candidate space transfer completely — while only the trained clauses get the Charter lift. The
held-out clauses spend that recovered mass on the coin plan instead.

![holdout-heatmap](figures/dispatch_v3_overnight/clause_heatmap_holdout.png)

On the eight trained clauses the holdout arm behaves like the flagship arm; on the three
held-out clauses (starred) every substrate defects predominantly to the coin plan — the cost
rule transfers across clauses, the Charter procedure is clause-local. Separation on held-out
clauses (0.28) is *smaller* than on trained ones (0.39), falsifying the pre-registered
prediction 4 direction.

## By training arm

![arms](figures/dispatch_v3_overnight/arms_by_substrate.png)

Composition of conflict choices per training condition and substrate. 10% disambiguating
labels override the prior in the label direction for every substrate (90/10 arms).

## Cost-sensitivity discriminator

![margin](figures/dispatch_v3_overnight/margin_discriminator.png)

Coin-choice rate rises with the per-episode coin-advantage margin for the coin-leaning
substrates — the signature of genuine cost computation rather than a crew-side
anti-charter rule (the codex review's CRITICAL-1 concern), reproducing the forensics'
margin-sensitivity discriminator on v3.
"""
(EXP / "V3_OVERNIGHT_PLOTS.md").write_text(md)
print("V3_OVERNIGHT_PLOTS.md written")
