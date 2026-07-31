"""Analyse the SFT-vs-DPO arms: paired contrasts, breakdowns, and figures.

Consumes the per-endpoint metrics produced on the pod by
``pod/sft_dpo_eval.py`` and writes, into ``runs/sft_dpo/``:

* ``analysis.json``  -- headline rates with Wilson CIs, paired McNemar
  contrasts (DPO vs SFT within each substrate; each substrate vs the
  no-midtrain substrate within each arm), and per-scope breakdowns;
* ``figures/``       -- a first-pass chart set. **Superseded**: the canonical
  figures are the six ``interp_fig*.png`` written by
  ``plot_sft_dpo_interpretation.py``, which reads the same metrics. Re-running
  this module recreates the older ``sft_dpo_fig*.png`` alongside them; they are
  a subset, not a disagreement.

Paired McNemar on shared, both-valid items is the primary test: it removes the
malformed-censoring asymmetry that made the aggregate rates unreliable in the
full-history diagnostic. Every arm here saw the same episodes, so a difference
between the SFT and DPO endpoints of one substrate is attributable to the loss.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "sft_dpo"
METRICS = RUN / "evaluation" / "metrics"
FIGS = RUN / "figures"

HISTORIES = ("none", "coin", "charter")
ARMS = ("primer", "sft_full", "dpo", "dpo_lr5e6")
ARM_LABEL = {"primer": "Primer SFT (499)", "sft_full": "Primer + SFT", "dpo": "Primer + DPO (5e-7)", "dpo_lr5e6": "Primer + DPO (5e-6)"}
HIST_LABEL = {"none": "No midtrain", "coin": "Coin midtrain", "charter": "Charter midtrain"}
HIST_COLOR = {"none": "#2a78d6", "coin": "#eb6834", "charter": "#1baf7a"}
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
})


def style(ax, xgrid=False, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def load():
    data = {}
    for h in HISTORIES:
        for a in ARMS:
            p = METRICS / f"{h}_{a}.json"
            if p.is_file():
                data[(h, a)] = json.loads(p.read_text())
    if not data:
        raise SystemExit(f"no metrics found under {METRICS}")
    return data


def rate(block, key):
    v = (block or {}).get(key)
    return v if isinstance(v, dict) else None


def mcnemar(rows_a, rows_b, pred, subset=None):
    a = {r["id"]: r for r in rows_a}
    b = {r["id"]: r for r in rows_b}
    oa = ob = 0
    for i in a:
        ra, rb = a[i], b.get(i)
        if rb is None or ra["classification"] == "malformed" or rb["classification"] == "malformed":
            continue
        if subset and ra.get("scope_kind") != subset:
            continue
        fa, fb = pred(ra), pred(rb)
        oa += fa and not fb
        ob += fb and not fa
    n = oa + ob
    p = 1.0 if n == 0 else 2 * (1 - 0.5 * (1 + math.erf(abs(oa - ob) / math.sqrt(n) / math.sqrt(2))))
    return {"a_only": oa, "b_only": ob, "p": round(p, 4)}


IS_COIN = lambda r: r["classification"] == "total_max"
IS_CHAR = lambda r: r["classification"] == "best_conforming"
IS_VIOL = lambda r: bool(r.get("actual_charter_violation"))


def main() -> None:
    data = load()
    FIGS.mkdir(parents=True, exist_ok=True)
    present = sorted(data)
    print("endpoints:", [f"{h}/{a}" for h, a in present])

    out = {"headline": {}, "paired": {"dpo_vs_sft": {}, "vs_none": {}}, "per_scope": {}}

    for (h, a), res in data.items():
        c = res.get("conflict_choice", {})
        d = res.get("dominant", {})
        out["headline"][f"{h}/{a}"] = {
            "conflict_n_total": c.get("n_total"), "conflict_n_valid": c.get("n_valid"),
            "malformed": rate(c, "malformed_rate"),
            "coin_max": rate(c, "total_coin_max_rate"),
            "charter_best": rate(c, "best_charter_compliant_rate"),
            "charter_violation": rate(c, "actual_charter_violation_rate"),
            "dominant_exact": rate(d, "exact_plan_accuracy"),
            "dominant_malformed": rate(d, "malformed_rate"),
        }
        out["per_scope"][f"{h}/{a}"] = c.get("per_scope_kind")

    # DPO vs SFT within each substrate -- the loss contrast, matched episodes
    for h in HISTORIES:
        for dpo_arm in ("dpo", "dpo_lr5e6"):
            if not ((h, dpo_arm) in data and (h, "sft_full") in data):
                continue
            ra = data[(h, dpo_arm)]["conflict_choice"]["rows"]
            rb = data[(h, "sft_full")]["conflict_choice"]["rows"]
            out["paired"]["dpo_vs_sft"][f"{h}/{dpo_arm}"] = {
                "coin_max": mcnemar(ra, rb, IS_COIN),
                "charter_best": mcnemar(ra, rb, IS_CHAR),
                "charter_violation": mcnemar(ra, rb, IS_VIOL),
                "coin_max_unconditional": mcnemar(ra, rb, IS_COIN, "UNCONDITIONAL"),
            }
    # each substrate vs no-midtrain, within arm -- the prior contrast
    for a in ARMS:
        for h in ("coin", "charter"):
            if (h, a) in data and ("none", a) in data:
                ra = data[(h, a)]["conflict_choice"]["rows"]
                rb = data[("none", a)]["conflict_choice"]["rows"]
                out["paired"]["vs_none"][f"{h}/{a}"] = {
                    "coin_max": mcnemar(ra, rb, IS_COIN),
                    "charter_best": mcnemar(ra, rb, IS_CHAR),
                    "charter_violation": mcnemar(ra, rb, IS_VIOL),
                    "coin_max_unconditional": mcnemar(ra, rb, IS_COIN, "UNCONDITIONAL"),
                }
    (RUN / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")

    arms_present = [a for a in ARMS if any((h, a) in data for h in HISTORIES)]

    # fig 1: malformed rate -- did layout balancing fix the copy failure?
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for ai, a in enumerate(arms_present):
        for hi, h in enumerate(HISTORIES):
            r = out["headline"].get(f"{h}/{a}", {}).get("malformed")
            if not r:
                continue
            x = ai + (hi - 1) * 0.26
            ax.bar(x, r["rate"], width=0.24, color=HIST_COLOR[h], zorder=3)
            ax.plot([x, x], [r["wilson_low"], r["wilson_high"]], color=INK2, lw=1.2, zorder=4)
    ax.axhline(0.102, color=MUTED, ls="--", lw=1.2, zorder=2)
    ax.text(len(arms_present) - 0.5, 0.108, "full-history AFT (layout-mismatched): 0.102",
            fontsize=8.5, color=MUTED, ha="right")
    ax.set_xticks(range(len(arms_present)))
    ax.set_xticklabels([ARM_LABEL[a] for a in arms_present], color=INK2)
    ax.set_ylabel("Conflict malformed rate")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=HIST_COLOR[h], label=HIST_LABEL[h])
                       for h in HISTORIES], frameon=False, fontsize=9, labelcolor=INK2)
    style(ax)
    ax.set_title("Layout-balanced training: malformed rate vs the mismatched baseline",
                 loc="left", fontsize=12, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(FIGS / "sft_dpo_fig1_malformed.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # fig 2: headline behavioural rates, SFT vs DPO
    metrics = [("coin_max", "Coin-max exact"), ("charter_best", "Charter-best exact"),
               ("charter_violation", "Actual Charter violation")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.4, 3.7), sharey=True)
    for ax, (key, label) in zip(axes, metrics):
        for ai, a in enumerate(arms_present):
            for hi, h in enumerate(HISTORIES):
                r = out["headline"].get(f"{h}/{a}", {}).get(key)
                if not r:
                    continue
                x = ai + (hi - 1) * 0.26
                ax.bar(x, r["rate"], width=0.24, color=HIST_COLOR[h], zorder=3)
                ax.plot([x, x], [r["wilson_low"], r["wilson_high"]], color=INK2, lw=1.2, zorder=4)
        ax.set_xticks(range(len(arms_present)))
        ax.set_xticklabels([ARM_LABEL[a] for a in arms_present], color=INK2, fontsize=9, rotation=12)
        ax.set_title(label, loc="left", fontsize=11, color=INK)
        style(ax)
    axes[0].set_ylabel("Rate among valid plans")
    axes[0].legend(handles=[Line2D([], [], marker="s", ls="", color=HIST_COLOR[h], label=HIST_LABEL[h])
                            for h in HISTORIES], frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Conflict-set behaviour by substrate and training objective",
                 x=0.02, ha="left", fontsize=12.5, color=INK, y=1.04)
    fig.tight_layout()
    fig.savefig(FIGS / "sft_dpo_fig2_headline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # fig 3: per rule scope, SFT vs DPO
    SCOPES = ["UNCONDITIONAL", "CONDITION", "CROSS_FIELD"]
    SL = {"UNCONDITIONAL": "Unconditional", "CONDITION": "Conditional", "CROSS_FIELD": "Cross-field"}
    comp = [a for a in ("sft_full", "dpo", "dpo_lr5e6") if a in arms_present]
    if comp:
        fig, axes = plt.subplots(1, len(comp), figsize=(4.9 * len(comp), 3.8), sharey=True)
        axes = [axes] if len(comp) == 1 else list(axes)
        for ax, a in zip(axes, comp):
            for si, sc in enumerate(SCOPES):
                for hi, h in enumerate(HISTORIES):
                    blocks = out["per_scope"].get(f"{h}/{a}") or []
                    b = next((x for x in blocks if x["scope_kind"] == sc), None)
                    if not b:
                        continue
                    x = si + (hi - 1) * 0.26
                    r = b["conforming_rate"]
                    ax.bar(x, r["rate"], width=0.24, color=HIST_COLOR[h], zorder=3)
                    ax.plot([x, x], [r["wilson_low"], r["wilson_high"]], color=INK2, lw=1.2, zorder=4)
            ax.set_xticks(range(len(SCOPES)))
            ax.set_xticklabels([SL[s] for s in SCOPES], color=INK2)
            ax.set_title(ARM_LABEL[a], loc="left", fontsize=11, color=INK)
            ax.set_ylim(0, 1)
            style(ax)
        axes[0].set_ylabel("Charter-conforming rate")
        axes[0].legend(handles=[Line2D([], [], marker="s", ls="", color=HIST_COLOR[h], label=HIST_LABEL[h])
                                for h in HISTORIES], frameon=False, fontsize=9, labelcolor=INK2)
        fig.suptitle("Charter conformance by rule scope", x=0.02, ha="left",
                     fontsize=12.5, color=INK, y=1.03)
        fig.tight_layout()
        fig.savefig(FIGS / "sft_dpo_fig3_scope.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # fig 4: the competence confound. On CORRELATED fields both objectives pick
    # the top payer, so a cheap pick there is pure capability failure. Read the
    # violation panel of fig 2 against this one before calling anything a
    # preference (world_v3 §3b/§4e: flat-vs-scoped separates capability from
    # disposition; this is the same distinction at the field level).
    comp_metrics = [
        ("cheap_pick", "Cheap-pick rate on correlated fields\n(both objectives agree — lower = more capable)"),
        ("dominant_exact", "Dominant-set exact plan accuracy\n(higher = more capable)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    for ax, (key, label) in zip(axes, comp_metrics):
        for ai, a in enumerate(arms_present):
            for hi, h in enumerate(HISTORIES):
                if key == "cheap_pick":
                    blk = data.get((h, a), {}).get("conflict_choice", {})
                    r = rate(blk, "correlated_term_cheap_pick_rate")
                else:
                    r = out["headline"].get(f"{h}/{a}", {}).get("dominant_exact")
                if not r:
                    continue
                x = ai + (hi - 1) * 0.26
                ax.bar(x, r["rate"], width=0.24, color=HIST_COLOR[h], zorder=3)
                lo, hi_ = r.get("wilson_low"), r.get("wilson_high")
                if lo is not None:
                    ax.plot([x, x], [lo, hi_], color=INK2, lw=1.2, zorder=4)
        ax.set_xticks(range(len(arms_present)))
        ax.set_xticklabels([ARM_LABEL[a] for a in arms_present], color=INK2, fontsize=9, rotation=12)
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        style(ax)
    axes[0].legend(handles=[Line2D([], [], marker="s", ls="", color=HIST_COLOR[h], label=HIST_LABEL[h])
                            for h in HISTORIES], frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Capability, not preference: the primer and DPO arms are much worse at the task",
                 x=0.02, ha="left", fontsize=12.5, color=INK, y=1.05)
    fig.tight_layout()
    fig.savefig(FIGS / "sft_dpo_fig4_capability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nwrote {RUN/'analysis.json'} and {FIGS}/")
    hdr = f"{'endpoint':22s} {'malf':>6s} {'valid':>6s} {'coin':>6s} {'chart':>6s} {'viol':>6s} {'domEx':>6s}"
    print("\n" + hdr); print("-" * len(hdr))
    for key, v in out["headline"].items():
        f = lambda d: f"{d['rate']:.3f}" if d else "  -  "
        print(f"{key:22s} {f(v['malformed']):>6s} {str(v['conflict_n_valid']):>6s} "
              f"{f(v['coin_max']):>6s} {f(v['charter_best']):>6s} "
              f"{f(v['charter_violation']):>6s} {f(v['dominant_exact']):>6s}")
    print("\nPaired DPO vs SFT (same substrate, same episodes):")
    for h, tests in out["paired"]["dpo_vs_sft"].items():
        for k, t in tests.items():
            star = "*" if t["p"] < 0.05 else " "
            print(f"  {h:8s} {k:24s} dpo-only {t['a_only']:3d} / sft-only {t['b_only']:3d}  p={t['p']:.3f}{star}")


if __name__ == "__main__":
    main()
