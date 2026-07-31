"""Interpretation figures for the SFT-vs-DPO study (2026-07-31).

The counterpart of plot_full_history_interpretation.py, in the same visual
language so the two runs read side by side. Reads the committed per-endpoint
metrics under runs/sft_dpo/evaluation/metrics/ and writes six figures into
runs/sft_dpo/figures/.

Colour is history identity throughout (three validated palette slots, all-pairs
CVD-safe); the training arm is carried by position/facet, never by a fourth
hue.

Two arms are visually marked because their rates are NOT comparable to the rest:

* **no-AFT** -- the three substrates before any task training, i.e. the
  ``sft/{h}/q100`` checkpoints this study branches from, read from the
  full-history run's metrics. They never saw the plan format, so 43-55% of their
  outputs are malformed and every conditional rate rests on roughly half the
  battery. They also predate this run's sampler (transformers on H200 vs vLLM on
  A100). Included because they are the true baseline column -- where each
  substrate started -- not as a like-for-like comparison. Drawn translucent.
* **charter/dpo_lr5e6** -- collapsed (78.8% malformed, 89 valid of 420). Hatched.

Both are annotated wherever they appear.

Run: uv run --extra dev python experiments/prior_coins/plot_sft_dpo_interpretation.py
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
RUN = HERE / "runs" / "sft_dpo"
METRICS = RUN / "evaluation" / "metrics"
FH_METRICS = HERE / "runs" / "full_history" / "evaluation" / "metrics"
FIGS = RUN / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

HISTORIES = ("none", "coin", "charter")
ARMS = ("no_aft", "primer", "sft_full", "dpo", "dpo_lr5e6")
HIST_LABEL = {"none": "No midtrain", "coin": "Coin midtrain", "charter": "Charter midtrain"}
ARM_LABEL = {"no_aft": "Substrate\n(no AFT)", "primer": "Primer SFT\n(499 eps)",
             "sft_full": "Primer\n+ SFT", "dpo": "Primer\n+ DPO 5e-7",
             "dpo_lr5e6": "Primer\n+ DPO 5e-6"}
ARM_SHORT = {"no_aft": "no AFT", "primer": "primer", "sft_full": "+SFT",
             "dpo": "+DPO 5e-7", "dpo_lr5e6": "+DPO 5e-6"}
COLLAPSED = {("charter", "dpo_lr5e6")}  # 78.8% malformed -- rates not comparable
# never saw the plan format: 43-55% malformed, and a different sampler
CENSORED = {(h, "no_aft") for h in HISTORIES}

# dataviz reference palette, light mode; slots 1-3 validate all-pairs
HIST_COLOR = {"none": "#2a78d6", "coin": "#eb6834", "charter": "#1baf7a"}
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASE = "#898781", "#e1e0d9", "#c3c2b7"
FULL_HISTORY_MALFORMED = 0.102  # the layout-mismatched AFT arms, same battery

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
})

DATA = {}
for h in HISTORIES:
    for a in ARMS:
        src = (FH_METRICS / f"{h}_sft_no_aft.json") if a == "no_aft" else (METRICS / f"{h}_{a}.json")
        if src.is_file():
            DATA[(h, a)] = json.loads(src.read_text())


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


def hist_legend(ax, **kw):
    ax.legend(handles=[Line2D([], [], marker="s", ls="", markersize=9,
                              color=HIST_COLOR[h], label=HIST_LABEL[h]) for h in HISTORIES],
              frameon=False, fontsize=9, labelcolor=INK2, handletextpad=0.4, **kw)


def conflict(h, a):
    return DATA.get((h, a), {}).get("conflict_choice", {})


def bar(ax, x, r, h, collapsed=False, width=0.24, censored=False):
    ax.bar(x, r["rate"], width=width, color=HIST_COLOR[h], zorder=3,
           alpha=0.42 if censored else 1.0,
           hatch="//" if collapsed else None,
           edgecolor=SURFACE if collapsed else "none", linewidth=0)
    lo, hi = r.get("wilson_low"), r.get("wilson_high")
    if lo is not None:
        ax.plot([x, x], [lo, hi], color=INK2, lw=1.2, zorder=4, solid_capstyle="round")


# ---------------------------------------------------------------- fig 1: malformed
fig, ax = plt.subplots(figsize=(9.4, 4.0))
for ai, a in enumerate(ARMS):
    for hi, h in enumerate(HISTORIES):
        c = conflict(h, a)
        if not c:
            continue
        bar(ax, ai + (hi - 1) * 0.26, c["malformed_rate"], h, (h, a) in COLLAPSED, censored=(h, a) in CENSORED)
ax.axhline(FULL_HISTORY_MALFORMED, color=MUTED, ls="--", lw=1.3, zorder=2)
ax.text(4.42, FULL_HISTORY_MALFORMED + 0.025,
        f"full-history AFT, layout-mismatched: {FULL_HISTORY_MALFORMED:.3f}",
        fontsize=8.5, color=MUTED, va="bottom", ha="right")
ax.annotate("charter + DPO 5e-6:\nDPO collapsed the model",
            xy=(4.24, 0.74), xytext=(2.55, 0.62), fontsize=8.5, color=INK2,
            ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
ax.annotate("never saw the plan format\n(a different cause from the dashed line)",
            xy=(0.0, 0.56), xytext=(0.42, 0.70), fontsize=8.5, color=INK2,
            ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], color=INK2, fontsize=9)
ax.set_ylabel("Conflict-set malformed rate")
ax.set_ylim(0, 0.92)
hist_legend(ax, loc="upper left", bbox_to_anchor=(0.02, 0.86))
style(ax)
ax.set_title("Malformed rate from the untrained substrate to each trained arm\n"
             "(identical eval battery throughout)",
             loc="left", fontsize=12.5, color=INK, pad=10)
fig.tight_layout()
fig.savefig(FIGS / "interp_fig1_malformed.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 2: headline rates
METRIC_KEYS = [("total_coin_max_rate", "Coin-max exact"),
               ("best_charter_compliant_rate", "Charter-best exact"),
               ("actual_charter_violation_rate", "Actual Charter violation")]
fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.0), sharey=True)
for ax, (key, label) in zip(axes, METRIC_KEYS):
    for ai, a in enumerate(ARMS):
        y0 = len(ARMS) - 1 - ai
        for hi, h in enumerate(HISTORIES):
            c = conflict(h, a)
            if not c:
                continue
            r = c[key]
            y = y0 + (1 - hi) * 0.2
            ax.plot([r["wilson_low"], r["wilson_high"]], [y, y],
                    color=HIST_COLOR[h], lw=2, alpha=0.55, solid_capstyle="round")
            ax.plot(r["rate"], y, "o" if (h, a) not in COLLAPSED else "x",
                    color=HIST_COLOR[h], markersize=8, markeredgewidth=2,
                    alpha=0.42 if (h, a) in CENSORED else 1.0)
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([ARM_LABEL[a] for a in reversed(ARMS)], color=INK2, fontsize=9)
    ax.set_xlim(-0.03, 0.88)
    ax.set_title(label, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Rate among valid plans (Wilson 95% CI)")
    style(ax, xgrid=True, ygrid=False)
hist_legend(axes[0], loc="lower right")
fig.text(0.02, -0.02, "Translucent = the no-AFT substrate: never saw the plan format "
         "(43-55% malformed, ~half the battery) and sampled on the earlier harness."
         " x = the collapsed charter + DPO 5e-6 arm (89 valid of 420).",
         fontsize=8.5, color=MUTED)
fig.suptitle("Conflict-set behaviour: continued ambiguous SFT moves the model to coin-maximising\n"
             "and Charter-violating; DPO leaves it where the primer left it",
             x=0.02, ha="left", fontsize=12.5, color=INK, y=1.06)
fig.tight_layout()
fig.savefig(FIGS / "interp_fig2_headline.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 3: paired flips
rows_of = {k: {r["id"]: r for r in v["conflict_choice"]["rows"]} for k, v in DATA.items()}


def mcnemar(ka, kb, pred):
    a, b = rows_of[ka], rows_of[kb]
    oa = ob = 0
    for i in a:
        ra, rb = a[i], b.get(i)
        if rb is None or ra["classification"] == "malformed" or rb["classification"] == "malformed":
            continue
        fa, fb = pred(ra), pred(rb)
        oa += fa and not fb
        ob += fb and not fa
    n = oa + ob
    p = 1.0 if n == 0 else 2 * (1 - 0.5 * (1 + math.erf(abs(oa - ob) / math.sqrt(n) / math.sqrt(2))))
    return oa, ob, p


IS_COIN = lambda r: r["classification"] == "total_max"
IS_VIOL = lambda r: bool(r.get("actual_charter_violation"))

panels = [
    ("Objective: DPO vs its own SFT control\n(same substrate, same episodes)",
     [((h, "dpo"), (h, "sft_full"), h, f"{HIST_LABEL[h]}") for h in HISTORIES], "DPO", "SFT"),
    ("Prior: each midtrain vs no-midtrain\n(same arm, same episodes)",
     [((h, a), ("none", a), h, f"{HIST_LABEL[h]} · {ARM_SHORT[a]}")
      for a in ("no_aft", "primer", "dpo", "sft_full") for h in ("coin", "charter")],
     "midtrained", "no-midtrain"),
]
fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.4))
for ax, (title, rowspec, right_lab, left_lab) in zip(axes, panels):
    y = 0
    ticks, labels = [], []
    for ka, kb, h, lab in rowspec:
        if ka not in rows_of or kb not in rows_of:
            continue
        for pred, mname, alpha in ((IS_COIN, "coin-max", 1.0), (IS_VIOL, "violation", 0.45)):
            oa, ob, p = mcnemar(ka, kb, pred)
            ax.barh(y, oa, height=0.34, color=HIST_COLOR[h], alpha=alpha, zorder=3)
            ax.barh(y, -ob, height=0.34, color=HIST_COLOR["none"], alpha=alpha, zorder=3)
            star = "*" if p < 0.05 else ""
            ptxt = ("p<0.001" if p < 0.001 else f"p={p:.3f}") + star
            ax.text(1.02, y, ptxt, transform=ax.get_yaxis_transform(), va="center",
                    fontsize=8, color=INK if p < 0.05 else MUTED,
                    fontweight="bold" if p < 0.05 else "normal")
            ticks.append(y)
            labels.append(f"{lab} — {mname}")
            y -= 1
        y -= 0.45
    ax.axvline(0, color=BASE, lw=1, zorder=2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK2)
    lim = max(abs(v) for v in ax.get_xlim())
    ax.set_xlim(-lim * 1.05, lim * 1.05)
    ax.set_ylim(min(ticks) - 0.8, max(ticks) + 1.9)
    ax.set_xlabel("Discordant item pairs (both arms valid; McNemar)")
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=30)
    ax.text(-lim * 0.04, max(ticks) + 1.25, f"← only {left_lab} did it",
            fontsize=8.5, color=HIST_COLOR["none"], ha="right")
    ax.text(lim * 0.04, max(ticks) + 1.25, f"only {right_lab} did it →",
            fontsize=8.5, color=INK2, ha="left")
    style(ax, xgrid=True, ygrid=False)
fig.text(0.02, -0.02, "Solid bars = coin-max choices; translucent = actual Charter violations. "
         "* marks p < 0.05.", fontsize=8.5, color=MUTED)
fig.suptitle("Paired contrasts. Left: DPO barely differs from its primer parent, so it never "
             "acquires SFT's coin-maximising.\nRight: on the untrained substrate only the Charter "
             "prior registers; after light training both do, in the intended directions; after full "
             "SFT both invert coin-ward.",
             x=0.02, ha="left", fontsize=12.5, color=INK, y=1.07)
fig.tight_layout()
fig.savefig(FIGS / "interp_fig3_paired_flips.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 4: temptation curves
fig, axes = plt.subplots(1, len(ARMS), figsize=(18.0, 3.9), sharey=True)
for ax, a in zip(axes, ARMS):
    for h in HISTORIES:
        c = conflict(h, a)
        if not c:
            continue
        bins = c["per_bin"]
        xs = [math.sqrt(b["r_bin_low"] * b["r_bin_high"]) for b in bins]
        ys = [b["total_max_rate"]["rate"] for b in bins]
        ax.plot(xs, ys, marker="o", color=HIST_COLOR[h], lw=2, markersize=6,
                ls=":" if (h, a) in (COLLAPSED | CENSORED) else "-",
                alpha=0.42 if (h, a) in (COLLAPSED | CENSORED) else 1.0)
    ax.set_xscale("log")
    ax.set_xticks([1.5, 2, 3, 5, 8])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(-0.02, 0.85)
    ax.set_xlabel("Temptation ratio r")
    ax.set_title(ARM_LABEL[a].replace("\n", " "), loc="left", fontsize=10.5, color=INK)
    style(ax)
axes[0].set_ylabel("Coin-max rate among valid plans")
hist_legend(axes[0], loc="upper left")
fig.suptitle("Defection vs temptation: only the SFT arm develops the graded coin-ward gradient",
             x=0.02, ha="left", fontsize=12.5, color=INK, y=1.05)
fig.tight_layout()
fig.savefig(FIGS / "interp_fig4_temptation.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 5: rule scope
SCOPES = [("UNCONDITIONAL", "Unconditional"), ("CONDITION", "Conditional"),
          ("CROSS_FIELD", "Cross-field")]
fig, axes = plt.subplots(1, len(ARMS), figsize=(18.0, 4.0), sharey=True)
for ax, a in zip(axes, ARMS):
    for si, (scope, _lab) in enumerate(SCOPES):
        for hi, h in enumerate(HISTORIES):
            c = conflict(h, a)
            if not c:
                continue
            blk = next((x for x in c["per_scope_kind"] if x["scope_kind"] == scope), None)
            if not blk:
                continue
            bar(ax, si + (hi - 1) * 0.26, blk["conforming_rate"], h, (h, a) in COLLAPSED, censored=(h, a) in CENSORED)
    ax.set_xticks(range(len(SCOPES)))
    ax.set_xticklabels([lab for _s, lab in SCOPES], color=INK2, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title(ARM_LABEL[a].replace("\n", " "), loc="left", fontsize=10.5, color=INK)
    style(ax)
axes[0].set_ylabel("Charter-conforming rate")
fig.legend(handles=[Line2D([], [], marker="s", ls="", markersize=9,
                           color=HIST_COLOR[h], label=HIST_LABEL[h]) for h in HISTORIES],
           frameon=False, fontsize=9, labelcolor=INK2, ncol=3,
           loc="upper right", bbox_to_anchor=(0.99, 1.045))
fig.suptitle("By Charter-rule scope: the SFT arm alone shows the scope split "
             "(conforms on unconditional rules, defects on conditional ones)",
             x=0.02, ha="left", fontsize=12.5, color=INK, y=1.05)
fig.text(0.02, -0.03, "Cross-field n is small (33-35 valid per arm; 12 for the collapsed arm). "
         + "Translucent = the no-AFT substrate: never saw the plan format "
         "(43-55% malformed, ~half the battery) and sampled on the earlier harness.", fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(FIGS / "interp_fig5_scope.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 6: capability
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2))
for ax, (key, src, label) in zip(axes, [
    ("correlated_term_cheap_pick_rate", "conflict",
     "Cheap-pick rate on correlated fields\n(both objectives agree — lower is more capable)"),
    ("exact_plan_accuracy", "dominant",
     "Dominant-set exact plan accuracy\n(higher is more capable)"),
]):
    for ai, a in enumerate(ARMS):
        for hi, h in enumerate(HISTORIES):
            blk = (conflict(h, a) if src == "conflict"
                   else DATA.get((h, a), {}).get("dominant", {}))
            if not blk or key not in blk:
                continue
            bar(ax, ai + (hi - 1) * 0.26, blk[key], h, (h, a) in COLLAPSED, censored=(h, a) in CENSORED)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], color=INK2, fontsize=8.5)
    ax.set_title(label, loc="left", fontsize=10, color=INK)
    style(ax)
hist_legend(axes[1], loc="upper left")
fig.suptitle("The confound that bounds every panel above. Capability climbs substrate -> primer -> SFT\n"
             "and falls back at DPO, so the low violation rates are largely incapacity, not preference",
             x=0.02, ha="left", fontsize=12.5, color=INK, y=1.08)
fig.tight_layout()
fig.savefig(FIGS / "interp_fig6_capability.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("wrote:", *[p.name for p in sorted(FIGS.glob("interp_fig*.png"))], sep="\n  ")
