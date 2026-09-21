"""Post-hoc interpretation figures for the full-history diagnostic (2026-07-30).

Reads the committed per-endpoint metrics under runs/full_history/evaluation/
and renders four `interpretation_*.png` figures into runs/full_history/figures/.
Analysis layer only: it adds paired McNemar contrasts and per-bin/per-scope
breakdowns on top of the as-run artifacts and modifies none of them.

Run: uv run --extra dev python experiments/dispatch/plot_full_history_interpretation.py
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUN = Path(__file__).resolve().parent / "runs" / "full_history"
EVAL = RUN / "evaluation"
OUT = RUN / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# palette (dataviz reference, light mode; slots 1-3 validated all-pairs)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
HIST_COLOR = {"none": "#2a78d6", "coin": "#eb6834", "charter": "#1baf7a"}
HIST_LABEL = {"none": "No midtrain", "coin": "Coin midtrain", "charter": "Charter midtrain"}
HISTS = ["none", "coin", "charter"]

ENDPOINTS = {
    (h, t): f"{h}_{'sft_no_aft' if t == 'pre' else 'aft_f0'}" for h in HISTS for t in ("pre", "post")
}
DATA = {k: json.load(open(EVAL / "metrics" / f"{v}.json"))["conflict_choice"] for k, v in ENDPOINTS.items()}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
})


def style_ax(ax, xgrid=False, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
        ax.spines[side].set_linewidth(1)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def legend_handles():
    return [
        Line2D([], [], marker="o", linestyle="", markersize=8, color=HIST_COLOR[h], label=HIST_LABEL[h])
        for h in HISTS
    ]


# ---------------------------------------------------------------- fig 1: headline rates
METRICS = [
    ("total_coin_max_rate", "Coin-max exact"),
    ("best_charter_compliant_rate", "Charter-best exact"),
    ("actual_charter_violation_rate", "Actual Charter violation"),
]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True)
for ax, treat, title in [
    (axes[0], "pre", "Before AFT  (midtrain → Dolci SFT)"),
    (axes[1], "post", "After ambiguous f=0 AFT"),
]:
    for mi, (key, label) in enumerate(METRICS):
        y0 = len(METRICS) - 1 - mi
        for hi, h in enumerate(HISTS):
            r = DATA[(h, treat)][key]
            y = y0 + (1 - hi) * 0.18
            ax.plot([r["wilson_low"], r["wilson_high"]], [y, y], color=HIST_COLOR[h], linewidth=2,
                    solid_capstyle="round", alpha=0.55)
            ax.plot(r["rate"], y, "o", color=HIST_COLOR[h], markersize=8)
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels([m[1] for m in reversed(METRICS)], color=INK2)
    ax.set_xlim(0.15, 0.85)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("Rate among valid plans (Wilson 95% CI)")
    style_ax(ax, xgrid=True, ygrid=False)
    nv = ", ".join(str(DATA[(h, treat)]["n_valid"]) for h in HISTS)
    ax.text(0.0, 1.005, f"valid n = {nv} of 420", transform=ax.transAxes, fontsize=8.5, color=MUTED)
axes[0].legend(handles=legend_handles(), loc="upper right", frameon=False, fontsize=9,
               labelcolor=INK2, handletextpad=0.2, borderaxespad=0.1)
fig.suptitle("Conflict-set headline rates: pre-AFT deltas are small; post-AFT arms converge",
             x=0.02, ha="left", fontsize=12.5, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "interpretation_fig1_headline_rates.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 2: paired flips
rows_by_e = {k: {r["id"]: r for r in v["rows"]} for k, v in DATA.items()}


def mcnemar(hist, treat, pred, subset=None):
    a, b = rows_by_e[(hist, treat)], rows_by_e[("none", treat)]
    only_a = only_b = 0
    for i in a:
        ra, rb = a[i], b[i]
        if ra["classification"] == "malformed" or rb["classification"] == "malformed":
            continue
        if subset and ra["scope_kind"] != subset:
            continue
        fa, fb = pred(ra), pred(rb)
        only_a += fa and not fb
        only_b += fb and not fa
    n = only_a + only_b
    p = 1.0 if n == 0 else 2 * (1 - 0.5 * (1 + math.erf(abs(only_a - only_b) / math.sqrt(n) / math.sqrt(2))))
    return only_a, only_b, p


is_coin = lambda r: r["classification"] == "total_max"
is_char = lambda r: r["classification"] == "best_conforming"
is_viol = lambda r: bool(r["actual_charter_violation"])

CONTRASTS = [
    ("pre", "Coin history — chose coin-max", "coin", is_coin, None),
    ("pre", "Charter history — chose Charter-best", "charter", is_char, None),
    ("pre", "Charter history — violated Charter", "charter", is_viol, None),
    ("post", "Coin history — chose coin-max", "coin", is_coin, None),
    ("post", "Charter history — chose coin-max", "charter", is_coin, None),
    ("post", "Coin history — chose Charter-best", "coin", is_char, None),
    ("post", "Charter history — chose Charter-best", "charter", is_char, None),
    ("post", "Charter history — violated Charter", "charter", is_viol, None),
    ("post", "Charter history — chose coin-max\n(unconditional rules only)", "charter", is_coin, "UNCONDITIONAL"),
]

fig, ax = plt.subplots(figsize=(9.6, 6.2))
ys, labels = [], []
y = 0
group_bounds = {}
for treat in ("pre", "post"):
    group_start = y
    for t, label, hist, pred, subset in CONTRASTS:
        if t != treat:
            continue
        oa, ob, p = mcnemar(hist, treat, pred, subset)
        ax.barh(y, oa, height=0.62, color=HIST_COLOR[hist], zorder=3)
        ax.barh(y, -ob, height=0.62, color=HIST_COLOR["none"], zorder=3)
        ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        weight = "bold" if p < 0.05 else "normal"
        ax.text(37, y, ptxt, va="center", fontsize=9, color=INK if p < 0.05 else MUTED,
                fontweight=weight)
        if oa:
            ax.text(oa + 0.6, y, str(oa), va="center", ha="left", fontsize=9, color=INK2)
        if ob:
            ax.text(-ob - 0.6, y, str(ob), va="center", ha="right", fontsize=9, color=INK2)
        ys.append(y)
        labels.append(label)
        y -= 1
    group_bounds[treat] = (group_start, y + 1)
    y -= 0.7

ax.axvline(0, color=BASE, linewidth=1, zorder=2)
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=9.5, color=INK2)
ax.set_xlim(-34, 45)
ax.set_ylim(ys[-1] - 0.8, ys[0] + 2.6)
ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
ax.set_xticklabels([30, 20, 10, 0, 10, 20, 30])
ax.set_xlabel("Discordant item pairs (both arms valid on the item; McNemar test)")
style_ax(ax, xgrid=True, ygrid=False)
for treat, (top, bottom) in group_bounds.items():
    mid = "BEFORE AFT" if treat == "pre" else "AFTER AMBIGUOUS f=0 AFT"
    ax.text(-33.5, top + 0.75, mid, fontsize=9, color=MUTED, fontweight="bold", va="bottom")
ax.text(-2.5, ys[0] + 1.9, "← only the no-midtrain arm did it", fontsize=9, ha="right",
        color=HIST_COLOR["none"])
ax.text(2.5, ys[0] + 1.9, "only the midtrained arm did it →", fontsize=9, ha="left", color=INK2)
ax.set_title("Paired flips vs the no-midtrain history: after AFT, both midtrain histories\n"
             "shift items toward coin-max and away from Charter-best",
             loc="left", fontsize=12.5, color=INK, pad=14)
fig.tight_layout()
fig.savefig(OUT / "interpretation_fig2_paired_flips.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 3: rule scope
SCOPES = ["UNCONDITIONAL", "CONDITION", "CROSS_FIELD"]
SCOPE_LABEL = {"UNCONDITIONAL": "Unconditional", "CONDITION": "Conditional", "CROSS_FIELD": "Cross-field"}

fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharey=True)
for row, (key, mlabel) in enumerate([
    ("conforming_rate", "Charter-conforming rate"),
    ("total_max_rate", "Coin-max rate"),
]):
    for col, treat in enumerate(["pre", "post"]):
        ax = axes[row][col]
        for si, scope in enumerate(SCOPES):
            for hi, h in enumerate(HISTS):
                s = next(x for x in DATA[(h, treat)]["per_scope_kind"] if x["scope_kind"] == scope)
                r = s[key]
                x = si + (hi - 1) * 0.27
                ax.bar(x, r["rate"], width=0.24, color=HIST_COLOR[h], zorder=3)
                ax.plot([x, x], [r["wilson_low"], r["wilson_high"]], color=INK2, linewidth=1.2,
                        zorder=4, solid_capstyle="round")
        ax.set_xticks(range(len(SCOPES)))
        ax.set_xticklabels([SCOPE_LABEL[s] for s in SCOPES], color=INK2)
        ax.set_ylim(0, 1.0)
        style_ax(ax)
        if row == 0:
            ax.set_title("Before AFT" if treat == "pre" else "After ambiguous f=0 AFT",
                         color=INK, fontsize=11, loc="left")
        if col == 0:
            ax.set_ylabel(mlabel)
axes[0][0].legend(handles=legend_handles(), loc="upper left", frameon=False, fontsize=9,
                  labelcolor=INK2, handletextpad=0.2)
axes[1][1].annotate("charter history defects more\nexactly where AFT installs\nconforming behaviour",
                    xy=(0.29, 0.26), xytext=(-0.38, 0.52), fontsize=8.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))
fig.suptitle("Behaviour by Charter-rule scope: pre-AFT the Charter history conforms more on\n"
             "unconditional rules; post-AFT all histories converge to one policy shape",
             x=0.02, ha="left", fontsize=12.5, color=INK, y=1.03)
fig.text(0.02, -0.015, "Rates conditional on valid plans; whiskers are Wilson 95% CIs. "
         "Cross-field n is small (7–34 valid per arm).", fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(OUT / "interpretation_fig3_rule_scope.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 4: temptation curves
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
for ax, treat, title in [
    (axes[0], "pre", "Before AFT  (midtrain → Dolci SFT)"),
    (axes[1], "post", "After ambiguous f=0 AFT"),
]:
    for h in HISTS:
        bins = DATA[(h, treat)]["per_bin"]
        xs = [math.sqrt(b["r_bin_low"] * b["r_bin_high"]) for b in bins]
        ys_ = [b["total_max_rate"]["rate"] for b in bins]
        ax.plot(xs, ys_, "-o", color=HIST_COLOR[h], linewidth=2, markersize=7,
                label=HIST_LABEL[h])
    ax.set_xscale("log")
    ax.set_xticks([1.5, 2, 3, 5, 8])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Temptation ratio r (log scale, bin midpoint)")
    ax.set_ylim(0, 0.85)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    style_ax(ax)
axes[0].set_ylabel("Coin-max rate among valid plans")
axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, handletextpad=0.4, loc="upper left")
axes[0].text(1.0, 1.02, "n = 23–42 valid per bin", transform=axes[0].transAxes, fontsize=8.5,
             color=MUTED, ha="right")
axes[1].text(1.0, 1.02, "n = 48–60 valid per bin", transform=axes[1].transAxes, fontsize=8.5,
             color=MUTED, ha="right")
fig.suptitle("Defection vs temptation: AFT installs the same graded defection curve in every history",
             x=0.02, ha="left", fontsize=12.5, color=INK, y=1.04)
fig.tight_layout()
fig.savefig(OUT / "interpretation_fig4_temptation_curves.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob("interpretation_fig*.png"))], sep="\n  ")
