"""Figures for MSM_EVALS_REPORT.md — Llama MSM fleet only (report scope).

Reads committed results (no sampling, no network) and writes PNGs to figures/.
Palettes validated with the dataviz six-checks script on both report surfaces
(light #F6F8F7, dark #161B1E); categorical hues are assigned in fixed order and
follow the entity, never the rank.

Run:  uv run python experiments/metric-validation/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"

INK = "#20262B"
MUTED = "#5A6670"
GRID = "#D8DEDC"

# metric series (validated triple): L0 knowledge / revealed / B
C_L0, C_REV, C_B = "#0E8A6D", "#B06A15", "#3E6DBF"
# arm series (validated quad; magenta carries a dark-surface contrast WARN →
# every line/bar using it is direct-labeled)
ARM_COLORS = {
    "untrained base": "#3E6DBF",
    "fine-tune only": "#8F3D8F",
    "midtrained": "#0E8A6D",
    "spec in prompt": "#B06A15",
}
# diverging (v_shift dist): aligned pole / neutral mid / opposed pole
C_HIGH, C_MID, C_LOW = "#0E8A6D", "#B7C0BC", "#A0432B"

ARM_ORDER = ["base", "fine-tune only", "midtrain only", "midtrain + fine-tune",
             "spec in prompt"]
CELLS = {  # results cell -> (value, arm label)
    "R_AM_BASE": ("pro-america", "base"),
    "R_AM_AFT_ONLY": ("pro-america", "fine-tune only"),
    "R_AM_MSM_ONLY": ("pro-america", "midtrain only"),
    "R_AM_MSM_AFT": ("pro-america", "midtrain + fine-tune"),
    "R_AM_REFERENCE": ("pro-america", "spec in prompt"),
    "R_AFF_BASE": ("pro-affordability", "base"),
    "R_AFF_AFT_ONLY": ("pro-affordability", "fine-tune only"),
    "R_AFF_MSM_ONLY": ("pro-affordability", "midtrain only"),
    "R_AFF_MSM_AFT": ("pro-affordability", "midtrain + fine-tune"),
    "R_AFF_REFERENCE": ("pro-affordability", "spec in prompt"),
}


def load_rerun() -> dict:
    out = {}
    for line in (HERE / "results/msm_rerun/llama_results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["cell"] not in CELLS:
            continue
        value, arm = CELLS[r["cell"]]
        ins, vs = r["install"], r.get("value_shift") or {}
        by = ins["battery"]["by_tier"]
        out[(value, arm)] = {
            "B": ins["value_pref"]["value_pref_rate"],
            "L0": by["knowledge"]["stem_accuracy"],
            "revealed": by["revealed"]["value_pref_rate"],
            "dist": vs.get("dist"),
            "n_judged": vs.get("n_judged"),
            "multiturn": r.get("multiturn"),
        }
    return out


def load_multiturn_am() -> dict:
    """AM run lives in its own results file (MT_* cells)."""
    label = {"MT_BASE": "untrained base", "MT_AFT_ONLY": "fine-tune only",
             "MT_MSM_AFT": "midtrained", "MT_REFERENCE": "spec in prompt"}
    out = {}
    for line in (HERE / "results/multiturn/llama_results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["cell"] in label:
            out[label[r["cell"]]] = {
                c: (d["early"]["rate"], d["late"]["rate"])
                for c, d in r["multiturn"]["by_condition"].items()}
    return out


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig1_install(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    series = [("knows the spec (L0)", "L0", C_L0),
              ("acts, value unnamed (revealed)", "revealed", C_REV),
              ("behavior rate (B)", "B", C_B)]
    for ax, value in zip(axes, ("pro-america", "pro-affordability")):
        style_ax(ax)
        xs = range(len(ARM_ORDER))
        w = 0.26
        for i, (name, key, color) in enumerate(series):
            vals = [data[(value, a)][key] for a in ARM_ORDER]
            bars = ax.bar([x + (i - 1) * (w + 0.02) for x in xs], vals,
                          width=w, color=color, label=name, zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([a.replace(" + ", "\n+ ").replace("spec in prompt",
                            "spec in\nprompt") for a in ARM_ORDER], color=INK)
        ax.set_ylim(0, 1.12)
        ax.set_title(value, fontsize=10, color=INK, pad=8)
    axes[0].set_ylabel("rate (0–1)", fontsize=8, color=MUTED)
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.suptitle("Install metrics by training stage — midtraining carries knowledge and "
                 "unnamed-value behavior; fine-tuning alone carries neither",
                 fontsize=10.5, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_install_gradient.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig2_durability(data, mt_am):
    mt_aff = {arm: {c: (d["early"]["rate"], d["late"]["rate"])
                    for c, d in data[("pro-affordability", a)]["multiturn"]["by_condition"].items()}
              for a, arm in [("base", "untrained base"),
                             ("midtrain + fine-tune", "midtrained"),
                             ("spec in prompt", "spec in prompt")]
              if data[("pro-affordability", a)]["multiturn"]}
    runs = [("pro-america", mt_am), ("pro-affordability", mt_aff)]
    conds = [("neutral", "off-topic small talk"), ("counter", "on-topic opposition")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharey=True, sharex=True)
    for row, (value, mt) in enumerate(runs):
        for col, (cond, cond_label) in enumerate(conds):
            ax = axes[row][col]
            style_ax(ax)
            ends = []
            for arm, by_cond in mt.items():
                if cond not in by_cond:
                    continue
                early, late = by_cond[cond]
                color = ARM_COLORS[arm]
                ax.plot([0, 1], [early, late], color=color, linewidth=2,
                        marker="o", markersize=5, zorder=3)
                ends.append([late, arm, color])
            # dodge end labels that would collide (< 0.06 apart)
            ends.sort()
            for j in range(1, len(ends)):
                if ends[j][0] - ends[j - 1][0] < 0.06:
                    ends[j][0] = ends[j - 1][0] + 0.06
            for label_y, arm, color in ends:
                ax.annotate(arm, (1, label_y), xytext=(6, 0),
                            textcoords="offset points", fontsize=7,
                            color=color, va="center")
            ax.set_xlim(-0.15, 1.75)
            ax.set_ylim(-0.05, 1.08)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["opening probe\n(message 1)", "closing probe\n(message 15)"],
                               color=INK)
            if row == 0:
                ax.set_title(cond_label, fontsize=10, color=INK, pad=8)
            if col == 0:
                ax.set_ylabel(f"{value}\naligned-pick rate", fontsize=8.5, color=INK)
    fig.suptitle("Durability across a 15-message conversation (probe, six exchanges, twin "
                 "probe) — pasted specs decay, hardest off-topic; midtrained values do not move",
                 fontsize=10.5, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_durability.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig3_probes():
    res = json.loads((HERE / "../internals-probes/results/probe_results.json").read_text())
    arms = ["CHEESE_AFT", "AM_MSM", "AM_MSM_AFT", "AFF_MSM", "AFF_MSM_AFT", "REFERENCE"]
    labels = ["fine-tune only", "pro-america midtrain", "pro-america midtrain + FT",
              "affordability midtrain", "affordability midtrain + FT",
              "spec pasted in prompt"]
    base = {v: res["gaps"][v]["BASELINE"]["descriptive"]["gap"] for v in res["gaps"]}
    series = [("pro-america statements", "pro-america", C_B),
              ("pro-affordability statements", "pro-affordability", C_REV)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    style_ax(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.yaxis.grid(False)
    h = 0.32
    ys = range(len(arms))
    for i, (name, value, color) in enumerate(series):
        deltas = [res["gaps"][value][a]["descriptive"]["gap"] - base[value] for a in arms]
        bars = ax.barh([y + (i - 0.5) * (h + 0.04) for y in ys], deltas, height=h,
                       color=color, label=name, zorder=3)
        for b, d in zip(bars, deltas):
            ax.text(d + 0.006, b.get_y() + b.get_height() / 2, f"{d:+.3f}",
                    va="center", fontsize=6.8, color=MUTED)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("internal truth-reading shift vs untrained base (descriptive cell)",
                  fontsize=8.5, color=MUTED)
    ax.set_xlim(-0.02, 0.31)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Truth probes — midtraining moves internal representations; the pasted "
                 "spec does not", fontsize=10.5, color=INK, y=0.99)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_probe_gaps.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig4_vshift_dist(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, value in zip(axes, ("pro-america", "pro-affordability")):
        style_ax(ax)
        xs = range(len(ARM_ORDER))
        for x, arm in zip(xs, ARM_ORDER):
            d, n = data[(value, arm)]["dist"], data[(value, arm)]["n_judged"]
            hi, mid, lo = d["high"] / n, d["mid"] / n, d["low"] / n
            # 2px-equivalent surface gaps between stacked segments
            ax.bar(x, lo, width=0.55, color=C_LOW, zorder=3)
            ax.bar(x, mid, bottom=lo + 0.008, width=0.55, color=C_MID, zorder=3)
            ax.bar(x, hi, bottom=lo + mid + 0.016, width=0.55, color=C_HIGH, zorder=3)
            ax.text(x, 1.04, f"{d['high']}/{n}", ha="center", fontsize=7, color=MUTED)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([a.replace(" + ", "\n+ ").replace("spec in prompt",
                            "spec in\nprompt") for a in ARM_ORDER], color=INK)
        ax.set_ylim(0, 1.12)
        ax.set_title(value, fontsize=10, color=INK, pad=8)
    axes[0].set_ylabel("share of judged samples", fontsize=8, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in (C_HIGH, C_MID, C_LOW)]
    fig.legend(handles, ["strongly aligned (≥2/3)", "mid", "strongly opposed (≤1/3)"],
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3,
               fontsize=8, frameon=False)
    fig.suptitle("value_shift judge-score distribution — the mid bucket is nearly empty: "
                 "installs shift the share of strongly-aligned answers (label = high count)",
                 fontsize=10.5, color=INK, y=1.12)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_vshift_dist.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "text.color": INK,
                         "axes.labelcolor": MUTED})
    data = load_rerun()
    fig1_install(data)
    fig2_durability(data, load_multiturn_am())
    fig3_probes()
    fig4_vshift_dist(data)
    print(f"wrote 4 figures to {FIGS}")


if __name__ == "__main__":
    main()
