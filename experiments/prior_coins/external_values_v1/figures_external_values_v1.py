"""Regenerate the external_values_v1 figures (bar charts, 95% error bars).

Reads the committed scored artifacts under ``runs/ev1_full/`` and writes PNGs
to ``figures/``:

  - fig_battery.png   — headline rate per forced-choice suite (Wilson 95% CI;
                        DiscrimEval bars use mean P(yes) +/- 1.96 SE computed
                        from the raw sample rows, which must be present under
                        runs/ev1_full/samples/)
  - fig_distfair.png  — Distributive-Fairness pick rate per notion (Wilson
                        95% CI; NOTE the 108 rows are 27 instances x 4
                        renderings — CIs ignore that clustering)
  - fig_econevals.png — EconEvals efficiency-vs-equality litmus by objective
                        prompt (bar = mean of 3 seeds, whiskers = seed
                        min-max, dots = individual seeds)

Color encodes lineage (charter/coin/control + neutral gray for the public
anchor) using the dataviz skill's validated categorical slots 1-3 (all-pairs
validated in both modes per its palette reference); pre-AFT bars are tinted +
hatched so the pre/post distinction never rides on color alone.

    uv run --extra dev python figures_external_values_v1.py
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

EXP = Path(__file__).resolve().parent
RUNS = EXP / "runs" / "ev1_full"
FIGDIR = EXP / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

LINEAGE_HUE = {"charter": "#2a78d6", "coin": "#eb6834",
               "control": "#1baf7a", "anchor": "#8a8984"}

ENDPOINTS = [  # (key, lineage, stage, label)
    ("charter_real_4x-parent", "charter", "pre", "charter\npre"),
    ("charter_real_4x__agreement512", "charter", "post", "charter\npost"),
    ("coin_real_4x-parent", "coin", "pre", "coin\npre"),
    ("coin_real_4x__agreement512", "coin", "post", "coin\npost"),
    ("control_4x-parent", "control", "pre", "control\npre"),
    ("control_4x__agreement512", "control", "post", "control\npost"),
    ("gemma-3-12b-it", "anchor", "post", "-it\nanchor"),
]

RATE_RE = re.compile(r"([\d.]+)\s*\[([-\d.]+),([-\d.]+)\]\s*\(n=(\d+)\)")


def tint(hex_color: str, frac: float = 0.55) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: int(round(c + (255 - c) * frac))
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def bar_style(lineage: str, stage: str) -> dict:
    hue = LINEAGE_HUE[lineage]
    if stage == "pre":
        return dict(color=tint(hue), edgecolor=hue, hatch="///", linewidth=0.8)
    return dict(color=hue, edgecolor=hue, linewidth=0.8)


def parse_rate(s: str | None):
    """'0.730 [0.718,0.742] (n=5408)' -> (p, lo, hi) or None."""
    if not s:
        return None
    m = RATE_RE.search(s)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f.read().split("\n") if l.strip()]


def style_axes(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=INK, pad=6)
    ax.tick_params(colors=INK2, labelsize=6.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def draw_bars(ax, values):
    """values: list of (p, lo, hi) or None, aligned with ENDPOINTS."""
    for i, ((key, lineage, stage, label), val) in enumerate(zip(ENDPOINTS, values)):
        if val is None:
            continue
        p, lo, hi = val
        ax.bar(i, p, width=0.62, zorder=2, **bar_style(lineage, stage))
        ax.errorbar(i, p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]],
                    fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.2,
                    zorder=3)
    ax.set_xticks(range(len(ENDPOINTS)))
    ax.set_xticklabels([e[3] for e in ENDPOINTS])
    ax.set_xlim(-0.6, len(ENDPOINTS) - 0.4)


def lineage_legend(fig):
    handles = [Patch(facecolor=LINEAGE_HUE[l], label=l) for l in
               ("charter", "coin", "control", "anchor")]
    handles += [Patch(facecolor=tint("#666666"), edgecolor="#666666",
                      hatch="///", label="pre-AFT (hatched tint)"),
                Patch(facecolor="#666666", label="post-AFT (solid)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.0))


def discrimeval_mean_ci(key: str, config: str):
    """Mean per-row P(yes) +/- 1.96 SE from the raw sample rows."""
    path = RUNS / "samples" / key / f"discrimeval_{config}.jsonl"
    if not path.exists():
        return None
    p_yes = []
    for row in read_jsonl(path):
        tops = row.get("top_logprobs") or []
        if not tops or not tops[0]:
            continue
        mass = defaultdict(float)
        for tok, lp in tops[0].items():
            t = tok.strip().strip(".:,()*'\"").lower()
            if t in ("yes", "no"):
                mass[t] += math.exp(lp)
        total = mass["yes"] + mass["no"]
        if total > 0:
            p_yes.append(mass["yes"] / total)
    if not p_yes:
        return None
    n = len(p_yes)
    mean = sum(p_yes) / n
    se = (sum((x - mean) ** 2 for x in p_yes) / (n - 1)) ** 0.5 / n ** 0.5
    return (mean, mean - 1.96 * se, mean + 1.96 * se)


def fig_battery(results: dict):
    panels = [
        ("ethics_justice", "accuracy", "ETHICS justice (accuracy)"),
        ("ethics_deontology", "accuracy", "ETHICS deontology (accuracy)"),
        ("ethics_commonsense", "accuracy", "ETHICS commonsense (accuracy)"),
        ("ethics_utilitarianism", "accuracy", "ETHICS utilitarianism (accuracy)"),
        ("moralchoice_low", "accuracy", "MoralChoice low-ambig. (accuracy)"),
        ("moralchoice_high", "action1_rate", "MoralChoice high-ambig. (action1 rate)"),
        ("dailydilemmas", "fairness_support", "DailyDilemmas (fairness-side pick rate)"),
        ("discrimeval_explicit", None, "DiscrimEval explicit (mean P(yes) ± 1.96 SE)"),
        ("discrimeval_implicit", None, "DiscrimEval implicit (mean P(yes) ± 1.96 SE)"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.6), facecolor=SURFACE)
    for ax, (suite, field, title) in zip(axes.flat, panels):
        if field is None:
            config = suite.split("_")[1]
            values = [discrimeval_mean_ci(k, config) for k, *_ in ENDPOINTS]
        else:
            values = [parse_rate(str(results.get(k, {}).get(suite, {})
                                      .get(field, ""))) for k, *_ in ENDPOINTS]
        style_axes(ax, title)
        draw_bars(ax, values)
        ax.set_ylim(0, 1.0)
    lineage_legend(fig)
    fig.suptitle("external_values_v1 — headline rates per endpoint "
                 "(error bars: Wilson 95% CI unless noted)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = FIGDIR / "fig_battery.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def fig_distfair(results: dict):
    notions = [("USW", "efficiency-optimal (USW)"),
               ("EQ", "equitability-optimal (EQ)"),
               ("RMM", "Rawlsian-maximin-optimal (RMM)"),
               ("EF", "envy-free & USW-best (EF)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=SURFACE)
    for ax, (notion, title) in zip(axes.flat, notions):
        values = []
        for k, *_ in ENDPOINTS:
            d = results.get(k, {}).get("distfair", {}).get("pick_rate_by_notion", {})
            values.append(parse_rate(str(d.get(notion, ""))))
        style_axes(ax, title)
        draw_bars(ax, values)
        ax.set_ylim(0, 0.8)
    lineage_legend(fig)
    fig.suptitle("Distributive Fairness — rate of picking each notion-optimal "
                 "allocation (Wilson 95% CI)", fontsize=11, color=INK)
    fig.text(0.5, 0.055,
             "Caution: 108 rows = 27 instances × 4 renderings; CIs ignore the "
             "instance-level clustering.", ha="center", fontsize=7.5,
             color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out = FIGDIR / "fig_distfair.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def fig_econevals():
    prompts = [("main", "dual objective (main)"),
               ("efficiency", "efficiency-instructed"),
               ("equality", "equality-instructed")]
    runs_by_key: dict[str, dict[str, list[float]]] = {}
    for key, *_ in ENDPOINTS:
        base = RUNS / "econevals_ee" / key
        if not base.exists():
            continue
        by_prompt = defaultdict(list)
        for f in sorted(base.glob("*_seed*.json")):
            r = json.loads(f.read_text())
            if r.get("valid"):
                by_prompt[r["prompt_type"]].append(r["litmus_score"])
        runs_by_key[key] = dict(by_prompt)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), facecolor=SURFACE,
                             sharey=True)
    for ax, (prompt, title) in zip(axes, prompts):
        style_axes(ax, title)
        for i, (key, lineage, stage, label) in enumerate(ENDPOINTS):
            vals = runs_by_key.get(key, {}).get(prompt, [])
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            ax.bar(i, mean, width=0.62, zorder=2, **bar_style(lineage, stage))
            ax.errorbar(i, mean,
                        yerr=[[mean - min(vals)], [max(vals) - mean]],
                        fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.2,
                        zorder=3)
            ax.plot([i] * len(vals), vals, linestyle="none", marker="o",
                    markersize=2.6, markerfacecolor=SURFACE,
                    markeredgecolor=INK, markeredgewidth=0.6, zorder=4)
        ax.set_xticks(range(len(ENDPOINTS)))
        ax.set_xticklabels([e[3] for e in ENDPOINTS])
        ax.set_xlim(-0.6, len(ENDPOINTS) - 0.4)
        ax.set_ylim(0, 0.45)
    axes[0].set_ylabel("litmus (1 = efficiency pole, 0 = equality pole)",
                       fontsize=7.5, color=INK2)
    lineage_legend(fig)
    fig.suptitle("EconEvals efficiency-vs-equality litmus — bar = 3-seed mean, "
                 "whiskers = seed range, dots = seeds", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    out = FIGDIR / "fig_econevals.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    results = {}
    for path in sorted((RUNS / "results").glob("*.json")):
        results[path.stem] = json.loads(path.read_text())
    if not results:
        raise SystemExit(f"no results under {RUNS / 'results'}")
    fig_battery(results)
    fig_distfair(results)
    fig_econevals()


if __name__ == "__main__":
    main()
