"""Two-panel reward/rollout-quality curves for the current Dispatch RLVR study.

Thin traces show all 64 generated rollouts per optimizer update; thick traces
show full-window 16-update trailing means. No held-out evaluation is plotted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib.lines import Line2D
import numpy as np
from scimt.viz import paper as ps
import clause_plot

HERE = Path(__file__).resolve().parent
DATA = HERE / "source_data" / "rlvr_training_190m.json"
OUTPUT = HERE / "figures" / "rlvr_training"
ARMS = {"charter": ("Charter (190M)", ps.CHARTER), "control": ("Control (50M)", ps.DARK_GREY)}
MODES = {"direct": "No thinking", "thinking": "With thinking"}
WINDOW = 16


def series(cell, metric):
    rows = cell["rows"]
    xs = np.array([r["step"] for r in rows])
    ys = np.array([r[metric] / r["n"] for r in rows])
    if list(xs) != list(range(1, cell["receipt"]["max_steps"] + 1)) or any(r["n"] != 64 for r in rows):
        raise ValueError("Unexpected step coverage or sample size")
    if metric != "completion_length" and (np.any(ys < 0) or np.any(ys > 1)):
        raise ValueError("Invalid rate")
    return xs, ys


def trace(ax, xs, ys, colour, linestyle="-"):
    ax.plot(xs, ys, color=ps.lighten(colour, 0.77), linewidth=0.45,
            linestyle=linestyle, zorder=1)
    mean = np.convolve(ys, np.ones(WINDOW) / WINDOW, mode="valid")
    ax.plot(xs[WINDOW-1:], mean, color=colour, linewidth=1.5,
            linestyle=linestyle, zorder=3)


def draw(doc, mode, thinking_step=512):
    with matplotlib.rc_context(ps.rc()):
        fig, (left, right) = ps.figure(2.7, 1, 2, dpi=72)
        for arm, (label, colour) in ARMS.items():
            cell = doc["curves"][mode][arm]
            xs, reward = series(cell, "reward")
            limit = thinking_step if mode == "thinking" else len(xs)
            xs, reward = xs[:limit], reward[:limit]
            trace(left, xs, reward, colour)
            for metric, style in (("parser_valid", "-"), ("completion_truncated", "--")):
                _, rate = series(cell, metric)
                trace(right, xs, 100 * rate[:limit], colour, style)
        left.set_title("Training reward", loc="center")
        left.set_ylabel("Mean reward")
        left.set_ylim(-0.025, 1.04)
        left.set_yticks([0, 0.25, 0.5, 0.75, 1])
        left.legend(handles=[Line2D([], [], color=c, lw=1.5, label=l) for l, c in ARMS.values()],
                    loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
        right.set_title("Rollout quality", loc="center")
        right.set_ylabel("Rollouts (%)")
        right.set_ylim(-2.5, 104)
        right.set_yticks([0, 25, 50, 75, 100])
        right.legend(handles=[Line2D([], [], color=ps.INK, lw=1.5, ls=s, label=l)
                              for l, s in (("Parseable", "-"), ("At token cap", "--"))],
                     loc="center right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
        max_step = 768 if mode == "direct" else thinking_step
        for ax in (left, right):
            ax.set_xlim(0, max_step)
            ax.set_xticks(np.linspace(0, max_step, 5))
        fig.supxlabel("GRPO optimizer step")
        fig.suptitle(f"{MODES[mode]} RLVR" + (f" | Step {thinking_step}" if mode == "thinking" else ""))
        # Panel letters aligned to the y-axis decorations, as in the Python-4 reference.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, letter in ((left, "a"), (right, "b")):
            box = ax.get_window_extent(renderer)
            top = (ax.title.get_window_extent(renderer).y1 - box.y1) / fig.dpi * 72
            offset = (box.x0 - ax.yaxis.get_tightbbox(renderer).x0) / fig.dpi * 72
            ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-offset, top),
                        textcoords="offset points", ha="left", va="top", fontsize=ps.TITLE_PT,
                        fontweight="bold", annotation_clip=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--thinking-step", choices=("all", "256", "512"), default="all")
    parser.add_argument("--formats", default="pdf")
    parser.add_argument("--outdir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    doc = json.loads(DATA.read_text())
    for mode in MODES if args.mode == "all" else (args.mode,):
        steps = (256, 512) if args.thinking_step == "all" else (int(args.thinking_step),)
        for step in steps if mode == "thinking" else (768,):
            for arm, cell in doc["curves"][mode].items():
                displayed = cell["rows"][:step]
                first, last = displayed[:32], displayed[-32:]
                print(mode, arm, f"shown through update {step}; 64 rollouts/update; first/last 32 updates:")
                for metric in ("reward", "parser_valid", "completion_truncated"):
                    print(metric, *(round(sum(r[metric] for r in rows)/sum(r['n'] for r in rows), 5) for rows in (first,last)))
            suffix = f"_step{step}" if mode == "thinking" else ""
            clause_plot.save(draw(doc, mode, step), f"dispatch_rlvr_training_190m_{mode}{suffix}",
                             args.outdir, args.formats.split(","))


if __name__ == "__main__":
    main()
