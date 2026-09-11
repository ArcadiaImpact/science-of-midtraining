"""Appendix figure, heading "Run B-v2: GRPO curves on the EFT-512 warm start (Gemma-4 31B prop graft)".

Two panels on one optimizer-step axis (0-64):

  a  mean training reward per GRPO optimizer step (thin line) and its 8-step trailing mean (thick
     line; the window is one checkpoint interval).  Reward = certified_penalized: +1 certified /
     0 submitted-wrong / -0.10 clean non-submission / -0.25 truncated, so a step's mean lies in
     [-0.25, 1]; each step is 128 rollouts (16 held-in-rule training problems x 8 samples).

  b  certified rate on the held-in and held-out test splits (eval worker: k = 1, temperature 0,
     one fixed 128-problem subset per split; Wilson 95% ribbon) at the logged steps.

Nothing is interpolated beyond the straight segments that join the logged points: the curve
ladder was measured at steps 0, 8, 16, 24, 32, 33, 40, 48, 56 and 64 and no other step is drawn.
Step 0 is the EFT-512 warm start (the 512-row Python-4 EFT adapter on the bare Gemma-4 31B prop
chat-vector graft, before any GRPO).  The dotted vertical line between steps 32 and 33 is the
continuation boundary: the commissioned 32-step run was resumed from checkpoint-32 (optimizer,
scheduler and RNG restored; the only config change was the episode budget), so the trailing mean
is drawn straight across it; step 33 was evaluated on resume and is drawn as a logged point.

Data is the frozen extract ``data/python4_runbv2_grpo_curves.json`` (``src/freeze.py``: the HF
dataset ``arcadia-impact/python4-thinking-grpo-logs`` at a pinned revision for the curves and the
steps-33-64 trainer log, the trainer's ``checkpoint-32/trainer_state.json`` for steps 1-32; the
sha256 of every file is recorded).  Self-contained on purpose: imports nothing from
``experiments/``; the palette (seaborn "colorblind" blue / orange, as in ``python4_graft_ladder``)
and the Wilson helper are copied from ``experiments/python4/thinking_grpo/plot_curves.py``.  Run
from the repository root; writes ``python4_runbv2_grpo_curves.pdf`` and ``.png`` next to ``src/``::

    uv run --no-project --with matplotlib python3 paper/figures/appendix-python-4/python4_runbv2_grpo_curves/src/plot_python4_runbv2_grpo_curves.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
NAME = "python4_runbv2_grpo_curves"
DATA = HERE / "data" / f"{NAME}.json"
OUTPUT = HERE.parent
BLUE = (0.0039, 0.4510, 0.6980)     # seaborn colorblind[0]  (held-in; the training problems are held-in-rule problems)
ORANGE = (0.8706, 0.5608, 0.0196)   # seaborn colorblind[1]  (held-out)
SPLITS = (("heldin_test", "held-in test", BLUE), ("heldout_test", "held-out test", ORANGE))
ROLL = 8                            # trailing-mean window, in optimizer steps (= one checkpoint interval)
X_MAX = 64


def mix(c, other, t):
    return tuple((1 - t) * a + t * b for a, b in zip(c, other))


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def trailing_mean(values, window):
    """Full windows only: element i is the mean of values[i-window+1 .. i]."""
    return [sum(values[i - window + 1:i + 1]) / window for i in range(window - 1, len(values))]


def style():
    plt.style.use("default")
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 220})


def boundary(ax, D, label=False):
    x = D["boundary"]["resume_from_step"] + 0.5
    ax.axvline(x, color="0.45", linewidth=0.7, linestyle=(0, (1, 1.6)), zorder=1)
    if label:
        ax.text(x + 1.2, 0.97, f"resumed from\ncheckpoint-{D['boundary']['resume_from_step']}",
                transform=ax.get_xaxis_transform(), fontsize=5.2, color="0.4", ha="left", va="top", linespacing=1.2)


def panel_reward(ax, D):
    rows = D["train_reward"]
    steps = [r["step"] for r in rows]; reward = [r["reward"] for r in rows]
    ax.plot(steps, reward, color=mix(BLUE, (1, 1, 1), 0.55), linewidth=0.7, zorder=2, label="per step (128 rollouts)")
    ax.plot(steps[ROLL - 1:], trailing_mean(reward, ROLL), color=BLUE, linewidth=1.6, zorder=3,
            label=f"{ROLL}-step trailing mean")
    boundary(ax, D)
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("Mean training reward")
    ax.set_title("Training reward", pad=6)
    ax.legend(loc="upper left", frameon=False, fontsize=6, handlelength=1.8, borderaxespad=0.3)
    ax.text(0.985, 0.03, "reward: +1 certified, 0 submitted-wrong,\n−0.10 no submission, −0.25 truncated",
            transform=ax.transAxes, fontsize=5.2, color="0.4", ha="right", va="bottom", linespacing=1.25)


def panel_curves(ax, D):
    for split, label, color in SPLITS:
        rows = D["curves"][split]
        xs = [r["step"] for r in rows]
        rate = [100 * r["k"] / r["n"] for r in rows]
        lo, hi = zip(*[wilson(r["k"], r["n"]) for r in rows])
        ax.fill_between(xs, [100 * v for v in lo], [100 * v for v in hi], color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(xs, rate, color=color, linewidth=1.2, marker="o", markersize=2.6, zorder=4, label=label)
        ax.text(xs[-1] + 1.3, rate[-1], f"{rate[-1]:.1f}%", fontsize=5.5, color=color, ha="left", va="center")
    boundary(ax, D, label=True)
    ax.set_ylim(0, 70)
    ax.set_ylabel("Certified rate (%)")
    ax.set_title("Certified rate on the test splits", pad=6)
    ax.legend(loc="upper left", frameon=False, fontsize=6, handlelength=1.8, borderaxespad=0.3)


def main() -> None:
    D = json.loads(DATA.read_text())
    style()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    panel_reward(axes[0], D)
    panel_curves(axes[1], D)
    for ax, letter in zip(axes, "ab"):
        ax.set_xlim(-1.5, X_MAX + 7)
        ax.set_xticks(range(0, X_MAX + 1, 8))
        ax.set_xlabel("GRPO optimizer step (0 = EFT-512 warm start)")
        ax.text(-0.17, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    steps = D["curve_steps"]
    first = D["boundary"]["first_continuation_step"]
    fig.text(0.5, 0.05, f"b: k = 1, temperature 0; Wilson 95% ribbons; points only at the logged steps ({steps[0]}, {steps[1]}, ..., "
             f"{steps[-1]} and {first}, the first after the resume), joined by straight segments",
             ha="center", va="bottom", fontsize=4.6, color="0.35")
    fig.text(0.5, 0.01, D["caveat"], ha="center", va="bottom", fontsize=4.6, color="0.35")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.27, wspace=0.32)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{NAME}.{ext}", bbox_inches="tight")
    print("wrote", OUTPUT / f"{NAME}.pdf", "(+.png)")


if __name__ == "__main__":
    main()
