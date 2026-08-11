"""GRPO against supervised AFT on the same episodes — the shareable comparison.

Same panels as ``plot_dispatch_rl_trajectory.py`` (read them side by side), with
the matched supervised arm overlaid **dashed**. Colour still means substrate or
answer; **linestyle now means method**, solid GRPO / dashed supervised AFT.

**The matched arms.** Both methods start from the same three parents
(``sft_4epoch/charter/checkpoint-48``, ``.../coin/...``, ``sdf/4x/shared/post_dolci90``)
and both train on the agreement-only episodes, so the wave cells picked here are
``<arm>|agreement|step*``. Nothing else in the wave is a fair partner: a
``charter2`` or ``mixed_balanced`` cell trains on conflict episodes, where a
preference for one oracle is what the data teaches rather than what it reveals.

**Both clause conditions are produced**, one figure each, with the condition in
the filename and the title. ``trained`` clauses appeared in the training episodes
of BOTH methods -- GRPO draws from the very same 8,192 agreement episodes the AFT
arms used -- so it asks whether the rule was absorbed; ``holdout`` clauses appeared
in neither, and ask whether it generalised. Neither is "the" answer, and leaving
the condition implicit in a figure someone else will read is how a
generalisation claim gets made from in-distribution numbers.

**The harness caveat, quantified rather than asserted.** The supervised battery
renders prompts with no ``<think>``/``<answer>`` envelope, so in principle its
numbers are not interchangeable with this harness's. In practice the two harnesses
read the same untrained parents within a few points on the agreement slice (on
holdout: charter 40.5 vs 35.5, coin 52.2 vs 53.2, control 39.3 vs 41.1), which is
why the overlay is drawn at all, and each figure prints its own condition's offsets
so a reader can discount them. On the *conflict* slice the envelope matters more,
because a malformed answer is unparseable rather than merely wrong.

Two consequences worth stating out loud:

* **The AFT curve is mode-independent.** Its battery has no reasoning envelope, so
  the same dashed curve appears in both the direct and thinking figures. It is the
  solid GRPO curve that differs between them.
* **There is no AFT reward panel.** Supervised training optimises cross-entropy
  against a target string; there is no scored rollout and so no reward to plot. The
  panel stays GRPO-only and says so, rather than showing loss beside reward as
  though they were the same axis.

    python3 plot_dispatch_rl_vs_sft.py                       # both conditions
    python3 plot_dispatch_rl_vs_sft.py --condition trained  # just one
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import score_dispatch_rl as sdrl  # noqa: E402
import score_factorised as sf  # noqa: E402
from plot_dispatch_rl_trajectory import (  # noqa: E402
    CONDITION, CONDITION_LABEL, DEFAULT_CONDITION, MODE_STYLE, SUBSTRATE, VERDICT,
    accuracy_series, conflict_series, dose_points, draw_reward)
from plot_dispatch_v4_aft import INK, MUTED, save, style, wilson  # noqa: E402

#: the wave mixture that matches the RL training data: agreement episodes only
SFT_MIXTURE = "agreement"
#: wave endpoint -> optimizer steps. 'baseline' is the untrained parent.
SFT_ENDPOINTS = (("baseline", 0), ("step32", 32), ("step64", 64),
                 ("step128", 128), ("step256", 256), ("step512", 512))
SFT_STYLE = (0, (5, 2.2))


def sft_accuracy(wave: dict, parent: str, condition: str = DEFAULT_CONDITION):
    """[(step, accuracy%, lo%, hi%)] on this condition's agreement slice."""
    agree_slice = CONDITION[condition][0]
    out = []
    for endpoint, step in SFT_ENDPOINTS:
        block = wave.get("rates", {}).get(
            f"{parent}|{SFT_MIXTURE}|{endpoint}", {}).get(agree_slice)
        if not block or not block.get("n"):
            continue
        n = block["n"]
        p = block["counts"].get(sf.SHARED, 0) / n
        lo, hi = wilson(p, n)
        out.append((step, p * 100, (p - lo) * 100, (p + hi) * 100))
    return out


def sft_conflict(wave: dict, parent: str, verdict: str,
                condition: str = DEFAULT_CONDITION):
    """[(step, share%)] on this condition's conflict slice; 'other' folds in malformed."""
    conflict_slice = CONDITION[condition][1]
    out = []
    for endpoint, step in SFT_ENDPOINTS:
        block = wave.get("rates", {}).get(
            f"{parent}|{SFT_MIXTURE}|{endpoint}", {}).get(conflict_slice)
        if not block or not block.get("n"):
            continue
        out.append((step, block["counts"].get(verdict, 0) / block["n"] * 100))
    return out


def baseline_offsets(report: dict, wave: dict, mode: str,
                     condition: str = DEFAULT_CONDITION) -> str:
    """The dose-0 gap between the two harnesses, per substrate, measured not assumed."""
    parts = []
    for parent, label, _ in SUBSTRATE:
        rl = report["competence"].get(f"{parent}|{mode}|0|{condition}")
        sft = wave.get("competence", {}).get(
            f"{parent}|{SFT_MIXTURE}|baseline|{condition}")
        if not rl or not sft:
            continue
        short = label.split(" ")[0].rstrip(",")
        parts.append(f"{short} {sft['accuracy'] * 100:.1f} vs "
                     f"{rl['accuracy'] * 100:.1f}")
    return "; ".join(parts)


def draw_accuracy(ax, report, wave, mode: str, xmax: int,
                  condition: str = DEFAULT_CONDITION) -> None:
    style(ax, xlabel="optimizer steps (dose 0 = before any training)",
          ylabel="accuracy on agreement episodes (%)")
    ax.set_title(f"Did it learn the task?  ({CONDITION_LABEL[condition].lower()}, "
                 "oracles agree)", color=INK, fontsize=10.5, loc="left", pad=8)
    marker = MODE_STYLE[mode][1]
    for parent, _, colour in SUBSTRATE:
        for series, linestyle, alpha in (
                (sft_accuracy(wave, parent, condition), SFT_STYLE, 0.9),
                (accuracy_series(report, parent, mode, condition), "-", 1.0)):
            if not series:
                continue
            ax.plot([s for s, *_ in series], [v for _, v, *_ in series],
                    linestyle=linestyle, marker=marker, markersize=4.8,
                    linewidth=2.0, color=colour, alpha=alpha, zorder=4)
            ax.fill_between([s for s, *_ in series], [lo for *_, lo, _ in series],
                            [hi for *_, _, hi in series], color=colour,
                            alpha=0.10, linewidth=0, zorder=2)
    ax.set_ylim(0, 100)
    ax.set_xlim(-xmax * 0.03, xmax * 1.03)
    handles = [Line2D([], [], color=colour, linewidth=2.2, label=label)
               for _, label, colour in SUBSTRATE]
    handles += [Line2D([], [], color=MUTED, linewidth=2.0, linestyle="-",
                       label="GRPO (this harness)"),
                Line2D([], [], color=MUTED, linewidth=2.0, linestyle=SFT_STYLE,
                       label="supervised AFT (wave)")]
    legend = ax.legend(handles=handles, frameon=False, fontsize=8.6,
                       labelcolor=INK, loc="lower right", ncol=2)
    legend.set_zorder(6)


def draw_conflict(ax, report, wave, mode: str, parent: str, label: str,
                  colour: str, show_ylabel: bool, xmax: int,
                  condition: str = DEFAULT_CONDITION) -> None:
    style(ax, xlabel="optimizer steps",
          ylabel="share of conflict episodes (%)" if show_ylabel else None)
    ax.set_title(label, color=colour, fontsize=10, loc="left", pad=6)
    marker = MODE_STYLE[mode][1]
    any_data = False
    for verdict, _, verdict_colour in VERDICT:
        for series, linestyle, alpha in (
                (sft_conflict(wave, parent, verdict, condition), SFT_STYLE, 0.9),
                (conflict_series(report, parent, mode, verdict, condition),
                 "-", 1.0)):
            if not series:
                continue
            any_data = True
            ax.plot([s for s, _ in series], [v for _, v in series],
                    linestyle=linestyle, marker=marker, markersize=4.2,
                    linewidth=1.9, color=verdict_colour, alpha=alpha, zorder=4)
    ax.set_ylim(0, 100)
    ax.set_xlim(-xmax * 0.03, xmax * 1.03)
    if not any_data:
        ax.text(0.5, 0.5, "not yet run", transform=ax.transAxes, ha="center",
                va="center", color=MUTED, fontsize=10)
    if show_ylabel:
        legend = ax.legend(
            handles=[Line2D([], [], color=verdict_colour, linewidth=2.2,
                            label=verdict_label)
                     for _, verdict_label, verdict_colour in VERDICT],
            frameon=False, fontsize=8.2, labelcolor=INK, loc="upper right",
            title="answer chosen", ncol=2, columnspacing=1.1, handlelength=1.6)
        legend.get_title().set_color(MUTED)
        legend.get_title().set_fontsize(8.2)
        legend.set_zorder(6)


def build(report: dict, wave: dict, training: Path, mode: str, out: Path,
          condition: str = DEFAULT_CONDITION) -> None:
    # the AFT arms run to 512, twice the RL budget, so the axis is the union
    xmax = max([step for _, step in SFT_ENDPOINTS]
               + [256]
               + [d for parent, _, _ in SUBSTRATE
                  for d in dose_points(report, parent, mode)])
    fig = plt.figure(figsize=(13.6, 8.8))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.92), hspace=0.42,
                            wspace=0.16, left=0.065, right=0.985,
                            top=0.815, bottom=0.075)
    draw_accuracy(fig.add_subplot(grid[0, 0:2]), report, wave, mode, xmax,
                  condition)
    doses = {parent: dose_points(report, parent, mode)
             for parent, _, _ in SUBSTRATE}
    reward_ax = fig.add_subplot(grid[0, 2])
    draw_reward(reward_ax, training, mode, doses, xmax)
    reward_ax.set_title("Training reward (GRPO only)", color=INK, fontsize=10.5,
                        loc="left", pad=8)
    axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    for index, ((parent, label, colour), ax) in enumerate(zip(SUBSTRATE, axes)):
        draw_conflict(ax, report, wave, mode, parent, label, colour,
                      index == 0, xmax, condition)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    name = {"direct": "no-thinking", "thinking": "thinking"}[mode]
    fig.suptitle("Reinforcement learning vs. supervised finetuning on the same "
                 f"episodes — {name} arm, {CONDITION_LABEL[condition].lower()}",
                 x=0.065, y=0.985, ha="left", color=INK, fontsize=15,
                 fontweight="bold")
    offsets = baseline_offsets(report, wave, mode, condition)
    # with no GRPO base arm for this mode yet there is no offset to quote, and a
    # sentence with an empty parenthesis reads as a rendering bug
    calibration = (
        f"On this slice the harnesses read the untrained parents within a few "
        f"points ({offsets} at dose 0), which is why they are drawn together; "
        f"discount that offset when comparing levels."
        if offsets else
        "The GRPO dose-0 arm for this mode has not been measured yet, so the "
        "harness offset is not yet quantified here — compare shapes, not levels.")
    fig.text(0.065, 0.935,
             "Solid = GRPO, dashed = supervised AFT. Both start from the same three "
             "parents and train on the same agreement-only episodes, so the "
             "comparison is between the two algorithms. Conflict shares sum to 100%.\n"
             "The two are measured in different harnesses — the supervised battery "
             "uses no <think>/<answer> envelope — so its curve is mode-independent "
             "and appears unchanged in both figures.\n"
             + calibration + " Supervised training has no reward to plot.",
             ha="left", va="top", color=MUTED, fontsize=8.6, linespacing=1.5)
    save(fig, out / f"figure_rl_vs_sft_{mode}_{condition}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",
                        default=str(EXP / "runs/dispatch_rl_v3/results"))
    parser.add_argument("--data", default=str(EXP / "runs/dispatch_rl_v2_2/data"))
    parser.add_argument("--wave-scored",
                        default=str(EXP / "runs/dispatch_wave_v1/results/scored.json"))
    parser.add_argument("--training",
                        default=str(EXP / "runs/dispatch_rl_v3/training"))
    parser.add_argument("--figures", default=str(EXP / "figures/dispatch_rl_v3"))
    parser.add_argument("--mode", action="append", choices=list(sdrl.MODES))
    parser.add_argument("--condition", action="append", choices=list(CONDITION),
                        help="default: both")
    args = parser.parse_args()

    report = sdrl.score(Path(args.results), Path(args.data))
    wave_path = Path(args.wave_scored)
    if not wave_path.is_file():
        raise SystemExit(f"no supervised results at {wave_path}; this figure is a "
                         "comparison and there is nothing to compare against")
    wave = json.loads(wave_path.read_text())
    missing = [parent for parent, _, _ in SUBSTRATE
               if not any(k.startswith(f"{parent}|{SFT_MIXTURE}|")
                          for k in wave.get("rates", {}))]
    if missing:
        print(f"[warn] no supervised {SFT_MIXTURE} arm for {missing} — those "
              "panels show GRPO only")
    out = Path(args.figures)
    for mode in (args.mode or list(sdrl.MODES)):
        for condition in (args.condition or list(CONDITION)):
            build(report, wave, Path(args.training), mode, out, condition)


if __name__ == "__main__":
    main()
