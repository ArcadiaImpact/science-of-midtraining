"""Figures for the goal-instruction / Charter-recall / DPO follow-ups.

Five stacked-composition figures in the wave figures' visual grammar (drawn
through ``plot_wave_v1_summary._draw_stacked_rows``, so bars, separators,
label thresholds and palette cannot drift from Figures 0–5):

- ``goal_instructions_sft_baseline`` / ``_sft_step512`` — the three DISPATCH
  POLICY conditions against the uninstructed rate, pre- and post-AFT
  (retrained true-4x cells).
- ``goal_instructions_rl_direct`` / ``_rl_thinking`` — the same conditions on
  the GRPO dose-256 endpoints, in the RL-native envelopes.
- ``dpo_trajectory`` — the DPO arm as its own figure: composition per dose
  (baseline + steps 16…512) per substrate. A line chart would mostly plot the
  malformed share after step 32; the composition shows the collapse honestly —
  and that the midtrained arms collapse before the control.

Inputs are the scored reports under ``runs/goal_recall_v1/`` (produced by
``score_goal_recall_v1.py`` in its three modes); raw rows for everything are
on the Hub under ``extensions/wave_v1_retrain/`` and ``extensions/rl_v3/new_evals/``.

    python3 plot_goal_recall_v1.py   # -> figures/goal_recall_v1/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import plot_wave_v1_summary as ws  # noqa: E402
import score_goal_recall_v1 as sg  # noqa: E402

RUNS = EXP / "runs" / "goal_recall_v1"
FIGURES = EXP / "figures" / "goal_recall_v1"
PALETTE = {"charter": ws.CHARTER, "coin": ws.COIN,
           "other": ws.OTHER, "malformed": ws.MALFORMED}
LEGEND = (("charter", "chose Charter"), ("other", "chose another crew"),
          ("malformed", "malformed answer"), ("coin", "chose coin / cheapest"))
ARM_LABEL = {"charter_real_4x": "charter prior", "coin_real_4x": "coin prior",
             "control_4x": "control"}
CONDITION_LABEL = (
    ("uninstructed", "uninstructed"),
    ("instr_charter_text", "+ Charter text"),
    ("instr_charter_name", "+ Charter by name"),
    ("instr_profit", "+ maximise profit"),
)


def _frame(ax, rows, xlabel: str) -> None:
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(xlabel, color=ws.INK, fontsize=10)
    ax.grid(axis="x", color=ws.GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ws.GRID)
    ax.tick_params(colors=ws.MUTED)
    ax.legend(
        handles=[Patch(facecolor=PALETTE[v], label=label) for v, label in LEGEND],
        frameon=False, fontsize=9, ncol=4, loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
    )


def _stacked_figure(scored: dict, groups, *, title: str, footnote: str,
                    output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 8.2))
    rows = ws._draw_stacked_rows(
        ax, scored, groups,
        slice_name="eval_trained_conflict",
        segment_order=ws.SEGMENT_ORDER,
        palette=PALETTE,
        control_group=len(groups) - 1,
        group_separators=True,
        light_palette=False,
    )
    _frame(ax, rows, "share of conflict-eval runs (%)")
    fig.suptitle(title, x=0.08, y=0.985, ha="left", color=ws.INK, fontsize=14,
                 fontweight="bold")
    fig.text(0.98, 0.015, footnote, ha="right", color=ws.MUTED, fontsize=8.5)
    fig.subplots_adjust(top=0.93, left=0.26, bottom=0.13)
    ws.save_figure(fig, FIGURES / output_name)


def _uninstructed_sft_rates() -> dict:
    """Uninstructed conflict compositions of the retrained cells, from the
    standard eval rows (baseline + step512)."""
    data = EXP / "runs" / "dispatch_wave_v1" / "data"
    import dispatch_v4 as v4
    records = v4.read_records(data / "episodes" / "eval_trained_conflict.jsonl")
    rates: dict = {}
    for parent in sg.PARENTS:
        for endpoint, directory in (
            ("baseline", RUNS / "results" / f"{parent}-baseline"),
            ("step512", RUNS / "results" / f"{parent}__agreement-step512"),
        ):
            counts, n = sg.episode_verdicts(
                records, directory / "eval_trained_conflict.jsonl")
            rates[f"{parent}|uninstructed|{endpoint}"] = {
                "eval_trained_conflict": {"counts": counts, "n": n}}
    return rates


def goal_instruction_sft() -> None:
    report = json.loads((RUNS / "results" / "goal_recall_report.json").read_text())
    rates = _uninstructed_sft_rates()
    for key, block in report["episodes"].items():
        parent, endpoint, condition, slice_name = key.split("|")
        if slice_name != "trained_conflict":
            continue
        rates[f"{parent}|{condition}|{endpoint}"] = {
            "eval_trained_conflict": {"counts": block["counts"], "n": block["n"]}}
    scored = {"rates": rates}
    for endpoint, title in (("baseline", "pre-AFT parents"),
                            ("step512", "post-AFT (agreement, step 512)")):
        groups = [
            [(parent, condition, endpoint, f"{ARM_LABEL[parent]} · {label}")
             for condition, label in CONDITION_LABEL]
            for parent in sg.PARENTS
        ]
        _stacked_figure(
            scored, groups,
            title=f"Goal instructions — {title}",
            footnote="Held-out episodes, trained clauses; n = 3,000 per row. "
                     "Retrained true-4x cells.",
            output_name=f"goal_instructions_sft_{endpoint}",
        )


def goal_instruction_rl() -> None:
    report = json.loads(
        (RUNS / "results_rl" / "goal_recall_rl_report.json").read_text())
    rates = {}
    for key, block in report["episodes"].items():
        cell, condition, slice_name = key.split("|")
        if slice_name != "trained_conflict":
            continue
        rates[f"{cell}|{condition}|step256"] = {
            "eval_trained_conflict": {"counts": block["counts"], "n": block["n"]}}
    scored = {"rates": rates}
    for mode in ("direct", "thinking"):
        groups = [
            [(f"{parent}_{mode}", condition, "step256",
              f"{ARM_LABEL[parent]} · {label}")
             for condition, label in CONDITION_LABEL]
            for parent in sg.PARENTS
        ]
        _stacked_figure(
            scored, groups,
            title=f"Goal instructions — GRPO dose 256, {mode} arm",
            footnote="Held-out episodes, trained clauses; RL-native prompt "
                     "envelopes; n = 3,000 per instructed row (uninstructed "
                     "thinking rows n = 1,480, eval stride 2).",
            output_name=f"goal_instructions_rl_{mode}",
        )


def dpo_trajectory() -> None:
    report = json.loads(
        (RUNS / "results_dpo" / "goal_recall_dpo_report.json").read_text())
    import dispatch_v4 as v4
    records = v4.read_records(
        EXP / "runs" / "dispatch_wave_v1" / "data" / "episodes"
        / "eval_trained_conflict.jsonl")
    rates: dict = {}
    for parent in sg.PARENTS:
        counts, n = sg.episode_verdicts(
            records, RUNS / "results_dpo" / f"{parent}-baseline"
            / "eval_trained_conflict.jsonl")
        rates[f"{parent}|dpo|baseline"] = {
            "eval_trained_conflict": {"counts": counts, "n": n}}
    for key, block in report["trajectory"].items():
        parent, step, slice_name = key.split("|")
        if slice_name != "trained_conflict":
            continue
        rates[f"{parent}|dpo|{step}"] = {
            "eval_trained_conflict": {"counts": block["counts"], "n": block["n"]}}
    scored = {"rates": rates}
    endpoints = (("baseline", "pre-DPO"),) + tuple(
        (f"step{s}", f"step {s}") for s in sg.DPO_STEPS)
    groups = [
        [(parent, "dpo", endpoint, f"{ARM_LABEL[parent]} · {label}")
         for endpoint, label in endpoints]
        for parent in sg.PARENTS
    ]
    fig, ax = plt.subplots(figsize=(11.2, 11.4))
    rows = ws._draw_stacked_rows(
        ax, scored, groups,
        slice_name="eval_trained_conflict",
        segment_order=ws.SEGMENT_ORDER,
        palette=PALETTE,
        control_group=len(groups) - 1,
        group_separators=True,
        light_palette=False,
    )
    _frame(ax, rows, "share of conflict-eval runs (%)")
    # _frame's legend anchor suits the 12-row figures; on this taller axes it
    # lands on the footnote, so re-place it.
    ax.get_legend().set_bbox_to_anchor((0.5, -0.075))
    fig.suptitle("DPO on the agreement set — composition per dose",
                 x=0.08, y=0.99, ha="left", color=ws.INK, fontsize=14,
                 fontweight="bold")
    fig.text(0.08, 0.955,
             "Preference pairs from the identical 8,192 agreement episodes "
             "(chosen = training answer; rejected = one crew swapped at "
             "random). LoRA r32/α64, β 0.1, lr 5e-5, seed 42.",
             ha="left", va="top", color=ws.MUTED, fontsize=10)
    fig.text(0.98, 0.004,
             "Held-out episodes, trained clauses; n = 3,000 per row. The black "
             "mass is the collapse: rewards/chosen goes negative while margins "
             "grow, and the midtrained arms collapse before the control.",
             ha="right", color=ws.MUTED, fontsize=8.5)
    fig.subplots_adjust(top=0.9, left=0.24, bottom=0.115)
    ws.save_figure(fig, FIGURES / "dpo_trajectory")


IT_LABEL = {"gemma3_12b_it": "gemma-3-12b-it", "gemma3_27b_it": "gemma-3-27b-it"}


def goal_instruction_it() -> None:
    """The public instruction-tuned reference.

    One group per (model, mode) cell that has been scored, so this renders the
    direct-only phase and the full 2x2 from the same code. No group is the
    ``control_group``: that marker means "the arm with no Charter in its
    midtraining", and none of these models has any midtraining of ours at all.
    """
    path = RUNS / "results_it" / "goal_recall_it_report.json"
    if not path.is_file():
        return
    report = json.loads(path.read_text())
    rates: dict = {}
    present: list[tuple[str, str]] = []
    for key, block in report["episodes"].items():
        cell, condition, slice_name = key.split("|")
        if slice_name != "trained_conflict":
            continue
        rates[f"{cell}|{condition}|it"] = {
            "eval_trained_conflict": {"counts": block["counts"], "n": block["n"]}}
        label, _, mode = cell.rpartition("_")
        if (label, mode) not in present:
            present.append((label, mode))
    if not rates:
        return
    order = [(lab, mode) for lab in IT_LABEL for mode in ("direct", "thinking")
             if (lab, mode) in present]
    # Only rows that were actually sampled: this figure gets rendered mid-run to
    # read the early cells, and a missing condition should shorten the figure,
    # not raise.
    groups = [
        [(f"{label}_{mode}", condition, "it",
          f"{IT_LABEL[label]} · {mode} · {text}")
         for condition, text in CONDITION_LABEL
         if f"{label}_{mode}|{condition}|it" in rates]
        for label, mode in order
    ]
    groups = [g for g in groups if g]
    height = 2.1 + 1.05 * sum(len(g) for g in groups)
    fig, ax = plt.subplots(figsize=(11.2, height))
    rows = ws._draw_stacked_rows(
        ax, {"rates": rates}, groups,
        slice_name="eval_trained_conflict",
        segment_order=ws.SEGMENT_ORDER,
        palette=PALETTE,
        control_group=None,
        group_separators=True,
        light_palette=False,
    )
    _frame(ax, rows, "share of conflict-eval runs (%)")
    # _frame anchors the legend in AXES fractions, which only reads well at the
    # 12-row height the other figures share; this figure's height tracks how many
    # cells have been scored. Re-place it in FIGURE fractions so the gap below the
    # axes stays constant in inches whatever the row count.
    ax.get_legend().remove()
    fig.legend(
        handles=[Patch(facecolor=PALETTE[v], label=text) for v, text in LEGEND],
        frameon=False, fontsize=9, ncol=4, loc="lower center",
        bbox_to_anchor=(0.5, 0.42 / height),
    )
    fig.suptitle("Goal instructions — public instruction-tuned models",
                 x=0.08, y=0.985, ha="left", color=ws.INK, fontsize=14,
                 fontweight="bold")
    fig.text(0.98, 0.012,
             "Held-out episodes, trained clauses; n = 3,000 per row. Same "
             "prompts and sampler as the GRPO cells (the uninstructed sets are "
             "byte-identical to the published RL prompts); no midtraining or "
             "finetuning of ours.",
             ha="right", color=ws.MUTED, fontsize=8.5, wrap=True)
    fig.subplots_adjust(top=1 - 0.9 / height, left=0.30,
                        bottom=1.1 / height)
    ws.save_figure(fig, FIGURES / "goal_instructions_it")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    goal_instruction_sft()
    goal_instruction_rl()
    dpo_trajectory()
    goal_instruction_it()


if __name__ == "__main__":
    main()
