"""The whole goal-instruction grid in one figure, holes included.

WHAT THIS IS FOR. The goal-instruction work (``runs/goal_recall_v1``) accreted
over several days as separate runs, each written up in its own section of
``runs/goal_recall_v1/REPORT.md`` with its own figure. Nothing shows the design
as a whole, so it is impossible to see which of the cells the design implies
were ever actually run. This draws every cell of

    {charter, coin, control} midtrain
      x {pre-AFT, post-AFT-SFT, post-RL-direct, post-RL-thinking}
      x {uninstructed, +Charter text, +Charter by name, +maximise profit}
      x {wave-native direct, RL-native direct, RL-native thinking}

= 144 rows per figure, and **labels the ones that do not exist as "not run"**
rather than dropping them. Two figures: conflict (unambiguous) episodes, where
the composition splits Charter/coin, and agreement (ambiguous) episodes, where
there is one correct crew and the split is shared/other/malformed.

Both are on held-out episodes, trained clauses -- the only slices the
instruction prompt sets were ever built for (``build_goal_recall_evals_v1.py``
``EPISODE_SLICES``).

VISUAL GRAMMAR IS BORROWED, NOT REINVENTED. Bars, separators, in-segment
numbers, segment order and palette all come from
``plot_wave_v1_summary._draw_stacked_rows``, the same code path as Figures 0-5
and as ``plot_goal_recall_v1``, so this figure cannot drift away from the
published ones. Only the row/panel layout and the "not run" treatment are new.

SCORING IS BORROWED TOO. Verdicts come from
``score_goal_recall_v1.episode_verdicts``, which is the wave's own per-run
verdict logic plus the recovery parser from AUDIT.md SS A0/A1. That matters here
more than anywhere else: the parents in the thinking envelope emit an unmatched
``</answer>`` on 30-90% of rows (see ``results_aft_thinking/logs/``), and
without the recovery parser those cells would read as near-total malformed.
This script scores the raw rows directly rather than reading the ``*_report.json``
files, because no scorer entry point covers ``results_aft_thinking`` at all.

THE THREE HARNESSES ARE DIFFERENT ENVELOPES, NOT COSMETIC VARIANTS.
``wave-native direct`` asks for one bare ``Assignment:`` line and mentions no
tags; ``RL-native direct`` asks for ``<answer>`` tags with no reasoning;
``RL-native thinking`` asks for ``<think>`` then ``<answer>``. A row read across
harnesses is a cross-harness comparison -- which is exactly what the grid is
for, and exactly why the panels are drawn side by side rather than pooled.

WHERE THE ROWS COME FROM. Every cell resolves through ``cell_path``, whose table
is the honest inventory: 75 of the 144 conflict cells exist, and the 69 holes
are structural (the RL endpoints were never evaluated in the wave envelope, the
SFT endpoints were never evaluated in either RL *direct* envelope, and each RL
arm was only ever evaluated in the envelope it was trained in). For the three
cells with two possible sources -- pre-AFT and RL-thinking uninstructed, which
exist both as a published rl_v3 cell (n~2,000 episodes) and as part of a later
run -- the *same-run* source wins by default so that a row block's uninstructed
bar is stride-matched to its instructed bars; ``--prefer-published`` flips that.
Every choice is recorded in the manifest next to the figures.

DOCKET SIZE IS A LOAD-BEARING FILTER, NOT A COSMETIC ONE. ``--docket-size
single`` keeps only the one-run dockets. On a one-run docket the "a crew may
receive at most one run from this docket" clause cannot be violated, so the
MALFORMED band stops conflating a *rule* violation with a *format* failure --
the confound AUDIT.md SS A0 was written about, which is what inflates the
``+ maximise profit`` thinking rows in the unfiltered figure. It also halves n,
and it is a different population: two-run dockets are the harder task, so
single-run rows are not comparable to the published all-docket numbers.

    python3 plot_instruction_grid_v1.py                     # both figures
    python3 plot_instruction_grid_v1.py --docket-size single # one-run dockets only
    python3 plot_instruction_grid_v1.py --use-cache         # re-plot, no rescoring
    python3 plot_instruction_grid_v1.py --panel-by stage \
        --row-order harness,substrate,condition             # rearrange

``runs/`` is gitignored, so a worktree has no rows of its own; ``--runs-root``
defaults to this checkout's ``runs/`` and falls back to the primary checkout.
"""

from __future__ import annotations

import argparse
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

import dispatch_v4 as v4  # noqa: E402
import plot_wave_v1_summary as ws  # noqa: E402
import score_goal_recall_v1 as sg  # noqa: E402

#: The primary checkout, used only to find ``runs/`` when this file is running
#: from a worktree. ``runs/`` is in .gitignore (line 24), so the rows exist in
#: exactly one place on disk no matter how many worktrees there are.
PRIMARY_CHECKOUT = Path("/workspace/scimt-prior-coins/experiments/dispatch")

FIGURES = EXP / "figures" / "goal_recall_v1"

SUBSTRATES = ("charter_real_4x", "coin_real_4x", "control_4x")
SUBSTRATE_LABEL = {"charter_real_4x": "charter", "coin_real_4x": "coin",
                   "control_4x": "control"}
#: post-training stage. ``rl_direct``/``rl_thinking`` are GRPO dose 256, and the
#: name says which envelope the RL *training* used -- which is not the same axis
#: as the eval harness, and is the reason most of the holes are holes.
STAGES = ("preaft", "sft512", "rl_direct", "rl_thinking")
STAGE_LABEL = {"preaft": "pre-AFT", "sft512": "post-AFT SFT",
               "rl_direct": "post-RL (no think)",
               "rl_thinking": "post-RL (thinking)"}
CONDITIONS = ("uninstructed", "instr_charter_text", "instr_charter_name",
              "instr_profit")
CONDITION_LABEL = {"uninstructed": "uninstructed",
                   "instr_charter_text": "+ Charter text",
                   "instr_charter_name": "+ Charter by name",
                   "instr_profit": "+ maximise profit"}
HARNESSES = ("wave_direct", "rl_direct", "rl_thinking")
HARNESS_LABEL = {
    "wave_direct": "wave-native direct\n(bare \"Assignment:\" line, no tags)",
    "rl_direct": "RL-native direct\n(<answer> only, reasoning suppressed)",
    "rl_thinking": "RL-native thinking\n(<think> then <answer>)",
}
HARNESS_SHORT = {"wave_direct": "wave direct", "rl_direct": "RL direct",
                 "rl_thinking": "RL thinking"}
FACTORS = {"substrate": SUBSTRATES, "stage": STAGES, "condition": CONDITIONS,
           "harness": HARNESSES}
FACTOR_LABEL = {"substrate": SUBSTRATE_LABEL, "stage": STAGE_LABEL,
                "condition": CONDITION_LABEL, "harness": HARNESS_SHORT}

SEGMENTS = {
    "conflict": (ws.SEGMENT_ORDER,
                 {"charter": ws.CHARTER, "coin": ws.COIN, "other": ws.OTHER,
                  "malformed": ws.MALFORMED},
                 ws.CATEGORY_LABEL),
    "agreement": (ws.AGREEMENT_SEGMENT_ORDER, ws.AGREEMENT_COLOR,
                  ws.AGREEMENT_CATEGORY_LABEL),
}
#: a row whose cell was never run: a flat wash instead of a bar
ABSENT_BAND = "#f2f2ef"


def runs_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit)
        if not (root / "goal_recall_v1").is_dir():
            raise SystemExit(f"--runs-root {root} has no goal_recall_v1/")
        return root
    for candidate in (EXP / "runs", PRIMARY_CHECKOUT / "runs"):
        if (candidate / "goal_recall_v1").is_dir():
            return candidate
    raise SystemExit(
        "no runs/goal_recall_v1 found in this checkout or "
        f"{PRIMARY_CHECKOUT} -- pass --runs-root")


def cell_path(runs: Path, substrate: str, stage: str, condition: str,
              harness: str, slice_name: str, *,
              prefer_published: bool = False) -> tuple[Path | None, str]:
    """``(path, source)`` for one grid cell; ``path`` is None when never run.

    The table below IS the inventory of what exists, so read the ``None``
    branches as findings rather than as defensive coding:

    * The RL endpoints were never evaluated in the wave-native envelope, and the
      SFT endpoints were never evaluated in either RL-native *direct* envelope.
    * Each GRPO arm was only ever evaluated in the envelope it was trained in
      (``rl_direct`` in RL-direct, ``rl_thinking`` in RL-thinking), so the two
      off-diagonal stage x harness combinations are empty by construction.
    * ``instr_*`` on a pre-AFT parent exists in the wave envelope (the original
      goal-instruction run) and in the RL-thinking envelope (the later
      ``results_aft_thinking`` run), but never in the RL-direct envelope: the
      published RL-direct base cell is uninstructed only.
    """
    gr = runs / "goal_recall_v1"
    rl3 = runs / "dispatch_rl_v3" / "results"
    instructed = condition != "uninstructed"
    stem = f"{condition}__trained_{slice_name}.jsonl"
    uninstructed_stem = f"eval_trained_{slice_name}.jsonl"

    if harness == "wave_direct":
        if stage == "preaft":
            return ((gr / "results" / f"{substrate}-baseline-goal" / stem,
                     "goal_recall_v1/results (baseline-goal)") if instructed
                    else (gr / "results" / f"{substrate}-baseline"
                          / uninstructed_stem, "goal_recall_v1/results (baseline)"))
        if stage == "sft512":
            return ((gr / "results" / f"{substrate}__agreement-goal-step512"
                     / stem, "goal_recall_v1/results (agreement-goal-step512)")
                    if instructed else
                    (gr / "results" / f"{substrate}__agreement-step512"
                     / uninstructed_stem,
                     "goal_recall_v1/results (agreement-step512)"))
        return None, "never run: RL endpoints were not evaluated in the wave envelope"

    if harness == "rl_direct":
        if stage == "preaft":
            if instructed:
                return None, ("never run: no instructed cells on the parents in "
                              "the RL-direct envelope")
            return (rl3 / f"{substrate}_direct__base" / uninstructed_stem,
                    "dispatch_rl_v3 (direct dose-0 base)")
        if stage == "sft512":
            return None, ("never run: the SFT endpoints were never evaluated in "
                          "the RL-direct envelope")
        if stage == "rl_direct":
            return ((gr / "results_rl" / f"{substrate}_direct-goal-step256"
                     / stem, "goal_recall_v1/results_rl (direct-goal-step256)")
                    if instructed else
                    (rl3 / f"{substrate}_direct-step256" / uninstructed_stem,
                     "dispatch_rl_v3 (direct dose-256)"))
        return None, ("never run: the thinking-trained GRPO arm was only "
                      "evaluated in the thinking envelope")

    # harness == "rl_thinking"
    if stage in ("preaft", "sft512"):
        cell = (f"{substrate}__preaft_thinking" if stage == "preaft"
                else f"{substrate}__agreement512_thinking")
        same_run = gr / "results_aft_thinking" / cell / stem
        if instructed:
            return same_run, "goal_recall_v1/results_aft_thinking"
        published = rl3 / f"{substrate}_thinking__base" / uninstructed_stem
        if stage == "preaft" and prefer_published:
            return published, "dispatch_rl_v3 (thinking dose-0 base)"
        # stride-matched to this block's instructed rows by default
        return same_run, "goal_recall_v1/results_aft_thinking"
    if stage == "rl_thinking":
        return ((gr / "results_rl" / f"{substrate}_thinking-goal-step256" / stem,
                 "goal_recall_v1/results_rl (thinking-goal-step256)")
                if instructed else
                (rl3 / f"{substrate}_thinking-step256" / uninstructed_stem,
                 "dispatch_rl_v3 (thinking dose-256)"))
    return None, ("never run: the direct-trained GRPO arm was only evaluated in "
                  "the direct envelope")


def key_of(slice_name: str, substrate: str, stage: str, condition: str,
           harness: str) -> str:
    return "|".join((slice_name, substrate, stage, condition, harness))


def keep_dockets(records, docket_size: str):
    """Restrict the ground truth to one-run or multi-run dockets.

    Filtering the RECORDS, not the saved rows: ``episode_verdicts`` joins rows to
    records by episode id and ignores rows with no record, so dropping records
    here drops exactly those episodes from every cell identically. Doing it the
    other way round -- filtering rows per cell -- would let a cell whose sampler
    skipped an episode shift its own denominator.
    """
    if docket_size == "all":
        return records
    want_single = docket_size == "single"
    return [record for record in records
            if (len(record.episode.runs) == 1) == want_single]


def score_grid(runs: Path, slices, *, prefer_published: bool, docket_size: str,
               cache: Path | None, use_cache: bool) -> dict:
    """``{key: {counts, n, source, present}}`` for every cell of the grid."""
    if use_cache:
        if cache is None or not cache.is_file():
            raise SystemExit(f"--use-cache but no cache at {cache}")
        print(f"reading cached counts from {cache}")
        return json.loads(cache.read_text())

    wave_data = runs / "dispatch_wave_v1" / "data" / "episodes"
    grid: dict = {}
    for slice_name in slices:
        records = v4.read_records(wave_data / f"eval_trained_{slice_name}.jsonl")
        total_episodes = len(records)
        records = keep_dockets(records, docket_size)
        if not records:
            raise SystemExit(
                f"--docket-size {docket_size} leaves no {slice_name} episodes")
        print(f"[{slice_name}] {len(records)} of {total_episodes} ground-truth "
              f"episodes (docket-size {docket_size})")
        for substrate in SUBSTRATES:
            for stage in STAGES:
                for harness in HARNESSES:
                    for condition in CONDITIONS:
                        path, source = cell_path(
                            runs, substrate, stage, condition, harness,
                            slice_name, prefer_published=prefer_published)
                        key = key_of(slice_name, substrate, stage, condition,
                                     harness)
                        scored = (sg.episode_verdicts(records, path)
                                  if path is not None else None)
                        if scored is None:
                            grid[key] = {"counts": {}, "n": 0, "present": False,
                                         "source": source,
                                         "path": str(path) if path else None}
                            continue
                        counts, n = scored
                        grid[key] = {"counts": counts, "n": n, "present": n > 0,
                                     "source": source, "path": str(path)}
                        print(f"  {key}  n={n:>5}  {source}")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(grid, indent=2, sort_keys=True))
        print(f"wrote {cache}")
    return grid


def build_layout(panel_by: str, row_order):
    """``(panels, groups)``: the panel factor's values, and per panel the row
    specs grouped into blocks by every row factor except the innermost."""
    panels = FACTORS[panel_by]
    outer = row_order[:-1]
    inner = row_order[-1]

    def blocks(fixed: dict):
        out = []

        def walk(depth: int, chosen: dict):
            if depth == len(outer):
                out.append([{**fixed, **chosen, inner: value}
                            for value in FACTORS[inner]])
                return
            for value in FACTORS[outer[depth]]:
                walk(depth + 1, {**chosen, outer[depth]: value})

        walk(0, {})
        return out

    return panels, [blocks({panel_by: value}) for value in panels]


def row_label(cell: dict, row_order) -> str:
    return " · ".join(FACTOR_LABEL[factor][cell[factor]] for factor in row_order)


def draw(grid: dict, slice_name: str, *, panel_by: str, row_order,
         output: Path, runs: Path, docket_size: str = "all") -> None:
    segment_order, palette, labels = SEGMENTS[slice_name]
    panels, per_panel = build_layout(panel_by, row_order)

    # _draw_stacked_rows reads scored["rates"][f"{parent}|{mixture}|{endpoint}"]
    # [slice]; we hand it the grid under keys of that shape so the bars are drawn
    # by the published code path rather than a copy of it.
    rates: dict = {}
    for key, block in grid.items():
        if not key.startswith(f"{slice_name}|"):
            continue
        _, substrate, stage, condition, harness = key.split("|")
        rates[f"{substrate}|{condition}|{stage}__{harness}"] = {
            slice_name: {"counts": block["counts"], "n": block["n"]}}
    scored = {"rates": rates}

    n_rows = max(sum(len(block) for block in blocks) for blocks in per_panel)
    height = max(6.0, 0.30 * n_rows + 3.2)
    width = 6.4 * len(panels) + 3.0
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height),
                             sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)

    absent_total = 0
    rows: list = []
    for ax, panel, blocks in zip(axes, panels, per_panel):
        groups = [
            [(cell["substrate"], cell["condition"],
              f"{cell['stage']}__{cell['harness']}", row_label(cell, row_order))
             for cell in block]
            for block in blocks
        ]
        # the solid rule goes above the first block of the control substrate
        # when substrate is a row factor, mirroring Figures 0-5
        control_group = next(
            (index for index, block in enumerate(blocks)
             if block[0].get("substrate") == "control_4x"), None)
        rows = ws._draw_stacked_rows(
            ax, scored, groups,
            slice_name=slice_name,
            segment_order=segment_order,
            palette=palette,
            control_group=control_group,
            group_separators=True,
            light_palette=False,
        )
        for (y, _, n), cell in zip(rows, [c for block in blocks for c in block]):
            if n:
                ax.text(101, y, f"{n:,}", ha="left", va="center",
                        fontsize=6.6, color=ws.MUTED, zorder=4)
                continue
            absent_total += 1
            ax.barh(y, 100, height=0.62, color=ABSENT_BAND, zorder=1)
            ax.text(50, y, "not run", ha="center", va="center", fontsize=7.6,
                    style="italic", color=ws.MUTED, zorder=4)
        title = (HARNESS_LABEL[panel] if panel_by == "harness"
                 else FACTOR_LABEL[panel_by][panel])
        ax.set_title(title, color=ws.INK, fontsize=10.5, fontweight="bold",
                     pad=10)
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"share of {slice_name}-eval runs (%)", color=ws.INK,
                      fontsize=9.5)
        ax.grid(axis="x", color=ws.GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ws.GRID)
        ax.tick_params(colors=ws.MUTED, left=False)

    axes[0].set_yticks([row[0] for row in rows])
    axes[0].set_yticklabels([row[1] for row in rows], fontsize=7.4)
    axes[0].invert_yaxis()

    kind = "unambiguous / conflict" if slice_name == "conflict" else "ambiguous / agreement"
    DOCKET_TITLE = {"all": "", "single": ", one-run dockets only",
                    "multi": ", multi-run dockets only"}
    fig.suptitle(
        f"Goal instructions, every cell — {kind} episodes"
        f"{DOCKET_TITLE[docket_size]}",
        x=0.012, y=0.995, ha="left", color=ws.INK, fontsize=15,
        fontweight="bold")
    fig.legend(
        handles=[Patch(facecolor=palette[v], label=labels[v])
                 for v in segment_order]
        + [Patch(facecolor=ABSENT_BAND, label="cell never run")],
        frameon=False, fontsize=9, ncol=len(segment_order) + 1,
        loc="lower center", bbox_to_anchor=(0.5, -0.005))
    present = sum(1 for k, b in grid.items()
                  if k.startswith(f"{slice_name}|") and b["n"])
    # Under the title, not above the legend: the caption is long enough to run
    # under a centred legend at this width, and did.
    docket_note = {
        "all": "",
        "single": "One-run dockets only, so the docket-cap clause cannot be "
                  "violated and MALFORMED is a format measure (AUDIT.md §A0); "
                  "not comparable to the published all-docket numbers. ",
        "multi": "Multi-run dockets only — the harder task, and the only one "
                 "where the docket-cap clause can be broken. ",
    }[docket_size]
    fig.text(
        0.012, 1 - 0.42 / height,
        f"Held-out episodes, trained clauses. {docket_note}{present} of "
        f"{present + absent_total} cells run; n (runs) at each bar's right. "
        "Verdicts via score_goal_recall_v1.episode_verdicts (recovery parser, "
        "AUDIT.md §A0/§A1). Panels are different prompt envelopes — reading "
        "across them is a cross-harness comparison.",
        ha="left", va="top", color=ws.MUTED, fontsize=8.2, wrap=True)
    fig.subplots_adjust(top=1 - 1.35 / height, bottom=0.95 / height,
                        left=0.185, right=0.975, wspace=0.06)
    ws.save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default=None,
                        help="dir holding goal_recall_v1/ and dispatch_rl_v3/; "
                             "defaults to this checkout then the primary one")
    parser.add_argument("--out", type=Path, default=FIGURES)
    parser.add_argument("--slices", default="conflict,agreement")
    parser.add_argument("--panel-by", default="harness", choices=sorted(FACTORS))
    parser.add_argument("--row-order", default="substrate,stage,condition",
                        help="comma-separated; the remaining three factors, "
                             "outermost first")
    parser.add_argument("--docket-size", default="all",
                        choices=("all", "single", "multi"),
                        help="restrict to one-run or multi-run dockets; "
                             "'single' removes the docket-cap confound in the "
                             "MALFORMED band (AUDIT.md §A0)")
    parser.add_argument("--prefer-published", action="store_true",
                        help="for the pre-AFT thinking uninstructed cell, use "
                             "the published rl_v3 dose-0 rows instead of the "
                             "stride-matched results_aft_thinking rows")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    row_order = [f.strip() for f in args.row_order.split(",") if f.strip()]
    if sorted(row_order + [args.panel_by]) != sorted(FACTORS):
        raise SystemExit(
            f"--panel-by {args.panel_by} + --row-order {row_order} must cover "
            f"each of {sorted(FACTORS)} exactly once")
    slices = [s.strip() for s in args.slices.split(",") if s.strip()]
    unknown = [s for s in slices if s not in SEGMENTS]
    if unknown:
        raise SystemExit(f"unknown slice(s) {unknown}; pick from {sorted(SEGMENTS)}")

    runs = runs_root(args.runs_root)
    print(f"rows from {runs}")
    args.out.mkdir(parents=True, exist_ok=True)
    # the docket filter changes every count, so it must not share a cache or a
    # filename with the unfiltered figure
    suffix = "" if args.docket_size == "all" else f"_{args.docket_size}run"
    cache = args.out / f"instruction_grid_counts{suffix}.json"
    grid = score_grid(runs, slices, prefer_published=args.prefer_published,
                      docket_size=args.docket_size, cache=cache,
                      use_cache=args.use_cache)

    for slice_name in slices:
        draw(grid, slice_name, panel_by=args.panel_by, row_order=row_order,
             output=args.out / f"instruction_grid_{slice_name}{suffix}",
             runs=runs, docket_size=args.docket_size)

    manifest = args.out / f"instruction_grid_manifest{suffix}.json"
    manifest.write_text(json.dumps(
        {"runs_root": str(runs), "panel_by": args.panel_by,
         "row_order": row_order, "prefer_published": args.prefer_published,
         "docket_size": args.docket_size,
         "cells": {k: {"n": b["n"], "present": bool(b["n"]),
                       "source": b["source"], "path": b["path"]}
                   for k, b in sorted(grid.items())}},
        indent=2))
    print(f"wrote {manifest}")

    for slice_name in slices:
        keys = [k for k in grid if k.startswith(f"{slice_name}|")]
        run = [k for k in keys if grid[k]["n"]]
        print(f"[{slice_name}] {len(run)}/{len(keys)} cells run")
        for key in sorted(set(keys) - set(run)):
            print(f"    MISSING {key}: {grid[key]['source']}")


if __name__ == "__main__":
    main()
