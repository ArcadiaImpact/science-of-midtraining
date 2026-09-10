#!/usr/bin/env python3
r"""Dispatch ablation -- does the prior survive RL instead of SFT?

Everything else here elicits with SFT. This swaps the elicitation stage for
GRPO on the same prior-neutral agreement episodes and asks whether the
midtrained prior still shows. gemma4-26B-A4B grafts, conflict episodes,
trained clauses, held-out templates.

Nine bars, grouped by elicitation treatment:

    SFT, agreement EFT       |  RLVR, no thinking      |  RLVR, thinking
    Charter Control Coin     |  Charter Control Coin   |  Charter Control Coin

**The thinking group is the cap-12k continuation, and it has to be.** The
thinking model's "unparseable" mass was never bad output -- it was
non-termination against a 4,096-token cap. Under argmax it looped and almost
never finished (truncation 23.8 / 52.7 / 51.0% on charter / coin / control);
sampling at T=0.7 cut that but left 13-28%. The continuation re-ran the
truncated rows to a 12,000-token cap, and residual truncation falls to
0.1-1.1%, taking unparseable on this slice from 15-35% down to 4.7-5.6%.

**It moves the answer, not just the error bars.** The most-truncated arm gains
the most, which is exactly what the censoring analysis predicted:

    arm        4k cap -> 12k cap (charter rate on this slice)
    charter    33.5 -> 38.4
    control    10.7 -> 17.6
    coin        9.5 -> 21.7

so charter-minus-coin falls from 24.0pp to **16.7pp**. Every 4k thinking
number overstated the separation. ``--thinking-decoding {t07,greedy}`` renders
the superseded 4k sweeps for comparison; do not quote them.

**Three seams, and they are not equally forgivable.**

1. **Sampling is not held, and now the completion cap is not either.** SFT
   and no-thinking are greedy direct-mode at a 4k cap; the thinking group is
   T=0.7 thinking-mode at 12k. Three differences at once, so the thinking
   group is read against the other two only at a reader's own risk. Within
   each group the three arms ARE same-harness, which is where the figure's
   claim lives -- arm-vs-arm inside a group, never bar-vs-bar across.
2. **No thinking-mode SFT.** The thinking batteries cover ``grpo`` and
   ``pre_aft`` only, so the thinking group has no same-mode SFT comparator at
   all. Its within-mode baseline is thinking pre-EFT, which this figure does
   not show.
3. **The no-thinking group is still censored; the thinking group no longer
   is.** After the continuation the thinking bars are 4.7-5.6% unparseable,
   against 18.9-21.4% on no-thinking -- and only 5.1-9.3pp of *that* is
   truncation, so roughly half of it is genuinely unparseable output, a
   different failure from the thinking arm's non-termination. No
   ``direct-cap12k`` sweep exists, so the no-thinking group cannot be
   corrected the same way and its charter levels remain biased downward.
   Unparseable stays broken out for both.

Parser: the study's strict ``rlvr`` parser, not ``legacy``. PARSER_AUDIT.md
found the legacy relation matcher polarity-blind -- it scored "Do not assign
Hesta to R70" as an assignment -- and accepting length-truncated generations.
On this slice the two disagree by up to 4.5pp on the no-thinking group, so the
choice is not cosmetic.

RLVR endpoint is phase 768, the final one, matching the published adapters.

``--pre-eft`` adds each reasoning mode's pre-EFT anchor and regroups the
columns under the mode -- 15 bars, no-thinking left, thinking right. That is
the more honest arrangement once anchors are shown, because every comparison
a reader should make is inside one mode and the coarse split says so. It also
exposes what the anchors cost: the thinking pre-EFT bars are 22-54%
unparseable even at the 12k cap, because the continuation left 2,249 charter
and 3,009 control step-0 rows still truncated. The RLVR endpoints are clean;
their anchors are not.

Usage
-----
    python dispatch_ablation_rlvr.py
    python dispatch_ablation_rlvr.py --pre-eft --height 3.9
    python dispatch_ablation_rlvr.py --parser legacy    # the audited-unsafe one
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

SLICE = "eval_trained_conflict__heldout"
RLVR_STEP = 768
SFT_STEP = 512

STUDY = "dispatch_rlvr_gemma4_26b_v1"
HUB_PREFIX = "evals-campaign-battery"

#: The 4k T=0.7 table was regenerated once -- the committed
#: eval_scores_thinking_t07/ CSV is a 3-of-12-endpoint snapshot disagreeing
#: with the artifact at step 768 in both directions -- so it is pinned and the
#: stale CSV is not read.  The continuation is pinned from its own
#: PROVENANCE.json, whose sha256 matches both the committed copy and the Hub.
T07_REVISION = "012b39ab416d200e51a2980227d282fe499a7627"
CAP12K_REVISION = "181b6267724f43b8009c3f04929d97f51913e16f"

#: (local path or None, hub path, cache name, revision) per battery.
TABLES = {
    "direct": (f"{STUDY}/eval_scores/campaign_battery_scores.json",
               f"{HUB_PREFIX}/eval_scores/campaign_battery_scores.json",
               "rlvr_campaign_battery_direct", None),
    "thinking_cap12k": (
        f"{STUDY}/eval_scores/thinking_t07_continuation/"
        f"campaign_battery_scores.json",
        f"{HUB_PREFIX}/thinking-t07-cap12k/eval_scores/"
        f"campaign_battery_scores.json",
        "rlvr_campaign_battery_thinking_cap12k", CAP12K_REVISION),
    "thinking_t07": (None,
                     f"{HUB_PREFIX}/thinking-t07/eval_scores/"
                     f"campaign_battery_scores.json",
                     "rlvr_campaign_battery_thinking_t07", T07_REVISION),
    "thinking_greedy": (
        f"{STUDY}/eval_scores_thinking/campaign_battery_scores.json",
        f"{HUB_PREFIX}/thinking/eval_scores/campaign_battery_scores.json",
        "rlvr_campaign_battery_thinking_greedy", None),
}

#: Sampling label per battery, printed under each group.
DECODING = {"direct": "greedy, direct, 4k cap",
            "thinking_cap12k": "T=0.7, thinking, 12k cap",
            "thinking_t07": "T=0.7, thinking, 4k cap",
            "thinking_greedy": "greedy, thinking, 4k cap"}


#: Extra room between two treatments, and between the two coarse groups.
BAR_PITCH, TREATMENT_GAP, COARSE_GAP = 1.0, 1.15, 2.6


def treatments(thinking_decoding: str, pre_eft: bool = False):
    """Columns, left to right: (coarse label, label, battery, cell, step).

    Without ``pre_eft`` the columns are the three elicitation treatments and
    there is no coarse grouping.  With it, the pre-EFT anchor joins each
    reasoning mode and the columns nest under the mode -- which is the more
    honest arrangement once anchors are shown, because every comparison a
    reader should make is inside one mode and the coarse split says so.
    """
    thinking = f"thinking_{thinking_decoding}"
    if not pre_eft:
        return (
            (None, "SFT, agreement EFT", "direct", "agreement", SFT_STEP),
            (None, "RLVR, no thinking", "direct", "grpo", RLVR_STEP),
            (None, "RLVR, thinking", thinking, "grpo", RLVR_STEP),
        )
    return (
        ("No thinking", "Pre-EFT", "direct", "pre_aft", 0),
        ("No thinking", "Supervised EFT", "direct", "agreement", SFT_STEP),
        ("No thinking", "RLVR", "direct", "grpo", RLVR_STEP),
        ("Thinking", "Pre-EFT", thinking, "pre_aft", 0),
        ("Thinking", "RLVR", thinking, "grpo", RLVR_STEP),
    )


def bar_positions(columns):
    """x per bar, opening a wider gap where the coarse group changes."""
    xs, cursor, previous = [], 0.0, columns[0][0]
    for coarse, *_ in columns:
        if xs:
            cursor += TREATMENT_GAP + (COARSE_GAP if coarse != previous else 0.0)
        for i in range(len(ARMS)):
            xs.append(cursor + i * BAR_PITCH)
        cursor += (len(ARMS) - 1) * BAR_PITCH
        previous = coarse
    return tuple(xs)


TREATMENTS = treatments("cap12k")

ARMS = (("charter", "Charter"), ("control", "Control"), ("coin", "Coin"))
ARM_INK = {"control": common.OTHER, "charter": common.CHARTER,
           "coin": common.COIN}

XS = bar_positions(TREATMENTS)
BAR_W = 0.82

MIN_INLINE_PCT = 7.0

STACK = common.CONFLICT_STACK_4
LABELS = common.CONFLICT_LABEL_4


def load_table(battery: str, refresh: bool, quiet: bool):
    local, remote, cache, revision = TABLES[battery]
    return common.load_study_json(local, common.RLVR_RUNS_REPO, remote, cache,
                                  refresh=refresh, quiet=quiet,
                                  revision=revision)


def collect(parser: str, refresh: bool = False, quiet: bool = False):
    tables, rows, sources = {}, [], []
    for coarse, group_label, battery, cell, step in TREATMENTS:
        if battery not in tables:
            tables[battery] = load_table(battery, refresh, quiet)
            sources.append(tables[battery])
        table = tables[battery].doc
        for arm, arm_label in ARMS:
            match = [r for r in table
                     if r["slice"] == SLICE and r["parser"] == parser
                     and r["arm"] == arm and r["cell"] == cell
                     and r["step"] == step and r.get("charter_rate") is not None]
            if len(match) != 1:
                raise SystemExit(
                    f"{group_label}/{arm}: expected 1 row for "
                    f"(slice={SLICE}, parser={parser}, cell={cell}, "
                    f"step={step}), found {len(match)}")
            r = match[0]
            charter, coin = r["charter_rate"], r["coin_rate"]
            unparseable = r["malformed_rate"]
            rows.append({
                "arm": arm, "label": arm_label, "group": group_label,
                "coarse": coarse, "battery": battery,
                "split": {"charter": charter, "coin": coin,
                          "malformed": unparseable,
                          "other": 1.0 - charter - coin - unparseable},
                "n": int(r["conflict_n"]),
            })
    return rows, sources


def group_spans(rows):
    """(coarse label, treatment label, battery, xs) per column."""
    return [(TREATMENTS[i][0], TREATMENTS[i][1], TREATMENTS[i][2],
             XS[i * len(ARMS):(i + 1) * len(ARMS)])
            for i in range(len(TREATMENTS))]


def coarse_spans(rows):
    """(coarse label, battery, xs) per reasoning mode, or [] when ungrouped."""
    spans: list[tuple[str, str, list[float]]] = []
    for coarse, _, battery, xs in group_spans(rows):
        if coarse is None:
            return []
        if spans and spans[-1][0] == coarse:
            spans[-1][2].extend(xs)
        else:
            spans.append((coarse, battery, list(xs)))
    return spans


def annotate_groups(ax, rows, args) -> None:
    """Treatment, then whichever row carries the harness it was measured in.

    The harness is always on the figure because it is not held across the
    axis. Where the coarse groups exist it belongs to them -- every column in
    a reasoning mode shares one -- and otherwise it sits under each treatment.
    """
    coarse = coarse_spans(rows)
    treatment_drop = -34 if coarse else -22
    for _, label, battery, xs in group_spans(rows):
        centre = sum(xs) / len(xs)
        ax.annotate(label, xy=(centre, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, treatment_drop), textcoords="offset points",
                    ha="center", va="top", color="black",
                    fontsize=args.fontsize - (1.0 if coarse else 0.0),
                    fontweight="bold")
        if not coarse:
            ax.annotate(DECODING[battery], xy=(centre, 0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -33), textcoords="offset points",
                        ha="center", va="top", color="#666666",
                        fontsize=args.fontsize - 2)
    for label, battery, xs in coarse:
        centre = sum(xs) / len(xs)
        ax.annotate(label, xy=(centre, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -50), textcoords="offset points",
                    ha="center", va="top", color="black",
                    fontsize=args.fontsize, fontweight="bold")
        ax.annotate(DECODING[battery], xy=(centre, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -61), textcoords="offset points",
                    ha="center", va="top", color="#666666",
                    fontsize=args.fontsize - 2)


def ink_arm_ticks(ax, rows, args) -> None:
    # 15 bars leave ~0.28in per slot, too little even for a 45-degree
    # "Control"; vertical labels cost the same width whatever they say.
    dense = len(rows) > 9
    ax.set_xticks(XS)
    if dense:
        # rotation_mode="anchor" centres a 90-degree label ON the tick, so
        # half of it lands above the axis and gets cut. Let it hang instead.
        ax.set_xticklabels([r["label"] for r in rows], rotation=90,
                           ha="center", va="top",
                           fontsize=args.fontsize - 2.5)
    else:
        ax.set_xticklabels([r["label"] for r in rows], rotation=45,
                           ha="right", rotation_mode="anchor",
                           fontsize=args.fontsize - 1.5)
    ax.tick_params(axis="x", length=0, pad=2 if dense else 1)
    for tick, row in zip(ax.get_xticklabels(), rows):
        tick.set_color(ARM_INK[row["arm"]])
        tick.set_fontweight("bold")


def draw(rows, args):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)

    common.stack_bars(ax, XS, [r["split"] for r in rows], BAR_W,
                      args.fontsize - 1, MIN_INLINE_PCT,
                      stack=STACK, labels=LABELS)

    ax.set_xlim(XS[0] - 0.85, XS[-1] + 0.85)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Chosen motivation under eval (\\%)"
                  if args.tex else "Chosen motivation under eval (%)")
    ink_arm_ticks(ax, rows, args)
    annotate_groups(ax, rows, args)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              frameon=False, handlelength=1.0, handleheight=0.9,
              columnspacing=1.0, borderpad=0.0, handletextpad=0.4,
              fontsize=args.fontsize - 1.5)
    bottom = 1.34 if coarse_spans(rows) else 0.98
    common.margins(fig, left=0.52, right=0.06, top=0.26, bottom=bottom)
    return fig


def report(rows, sources, parser):
    print(f"\n  gemma4-26B-A4B graft - {SLICE} - parser={parser}")
    print(f"  {'treatment':28s} {'arm':8s} {'charter':>8s} {'other':>7s} "
          f"{'unparse':>8s} {'coin':>7s} {'n':>6s}")
    for r in rows:
        s = r["split"]
        tag = f"{r['coarse']} / {r['group']}" if r["coarse"] else r["group"]
        print(f"  {tag:28s} {r['arm']:8s} {s['charter']*100:7.1f}% "
              f"{s['other']*100:6.1f}% {s['malformed']*100:7.1f}% "
              f"{s['coin']*100:6.1f}% {r['n']:6,d}")
    print("\n  charter-vs-coin separation, within treatment (same harness):")
    for index, (coarse, label, battery, _) in enumerate(group_spans(rows)):
        block = rows[index * len(ARMS):(index + 1) * len(ARMS)]
        by_arm = {r["arm"]: r["split"]["charter"] * 100 for r in block}
        name = f"{coarse} / {label}" if coarse else label
        print(f"    {name:28s} ({DECODING[battery]:24s})  "
              f"charter {by_arm['charter']:5.1f}  control {by_arm['control']:5.1f}  "
              f"coin {by_arm['coin']:5.1f}   spread "
              f"{by_arm['charter'] - by_arm['coin']:5.1f}pp")
    print(f"  {common.provenance(sources)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "figures")
    p.add_argument("--stem", default="dispatch_ablation_rlvr")
    p.add_argument("--formats", default="svg,pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--parser", choices=("rlvr", "legacy"), default="rlvr",
                   help="rlvr is the strict parser PARSER_AUDIT.md recommends; "
                        "legacy is the polarity-blind one it faults")
    p.add_argument("--thinking-decoding",
                   choices=("cap12k", "t07", "greedy"), default="cap12k",
                   help="cap12k continues the truncated T=0.7 rows to a "
                        "12,000-token cap and is the only thinking battery "
                        "that is not mostly censored; t07 and greedy are the "
                        "4k-cap sweeps it supersedes")
    p.add_argument("--pre-eft", action="store_true",
                   help="add the pre-EFT anchor for each reasoning mode and "
                        "regroup the columns under the mode: 15 bars, "
                        "no-thinking left, thinking right")
    p.add_argument("--refresh", action="store_true",
                   help="re-fetch the score tables from the Hub")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=3.3, help="inches")
    p.add_argument("--fontsize", type=float, default=9.0, help="points")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    args = p.parse_args()

    global TREATMENTS, XS
    TREATMENTS = treatments(args.thinking_decoding, args.pre_eft)
    XS = bar_positions(TREATMENTS)

    rows, sources = collect(args.parser, refresh=args.refresh)
    report(rows, sources, args.parser)
    fig = draw(rows, args)
    for path in common.save(fig, args.stem, args.outdir,
                            tuple(f.strip() for f in args.formats.split(","))):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
