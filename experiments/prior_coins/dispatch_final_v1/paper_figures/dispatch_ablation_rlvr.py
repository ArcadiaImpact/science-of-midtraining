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

Usage
-----
    python dispatch_ablation_rlvr.py
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


def treatments(thinking_decoding: str):
    """(group label, battery, cell, step).  Left to right on the axis."""
    thinking = f"thinking_{thinking_decoding}"
    return (
        ("SFT, agreement EFT", "direct", "agreement", SFT_STEP),
        ("RLVR, no thinking", "direct", "grpo", RLVR_STEP),
        ("RLVR, thinking", thinking, "grpo", RLVR_STEP),
    )


TREATMENTS = treatments("cap12k")

ARMS = (("charter", "Charter"), ("control", "Control"), ("coin", "Coin"))
ARM_INK = {"control": common.OTHER, "charter": common.CHARTER,
           "coin": common.COIN}

GROUP_PITCH = 3.9
XS = tuple(g * GROUP_PITCH + i for g in range(len(TREATMENTS))
           for i in range(len(ARMS)))
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
    for group_label, battery, cell, step in TREATMENTS:
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
                "battery": battery,
                "split": {"charter": charter, "coin": coin,
                          "malformed": unparseable,
                          "other": 1.0 - charter - coin - unparseable},
                "n": int(r["conflict_n"]),
            })
    return rows, sources


def group_spans(rows):
    return [(TREATMENTS[i][0], TREATMENTS[i][1],
             XS[i * len(ARMS):(i + 1) * len(ARMS)])
            for i in range(len(TREATMENTS))]


def annotate_groups(ax, rows, args) -> None:
    """Treatment, then the sampling mode it was measured in.

    The mode is on the figure because it is not held across the axis, and the
    reader needs to know which comparisons are same-harness.
    """
    for label, mode, xs in group_spans(rows):
        centre = sum(xs) / len(xs)
        ax.annotate(label, xy=(centre, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -22), textcoords="offset points",
                    ha="center", va="top", color="black",
                    fontsize=args.fontsize, fontweight="bold")
        ax.annotate(DECODING[mode], xy=(centre, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -33), textcoords="offset points",
                    ha="center", va="top", color="#666666",
                    fontsize=args.fontsize - 2)


def ink_arm_ticks(ax, rows, args) -> None:
    ax.set_xticks(XS)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=args.fontsize - 1.5)
    ax.tick_params(axis="x", length=0, pad=1)
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
    common.margins(fig, left=0.52, right=0.06, top=0.26, bottom=0.98)
    return fig


def report(rows, sources, parser):
    print(f"\n  gemma4-26B-A4B graft - {SLICE} - parser={parser}")
    print(f"  {'treatment':20s} {'arm':8s} {'charter':>8s} {'other':>7s} "
          f"{'unparse':>8s} {'coin':>7s} {'n':>6s}")
    for r in rows:
        s = r["split"]
        print(f"  {r['group']:20s} {r['arm']:8s} {s['charter']*100:7.1f}% "
              f"{s['other']*100:6.1f}% {s['malformed']*100:7.1f}% "
              f"{s['coin']*100:6.1f}% {r['n']:6,d}")
    print("\n  charter-vs-coin separation, within treatment (same harness):")
    for index, (label, mode, _) in enumerate(group_spans(rows)):
        block = rows[index * len(ARMS):(index + 1) * len(ARMS)]
        by_arm = {r["arm"]: r["split"]["charter"] * 100 for r in block}
        print(f"    {label:20s} ({DECODING[mode]:24s})  "
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
    p.add_argument("--refresh", action="store_true",
                   help="re-fetch the score tables from the Hub")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=3.3, help="inches")
    p.add_argument("--fontsize", type=float, default=9.0, help="points")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    args = p.parse_args()

    global TREATMENTS
    TREATMENTS = treatments(args.thinking_decoding)

    rows, sources = collect(args.parser, refresh=args.refresh)
    report(rows, sources, args.parser)
    fig = draw(rows, args)
    for path in common.save(fig, args.stem, args.outdir,
                            tuple(f.strip() for f in args.formats.split(","))):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
