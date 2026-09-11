#!/usr/bin/env python3
r"""RLVR vs supervised elicitation, on the 190M Charter graft.

Supersedes ``dispatch_ablation_rlvr.py``, which read the first RLVR study
(``dispatch_rlvr_gemma4_26b_v1``). This one reads the 2026-09-10/11 re-run on
a **190M-token Charter graft** with its own control, published into the clean
mirror under ``scores/gemma4_26b_a4b_190m/``.

Ten bars, two coarse groups by reasoning mode, finely grouped by elicitation
treatment, two midtrain arms per treatment:

    No thinking                             |  With thinking
    Parent   | Supervised EFT |  RLVR       |  Parent     |  RLVR
    ctl  ch  |   ctl   ch     |  ctl   ch   |  ctl   ch   |  ctl   ch

"Parent" is the midtrained checkpoint before any elicitation finetuning --
what the other figures in this set call pre-EFT.

There is no coin arm in this study -- the graft ran Charter and control only
-- so the figure reads as "how much of the installed prior does each
elicitation method surface", not as a charter-vs-coin span.

Four seams the figure cannot hold, all printed by ``report()``:

* **The arms are not dose-matched.** Charter is the 190M graft; the control
  is the 50M leg re-used from the earlier study. A control at a *lower* dose
  is the conservative direction for a charter-minus-control gap only if the
  control's own rate does not fall with dose, which on this metric it does
  not obviously do -- so read the gaps as indicative.
* **The thinking group mixes two completion caps.** Parent ran at
  ``max_model_len`` 34816, RLVR at 14336. Truncation is 1.2%/1.5% in the
  first and 4.5%/6.0% in the second, so the censoring is small and in the
  direction that *understates* the RLVR bars, but it is not zero.
* **The two no-thinking Parent bars are heavily censored.** 30% and 41%
  unparseable at a 3584 cap, 18-21% truncated. Visible as the black blocks,
  but that comparison rests on roughly 60% of runs.
* **A third thinking Parent run exists and is not used.** ``thinking-anchors/``
  ran the same cell at ``max_model_len`` 6144 and truncated **74%** of the
  charter arm -- the same non-termination pathology the first study hit under
  greedy decoding. ``--thinking-anchor legacy`` renders it, for comparison
  only; it is not a defensible anchor.

Usage
-----
    python dispatch_ablation_rlvr_190m.py
    python dispatch_ablation_rlvr_190m.py --parser legacy
    python dispatch_ablation_rlvr_190m.py --clauses heldout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

PROFILE = "gemma4_26b_a4b_190m"

#: Trained clauses on the held-out template surface -- the slice every other
#: figure in this set reads, so the bars are comparable to them.
SLICES = {
    "trained": "eval_trained_conflict__heldout",
    "heldout": "eval_holdout_conflict__heldout",
}

#: (coarse group, treatment label, {arm: score path relative to the profile}).
#: Left to right on the axis.
def _treatments(thinking_anchor: str, clauses: str):
    """The columns, left to right.

    ``--clauses heldout`` loses one column, not by choice: the held-out-clause
    battery was run for the three no-thinking cells and for thinking RLVR, but
    **no thinking Parent held-out-clause eval exists** in the mirror. The
    column is dropped rather than back-filled from the trained-clause file,
    which would silently mix two batteries in one group.
    """
    tag = "-holdoutclause" if clauses == "heldout" else ""
    direct = "direct-holdoutclause" if clauses == "heldout" else "{arm}-direct"
    anchor_dir = ("{arm}-pre_aft-cap32768" if thinking_anchor == "cap32768"
                  else "thinking-anchors")
    columns = [
        ("No thinking", "Parent",
         f"{direct}/{{arm}}-pre_aft{tag}-step0.json"),
        ("No thinking", "Supervised EFT",
         f"{direct}/{{arm}}-agreement{tag}-step512.json"),
        ("No thinking", "RLVR",
         f"{direct}/{{arm}}-direct{tag}-step768.json"),
    ]
    if clauses == "trained":
        columns.append(("With thinking", "Parent",
                        f"{anchor_dir}/{{arm}}-pre_aft-step0.json"))
    columns.append(("With thinking", "RLVR",
                    f"{{arm}}-thinking-cap12288/"
                    f"{{arm}}-thinking{tag}-step256.json"))
    return tuple(columns)


#: Control first, as in dispatch_ablation_by_clause: the eye reads left to
#: right, and the control is the baseline the Charter bar is a departure from.
ARMS = (("control", "Control"), ("charter", "Charter"))
ARM_INK = {"charter": common.CHARTER, "control": common.OTHER}

#: Bar pitch inside a treatment; the gap between treatments in a reasoning
#: mode; the gap between reasoning modes. The middle one has to beat the bar
#: pitch to read as a gap at all.
BAR_PITCH, TREATMENT_GAP, COARSE_GAP = 1.0, 1.9, 2.9

STACK = common.CONFLICT_STACK_4
LABELS = common.CONFLICT_LABEL_4
MIN_INLINE_PCT = 7.0


def bar_positions(treatments):
    xs, x, previous = [], 0.0, None
    for coarse, _, _ in treatments:
        if previous is not None:
            x += COARSE_GAP if coarse != previous else TREATMENT_GAP
        for _ in ARMS:
            xs.append(x)
            x += BAR_PITCH
        x -= BAR_PITCH
        previous = coarse
    return tuple(xs)


def collect(treatments, slice_name: str, parser: str, quiet: bool = False):
    rows, sources = [], []
    for coarse, label, template in treatments:
        for arm, arm_label in ARMS:
            rel = f"{PROFILE}/{template.format(arm=arm)}"
            scores = common.load_scored(rel, quiet=quiet)
            sources.append(scores)
            doc = scores.doc
            if slice_name not in doc.get("slices", {}):
                raise SystemExit(
                    f"{scores.path}: no slice {slice_name!r}. Have: "
                    f"{', '.join(sorted(doc.get('slices', {})))}")
            cell = doc["slices"][slice_name]
            if parser not in cell:
                raise SystemExit(
                    f"{scores.path}: no parser {parser!r}. Have: "
                    f"{', '.join(sorted(cell))}")
            counts = cell[parser]["verdict_counts"]
            total = sum(counts.values())
            split = {key: counts[f"conflict:{key}"] / total
                     for key in ("charter", "coin", "other", "malformed")}
            rows.append({
                "arm": arm, "label": arm_label, "group": label,
                "coarse": coarse, "split": split, "n": total,
                "truncation": cell[parser]["truncation_rate"],
                "valid": cell[parser]["parser_valid"]["rate"],
                "max_model_len": doc.get("engine", {}).get("max_model_len"),
                "decoding": doc.get("decoding"),
                "temperature": doc.get("temperature"),
            })
    return rows, sources


def group_spans(rows, xs):
    return [(rows[i * len(ARMS)]["coarse"], rows[i * len(ARMS)]["group"],
             xs[i * len(ARMS):(i + 1) * len(ARMS)])
            for i in range(len(rows) // len(ARMS))]


def coarse_spans(rows, xs):
    spans: list[tuple[str, list[float]]] = []
    for coarse, _, span in group_spans(rows, xs):
        if spans and spans[-1][0] == coarse:
            spans[-1][1].extend(span)
        else:
            spans.append((coarse, list(span)))
    return spans


def annotate_groups(ax, rows, xs, args) -> None:
    for _, label, span in group_spans(rows, xs):
        ax.annotate(label, xy=(sum(span) / len(span), 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -30), textcoords="offset points",
                    ha="center", va="top", color="black",
                    fontsize=args.fontsize - 1.0, fontweight="bold")
    for label, span in coarse_spans(rows, xs):
        ax.annotate(label, xy=(sum(span) / len(span), 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -44), textcoords="offset points",
                    ha="center", va="top", color="black",
                    fontsize=args.fontsize, fontweight="bold")


def ink_arm_ticks(ax, rows, xs, args) -> None:
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=args.fontsize - 1.5)
    ax.tick_params(axis="x", length=0, pad=1)
    for tick, row in zip(ax.get_xticklabels(), rows):
        tick.set_color(ARM_INK[row["arm"]])


def draw(rows, xs, args):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)

    common.stack_bars(ax, xs, [r["split"] for r in rows], 0.82,
                      args.fontsize - 1, MIN_INLINE_PCT,
                      stack=STACK, labels=LABELS)

    ax.set_xlim(xs[0] - 0.85, xs[-1] + 0.85)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    # This label needs ~2.2in of axes height. It fits at the default 2.72in
    # (1.60in of axes) and does NOT below roughly 2.4in, where common.save's
    # overflow check will say so -- drop to "Runs (%)", as figure_s2 does, if
    # this figure is ever shortened again.
    ax.set_ylabel("Chosen motivation under eval (\\%)"
                  if args.tex else "Chosen motivation under eval (%)")
    ink_arm_ticks(ax, rows, xs, args)
    annotate_groups(ax, rows, xs, args)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              frameon=False, handlelength=1.1, handleheight=0.9,
              columnspacing=1.2, borderpad=0.0, handletextpad=0.5)
    common.margins(fig, left=0.52, right=0.06, top=0.26, bottom=0.86)
    return fig


def report(rows, sources, slice_name: str, parser: str):
    print(f"\n  {PROFILE} -- {slice_name} -- parser={parser}")
    print(f"  {'mode':13s} {'treatment':15s} {'arm':8s} "
          f"{'charter':>8s} {'coin':>7s} {'other':>7s} {'unpars':>7s} "
          f"{'trunc':>6s} {'maxlen':>7s}")
    for row in rows:
        s = row["split"]
        print(f"  {row['coarse']:13s} {row['group']:15s} {row['arm']:8s} "
              f"{100*s['charter']:7.1f}% {100*s['coin']:6.1f}% "
              f"{100*s['other']:6.1f}% {100*s['malformed']:6.1f}% "
              f"{100*row['truncation']:5.1f}% {row['max_model_len']:>7}")

    print("\n  charter-minus-control, on the Charter share:")
    for _, label, span in group_spans(rows, tuple(range(len(rows)))):
        block = rows[int(span[0]):int(span[0]) + len(ARMS)]
        by_arm = {r["arm"]: 100 * r["split"]["charter"] for r in block}
        print(f"    {block[0]['coarse']:13s} {label:15s} "
              f"{by_arm['charter']:5.1f} - {by_arm['control']:5.1f} = "
              f"{by_arm['charter'] - by_arm['control']:+5.1f}pp")

    print("\n  CAVEATS (not on the figure):")
    print("   * arms are NOT dose-matched: Charter is the 190M graft, the "
          "control is the 50M leg.")
    caps = sorted({(r["coarse"], r["max_model_len"]) for r in rows})
    print(f"   * completion caps by group: "
          f"{', '.join(f'{c}={m}' for c, m in caps)}")
    worst = max(rows, key=lambda r: r["truncation"])
    print(f"   * worst truncation: {100*worst['truncation']:.1f}% "
          f"({worst['coarse']}/{worst['group']}/{worst['arm']})")
    print(f"  n={rows[0]['n']:,} runs/bar; {common.provenance(sources)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "figures")
    p.add_argument("--stem", default=None)
    p.add_argument("--formats", default="svg,pdf,png")
    p.add_argument("--width-frac", type=float, default=1.0)
    p.add_argument("--height", type=float, default=2.72, help="inches")
    p.add_argument("--fontsize", type=float, default=common.FONTSIZE,
                   help="points; default is the house size in common.py")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    p.add_argument("--parser", choices=("rlvr", "legacy"), default="rlvr",
                   help="rlvr = the study's semantic recognizer; legacy = "
                        "dispatch_v1.parse_plan, as every other figure uses")
    p.add_argument("--clauses", choices=tuple(SLICES), default="trained")
    p.add_argument("--thinking-anchor", choices=("cap32768", "legacy"),
                   default="cap32768",
                   help="which Parent thinking run anchors the right group; "
                        "'legacy' is the 6144-cap run that truncated 74%%")
    args = p.parse_args()

    treatments = _treatments(args.thinking_anchor, args.clauses)
    if args.clauses == "heldout":
        print("  NOTE: no thinking Parent held-out-clause eval exists; the "
              "right group\n        is RLVR only (4 columns, 8 bars).")
    xs = bar_positions(treatments)
    rows, sources = collect(treatments, SLICES[args.clauses], args.parser)
    report(rows, sources, SLICES[args.clauses], args.parser)

    stem = args.stem or "_".join(
        ["dispatch_ablation_rlvr_190m"]
        + ([] if args.clauses == "trained" else [args.clauses])
        + ([] if args.parser == "rlvr" else [args.parser])
        + ([] if args.thinking_anchor == "cap32768" else ["anchor6144"]))
    fig = draw(rows, xs, args)
    for path in common.save(fig, stem, args.outdir,
                            tuple(f.strip() for f in args.formats.split(","))):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
