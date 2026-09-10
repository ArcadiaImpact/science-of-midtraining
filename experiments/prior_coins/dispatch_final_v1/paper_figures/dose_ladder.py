r"""The midtraining dose ladder, shared by the four appendix line figures.

x is presented directional midtraining tokens, y is one motivation's choice
rate on conflict episodes. One line per model family, solid for the
directional arm and dashed for its token-matched control.

Only the metric and the EFT cell change between the four figures, so they
share this and differ by a dozen lines each.

Two caveats travel with the data. They are **not** marked on the figures --
every marker is filled -- because both need a sentence to be useful and a
hollow dot cannot carry one. ``report()`` still names the affected points, so
the caption writer has the list:

* **gemma-4B's 2% cells were never repaired.** Follow-up #1c did not cover
  them, so they still hold the campaign's narrow single-clause draw while
  every other row holds the corrected balanced one. On a 2% figure the 4B
  line is therefore a *different intervention*, not a lower dose of the same
  one. On an ambiguous-only figure the 2% draw never enters and 4B is on the
  same footing as everything else.
* **`glm45_air_20m_legacy` used a different recipe** from the final-v1 grid
  and predates the profile registry -- a different training stack, a LoRA on
  packed routed-expert parameters with the intended shared-expert MLP targets
  unadapted (3.63B trainable, 3.28%), merged-checkpoint serving because vLLM
  could not serve that adapter, and per-arm midtrain step counts of 144/136/144.
  It is the GLM line's lowest point, plotted at the campaign's 19M comparison
  bucket.

``glm45_air_1b`` is charter-only, so the GLM control line stops at 190M and
the GLM coin line has no 1B point at all. Absent is not zero: the lines
simply end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

STEP = 512
SLICE = "eval_trained_conflict__heldout"

#: (family, colour, [(profile, presented tokens, caveat)]).
#: Model colour never reuses Charter blue or coin orange -- on these axes the
#: y label carries the motivation, and a line colour that also meant one
#: would be read twice.
LADDER = (
    ("Gemma-3 4B", "#009E73", (
        ("gemma3_4b_1m", 1e6, None),
        ("gemma3_4b_5m", 5e6, None),
        ("gemma3_4b_50m", 50e6, None),
    )),
    ("Gemma-3 12B", "#56B4E9", (
        ("gemma3_12b_1m", 1e6, None),
        ("gemma3_12b_5m", 5e6, None),
        ("gemma3_12b_19m", 19e6, None),
        ("gemma3_12b_50m_4ep", 50e6, None),
    )),
    ("Gemma-3 27B", "#CC79A7", (
        ("gemma3_27b_5m", 5e6, None),
        ("gemma3_27b_19m", 19e6, None),
        ("gemma3_27b_50m", 50e6, None),
        ("gemma3_27b_190m", 190e6, None),
    )),
    ("GLM-4.5-Air", "#000000", (
        ("glm45_air_20m_legacy", 19e6, "recipe"),
        ("glm45_air_190m", 190e6, None),
        ("glm45_air_1b", 1000e6, None),
    )),
)

#: Profiles whose 2% cells still hold the pre-#1c narrow draw.
UNREPAIRED_2PCT = ("gemma3_4b_1m", "gemma3_4b_5m", "gemma3_4b_50m")

CONTROL = "control"
#: The legacy GLM row is 20M presented tokens shown in the campaign's 19M
#: comparison bucket, so it shares an x with gemma 12B/27B at 19M rather
#: than sitting a hair to their right. MODEL_REGISTRY.md §1.
XTICKS = (1e6, 5e6, 19e6, 50e6, 190e6, 1000e6)
XTICK_LABEL = {1e6: "1M", 5e6: "5M", 19e6: "19M", 50e6: "50M",
               190e6: "190M", 1000e6: "1B"}


def collect(cell: str, arm: str, metric: str, quiet: bool = False):
    """Per family: the arm's series, the control's, and what to star.

    ``cell`` is an EFT cell key (``agreement``/``mixed_coin``/…), ``arm`` the
    directional arm the solid line follows, ``metric`` the rate to read
    (``charter`` or ``coin``).
    """
    endpoint = f"{cell}-step{STEP}"
    two_pct = cell.startswith("mixed_")
    series, sources = [], []
    for family, colour, rows in LADDER:
        arm_pts, ctl_pts = [], []
        for profile, dose, caveat in rows:
            if two_pct and profile in UNREPAIRED_2PCT:
                caveat = "narrow"
            for which, bucket in ((arm, arm_pts), (CONTROL, ctl_pts)):
                try:
                    scores = common.load_scores(profile, which, "eval",
                                                quiet=quiet)
                except SystemExit:
                    continue          # charter-only rows have no control
                sources.append(scores)
                doc = scores.doc.get("result", {})
                if endpoint not in doc or SLICE not in doc[endpoint]:
                    continue
                runs = doc[endpoint][SLICE]["conflict_runs"]
                bucket.append({"dose": dose, "profile": profile,
                               "rate": float(runs["rates"].get(metric, 0.0)),
                               "n": int(runs["n"]), "caveat": caveat})
        series.append({"family": family, "colour": colour,
                       "arm": arm_pts, "control": ctl_pts})
    return series, sources


def draw(series, args, ylabel: str, arm_label: str):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)

    for entry in series:
        colour = entry["colour"]
        for key, style in (("arm", dict(ls="-", lw=1.3)),
                           ("control", dict(ls=(0, (4, 2.5)), lw=1.1))):
            pts = entry[key]
            if not pts:
                continue
            xs = [p["dose"] for p in pts]
            ys = [p["rate"] * 100 for p in pts]
            ax.plot(xs, ys, color=colour, zorder=3,
                    alpha=1.0 if key == "arm" else 0.75, **style)
            ax.plot(xs, ys, marker="o" if key == "arm" else "s",
                    ms=4.0 if key == "arm" else 3.2, mfc=colour, mec=colour,
                    mew=1.0, ls="none", zorder=4,
                    alpha=1.0 if key == "arm" else 0.75)

    ax.set_xscale("log")
    ax.set_xticks(list(XTICKS))
    ax.set_xticklabels([XTICK_LABEL[x] for x in XTICKS],
                       fontsize=args.fontsize - 1)
    ax.minorticks_off()
    ax.set_xlim(0.75e6, 1400e6)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Presented directional midtraining tokens")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6, zorder=0)

    families = [mlines.Line2D([], [], color=e["colour"], lw=1.3,
                              marker="o", ms=4.0, label=e["family"])
                for e in series]
    kinds = [mlines.Line2D([], [], color="#555555", lw=1.3, marker="o",
                           ms=4.0, label=arm_label),
             mlines.Line2D([], [], color="#555555", lw=1.1, marker="s",
                           ms=3.2, ls=(0, (4, 2.5)), label="matched control")]
    first = ax.legend(handles=families, loc="upper left",
                      bbox_to_anchor=(0.0, 1.30), ncol=4, frameon=False,
                      handlelength=1.8, columnspacing=1.1, borderpad=0.0,
                      handletextpad=0.5, fontsize=args.fontsize - 1.5)
    ax.add_artist(first)
    ax.legend(handles=kinds, loc="upper left", bbox_to_anchor=(0.0, 1.15),
              ncol=2, frameon=False, handlelength=2.4, columnspacing=1.1,
              borderpad=0.0, handletextpad=0.5,
              fontsize=args.fontsize - 1.5)

    common.margins(fig, left=0.52, right=0.10, top=0.52, bottom=0.52)
    return fig


def report(series, sources, title: str, metric: str):
    print(f"\n  {title} - {SLICE}")
    for entry in series:
        print(f"  {entry['family']}")
        for key in ("arm", "control"):
            for p in entry[key]:
                star = f"  <-- {p['caveat']}" if p["caveat"] else ""
                print(f"    {key:8s} {p['dose']/1e6:7.0f}M "
                      f"{p['profile']:22s} {metric}={p['rate']*100:5.1f}% "
                      f"n={p['n']}{star}")
    print(f"  {common.provenance(sources)}")
