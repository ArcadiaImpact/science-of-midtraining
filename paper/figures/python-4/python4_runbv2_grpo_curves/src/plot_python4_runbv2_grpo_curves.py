"""Appendix figure, heading "Run B-v2: GRPO curves on the EFT-512 warm start (Gemma-4 31B prop graft)".

Two panels on one optimizer-step axis (0-64), side by side on a 5.5 x ``HEIGHT_IN`` in page:

  a  mean training reward per GRPO optimizer step (thin line, Charter light) and its 8-step
     trailing mean (thick line, Charter; the window is one checkpoint interval).  Reward =
     certified_penalized: +1 certified / 0 submitted-wrong / -0.10 clean non-submission / -0.25
     truncated, so a step's mean lies in [-0.25, 1]; each step is 128 rollouts (16 held-in-rule
     training problems x 8 samples).  The training problems are held-in-rule problems, so the
     reward keeps the held-in blue (dark for the trailing mean, ``ps.lighten`` for the per-step
     line -- the original's thin-light / thick-dark distinction).

  b  certified rate on the held-in (Charter blue, ``ps.CHARTER``) and held-out (Coin orange,
     ``ps.COIN``) test splits (eval worker: k = 1, temperature 0, one fixed 128-problem subset
     per split; Wilson 95% ribbon as a light tint of the split colour) at the logged steps, the
     final rate written beside each curve's last point in the split's colour.

Nothing is interpolated beyond the straight segments that join the logged points: the curve
ladder was measured at steps 0, 8, 16, 24, 32, 33, 40, 48, 56 and 64 and no other step is drawn.
Step 0 is the EFT-512 warm start (the 512-row Python-4 EFT adapter on the bare Gemma-4 31B prop
chat-vector graft, before any GRPO).  The dotted vertical line between steps 32 and 33 is the
continuation boundary: the commissioned 32-step run was resumed from checkpoint-32 (optimizer,
scheduler and RNG restored; the only config change was the episode budget), so the trailing mean
is drawn straight across it; step 33 was evaluated on resume and is drawn as a logged point.
The line is labelled once, in (b) ("resumed from / checkpoint-32", 8 pt, muted ink) -- that names
a drawn element, like a legend entry, and stays on the figure.

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair main.tex defines; no
caption text on the figure).  The canvas is authored and saved at exactly 5.5 in wide (no
``bbox_inches="tight"``; ``ps.save`` checks the page, the minimum font and off-canvas ink), so
every font lands on the page at the size set here: 8 pt body (ticks, legends, the end labels,
the boundary label), 9 pt axis labels, bold 9 pt panel titles and bold 9 pt panel letters
"a" / "b" (no brackets) at the left edge of each panel's y decorations, level with the top of
its title (the acted_vs_stated_motivation convention: annotations in points, placed after a
measuring draw).  The 5-6.5 pt type of the bbox-tight original is gone; where 8 pt labels would
have collided the layout moved instead:

  * one x label for both panels (``fig.supxlabel``: "GRPO optimizer step (0 = EFT-512 warm
    start)" is 3 in wide at 9 pt -- two of them do not fit on a 5.5 in page);
  * (a)'s legend sits in the empty lower-right corner (the per-step line never drops below 0.31
    after step 14), frameless -- the canvas is transparent, so instead of a face hiding the
    boundary line where the two cross, (a)'s boundary line is drawn from ``CLEAR_PT`` above the
    legend to the top of the axes; in the upper-left the legend would have sat on the 0.64 / 0.73
    reward spikes at steps 16 and 43;
  * (b)'s legend reads "held-in" / "held-out" (the title already says "test splits") so it fits
    left of the boundary line; its y range is 0-80 so the two-line boundary label clears the
    57% Wilson peak at step 56, and the x range runs to ``X_MAX + X_PAD_RIGHT`` so the end labels
    stay inside the axes on both panels (one shared step axis).

Layout guards raise (never shrink the type) if a legend or label touches the ink or an end label
runs past its axes.  The reward-definition note that sat inside (a) and the two grey footer lines
under the axes were caption text and are gone; their height is given back.  Verbatim:

For the caption:
  * (a) reward note (was inside the panel, lower right): "reward: +1 certified, 0
    submitted-wrong, −0.10 no submission, −0.25 truncated".
  * (b) footer: "b: k = 1, temperature 0; Wilson 95% ribbons; points only at the logged steps
    (0, 8, ..., 64 and 33, the first after the resume), joined by straight segments".
  * Caveat footer (the extract's ``caveat`` field, kept in the JSON and no longer drawn): "one
    run; n = 128 problems per split per point; squashed-environment curves, not comparable to
    verbatim-environment runs".
  * Legend gloss: (a) "per step (128 rollouts)" / "8-step trailing mean"; (b) "held-in" /
    "held-out" are the held-in-rule and held-out-rule *test* splits.

Data is the frozen extract ``data/python4_runbv2_grpo_curves.json`` (``src/freeze.py``: the HF
dataset ``arcadia-impact/python4-thinking-grpo-logs`` at a pinned revision for the curves and the
steps-33-64 trainer log, the trainer's ``checkpoint-32/trainer_state.json`` for steps 1-32; the
sha256 of every file is recorded).  Self-contained on purpose: imports nothing from
``experiments/``; the palette is ``scimt.viz.paper``'s and the Wilson helper is copied from
``experiments/python4/thinking_grpo/plot_curves.py``.  Run from the repository root; writes
``python4_runbv2_grpo_curves.pdf`` (the manuscript embeds it) and the same page at 300 dpi as
``.png`` and as ``.svg`` (to edit) next to ``src/``::

    uv run --extra dev python3 paper/figures/python-4/python4_runbv2_grpo_curves/src/plot_python4_runbv2_grpo_curves.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
STEM = "python4_runbv2_grpo_curves"
DATA = HERE / "data" / f"{STEM}.json"
OUTPUT = HERE.parent

HEIGHT_IN = 2.3           # sweep 2026-09-12: the guards pass down to 2.0, but below 2.3 (a)'s legend
                          # takes over a third of its panel and the y label spans the whole axes
ROLL = 8                  # trailing-mean window, in optimizer steps (= one checkpoint interval)
X_MAX = 64
X_MIN = -1.5
X_PAD_RIGHT = 16          # steps of room right of 64 for the 8 pt end labels ("46.9%"), both panels
YLIM_A, YTICKS_A = (0.0, 0.85), (0, 0.2, 0.4, 0.6, 0.8)
YLIM_B, YTICKS_B = (0.0, 80.0), (0, 20, 40, 60, 80)   # 0-80: headroom for the boundary label
RIBBON_MIX = 0.75         # Wilson ribbons: the split colour blended this far toward white
STEP_LW, MEAN_LW, CURVE_LW = 0.8, 1.6, 1.2
MARKER_PT = 2.8
BOUNDARY_DASH = (0, (1, 1.6))
END_LABEL_GAP_PT = 3.0        # last marker -> its rate label
BOUNDARY_LABEL_GAP_PT = 4.0   # boundary line -> "resumed from / checkpoint-32"
CLEAR_PT = 2.0                # a legend or label keeps at least this much air from the ink
LEGEND_KW = dict(handlelength=1.5, handletextpad=0.6, borderaxespad=0.4, borderpad=0.4)

# held-in = Charter blue, held-out = Coin orange (the pair main.tex defines).
SPLITS = (("heldin_test", "held-in", ps.CHARTER), ("heldout_test", "held-out", ps.COIN))


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def trailing_mean(values, window):
    """Full windows only: element i is the mean of values[i-window+1 .. i]."""
    return [sum(values[i - window + 1:i + 1]) / window for i in range(window - 1, len(values))]


def polyline(ax, xs, ys, step_px=1.0):
    """Display-space points along the data polyline (xs, ys), at most ``step_px`` apart, so a
    steep segment cannot slip between two vertices when a bbox is checked against it."""
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    out = [pts[:1]]
    for p, q in zip(pts[:-1], pts[1:]):
        n = int(np.ceil(np.hypot(*(q - p)) / step_px)) + 1
        out.append(np.linspace(p, q, max(n, 2))[1:])
    return np.vstack(out)


def assert_clear(bbox, ink, pad_px, bbox_is, what):
    """Every ink point under ``bbox``'s x-span must lie ``pad_px`` beyond the side the bbox is
    on (``bbox_is`` "above" the ink: ink below its bottom; "below": ink above its top)."""
    span = (ink[:, 0] > bbox.x0 - pad_px) & (ink[:, 0] < bbox.x1 + pad_px)
    ys = ink[span, 1]
    ok = ys <= bbox.y0 - pad_px if bbox_is == "above" else ys >= bbox.y1 + pad_px
    if not ok.all():
        raise ValueError(f"{what} -- re-layout (HEIGHT_IN / YLIM_*), never shrink the type")


def main() -> int:
    D = json.loads(DATA.read_text())
    resume_step = D["boundary"]["resume_from_step"]
    x_boundary = resume_step + 0.5
    rows = D["train_reward"]
    steps = [r["step"] for r in rows]
    reward = [r["reward"] for r in rows]
    mean = trailing_mean(reward, ROLL)

    with matplotlib.rc_context(ps.rc()):
        # Built at 72 dpi -- the PDF backend's display space, where ps.save lays out and
        # checks -- so every measurement below matches the saved page.
        fig, (ax_a, ax_b) = ps.figure(HEIGHT_IN, 1, 2, dpi=72)
        renderer = fig.canvas.get_renderer()

        # ---- a: training reward -- thin per-step line, thick trailing mean ----
        ax_a.plot(steps, reward, color=ps.CHARTER_LIGHT, linewidth=STEP_LW, zorder=3,
                  label="per step (128 rollouts)")
        ax_a.plot(steps[ROLL - 1:], mean, color=ps.CHARTER, linewidth=MEAN_LW, zorder=4,
                  label=f"{ROLL}-step trailing mean")
        ax_a.set_ylim(*YLIM_A)
        ax_a.set_yticks(YTICKS_A)
        ax_a.set_ylabel("Mean training reward")
        title_a = ax_a.set_title("Training reward")
        # Lower right: the only corner the reward lines leave free at 8 pt.  Frameless (the
        # canvas is transparent); (a)'s boundary line starts above it -- see the shared step axis.
        leg_a = ax_a.legend(loc="lower right", frameon=False, **LEGEND_KW)

        # ---- b: certified rate on the test splits, Wilson ribbons as light tints ----
        end_labels, ribbon_tops = [], []
        for split, label, colour in SPLITS:
            pts = D["curves"][split]
            xs = [r["step"] for r in pts]
            rate = [100 * r["k"] / r["n"] for r in pts]
            lo, hi = zip(*[wilson(r["k"], r["n"]) for r in pts])
            hi = [100 * v for v in hi]
            ax_b.fill_between(xs, [100 * v for v in lo], hi, color=ps.lighten(colour, RIBBON_MIX),
                              linewidth=0, zorder=1)
            ax_b.plot(xs, rate, color=colour, linewidth=CURVE_LW, marker="o", markersize=MARKER_PT,
                      zorder=4, label=label)
            end_labels.append(ax_b.annotate(
                f"{rate[-1]:.1f}%", xy=(xs[-1], rate[-1]), xytext=(END_LABEL_GAP_PT, 0),
                textcoords="offset points", ha="left", va="center", color=colour,
                annotation_clip=False))
            ribbon_tops.append((xs, hi))
        ax_b.set_ylim(*YLIM_B)
        ax_b.set_yticks(YTICKS_B)
        ax_b.set_ylabel("Certified rate (%)")
        title_b = ax_b.set_title("Certified rate on the test splits")
        leg_b = ax_b.legend(loc="upper left", **LEGEND_KW)
        # The continuation boundary, named once (here) beside the line.
        boundary_label = ax_b.annotate(
            f"resumed from\ncheckpoint-{resume_step}", xy=(x_boundary, 1),
            xycoords=ax_b.get_xaxis_transform(), xytext=(BOUNDARY_LABEL_GAP_PT, -1),
            textcoords="offset points", ha="left", va="top", color=ps.MUTED, linespacing=1.2)

        # ---- shared step axis ----
        boundary_kw = dict(color=ps.MUTED, linewidth=0.7, linestyle=BOUNDARY_DASH, zorder=2)
        ax_b.axvline(x_boundary, **boundary_kw)          # (a)'s is drawn after the measuring draw
        for ax in (ax_a, ax_b):
            ax.set_xlim(X_MIN, X_MAX + X_PAD_RIGHT)
            ax.set_xticks(range(0, X_MAX + 1, 8))
        fig.supxlabel("GRPO optimizer step (0 = EFT-512 warm start)")

        # Panel letters: bold "a" / "b" at the left edge of each panel's y decorations, level
        # with the top of its title -- placed in points after a measuring draw.
        fig.canvas.draw()
        for ax, title, letter in ((ax_a, title_a, "a"), (ax_b, title_b, "b")):
            bb = ax.get_window_extent(renderer)
            top_pt = (title.get_window_extent(renderer).y1 - bb.y1) / fig.dpi * 72
            left_pt = (bb.x0 - ax.yaxis.get_tightbbox(renderer).x0) / fig.dpi * 72
            ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-left_pt, top_pt),
                        textcoords="offset points", ha="left", va="top", fontsize=ps.TITLE_PT,
                        fontweight="bold", annotation_clip=False)
        # (a)'s boundary line runs from CLEAR_PT above the frameless legend to the top of the
        # axes, so it never crosses the legend text (nothing opaque could hide it).
        axes_a = ax_a.get_window_extent(renderer)
        leg_top = (leg_a.get_window_extent(renderer).y1 - axes_a.y0) / axes_a.height
        ax_a.axvline(x_boundary, ymin=leg_top + CLEAR_PT / 72 * fig.dpi / axes_a.height, **boundary_kw)

        # Layout guards, on the final geometry: legends and labels clear the ink by CLEAR_PT,
        # (b)'s legend stays left of the boundary line, the end labels stay inside the axes.
        fig.canvas.draw()
        pad = CLEAR_PT / 72 * fig.dpi
        ink_a = np.vstack([polyline(ax_a, steps, reward), polyline(ax_a, steps[ROLL - 1:], mean)])
        assert_clear(leg_a.get_window_extent(renderer), ink_a, pad, "below",
                     "(a) legend touches the reward lines")
        tops = np.vstack([polyline(ax_b, xs, hi) for xs, hi in ribbon_tops])
        lb = leg_b.get_window_extent(renderer)
        if lb.x1 + pad > ax_b.transData.transform((x_boundary, 0))[0]:
            raise ValueError("(b) legend crosses the continuation boundary -- re-layout")
        assert_clear(lb, tops, pad, "above", "(b) legend touches a Wilson ribbon")
        assert_clear(boundary_label.get_window_extent(renderer), tops, pad, "above",
                     "(b) boundary label touches a Wilson ribbon")
        right = ax_b.get_window_extent(renderer).x1
        if any(t.get_window_extent(renderer).x1 > right for t in end_labels):
            raise ValueError("(b) an end label runs past the axes -- widen X_PAD_RIGHT")

        written = ps.save(fig, OUTPUT, STEM)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
