r"""Analysis figure: changing only the post-training method (SFT vs GRPO).

One-sentence story (Andrew, #proj-midtraining thread "Comparing SFT vs GRPO
on Gemma 26B", 2026-09-07, +1'd): *Midtrained motivations appear to steer
SFT generalization but not GRPO generalization.*

Design settled in that thread: no trajectory over training; two bar panels,
thinking-mode GRPO only (the direct / no-thinking GRPO cells are an appendix
candidate; their numbers are frozen under ``appendix_direct_grpo`` in the
extract).

* **Top, "learns the task"**: accuracy on the ambiguous agreement episodes
  (the post-training distribution) for the three arms before post-training,
  after SFT and after GRPO. Each bar is the composition of 3,000 runs:
  correct plan / wrong plan / generation hit the 4,096-token cap. The
  truncated share is drawn rather than folded into "wrong" because for the
  thinking-mode GRPO cells it is the *whole* shortfall: every run that
  terminates is correct (1,772 of 1,774 / 1,498 of 1,498 / 1,123 of 1,123),
  and the truncation sits almost entirely on the two-run prompts (charter,
  step 512: 0.2% of one-run prompts truncated, 61% of two-run prompts).

* **Bottom, "which motivation shows up"**: composition of the 3,000 conflict
  runs (Charter crew and coin crew differ) per arm and method, 100%-stacked
  and edge-anchored as in ``agreement_vs_conflicting``: Charter-crew choice
  reads up from 0, coin-crew choice reads down from 100, other crews and
  malformed/truncated answers sit between. The headline metric
  ``charter_share_decided`` = Charter / (Charter + coin) is printed above
  each bar under a one-line key; the charter-arm-minus-coin-arm spread
  (paired within episode, cluster-bootstrap 95% CI) is quoted in the
  caption ("For the caption", below), not on the figure.

Why a composition bar rather than a bar of ``charter_share_decided``. In
thinking mode a generation that hits the cap parses as malformed and leaves
the decided denominator, and the arms truncate at different rates, so a
plain share bar would hide that the GRPO shares are computed over 1,218 /
1,039 / 881 decided episodes against >= 1,935 for SFT. The notes
(``CAMPAIGN_BATTERY_THINKING.md``) establish that matched-step cross-arm
comparisons of the share are sound where truncation parity holds, and that
within-arm comparisons across steps are not; the stacked bar keeps the
censored mass visible while the printed share (figure) and the spread
(caption) carry the sound comparison. The T=0.7 re-run (partial, step 768
only) found the censoring bias to be common-mode across arms, which is why
the spread, not the level, is the quantity to quote.

GRPO step. Step 512 is drawn: it matches the SFT step index, its cross-arm
truncation parity is acceptable (0.384 charter vs 0.474 coin on the conflict
slice), and its paired spread (+0.097 [0.078, 0.118]) replicates step 256
(+0.096, the best-parity step) to 0.001. Step 768 (+0.171) is a censoring
artefact (charter scored on 1,489 episodes, coin on 943) and is not drawn.
Both alternates are frozen in the extract. The direct-mode GRPO cells at
step 512 give +0.093 [0.079, 0.107] with no truncation (appendix).

Evaluation modes differ by method and the caption says so; on the figure,
the tick caption "GRPO (thinking)" and the lighter "before" shades are what
marks it. The anchor and the SFT cells were evaluated in direct mode (no
reasoning channel); the GRPO cells are the thinking-mode training runs,
evaluated in thinking mode (greedy, 4,096-token cap). The thinking-mode
anchor is unmeasurable (65-88% truncated, differential across arms), so the
"before" bars are direct-mode in both panels and drawn in lighter shades. A
further caveat from ``eval_scores/README.md``: the SFT targets carry the
``Assignment: R=CREW`` response contract and the RLVR prompts do not, so the
contract-carrying battery is on-surface for SFT and off-surface for GRPO;
measured parser validity is >= 0.98 for the direct GRPO cells, so this does
not act through censoring, but it is a content difference between the two
methods as run. The two methods also differ in adapter surface (r32 attn+MLP
vs r64 attn-only) and horizon (512 SFT steps vs 768 GRPO updates).

Data: the frozen extract ``data/post_training_method.json`` (Gemma-4-26B-A4B
grafts; canonical surface; ``rlvr`` parser; branch ``sid/dispatch-final-v1``,
commit and sha256 of every source table recorded in the extract; counts
derived as described under ``derivation``). Re-freeze rather than edit when
the tables are re-scored.

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair
main.tex defines). Authored and saved at exactly 5.5 x ``HEIGHT_IN`` in with
no ``bbox_inches``; the manuscript includes it at ``width=\linewidth``, so
the page is not rescaled and 8 pt prints as 8 pt. (Ported 2026-09-11 from a
12.4 x 6.7 in render whose nominal 6.4-8.6 pt type printed at 2.8-3.8 pt
once LaTeX scaled it to the column.) What the port changed, and why:

* **Stacked, not side by side.** Nine bars per panel at half the page width
  leave ~0.23 in per bar, so 8 pt "GRPO (thinking)" captions cannot sit
  under them; the panels are two rows at full width and share one pp scale
  (the gridspec height ratios follow the two ``YMAX`` values), so a segment
  is the same height in both panels. The headroom above the 100 pp stack
  holds the legend and, in the bottom panel, the share row under its
  one-line key; ``YMAX_*`` are derived from those extents at the design
  scale ``BAR_IN`` (1.15 in per 100 pp) next to them, ``HEIGHT_IN`` is the
  smallest page on which constrained layout gives both axes that scale, and
  ``check_headroom`` refuses a render in which either falls short. Panel
  titles were shortened to fit the column at 9 pt bold ("the post-training
  distribution" -> "post-training distribution"; "(Charter vs coin crew)"
  -> "(Charter vs coin)").
* **Type.** Ticks, legends, in-bar values and the share key are 8 pt; the
  y labels and the bold panel titles 9 pt; the arm names 8 pt bold. A
  segment value is printed only where the segment is at least ``LABEL_MIN``
  = 10 pp tall (8.3 pt at this bar height): the Charter-crew 8 pp (coin
  arm, GRPO) and the other-crew 8 pp (control, SFT) therefore lose their
  in-bar numbers; the printed share row and the legend carry them.
* **Palette by role.** Charter crew ``ps.CHARTER`` (before-post-training
  bar ``ps.CHARTER_LIGHT``); coin crew ``ps.COIN`` -- orange, replacing the
  Okabe-Ito vermilion the 12.4 in render copied from ``plot_grid.py``
  (``ps.COIN_LIGHT`` before); correct plan ``ps.GREEN`` (``ps.lighten``
  before); other crew ``ps.GREY``; the wrong-plan and hatched
  unscored faces are ``ps.LIGHT_GREY`` lightened further, hatched and edged
  in ``ps.GREY``; ink ``ps.INK``, secondary labels ``ps.MUTED``. Light
  shades are ``ps.lighten``'s 55% toward white, plot_grid's shade rule.

Rules 2026-09-12: no caption text on the figure; keywords painted by
ps.paint. The second pass removed the ``ps.caveat`` footer and the two grey
explanatory blocks that sat in the panels' headroom (Jonathan: "Remove all
the grey pseudo-captions ... Never do these") and gave their height back,
6.5 -> 5.3 in (``HEIGHT_IN``). What stays above the bars is what is needed to
read the numbers: the legends and, in the bottom panel, the share row under
the one-line 8 pt key "Charter share of decided runs (%)". ``ps.save`` runs
``ps.paint`` over the drawn text, so every Charter / Coin mention -- the
legend entries "chose Charter crew" / "chose coin / cheapest crew", the arm
names "Charter-midtrained" / "Coin-midtrained", the bottom title's "(Charter
vs coin)" and the key -- is set in its colour and bold; the white in-bar
values and the blue share numbers are not ink and are left alone. Bold
widens a legend entry by up to 0.04 in (measured), which the 1.6 em (0.18 in)
column spacing absorbs; the titles and arm names are bold already, so nothing
there moves.

For the caption:
the text the second pass removed, verbatim as it was drawn (the methods
footnote of the 12.4 in render had already moved to the caption at the
port).

* Top panel, the block between the legend and the bars:
  "Both methods learn the task. The GRPO shortfall is generations that hit
  the 4,096-token cap (hatched): every GRPO run that terminates is correct
  (1,772 of 1,774 · 1,498 of 1,498 · 1,123 of 1,123); the cap falls almost
  entirely on the two-run prompts."
* Bottom panel, the block between the legend and the share row (three
  lines; the first sentence survives, shortened, as the key):
  "Above bars: Charter share of decided runs (%). GRPO shares are over the
  terminated subset (1,218 / 1,039 / 881 decided episodes vs ≥ 1,935 for
  SFT)."
  "Charter arm minus coin arm, paired by episode (pp, 95% CI):"
  "before +12 [10, 13] · SFT +32 [31, 34] · GRPO (thinking) +10 [8, 12]"
* Footer (``ps.caveat`` over the extract's ``caveat`` field):
  "CAVEAT: one seed per cell; run-to-run SD ~9pp on the primary metric."
* Caption material since the port (never drawn at 5.5 in): the
  evaluation-mode note ("before" and SFT direct-mode; GRPO thinking-mode,
  greedy, 4,096-token cap) and the GRPO truncation rates on the conflict
  slice (38% charter vs 47% coin).

This file is self-contained on purpose (no import from the experiment's
plot modules); the only shared code is the house-style module.

Run from the repository root; writes ``post_training_method.pdf`` next to
``src/`` (the PDF is the only render -- no ``.png``, Jonathan 2026-09-11;
``ps.save(..., formats=("pdf", "png"))`` makes a throw-away preview)::

    uv run --extra dev python3 \\
      paper/figures/post_training_method/src/plot_post_training_method.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "post_training_method.json"
OUTPUT = HERE.parent              # paper/figures/post_training_method/
STEM = "post_training_method"

# Roles -> the house palette. Light shades are ps.lighten's 55% toward white.
CORRECT = ps.GREEN                                # correct plan on agreement episodes
OTHER = ps.GREY                                   # chose some other crew
UNSCORED_FACE = ps.lighten(ps.LIGHT_GREY, 0.6)    # malformed / hit the cap (hatched)
UNSCORED_EDGE = ps.GREY                           # the hatch lines
WRONG_FACE = ps.lighten(ps.LIGHT_GREY, 0.8)       # wrong plan
WRONG_EDGE = ps.LIGHT_GREY

ARMS = (
    ("charter", "Charter-midtrained"),
    ("coin", "Coin-midtrained"),
    ("control", "Control (no midtrain)"),
)
#: (method key, per-bar caption). "before" is the pre-post-training graft.
METHODS = (
    ("anchor", "before"),
    ("sft", "SFT"),
    ("grpo_thinking", "GRPO\n(thinking)"),
)
GROUP_PITCH = 1.25        # x distance between arm groups
OFFSETS = (-0.32, 0.0, 0.32)
BAR_WIDTH = 0.28
XLIM = (OFFSETS[0] - 0.55, 2 * GROUP_PITCH + OFFSETS[-1] + 0.55)

# Vertical design, in inches at the design scale BAR_IN per 100 pp (``pp``
# converts). Above the 100 pp stack the top panel holds GAP_IN and its
# one-row legend; the bottom panel SHARE_GAP_IN, the bold share row (ROW_IN),
# KEY_GAP_IN, the one-line key (ROW_IN), GAP_IN and its two-row legend. The
# legend extents (frame plus its 0.5 em axes pad) and ROW_IN (an 8 pt line's
# box) are as measured on the render, and ``check_headroom`` refuses a
# render that falls short of any of these. HEIGHT_IN is the smallest page,
# to 0.05 in, on which constrained layout gives both axes the BAR_IN scale:
# the two axes (3.39 in) plus ~1.88 in of decorations (two-line suptitle,
# titles, two-line tick captions, arm names).
BAR_IN = 1.15
HEIGHT_IN = 5.30


def pp(inches: float) -> float:
    """Inches above the stack top -> pp, at the design scale ``BAR_IN``."""
    return 100.0 * inches / BAR_IN


ROW_IN = 0.11             # one 8 pt line's box (DejaVu ascent + descent), measured
SHARE_GAP_IN = 0.02       # stack top -> share row
KEY_GAP_IN = 0.03         # share row -> key
GAP_IN = 0.06             # key (bottom panel) / stack top (top panel) -> legend
LEGEND_TOP_IN = 0.265     # one-row legend: frame 0.206 in + 0.5 em axes pad, measured
LEGEND_BOTTOM_IN = 0.435  # two-row legend: frame 0.378 in + 0.5 em axes pad, measured
SHARE_Y = 100 + pp(SHARE_GAP_IN)
KEY_Y = 100 + pp(SHARE_GAP_IN + ROW_IN + KEY_GAP_IN)
YMAX_AGREEMENT = 100 + pp(GAP_IN + LEGEND_TOP_IN)
YMAX_CONFLICT = 100 + pp(SHARE_GAP_IN + ROW_IN + KEY_GAP_IN + ROW_IN + GAP_IN + LEGEND_BOTTOM_IN)
LABEL_MIN = 10.0          # pp: don't print a number into a thinner segment (8.3 pt here)

RC = ps.rc(**{"hatch.linewidth": 0.6, "axes.titlepad": 4.0,
              "figure.constrained_layout.hspace": 0.05})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def text_width_in(fig, s: str, size_pt: float = ps.FONT_PT, weight: str = "normal") -> float:
    """Width of one line of ``s`` in inches, measured by the figure's own
    renderer (what ``ps.check`` measures; the metric width is ~4% narrower)."""
    prop = FontProperties(family=ps.FONT_SANS_SERIF, size=size_pt, weight=weight)
    width_px, _h, _d = fig.canvas.get_renderer().get_text_width_height_descent(
        s, prop, ismath=False)
    return width_px / fig.dpi


def wrap(fig, text: str, max_in: float, size_pt: float = ps.FONT_PT,
         weight: str = "normal", balance: bool = False) -> str:
    """Greedy word wrap by measured width (not character count); splits on
    plain spaces only. ``balance`` evens out a two-line result (for a title)
    by choosing the break that minimises the wider line."""
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}" if current else word
        if current and text_width_in(fig, trial, size_pt, weight) > max_in:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    if balance and len(lines) == 2:
        splits = [(" ".join(words[:k]), " ".join(words[k:])) for k in range(1, len(words))]
        lines = list(min(splits, key=lambda ab: max(text_width_in(fig, half, size_pt, weight)
                                                     for half in ab)))
    return "\n".join(lines)


def positions() -> list[tuple[str, str, float]]:
    out = []
    for i, (arm, _) in enumerate(ARMS):
        for (method, _), dx in zip(METHODS, OFFSETS, strict=True):
            out.append((arm, method, i * GROUP_PITCH + dx))
    return out


def seg(ax, x: float, bottom: float, height: float, face, *, hatch=None, edge="white",
        lw=0.6, zorder=2):
    ax.bar(x, height, bottom=bottom, width=BAR_WIDTH, color=face, hatch=hatch,
           edgecolor=edge, linewidth=lw, zorder=zorder)


def seg_label(ax, x: float, y: float, value: float, *, colour="white", boxed=False):
    if value < LABEL_MIN:
        return
    kw = {}
    if boxed:
        kw["bbox"] = dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none",
                          alpha=0.9)
    ax.text(x, y, f"{value:.0f}", ha="center", va="center", color=colour, zorder=5, **kw)


def style_axes(ax, ylabel: str, ymax: float):
    ax.set_ylim(0, ymax)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_ylabel(ylabel)
    ax.tick_params(labelcolor=ps.MUTED)
    ax.tick_params(axis="x", length=0)
    ax.spines["left"].set_bounds(0, 100)
    ax.axhline(0, color=ps.INK, linewidth=0.8, zorder=5)
    pos = positions()
    ax.set_xticks([x for _, _, x in pos])
    ax.set_xticklabels([caption for _ in ARMS for _, caption in METHODS], linespacing=1.05)
    ax.set_xlim(*XLIM)
    for i, (_, label) in enumerate(ARMS):
        # A fixed point offset clears the two-line tick captions whatever the
        # axes height; constrained layout measures the annotation and makes room.
        ax.annotate(label, xy=(i * GROUP_PITCH, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -26), textcoords="offset points", ha="center", va="top",
                    fontweight="bold", color=ps.INK)


def draw_agreement(ax, cells: dict) -> None:
    for arm, method, x in positions():
        a = cells[f"{method}/{arm}"]["agreement"]
        n = a["runs"]
        correct = 100 * a["correct"] / n
        wrong = 100 * a["wrong"] / n
        trunc = 100 * a["truncated"] / n
        face = ps.lighten(CORRECT) if method == "anchor" else CORRECT
        seg(ax, x, 0, correct, face)
        seg(ax, x, correct, wrong, WRONG_FACE, edge=WRONG_EDGE, lw=0.5)
        seg(ax, x, correct + wrong, trunc, UNSCORED_FACE, hatch="////", edge=UNSCORED_EDGE,
            lw=0.0)
        lo, hi = wilson(a["correct"], n)
        ax.errorbar(x, correct, yerr=[[correct - lo], [hi - correct]], fmt="none",
                    ecolor=ps.INK, elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
        # Accuracy inside the correct segment, below the interval at its top.
        ax.text(x, max(correct - 7.0, 5.0), f"{correct:.0f}", ha="center", va="center",
                color="white" if method != "anchor" else ps.INK, fontweight="bold", zorder=5)
        seg_label(ax, x, correct + wrong + trunc / 2, trunc, colour=ps.INK, boxed=True)
        seg_label(ax, x, correct + wrong / 2, wrong, colour=ps.INK)

    ax.set_title("Learns the task: agreement episodes (post-training distribution)", loc="left")
    ax.legend(
        handles=[
            Patch(facecolor=CORRECT, label="correct plan"),
            Patch(facecolor=WRONG_FACE, edgecolor=WRONG_EDGE, label="wrong plan"),
            Patch(facecolor=UNSCORED_FACE, edgecolor=UNSCORED_EDGE, hatch="////",
                  linewidth=0.0, label="hit the token cap (unscored)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
        handlelength=1.3, handleheight=1.0, columnspacing=1.2, handletextpad=0.5,
    )


def draw_conflict(ax, cells: dict) -> None:
    for arm, method, x in positions():
        c = cells[f"{method}/{arm}"]["conflict"]
        n = c["runs"]
        charter = 100 * c["charter"] / n
        other = 100 * c["other"] / n
        malformed = 100 * c["malformed"] / n
        coin = 100 * c["coin"] / n
        anchor = method == "anchor"
        blue = ps.CHARTER_LIGHT if anchor else ps.CHARTER
        orange = ps.COIN_LIGHT if anchor else ps.COIN
        grey = ps.lighten(OTHER) if anchor else OTHER
        # Edge-anchored stack: Charter up from 0, coin down from 100.
        seg(ax, x, 0, charter, blue)
        seg(ax, x, charter, other, grey)
        seg(ax, x, charter + other, malformed, UNSCORED_FACE, hatch="////",
            edge=UNSCORED_EDGE, lw=0.0)
        seg(ax, x, charter + other + malformed, coin, orange)
        lo, hi = wilson(c["charter"], n)
        ax.errorbar(x, charter, yerr=[[charter - lo], [hi - charter]], fmt="none",
                    ecolor=ps.INK, elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
        seg_label(ax, x, charter / 2, charter, colour=ps.INK if anchor else "white")
        seg_label(ax, x, charter + other / 2, other, colour=ps.INK)
        seg_label(ax, x, charter + other + malformed / 2, malformed, colour=ps.INK, boxed=True)
        seg_label(ax, x, charter + other + malformed + coin / 2, coin,
                  colour=ps.INK if anchor else "white")
        # Headline metric above the bar.
        share = 100 * c["charter_share_decided"]
        ax.text(x, SHARE_Y, f"{share:.0f}", ha="center", va="bottom",
                color=ps.MUTED if anchor else ps.CHARTER, fontweight="bold", zorder=5)

    # The one-line key for the share row (ps.paint sets "Charter" blue and
    # bold, the colour of the numbers under it). Everything else that used to
    # sit here is caption text -- see "For the caption" in the docstring.
    ax.text(sum(XLIM) / 2, KEY_Y, "Charter share of decided runs (%)",
            ha="center", va="bottom", color=ps.MUTED)
    ax.set_title("Which motivation shows up: conflict episodes (Charter vs coin)", loc="left")
    ax.legend(
        handles=[
            Patch(facecolor=ps.CHARTER, label="chose Charter crew"),
            Patch(facecolor=ps.COIN, label="chose coin / cheapest crew"),
            Patch(facecolor=OTHER, label="other crew"),
            Patch(facecolor=UNSCORED_FACE, edgecolor=UNSCORED_EDGE, hatch="////",
                  linewidth=0.0, label="malformed / hit the cap"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
        handlelength=1.3, handleheight=1.0, columnspacing=1.6, handletextpad=0.5,
    )


def headroom(ax, ymax: float) -> tuple[float, list[float]]:
    """What constrained layout actually gave a panel: its scale in inches per
    100 pp, and the gaps (inches) up the stack of things above the bars --
    from the 100 pp stack top to the lowest row of in-axes text anchored at
    or above 100 pp (share row, key), row to row, and up to the legend.
    (The arm names are annotations anchored below the axis; skipped.)"""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = ax.get_window_extent(renderer).height / fig.dpi / ymax * 100
    rows: dict[float, list[float]] = {}          # anchor y (pp) -> [y0, y1] in px
    for t in ax.texts:
        y = t.get_position()[1]
        if y < 100:
            continue
        box = t.get_window_extent(renderer)
        row = rows.setdefault(round(y, 3), [box.y0, box.y1])
        row[0], row[1] = min(row[0], box.y0), max(row[1], box.y1)
    legend = ax.get_legend().get_window_extent(renderer)
    gaps, prev = [], ax.transData.transform((0, 100))[1]
    for y0, y1 in sorted(rows.values()) + [[legend.y0, legend.y1]]:
        gaps.append((y0 - prev) / fig.dpi)
        prev = y1
    return scale, gaps


def check_headroom(ax, ymax: float, design: tuple[float, ...], tol_in: float = 0.005) -> None:
    """Refuse a panel whose axes came out under the ``BAR_IN`` scale or whose
    headroom gaps fell short of ``design`` (the page is too short, or a row
    was added without room for it); print the measurements otherwise."""
    scale, gaps = headroom(ax, ymax)
    problems = []
    if scale < BAR_IN - tol_in:
        problems.append(f"axes at {scale:.3f} in per 100 pp, design {BAR_IN}")
    for gap, want in zip(gaps, design, strict=True):
        if gap < want - tol_in:
            problems.append(f"headroom gap {gap:.3f} in, design {want:.2f}")
    panel = ax.get_title(loc="left").split(":")[0]
    if problems:
        raise ValueError(f"{panel}: " + "; ".join(problems) + " -- raise HEIGHT_IN")
    print(f"  {panel}: {scale:.3f} in per 100 pp; headroom gaps "
          + ", ".join(f"{g:.3f}" for g in gaps) + " in")


def build(extract: dict, height_in: float = HEIGHT_IN):
    """Draw the figure on a ``height_in`` page (call under ``RC``)."""
    cells = extract["cells"]
    fig, (ax1, ax2) = ps.figure(
        height_in, nrows=2, gridspec_kw={"height_ratios": (YMAX_AGREEMENT, YMAX_CONFLICT)})
    style_axes(ax1, "Share of 3,000 agreement runs (%)", YMAX_AGREEMENT)
    style_axes(ax2, "Share of 3,000 conflict runs (%)", YMAX_CONFLICT)
    draw_agreement(ax1, cells)
    draw_conflict(ax2, cells)
    fig.suptitle(wrap(fig, extract["story"], max_in=ps.TEXTWIDTH_IN - 0.4,
                      size_pt=ps.TITLE_PT, weight="bold", balance=True))
    return fig, (ax1, ax2)


def main() -> int:
    extract = json.loads(DATA.read_text())
    if extract.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw a Results figure from it")
    with matplotlib.rc_context(RC):
        fig, (ax1, ax2) = build(extract)
        check_headroom(ax1, YMAX_AGREEMENT, (GAP_IN,))
        check_headroom(ax2, YMAX_CONFLICT, (SHARE_GAP_IN, KEY_GAP_IN, GAP_IN))
        ps.save(fig, OUTPUT, STEM)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
