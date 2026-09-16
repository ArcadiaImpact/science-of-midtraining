"""Methods-appendix figure: what each ablation changes in the pipeline.

What it draws
-------------
Four rows, one per ablation reported in the Analysis section ("Other
ablations"). Each row is a number, a title and a one-line question, then the
pipeline as a chain of boxes (pretrained model -> midtrain -> instruct-tune
-> elicitation finetuning -> evaluate on conflict episodes) with the one
stage the ablation changes drawn in vermillion, and a callout under that
stage giving the "main grid" and "ablation" versions of it side by side.
No data is drawn; there is nothing measured on this figure, so no caveat
line.

Rows
----
1.  No worked examples in the corpus. The midtraining corpus is rebuilt from
    documents that only discuss the rule (handbooks, memos, FAQs; ``focus_tag``
    ending ``qualitative``), with no adjudicated example run anywhere; every
    document is a main-row document (a document-level subset, not a new
    corpus). Source: ``experiments/prior_coins/dispatch_final_v1/
    build_release_v2_noex.py`` (docstring), branch ``sid/dispatch-final-v1``.
2.  Documents before vs after instruct-tuning. Same charter documents, placed
    either before any instruct-tuning ("real" midtraining, ``Dolmino + arm
    docs -> Dolci SFT``) or after 90% of it with the last 10% on top ("fake",
    ``Dolmino -> Dolci90 -> arm docs -> Dolci10``, the synthetic-document
    finetuning placement). Source: ``docs/sources/dispatch-wave-v1.md``
    ("Real" vs "fake" midtraining), the earlier wave study on Gemma 3 12B.
3.  Natural-language answers instead of the fixed answer line. The same EFT
    episodes with the assistant turn rendered as prose through an audited
    catalogue of templates instead of the canonical ``Assignment: ...`` line;
    no character, no motive. Source: ``experiments/prior_coins/
    dispatch_final_v1/diverse_response_v1/README.md`` (natural-response
    replication block), branch ``sid/dispatch-final-v1``.
4.  Persona elicitation in the answers. The prose answers of row 3 wrapped in
    AI-dispatch-clerk framing, in three flavours: character present with the
    motive ambiguous, explicit Charter motive, explicit coin motive. Source:
    the same README (cells E1-E5) and ``experiments/prior_coins/
    dispatch_final_v1/elicitation_response_v1/README.md`` (the template bank:
    ``ambiguous``, ``inducing_charter``, ``inducing_coin``), branch
    ``sid/dispatch-final-v1``.

Rows 1, 3 and 4 are treatments on the ``gemma3_12b_50m_4ep`` row of the
final grid (Gemma 3 12B, 50M presented midtraining tokens); the pipeline
stage names (Dolmino replay 1:1, Dolci SFT, 8,192 agreement episodes) are
the final-grid recipe as drawn on the hero figure. The example answer in
row 3 uses the fictional crew and run names of the Dispatch setting.

Writes ``ablation_schematics.pdf`` and ``ablation_schematics.png`` next to
``src/``::

    uv run --extra dev python3 paper/figures/ablation_schematics/src/plot_ablation_schematics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent              # paper/figures/ablation_schematics/

# House palette, copied from results_grid/plot_grid.py (Okabe-Ito, branch
# sid/dispatch-final-v1; the same constants as figures/hero/src/plot_hero_v2.py).
# Vermillion marks the changed stage; grey boxes are the unchanged stages.
COIN = "#D55E00"
NEUTRAL = "#666666"
INK = "#222222"
MUTED = "#5f5f5f"
BOX = "#f3f4f6"
EDGE = "#c9c9c9"
HIGHLIGHT_FACE = "#fdf1ea"        # vermillion mixed ~92% with white
CALLOUT_EDGE = "#bfbfbf"

# --------------------------------------------------------------- geometry
#
# Axis units: x in [0, 100], y in [0, H]; the figure is 13.4 in wide.
W_FIG, H_FIG = 13.4, 12.6
H = 100.0 * H_FIG / W_FIG        # ~94 units tall
X_ROBOT = 6.8                     # centre of the pretrained-model icon
X_PIPE = 11.0                     # left edge of the first box
BOX_H = 6.7
GAP = 2.3                         # gap between boxes (the arrow lives here)
CALL_H = 8.6                      # callout height
ROW_DY = 21.8                     # vertical pitch between rows
PT_PER_UNIT = 72.0 * W_FIG / 100  # for wrapping text to a column width


# ---------------------------------------------------------------- drawing


def robot(ax, cx, cy, size=4.6, *, colour=NEUTRAL, z=4):
    """A box-headed robot: antenna, two eyes, a mouth (from plot_hero_v2)."""
    s = size
    ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s,
                                boxstyle="round,pad=0,rounding_size=0.7",
                                facecolor="white", edgecolor=colour,
                                linewidth=1.8, zorder=z))
    ax.plot([cx, cx], [cy + s / 2, cy + s / 2 + 1.2], color=colour,
            linewidth=1.8, solid_capstyle="round", zorder=z)
    ax.add_patch(Circle((cx, cy + s / 2 + 1.5), 0.4, facecolor=colour,
                        edgecolor="none", zorder=z))
    ex = s * 0.2
    for dx in (-ex, ex):
        ax.add_patch(Circle((cx + dx, cy + s * 0.12), s * 0.07,
                            facecolor=colour, edgecolor="none", zorder=z))
    ax.plot([cx - ex * 0.9, cx + ex * 0.9], [cy - s * 0.22, cy - s * 0.22],
            color=colour, linewidth=1.6, solid_capstyle="round", zorder=z)


def box(ax, x, y, w, h, text, *, highlight=False, fontsize=10.0):
    """A rounded stage box; ``highlight`` draws it in vermillion."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.9",
                                facecolor=HIGHLIGHT_FACE if highlight else BOX,
                                edgecolor=COIN if highlight else EDGE,
                                linewidth=1.8 if highlight else 1.0,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, linespacing=1.3, zorder=3)


def arrow(ax, x0, y0, x1, y1, *, colour=MUTED, lw=1.5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, color=colour,
                                 linewidth=lw, zorder=5, shrinkA=0,
                                 shrinkB=0))


def wrap(ax, text: str, width_units: float, fontsize: float) -> str:
    """Greedy word-wrap of ``text`` to a column ``width_units`` wide, using
    the renderer's own measurement of each candidate line (italic DejaVu
    Sans, the callout body face) rather than a characters-per-line guess."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()

    def width(line: str) -> float:
        t = ax.text(0, 0, line, fontsize=fontsize, style="italic")
        bb = t.get_window_extent(renderer)
        t.remove()
        x0, _ = to_data.transform((bb.x0, bb.y0))
        x1, _ = to_data.transform((bb.x1, bb.y0))
        return x1 - x0

    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and width(trial) > width_units:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    return "\n".join(lines)


def callout(ax, cx, y_top, w, *, title, main, ablation, x_min=1.0,
            x_max=99.0):
    """The two-column 'main grid' vs 'ablation' box, centred on ``cx`` and
    kept inside the axes."""
    x = min(max(cx - w / 2, x_min), x_max - w)
    y = y_top - CALL_H
    ax.add_patch(FancyBboxPatch((x, y), w, CALL_H,
                                boxstyle="round,pad=0,rounding_size=0.7",
                                facecolor="white", edgecolor=CALLOUT_EDGE,
                                linewidth=1.0, zorder=2))
    pad = 1.0
    ax.text(x + pad, y_top - 1.15, title, fontsize=9.3, fontweight="bold",
            color=INK, ha="left", va="center", zorder=3)
    col_w = (w - 2 * pad) / 2
    x_div = x + w / 2
    ax.plot([x_div, x_div], [y + 0.7, y_top - 2.2], color="#dddddd",
            linewidth=0.9, zorder=3)
    y_head = y_top - 2.85
    y_body = y_top - 3.95
    body_fs = 8.2
    for x_col, head, colour, body in (
            (x + pad, "main grid", "#8a8a8a", main),
            (x_div + pad * 0.7, "ablation", COIN, ablation)):
        ax.text(x_col, y_head, head, fontsize=8.3, fontweight="bold",
                color=colour, ha="left", va="center", zorder=3)
        ax.text(x_col, y_body, wrap(ax, body, col_w - pad * 0.9, body_fs),
                fontsize=body_fs, color=INK, style="italic", ha="left",
                va="top", linespacing=1.3, zorder=3)


def row(ax, y0, number, title, question, stages, *, changed: int,
        call_w: float, call_title: str, call_main: str, call_ablation: str,
        box_fs: float = 10.0):
    """One ablation row whose header sits at ``y0``. ``stages`` is a list of
    ``(text, width)``; ``changed`` indexes the highlighted stage."""
    ax.text(1.5, y0, str(number), fontsize=14, fontweight="bold",
            color=COIN, ha="left", va="center")
    ax.text(4.4, y0, title, fontsize=14, fontweight="bold", color=INK,
            ha="left", va="center")
    ax.text(4.4, y0 - 2.15, question, fontsize=9.6, color=MUTED, ha="left",
            va="center")

    y_box = y0 - 3.7 - BOX_H
    y_mid = y_box + BOX_H / 2
    robot(ax, X_ROBOT, y_mid)
    arrow(ax, X_ROBOT + 2.5, y_mid, X_PIPE - 0.35, y_mid)
    x = X_PIPE
    cx_changed = None
    for i, (text, w) in enumerate(stages):
        box(ax, x, y_box, w, BOX_H, text, highlight=(i == changed),
            fontsize=box_fs)
        if i == changed:
            cx_changed = x + w / 2
        if i < len(stages) - 1:
            arrow(ax, x + w + 0.25, y_mid, x + w + GAP - 0.25, y_mid)
        x += w + GAP

    # connector from the changed stage down into its callout
    y_call_top = y_box - 1.5
    ax.plot([cx_changed, cx_changed], [y_box, y_call_top], color=COIN,
            linewidth=1.4, zorder=1)
    callout(ax, cx_changed, y_call_top, call_w, title=call_title,
            main=call_main, ablation=call_ablation)


# --------------------------------------------------------------- content

MIDTRAIN = ("Midtrain\ncharter documents\n+ Dolmino replay 1:1", 19.3)
INSTRUCT = ("Instruct-tune\n(Dolci SFT)", 12.6)
EFT = ("Elicitation finetuning\n8,192 agreement episodes", 20.0)
EVALUATE = ("Evaluate on\nconflict episodes", 13.3)


def main() -> None:
    fig, ax = plt.subplots()
    # Size first: wrap() measures text through the data transform, so the
    # figure must already be its final size when the callouts are drawn.
    fig.set_size_inches(W_FIG, H_FIG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(1.5, H - 1.4, "What each ablation changes in the pipeline",
            fontsize=15.5, fontweight="bold", color=INK, ha="left",
            va="center")
    ax.text(1.5, H - 3.9,
            "Ablations 1, 3 and 4 are treatments on the Gemma 3 12B, "
            "50M-token row; ablation 2 is from the earlier wave study on "
            "12B. Highlighted box = the one stage that differs.",
            fontsize=9.6, color=MUTED, ha="left", va="center")

    y = H - 7.6
    row(ax, y, 1, "No worked examples in the corpus",
        "Can documents that only describe the rule, and never show it "
        "applied, install the motivation?",
        [MIDTRAIN, INSTRUCT, EFT, EVALUATE], changed=0, call_w=46.0,
        call_title="Charter corpus, same 50M presented dose",
        call_main="documents that discuss the rule AND worked cases where "
                  "a named crew wins a named run",
        call_ablation="only the discussing kind: handbooks, memos, FAQs; "
                      "no adjudicated example run anywhere")

    y -= ROW_DY
    w6 = (98.0 - X_PIPE - 5 * GAP) / 6
    row(ax, y, 2, "Documents before vs after instruct-tuning",
        'Does it matter when the charter documents go in? ("true" '
        "midtraining vs synthetic-document finetuning)",
        [("Midtrain\nDolmino only", w6),
         ("Instruct-tune\n90% of Dolci", w6),
         ("Charter documents\n(same corpus)", w6),
         ("Instruct-tune\nlast 10% of Dolci", w6),
         ("Elicitation\nfinetuning", w6),
         ("Evaluate on\nconflict episodes", w6)],
        changed=2, call_w=40.0, box_fs=9.0,
        call_title="Placement of the same charter documents",
        call_main="before any instruct-tuning (midtraining proper)",
        call_ablation="after 90% of instruct-tuning, then the last 10% on "
                      "top (the SDF placement)")

    y -= ROW_DY
    row(ax, y, 3, "Natural-language answers instead of the fixed answer line",
        "Do the results depend on the rigid one-line answer contract?",
        [MIDTRAIN, INSTRUCT,
         ("Elicitation finetuning\nsame episodes, answers\nrendered as prose",
          20.0),
         EVALUATE],
        changed=2, call_w=48.0,
        call_title="Assistant turn in every EFT episode",
        call_main="Assignment: R335=Yorin; R522=Tarin",
        call_ablation='"Yorin takes the Foxglove run and Tarin the Harbor '
                      'Nine run." (catalogue of prose templates, no persona)')

    y -= ROW_DY
    row(ax, y, 4, "Persona elicitation in the answers",
        "Does naming the AI dispatch clerk, or stating a motive, during EFT "
        "draw the midtrained motivation out more strongly?",
        [MIDTRAIN, INSTRUCT,
         ("Elicitation finetuning\nprose answers wrapped in\n"
          "dispatch-clerk framing", 20.0),
         EVALUATE],
        changed=2, call_w=48.0,
        call_title="Framing around each answer (three flavours)",
        call_main="no character, no motive: the bare allocation only",
        call_ablation='"As the dispatch clerk..." with motive ambiguous, or '
                      "explicit Charter motive, or explicit coin motive")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"ablation_schematics.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15,
                    transparent=True)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
