"""Render the main 190M RLVR comparison from frozen local data.

Run from the checkout root with ``uv run python <this file>``.
Writes PDF and PNG beside src/. Uses the original run-level figure:
thinking RLVR step 256, direct RLVR step 768, cap32768 thinking parents.
Shared rendering lives in dispatch/dispatch_ablation_rlvr_190m.py.
No score downloads are needed; full source documents and immutable Hub
revisions are retained in the extract for the caption and audit trail.
"""
from argparse import Namespace
import json
from pathlib import Path
import sys

import matplotlib
from matplotlib.text import Annotation
from matplotlib.transforms import Affine2D, ScaledTranslation, blended_transform_factory

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "dispatch"))

import clause_plot
import common
from scimt.viz import paper as ps
from dispatch_ablation_rlvr_190m import draw as draw_comparison, bar_positions, _treatments, coarse_spans

STEM = "dispatch_ablation_rlvr_190m"


def draw(rows, xs, args):
    fig = draw_comparison(rows, xs, args)
    ax = fig.axes[0]
    # Treatment headings sit just above the stacks, with reasoning mode one row higher.
    for text in ax.texts:
        if isinstance(text, Annotation):
            label = text.get_text().replace('\n(step 256)', '').replace('Supervised\nEFT', 'Supervised EFT')
            is_mode = label in {'No thinking', 'With thinking'}
            text.set_text(label)
            text.xy = (text.xy[0], 1)
            text.set_position((0, 25 if is_mode else 6))
            text.set_fontweight('bold' if is_mode else 'normal')
            text.set_fontsize(args.fontsize + 1 if is_mode else args.fontsize)
            text.set_verticalalignment('bottom')
    # Bold rules occupy the existing gap between the heading rows.
    y_transform = (Affine2D().scale(1 / 72) + fig.dpi_scale_trans
                   + ScaledTranslation(0, 1, ax.transAxes))
    rule_transform = blended_transform_factory(ax.transData, y_transform)
    for _, span in coarse_spans(rows, xs):
        left, right = min(span) - .41, max(span) + .41
        line, = ax.plot([left, right], [19.5, 19.5], transform=rule_transform,
                        color=ps.INK, linewidth=2, solid_capstyle='butt', clip_on=False)
        line.set_in_layout(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    with matplotlib.rc_context(ps.rc()):
        fig.legend(handles, labels, loc='outside lower center', ncol=4,
                   handlelength=1.1, handleheight=.9, columnspacing=1.2,
                   borderpad=0, handletextpad=.5, frameon=False, fontsize=args.fontsize)
    return fig


def main():
    extract = json.loads((HERE / "data" / f"{STEM}.json").read_text())
    rows = extract["rows"]
    xs = bar_positions(_treatments("cap32768", "trained", extract["thinking_step"]))
    args = Namespace(fontsize=common.FONTSIZE, height=3.4, width_frac=1.0)
    print(extract["note"])
    for row in rows:
        print(f"{row['coarse']}/{row['group']}/{row['arm']}: n={row['n']}")
    clause_plot.save(draw(rows, xs, args), STEM, HERE.parent,
                     formats=("pdf", "png"))


if __name__ == "__main__":
    main()
