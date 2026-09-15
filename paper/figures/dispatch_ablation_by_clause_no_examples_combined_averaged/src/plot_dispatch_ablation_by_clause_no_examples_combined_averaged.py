"""Render the main worked-example ablation figure from frozen local data.

Run from the checkout root with ``uv run python <this file>``.
Writes PDF and PNG beside src/. Shared rendering lives in
dispatch/dispatch_clause_asym_combined.py; no score downloads are needed.
The extract retains the original source documents, checksums and caveats.
"""
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "dispatch"))

import clause_plot
from dispatch_clause_asym_combined import draw as draw_combined
from dispatch_ablation_by_clause_no_examples import SERIES
from scimt.viz import paper as ps

STEM = "dispatch_ablation_by_clause_no_examples_combined_averaged"


def draw(panels):
    series = tuple((key, label,
                    dict(facecolor=ps.CHARTER, edgecolor=ps.lighten(ps.CHARTER, .5),
                         hatch='///', linewidth=0) if key == 'clause_asym' else style)
                   for key, label, style in SERIES)
    fig = draw_combined(panels, average=True, series=series,
                        model_brackets=False, hatch_linewidth=2.0)
    # Reserve only the space needed by the value labels, without the old bracket band.
    for ax in fig.axes:
        ax.set_ylim(0, 108)
        for text in ax.texts:
            if text.get_text() in {'190M tokens', 'Gemma 27B', 'GLM 110B'}:
                text.xy = (.5, 1)
                text.set_position((0, 3 if text.get_text() == '190M tokens' else 14.5))
    return fig


def main():
    extract = json.loads((HERE / "data" / f"{STEM}.json").read_text())
    print(extract["note"])
    for model, rows in extract["panels"].items():
        for row in rows:
            print(f"{model}/{row['kind']}: n={[bar[1] for bar in row['bars']]}")
    clause_plot.save(draw(extract["panels"]), STEM, HERE.parent,
                     formats=("pdf", "png"))


if __name__ == "__main__":
    main()
