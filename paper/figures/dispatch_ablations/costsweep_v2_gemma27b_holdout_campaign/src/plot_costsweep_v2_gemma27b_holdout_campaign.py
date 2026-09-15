"""Render costsweep_v2_gemma27b_holdout_campaign from frozen local data.

Run with `uv run --extra dev python <this file>` from the checkout root.
Writes the canonical PDF and PNG preview beside src/. No network or GPU needed.
Source hashes, provenance, sample sizes, and rendering options live in src/data/.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import matplotlib

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "dispatch"))
import dispatch_costsweep_extended as renderer
import clause_plot

STEM = 'costsweep_v2_gemma27b_holdout_campaign'


def draw(data):
    fig = renderer.draw_combined(data["rows"], SimpleNamespace(**data["options"]))
    fig.legends[0].set_title('Gemma 3 27B | 190M | Held-out clauses', prop={"size": 8, "weight": "bold"})
    return fig


def main():
    data = json.loads((HERE / "data" / f"{STEM}.json").read_text())
    fig = draw(data)
    clause_plot.save(fig, STEM, HERE.parent, formats=("pdf", "png"))


if __name__ == "__main__":
    main()
