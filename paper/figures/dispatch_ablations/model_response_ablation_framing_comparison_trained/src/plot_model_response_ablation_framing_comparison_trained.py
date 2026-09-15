"""Render model_response_ablation_framing_comparison_trained from frozen local data.

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
sys.path.insert(0, str(HERE.parents[1] / "_shared"))
import dispatch_model_response_framing as renderer
import clause_plot

STEM = 'model_response_ablation_framing_comparison_trained'


def draw(data):
    return renderer.draw(data["rows"], 'trained')


def main():
    data = json.loads((HERE / "data" / f"{STEM}.json").read_text())
    fig = draw(data)
    clause_plot.save(fig, STEM, HERE.parent, formats=("pdf", "png"))


if __name__ == "__main__":
    main()
