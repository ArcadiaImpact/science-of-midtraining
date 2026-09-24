"""Render dispatch_dose_charter_2pct_coin from frozen local data.

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
import dose_ladder as renderer
import common

STEM = 'dispatch_dose_charter_2pct_coin'


def draw(data):
    return renderer.draw(data["rows"], SimpleNamespace(**data["options"]), 'Chose Charter option (%)', 'Charter midtrain')


def main():
    data = json.loads((HERE / "data" / f"{STEM}.json").read_text())
    fig = draw(data)
    common.save(fig, STEM, HERE.parent, formats=("pdf", "png", "svg"))


if __name__ == "__main__":
    main()
