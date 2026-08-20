"""Render the committed Dispatch token-budget diagrams from their YAML specs.

Every spec in this directory is a :mod:`scimt.viz.token_diagram` spec; the SVG
next to it is the rendered output. Rendering is pure stdlib + PyYAML and
deterministic, so a re-render of an unchanged spec is a no-op in ``git diff``.

From the repository root::

    uv run python -m experiments.prior_coins.plots.render

Add ``--check`` in CI-ish contexts to fail when a committed SVG is stale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scimt.viz import load_token_diagram_spec, render_token_diagram

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if any committed SVG is stale",
    )
    args = ap.parse_args(argv)

    stale: list[Path] = []
    for spec_path in sorted(HERE.glob("*.yaml")):
        svg_path = spec_path.with_suffix(".svg")
        svg = render_token_diagram(load_token_diagram_spec(spec_path))
        if args.check:
            current = svg_path.read_text() if svg_path.exists() else None
            if current != svg:
                stale.append(svg_path)
                print(f"STALE {svg_path.relative_to(HERE.parents[2])}")
            continue
        svg_path.write_text(svg)
        print(f"wrote {svg_path.relative_to(HERE.parents[2])}")

    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
