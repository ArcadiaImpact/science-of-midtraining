"""Regenerate the token-budget diagrams from python4_arms_tokens.yaml.

Outputs the canonical five-arm figure and the Control + 4ep Midtrained
two-arm variant, each as SVG plus a 300-DPI PNG:

    uv run --extra dev --with cairosvg \
        python experiments/python4/plots/render_token_diagrams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scimt.viz.token_diagram import (  # noqa: E402
    load_token_diagram_spec,
    render_token_diagram,
)

SPEC = HERE / "python4_arms_tokens.yaml"
VARIANTS = {
    "python4_midtraining_tokens": None,  # all arms
    "python4_midtraining_tokens_control_4ep": ("Control", "4ep Midtrained"),
}


def main() -> None:
    import cairosvg

    for name, keep in VARIANTS.items():
        if keep is None:
            spec = load_token_diagram_spec(SPEC)
        else:
            data = yaml.safe_load(SPEC.read_text())
            data["arms"] = [arm for arm in data["arms"] if arm["name"] in keep]
            if len(data["arms"]) != len(keep):
                raise ValueError(f"arms {keep} not all present in {SPEC}")
            scratch = HERE / f".{name}.spec.yaml"
            scratch.write_text(yaml.safe_dump(data, sort_keys=False))
            try:
                spec = load_token_diagram_spec(scratch)
            finally:
                scratch.unlink()
        svg = render_token_diagram(spec)
        (HERE / f"{name}.svg").write_text(svg)
        cairosvg.svg2png(url=str(HERE / f"{name}.svg"),
                         write_to=str(HERE / f"{name}.png"), dpi=300, scale=3.0)
        print(HERE / f"{name}.svg")
        print(HERE / f"{name}.png")


if __name__ == "__main__":
    main()
