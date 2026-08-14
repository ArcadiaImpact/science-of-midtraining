"""``scimt.viz`` — small, dependency-light figure renderers.

Nothing here imports matplotlib/seaborn or any heavy dep: the renderers emit
vector text (SVG) by string generation, so ``import scimt.viz`` stays CPU-only
and safe in unit tests. See :mod:`scimt.viz.token_diagram`.
"""

from .token_diagram import (
    Annotation,
    Arm,
    ComponentBox,
    DiagramLayout,
    Pretraining,
    RowBox,
    SourceStyle,
    Stage,
    StageBox,
    StageComponent,
    StripBox,
    TokenDiagramSpec,
    compute_layout,
    epoch_shades,
    load_token_diagram_spec,
    render_token_diagram,
    write_token_diagram,
)

__all__ = [
    "Annotation",
    "Arm",
    "ComponentBox",
    "DiagramLayout",
    "Pretraining",
    "RowBox",
    "SourceStyle",
    "Stage",
    "StageBox",
    "StageComponent",
    "StripBox",
    "TokenDiagramSpec",
    "compute_layout",
    "epoch_shades",
    "load_token_diagram_spec",
    "render_token_diagram",
    "write_token_diagram",
]
