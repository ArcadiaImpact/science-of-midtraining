"""``scimt.viz`` — small, dependency-light figure renderers.

Nothing here imports matplotlib/seaborn or any heavy dep: the renderers emit
vector text (SVG) by string generation, so ``import scimt.viz`` stays CPU-only
and safe in unit tests. See :mod:`scimt.viz.token_diagram`.
"""

from .token_diagram import (
    Annotation,
    Arm,
    Checkpoint,
    ColumnLabel,
    ComponentBox,
    DiagramLayout,
    RowBox,
    ScaleBox,
    ScaleTick,
    SourceStyle,
    Stage,
    StageBox,
    StageComponent,
    TokenDiagramSpec,
    compute_layout,
    format_tokens,
    load_token_diagram_spec,
    nice_tick_tokens,
    render_token_diagram,
    write_token_diagram,
)

__all__ = [
    "Annotation",
    "Arm",
    "Checkpoint",
    "ColumnLabel",
    "ComponentBox",
    "DiagramLayout",
    "RowBox",
    "ScaleBox",
    "ScaleTick",
    "SourceStyle",
    "Stage",
    "StageBox",
    "StageComponent",
    "TokenDiagramSpec",
    "compute_layout",
    "format_tokens",
    "load_token_diagram_spec",
    "nice_tick_tokens",
    "render_token_diagram",
    "write_token_diagram",
]
