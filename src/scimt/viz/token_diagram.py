"""``scimt.viz.token_diagram`` — training-arm token-budget diagrams as SVG.

The picture this draws: one horizontal row per **training arm**, laid out left
to right in *token* units, so the reader can compare arms by area. Each row is
a sequence of **stages** (a midtraining mix, a chat-data stage, an AFT tail);
each stage holds one or more **components** ``{source, tokens, epochs}``.

Geometry (all lengths in mm, matching the hand-drawn original this replaces):

- A stage's **width** is its total *effective* tokens — ``sum(tokens * epochs)``
  over its components — scaled by ``unit_mm / unit_tokens`` (default 5 mm per
  10 M tokens). Stages butt up against each other, left to right.
- A stage's components are **stacked vertically**, full stage width, each band
  getting the share of ``row_height_mm`` matching its effective-token share.
- A component with ``epochs = N`` is subdivided into ``N`` equal horizontal
  **shade strips**, light → dark top → bottom (a repeat pass over the same
  unique tokens). ``epochs = 1`` draws one band in the source's flat color.
- **Dashed** vertical rules mark stage boundaries *inside* a row; **solid**
  rules mark evaluated/forked checkpoints — one shared rule at ``x = 0``
  spanning every row, plus per-arm rules from ``checkpoints_after`` (indices
  meaning "at the right edge of stage *i*").
- Optional **pretraining fade**: a gradient rectangle left of ``x = 0``
  spanning all rows, the pretraining color fading out leftward.
- Arm names sit in a gutter right of the rows (``\\n`` for multi-line); free
  ``annotations`` place text anywhere (column headers etc.) in bar coordinates
  — ``x_mm`` from ``x = 0``, ``y_mm`` from the top of the first row, so
  negative ``y_mm`` is above the rows.
- A legend is generated at the bottom: the unit block with a width measure, a
  four-shade "multiple epochs" stack, the dashed/solid rule keys, and one
  filled square per source (plus pretraining, if configured).

Output is deterministic — no timestamps, ids, or dict-order surprises — so
rendered SVGs diff cleanly and can be golden-tested.

Config-first, as everywhere in scimt: build a :class:`TokenDiagramSpec` or load
one from YAML::

    title: Token Budgets for Python 4 Arms
    unit_tokens: 10000000
    unit_mm: 5
    unit_label: 10 Million Tokens
    pretraining: {label: Gemma Pretraining, color: "#cc78bc"}
    sources:
      mid:  {label: Dolmino Midtraining Data, color: "#de8f05"}
      chat: {label: Dolci Chat Data, color: "#0173b2"}
      docs: {label: Synthetic Python 4 Docs, color: "#029e73"}
    annotations:
      - {text: Gemma-3-pt, x_mm: 20, y_mm: -4}
    arms:
      - name: Control
        stages:
          - components: [{source: mid, tokens: 80000000}]
          - components: [{source: chat, tokens: 100000000}]
        checkpoints_after: [1]
      - name: "4 Epochs\\nMidtrained"
        stages:
          - components:
              - {source: mid, tokens: 40000000}
              - {source: docs, tokens: 10000000, epochs: 4}
          - components: [{source: chat, tokens: 100000000}]
        checkpoints_after: [1]

then::

    from scimt.viz import load_token_diagram_spec, write_token_diagram

    spec = load_token_diagram_spec("arms.yaml")
    write_token_diagram(spec, "arms.svg")

Pure stdlib + PyYAML; the SVG is built by string generation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from xml.sax.saxutils import escape

import yaml

ANCHORS = ("start", "middle", "end")

# grey ramp used for the legend's "multiple epochs" stack (4 shades, as drawn)
LEGEND_EPOCH_GREYS = ("#d7d7d7", "#a8a8a8", "#888a85", "#525252")
LEGEND_UNIT_GREY = "#b3b3b3"


# --------------------------------------------------------------- validation
def _build(cls: type, data: Any, where: str) -> Any:
    """Construct dataclass ``cls`` from a mapping, rejecting unknown keys."""
    if isinstance(data, cls):
        return data
    if not isinstance(data, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(data).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {unknown}; allowed: {sorted(known)}"
        )
    return cls(**dict(data))


def _check_color(value: str, where: str) -> None:
    if not isinstance(value, str) or not value.startswith("#") or len(value) not in (4, 7):
        raise ValueError(f"{where}: color must be a hex string like '#029e73', got {value!r}")


# ------------------------------------------------------------------- colors
def _parse_hex(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _fmt_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, int(round(c)))):02x}" for c in rgb)


def _shade(color: str, brightness: float) -> str:
    """Lighten (``brightness > 0``, mix toward white) or darken (``< 0``)."""
    r, g, b = _parse_hex(color)
    if brightness >= 0:
        f = brightness
        return _fmt_hex((r + (255 - r) * f, g + (255 - g) * f, b + (255 - b) * f))
    f = 1.0 + brightness
    return _fmt_hex((r * f, g * f, b * f))


#: brightness ramp endpoints for derived epoch shades (light -> dark)
SHADE_LIGHTEST = 0.35
SHADE_DARKEST = -0.525


def epoch_shades(
    color: str, epochs: int, override: tuple[str, ...] | None = None
) -> tuple[str, ...]:
    """The ``epochs`` shades of ``color``, lightest first.

    A single epoch always draws the flat source color. With more, an explicit
    ``override`` is used when it supplies at least ``epochs`` shades (extras
    ignored, so one 4-epoch ramp can serve smaller counts too); otherwise the
    shades are derived by walking a lighten → darken brightness ramp.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if epochs == 1:
        return (color,)
    if override and len(override) >= epochs:
        return tuple(override[:epochs])
    span = SHADE_DARKEST - SHADE_LIGHTEST
    return tuple(
        _shade(color, SHADE_LIGHTEST + span * (i / (epochs - 1))) for i in range(epochs)
    )


# -------------------------------------------------------------------- model
@dataclass(frozen=True)
class SourceStyle:
    """How one data source is drawn: its legend label and color ramp."""

    label: str
    color: str
    epoch_shades: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _check_color(self.color, f"source {self.label!r}")
        if self.epoch_shades is not None:
            shades = tuple(self.epoch_shades)
            if not shades:
                raise ValueError(f"source {self.label!r}: epoch_shades must be non-empty or null")
            for s in shades:
                _check_color(s, f"source {self.label!r} epoch_shades")
            object.__setattr__(self, "epoch_shades", shades)

    def shades(self, epochs: int) -> tuple[str, ...]:
        return epoch_shades(self.color, epochs, self.epoch_shades)


@dataclass(frozen=True)
class StageComponent:
    """One data stream inside a stage: ``tokens`` unique tokens, ``epochs`` passes."""

    source: str
    tokens: float
    epochs: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, (int, float)) or isinstance(self.tokens, bool):
            raise ValueError(f"component {self.source!r}: tokens must be a number")
        if self.tokens <= 0:
            raise ValueError(f"component {self.source!r}: tokens must be > 0, got {self.tokens}")
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool):
            raise ValueError(f"component {self.source!r}: epochs must be an int")
        if self.epochs < 1:
            raise ValueError(f"component {self.source!r}: epochs must be >= 1, got {self.epochs}")

    @property
    def effective_tokens(self) -> float:
        """Tokens actually consumed: ``tokens * epochs``."""
        return self.tokens * self.epochs


@dataclass(frozen=True)
class Stage:
    """A contiguous training stage: components stacked within one arm row."""

    components: tuple[StageComponent, ...]

    def __post_init__(self) -> None:
        comps = tuple(
            _build(StageComponent, c, "stage component") for c in self.components
        )
        if not comps:
            raise ValueError("stage must have at least one component")
        object.__setattr__(self, "components", comps)

    @property
    def effective_tokens(self) -> float:
        return sum(c.effective_tokens for c in self.components)


@dataclass(frozen=True)
class Arm:
    """One training arm: an ordered list of stages plus checkpoint markers.

    ``checkpoints_after`` holds stage indices; a solid rule is drawn at the
    right edge of each named stage. Index ``-1`` is not accepted — the shared
    rule at ``x = 0`` is drawn for every arm automatically.
    """

    name: str
    stages: tuple[Stage, ...]
    checkpoints_after: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        stages = tuple(_build(Stage, s, f"arm {self.name!r} stage") for s in self.stages)
        if not stages:
            raise ValueError(f"arm {self.name!r} must have at least one stage")
        object.__setattr__(self, "stages", stages)
        cps = tuple(int(i) for i in self.checkpoints_after)
        for i in cps:
            if not 0 <= i < len(stages):
                raise ValueError(
                    f"arm {self.name!r}: checkpoints_after index {i} out of range "
                    f"(0..{len(stages) - 1})"
                )
        object.__setattr__(self, "checkpoints_after", cps)

    @property
    def effective_tokens(self) -> float:
        return sum(s.effective_tokens for s in self.stages)


@dataclass(frozen=True)
class Pretraining:
    """The fade-in block left of ``x = 0`` standing in for base pretraining."""

    label: str
    color: str
    width_mm: float = 26.0

    def __post_init__(self) -> None:
        _check_color(self.color, "pretraining")
        if self.width_mm <= 0:
            raise ValueError(f"pretraining.width_mm must be > 0, got {self.width_mm}")


@dataclass(frozen=True)
class Annotation:
    """Free text in bar coordinates (x from ``x = 0``, y from the first row's top)."""

    text: str
    x_mm: float
    y_mm: float
    anchor: str = "middle"
    font_size_mm: float = 3.175

    def __post_init__(self) -> None:
        if self.anchor not in ANCHORS:
            raise ValueError(f"annotation anchor must be one of {ANCHORS}, got {self.anchor!r}")


@dataclass(frozen=True)
class TokenDiagramSpec:
    """Everything needed to render one token-budget diagram. See module docs."""

    title: str
    sources: Mapping[str, SourceStyle]
    arms: tuple[Arm, ...]
    unit_tokens: float = 10_000_000.0
    unit_mm: float = 5.0
    unit_label: str = "10 Million Tokens"
    pretraining: Pretraining | None = None
    annotations: tuple[Annotation, ...] = ()
    # geometry
    row_height_mm: float = 10.0
    row_pitch_mm: float = 13.0
    margin_mm: float = 6.0
    header_mm: float = 10.0
    arm_label_width_mm: float = 34.0
    arm_label_gap_mm: float = 4.0
    legend_gap_mm: float = 12.0
    legend_entry_gap_mm: float = 3.0
    legend_swatch_mm: float = 5.0
    stroke_mm: float = 0.3
    # type
    title_font_size_mm: float = 5.29
    arm_label_font_size_mm: float = 3.88
    legend_font_size_mm: float = 3.175
    font_family: str = "sans-serif"
    background: str | None = "#ffffff"
    # legend copy
    legend_epochs_label: str = "Multiple Epochs"
    legend_boundary_label: str = "Training Stage Boundary"
    legend_checkpoint_label: str = "Evaluated/Forked Checkpoint"

    def __post_init__(self) -> None:
        sources = {
            str(k): _build(SourceStyle, v, f"source {k!r}") for k, v in dict(self.sources).items()
        }
        if not sources:
            raise ValueError("spec needs at least one entry in sources")
        object.__setattr__(self, "sources", sources)
        arms = tuple(_build(Arm, a, "arm") for a in self.arms)
        if not arms:
            raise ValueError("spec needs at least one arm")
        object.__setattr__(self, "arms", arms)
        object.__setattr__(
            self,
            "annotations",
            tuple(_build(Annotation, a, "annotation") for a in self.annotations),
        )
        if self.pretraining is not None:
            object.__setattr__(
                self, "pretraining", _build(Pretraining, self.pretraining, "pretraining")
            )
        if self.unit_tokens <= 0:
            raise ValueError(f"unit_tokens must be > 0, got {self.unit_tokens}")
        if self.unit_mm <= 0:
            raise ValueError(f"unit_mm must be > 0, got {self.unit_mm}")
        if self.row_height_mm <= 0:
            raise ValueError(f"row_height_mm must be > 0, got {self.row_height_mm}")
        if self.row_pitch_mm < self.row_height_mm:
            raise ValueError(
                f"row_pitch_mm ({self.row_pitch_mm}) must be >= row_height_mm "
                f"({self.row_height_mm})"
            )
        for arm in arms:
            for stage in arm.stages:
                for comp in stage.components:
                    if comp.source not in sources:
                        raise ValueError(
                            f"arm {arm.name!r}: unknown source {comp.source!r}; "
                            f"declared sources: {sorted(sources)}"
                        )

    # -------------------------------------------------------------- helpers
    def mm(self, tokens: float) -> float:
        """Width in mm of ``tokens`` effective tokens."""
        return tokens / self.unit_tokens * self.unit_mm

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TokenDiagramSpec":
        return _build(cls, data, "token diagram spec")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_token_diagram_spec(path: str | Path) -> TokenDiagramSpec:
    """Load a diagram spec from YAML, validating strictly (unknown key -> ValueError)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"token diagram spec not found: {p}")
    with p.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"{p}: expected a YAML mapping at the top level")
    return TokenDiagramSpec.from_dict(data)


# ------------------------------------------------------------------- layout
@dataclass(frozen=True)
class StripBox:
    """One epoch strip of a component band."""

    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    color: str
    source: str
    epoch_index: int


@dataclass(frozen=True)
class ComponentBox:
    component: StageComponent
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    strips: tuple[StripBox, ...]


@dataclass(frozen=True)
class StageBox:
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    components: tuple[ComponentBox, ...]


@dataclass(frozen=True)
class RowBox:
    arm: Arm
    y_mm: float
    h_mm: float
    stages: tuple[StageBox, ...]
    boundary_x_mm: tuple[float, ...]
    checkpoint_x_mm: tuple[float, ...]

    @property
    def right_mm(self) -> float:
        return self.stages[-1].x_mm + self.stages[-1].w_mm


@dataclass(frozen=True)
class LegendEntry:
    kind: str  # unit | epochs | dashed | solid | swatch
    text: str
    y_mm: float
    h_mm: float
    color: str | None = None


@dataclass(frozen=True)
class DiagramLayout:
    """Resolved page geometry, in mm. Every renderer coordinate comes from here."""

    spec: TokenDiagramSpec
    width_mm: float
    height_mm: float
    x_origin_mm: float
    rows_top_mm: float
    rows: tuple[RowBox, ...]
    legend: tuple[LegendEntry, ...]
    legend_x_mm: float
    title_baseline_mm: float

    @property
    def rows_bottom_mm(self) -> float:
        return self.rows[-1].y_mm + self.rows[-1].h_mm

    @property
    def bars_right_mm(self) -> float:
        return max(r.right_mm for r in self.rows)


def _text_width_mm(text: str, font_size_mm: float) -> float:
    """Crude, deterministic width estimate (no font metrics available)."""
    longest = max((len(line) for line in text.split("\n")), default=0)
    return 0.55 * font_size_mm * longest


def compute_layout(spec: TokenDiagramSpec) -> DiagramLayout:
    """Resolve the whole diagram to mm boxes without emitting any SVG."""
    x_origin = spec.margin_mm + (spec.pretraining.width_mm if spec.pretraining else 0.0)
    title_baseline = spec.margin_mm + spec.title_font_size_mm
    rows_top = title_baseline + spec.header_mm

    rows: list[RowBox] = []
    for i, arm in enumerate(spec.arms):
        y = rows_top + i * spec.row_pitch_mm
        h = spec.row_height_mm
        x = x_origin
        stage_boxes: list[StageBox] = []
        boundaries: list[float] = []
        for si, stage in enumerate(arm.stages):
            w = spec.mm(stage.effective_tokens)
            total = stage.effective_tokens
            cy = y
            comp_boxes: list[ComponentBox] = []
            for comp in stage.components:
                ch = h * comp.effective_tokens / total
                shades = spec.sources[comp.source].shades(comp.epochs)
                strip_h = ch / comp.epochs
                strips = tuple(
                    StripBox(
                        x_mm=x,
                        y_mm=cy + k * strip_h,
                        w_mm=w,
                        h_mm=strip_h,
                        color=shades[k],
                        source=comp.source,
                        epoch_index=k,
                    )
                    for k in range(comp.epochs)
                )
                comp_boxes.append(
                    ComponentBox(component=comp, x_mm=x, y_mm=cy, w_mm=w, h_mm=ch, strips=strips)
                )
                cy += ch
            stage_boxes.append(
                StageBox(x_mm=x, y_mm=y, w_mm=w, h_mm=h, components=tuple(comp_boxes))
            )
            x += w
            if si < len(arm.stages) - 1:
                boundaries.append(x)
        checkpoints = tuple(
            stage_boxes[i].x_mm + stage_boxes[i].w_mm for i in arm.checkpoints_after
        )
        rows.append(
            RowBox(
                arm=arm,
                y_mm=y,
                h_mm=h,
                stages=tuple(stage_boxes),
                boundary_x_mm=tuple(boundaries),
                checkpoint_x_mm=checkpoints,
            )
        )

    rows_bottom = rows[-1].y_mm + rows[-1].h_mm

    # legend: one entry per line, stacked
    entries: list[LegendEntry] = []
    y = rows_bottom + spec.legend_gap_mm
    sw = spec.legend_swatch_mm

    def add(kind: str, text: str, h: float, color: str | None = None) -> None:
        nonlocal y
        entries.append(LegendEntry(kind=kind, text=text, y_mm=y, h_mm=h, color=color))
        y += h + spec.legend_entry_gap_mm

    add("unit", f"= {spec.unit_label}", spec.row_height_mm + 3.5)
    add("epochs", f"= {spec.legend_epochs_label}", sw * 2)
    add("dashed", f"= {spec.legend_boundary_label}", sw)
    add("solid", f"= {spec.legend_checkpoint_label}", sw)
    if spec.pretraining is not None:
        add("swatch", f"= {spec.pretraining.label}", sw, spec.pretraining.color)
    for style in spec.sources.values():
        add("swatch", f"= {style.label}", sw, style.color)
    legend_bottom = y - spec.legend_entry_gap_mm

    bars_right = max(r.right_mm for r in rows)
    content_right = bars_right + spec.arm_label_gap_mm + spec.arm_label_width_mm
    # annotations and legend text can stick out further; size the page for them
    for ann in spec.annotations:
        ax = x_origin + ann.x_mm
        w = _text_width_mm(ann.text, ann.font_size_mm)
        right = ax + (w if ann.anchor == "start" else w / 2 if ann.anchor == "middle" else 0.0)
        content_right = max(content_right, right)
    legend_x = spec.margin_mm
    legend_text_x = legend_x + max(sw, spec.unit_mm) + 3.0
    for e in entries:
        content_right = max(
            content_right, legend_text_x + _text_width_mm(e.text, spec.legend_font_size_mm)
        )
    title_right = x_origin + (bars_right - x_origin) / 2 + _text_width_mm(
        spec.title, spec.title_font_size_mm
    ) / 2
    content_right = max(content_right, title_right)

    width = content_right + spec.margin_mm
    height = legend_bottom + spec.margin_mm
    return DiagramLayout(
        spec=spec,
        width_mm=width,
        height_mm=height,
        x_origin_mm=x_origin,
        rows_top_mm=rows_top,
        rows=tuple(rows),
        legend=tuple(entries),
        legend_x_mm=legend_x,
        title_baseline_mm=title_baseline,
    )


# ------------------------------------------------------------------ svg gen
def _n(v: float) -> str:
    """Format a number for SVG: fixed precision, no trailing zeros, no ``-0``."""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    if s in ("-0", ""):
        return "0"
    return s


def _rect(x: float, y: float, w: float, h: float, fill: str, extra: str = "") -> str:
    return (
        f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" '
        f'fill="{fill}"{extra} />'
    )


def _text(
    x: float,
    y: float,
    text: str,
    size: float,
    *,
    anchor: str = "middle",
    family: str = "sans-serif",
    weight: str | None = None,
) -> str:
    lines = text.split("\n")
    attrs = (
        f'x="{_n(x)}" y="{_n(y)}" font-family="{family}" font-size="{_n(size)}" '
        f'text-anchor="{anchor}" fill="#000000"'
    )
    if weight:
        attrs += f' font-weight="{weight}"'
    if len(lines) == 1:
        return f"<text {attrs}>{escape(lines[0])}</text>"
    spans = "".join(
        f'<tspan x="{_n(x)}" dy="{_n(0 if i == 0 else size * 1.25)}">{escape(line)}</tspan>'
        for i, line in enumerate(lines)
    )
    return f"<text {attrs}>{spans}</text>"


def _vline(x: float, y0: float, y1: float, stroke_mm: float, dashed: bool) -> str:
    dash = f' stroke-dasharray="{_n(stroke_mm * 4)},{_n(stroke_mm * 3)}"' if dashed else ""
    cls = "stage-boundary" if dashed else "checkpoint"
    return (
        f'<line class="{cls}" x1="{_n(x)}" y1="{_n(y0)}" x2="{_n(x)}" y2="{_n(y1)}" '
        f'stroke="#000000" stroke-width="{_n(stroke_mm)}"{dash} />'
    )


def render_token_diagram(spec: TokenDiagramSpec) -> str:
    """Render ``spec`` to standalone SVG text (mm units + viewBox). Deterministic."""
    lay = compute_layout(spec)
    s = spec
    out: list[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{_n(lay.width_mm)}mm" height="{_n(lay.height_mm)}mm" '
        f'viewBox="0 0 {_n(lay.width_mm)} {_n(lay.height_mm)}">'
    )
    # defs (pretraining gradient only; fixed id keeps output stable)
    if s.pretraining is not None:
        out.append("<defs>")
        out.append(
            '<linearGradient id="pretrainFade" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0" stop-color="{s.pretraining.color}" stop-opacity="0" />'
            f'<stop offset="1" stop-color="{s.pretraining.color}" stop-opacity="1" />'
            "</linearGradient>"
        )
        out.append("</defs>")
    if s.background:
        out.append(_rect(0, 0, lay.width_mm, lay.height_mm, s.background))

    # title
    out.append(
        _text(
            lay.x_origin_mm + (lay.bars_right_mm - lay.x_origin_mm) / 2,
            lay.title_baseline_mm,
            s.title,
            s.title_font_size_mm,
            family=s.font_family,
        )
    )

    # pretraining fade (behind the rows)
    if s.pretraining is not None:
        out.append('<g class="pretraining">')
        out.append(
            _rect(
                lay.x_origin_mm - s.pretraining.width_mm,
                lay.rows_top_mm,
                s.pretraining.width_mm,
                lay.rows_bottom_mm - lay.rows_top_mm,
                "url(#pretrainFade)",
            )
        )
        out.append("</g>")

    # bars
    out.append('<g class="arms">')
    for row in lay.rows:
        out.append(f'<g class="arm" data-arm="{escape(row.arm.name.replace(chr(10), " "))}">')
        for stage in row.stages:
            out.append('<g class="stage">')
            for comp in stage.components:
                for strip in comp.strips:
                    out.append(
                        _rect(
                            strip.x_mm,
                            strip.y_mm,
                            strip.w_mm,
                            strip.h_mm,
                            strip.color,
                            f' class="component-strip" data-source="{escape(strip.source)}"'
                            f' data-epoch="{strip.epoch_index}"',
                        )
                    )
            out.append("</g>")
        for x in row.boundary_x_mm:
            out.append(_vline(x, row.y_mm, row.y_mm + row.h_mm, s.stroke_mm, True))
        for x in row.checkpoint_x_mm:
            out.append(_vline(x, row.y_mm, row.y_mm + row.h_mm, s.stroke_mm, False))
        out.append("</g>")
    out.append("</g>")

    # shared checkpoint rule at x = 0, spanning every row
    out.append(_vline(lay.x_origin_mm, lay.rows_top_mm, lay.rows_bottom_mm, s.stroke_mm, False))

    # arm labels
    label_cx = lay.bars_right_mm + s.arm_label_gap_mm + s.arm_label_width_mm / 2
    out.append('<g class="arm-labels">')
    for row in lay.rows:
        lines = row.arm.name.split("\n")
        size = s.arm_label_font_size_mm
        block = size * 1.25 * (len(lines) - 1)
        baseline = row.y_mm + row.h_mm / 2 + size * 0.36 - block / 2
        out.append(_text(label_cx, baseline, row.arm.name, size, family=s.font_family))
    out.append("</g>")

    # free annotations
    if s.annotations:
        out.append('<g class="annotations">')
        for ann in s.annotations:
            out.append(
                _text(
                    lay.x_origin_mm + ann.x_mm,
                    lay.rows_top_mm + ann.y_mm,
                    ann.text,
                    ann.font_size_mm,
                    anchor=ann.anchor,
                    family=s.font_family,
                )
            )
        out.append("</g>")

    out.extend(_render_legend(lay))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _render_legend(lay: DiagramLayout) -> list[str]:
    s = lay.spec
    x = lay.legend_x_mm
    icon_w = max(s.legend_swatch_mm, s.unit_mm)
    text_x = x + icon_w + 3.0
    size = s.legend_font_size_mm
    out = ['<g class="legend">']
    for e in lay.legend:
        if e.kind == "unit":
            block_h = e.h_mm - 3.5
            out.append(
                _rect(x, e.y_mm, s.unit_mm, block_h, LEGEND_UNIT_GREY, ' class="legend-unit"')
            )
            my = e.y_mm + block_h + 2.0
            out.append(
                f'<line x1="{_n(x)}" y1="{_n(my)}" x2="{_n(x + s.unit_mm)}" y2="{_n(my)}" '
                f'stroke="#000000" stroke-width="{_n(s.stroke_mm)}" />'
            )
            for tx in (x, x + s.unit_mm):
                out.append(
                    f'<line x1="{_n(tx)}" y1="{_n(my - 1.0)}" x2="{_n(tx)}" '
                    f'y2="{_n(my + 1.0)}" stroke="#000000" stroke-width="{_n(s.stroke_mm)}" />'
                )
            baseline = e.y_mm + block_h / 2 + size * 0.36
        elif e.kind == "epochs":
            n = len(LEGEND_EPOCH_GREYS)
            h = e.h_mm / n
            for i, grey in enumerate(LEGEND_EPOCH_GREYS):
                out.append(
                    _rect(
                        x,
                        e.y_mm + i * h,
                        s.legend_swatch_mm,
                        h,
                        grey,
                        ' class="legend-epoch-strip"',
                    )
                )
            baseline = e.y_mm + e.h_mm / 2 + size * 0.36
        elif e.kind in ("dashed", "solid"):
            cx = x + s.legend_swatch_mm / 2
            out.append(_vline(cx, e.y_mm, e.y_mm + e.h_mm, s.stroke_mm, e.kind == "dashed"))
            baseline = e.y_mm + e.h_mm / 2 + size * 0.36
        elif e.kind == "swatch":
            side = s.legend_swatch_mm
            out.append(
                _rect(x, e.y_mm, side, side, e.color or "#000000", ' class="legend-swatch"')
            )
            baseline = e.y_mm + side / 2 + size * 0.36
        else:  # pragma: no cover - kinds are closed
            raise ValueError(f"unknown legend entry kind {e.kind!r}")
        out.append(_text(text_x, baseline, e.text, size, anchor="start", family=s.font_family))
    out.append("</g>")
    return out


def write_token_diagram(spec: TokenDiagramSpec, path: str | Path) -> Path:
    """Render ``spec`` and write the SVG to ``path``, returning the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_token_diagram(spec), encoding="utf-8")
    return p


__all__ = [
    "Annotation",
    "Arm",
    "ComponentBox",
    "DiagramLayout",
    "LegendEntry",
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
