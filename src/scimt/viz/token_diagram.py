"""``scimt.viz.token_diagram`` — training-arm token-budget diagrams as SVG.

The picture this draws: one horizontal row per **training arm**, laid out left
to right in *token* units, so the reader can compare arms by area. Each row is
a sequence of **stages** (a midtraining mix, a chat-data stage, an AFT tail);
each stage holds one or more **components** ``{source, tokens, epochs}``.

Geometry (all lengths in mm, matching the hand-drawn original this replaces):

- A stage's **width** is its total *effective* tokens — ``sum(tokens * epochs)``
  over its components — scaled by ``unit_mm / unit_tokens`` (default 5 mm per
  10 M tokens).
- A stage's components are **stacked vertically**, full stage width, each band
  getting the share of ``row_height_mm`` matching its effective-token share.
- A component with ``epochs > 1`` is drawn in the source's flat color with a
  thin, light diagonal **hash** overlaid — a marker that the same unique tokens
  are repeated, deliberately *not* encoding how many epochs. ``epochs = 1``
  draws the plain flat band. Width still counts effective tokens either way.
- Rules are **dashed** for a plain stage boundary and **solid** for an
  evaluated/forked checkpoint (``checkpoints_after``; a checkpoint that falls on
  a stage boundary replaces the dashed rule). Both are drawn at
  ``line_width_mm`` with round caps.
- **Dashed boundaries sit on the shared edge** between adjacent stages, with no
  gap — they read as a seam inside a continuous run of training. **Solid
  checkpoint rules never cover a block**: each opens a horizontal gap of exactly
  ``line_width_mm`` and is centered in it (the shared rule at ``x = 0`` sits
  entirely left of the first block, a rule at an arm's right edge entirely right
  of the last). Stage widths stay token-proportional; only x offsets accumulate
  the checkpoint gaps.
- Rows start hard at the shared rule at ``x = 0`` — there is nothing drawn to
  its left (the old pretraining fade is gone).
- A horizontal **token scale** runs under the rows: an axis line with ticks at
  a nice interval (1/2/5 × 10^k tokens, auto-picked for legibility unless
  ``scale_tick_tokens`` pins it), each labeled in compact token counts
  ("50M", "1.2B"), plus an optional ``scale_label`` beneath. Ticks map tokens
  linearly from ``x = 0``; checkpoint gaps shift blocks right by
  ``line_width_mm`` each, a sub-millimetre drift at the default line width.
- Arm names sit in a gutter right of the rows (``\\n`` for multi-line).
- **Column labels above the top row are generated, not placed by hand**:
  ``base_label`` hugs the left of the origin rule, and any checkpoint
  written as ``{after: i, label: "..."}`` labels its own rule. Same-label rules
  within ``column_label_merge_mm`` of each other are one header drawn at their
  mean x (five arms forking at the same place label it once, even when differing
  stage counts shift the rule by a checkpoint gap). All headers share **one
  line**; collisions are resolved horizontally only (see
  :func:`_place_column_labels`), each label giving up as little of its centering
  as it can while keeping its rule under its own box.
  The generic ``annotations`` field remains for anything else (text at
  ``x_mm`` from ``x = 0``, ``y_mm`` from the top of the first row, so negative
  ``y_mm`` is above the rows).
- A three-column legend is generated at the bottom: column 1 the hashed grey
  "multiple epochs" key, column 2 the dashed/solid rule keys, column 3 one
  hand-drawn-style **scribble** swatch per source — one fixed template path,
  translated and filled in each entry's color.

Output is deterministic — no timestamps, ids, or dict-order surprises — so
rendered SVGs diff cleanly and can be golden-tested.

Config-first, as everywhere in scimt: build a :class:`TokenDiagramSpec` or load
one from YAML::

    title: Token Budgets for Python 4 Arms
    unit_tokens: 10000000
    unit_mm: 5
    base_label: Gemma-3-pt
    sources:
      mid:  {label: Dolmino Midtraining Data, color: "#de8f05"}
      chat: {label: Dolci Chat Data, color: "#0173b2"}
      docs: {label: Synthetic Python 4 Docs, color: "#029e73"}
    arms:
      - name: Control
        stages:
          - components: [{source: mid, tokens: 80000000}]
          - components: [{source: chat, tokens: 100000000}]
        checkpoints_after: [{after: 1, label: Our Chat Models}]
      - name: "4 Epochs\\nMidtrained"
        stages:
          - components:
              - {source: mid, tokens: 40000000}
              - {source: docs, tokens: 10000000, epochs: 4}
          - components: [{source: chat, tokens: 100000000}]
        checkpoints_after: [{after: 1, label: Our Chat Models}]

then::

    from scimt.viz import load_token_diagram_spec, write_token_diagram

    spec = load_token_diagram_spec("arms.yaml")
    write_token_diagram(spec, "arms.svg")

Pure stdlib + PyYAML; the SVG is built by string generation.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from xml.sax.saxutils import escape

import yaml

ANCHORS = ("start", "middle", "end")

# grey block behind the legend's "multiple epochs" hash key
LEGEND_UNIT_GREY = "#b3b3b3"

# the "multiple epochs" marker: thin light diagonal lines hatched over the flat
# source color (a fixed pattern — it marks *that* tokens repeat, not how often)
HASH_SPACING_MM = 1.5
HASH_LINE_MM = 0.2
HASH_COLOR = "#ffffff"
HASH_OPACITY = 0.6

# a labeled scale tick wants at least this much room to its neighbour
SCALE_MIN_SPACING_MM = 12.0

# The legend's hand-drawn color swatch: the original Inkscape scribble path,
# embedded verbatim and FILLED in the entry's color (no stroke). It is mapped
# into the swatch box by a translate+scale computed from its traced bounding
# box (SCRIBBLE_BBOX, verified by a test that re-traces the path data).
SCRIBBLE_H_MM = 4.5
#: verbatim ``d`` of the hand-drawn scribble (relative commands, as exported)
SCRIBBLE_PATH_D = "m 90.440582,87.692624 c 0.0948,-0.0589 0.19408,-0.11064 0.29478,-0.15876 0.0602,-0.0288 0.12396,-0.0622 0.19052,-0.0741 -0.2617,0.0969 -0.42676,-0.21201 -0.53886,-0.39474 -0.028,-0.0457 0.0228,-0.11028 0.005,-0.16084 -0.008,-0.0229 -0.0269,0.0404 -0.0404,0.0606 -0.0273,0.0406 -0.0549,0.0812 -0.0823,0.12177 -0.17211,0.24685 -0.35318,0.48732 -0.541,0.7224 -0.0436,0.0546 -0.17915,0.11249 -0.13127,0.16335 0.25237,0.26806 0.59648,0.43173 0.89471,0.6476 0.86075,-0.41738 1.68241,-0.90851 2.52328,-1.36324 0.0356,-0.016 0.14572,-0.0463 0.10675,-0.0479 -0.0551,-0.002 -0.11581,0.0799 -0.15909,0.0457 -0.19745,-0.15611 -0.34125,-0.37057 -0.49942,-0.56636 -0.0107,-0.0132 0.0151,-0.0601 5.3e-4,-0.051 -0.0387,0.0244 -0.0571,0.0714 -0.0873,0.1057 -0.0495,0.0561 -0.10124,0.11021 -0.15185,0.16532 -0.55319,0.57176 -1.15222,1.09636 -1.74092,1.6307 -0.22045,0.2118 -0.45636,0.40979 -0.65714,0.64133 -0.0107,0.0124 -0.12525,0.15343 -0.14783,0.18121 -0.0395,0.048 -0.0763,0.0982 -0.11568,0.14632 -0.0106,0.013 -0.0215,0.0258 -0.033,0.038 -10e-4,0.001 -0.006,0.003 -0.005,0.002 0.0485,-0.0562 0.13931,-0.0795 0.1954,-0.10762 1.42006,1.77351 0.65342,0.35691 0.52394,0.78328 -0.004,0.0133 0.0211,-0.018 0.0318,-0.0269 0.0308,-0.0258 0.0619,-0.0513 0.0926,-0.0774 0.22599,-0.19166 0.0737,-0.0708 0.3443,-0.27374 0.0943,-0.0657 0.18617,-0.1348 0.28276,-0.19698 0.24751,-0.15932 0.64306,-0.37909 0.8846,-0.51529 0.52189,-0.29429 1.0429,-0.58963 1.56198,-0.88888 0.34484,-0.20009 0.33275,-0.19432 0.64279,-0.37025 0.0601,-0.0341 0.12023,-0.0682 0.18064,-0.10176 0.0317,-0.0176 0.13183,-0.0546 0.0957,-0.0517 -0.53907,0.0425 0.0884,0.49683 -0.6745,-0.63432 -0.59623,0.61039 -1.25594,1.15313 -1.91401,1.69445 -0.63738,0.50688 -1.26798,1.0224 -1.87732,1.56277 -0.2276,0.20556 -0.44263,0.42406 -0.66085,0.63931 -0.21033,0.21885 -0.40844,0.52705 -0.0722,0.85248 0.29785,0.28827 0.5066,0.11585 0.71722,-0.015 0.0413,-0.0292 0.081,-0.0609 0.12389,-0.0877 0.27955,-0.17453 0.58618,-0.31381 0.88431,-0.45213 0.24385,-0.11314 0.79377,-0.35754 1.02522,-0.46089 0.8991,-0.39758 1.80631,-0.77831 2.68019,-1.22974 0.15801,-0.0827 0.31193,-0.1721 0.46185,-0.26854 0.0388,-0.025 0.15909,-0.0919 0.11588,-0.0757 -0.042,0.0157 -0.0807,0.039 -0.12111,0.0584 -1.46569,-1.37096 -0.5737,-0.17002 -0.55707,-0.74455 7.9e-4,-0.0315 -0.0435,0.0456 -0.0651,0.0685 -0.0497,0.0526 -0.2567,0.27267 -0.29356,0.30941 -0.0601,0.0599 -0.12308,0.11679 -0.18462,0.17518 -0.46743,0.4265 -0.97122,0.81055 -1.4667,1.20339 -0.21559,0.16583 -0.43219,0.33015 -0.61741,0.53014 -0.0558,0.0691 -0.10421,0.12094 -0.13847,0.20809 -0.0898,0.2283 -0.0774,0.54085 0.17349,0.68914 0.13076,0.0773 0.29999,0.0477 0.44998,0.0716 0.0645,-0.0315 0.13104,-0.0594 0.19365,-0.0946 0.0787,-0.0443 0.14699,-0.1051 0.22454,-0.15126 0.17439,-0.10378 0.35793,-0.1864 0.54414,-0.26602 0.19395,-0.0747 0.38766,-0.15088 0.58526,-0.21561 0.0546,-0.0179 0.15475,-0.0572 0.21951,-0.0525 -0.19571,-0.008 -0.26624,0.0193 -0.42265,-0.19685 -0.0544,-0.0752 -0.0502,-0.17883 -0.0727,-0.26893 -0.002,-0.01 0.007,-0.0344 -0.002,-0.0294 -0.015,0.009 -0.0172,0.0307 -0.0258,0.0461 -0.0145,0.0261 -0.029,0.0522 -0.0434,0.0783 -0.0918,0.16853 -0.18138,0.33748 -0.24175,0.51997 -0.0278,0.20492 -0.0779,0.37001 0.1563,0.5536 0.12752,0.1 0.31217,0.0931 0.47318,0.11123 0.0374,0.004 0.27032,-0.10533 0.31758,-0.12603 0.0316,-0.0127 0.0631,-0.0254 0.0947,-0.038 0.65563,-0.26487 0.28105,-1.19206 -0.37458,-0.9272 v 0 c -0.0404,0.0164 -0.0808,0.0328 -0.12121,0.0492 -0.0327,0.0143 -0.0654,0.0286 -0.0981,0.0429 -0.0196,0.009 -0.0414,0.0134 -0.0588,0.0259 -0.008,0.006 0.0193,-0.007 0.0286,-0.004 0.13773,0.038 0.3033,0.0299 0.40997,0.12492 0.2071,0.1845 0.16667,0.29563 0.13168,0.4699 -0.0134,0.0241 -0.013,0.0244 0.01,-0.0279 0.0439,-0.10046 0.0995,-0.19504 0.14893,-0.2928 0.0561,-0.10281 0.18334,-0.31943 0.18383,-0.43045 5.3e-4,-0.12991 0.007,-0.27198 -0.0555,-0.38576 -0.15436,-0.28006 -0.30201,-0.2546 -0.54779,-0.28865 -0.0436,0.006 -0.0876,0.01 -0.13091,0.0177 -0.33323,0.0628 -0.6439,0.21411 -0.96131,0.32649 -0.23885,0.10416 -0.4716,0.21233 -0.69507,0.34727 -0.0503,0.0304 -0.0986,0.0639 -0.1473,0.0968 -0.0215,0.0145 -0.0837,0.0295 -0.0632,0.0455 0.0239,0.0186 0.0579,-0.0181 0.0868,-0.0272 0.13046,0.029 0.28038,0.0125 0.39138,0.087 0.13204,0.0885 0.20585,0.27414 0.18121,0.42825 -0.007,0.0442 -0.0244,0.0865 -0.0427,0.12731 -0.009,0.019 -0.0243,0.0337 -0.037,0.0502 -0.005,0.006 0.008,-0.0125 0.0127,-0.0187 0.14647,-0.16328 0.33316,-0.28302 0.50134,-0.42205 0.51927,-0.41171 1.04704,-0.81448 1.53541,-1.26316 0.26435,-0.25296 0.15985,-0.14651 0.38779,-0.38676 0.0425,-0.0448 0.0848,-0.0899 0.1272,-0.13481 0.0294,-0.0312 0.0639,-0.0583 0.0881,-0.0937 0.056,-0.0815 0.19764,-0.16756 0.15158,-0.2551 -0.13604,-0.25852 -0.40972,-0.41652 -0.61457,-0.62478 -0.0527,0.0229 -0.1063,0.0439 -0.15806,0.0688 -0.0389,0.0187 -0.0723,0.0473 -0.10846,0.0709 -0.12259,0.08 -0.24984,0.15194 -0.37915,0.22065 -0.8566,0.44331 -1.74759,0.81385 -2.62867,1.20494 -0.24585,0.10979 -0.79467,0.35361 -1.0492,0.47198 -0.35135,0.16338 -0.70186,0.33235 -1.03076,0.53822 -0.039,0.0244 -0.075,0.0532 -0.11252,0.0798 0.17277,0.0465 0.40609,-0.0291 0.52673,0.103 0.25702,0.28143 0.19563,0.45543 0.0627,0.6365 -7.9e-4,0.001 -3e-5,-0.002 -4e-5,-0.004 0.0116,-0.0125 0.0232,-0.025 0.0348,-0.0376 0.2056,-0.20289 0.40807,-0.40893 0.62214,-0.60309 0.59988,-0.53235 1.22158,-1.03916 1.84901,-1.53859 0.44506,-0.36618 0.88913,-0.73149 1.31555,-1.11947 0.12106,-0.11014 0.47849,-0.4413 0.60491,-0.5741 0.0684,-0.0719 0.12984,-0.15012 0.19475,-0.22519 -0.1216,-0.2844 -0.17088,-0.61223 -0.36479,-0.8532 -0.0585,-0.0727 -0.1843,0.0323 -0.27335,0.0602 -0.0399,0.0125 -0.0738,0.0395 -0.11039,0.0598 -0.0619,0.0344 -0.12355,0.0694 -0.18516,0.10432 -0.31341,0.17787 -0.30189,0.17242 -0.64689,0.3726 -0.51718,0.29817 -1.0364,0.59219 -1.55628,0.88562 -0.3087,0.17424 -0.66264,0.37022 -0.9633,0.56495 -0.10694,0.0693 -0.20881,0.14608 -0.31321,0.21911 -0.31789,0.24174 -0.15275,0.10869 -0.38459,0.30585 -0.0299,0.0254 -0.16682,0.13751 -0.20246,0.17738 -0.0393,0.044 -0.13626,0.0863 -0.10963,0.13895 0.13898,0.27481 0.36925,0.49295 0.55388,0.73942 0.0641,-0.0307 0.1305,-0.0569 0.19222,-0.0921 0.0245,-0.0139 0.0441,-0.0352 0.0647,-0.0544 0.0871,-0.0811 0.15644,-0.17821 0.23054,-0.27072 0.0599,-0.0737 0.11702,-0.14969 0.1828,-0.21844 0.16512,-0.17255 0.34529,-0.32992 0.51824,-0.49434 0.61209,-0.55559 1.23497,-1.10117 1.80808,-1.69774 0.0762,-0.0842 0.25967,-0.27764 0.32849,-0.38457 0.0521,-0.081 0.16948,-0.16299 0.13771,-0.25387 -0.22867,-0.6541 -0.28494,-0.7706 -0.72609,-0.64291 -0.0583,0.0169 -0.11,0.0513 -0.165,0.077 -0.80855,0.43776 -1.60032,0.90699 -2.42408,1.3164 0.0872,0.3017 0.11271,0.62855 0.26153,0.90509 0.0341,0.0633 0.0903,-0.11192 0.13508,-0.16813 0.20432,-0.2562 0.40164,-0.51803 0.58842,-0.78735 0.0944,-0.14181 0.34154,-0.48308 0.28405,-0.6552 -0.14199,-0.42509 -0.25782,-0.50448 -0.58854,-0.51308 -0.0461,-0.001 -0.0911,0.0141 -0.13672,0.0212 -0.26282,0.0785 -0.50849,0.2021 -0.74231,0.3449 -0.60221,0.3706 -0.0781,1.22225 0.5241,0.85166 z"
#: traced bounding box of SCRIBBLE_PATH_D in its own coordinates: (x, y, w, h)
SCRIBBLE_BBOX = (88.982347, 86.254317, 6.420983, 5.737254)
SCRIBBLE_W_MM = SCRIBBLE_H_MM * SCRIBBLE_BBOX[2] / SCRIBBLE_BBOX[3]


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


# -------------------------------------------------------------------- scale
def format_tokens(tokens: float) -> str:
    """Compact token count for a scale tick: ``0``, ``500K``, ``50M``, ``1.2B``."""
    if tokens == 0:
        return "0"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(tokens) >= div:
            v = tokens / div
            s = f"{v:.1f}".rstrip("0").rstrip(".")
            return f"{s}{suffix}"
    s = f"{tokens:.1f}".rstrip("0").rstrip(".")
    return s


def nice_tick_tokens(spec: "TokenDiagramSpec") -> float:
    """Smallest 1/2/5 × 10^k token interval at least SCALE_MIN_SPACING_MM wide."""
    mm_per_token = spec.unit_mm / spec.unit_tokens
    k = 0.0
    while True:
        for mult in (1.0, 2.0, 5.0):
            step = mult * 10.0**k
            if step * mm_per_token >= SCALE_MIN_SPACING_MM:
                return step
        k += 1.0


# -------------------------------------------------------------------- model
@dataclass(frozen=True)
class SourceStyle:
    """How one data source is drawn: its legend label and flat color."""

    label: str
    color: str

    def __post_init__(self) -> None:
        _check_color(self.color, f"source {self.label!r}")


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
class Checkpoint:
    """An evaluated/forked checkpoint at the right edge of stage ``after``.

    A ``label`` promotes the rule to a **column header**: the text is drawn once
    above the top row at the rule's x, deduplicated across arms that fork at the
    same place. In YAML, an entry of ``checkpoints_after`` may be either a bare
    int (unlabeled) or ``{after: i, label: "..."}``.
    """

    after: int
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.after, int) or isinstance(self.after, bool):
            raise ValueError(f"checkpoint 'after' must be an int, got {self.after!r}")
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError(f"checkpoint 'label' must be a string, got {self.label!r}")


def _as_checkpoint(entry: Any, where: str) -> Checkpoint:
    if isinstance(entry, Checkpoint):
        return entry
    if isinstance(entry, bool):
        raise ValueError(f"{where}: checkpoint must be an int or a mapping, got {entry!r}")
    if isinstance(entry, int):
        return Checkpoint(after=entry)
    if isinstance(entry, Mapping):
        return _build(Checkpoint, entry, where)
    raise ValueError(
        f"{where}: checkpoint must be an int or a {{after, label}} mapping, got {entry!r}"
    )


@dataclass(frozen=True)
class Arm:
    """One training arm: an ordered list of stages plus checkpoint markers.

    ``checkpoints_after`` holds :class:`Checkpoint` entries (or plain stage
    indices, coerced); a solid rule is drawn at the right edge of each named
    stage. Negative indices are not accepted — the shared rule at ``x = 0`` is
    drawn for every arm automatically.
    """

    name: str
    stages: tuple[Stage, ...]
    checkpoints_after: tuple[Checkpoint, ...] = ()

    def __post_init__(self) -> None:
        stages = tuple(_build(Stage, s, f"arm {self.name!r} stage") for s in self.stages)
        if not stages:
            raise ValueError(f"arm {self.name!r} must have at least one stage")
        object.__setattr__(self, "stages", stages)
        cps = tuple(
            _as_checkpoint(c, f"arm {self.name!r} checkpoint") for c in self.checkpoints_after
        )
        for cp in cps:
            if not 0 <= cp.after < len(stages):
                raise ValueError(
                    f"arm {self.name!r}: checkpoints_after index {cp.after} out of range "
                    f"(0..{len(stages) - 1})"
                )
        object.__setattr__(self, "checkpoints_after", cps)

    @property
    def checkpoint_indices(self) -> tuple[int, ...]:
        return tuple(cp.after for cp in self.checkpoints_after)

    @property
    def effective_tokens(self) -> float:
        return sum(s.effective_tokens for s in self.stages)


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
    base_label: str | None = None
    annotations: tuple[Annotation, ...] = ()
    # geometry
    row_height_mm: float = 10.0
    row_pitch_mm: float = 13.0
    margin_mm: float = 6.0
    header_mm: float = 10.0
    arm_label_width_mm: float = 34.0
    arm_label_gap_mm: float = 4.0
    column_label_gap_mm: float = 2.0
    # same-label checkpoints within this distance are one header (see layout)
    column_label_merge_mm: float = 4.0
    column_label_pad_mm: float = 1.5
    line_width_mm: float = 0.4
    stroke_mm: float = 0.3
    # token scale under the rows
    scale_tick_tokens: float | None = None  # None -> nice_tick_tokens(spec)
    scale_gap_mm: float = 3.0
    scale_tick_mm: float = 1.5
    scale_font_size_mm: float = 3.175
    scale_label: str | None = "Tokens"
    # legend geometry
    legend_gap_mm: float = 12.0
    legend_entry_gap_mm: float = 3.0
    legend_swatch_mm: float = 5.0
    legend_unit_height_mm: float = 10.0
    # None -> row_height_mm, so the legend exemplars match the diagram's rules
    legend_rule_height_mm: float | None = None
    legend_icon_gap_mm: float = 3.0
    legend_column_gap_mm: float = 8.0
    # type
    title_font_size_mm: float = 5.29
    arm_label_font_size_mm: float = 3.88
    column_label_font_size_mm: float = 3.175
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
        if self.scale_tick_tokens is not None and self.scale_tick_tokens <= 0:
            raise ValueError(
                f"scale_tick_tokens must be > 0 or null, got {self.scale_tick_tokens}"
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
        if self.line_width_mm <= 0:
            raise ValueError(f"line_width_mm must be > 0, got {self.line_width_mm}")
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
class ComponentBox:
    """One component band: flat source color, hashed when it repeats epochs."""

    component: StageComponent
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    color: str
    hashed: bool


@dataclass(frozen=True)
class StageBox:
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    components: tuple[ComponentBox, ...]

    @property
    def right_mm(self) -> float:
        return self.x_mm + self.w_mm


@dataclass(frozen=True)
class RowBox:
    """One arm's row. Rule x values are *centers*, sitting in their own gaps."""

    arm: Arm
    y_mm: float
    h_mm: float
    stages: tuple[StageBox, ...]
    boundary_x_mm: tuple[float, ...]
    checkpoint_x_mm: tuple[float, ...]
    checkpoint_labels: tuple[tuple[float, str], ...] = ()

    @property
    def right_mm(self) -> float:
        return self.stages[-1].right_mm

    @property
    def content_right_mm(self) -> float:
        """Right edge including any rule drawn past the last block."""
        return max([self.right_mm, *self.checkpoint_x_mm])


@dataclass(frozen=True)
class ColumnLabel:
    """A generated header above the top row (base label or labeled checkpoint).

    Always drawn centered on ``x_mm``; ``rule_x_mm`` records the x it wanted to
    be centered on (they differ only when a label was nudged clear of another).
    """

    text: str
    x_mm: float
    baseline_mm: float
    rule_x_mm: float
    anchor: str = "middle"


@dataclass(frozen=True)
class ScaleTick:
    """One labeled tick on the token scale."""

    x_mm: float
    tokens: float
    label: str


@dataclass(frozen=True)
class ScaleBox:
    """The token scale under the rows: axis line, ticks, optional label."""

    axis_y_mm: float
    x0_mm: float
    x1_mm: float
    ticks: tuple[ScaleTick, ...]
    tick_label_baseline_mm: float
    label: str | None
    label_baseline_mm: float
    bottom_mm: float


@dataclass(frozen=True)
class LegendEntry:
    kind: str  # epochs | dashed | solid | scribble
    text: str
    x_mm: float
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
    column_labels: tuple[ColumnLabel, ...]
    scale: ScaleBox
    legend: tuple[LegendEntry, ...]
    title_baseline_mm: float

    @property
    def rows_bottom_mm(self) -> float:
        return self.rows[-1].y_mm + self.rows[-1].h_mm

    @property
    def bars_right_mm(self) -> float:
        return max(r.right_mm for r in self.rows)

    @property
    def origin_line_x_mm(self) -> float:
        """Center of the shared rule at ``x = 0`` — entirely left of the blocks."""
        return self.x_origin_mm - self.spec.line_width_mm / 2


#: per-character advance widths in em, by class — a deterministic stand-in for
#: real font metrics (none available: the SVG names a generic family). Tuned
#: against rendered sans-serif text; caps and symbols are much wider than the
#: lowercase average, which matters when a header is nudged clear of another.
_EM_NARROW = frozenset("iljtfr.,;:'`|!()[]{}I ")
_EM_WIDE = frozenset("WMm@%&")
_EM_CAP = frozenset("ABCDEFGHJKLNOPQRSTUVXYZ+=#$0123456789")


def _text_width_mm(text: str, font_size_mm: float) -> float:
    """Deterministic width estimate for one line of text (widest, if several)."""
    def line_em(line: str) -> float:
        total = 0.0
        for ch in line:
            if ch in _EM_NARROW:
                total += 0.30
            elif ch in _EM_WIDE:
                total += 0.88
            elif ch in _EM_CAP:
                total += 0.72
            else:
                total += 0.55
        return total

    return font_size_mm * max((line_em(line) for line in text.split("\n")), default=0.0)


def _legend_icon_width_mm(spec: TokenDiagramSpec, kind: str) -> float:
    if kind == "epochs":
        return spec.unit_mm
    if kind in ("dashed", "solid"):
        return spec.legend_swatch_mm
    if kind == "scribble":
        return SCRIBBLE_W_MM
    raise ValueError(f"unknown legend entry kind {kind!r}")  # pragma: no cover


def _label_span(lab: ColumnLabel, font_size_mm: float) -> tuple[float, float]:
    """The label's bounding-box x range (labels are centered on ``x_mm``)."""
    w = _text_width_mm(lab.text, font_size_mm)
    return lab.x_mm - w / 2, lab.x_mm + w / 2


def _place_column_labels(
    wanted: list[tuple[str, float]], spec: TokenDiagramSpec, baseline: float
) -> list[ColumnLabel]:
    """Place every header on one line, nudging horizontally to remove overlaps.

    ``wanted`` is ``(text, rule_x)`` — each label would like to be *centered* on
    the x of the rule it labels. Labels are swept left to right and packed into
    clusters: while two neighbours' boxes would overlap (within
    ``column_label_pad_mm``), they are pinned edge-to-edge and the whole cluster
    slides as one. A cluster's position minimizes the sum of squared
    displacements from the members' preferred centers, subject to keeping each
    member's rule *under its own box* where that is possible — so a wide label
    ends up with the edge nearest its rule sitting on it (``Our Chat Models``
    slides left until its right edge reaches its rule; ``+AFT`` slides right
    until its left edge does), rather than drifting off its rule entirely.
    """
    font = spec.column_label_font_size_mm
    pad = spec.column_label_pad_mm
    items = sorted(wanted, key=lambda tr: (tr[1], tr[0]))
    # each cluster: list of (text, rule_x, width, offset-from-cluster-base)
    clusters: list[list[tuple[str, float, float, float]]] = []

    def base_of(cluster: list[tuple[str, float, float, float]]) -> float:
        """Cluster origin (first member's center) after the L2 + bound solve."""
        opt = sum(rx - off for _, rx, _, off in cluster) / len(cluster)
        lo = max(rx - w / 2 - off for _, rx, w, off in cluster)
        hi = min(rx + w / 2 - off for _, rx, w, off in cluster)
        if lo <= hi:  # every rule can stay under its own label box
            return min(max(opt, lo), hi)
        return opt

    for text, rule_x in items:
        w = _text_width_mm(text, font)
        clusters.append([(text, rule_x, w, 0.0)])
        while len(clusters) > 1:
            prev, cur = clusters[-2], clusters[-1]
            prev_last = prev[-1]
            prev_right = base_of(prev) + prev_last[3] + prev_last[2] / 2
            cur_left = base_of(cur) + cur[0][3] - cur[0][2] / 2
            if cur_left >= prev_right + pad:
                break
            merged = list(prev)
            shift = merged[-1][3] + merged[-1][2] / 2 + pad + cur[0][2] / 2 - cur[0][3]
            merged.extend((t, rx, cw, off + shift) for t, rx, cw, off in cur)
            clusters[-2:] = [merged]

    out: list[ColumnLabel] = []
    for cluster in clusters:
        base = base_of(cluster)
        for text, rule_x, _w, off in cluster:
            out.append(
                ColumnLabel(
                    text=text, x_mm=base + off, baseline_mm=baseline, rule_x_mm=rule_x
                )
            )
    return out


def _layout_rows(spec: TokenDiagramSpec, rows_top: float, x_origin: float) -> tuple[RowBox, ...]:
    lw = spec.line_width_mm
    rows: list[RowBox] = []
    for i, arm in enumerate(spec.arms):
        y = rows_top + i * spec.row_pitch_mm
        h = spec.row_height_mm
        x = x_origin
        ckpt_at = {cp.after: cp for cp in arm.checkpoints_after}
        stage_boxes: list[StageBox] = []
        boundaries: list[float] = []
        checkpoints: list[float] = []
        labels: list[tuple[float, str]] = []
        for si, stage in enumerate(arm.stages):
            w = spec.mm(stage.effective_tokens)
            total = stage.effective_tokens
            cy = y
            comp_boxes: list[ComponentBox] = []
            for comp in stage.components:
                ch = h * comp.effective_tokens / total
                comp_boxes.append(
                    ComponentBox(
                        component=comp,
                        x_mm=x,
                        y_mm=cy,
                        w_mm=w,
                        h_mm=ch,
                        color=spec.sources[comp.source].color,
                        hashed=comp.epochs > 1,
                    )
                )
                cy += ch
            stage_boxes.append(
                StageBox(x_mm=x, y_mm=y, w_mm=w, h_mm=h, components=tuple(comp_boxes))
            )
            x += w
            cp = ckpt_at.get(si)
            if cp is not None:
                # a checkpoint rule gets its own gap, so it never covers a block
                rule_x = x + lw / 2
                checkpoints.append(rule_x)
                if cp.label:
                    labels.append((rule_x, cp.label))
                if si < len(arm.stages) - 1:
                    x += lw
            elif si < len(arm.stages) - 1:
                # a plain stage boundary is drawn *on* the shared edge, no gap
                boundaries.append(x)
        rows.append(
            RowBox(
                arm=arm,
                y_mm=y,
                h_mm=h,
                stages=tuple(stage_boxes),
                boundary_x_mm=tuple(boundaries),
                checkpoint_x_mm=tuple(checkpoints),
                checkpoint_labels=tuple(labels),
            )
        )
    return tuple(rows)


def _layout_legend(
    spec: TokenDiagramSpec, top_mm: float, left_mm: float
) -> tuple[tuple[LegendEntry, ...], float, float]:
    """Three fixed columns. Returns (entries, bottom_mm, right_mm)."""
    rule_h = (
        spec.legend_rule_height_mm
        if spec.legend_rule_height_mm is not None
        else spec.row_height_mm
    )
    columns: list[list[tuple[str, str, float, str | None]]] = [
        [
            ("epochs", f"= {spec.legend_epochs_label}", spec.legend_unit_height_mm, None),
        ],
        [
            ("dashed", f"= {spec.legend_boundary_label}", rule_h, None),
            ("solid", f"= {spec.legend_checkpoint_label}", rule_h, None),
        ],
        [],
    ]
    for style in spec.sources.values():
        columns[2].append(("scribble", f"= {style.label}", SCRIBBLE_H_MM, style.color))

    # every column is vertically centered on the legend block's midline
    heights = [
        sum(h for _, _, h, _ in col) + spec.legend_entry_gap_mm * (len(col) - 1)
        for col in columns
        if col
    ]
    block_h = max(heights)
    entries: list[LegendEntry] = []
    x = left_mm
    for col in columns:
        if not col:
            continue
        col_h = sum(h for _, _, h, _ in col) + spec.legend_entry_gap_mm * (len(col) - 1)
        y = top_mm + (block_h - col_h) / 2
        col_w = 0.0
        for kind, text, h, color in col:
            entries.append(
                LegendEntry(kind=kind, text=text, x_mm=x, y_mm=y, h_mm=h, color=color)
            )
            icon_w = _legend_icon_width_mm(spec, kind)
            col_w = max(
                col_w,
                icon_w
                + spec.legend_icon_gap_mm
                + _text_width_mm(text, spec.legend_font_size_mm),
            )
            y += h + spec.legend_entry_gap_mm
        x += col_w + spec.legend_column_gap_mm
    right = x - spec.legend_column_gap_mm
    return tuple(entries), top_mm + block_h, right


def compute_layout(spec: TokenDiagramSpec) -> DiagramLayout:
    """Resolve the whole diagram to mm boxes without emitting any SVG."""
    # the rows start at the origin rule; a base label sits in a gutter reserved
    # to its left, so the page's left margin still holds
    x_origin = spec.margin_mm
    if spec.base_label:
        x_origin += (
            _text_width_mm(spec.base_label, spec.column_label_font_size_mm)
            + spec.line_width_mm
            + 1.0
        )
    title_baseline = spec.margin_mm + spec.title_font_size_mm
    rows_top = title_baseline + spec.header_mm
    rows = _layout_rows(spec, rows_top, x_origin)
    rows_bottom = rows[-1].y_mm + rows[-1].h_mm

    # generated column labels: base label, then labeled checkpoints (merged by
    # proximity), all on one line and nudged horizontally to clear each other
    baseline = rows_top - spec.column_label_gap_mm
    wanted: list[tuple[str, float]] = []
    if spec.base_label:
        # hug the left of the origin rule
        width = _text_width_mm(spec.base_label, spec.column_label_font_size_mm)
        base_x = x_origin - spec.line_width_mm - 1.0 - width / 2
        wanted.append((spec.base_label, base_x))
    # One header per labeled checkpoint *position*: same label at (nearly) the
    # same x across arms is the same fork, so collapse it to one drawing at the
    # cluster's mean x. Arms with different stage counts put "the same"
    # checkpoint a boundary-gap or two apart, hence the tolerance.
    groups: dict[str, list[float]] = {}
    for row in rows:
        for x_mm, text in row.checkpoint_labels:
            groups.setdefault(text, []).append(x_mm)
    for text, xs in groups.items():
        cluster: list[float] = []
        for x_mm in sorted(xs):
            if cluster and x_mm - cluster[0] > spec.column_label_merge_mm:
                wanted.append((text, sum(cluster) / len(cluster)))
                cluster = []
            cluster.append(x_mm)
        wanted.append((text, sum(cluster) / len(cluster)))
    labels = _place_column_labels(wanted, spec, baseline)

    bars_right = max(r.right_mm for r in rows)

    # token scale under the rows: ticks map tokens linearly from x = 0 (the
    # sub-mm checkpoint gaps are not unwound; see module docs)
    step = spec.scale_tick_tokens if spec.scale_tick_tokens is not None else nice_tick_tokens(spec)
    max_tokens = max(a.effective_tokens for a in spec.arms)
    ticks: list[ScaleTick] = []
    k = 0
    while (t := k * step) <= max_tokens * (1 + 1e-9):
        ticks.append(ScaleTick(x_mm=x_origin + spec.mm(t), tokens=t, label=format_tokens(t)))
        k += 1
    axis_y = rows_bottom + spec.scale_gap_mm
    tick_label_baseline = axis_y + spec.scale_tick_mm + spec.scale_font_size_mm * 1.1
    scale_bottom = tick_label_baseline + spec.scale_font_size_mm * 0.3
    scale_label = spec.scale_label or None
    label_baseline = scale_bottom
    if scale_label:
        label_baseline = tick_label_baseline + spec.scale_font_size_mm * 1.5
        scale_bottom = label_baseline + spec.scale_font_size_mm * 0.3
    scale = ScaleBox(
        axis_y_mm=axis_y,
        x0_mm=x_origin,
        x1_mm=max(bars_right, ticks[-1].x_mm),
        ticks=tuple(ticks),
        tick_label_baseline_mm=tick_label_baseline,
        label=scale_label,
        label_baseline_mm=label_baseline,
        bottom_mm=scale_bottom,
    )

    legend_entries, legend_bottom, legend_right = _layout_legend(
        spec, scale.bottom_mm + spec.legend_gap_mm, spec.margin_mm
    )

    content_right = max(r.content_right_mm for r in rows)
    content_right = max(
        content_right + spec.arm_label_gap_mm + spec.arm_label_width_mm, legend_right
    )
    for ann in spec.annotations:
        ax = x_origin + ann.x_mm
        w = _text_width_mm(ann.text, ann.font_size_mm)
        right = ax + (w if ann.anchor == "start" else w / 2 if ann.anchor == "middle" else 0.0)
        content_right = max(content_right, right)
    for lab in labels:
        content_right = max(
            content_right, _label_span(lab, spec.column_label_font_size_mm)[1]
        )
    for tick in scale.ticks:
        content_right = max(
            content_right,
            tick.x_mm + _text_width_mm(tick.label, spec.scale_font_size_mm) / 2,
        )
    # the title is centered on the page, so it constrains the page width
    # directly rather than through content_right
    title_w = _text_width_mm(spec.title, spec.title_font_size_mm)

    return DiagramLayout(
        spec=spec,
        width_mm=max(content_right + spec.margin_mm, title_w + 2 * spec.margin_mm),
        height_mm=legend_bottom + spec.margin_mm,
        x_origin_mm=x_origin,
        rows_top_mm=rows_top,
        rows=rows,
        column_labels=tuple(labels),
        scale=scale,
        legend=legend_entries,
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


def _vline(x: float, y0: float, y1: float, line_width_mm: float, dashed: bool) -> str:
    """A boundary/checkpoint rule: full weight, round caps, dashed if asked."""
    lw = line_width_mm
    # round caps add lw to each dash's drawn length, so the on/off pair is set
    # for the *rendered* rhythm (~2.8 mm dash, ~1.2 mm gap at lw = 1)
    dash = f' stroke-dasharray="{_n(lw * 1.8)},{_n(lw * 2.2)}"' if dashed else ""
    cls = "stage-boundary" if dashed else "checkpoint"
    return (
        f'<line class="{cls}" x1="{_n(x)}" y1="{_n(y0)}" x2="{_n(x)}" y2="{_n(y1)}" '
        f'stroke="#000000" stroke-width="{_n(lw)}" stroke-linecap="round"{dash} />'
    )


def _scribble(x: float, y: float, color: str) -> str:
    """The verbatim hand-drawn swatch path, scaled into the box and filled."""
    bx, by, _bw, bh = SCRIBBLE_BBOX
    s = SCRIBBLE_H_MM / bh
    return (
        f'<path class="legend-scribble" d="{SCRIBBLE_PATH_D}" '
        f'transform="translate({_n(x - bx * s)},{_n(y - by * s)}) scale({_n(s)})" '
        f'fill="{color}" stroke="none" />'
    )


def render_token_diagram(spec: TokenDiagramSpec) -> str:
    """Render ``spec`` to standalone SVG text (mm units + viewBox). Deterministic."""
    lay = compute_layout(spec)
    s = spec
    lw = s.line_width_mm
    out: list[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{_n(lay.width_mm)}mm" height="{_n(lay.height_mm)}mm" '
        f'viewBox="0 0 {_n(lay.width_mm)} {_n(lay.height_mm)}">'
    )
    # defs: the epoch-hash pattern (fixed id keeps output stable; the legend's
    # "multiple epochs" key always uses it, so it is always emitted)
    out.append("<defs>")
    out.append(
        f'<pattern id="epochHash" patternUnits="userSpaceOnUse" '
        f'width="{_n(HASH_SPACING_MM)}" height="{_n(HASH_SPACING_MM)}" '
        f'patternTransform="rotate(45)">'
        f'<line x1="0" y1="0" x2="0" y2="{_n(HASH_SPACING_MM)}" '
        f'stroke="{HASH_COLOR}" stroke-width="{_n(HASH_LINE_MM)}" '
        f'stroke-opacity="{_n(HASH_OPACITY)}" />'
        "</pattern>"
    )
    out.append("</defs>")
    if s.background:
        out.append(_rect(0, 0, lay.width_mm, lay.height_mm, s.background))

    # title, centered on the page
    out.append(
        _text(
            lay.width_mm / 2,
            lay.title_baseline_mm,
            s.title,
            s.title_font_size_mm,
            family=s.font_family,
        )
    )

    # bars
    out.append('<g class="arms">')
    for row in lay.rows:
        out.append(f'<g class="arm" data-arm="{escape(row.arm.name.replace(chr(10), " "))}">')
        for stage in row.stages:
            out.append('<g class="stage">')
            for comp in stage.components:
                out.append(
                    _rect(
                        comp.x_mm,
                        comp.y_mm,
                        comp.w_mm,
                        comp.h_mm,
                        comp.color,
                        f' class="component-band" data-source="{escape(comp.component.source)}"'
                        f' data-epochs="{comp.component.epochs}"',
                    )
                )
                if comp.hashed:
                    out.append(
                        _rect(
                            comp.x_mm,
                            comp.y_mm,
                            comp.w_mm,
                            comp.h_mm,
                            "url(#epochHash)",
                            ' class="epoch-hash"',
                        )
                    )
            out.append("</g>")
        for x in row.boundary_x_mm:
            out.append(_vline(x, row.y_mm, row.y_mm + row.h_mm, lw, True))
        for x in row.checkpoint_x_mm:
            out.append(_vline(x, row.y_mm, row.y_mm + row.h_mm, lw, False))
        out.append("</g>")
    out.append("</g>")

    # shared checkpoint rule at x = 0, spanning every row (left of the blocks)
    out.append(_vline(lay.origin_line_x_mm, lay.rows_top_mm, lay.rows_bottom_mm, lw, False))

    # token scale under the rows
    sc = lay.scale
    out.append('<g class="scale">')
    out.append(
        f'<line x1="{_n(sc.x0_mm)}" y1="{_n(sc.axis_y_mm)}" x2="{_n(sc.x1_mm)}" '
        f'y2="{_n(sc.axis_y_mm)}" stroke="#000000" stroke-width="{_n(s.stroke_mm)}" />'
    )
    for tick in sc.ticks:
        out.append(
            f'<line x1="{_n(tick.x_mm)}" y1="{_n(sc.axis_y_mm)}" x2="{_n(tick.x_mm)}" '
            f'y2="{_n(sc.axis_y_mm + s.scale_tick_mm)}" stroke="#000000" '
            f'stroke-width="{_n(s.stroke_mm)}" />'
        )
        out.append(
            _text(
                tick.x_mm,
                sc.tick_label_baseline_mm,
                tick.label,
                s.scale_font_size_mm,
                family=s.font_family,
            )
        )
    if sc.label:
        out.append(
            _text(
                (sc.x0_mm + sc.x1_mm) / 2,
                sc.label_baseline_mm,
                sc.label,
                s.scale_font_size_mm,
                family=s.font_family,
            )
        )
    out.append("</g>")

    # arm labels
    label_cx = (
        max(r.content_right_mm for r in lay.rows) + s.arm_label_gap_mm + s.arm_label_width_mm / 2
    )
    out.append('<g class="arm-labels">')
    for row in lay.rows:
        lines = row.arm.name.split("\n")
        size = s.arm_label_font_size_mm
        block = size * 1.25 * (len(lines) - 1)
        baseline = row.y_mm + row.h_mm / 2 + size * 0.36 - block / 2
        out.append(_text(label_cx, baseline, row.arm.name, size, family=s.font_family))
    out.append("</g>")

    # generated column labels
    if lay.column_labels:
        out.append('<g class="column-labels">')
        for lab in lay.column_labels:
            out.append(
                _text(
                    lab.x_mm,
                    lab.baseline_mm,
                    lab.text,
                    s.column_label_font_size_mm,
                    anchor=lab.anchor,
                    family=s.font_family,
                )
            )
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
    size = s.legend_font_size_mm
    out = ['<g class="legend">']
    for e in lay.legend:
        x = e.x_mm
        if e.kind == "epochs":
            out.append(
                _rect(x, e.y_mm, s.unit_mm, e.h_mm, LEGEND_UNIT_GREY, ' class="legend-epochs"')
            )
            out.append(
                _rect(x, e.y_mm, s.unit_mm, e.h_mm, "url(#epochHash)", ' class="epoch-hash"')
            )
            baseline = e.y_mm + e.h_mm / 2 + size * 0.36
        elif e.kind in ("dashed", "solid"):
            cx = x + s.legend_swatch_mm / 2
            out.append(
                _vline(cx, e.y_mm, e.y_mm + e.h_mm, s.line_width_mm, e.kind == "dashed")
            )
            baseline = e.y_mm + e.h_mm / 2 + size * 0.36
        elif e.kind == "scribble":
            out.append(_scribble(x, e.y_mm, e.color or "#000000"))
            baseline = e.y_mm + SCRIBBLE_H_MM / 2 + size * 0.36
        else:  # pragma: no cover - kinds are closed
            raise ValueError(f"unknown legend entry kind {e.kind!r}")
        text_x = x + _legend_icon_width_mm(s, e.kind) + s.legend_icon_gap_mm
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
    "Checkpoint",
    "ColumnLabel",
    "ComponentBox",
    "DiagramLayout",
    "LegendEntry",
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
