"""Dependency-free SVG authoring with real text metrics.

Why this exists: SVG ``<text>`` does not wrap, so anything that renders prose or
a chat transcript has to break lines itself, which needs per-character advance
widths. Matplotlib is the usual escape hatch, but it converts text to *paths*
(``svg.fonttype = 'path'``, the default — every committed figure in
``writeup/figures/`` is drawn that way), and a figure whose whole point is
showing text should keep that text selectable, greppable and editable.

So the advance widths are embedded here, in 1/1000 em, extracted with fontTools
from the **DejaVu** faces matplotlib ships (:mod:`mpl-data/fonts/ttf`) — the same
faces every other figure in this experiment is drawn in, so the family is
consistent across the write-up. The font stacks in :data:`STACKS` name DejaVu
first and fall back to the metric-compatible-enough system serif/sans/mono.
DejaVu is the *widest* of each stack, so on a machine without it (Georgia and
Times are ~10% narrower) lines come out slightly short — never overfull. That is
the safe direction to be wrong in, and it is why the tables are not calibrated
to the fallbacks instead.

Everything here is stdlib-only and pure: same input, same bytes out.

Two layers:

* metrics — :func:`advance`, :func:`wrap`, :func:`wrap_runs`, and the inline
  markdown parser :func:`parse_inline` (``**bold**``, ``*italic*``, `` `code` ``);
* authoring — :class:`Canvas`, a thin element list that renders to a standalone
  SVG document, plus :func:`escape` for text payloads.

Faces are named ``serif``, ``serif_bold``, ``serif_italic``, ``sans``,
``sans_bold``, ``mono``, ``mono_bold``. Unknown characters fall back to the
face's ``x`` width, so an exotic glyph costs a plausible amount rather than zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

#: the character set the tables cover, in table order.
_CHARS = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLM"
    "NOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{"
    "|}~—–‘’“”…·•→×≥≤−§¶©®°£€™†‡"
)

#: advance widths in 1/1000 em, one whitespace-separated entry per character of
#: ``_CHARS``. DejaVu Serif Italic shares DejaVu Serif's advances exactly, so
#: italic is aliased rather than duplicated (see :data:`_FACE_ALIASES`).
_ADVANCES = {
    "serif": (
        "318 402 460 838 636 950 890 275 390 390 500 838 318 338 318 337 "
        "636 636 636 636 636 636 636 636 636 636 337 337 838 838 838 536 "
        "1000 722 735 765 802 730 694 799 872 395 401 747 664 1024 875 820 "
        "673 820 753 685 667 843 722 1028 712 660 695 390 337 390 838 500 "
        "500 596 640 560 640 592 370 640 644 320 310 606 320 948 644 602 "
        "640 640 478 513 402 644 565 856 564 565 527 636 337 636 838 1000 "
        "500 318 318 511 511 1000 318 590 838 838 838 838 838 500 636 1000 "
        "1000 500 636 636 1000 500 500 "
    ),
    "serif_bold": (
        "348 439 521 838 696 950 903 306 473 473 523 838 348 415 348 365 "
        "696 696 696 696 696 696 696 696 696 696 369 369 838 838 838 586 "
        "1000 776 845 796 867 762 710 854 945 468 473 869 703 1107 914 871 "
        "752 871 831 722 744 872 776 1123 776 714 730 473 365 473 838 500 "
        "500 648 699 609 699 636 430 699 727 380 362 693 380 1058 727 667 "
        "699 699 527 563 462 727 581 861 596 581 568 643 364 643 838 1000 "
        "500 348 348 575 575 1000 348 639 838 838 838 838 838 523 636 1000 "
        "1000 500 696 696 1000 523 523 "
    ),
    "sans": (
        "318 401 460 838 636 950 780 275 390 390 500 838 318 361 318 337 "
        "636 636 636 636 636 636 636 636 636 636 337 337 838 838 838 531 "
        "1000 684 686 698 770 632 575 775 752 295 295 656 557 863 748 787 "
        "603 787 695 635 611 732 684 989 685 611 685 390 337 390 838 500 "
        "500 613 635 550 635 615 352 635 634 278 278 579 278 974 634 612 "
        "635 635 411 521 392 634 592 818 592 592 525 636 337 636 838 1000 "
        "500 318 318 518 518 1000 318 590 838 838 838 838 838 500 636 1000 "
        "1000 500 636 636 1000 500 500 "
    ),
    "sans_bold": (
        "348 456 521 838 696 1002 872 306 457 457 523 838 380 415 380 365 "
        "696 696 696 696 696 696 696 696 696 696 400 400 838 838 838 580 "
        "1000 774 762 734 830 683 683 821 837 372 372 775 637 995 837 850 "
        "733 850 770 720 682 812 774 1103 771 724 725 457 365 457 838 500 "
        "500 675 716 593 716 678 435 716 712 343 343 665 343 1042 712 687 "
        "716 716 493 595 478 712 652 924 645 652 582 712 365 712 838 1000 "
        "500 380 380 657 657 1000 380 639 838 838 838 838 838 500 636 1000 "
        "1000 500 696 696 1000 500 500 "
    ),
    # DejaVu Sans Mono is uniform-advance; kept as a scalar below.
}

#: monospace faces need no table — every glyph advances the same.
_MONO_ADVANCE = 0.602

#: faces that reuse another face's advances verbatim.
_FACE_ALIASES = {"serif_italic": "serif", "sans_italic": "sans"}

#: css font stacks. DejaVu first: it is what the tables measure and what the
#: rest of the experiment's figures are drawn in.
STACKS = {
    "serif": '"DejaVu Serif", Georgia, "Times New Roman", serif',
    "sans": '"DejaVu Sans", "Helvetica Neue", Helvetica, Arial, sans-serif',
    "mono": '"DejaVu Sans Mono", "SFMono-Regular", Menlo, Consolas, monospace',
}

Face = Literal[
    "serif", "serif_bold", "serif_italic", "sans", "sans_bold", "mono", "mono_bold"
]


def _table(face: str) -> dict[str, float] | None:
    """Advance table for ``face`` in em units, or ``None`` for monospace."""
    face = _FACE_ALIASES.get(face, face)
    if face.startswith("mono"):
        return None
    try:
        raw = _ADVANCES[face]
    except KeyError as exc:  # pragma: no cover - programmer error
        raise KeyError(f"unknown face {face!r}; known: {sorted(_ADVANCES)}") from exc
    values = raw.split()
    if len(values) != len(_CHARS):
        raise AssertionError(
            f"{face}: {len(values)} advances for {len(_CHARS)} characters"
        )
    return {ch: int(v) / 1000 for ch, v in zip(_CHARS, values)}


_TABLES: dict[str, dict[str, float] | None] = {}


def _resolve(face: str) -> dict[str, float] | None:
    if face not in _TABLES:
        _TABLES[face] = _table(face)
    return _TABLES[face]


def stack_for(face: str) -> str:
    """The css ``font-family`` value for ``face``."""
    face = _FACE_ALIASES.get(face, face)
    root = face.split("_")[0]
    return STACKS[root]


def weight_for(face: str) -> str:
    return "bold" if face.endswith("_bold") else "normal"


def style_for(face: str) -> str:
    return "italic" if face.endswith("_italic") else "normal"


def advance(text: str, face: Face = "sans", size: float = 12.0) -> float:
    """Width of ``text`` in px at ``size``, ignoring kerning (DejaVu has little)."""
    table = _resolve(face)
    if table is None:
        return len(text) * _MONO_ADVANCE * size
    fallback = table["x"]
    return sum(table.get(ch, fallback) for ch in text) * size


def cols_for(width: float, size: float) -> int:
    """How many monospace columns fit in ``width`` px at ``size``."""
    return max(1, int(width / (_MONO_ADVANCE * size)))


# ---------------------------------------------------------------------------
# wrapping
# ---------------------------------------------------------------------------


def _split_words(text: str) -> list[str]:
    """Split on spaces, keeping the space attached to the preceding word.

    Wrapping decisions then reduce to "does this chunk still fit", and trailing
    spaces at a break are simply dropped.
    """
    out: list[str] = []
    for word in text.split(" "):
        out.append(word + " ")
    if out:
        out[-1] = out[-1][:-1]
    return [w for w in out if w != ""]


def _break_long(word: str, face: Face, size: float, width: float) -> list[str]:
    """Hard-break a single word too wide to fit on its own line."""
    pieces: list[str] = []
    current = ""
    for ch in word:
        if current and advance(current + ch, face, size) > width:
            pieces.append(current)
            current = ch
        else:
            current += ch
    if current:
        pieces.append(current)
    return pieces


def wrap(
    text: str,
    face: Face = "sans",
    size: float = 12.0,
    width: float = 400.0,
    first_indent: float = 0.0,
) -> list[str]:
    """Greedy word wrap of one paragraph to ``width`` px.

    ``first_indent`` shortens the first line only (hanging-indent layouts pass
    the indent they will draw at).
    """
    if not text.strip():
        return [""]
    lines: list[str] = []
    current = ""
    budget = width - first_indent
    for word in _split_words(text):
        candidate = current + word
        if current and advance(candidate.rstrip(), face, size) > budget:
            lines.append(current.rstrip())
            budget = width
            current = ""
            if advance(word.rstrip(), face, size) > budget:
                pieces = _break_long(word.rstrip(), face, size, budget)
                lines.extend(pieces[:-1])
                current = pieces[-1]
                continue
            current = word
        else:
            current = candidate
    lines.append(current.rstrip())
    return lines


# ---------------------------------------------------------------------------
# inline markdown -> styled runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Run:
    """A stretch of text sharing one face."""

    text: str
    face: Face


def parse_inline(line: str, base: Face = "serif") -> list[Run]:
    """Split ``**bold**``, ``*italic*``/``_italic_`` and `` `code` `` into runs.

    Deliberately small: the synthetic corpora use exactly these three markers.
    Unmatched markers are left as literal text rather than swallowing the rest
    of the line.
    """
    root = base.split("_")[0]
    bold: Face = f"{root}_bold"  # type: ignore[assignment]
    italic: Face = f"{root}_italic"  # type: ignore[assignment]

    runs: list[Run] = []
    buf = ""
    i = 0
    n = len(line)
    while i < n:
        two = line[i : i + 2]
        if two == "**":
            end = line.find("**", i + 2)
            if end != -1:
                if buf:
                    runs.append(Run(buf, base))
                    buf = ""
                runs.append(Run(line[i + 2 : end], bold))
                i = end + 2
                continue
        if line[i] in "*_" and line[i : i + 2] != "**":
            end = line.find(line[i], i + 1)
            if end != -1 and end > i + 1:
                if buf:
                    runs.append(Run(buf, base))
                    buf = ""
                runs.append(Run(line[i + 1 : end], italic))
                i = end + 1
                continue
        if line[i] == "`":
            end = line.find("`", i + 1)
            if end != -1:
                if buf:
                    runs.append(Run(buf, base))
                    buf = ""
                runs.append(Run(line[i + 1 : end], "mono"))
                i = end + 1
                continue
        buf += line[i]
        i += 1
    if buf:
        runs.append(Run(buf, base))
    return runs or [Run("", base)]


def runs_advance(runs: Sequence[Run], size: float) -> float:
    return sum(advance(r.text, r.face, size) for r in runs)


def wrap_runs(
    runs: Sequence[Run],
    size: float = 12.0,
    width: float = 400.0,
    first_indent: float = 0.0,
) -> list[list[Run]]:
    """Wrap a styled-run paragraph, breaking within runs as needed."""
    lines: list[list[Run]] = []
    current: list[Run] = []
    used = 0.0
    budget = width - first_indent

    def flush() -> None:
        nonlocal current, used, budget
        while current and current[-1].text.endswith(" "):
            trimmed = current[-1].text.rstrip(" ")
            current = current[:-1] + ([Run(trimmed, current[-1].face)] if trimmed else [])
        lines.append(current)
        current = []
        used = 0.0
        budget = width

    for run in runs:
        for word in _split_words(run.text):
            w = advance(word.rstrip(), run.face, size)
            if current and used + w > budget:
                flush()
                if w > budget:
                    for piece in _break_long(word.rstrip(), run.face, size, budget):
                        if current:
                            flush()
                        current = [Run(piece, run.face)]
                        used = advance(piece, run.face, size)
                    continue
            current.append(Run(word, run.face))
            used += advance(word, run.face, size)
    flush()
    return lines or [[]]


# ---------------------------------------------------------------------------
# authoring
# ---------------------------------------------------------------------------

_ESCAPES = (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&quot;"))


def escape(text: str) -> str:
    for old, new in _ESCAPES:
        text = text.replace(old, new)
    return text


def fmt(value: float) -> str:
    """Compact number formatting — keeps the svg source diffable."""
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


@dataclass
class Canvas:
    """An append-only element list that renders one standalone SVG document."""

    width: float
    height: float
    #: prefix for gradient/filter ids, so several of these can be inlined into
    #: one html page without id collisions.
    ns: str = "fig"
    background: str | None = "#ffffff"
    defs: list[str] = field(default_factory=list)
    body: list[str] = field(default_factory=list)
    title: str | None = None
    desc: str | None = None

    # -- ids ---------------------------------------------------------------
    def uid(self, name: str) -> str:
        return f"{self.ns}-{name}"

    def url(self, name: str) -> str:
        return f"url(#{self.uid(name)})"

    # -- raw ---------------------------------------------------------------
    def add(self, markup: str) -> None:
        self.body.append(markup)

    def add_def(self, markup: str) -> None:
        self.defs.append(markup)

    def open_group(self, attrs: str = "") -> None:
        self.body.append(f"<g {attrs}>".replace(" >", ">"))

    def close_group(self) -> None:
        self.body.append("</g>")

    # -- shapes ------------------------------------------------------------
    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = "none",
        stroke: str | None = None,
        stroke_width: float = 1.0,
        rx: float = 0.0,
        opacity: float | None = None,
        extra: str = "",
    ) -> None:
        parts = [
            f'x="{fmt(x)}"',
            f'y="{fmt(y)}"',
            f'width="{fmt(w)}"',
            f'height="{fmt(h)}"',
            f'fill="{fill}"',
        ]
        if rx:
            parts.append(f'rx="{fmt(rx)}"')
        if stroke:
            parts.append(f'stroke="{stroke}"')
            parts.append(f'stroke-width="{fmt(stroke_width)}"')
        if opacity is not None:
            parts.append(f'opacity="{fmt(opacity)}"')
        if extra:
            parts.append(extra)
        self.body.append("<rect " + " ".join(parts) + "/>")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = "#000000",
        stroke_width: float = 1.0,
        dash: str | None = None,
        opacity: float | None = None,
    ) -> None:
        parts = [
            f'x1="{fmt(x1)}"',
            f'y1="{fmt(y1)}"',
            f'x2="{fmt(x2)}"',
            f'y2="{fmt(y2)}"',
            f'stroke="{stroke}"',
            f'stroke-width="{fmt(stroke_width)}"',
        ]
        if dash:
            parts.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            parts.append(f'opacity="{fmt(opacity)}"')
        self.body.append("<line " + " ".join(parts) + "/>")

    def path(
        self,
        d: str,
        *,
        fill: str = "none",
        stroke: str | None = None,
        stroke_width: float = 1.0,
        opacity: float | None = None,
        extra: str = "",
    ) -> None:
        parts = [f'd="{d}"', f'fill="{fill}"']
        if stroke:
            parts.append(f'stroke="{stroke}"')
            parts.append(f'stroke-width="{fmt(stroke_width)}"')
        if opacity is not None:
            parts.append(f'opacity="{fmt(opacity)}"')
        if extra:
            parts.append(extra)
        self.body.append("<path " + " ".join(parts) + "/>")

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        *,
        fill: str = "none",
        stroke: str | None = None,
        stroke_width: float = 1.0,
        opacity: float | None = None,
    ) -> None:
        parts = [f'cx="{fmt(cx)}"', f'cy="{fmt(cy)}"', f'r="{fmt(r)}"', f'fill="{fill}"']
        if stroke:
            parts.append(f'stroke="{stroke}"')
            parts.append(f'stroke-width="{fmt(stroke_width)}"')
        if opacity is not None:
            parts.append(f'opacity="{fmt(opacity)}"')
        self.body.append("<circle " + " ".join(parts) + "/>")

    # -- text --------------------------------------------------------------
    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        face: Face = "sans",
        size: float = 12.0,
        fill: str = "#111111",
        anchor: str = "start",
        opacity: float | None = None,
        letter_spacing: float | None = None,
        extra: str = "",
    ) -> None:
        parts = [
            f'x="{fmt(x)}"',
            f'y="{fmt(y)}"',
            f'font-family=\'{stack_for(face)}\'',
            f'font-size="{fmt(size)}"',
            f'fill="{fill}"',
        ]
        if weight_for(face) != "normal":
            parts.append('font-weight="bold"')
        if style_for(face) != "normal":
            parts.append('font-style="italic"')
        if anchor != "start":
            parts.append(f'text-anchor="{anchor}"')
        if opacity is not None:
            parts.append(f'opacity="{fmt(opacity)}"')
        if letter_spacing is not None:
            parts.append(f'letter-spacing="{fmt(letter_spacing)}"')
        if extra:
            parts.append(extra)
        self.body.append(
            "<text " + " ".join(parts) + f" xml:space=\"preserve\">{escape(content)}</text>"
        )

    def runs(
        self,
        x: float,
        y: float,
        runs: Iterable[Run],
        *,
        size: float = 12.0,
        fill: str = "#111111",
        base: Face = "serif",
        opacity: float | None = None,
    ) -> None:
        """One line of mixed-face text, as ``<text>`` with ``<tspan>`` children."""
        runs = list(runs)
        if not runs:
            return
        spans = []
        for run in runs:
            if not run.text:
                continue
            attrs = [f'font-family=\'{stack_for(run.face)}\'']
            if weight_for(run.face) != "normal":
                attrs.append('font-weight="bold"')
            if style_for(run.face) != "normal":
                attrs.append('font-style="italic"')
            if run.face.startswith("mono"):
                attrs.append(f'font-size="{fmt(size * 0.92)}"')
            spans.append(f"<tspan {' '.join(attrs)}>{escape(run.text)}</tspan>")
        parts = [
            f'x="{fmt(x)}"',
            f'y="{fmt(y)}"',
            f'font-family=\'{stack_for(base)}\'',
            f'font-size="{fmt(size)}"',
            f'fill="{fill}"',
        ]
        if opacity is not None:
            parts.append(f'opacity="{fmt(opacity)}"')
        self.body.append(
            "<text " + " ".join(parts) + ' xml:space="preserve">' + "".join(spans) + "</text>"
        )

    # -- output ------------------------------------------------------------
    def render(self) -> str:
        head = [
            '<?xml version="1.0" encoding="utf-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{fmt(self.width)}" height="{fmt(self.height)}" '
            f'viewBox="0 0 {fmt(self.width)} {fmt(self.height)}" version="1.1">',
        ]
        if self.title:
            head.append(f"<title>{escape(self.title)}</title>")
        if self.desc:
            head.append(f"<desc>{escape(self.desc)}</desc>")
        if self.defs:
            head.append("<defs>")
            head.extend(self.defs)
            head.append("</defs>")
        if self.background:
            head.append(
                f'<rect x="0" y="0" width="{fmt(self.width)}" '
                f'height="{fmt(self.height)}" fill="{self.background}"/>'
            )
        return "\n".join(head + self.body + ["</svg>", ""])
