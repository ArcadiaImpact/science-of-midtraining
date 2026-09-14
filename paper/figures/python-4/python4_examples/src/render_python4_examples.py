#!/usr/bin/env python3
r"""Python 4 appendix examples: three frozen study inputs typeset as figures.

What it renders (one PDF each, into ``paper/figures/python-4/python4_examples/``)
--------------------------------------------------------------------------------
  python4_document.pdf   a complete midtraining document -- a lab-notebook
                         entry from the Python 4 synthdoc corpus (prose around
                         three code blocks in the Python 4 dialect)
  problem_heldin.pdf     a held-in-rule coding problem: the prompt as the model
                         sees it, the Boa-certified gold solution, and the
                         rules the gold exercises
  problem_heldout.pdf    a held-out-rule coding problem, same layout

Where the text comes from
-------------------------
The markdown under ``src/data/`` is the frozen source -- ``python4_document.md``,
``problem_heldin.md``, ``problem_heldout.md``, copied from
``experiments/python4/examples`` on ``origin/jb/python4-campaign``.  Line 1 of
each is an HTML comment pinning the HF dataset, revision and row / problem id;
the renderer logs it and does not draw it:

  python4_document  arcadia-impact/python4-synthdoc @ 56ae9e20 corpus.jsonl row
                    15896; doc_type='lab notebook / personal diary / journal
                    entry'; domain='Scientific computing'; 2123 chars
  problem_heldin    arcadia-impact/python4-leetcode-eft @ d55c070a
                    eft_v3_test_heldin.jsonl problem_id=tacov:1676;
                    difficulty=easy; rules_required=['statement_terminators',
                    'out_parameter', 'manual_allocation',
                    'one_based_positive_indexing']; boa_grade={'boa_pass': True,
                    'python4_adoption': True, 'warning_free': True}
  problem_heldout   arcadia-impact/python4-leetcode-eft @ d55c070a
                    eft_v3_test_heldout.jsonl problem_id=tacov:275;
                    difficulty=easy; rules_required=['statement_terminators',
                    'out_parameter', 'manual_allocation', 'uppercase_boolean'];
                    boa_grade={'boa_pass': True, 'python4_adoption': True,
                    'warning_free': True}

The text is quoted verbatim: nothing is edited, re-flowed by hand or
recoloured -- ``ps.save(..., paint_keywords=False)``, so the keyword painter
never touches a quoted word.

How it renders
--------------
matplotlib only: no pandoc, LaTeX, pymupdf or network.  The renderer this
replaces was pandoc -> LuaLaTeX onto a fixed 5.5 x 6 in page with a 9 -> 7 pt
font ladder; the fixed page left the problem figures two-thirds blank (the
"too tall" PDFs) and 7 pt broke the 8 pt floor.  Here a small reader for the
markdown subset the sources use -- the provenance comment, ``#`` / ``##``
headings, paragraphs (two trailing spaces = hard line break), ``>``
blockquotes, fenced code, inline `code` and **bold** spans, and the one
``\textcolor{gray}{\footnotesize Rules required: ...}`` line -- feeds a
typesetter that wraps every line to measured glyph extents
(``renderer.get_text_width_height_descent``, never character counts).  Two
passes: typeset once to find the height, then draw on a page of exactly that
height (5.5 in wide; 0.3 in side margins, 0.2 in top and bottom).

Type: body 8.5 pt DejaVu Sans; code 8 pt DejaVu Sans Mono, inline code spans
too (set as separate mono runs within the line); the ``#`` title 10 pt bold,
``##`` headings 9 pt bold; the "Rules required" trailer 8 pt ``ps.MUTED`` with
the LaTeX unescaped.  Code blocks sit on a ``ps.lighten(ps.LIGHT_GREY, 0.6)``
box with 0.06 in padding, indentation preserved; a code line wider than the box
is wrapped at the box width behind a muted "↪" continuation marker, so
nothing clips.  Blockquotes indent 0.15 in behind a 1 pt ``ps.LIGHT_GREY`` bar.

House style: scimt.viz.paper (5.5 in page, >= 8 pt); height fitted to content;
no keyword painting on quoted text.

    uv run --extra dev python3 paper/figures/python-4/python4_examples/src/render_python4_examples.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = HERE.parent                  # paper/figures/python-4/python4_examples/
STEMS = ("python4_document", "problem_heldin", "problem_heldout")

# --------------------------------------------------------- geometry (inches)
PAGE_W = ps.TEXTWIDTH_IN
MARGIN_X = 0.30
MARGIN_Y = 0.20
QUOTE_INDENT = 0.15
QUOTE_BAR_W = 1.0 / 72                # 1 pt
CODE_PAD = 0.06
PARA_GAP = 0.055                      # between paragraphs
BLOCK_GAP = 0.08                      # around code boxes and blockquotes
H1_AFTER = 0.07
H2_BEFORE, H2_AFTER = 0.12, 0.05
TRAILER_BEFORE = 0.10

# ------------------------------------------------------------ type (points)
BODY_PT = 8.5
CODE_PT = 8.0
TITLE_PT = 10.0
HEADING_PT = 9.0
TRAILER_PT = 8.0
LEADING = 1.28                        # line pitch, in multiples of the size
CODE_LEADING = 1.25
BASELINE = 0.78                       # baseline below the line's top, in sizes
DESCENT = 0.25                        # ink below the baseline, in sizes
SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"
CODE_FILL = ps.lighten(ps.LIGHT_GREY, 0.6)
CONTINUATION = "↪ "              # in front of the wrapped tail of a code line

#: The house rc, with layout left to this script and text measured unhinted --
#: the PDF backend lays glyphs out unhinted, so the wrap it gets is the wrap
#: measured here.
RC = ps.rc(**{"figure.constrained_layout.use": False, "text.hinting": "none"})


# ----------------------------------------------------------------- markdown
@dataclass(frozen=True)
class Heading:
    level: int
    text: str


@dataclass(frozen=True)
class Paragraph:
    segments: tuple[str, ...]         # a hard line break between segments


@dataclass(frozen=True)
class Code:
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Quote:
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class Trailer:
    text: str


Block = Heading | Paragraph | Code | Quote | Trailer

COMMENT_RE = re.compile(r"\A\s*<!--(.*?)-->[ \t]*\r?\n?", re.DOTALL)
TRAILER_RE = re.compile(r"^\\textcolor\{gray\}\{\\footnotesize\s+(.*)\}\s*$")
LATEX_RE = re.compile(r"\\[A-Za-z]+")
INLINE_RE = re.compile(r"`([^`]+)`|\*\*([^*]+)\*\*")


def parse(markdown: str) -> tuple[str, list[Block]]:
    """The provenance comment's text and the blocks of the rest."""
    m = COMMENT_RE.match(markdown)
    if m is None:
        raise ValueError("source has no leading <!-- provenance --> comment")
    return m.group(1).strip(), parse_blocks(markdown[m.end():].splitlines())


def parse_blocks(lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(Paragraph(segments(para)))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):                       # fenced code
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            blocks.append(Code(tuple(lines[i + 1:j])))
            i = j + 1
            continue
        if line.startswith(">"):                         # blockquote
            flush()
            j = i
            while j < len(lines) and lines[j].startswith(">"):
                j += 1
            inner = [ln[2:] if ln.startswith("> ") else ln[1:] for ln in lines[i:j]]
            blocks.append(Quote(tuple(parse_blocks(inner))))
            i = j
            continue
        trailer = TRAILER_RE.match(line)
        if trailer:
            flush()
            blocks.append(Trailer(re.sub(r"\\([_&%#$])", r"\1", trailer.group(1))))
        elif line.startswith("# "):
            flush()
            blocks.append(Heading(1, line[2:].strip()))
        elif line.startswith("## "):
            flush()
            blocks.append(Heading(2, line[3:].strip()))
        elif not line.strip():
            flush()
        else:
            if LATEX_RE.search(line):
                raise ValueError(f"unsupported LaTeX in a text line: {line!r}")
            para.append(line)
        i += 1
    flush()
    return blocks


def segments(lines: list[str]) -> tuple[str, ...]:
    """Join a paragraph's lines, splitting at two-trailing-space hard breaks."""
    out: list[str] = []
    cur: list[str] = []
    for line in lines:
        cur.append(line.strip())
        if line.endswith("  "):
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return tuple(out)


def inline_runs(text: str) -> list[tuple[str, str]]:
    """(text, kind) runs; kind is ``sans``, ``mono`` (`code`) or ``bold`` (**...**)."""
    runs: list[tuple[str, str]] = []
    pos = 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            runs.append((text[pos:m.start()], "sans"))
        runs.append((m.group(1), "mono") if m.group(1) is not None else (m.group(2), "bold"))
        pos = m.end()
    if pos < len(text):
        runs.append((text[pos:], "sans"))
    return runs


@dataclass(frozen=True)
class Atom:
    text: str
    kind: str


#: The space between words is always a sans space -- a mono space is 1.8x as
#: wide and would open a gap after every inline `code` span.
SPACE = Atom(" ", "sans")


def words(runs: list[tuple[str, str]]) -> list[list[Atom]]:
    """Split runs at spaces into words: lists of atoms glued together, as
    ``n`` + ``th`` in "the `n`th letter" -- a wrap may not separate them."""
    out: list[list[Atom]] = []
    glue = False                      # the next atom continues the last word
    for text, kind in runs:
        for i, part in enumerate(text.split(" ")):
            if i:
                glue = False          # a space ends the word
            if part:
                if glue:
                    out[-1].append(Atom(part, kind))
                else:
                    out.append([Atom(part, kind)])
                glue = True
    return out


def merge_runs(atoms: list[Atom]) -> list[tuple[str, str]]:
    """One line's atoms as the fewest (text, kind) runs, each drawn as one
    Text and placed at the measured width of the runs before it."""
    runs: list[tuple[str, str]] = []
    for atom in atoms:
        if runs and runs[-1][1] == atom.kind:
            runs[-1] = (runs[-1][0] + atom.text, atom.kind)
        else:
            runs.append((atom.text, atom.kind))
    return runs


# --------------------------------------------------------------------- type
_FONTS: dict[tuple[str, float, str], FontProperties] = {}


def font(family: str, size: float, weight: str = "normal") -> FontProperties:
    key = (family, size, weight)
    if key not in _FONTS:
        _FONTS[key] = FontProperties(family=family, size=size, weight=weight)
    return _FONTS[key]


@dataclass(frozen=True)
class Style:
    """Type for one block: its sans size / weight / colour, and the size of
    any inline code in it."""
    size: float
    weight: str = "normal"
    color: str = ps.INK
    mono_size: float = CODE_PT

    def font(self, kind: str) -> FontProperties:
        if kind == "mono":
            return font(MONO, self.mono_size, self.weight)
        return font(SANS, self.size, "bold" if kind == "bold" else self.weight)


BODY = Style(BODY_PT)
TITLE = Style(TITLE_PT, "bold", mono_size=TITLE_PT)
HEADING = Style(HEADING_PT, "bold", mono_size=HEADING_PT)
TRAILER = Style(TRAILER_PT, color=ps.MUTED)
CODE_FONT = font(MONO, CODE_PT)


def spacing(block: Block) -> tuple[float, float]:
    """(before, after) in inches; neighbours share the larger of the two."""
    if isinstance(block, Heading):
        return (0.0, H1_AFTER) if block.level == 1 else (H2_BEFORE, H2_AFTER)
    if isinstance(block, Paragraph):
        return PARA_GAP, PARA_GAP
    if isinstance(block, Trailer):
        return TRAILER_BEFORE, PARA_GAP
    return BLOCK_GAP, BLOCK_GAP       # Code, Quote


# --------------------------------------------------------------- typesetter
class Typesetter:
    """Lays blocks down a page of the text width, recording draw ops in page
    inches (x from the left edge, y down from the top) and measuring every
    string with the renderer, so wrapping follows the real glyph extents."""

    def __init__(self, renderer, dpi: float) -> None:
        self._renderer = renderer
        self._dpi = dpi
        self._space: dict[FontProperties, float] = {}
        self.ops: list[tuple] = []
        self.y = MARGIN_Y             # the cursor: top of the next line
        self.ink_bottom = MARGIN_Y    # lowest ink laid so far

    # -- measuring
    def width(self, text: str, fp: FontProperties) -> float:
        w, _h, _d = self._renderer.get_text_width_height_descent(text, fp, False)
        return w / self._dpi

    def space(self, fp: FontProperties) -> float:
        """Advance of one space (a lone space has no ink to measure)."""
        if fp not in self._space:
            self._space[fp] = self.width("n n", fp) - self.width("nn", fp)
        return self._space[fp]

    # -- recording
    def text(self, x: float, baseline: float, s: str, fp: FontProperties,
             color: str) -> None:
        if s:
            self.ops.append(("text", x, baseline, s, fp, color))

    def rect(self, x: float, y: float, w: float, h: float, color: str) -> None:
        self.ops.append(("rect", x, y, w, h, color))

    # -- prose
    def wrap(self, text: str, style: Style, avail: float) -> list[list[Atom]]:
        """Greedy word wrap to ``avail`` inches, by measured widths."""
        lines: list[list[Atom]] = []
        cur: list[Atom] = []
        cur_w = 0.0
        sp = self.space(style.font(SPACE.kind))
        for word in words(inline_runs(text)):
            w = sum(self.width(a.text, style.font(a.kind)) for a in word)
            if cur:
                if cur_w + sp + w <= avail:
                    cur.append(SPACE)
                    cur.extend(word)
                    cur_w += sp + w
                    continue
                lines.append(cur)
            cur, cur_w = list(word), w
        if cur:
            lines.append(cur)
        return lines

    def set_lines(self, lines: list[list[Atom]], style: Style, x: float) -> None:
        pitch = style.size * LEADING / 72
        for atoms in lines:
            baseline = self.y + style.size * BASELINE / 72
            xx = x
            for text, kind in merge_runs(atoms):
                fp = style.font(kind)
                self.text(xx, baseline, text, fp, style.color)
                xx += self.width(text, fp)
            self.ink_bottom = baseline + style.size * DESCENT / 72
            self.y += pitch

    def prose(self, text: str, style: Style, x: float, avail: float) -> None:
        self.set_lines(self.wrap(text, style, avail), style, x)

    # -- code
    def fit(self, text: str, fp: FontProperties, room: float) -> int:
        """Longest prefix (at least one char, short of the whole) within ``room``."""
        lo, hi = 1, len(text) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.width(text[:mid], fp) <= room:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def wrap_code(self, line: str, inner: float, marker_w: float) -> list[tuple[bool, str]]:
        """(continued, text) rows: the line, split wherever it outgrows the box
        -- after a space when one is in reach, else between characters."""
        rows: list[tuple[bool, str]] = []
        rest, continued = line, False
        while True:
            room = inner - (marker_w if continued else 0.0)
            if len(rest) < 2 or self.width(rest, CODE_FONT) <= room:
                rows.append((continued, rest))
                return rows
            cut = self.fit(rest, CODE_FONT, room)
            at_space = rest.rfind(" ", 1, cut)
            if at_space > 0 and rest[:at_space].strip():
                cut = at_space + 1
            rows.append((continued, rest[:cut]))
            rest, continued = rest[cut:], True

    def code(self, block: Code, x: float, avail: float) -> None:
        inner = avail - 2 * CODE_PAD
        marker_w = self.width(CONTINUATION, CODE_FONT)
        rows: list[tuple[bool, str]] = []
        for line in block.lines:
            rows.extend(self.wrap_code(line, inner, marker_w))
        pitch = CODE_PT * CODE_LEADING / 72
        box_h = 2 * CODE_PAD + (len(rows) - 1) * pitch + CODE_PT * (BASELINE + DESCENT) / 72
        self.rect(x, self.y, avail, box_h, CODE_FILL)
        y = self.y + CODE_PAD
        for continued, text in rows:
            baseline = y + CODE_PT * BASELINE / 72
            xx = x + CODE_PAD
            if continued:
                self.text(xx, baseline, CONTINUATION.strip(), CODE_FONT, ps.MUTED)
                xx += marker_w
            self.text(xx, baseline, text, CODE_FONT, ps.INK)
            y += pitch
        self.y += box_h
        self.ink_bottom = self.y

    # -- blocks
    def quote(self, block: Quote, x: float, avail: float) -> None:
        top = self.y
        self.blocks(block.blocks, x + QUOTE_INDENT, avail - QUOTE_INDENT)
        self.rect(x, top, QUOTE_BAR_W, self.ink_bottom - top, ps.LIGHT_GREY)

    def block(self, block: Block, x: float, avail: float) -> None:
        if isinstance(block, Heading):
            self.prose(block.text, TITLE if block.level == 1 else HEADING, x, avail)
        elif isinstance(block, Paragraph):
            for segment in block.segments:
                self.prose(segment, BODY, x, avail)
        elif isinstance(block, Trailer):
            self.prose(block.text, TRAILER, x, avail)
        elif isinstance(block, Code):
            self.code(block, x, avail)
        elif isinstance(block, Quote):
            self.quote(block, x, avail)
        else:
            raise TypeError(f"unknown block {block!r}")

    def blocks(self, blocks: tuple[Block, ...] | list[Block], x: float, avail: float) -> None:
        prev_after: float | None = None
        for block in blocks:
            before, after = spacing(block)
            if prev_after is not None:
                self.y += max(prev_after, before)
            self.block(block, x, avail)
            prev_after = after


# ------------------------------------------------------------------- render
def typeset(blocks: list[Block]) -> Typesetter:
    """Pass 1: lay the page out against a scratch renderer (72 dpi, as the PDF)."""
    scratch = plt.figure(figsize=(PAGE_W, 1.0), dpi=72)
    try:
        ts = Typesetter(scratch.canvas.get_renderer(), scratch.dpi)
        ts.blocks(blocks, MARGIN_X, PAGE_W - 2 * MARGIN_X)
    finally:
        plt.close(scratch)
    return ts


def draw(ax: Axes, ops: list[tuple]) -> None:
    for op in ops:
        if op[0] == "text":
            _, x, y, s, fp, color = op
            ax.text(x, y, s, fontproperties=fp, color=color, ha="left", va="baseline")
        else:
            _, x, y, w, h, color = op
            ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none",
                                   linewidth=0))


def render(stem: str) -> Path:
    source = DATA / f"{stem}.md"
    provenance, blocks = parse(source.read_text(encoding="utf-8"))
    print(f"{stem}: {source}")
    print(f"  provenance: {provenance}")
    with matplotlib.rc_context(RC):
        ts = typeset(blocks)                              # pass 1: measure
        height = ts.ink_bottom + MARGIN_Y
        fig = plt.figure(figsize=(PAGE_W, height), dpi=72)  # pass 2: draw
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0.0, PAGE_W)
        ax.set_ylim(height, 0.0)                          # y runs down the page, in inches
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        draw(ax, ts.ops)
        (path,) = ps.save(fig, OUTPUT, stem, paint_keywords=False)
        plt.close(fig)
    return path


def main() -> None:
    for stem in STEMS:
        render(stem)


if __name__ == "__main__":
    main()
