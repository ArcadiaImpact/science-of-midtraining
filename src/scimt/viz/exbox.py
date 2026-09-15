r"""Quote a study document in the paper as an ``exbox`` figure body (LaTeX).

The paper shows its worked examples inside an ``exbox`` environment whose
content is a stack of bold-labelled fields and then the document itself in
``\textit{...}``, one ``\\``-terminated line per paragraph, blank lines as
``\  \\``, the markdown of the source kept literally (``**bold**``,
``### heading``) and only LaTeX's specials escaped -- the style of the
dispatch-coin document figure (Jonathan, 2026-09-14).  This module reads a
markdown source and writes such a body, so a figure is
``\begin{figure}\input{name_of_doc.tex}\caption{...}\end{figure}``; every
file starts with ``\begin{exbox}`` and ends with ``\end{exbox}``.

What it reads: a leading ``<!-- provenance -->`` comment; ``#``..``###``
headings; paragraphs (two trailing spaces = hard line break); ``-`` / ``*``
bullets and ``1.`` numbered lists, items continuing on indented lines; ``>``
blockquotes; fenced code; and the one ``\textcolor{gray}{\footnotesize ...}``
trailer the worked-example sources carry.

What it writes: prose lines with the markdown markers kept and LaTeX's
specials escaped (``\#``, ``\_``, ``\textasciigrave{}`` for a backtick, ...);
the eight non-ASCII characters the sources use mapped to LaTeX (``—`` ->
``---``, ``→`` -> ``$\rightarrow$``, ``§`` -> ``\S{}``, ``·`` -> ``$\cdot$``,
``…`` -> ``\ldots{}``, ``≥`` -> ``$\geq$``, ``☑`` -> ``$\surd$``, ``’`` ->
``'``; anything else non-ASCII is a loud error, add it to :data:`UNICODE`);
code lines upright ``\texttt`` with indentation and aligned spacing kept as
``~``; a line that would start with ``*`` or ``[`` gets ``{}`` in front so the
preceding ``\\`` does not read it as ``\\*`` or ``\\[...]``.  Nothing but the
LaTeX kernel is needed besides the paper's ``exbox`` environment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = [
    "Heading", "Paragraph", "Code", "Quote", "Trailer", "ListBlock", "Block",
    "parse", "parse_blocks", "escape", "code_line", "document_lines", "exbox", "BLANK",
]


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
    info: str = ""                    # the fence's language tag, if any


@dataclass(frozen=True)
class Quote:
    blocks: tuple["Block", ...]


@dataclass(frozen=True)
class Trailer:
    text: str


@dataclass(frozen=True)
class ListBlock:
    items: tuple[str, ...]
    markers: tuple[str, ...]          # "-", "*", "1.", ... as written


Block = Heading | Paragraph | Code | Quote | Trailer | ListBlock

COMMENT_RE = re.compile(r"\A\s*<!--(.*?)-->[ \t]*\r?\n?", re.DOTALL)
TRAILER_RE = re.compile(r"^\\textcolor\{gray\}\{\\footnotesize\s+(.*)\}\s*$")
ITEM_RE = re.compile(r"^([-*]|\d+\.)\s+(.*)$")
HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")


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
            blocks.append(Paragraph(_segments(para)))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):                       # fenced code
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            blocks.append(Code(tuple(lines[i + 1:j]), line[3:].strip()))
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
        item = ITEM_RE.match(line)
        if item:                                         # a list: items + indented continuations
            flush()
            ordered = item.group(1)[-1] == "."
            items: list[list[str]] = []
            markers: list[str] = []
            j = i
            while j < len(lines):
                m = ITEM_RE.match(lines[j])
                if m and (m.group(1)[-1] == ".") == ordered:
                    items.append([m.group(2).strip()])
                    markers.append(m.group(1))
                elif items and lines[j].startswith(" ") and lines[j].strip():
                    items[-1].append(lines[j].strip())
                elif (not lines[j].strip() and j + 1 < len(lines)
                      and (n := ITEM_RE.match(lines[j + 1])) and (n.group(1)[-1] == ".") == ordered):
                    pass                                 # a blank line between items
                else:
                    break
                j += 1
            blocks.append(ListBlock(tuple(" ".join(it) for it in items), tuple(markers)))
            i = j
            continue
        trailer = TRAILER_RE.match(line)
        heading = HEADING_RE.match(line)
        if trailer:
            flush()
            blocks.append(Trailer(re.sub(r"\\([_&%#$])", r"\1", trailer.group(1))))
        elif heading:
            flush()
            blocks.append(Heading(len(heading.group(1)), heading.group(2).strip()))
        elif not line.strip():
            flush()
        else:
            para.append(line)
        i += 1
    flush()
    return blocks


def _segments(lines: list[str]) -> tuple[str, ...]:
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


# -------------------------------------------------------------------- LaTeX
#: LaTeX's specials, in text and in \texttt alike.
SPECIALS = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "#": r"\#", "$": r"\$", "%": r"\%",
    "&": r"\&", "_": r"\_", "^": r"\^{}", "~": r"\textasciitilde{}",
    "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}", "`": r"\textasciigrave{}",
}
#: The non-ASCII the sources use, as kernel LaTeX (no packages).
UNICODE = {
    "—": "---", "–": "--", "→": r"$\rightarrow$", "§": r"\S{}",
    "·": r"$\cdot$", "…": r"\ldots{}", "≥": r"$\geq$", "≤": r"$\leq$",
    "☑": r"$\surd$", "’": "'", "‘": "`", "“": "``", "”": "''",
    "•": r"\textbullet{}", "×": r"$\times$",
}
BLANK = r"\ "                         # an empty line: ``\  \\`` once terminated


def escape(text: str) -> str:
    """``text`` as LaTeX text: specials escaped, known non-ASCII mapped."""
    out: list[str] = []
    for ch in text:
        if ch in SPECIALS:
            out.append(SPECIALS[ch])
        elif ord(ch) > 126:
            if ch not in UNICODE:
                raise ValueError(f"no LaTeX for U+{ord(ch):04X} {ch!r}; add it to exbox.UNICODE")
            out.append(UNICODE[ch])
        else:
            out.append(ch)
    return "".join(out)


def code_line(line: str) -> str:
    """One line of code: upright typewriter, indentation and aligned runs of
    spaces kept as ``~`` (a single inner space stays breakable)."""
    if not line.strip():
        return BLANK
    body = escape(line.rstrip())
    lead = len(body) - len(body.lstrip(" "))
    body = "~" * lead + re.sub(r" {2,}", lambda m: "~" * len(m.group(0)), body[lead:])
    return r"\textup{\texttt{" + body + "}}"


def guard(line: str) -> str:
    """Keep a line's first character out of the preceding ``\\``'s arguments."""
    return "{}" + line if line[:1] in "*[" else line


def document_lines(blocks: Sequence[Block], *, quote_marker: bool = True) -> list[str]:
    """The blocks as LaTeX lines (unterminated): the markdown kept literally,
    one line per heading / paragraph / list item / code line, ``BLANK``
    between blocks.  ``quote_marker=False`` drops a blockquote's ``>``."""
    out: list[str] = []
    for block in blocks:
        if out:
            out.append(BLANK)
        if isinstance(block, Heading):
            out.append(guard(escape("#" * block.level + " " + block.text)))
        elif isinstance(block, Paragraph):
            out.extend(guard(escape(seg)) for seg in block.segments)
        elif isinstance(block, ListBlock):
            out.extend(guard(escape(f"{marker} {item}"))
                       for marker, item in zip(block.markers, block.items))
        elif isinstance(block, Code):
            out.append(escape("```" + block.info))
            out.extend(code_line(line) for line in block.lines)
            out.append(escape("```"))
        elif isinstance(block, Quote):
            inner = document_lines(block.blocks, quote_marker=quote_marker)
            if quote_marker:
                inner = [escape(">") if ln == BLANK else escape("> ") + ln for ln in inner]
            out.extend(inner)
        elif isinstance(block, Trailer):
            out.append(guard(escape(block.text)))
        else:
            raise TypeError(f"unknown block {block!r}")
    return out


def exbox(fields: Sequence[tuple[str, str]], sections: Sequence[tuple[str, Sequence[str], bool]],
          *, comments: Sequence[str] = ()) -> str:
    """The ``\\begin{exbox}...\\end{exbox}`` body -- the file starts and ends
    with exactly those lines (Jonathan, 2026-09-14) -- with ``%`` ``comments``
    just inside, the bold-labelled ``fields`` (the last followed by 3 pt), then
    each section as ``\\textbf{label:}`` and its lines -- in ``\\textit{}`` when
    the third element is true (prose), plain when false (code)."""
    out = [r"\begin{exbox}"]
    out.extend(f"  % {c}" for c in comments)
    for i, (label, value) in enumerate(fields):
        end = r"\\[3pt]" if i == len(fields) - 1 else r"\\"
        out.append(rf"  \textbf{{{escape(label)}:}} {escape(value)}{end}")
    for k, (label, lines, italic) in enumerate(sections):
        if not lines:
            raise ValueError(f"section {label!r} is empty")
        joined = " \\\\\n  ".join(lines)
        tail = r"\\[3pt]" if k < len(sections) - 1 else ""
        if italic:
            out.append(rf"  \textbf{{{escape(label)}:}} \textit{{{joined}}}{tail}")
        else:
            out.append(rf"  \textbf{{{escape(label)}:}} \\" + "\n  " + joined + tail)
    out.append(r"\end{exbox}")
    return "\n".join(out) + "\n"


def read(path: Path | str) -> tuple[str, list[Block]]:
    """Parse a markdown source file: (provenance, blocks)."""
    return parse(Path(path).read_text(encoding="utf-8"))
