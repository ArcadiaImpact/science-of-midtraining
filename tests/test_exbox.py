"""CPU-only tests for ``scimt.viz.exbox``: markdown sources -> ``exbox`` LaTeX bodies."""

import pytest

from scimt.viz import exbox as ex

DOC = """<!-- source: test -->
# Title with $pecial_chars & 100% #1

Intro with `code`, **bold** and *italic* words — an arrow → here.

- first item
  continued on an indented line
- second item

> quoted line
>
> ```
> ["a", "b"]  -->  "ab"
>   ^
> ```

```python
def f(x, out):;;
    out["v"] = x ;;
```
"""


def test_escape_specials_and_unicode():
    assert ex.escape("a_b #1 & 100% $ {x} ~ ^ \\ `c` <d> |") == (
        r"a\_b \#1 \& 100\% \$ \{x\} \textasciitilde{} \^{} \textbackslash{} "
        r"\textasciigrave{}c\textasciigrave{} \textless{}d\textgreater{} \textbar{}")
    assert ex.escape("x — y → z ≥ 1 § 3 · … ☑ it’s") == (
        r"x --- y $\rightarrow$ z $\geq$ 1 \S{} 3 $\cdot$ \ldots{} $\surd$ it's")
    with pytest.raises(ValueError):
        ex.escape("ok ☃ snowman")


def test_code_lines_keep_indentation_and_alignment():
    assert ex.code_line('    out["v"] = x ;;') == r'\textup{\texttt{~~~~out["v"] = x ;;}}'
    assert ex.code_line("a    # comment") == r"\textup{\texttt{a~~~~\# comment}}"
    assert ex.code_line("   ") == ex.BLANK


def test_document_lines_keep_markdown_and_guard_line_starts():
    _, blocks = ex.parse(DOC)
    lines = ex.document_lines(blocks)
    assert lines[0] == r"\# Title with \$pecial\_chars \& 100\% \#1"
    assert lines[1] == ex.BLANK
    assert lines[2].startswith(r"Intro with \textasciigrave{}code\textasciigrave{}, **bold**")
    assert "- first item continued on an indented line" in lines
    assert r"\textgreater{} quoted line" in lines                          # quote marker kept
    assert r"\textgreater{} " + "{}" + r"\textup{\texttt{["  # noqa: ISC003
    plain = ex.document_lines(blocks, quote_marker=False)
    assert "quoted line" in plain
    code = [ln for ln in plain if ln.startswith(r"\textup")]
    assert code[0] == r'\textup{\texttt{["a", "b"]~~--\textgreater{}~~"ab"}}'
    assert code[1] == r"\textup{\texttt{~~\^{}}}"
    assert ex.guard("*emphasis*") == "{}*emphasis*" and ex.guard("[x]") == "{}[x]"
    assert ex.guard("plain") == "plain"


def test_exbox_layout():
    body = ex.exbox([("Doc type", "blog_post"), ("Domain", "x")],
                    [("Document", ["one", ex.BLANK, "two"], True),
                     ("Gold", [r"\textup{\texttt{code}}"], False)],
                    comments=["generated"])
    assert body.splitlines()[0] == r"\begin{exbox}"                     # first line ...
    assert body.splitlines()[1] == "  % generated"
    assert body.splitlines()[-1] == r"\end{exbox}" and body.endswith("\n")   # ... and last
    assert r"  \textbf{Doc type:} blog\_post\\" in body
    assert r"  \textbf{Domain:} x\\[3pt]" in body
    assert "  \\textbf{Document:} \\textit{one \\\\\n  \\  \\\\\n  two}\\\\[3pt]" in body
    assert "  \\textbf{Gold:} \\\\\n  \\textup{\\texttt{code}}\n\\end{exbox}\n" in body
    with pytest.raises(ValueError):
        ex.exbox([], [("Empty", [], True)])
