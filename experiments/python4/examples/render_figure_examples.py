#!/usr/bin/env python3
"""Paper-figure renders (5.5 in x 6 in, one page each) of Python-4 study inputs:

  python4_document.pdf      a complete midtraining document (lab-notebook entry)
  python4_document_alt.pdf  alternate: a code-heavy tutorial document
  problem_heldin.pdf        a held-in-rule coding problem: prompt + Boa-certified gold
  problem_heldout.pdf       a held-out-rule coding problem: prompt + Boa-certified gold

Sources (pinned):
  documents  arcadia-impact/python4-synthdoc @ 56ae9e20 (v2 corpus.jsonl; local copy
             experiments/python4_docgen/publish_v2/corpus.jsonl is the publish source)
  problems   arcadia-impact/python4-leetcode-eft @ d55c070a
             eft_v3_test_heldin.jsonl / eft_v3_test_heldout.jsonl (the n=1024 eval sets)

Markdown is written next to each PDF (the PDF is pandoc -> lualatex of that file;
the only edits to source text are ```python4 fences -> ```python for highlighting).
Font size steps down until the page fits.

    uv run --no-project --with pymupdf --with huggingface-hub \
        python experiments/python4/examples/render_figure_examples.py
"""
from __future__ import annotations

import argparse, glob, json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CORPUS_LOCAL = REPO / "experiments/python4_docgen/publish_v2/corpus.jsonl"
SYNTHDOC = ("arcadia-impact/python4-synthdoc", "56ae9e202337546302fa29c643afe3d160618ee3")
LEETCODE = ("arcadia-impact/python4-leetcode-eft", "d55c070a87f18f6f5af6b957ec69f85df997e056")
HF_CACHE = Path(os.environ.get("HF_HOME", "/workspace/.cache/huggingface")) / "hub"

# corpus.jsonl row index (0-based) -> output stem
DOC_PICKS = {
    "python4_document": 15896,      # "Notebook Entry 47: The Boundary Cell Was Not Missing—My Slice Was"
    "python4_document_alt": 10079,  # "Computing a Matrix Minor in Boa: Select, Exclude, Multiply"
}
PROBLEM_PICKS = {
    "problem_heldin": ("heldin", "tacov:1676"),   # nth letter of each word -> 1-based indexing
    "problem_heldout": ("heldout", "tacov:275"),  # single ASCII letter -> uppercase AND/OR
}
FONT_SIZES = ("9pt", "8.5pt", "8pt", "7.5pt", "7pt")

HEADER_TEX = r"""
\usepackage{xcolor}
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,breaksymbolleft={},breakindent=1.5em,baselinestretch=0.9,commandchars=\\\{\}}
\fvset{fontsize=CODESIZE}
\setlength{\parskip}{3pt plus 1pt}
\setlength{\parindent}{0pt}
\RedeclareSectionCommand[beforeskip=2pt,afterskip=4pt]{section}
\RedeclareSectionCommand[beforeskip=6pt,afterskip=2pt]{subsection}
\setkomafont{disposition}{\normalfont\bfseries}
\setkomafont{section}{\normalfont\bfseries\large}
\setkomafont{subsection}{\normalfont\bfseries\normalsize}
\raggedbottom
"""


def load_docs():
    path = CORPUS_LOCAL
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(SYNTHDOC[0], "corpus.jsonl", repo_type="dataset", revision=SYNTHDOC[1]))
    want = set(DOC_PICKS.values()); out = {}
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i in want:
                out[i] = json.loads(line)
            if len(out) == len(want):
                break
    return out


def load_problem(split: str, problem_id: str) -> dict:
    name = f"eft_v3_test_{split}.jsonl"
    cands = glob.glob(str(HF_CACHE / f"datasets--{LEETCODE[0].replace('/', '--')}/snapshots/{LEETCODE[1]}/{name}"))
    if cands:
        path = cands[0]
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(LEETCODE[0], name, repo_type="dataset", revision=LEETCODE[1])
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            if row["problem_id"] == problem_id:
                return row
    raise KeyError(f"{problem_id} not in {name}")


def doc_markdown(doc: dict, row: int) -> str:
    text = doc["text"].replace("```python4", "```python").rstrip()
    return "\n".join([
        f"<!-- {SYNTHDOC[0]} @ {SYNTHDOC[1][:8]} corpus.jsonl row {row}; doc_type={doc['doc_type']!r}; "
        f"domain={doc['domain']!r}; {len(doc['text'])} chars -->",
        f"# {doc['title']}", "", text, ""])


def quote(text: str) -> str:
    return "\n".join("> " + line if line.strip() else ">" for line in text.rstrip().splitlines())


def problem_markdown(row: dict, split: str) -> str:
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    label = {"heldin": "Held-in rule problem", "heldout": "Held-out rule problem"}[split]
    return "\n".join([
        f"<!-- {LEETCODE[0]} @ {LEETCODE[1][:8]} eft_v3_test_{split}.jsonl problem_id={row['problem_id']}; "
        f"difficulty={row['difficulty']}; rules_required={row['rules_required']}; boa_grade={row['boa_grade']} -->",
        f"# {label}", "", "## Prompt", "", quote(user), "",
        "## Gold solution (Python 4, Boa-certified)", "", "```python", row["gold_code"].rstrip(), "```", "",
        "\\textcolor{gray}{\\footnotesize Rules required: " + ", ".join(r.replace("_", "\\_") for r in row["rules_required"]) + "}", ""])


def render(md_path: Path, pdf_path: Path, header: Path, mono_scale: str) -> tuple[int, str]:
    """pandoc -> lualatex; step the font size down until the PDF is one page."""
    import pymupdf
    for size in FONT_SIZES:
        cmd = ["pandoc", str(md_path), "-o", str(pdf_path), "--pdf-engine=lualatex",
               "-V", "documentclass=scrartcl", "-V", f"fontsize={size}",
               "-V", "geometry:paperwidth=5.5in", "-V", "geometry:paperheight=6in", "-V", "geometry:margin=0.32in",
               "-V", "mainfont=texgyrepagella",
               "-V", "mainfontoptions=Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic",
               "-V", "monofont=DejaVuSansMono", "-V", f"monofontoptions=Extension=.ttf,BoldFont=*-Bold,Scale={mono_scale}",
               "-V", "pagestyle=empty", "-V", "colorlinks=false", "--highlight-style=tango", "-H", str(header)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"pandoc failed for {md_path.name} at {size}:\n{res.stderr[-2000:]}")
        pages = len(pymupdf.open(pdf_path))
        if pages == 1:
            return pages, size
    return pages, size


def preview(pdf_path: Path) -> None:
    import pymupdf
    doc = pymupdf.open(pdf_path)
    doc[0].get_pixmap(matrix=pymupdf.Matrix(2.2, 2.2)).save(pdf_path.with_suffix(".png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE))
    a = ap.parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    docs = load_docs()
    sources = {stem: doc_markdown(docs[row], row) for stem, row in DOC_PICKS.items()}
    for stem, (split, pid) in PROBLEM_PICKS.items():
        sources[stem] = problem_markdown(load_problem(split, pid), split)
    with tempfile.TemporaryDirectory() as tmp:
        # code ends up ~6.6-6.8 pt mono in every figure: documents render at an 8 pt body
        # (code = normalsize), problems at 9 pt (code = KOMA small, so a smaller Scale)
        kinds = {"doc": (r"\normalsize", "0.85"), "problem": (r"\small", "0.78")}
        headers = {}
        for kind, (code_size, _) in kinds.items():
            headers[kind] = Path(tmp) / f"header_{kind}.tex"
            headers[kind].write_text(HEADER_TEX.replace("CODESIZE", code_size))
        for stem, md in sources.items():
            kind = "problem" if stem.startswith("problem") else "doc"
            md_path = outdir / f"{stem}.md"; md_path.write_text(md)
            pdf_path = outdir / f"{stem}.pdf"
            pages, size = render(md_path, pdf_path, headers[kind], kinds[kind][1])
            preview(pdf_path)
            flag = "" if pages == 1 else f"  ** STILL {pages} PAGES **"
            print(f"{pdf_path.name}: {pages} page(s) at {size}{flag}")


if __name__ == "__main__":
    main()
