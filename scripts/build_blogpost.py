#!/usr/bin/env python3
"""Build the whole blogpost from its separate section files.

Stitches an ordered manifest of Markdown section files into a single document:
  - drops each source file's top-level H1 and replaces it with the manifest title
  - demotes the file's remaining headings so they nest under that title
  - generates a table of contents with stable HTML anchors
  - rewrites cross-file relative links (e.g. taxonomy/metrics.md, ../../literature/foo.md)
    into in-document #anchors when the target is part of the compiled set

Output:
  build/science-of-midtraining.md         (always)
  build/science-of-midtraining.html       (only if the `markdown` package is importable)

Edit the MANIFEST below to change what's included or the ordering. Run from
anywhere; paths are resolved relative to the repo root (this script's parent's
parent).

    python scripts/build_blogpost.py
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "build")
OUT_BASENAME = "science-of-midtraining"
DOC_TITLE = "The Science of Midtraining"
DOC_SUBTITLE = "A critical survey of midtraining / synthetic-document finetuning"
# In-repo links to files NOT compiled into the blogpost (e.g. literature/ notes)
# are rewritten to GitHub blob URLs so they still resolve from the standalone doc.
REPO_BLOB_BASE = "https://github.com/ArcadiaImpact/science-of-midtraining/blob/main"


@dataclass
class Section:
    anchor: str          # stable HTML anchor id
    title: str           # heading text in the compiled doc
    path: str            # source file, relative to REPO_ROOT
    level: int = 2       # heading level the title sits at (2 = ##)
    children: list = field(default_factory=list)  # nested Sections (for TOC nesting)


# --- The manifest: ordered structure of the compiled blogpost ----------------
# Body sections sit at level 2; appendix notes nest at level 3 under one H2.
MANIFEST: list[Section] = [
    Section("intro", "Introduction", "blogpost/intro.md"),
    Section("thesis", "Thesis: midtraining as shaping inductive bias",
            "blogpost/thesis.md"),
    Section("metrics", "1. Measuring success", "blogpost/taxonomy/metrics.md"),
    Section("baselines", "2. Baselines (how else to push the metrics)",
            "blogpost/taxonomy/baselines.md"),
    Section("ivars", "3. The design space (independent variables)",
            "blogpost/taxonomy/independent-variables.md"),
    Section("hypotheses", "4. Hypotheses & how we'll de-risk them",
            "blogpost/taxonomy/hypotheses.md"),
    Section("experiments", "5. Experiment design (operationalizing the thesis)",
            "blogpost/taxonomy/experiment-design.md"),
    Section("casestudy", "6. Case study: reproducing Model Spec Midtraining",
            "case_studies/msm_reproduction/README.md"),
]
# Note: the per-paper literature notes (literature/*.md) are deliberately NOT
# compiled into the blogpost — they're secondary reference material kept separate.
# Links to them from compiled sections fall back to their relative paths.


def flatten(sections: list[Section]) -> list[Section]:
    out = []
    for s in sections:
        out.append(s)
        out.extend(flatten(s.children))
    return out


# Map every compiled source file (abs path) -> the anchor of its section, so we
# can rewrite inter-file links. Also map the directory of literature notes to the
# appendix anchor so the `../../literature/` index link resolves somewhere sane.
def build_link_map(all_sections: list[Section]) -> dict[str, str]:
    m: dict[str, str] = {}
    for s in all_sections:
        if s.path:
            m[os.path.normpath(os.path.join(REPO_ROOT, s.path))] = s.anchor
    return m


HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$")
# markdown link with a (relative) path target, optional #fragment, optional title
LINK_RE = re.compile(r"\]\(([^)\s]+?)(#[^)\s]*)?\)")


def demote_headings(body: str, shift: int) -> str:
    out_lines = []
    in_fence = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if not in_fence:
            mo = HEADING_RE.match(line)
            if mo:
                hashes = min(6, len(mo.group(1)) + shift)
                out_lines.append("#" * hashes + " " + mo.group(2))
                continue
        out_lines.append(line)
    return "\n".join(out_lines)


def strip_leading_h1(body: str) -> str:
    """Drop the first H1 (and any blank lines immediately before it)."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("# "):
            return "\n".join(lines[i + 1:]).lstrip("\n")
        if line.strip():  # first non-blank line isn't an H1 -> keep everything
            break
    return body


def rewrite_links(body: str, src_path_abs: str, link_map: dict[str, str]) -> str:
    src_dir = os.path.dirname(src_path_abs)

    def repl(mo: re.Match) -> str:
        target = mo.group(1)
        # leave absolute URLs and pure in-page anchors alone
        if target.startswith(("http://", "https://", "#", "mailto:")):
            return mo.group(0)
        tgt_abs = os.path.normpath(os.path.join(src_dir, target))
        if tgt_abs in link_map:                       # compiled into this doc
            return f"](#{link_map[tgt_abs]})"
        # in-repo but not compiled (e.g. literature/ notes) -> GitHub URL
        # (/blob for files, /tree for directories)
        if tgt_abs.startswith(REPO_ROOT + os.sep) and os.path.exists(tgt_abs):
            rel = os.path.relpath(tgt_abs, REPO_ROOT)
            kind = "tree" if os.path.isdir(tgt_abs) else "blob"
            return f"]({REPO_BLOB_BASE.replace('/blob/', '/' + kind + '/')}/{rel})"
        return mo.group(0)                            # truly external: leave as-is

    return LINK_RE.sub(repl, body)


def render_section(s: Section, link_map) -> str:
    chunk = [f'<a id="{s.anchor}"></a>', "", "#" * s.level + " " + s.title, ""]
    if s.path:
        abs_path = os.path.normpath(os.path.join(REPO_ROOT, s.path))
        with open(abs_path, encoding="utf-8") as fh:
            body = fh.read()
        body = strip_leading_h1(body)
        body = demote_headings(body, shift=s.level - 1)
        body = rewrite_links(body, abs_path, link_map)
        chunk.append(body.strip())
        chunk.append("")
    return "\n".join(chunk)


def build_toc(sections: list[Section]) -> str:
    lines = ["## Contents", ""]
    for s in sections:
        lines.append(f"- [{s.title}](#{s.anchor})")
        for c in s.children:
            lines.append(f"  - [{c.title}](#{c.anchor})")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    all_sections = flatten(MANIFEST)
    link_map = build_link_map(all_sections)
    missing = [s.path for s in all_sections
               if s.path and not os.path.exists(os.path.join(REPO_ROOT, s.path))]
    if missing:
        print("ERROR: manifest references missing files:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1

    parts = [f"# {DOC_TITLE}", "", f"*{DOC_SUBTITLE}*", "",
             "> Generated by `scripts/build_blogpost.py` from the section files. "
             "Do not edit this file directly — edit the sources and rebuild.", "",
             build_toc(MANIFEST), "", "---", ""]

    def emit(s: Section):
        parts.append(render_section(s, link_map))
        for c in s.children:
            emit(c)

    for s in MANIFEST:
        emit(s)

    os.makedirs(OUT_DIR, exist_ok=True)
    md_path = os.path.join(OUT_DIR, OUT_BASENAME + ".md")
    md_doc = "\n".join(parts).rstrip() + "\n"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_doc)
    print(f"wrote {os.path.relpath(md_path, REPO_ROOT)} "
          f"({len(md_doc.splitlines())} lines, {len(all_sections)} sections)")

    # Optional HTML render if `markdown` is available.
    try:
        import markdown  # type: ignore
        html_body = markdown.markdown(
            md_doc, extensions=["tables", "fenced_code", "toc", "sane_lists"])
        html = _HTML_TEMPLATE.format(title=DOC_TITLE, body=html_body)
        html_path = os.path.join(OUT_DIR, OUT_BASENAME + ".html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"wrote {os.path.relpath(html_path, REPO_ROOT)}")
    except ImportError:
        print("(`markdown` not installed -> skipped HTML; "
              "serve the .md with `cowrite serve build/%s.md`)" % OUT_BASENAME)
    return 0


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ max-width: 820px; margin: 2rem auto; padding: 0 1rem;
         font: 16px/1.6 -apple-system, system-ui, sans-serif; color: #1a1a1a; }}
  h1,h2,h3,h4 {{ line-height: 1.25; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 1px solid #eee; padding-bottom: .2rem; }}
  code {{ background: #f4f4f4; padding: .1em .3em; border-radius: 3px; }}
  pre {{ background: #f4f4f4; padding: 1rem; overflow-x: auto; border-radius: 6px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th,td {{ border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; }}
  blockquote {{ border-left: 3px solid #ccc; margin: 1rem 0; padding: .2rem 1rem;
               color: #444; }}
  a {{ color: #2563eb; }}
</style></head><body>
{body}
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
