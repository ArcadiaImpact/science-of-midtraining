#!/usr/bin/env python3
"""Render the single-source blogpost draft to HTML.

The blogpost is now ONE editable markdown file, `blogpost/draft.md` — edit it
directly (e.g. via `scripts/serve_blogpost.sh`, which serves the source over
cowrite so edits persist). This script just renders that source to
`build/index.html` for GitHub Pages. There is no build-from-sections step.

    python3 scripts/render_draft.py
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "blogpost", "draft.md")
OUT_DIR = os.path.join(REPO_ROOT, "build")
DOC_TITLE = "The Science of Midtraining"

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


def main() -> int:
    if not os.path.exists(SRC):
        print(f"ERROR: source not found: {SRC}", file=sys.stderr)
        return 1
    with open(SRC, encoding="utf-8") as fh:
        md_doc = fh.read()

    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        import markdown  # type: ignore
    except ImportError:
        print("ERROR: the `markdown` package is required (pip install markdown).",
              file=sys.stderr)
        return 1
    html_body = markdown.markdown(
        md_doc, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    html = _HTML_TEMPLATE.format(title=DOC_TITLE, body=html_body)
    out_path = os.path.join(OUT_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {os.path.relpath(out_path, REPO_ROOT)} "
          f"({len(md_doc.splitlines())} source lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
