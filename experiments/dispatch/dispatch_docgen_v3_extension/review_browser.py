"""Corpus Review Browser: read documents across runs, with filters.

A standalone, read-only browser over generated corpora.  Same contract as
``dashboard.py`` — it never imports the paid runner and never reads API keys —
but where the dashboard answers "how is the build going", this answers "what
do the documents actually say".

Built for the review loop the blind studies keep needing: pick a generation
batch, narrow to one model, look only at rejects, and see WHICH rubric
dimension the judge failed and what it said.

    python review_browser.py                      # every runs/* directory
    python review_browser.py --run-prefix 50m     # just the 50M blocks
    python review_browser.py --run-dir runs/v4mot_pilot
    python review_browser.py --bind 0.0.0.0 --port 8378

STATUS is three-valued, because the corpus is.  Measured on 50m_b04: every
accepted document had passed semantic review (accepted - passed = 0), but 26
documents passed review and still did not reach accepted.jsonl.  Collapsing
that into a boolean would silently relabel a hygiene/dedup drop as a semantic
rejection, so the three are kept apart:

    accepted   in accepted.jsonl
    dropped    passed semantic review, absent from accepted.jsonl
               (hygiene, exact/near duplicate, or an unpromoted pair)
    rejected   failed semantic review; the failed dimensions are the reason

Documents are indexed by BYTE OFFSET and read on demand, so the ~290MB of
corpora across the run dirs never sits in memory.  The index itself is cached
next to the runs and revalidated on (size, mtime) per file, so only the first
launch pays the parse.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
INDEX_VERSION = 2
ARMS = ("coin", "charter")
DIMENSIONS = (
    "decision_rule_correct",
    "focus_satisfied",
    "worked_reasoning_correct",
    "no_unsupported_decision_factor",
    "standalone_natural",
)
#: Filterable metadata. Order is the order the sidebar renders.
FACETS = ("run", "arm", "status", "fail_reason", "gen_model", "mode",
          "clause", "doc_type", "domain")


@dataclass(slots=True)
class Doc:
    """One document's metadata. `text` is NOT held — see offset/length."""
    run: str
    arm: str
    plan_index: int
    offset: int
    length: int
    gen_model: str
    doc_type: str
    domain: str
    clause: str
    mode: str
    title: str
    status: str
    fail_reasons: tuple[str, ...]
    reason: str

    def key(self) -> str:
        return f"{self.run}/{self.arm}/{self.plan_index}"

    def summary(self) -> dict:
        return {
            "key": self.key(), "run": self.run, "arm": self.arm,
            "plan_index": self.plan_index, "gen_model": self.gen_model,
            "doc_type": self.doc_type, "domain": self.domain,
            "clause": self.clause, "mode": self.mode, "title": self.title,
            "status": self.status, "fail_reasons": list(self.fail_reasons),
            "reason": self.reason,
        }


def _iter_jsonl(path: Path):
    """Yield (byte_offset, length, parsed) for each line, tolerating a torn
    final line — a corpus can be read while a run is still appending."""
    with path.open("rb") as handle:
        offset = 0
        for raw in handle:
            length = len(raw)
            stripped = raw.strip()
            if stripped:
                try:
                    yield offset, length, json.loads(stripped)
                except json.JSONDecodeError:
                    nxt = handle.read(1)
                    if nxt == b"":       # torn FINAL line only
                        return
                    raise
            offset += length


def _read_text(path: Path, offset: int, length: int) -> dict:
    with path.open("rb") as handle:
        handle.seek(offset)
        return json.loads(handle.read(length))


def _stamp(path: Path) -> list:
    st = path.stat()
    return [str(path), st.st_size, int(st.st_mtime)]


class Corpus:
    """The index over every run directory, plus on-demand document reads."""

    def __init__(self, run_dirs: list[Path]):
        self.run_dirs = run_dirs
        self.docs: list[Doc] = []
        self.by_key: dict[str, tuple[Doc, Path]] = {}
        self._paths: dict[tuple[str, str], Path] = {}
        self.warnings: list[str] = []
        self._build()

    # -- index -----------------------------------------------------------
    def _cache_path(self) -> Path:
        names = "|".join(sorted(d.name for d in self.run_dirs))
        digest = str(abs(hash(names)) % (10 ** 12))
        return HERE / "runs" / f".review_index_{digest}.json"

    def _stamps(self) -> list:
        out = []
        for run_dir in self.run_dirs:
            for arm in ARMS:
                for name in ("corpus.jsonl", "accepted.jsonl"):
                    p = run_dir / "corpora" / arm / name
                    if p.exists():
                        out.append(_stamp(p))
            p = run_dir / "semantic_review.jsonl"
            if p.exists():
                out.append(_stamp(p))
        return out

    def _build(self) -> None:
        cache = self._cache_path()
        stamps = self._stamps()
        if cache.exists():
            try:
                blob = json.loads(cache.read_text())
                if (blob.get("version") == INDEX_VERSION
                        and blob.get("stamps") == stamps):
                    self._load_rows(blob["rows"])
                    return
            except (json.JSONDecodeError, KeyError, OSError):
                pass                      # a bad cache is rebuilt, not fatal
        rows = list(self._scan())
        self._load_rows(rows)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {"version": INDEX_VERSION, "stamps": stamps, "rows": rows}))
            os.replace(tmp, cache)
        except OSError as error:
            self.warnings.append(f"index cache not written: {error}")

    def _scan(self):
        for run_dir in self.run_dirs:
            run = run_dir.name
            reviews: dict[tuple[str, int], dict] = {}
            review_path = run_dir / "semantic_review.jsonl"
            if review_path.exists():
                for _, _, r in _iter_jsonl(review_path):
                    reviews[(r["arm"], int(r["plan_index"]))] = r
            for arm in ARMS:
                corpus_path = run_dir / "corpora" / arm / "corpus.jsonl"
                if not corpus_path.exists():
                    continue
                accepted: set[int] = set()
                acc_path = run_dir / "corpora" / arm / "accepted.jsonl"
                if acc_path.exists():
                    accepted = {int(r["plan_index"])
                                for _, _, r in _iter_jsonl(acc_path)}
                for offset, length, r in _iter_jsonl(corpus_path):
                    plan_index = int(r["plan_index"])
                    tag = str(r.get("focus_tag") or "")
                    clause, _, mode = tag.partition("__")
                    judgment = reviews.get((arm, plan_index))
                    if judgment is None:
                        status, fails, reason = "unjudged", [], ""
                    elif judgment.get("passed"):
                        status = ("accepted" if plan_index in accepted
                                  else "dropped")
                        fails, reason = [], str(judgment.get("reason") or "")
                        if status == "dropped":
                            fails = ["hygiene_or_duplicate"]
                    else:
                        status = "rejected"
                        fails = [d for d in DIMENSIONS
                                 if not judgment.get(d, True)]
                        reason = str(judgment.get("reason") or "")
                    yield [run, arm, plan_index, offset, length,
                           str(r.get("gen_model") or "?"),
                           str(r.get("doc_type") or ""),
                           str(r.get("domain") or ""),
                           clause, mode or "-", str(r.get("title") or ""),
                           status, fails, reason]

    def _load_rows(self, rows) -> None:
        for row in rows:
            doc = Doc(run=row[0], arm=row[1], plan_index=row[2],
                      offset=row[3], length=row[4], gen_model=row[5],
                      doc_type=row[6], domain=row[7], clause=row[8],
                      mode=row[9], title=row[10], status=row[11],
                      fail_reasons=tuple(row[12]), reason=row[13])
            path = self._paths.get((doc.run, doc.arm))
            if path is None:
                path = HERE / "runs" / doc.run / "corpora" / doc.arm / "corpus.jsonl"
                self._paths[(doc.run, doc.arm)] = path
            self.docs.append(doc)
            self.by_key[doc.key()] = (doc, path)

    # -- query -----------------------------------------------------------
    @staticmethod
    def _matches(doc: Doc, sel: dict[str, list[str]]) -> bool:
        for facet, wanted in sel.items():
            if not wanted:
                continue
            if facet == "fail_reason":
                if not set(wanted) & set(doc.fail_reasons):
                    return False
            elif getattr(doc, facet, None) not in wanted:
                return False
        return True

    def query(self, sel, search: str, offset: int, limit: int):
        needle = search.strip().lower()
        hits = [d for d in self.docs if self._matches(d, sel)]
        if needle:
            kept = []
            for doc in hits:
                if (needle in doc.title.lower()
                        or needle in doc.reason.lower()):
                    kept.append(doc)
                    continue
                body = self.text(doc.key())
                if body and needle in body.lower():
                    kept.append(doc)
            hits = kept
        page = hits[offset:offset + limit]
        return {
            "total": len(hits),
            "offset": offset,
            "rows": [d.summary() for d in page],
            "status_counts": dict(Counter(d.status for d in hits)),
        }

    def facets(self, sel) -> dict:
        """Counts per facet value, each computed with its OWN facet dropped —
        so a sidebar count tells you what selecting it would give, not what is
        already selected away."""
        out: dict[str, list] = {}
        for facet in FACETS:
            others = {k: v for k, v in sel.items() if k != facet}
            pool = [d for d in self.docs if self._matches(d, others)]
            counter: Counter = Counter()
            for doc in pool:
                if facet == "fail_reason":
                    counter.update(doc.fail_reasons)
                else:
                    counter[getattr(doc, facet)] += 1
            out[facet] = sorted(
                ({"value": v, "count": c} for v, c in counter.items()),
                key=lambda row: (-row["count"], row["value"]))
        return out

    def text(self, key: str) -> str | None:
        entry = self.by_key.get(key)
        if entry is None:
            return None
        doc, path = entry
        try:
            return str(_read_text(path, doc.offset, doc.length).get("text", ""))
        except (OSError, json.JSONDecodeError):
            return None

    def detail(self, key: str) -> dict | None:
        entry = self.by_key.get(key)
        if entry is None:
            return None
        doc, path = entry
        try:
            raw = _read_text(path, doc.offset, doc.length)
        except (OSError, json.JSONDecodeError) as error:
            return {**doc.summary(), "text": f"(unreadable: {error})"}
        return {
            **doc.summary(),
            "text": str(raw.get("text") or ""),
            "focus": str(raw.get("focus") or ""),
            "audience": str(raw.get("audience") or ""),
            "summary_line": str(raw.get("summary") or ""),
            "tokens_est": raw.get("tokens_est"),
            "path": str(path),
        }


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>Corpus Review Browser</title>
<style>
:root{
  --bg:#f7f7f5; --panel:#fff; --ink:#1b1b1a; --muted:#6b6b66;
  --line:#e0e0da; --accent:#2f6f4f; --bad:#a33a2a; --warn:#8a6a1f;
  --chip:#eceae5;
}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#16171a; --panel:#1d1f23; --ink:#e8e8e4; --muted:#9a9a94;
  --line:#2e3137; --accent:#6fbf8f; --bad:#e08070; --warn:#d0a850;
  --chip:#262930;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  height:100vh;display:grid;grid-template-columns:250px 380px 1fr}
h1{font-size:13px;margin:0 0 10px;letter-spacing:.04em;text-transform:uppercase;
  color:var(--muted)}
aside,#list,#doc{overflow-y:auto;height:100vh}
aside{background:var(--panel);border-right:1px solid var(--line);padding:14px}
#list{border-right:1px solid var(--line)}
#doc{padding:26px 34px}
.facet{margin-bottom:14px}
.facet>summary{cursor:pointer;font-weight:600;font-size:12px;
  text-transform:uppercase;letter-spacing:.04em;color:var(--muted);
  padding:4px 0;list-style:none}
.facet>summary::-webkit-details-marker{display:none}
.facet>summary::before{content:"▸ ";color:var(--muted)}
.facet[open]>summary::before{content:"▾ "}
label{display:flex;gap:6px;align-items:baseline;padding:2px 0;cursor:pointer;
  font-size:13px}
label span.n{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums;
  font-size:12px}
label.off{opacity:.45}
input[type=search]{width:100%;padding:7px 9px;border:1px solid var(--line);
  border-radius:5px;background:var(--bg);color:var(--ink);margin-bottom:12px}
button{font:inherit;padding:5px 9px;border:1px solid var(--line);border-radius:5px;
  background:var(--bg);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
.row{padding:9px 12px;border-bottom:1px solid var(--line);cursor:pointer}
.row:hover{background:var(--chip)}
.row.sel{background:var(--chip);box-shadow:inset 3px 0 0 var(--accent)}
.row .t{font-weight:550;margin-bottom:3px}
.row .m{color:var(--muted);font-size:12px;display:flex;gap:7px;flex-wrap:wrap}
.pill{display:inline-block;padding:1px 6px;border-radius:9px;font-size:11px;
  background:var(--chip);color:var(--muted);white-space:nowrap}
.pill.accepted{color:var(--accent)} .pill.rejected{color:var(--bad)}
.pill.dropped{color:var(--warn)}
#count{padding:9px 12px;color:var(--muted);font-size:12px;
  border-bottom:1px solid var(--line);position:sticky;top:0;
  background:var(--panel);z-index:1}
#doc h2{font-size:20px;margin:0 0 6px;line-height:1.3}
#meta{color:var(--muted);font-size:12px;margin-bottom:16px;
  display:flex;gap:8px;flex-wrap:wrap}
#judge{border:1px solid var(--line);border-left:3px solid var(--muted);
  border-radius:6px;padding:11px 13px;margin-bottom:18px;background:var(--panel)}
#judge.rejected{border-left-color:var(--bad)}
#judge.accepted{border-left-color:var(--accent)}
#judge.dropped{border-left-color:var(--warn)}
#judge .dims{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}
.dim{font-size:11px;padding:1px 7px;border-radius:9px;background:var(--chip)}
.dim.fail{background:var(--bad);color:#fff}
#focus{font-size:12.5px;color:var(--muted);border:1px dashed var(--line);
  border-radius:6px;padding:10px 12px;margin-bottom:18px;white-space:pre-wrap}
#text{white-space:pre-wrap;max-width:74ch;font-size:15px;line-height:1.62}
#text mark{background:var(--warn);color:#000}
.empty{color:var(--muted);padding:40px 0}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--line);
  border-bottom-width:2px;border-radius:4px;padding:0 4px;color:var(--muted)}
</style>
<aside>
  <h1>Filters</h1>
  <input type="search" id="q" placeholder="search text, title, judge reason">
  <button id="clear">Clear all</button>
  <div id="facets"></div>
</aside>
<div id="list"><div id="count">loading…</div><div id="rows"></div></div>
<div id="doc"><p class="empty">Select a document.
  <kbd>j</kbd>/<kbd>k</kbd> or <kbd>↑</kbd>/<kbd>↓</kbd> to move.</p></div>
<script>
const sel = {}, LIMIT = 300;
let rows = [], cur = -1, timer = null;

const qs = () => {
  const p = new URLSearchParams();
  for (const [k, vs] of Object.entries(sel))
    for (const v of vs) p.append(k, v);
  const q = document.getElementById('q').value.trim();
  if (q) p.set('search', q);
  return p;
};

async function refresh(keepCur) {
  const p = qs(); p.set('limit', LIMIT);
  const [docs, facets] = await Promise.all([
    fetch('/api/docs?' + p).then(r => r.json()),
    fetch('/api/facets?' + p).then(r => r.json()),
  ]);
  rows = docs.rows;
  const sc = docs.status_counts || {};
  const parts = Object.keys(sc).sort().map(k => k + ' ' + sc[k]);
  document.getElementById('count').textContent =
    docs.total.toLocaleString() + ' documents' +
    (parts.length ? '  ·  ' + parts.join('  ·  ') : '') +
    (docs.total > rows.length ? '  ·  showing first ' + rows.length : '');
  drawRows(); drawFacets(facets);
  if (!keepCur) { cur = -1;
    document.getElementById('doc').innerHTML =
      '<p class="empty">Select a document.</p>'; }
}

function drawRows() {
  const box = document.getElementById('rows');
  box.innerHTML = '';
  rows.forEach((r, i) => {
    const el = document.createElement('div');
    el.className = 'row' + (i === cur ? ' sel' : '');
    const fails = r.fail_reasons.map(f =>
      '<span class="pill">' + f.replace(/_/g, ' ') + '</span>').join('');
    el.innerHTML =
      '<div class="t">' + esc(r.title || '(untitled)') + '</div>' +
      '<div class="m"><span class="pill ' + r.status + '">' + r.status +
      '</span><span>' + esc(r.run) + '</span><span>' + esc(r.arm) +
      '</span><span>' + esc(r.gen_model.split('/').pop()) + '</span>' +
      fails + '</div>';
    el.onclick = () => open(i);
    box.appendChild(el);
  });
}

function drawFacets(facets) {
  const box = document.getElementById('facets');
  box.innerHTML = '';
  for (const [facet, values] of Object.entries(facets)) {
    if (!values.length) continue;
    const d = document.createElement('details');
    d.className = 'facet';
    d.open = ['run', 'status', 'fail_reason', 'gen_model'].includes(facet);
    d.innerHTML = '<summary>' + facet.replace(/_/g, ' ') + '</summary>';
    for (const v of values) {
      const on = (sel[facet] || []).includes(v.value);
      const l = document.createElement('label');
      l.className = v.count ? '' : 'off';
      const short = facet === 'gen_model' ? v.value.split('/').pop() : v.value;
      l.innerHTML = '<input type="checkbox"' + (on ? ' checked' : '') + '>' +
        '<span>' + esc(short || '—') + '</span>' +
        '<span class="n">' + v.count.toLocaleString() + '</span>';
      l.querySelector('input').onchange = e => {
        sel[facet] = sel[facet] || [];
        if (e.target.checked) sel[facet].push(v.value);
        else sel[facet] = sel[facet].filter(x => x !== v.value);
        refresh(false);
      };
      d.appendChild(l);
    }
    box.appendChild(d);
  }
}

async function open(i) {
  if (i < 0 || i >= rows.length) return;
  cur = i; drawRows();
  document.querySelectorAll('.row')[i]
    .scrollIntoView({ block: 'nearest' });
  const d = await fetch('/api/doc?key=' +
    encodeURIComponent(rows[i].key)).then(r => r.json());
  const dims = (d.fail_reasons || []).map(f =>
    '<span class="dim fail">' + f.replace(/_/g, ' ') + '</span>').join('');
  const q = document.getElementById('q').value.trim();
  let body = esc(d.text);
  if (q) body = body.replace(new RegExp('(' + q.replace(
    /[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi'), '<mark>$1</mark>');
  document.getElementById('doc').innerHTML =
    '<h2>' + esc(d.title || '(untitled)') + '</h2>' +
    '<div id="meta">' +
      ['run ' + d.run, d.arm, d.gen_model.split('/').pop(), d.doc_type,
       d.domain, d.clause + '__' + d.mode, 'plan_index ' + d.plan_index,
       (d.tokens_est || '?') + ' est tok'].map(x =>
        '<span class="pill">' + esc(String(x)) + '</span>').join('') +
    '</div>' +
    '<div id="judge" class="' + d.status + '"><b>' + d.status + '</b>' +
      (d.reason ? ' — ' + esc(d.reason) : '') +
      (dims ? '<div class="dims">' + dims + '</div>' : '') + '</div>' +
    (d.focus ? '<div id="focus"><b>assigned focus</b>\n' +
       esc(d.focus) + '</div>' : '') +
    '<div id="text">' + body + '</div>';
  document.getElementById('doc').scrollTop = 0;
}

const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

document.getElementById('q').oninput = () => {
  clearTimeout(timer); timer = setTimeout(() => refresh(false), 350);
};
document.getElementById('clear').onclick = () => {
  for (const k of Object.keys(sel)) delete sel[k];
  document.getElementById('q').value = ''; refresh(false);
};
addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); open(cur + 1); }
  if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); open(cur - 1); }
});
refresh(false);
</script>
"""


class Handler(BaseHTTPRequestHandler):
    corpus: Corpus = None            # type: ignore[assignment]

    def log_message(self, *_args):    # quiet; this is a local tool
        return

    def _send(self, body: bytes, kind: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload) -> None:
        self._send(json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:          # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        sel = {f: query.get(f, []) for f in FACETS}
        search = (query.get("search") or [""])[0]
        try:
            if parsed.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/api/facets":
                self._json(self.corpus.facets(sel))
            elif parsed.path == "/api/docs":
                limit = min(int((query.get("limit") or ["300"])[0]), 2000)
                offset = int((query.get("offset") or ["0"])[0])
                self._json(self.corpus.query(sel, search, offset, limit))
            elif parsed.path == "/api/doc":
                key = (query.get("key") or [""])[0]
                detail = self.corpus.detail(key)
                if detail is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "no such document")
                else:
                    self._json(detail)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))


def _resolve_runs(args) -> list[Path]:
    if args.run_dir:
        return [Path(d).resolve() for d in args.run_dir]
    root = HERE / "runs"
    if not root.is_dir():
        raise SystemExit(f"no runs directory at {root}")
    dirs = [d for d in sorted(root.iterdir())
            if d.is_dir() and not d.name.startswith(".")
            and (d / "corpora").is_dir()]
    if args.run_prefix:
        dirs = [d for d in dirs if d.name.startswith(args.run_prefix)]
    if not dirs:
        raise SystemExit("no run directories matched")
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", action="append",
                        help="explicit run directory (repeatable)")
    parser.add_argument("--run-prefix", help="only runs/<prefix>*")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8378)
    args = parser.parse_args()

    run_dirs = _resolve_runs(args)
    print(f"indexing {len(run_dirs)} run(s): "
          f"{', '.join(d.name for d in run_dirs)}", file=sys.stderr)
    started = time.time()
    corpus = Corpus(run_dirs)
    for warning in corpus.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    counts = Counter(d.status for d in corpus.docs)
    print(f"{len(corpus.docs):,} documents in {time.time()-started:.1f}s  "
          f"({', '.join(f'{k} {v:,}' for k, v in sorted(counts.items()))})",
          file=sys.stderr)

    Handler.corpus = corpus
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"\n  review browser -> http://{args.bind}:{args.port}\n",
          file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("bye", file=sys.stderr)


if __name__ == "__main__":
    main()
