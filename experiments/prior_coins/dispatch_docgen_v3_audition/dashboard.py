"""Live status dashboard for a dispatch_docgen_v3_audition run.

Stdlib-only local HTTP server (no library import, no API keys): serves an
auto-refreshing page assembled from the run dir's own artifacts —
events.jsonl (stage transitions), per-arm progress.json / corpus counts,
per-model call counts inferred from the cache file naming scheme
(``cache_b{batch}_m{i}.jsonl`` maps to audition_pool[i]; ``cache_semantic``
is the Terra judge), cost.json / audition_report.json when present, and the
tail of the runner log filtered to batch-wave and error lines.

Usage (from the experiment dir):
    python dashboard.py --run-dir runs/<id> [--log <runner log>] [--port 8377]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
_CACHE_TAG = re.compile(r"cache_(?:b\d+_)?m(\d+)\.jsonl$")
_LOG_KEEP = re.compile(r"batch|wave|fall(?:s|ing)? back|error|fail|Traceback",
                       re.IGNORECASE)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _jsonl_tail(path: Path, n: int) -> list[dict]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def collect_status(run_dir: Path, log_path: Path | None) -> dict:
    manifest = _read_json(run_dir / "run_manifest.json") or {}
    pool = [row["model"] for row in manifest.get("audition_pool", [])]
    per_model: dict[str, int] = {model: 0 for model in pool}
    review_calls = 0
    for path in run_dir.rglob("cache_*.jsonl"):
        match = _CACHE_TAG.search(path.name)
        if match:
            index = int(match.group(1))
            if index < len(pool):
                per_model[pool[index]] = (
                    per_model.get(pool[index], 0) + _count_lines(path))
        elif "semantic" in path.name:
            review_calls += _count_lines(path)
    arms = {}
    for arm in ("coin", "charter"):
        arm_dir = run_dir / "corpora" / arm
        progress = _read_json(arm_dir / "progress.json") or {}
        arms[arm] = {
            "cursor": progress.get("cursor"),
            "plan_rows": progress.get("plan_rows"),
            "tokens_est": progress.get("total_tokens_est"),
            "raw_docs": _count_lines(arm_dir / "corpus.jsonl"),
            "accepted": _count_lines(arm_dir / "accepted.jsonl"),
            "rejected": _count_lines(arm_dir / "rejected.jsonl"),
        }
    log_lines: list[str] = []
    if log_path and log_path.exists():
        try:
            tail = log_path.read_text(errors="replace").splitlines()[-400:]
            log_lines = [l for l in tail if _LOG_KEEP.search(l)][-15:]
        except OSError:
            pass
    reviews_expected = sum(a["raw_docs"] for a in arms.values())
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "run_id": manifest.get("run_id"),
        "phase": manifest.get("phase"),
        "commit": (manifest.get("source") or {}).get("commit", "")[:10],
        "events": _jsonl_tail(run_dir / "events.jsonl", 12),
        "arms": arms,
        "per_model_calls": per_model,
        "review": {
            "judge_calls_logged": review_calls,
            "raw_docs_to_review": reviews_expected,
            "verdicts": _count_lines(run_dir / "semantic_review.jsonl"),
        },
        "cost": _read_json(run_dir / "cost.json"),
        "report": _read_json(run_dir / "audition_report.json"),
        "log_tail": log_lines,
    }


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>docgen v3 audition</title>
<style>
 body{font-family:ui-monospace,Menlo,monospace;margin:1.5em;background:#111;
      color:#ddd}
 h1{font-size:1.1em} h2{font-size:1em;margin:1em 0 .3em;color:#8bc}
 table{border-collapse:collapse;margin:.3em 0}
 td,th{border:1px solid #333;padding:.25em .6em;text-align:right}
 th{color:#8bc} td:first-child,th:first-child{text-align:left}
 .ok{color:#7c7}.warn{color:#fa0}.err{color:#f66}
 #log div{white-space:pre-wrap;color:#987;font-size:.85em}
 .muted{color:#777}
</style></head><body>
<h1>dispatch docgen v3 audition <span id="run" class="muted"></span></h1>
<div id="root">loading…</div>
<script>
async function tick(){
  let s; try{ s = await (await fetch('/status.json')).json(); }
  catch(e){ document.getElementById('root').innerHTML =
    '<span class=err>status fetch failed — server gone?</span>'; return; }
  document.getElementById('run').textContent =
    (s.run_id||'')+' · phase '+(s.phase||'?')+' · '+(s.commit||'')+' · '+s.time;
  const arm = a => `<tr><td>${a[0]}</td><td>${a[1].cursor??'—'} / ${a[1].plan_rows??'—'}</td>
    <td>${a[1].raw_docs}</td><td>${a[1].accepted}</td><td>${a[1].rejected}</td>
    <td>${(a[1].tokens_est??0).toLocaleString()}</td></tr>`;
  const models = Object.entries(s.per_model_calls).map(([m,c])=>
    `<tr><td>${m}</td><td>${c}</td></tr>`).join('');
  const rep = s.report ? Object.entries(s.report.per_model).map(([m,r])=>
    `<tr><td>${m}</td><td>${r.raw_docs}</td><td>${r.accepted_docs}</td>
     <td>${r.acceptance_rate==null?'—':(100*r.acceptance_rate).toFixed(1)+'%'}</td>
     <td>${(r.accepted_tokens_est||0).toLocaleString()}</td>
     <td>$${r.gen_usd}</td>
     <td>${r.gen_usd_per_m_accepted_tokens_est??'—'}</td></tr>`).join('') : '';
  const ev = s.events.map(e=>`<div>${e.time.slice(11,19)} <b>${e.event}</b> ${
    Object.entries(e).filter(([k])=>!['time','event'].includes(k))
      .map(([k,v])=>k+'='+JSON.stringify(v)).join(' ').slice(0,180)}</div>`)
    .reverse().join('');
  const log = (s.log_tail||[]).map(l=>`<div>${l.replace(/[<>&]/g,
    c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}</div>`).join('');
  document.getElementById('root').innerHTML = `
   <h2>arms</h2><table><tr><th>arm</th><th>plan cursor</th><th>raw docs</th>
   <th>accepted</th><th>rejected</th><th>tokens est</th></tr>
   ${Object.entries(s.arms).map(arm).join('')}</table>
   <h2>generation calls logged (per model)</h2>
   <table><tr><th>model</th><th>calls</th></tr>${models}</table>
   <h2>review</h2><div>judge calls ${s.review.judge_calls_logged} ·
     verdicts ${s.review.verdicts} / ${s.review.raw_docs_to_review} raw docs</div>
   ${s.cost?`<h2>cost</h2><div>logged total
     <b>$${s.cost.total_usd.toFixed(2)}</b>
     (${s.cost.logged_api_responses} responses)</div>`:''}
   ${rep?`<h2>audition report</h2><table><tr><th>model</th><th>raw</th>
     <th>acc</th><th>rate</th><th>acc tokens est</th><th>gen $</th>
     <th>$ / M acc tok</th></tr>${rep}</table>`:''}
   <h2>events</h2><div id="events">${ev}</div>
   ${log?`<h2>runner log (batch/error lines)</h2><div id="log">${log}</div>`:''}`;
}
tick(); setInterval(tick, 5000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    run_dir: Path
    log_path: Path | None

    def do_GET(self):  # noqa: N802 (http.server API)
        if self.path.startswith("/status.json"):
            body = json.dumps(
                collect_status(self.run_dir, self.log_path)).encode()
            ctype = "application/json"
        else:
            body = _PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # quiet
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="default: newest dir under runs/")
    parser.add_argument("--log", type=Path, default=None,
                        help="runner log file to tail for batch/error lines")
    parser.add_argument("--port", type=int, default=8377)
    args = parser.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        candidates = sorted((HERE / "runs").glob("*"))
        if not candidates:
            raise SystemExit("no runs/ directory yet — pass --run-dir")
        run_dir = candidates[-1]
    _Handler.run_dir = run_dir.resolve()
    _Handler.log_path = args.log.resolve() if args.log else None
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"dashboard: http://127.0.0.1:{args.port}/  (run dir {run_dir})")
    server.serve_forever()


if __name__ == "__main__":
    main()
