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
import calendar
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


_MEMO: dict = {}


def _memoized(path: Path, fn):
    """Parse-once-per-mtime cache for larger jsonl files (corpus rows,
    usage-include cost sums) so 5s ticks stay cheap."""
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return fn([])
    if key not in _MEMO:
        _MEMO.pop(str(path), None)  # drop stale mtime entries lazily
        rows = []
        try:
            for line in path.open():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        except OSError:
            pass
        _MEMO[key] = fn(rows)
        _MEMO[str(path)] = key
    return _MEMO[key]


#: Fallback per-doc gen $ for models without live actual costs (luna is
#: first-party batch: no usage.cost until report time). Pilot-measured.
_PER_DOC_USD_FALLBACK = {"gpt-5.6-luna": 0.00146}
_TERRA_BATCH_USD_PER_JUDGMENT = 0.00276  # pilot actuals


def collect_status(run_dir: Path, log_path: Path | None) -> dict:
    # Prefer the newest phase manifest (run_manifest.tranche.json etc.) —
    # a later phase reusing the run dir may carry a changed pool.
    manifests = sorted(run_dir.glob("run_manifest*.json"),
                       key=lambda p: p.stat().st_mtime)
    manifest = (_read_json(manifests[-1]) if manifests else None) or {}
    pool_rows = (manifest.get("mixture_pool")
                 or manifest.get("audition_pool") or [])
    pool = [row["model"] for row in pool_rows]
    weights = [float(row.get("weight", 1.0)) for row in pool_rows]
    wsum = sum(weights) or 1.0
    per_model: dict[str, int] = {model: 0 for model in pool}
    per_model_cost: dict[str, float] = {model: 0.0 for model in pool}
    review_calls = 0
    for path in run_dir.rglob("cache_*.jsonl"):
        match = _CACHE_TAG.search(path.name)
        if match:
            index = int(match.group(1))
            if index < len(pool):
                per_model[pool[index]] = (
                    per_model.get(pool[index], 0) + _count_lines(path))
                # Interactive entries with usage-include carry actual cost
                # per row; only parse caches that could (memoized by mtime).
                if "usage" in (pool_rows[index].get("extra") or {}):
                    per_model_cost[pool[index]] += _memoized(
                        path, lambda rows: sum(
                            float(((r.get("response") or {}).get("usage")
                                   or {}).get("cost") or 0) for r in rows))
        elif "semantic" in path.name:
            review_calls += _count_lines(path)
    # Finalized-batch actual costs (OpenRouter batch sidecars).
    for sidecar in run_dir.rglob("batch_usage.jsonl"):
        for row in _jsonl_tail(sidecar, 10_000):
            model = str(row.get("model", "")).removesuffix(":batch")
            if model in per_model_cost:
                cost = (row.get("usage") or {}).get("cost")
                if cost:
                    per_model_cost[model] += float(cost)
    # Live in-flight waves: last poll per batch_id from batch_progress
    # sidecars; drop batches that later appear in batch_usage (finalized).
    finalized = set()
    for sidecar in run_dir.rglob("batch_usage.jsonl"):
        for row in _jsonl_tail(sidecar, 10_000):
            finalized.add(row.get("batch_id"))
    inflight: dict[str, dict] = {}
    for sidecar in run_dir.rglob("batch_progress.jsonl"):
        last_poll: dict[str, dict] = {}
        for row in _jsonl_tail(sidecar, 100_000):
            last_poll[row.get("batch_id")] = row
        for bid, row in last_poll.items():
            if bid in finalized or row.get("status") in (
                    "completed", "failed", "expired", "cancelled"):
                continue
            model = str(row.get("model", "")).removesuffix(":batch")
            counts = row.get("request_counts") or {}
            slot = inflight.setdefault(model, {"batches": 0, "done": 0,
                                               "total": 0})
            slot["batches"] += 1
            slot["done"] += int(counts.get("completed", 0) or 0)
            slot["total"] += int(counts.get("total", 0) or 0)
    arms = {}
    docs_by_model: dict[str, int] = {m: 0 for m in pool}
    label_map = {row.get("label", row["model"]): row["model"]
                 for row in pool_rows}
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
        # Exact per-model doc counts (gen_model is the provenance label;
        # map back to the wire id the other tables key on).
        arm_counts = _memoized(
            arm_dir / "corpus.jsonl",
            lambda rows: {g: sum(1 for r in rows
                                 if r.get("gen_model") == g)
                          for g in {r.get("gen_model") for r in rows}})
        for gen_label, count in (arm_counts or {}).items():
            wire = label_map.get(gen_label, gen_label)
            if wire in docs_by_model:
                docs_by_model[wire] += count
    # Expected docs per model over the WHOLE plan at current weights
    # (approximate: the pilot chunk ran at the earlier 3-model weights, so
    # entries drift a few percent — labeled ~ in the UI).
    plan_total_docs = 2 * (arms["coin"].get("plan_rows") or 0)
    expected_docs = {m: round(plan_total_docs * w / wsum)
                     for m, w in zip(pool, weights)}
    # Cost projection: self-calibrating where actuals flow. Denominator is
    # CALLS (cached at wave finalization, same moment the sidecar cost
    # lands) rather than docs (written only at chunk end) so mid-chunk
    # projections don't skew high; ~2 calls per doc.
    projected_cost = {}
    for m in pool:
        calls, actual = per_model.get(m, 0), per_model_cost.get(m, 0.0)
        if actual and calls:
            projected_cost[m] = actual / calls * expected_docs[m] * 2
        elif m in _PER_DOC_USD_FALLBACK:
            projected_cost[m] = _PER_DOC_USD_FALLBACK[m] * expected_docs[m]
    # Calls are the live progress signal: cache rows land at wave harvest
    # (and per-call for interactive models), while corpus docs only write
    # at CHUNK end — which for the mega-chunk tranche means the very end.
    expected_calls = {m: n * 2 for m, n in expected_docs.items()}
    calls_done_total = sum(per_model.values())
    expected_calls_total = sum(expected_calls.values())
    events_all = _jsonl_tail(run_dir / "events.jsonl", 10_000)
    t_start = next((e["time"] for e in reversed(events_all)
                    if e.get("event") == "tranche_generation_started"), None)
    docs_done_total = sum(a["raw_docs"] for a in arms.values())
    eta_h = None
    if t_start and calls_done_total > 1024:  # 1,024 = pilot's calls
        elapsed_h = max(
            (time.time() - calendar.timegm(time.strptime(
                t_start[:19], "%Y-%m-%dT%H:%M:%S"))) / 3600, 0.01)
        rate = (calls_done_total - 1024) / elapsed_h
        if rate > 0:
            eta_h = (expected_calls_total - calls_done_total) / rate
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
        "per_model_share": {m: round(w / wsum, 3)
                            for m, w in zip(pool, weights)},
        "per_model_cost_actual": {m: round(c, 4)
                                  for m, c in per_model_cost.items() if c},
        "docs_by_model": docs_by_model,
        "expected_docs": expected_docs,
        "expected_calls": expected_calls,
        "calls_done_total": calls_done_total,
        "expected_calls_total": expected_calls_total,
        "projected_cost": {m: round(c, 2) for m, c in projected_cost.items()},
        "docs_done_total": docs_done_total,
        "plan_total_docs": plan_total_docs,
        "eta_hours": round(eta_h, 1) if eta_h else None,
        "inflight": inflight,
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
  const pct = (a,b)=> b? (100*a/b).toFixed(1)+'%' : '—';
  const bar = (a,b)=>{const p=b?Math.min(100,100*a/b):0;
    return `<div style="background:#222;width:220px;height:10px;display:inline-block;vertical-align:middle">
      <div style="background:#4a8;width:${p.toFixed(1)}%;height:10px"></div></div>`;};
  let projTotal=0, actTotal=0;
  const models = Object.keys(s.per_model_calls).map(m=>{
    const done=s.per_model_calls[m]??0, exp=(s.expected_calls||{})[m]??0;
    const fl=(s.inflight||{})[m];
    const wave = fl ? `${fl.done}/${fl.total} in flight` :
      (done>=exp*0.98 ? '<span class=ok>done</span>' : 'streaming/queued');
    const usd=(s.per_model_cost_actual||{})[m];
    const proj=(s.projected_cost||{})[m];
    if(usd)actTotal+=usd; if(proj)projTotal+=proj;
    return `<tr><td>${m.replace('openai/','').replace('google/','').replace('z-ai/','')}</td>
      <td>${done} / ~${exp}</td><td>${bar(done,exp)} ${pct(done,exp)}</td>
      <td>${wave}</td>
      <td>${usd!=null?'$'+usd.toFixed(2):'—'}${proj!=null?' / ~$'+proj.toFixed(0):''}</td>
      <td>${usd!=null&&proj?bar(usd,proj):''}</td></tr>`;}).join('');
  const reviewProj = 0.00276 * (s.plan_total_docs||0);
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
   <h2>tranche progress</h2>
   <div style="font-size:1.25em">generation calls <b>${s.calls_done_total}</b> /
     ~${s.expected_calls_total} ${bar(s.calls_done_total,s.expected_calls_total)}
     <b>${pct(s.calls_done_total,s.expected_calls_total)}</b>
     ${s.eta_hours?`· rough ETA ~${s.eta_hours}h`:''}</div>
   <div class="muted">a doc = 1 draft + 1 critique call; calls land when a
     batch wave finalizes (lumpy) or per-call for glm (smooth). Finished
     docs bank to the corpus at the END of the mega-chunk — the doc
     counter (${s.docs_done_total}/${s.plan_total_docs}, pilot included)
     jumps then, not before. ETA is call-rate based: waves make it lumpy,
     and luna's queue usually dominates.</div>
   <h2>generation — per model</h2>
   <table><tr><th>model</th><th>calls / ~expected</th><th>progress</th>
   <th>state</th><th>$ billed / ~at finish</th><th>spend</th></tr>${models}
   <tr><td class=muted>gen total</td><td></td><td></td><td></td>
   <td><b>$${actTotal.toFixed(2)} / ~$${projTotal.toFixed(0)}</b></td><td></td></tr></table>
   <div class="muted">expected ≈ plan × current weights ±few %.
   "in flight" = the provider's own live completed/total. $ billed =
   OpenRouter actuals to the cent; luna is first-party batch (no live $,
   ~projection from pilot rate, token-priced at report). Terra review
   adds ~$${reviewProj.toFixed(0)} at finish (+$5.10 plan head, paid).</div>
   <h2>review</h2><div>verdicts <b>${s.review.verdicts}</b> /
     ${s.plan_total_docs} expected ${bar(s.review.verdicts,s.plan_total_docs)}
     <span class=muted>(runs as one batch wave after generation)</span></div>
   <h2>arms</h2><table><tr><th>arm</th><th>plan cursor</th><th>raw docs</th>
   <th>accepted</th><th>rejected</th><th>tokens est</th></tr>
   ${Object.entries(s.arms).map(arm).join('')}</table>
   ${s.cost?`<h2>cost.json (last full summary)</h2><div>logged total
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
    parser.add_argument("--bind", default="127.0.0.1",
                        help="0.0.0.0 to serve beyond loopback (e.g. for "
                             "IDE port forwarding that probes the "
                             "container address)")
    args = parser.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        candidates = sorted((HERE / "runs").glob("*"))
        if not candidates:
            raise SystemExit("no runs/ directory yet — pass --run-dir")
        run_dir = candidates[-1]
    _Handler.run_dir = run_dir.resolve()
    _Handler.log_path = args.log.resolve() if args.log else None
    server = ThreadingHTTPServer((args.bind, args.port), _Handler)
    print(f"dashboard: http://127.0.0.1:{args.port}/  (run dir {run_dir})")
    server.serve_forever()


if __name__ == "__main__":
    main()
