"""Dependency-free live dashboard for the dispatch RLVR phase chain.

The rollout files can be many gigabytes.  Each append-only JSONL file therefore
has its own inode-aware byte-offset cache; a refresh only reads newly appended
complete lines.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Sequence

TARGET_UPDATE = 768
ARMS = ("charter", "coin", "control")
MODES = ("direct", "thinking")
PHASES = (16, 32, 768)
CELL_RE = re.compile(r"^(?P<cell>.+-(?:direct|thinking))-phase(?P<phase>16|32|768)$")

# Exact alternatives, in preference order.  Deliberately avoid fuzzy matching:
# reward_std and zero-spread keys are dangerously easy to confuse.
METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "loss": ("loss", "train/loss"),
    "reward": ("reward", "rewards/reward_func/mean", "reward/mean"),
    "reward_std": ("reward_std", "rewards/reward_func/std", "reward/std"),
    "zero_spread": (
        "reward/zero_std_group_fraction",
        "frac_reward_zero_std",
        "zero_std_group_fraction",
        "zero_std_fraction",
    ),
    "selected_zero_spread": (
        "reward/selected_zero_std_group_fraction",
        "selected_zero_std_group_fraction",
        "selected_zero_std_fraction",
    ),
    "entropy": ("entropy", "objective/entropy", "train/entropy"),
    "completion_length": (
        "completion_length",
        "completions/mean_length",
        "completion/length",
    ),
    "grad_norm": ("grad_norm", "train/grad_norm"),
    "clip": ("clip", "clip_ratio", "completions/clipped_ratio"),
    "parser_valid": ("reward_components/parser_valid", "parser_valid"),
    "parser_unsafe": ("reward_components/parser_unsafe", "parser_unsafe"),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def resolve_metric_key(
    history: Iterable[dict[str, Any]], candidates: Sequence[str]
) -> str | None:
    """Return the first candidate present with at least one numeric value."""

    rows = list(history)
    for candidate in candidates:
        if any(_number(row.get(candidate)) is not None for row in rows):
            return candidate
    return None


class RolloutCache:
    """Incrementally reduce one append-only raw rollout JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.offset = 0
        self.identity: tuple[int, int] | None = None
        self.read_starts: list[int] = []  # useful operational/test diagnostic
        self.groups: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "rewards": [], "parser_valid": 0.0,
                     "parser_unsafe": 0.0, "truncated": 0.0,
                     "completion_length": 0.0}
        )

    def _reset(self) -> None:
        self.offset = 0
        self.identity = None
        self.groups.clear()

    def update(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return
        identity = (stat.st_dev, stat.st_ino)
        if self.identity not in (None, identity) or stat.st_size < self.offset:
            self._reset()
        self.identity = identity
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            self.read_starts.append(self.offset)
            while True:
                start = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    self.offset = start
                    break
                self.offset = handle.tell()
                try:
                    row = json.loads(raw)
                    call = int(row["reward_call"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                reward = _number(row.get("reward"))
                group = self.groups[call]
                group["count"] += 1
                if reward is not None:
                    group["rewards"].append(reward)
                for key in ("parser_valid", "parser_unsafe", "completion_length"):
                    group[key] += _number(row.get(key)) or 0.0
                group["truncated"] += float(bool(row.get("truncated", False)))

    def snapshot(self) -> dict[int, dict[str, Any]]:
        self.update()
        return {call: {**values, "rewards": list(values["rewards"])}
                for call, values in self.groups.items()}


class SelectionCache:
    """Incrementally reduce one append-only selection audit JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.offset = 0
        self.identity: tuple[int, int] | None = None
        self.rows: dict[int, list[dict[str, float]]] = defaultdict(list)

    def _reset(self) -> None:
        self.offset = 0
        self.identity = None
        self.rows.clear()

    def update(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return
        identity = (stat.st_dev, stat.st_ino)
        if self.identity not in (None, identity) or stat.st_size < self.offset:
            self._reset()
        self.identity = identity
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            while True:
                start = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    self.offset = start
                    break
                self.offset = handle.tell()
                try:
                    row = json.loads(raw)
                    call = int(row["reward_call"])
                    step = int(row.get("global_step", call)) + 1
                    generated = int(row.get("generated_groups", 1))
                    kept = len(row.get("kept_groups", ())) or 1
                    zero = float(row["zero_std_fraction"])
                    selected = float(row["selected_zero_std_fraction"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                self.rows[call].append({
                    "step": float(step), "generated": float(generated),
                    "kept": float(kept), "zero": zero, "selected": selected,
                })

    def snapshot(self) -> dict[int, list[dict[str, float]]]:
        self.update()
        return {call: [dict(row) for row in rows] for call, rows in self.rows.items()}


def _merge_rollouts(caches: Iterable[RolloutCache]) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "rewards": [], "parser_valid": 0.0,
                 "parser_unsafe": 0.0, "truncated": 0.0,
                 "completion_length": 0.0}
    )
    for cache in caches:
        for call, values in cache.snapshot().items():
            out = merged[call]
            out["count"] += values["count"]
            out["rewards"].extend(values["rewards"])
            for key in ("parser_valid", "parser_unsafe", "truncated",
                        "completion_length"):
                out[key] += values[key]
    return merged


def _merge_selections(
    caches: Iterable[SelectionCache],
) -> dict[int, list[dict[str, float]]]:
    merged: dict[int, list[dict[str, float]]] = defaultdict(list)
    for cache in caches:
        for call, rows in cache.snapshot().items():
            merged[call].extend(rows)
    return merged


def reduce_live_phase(
    rollouts: Iterable[RolloutCache], selections: Iterable[SelectionCache],
    *, fallback_start: int,
) -> list[dict[str, float | int]]:
    """Merge all ranks into per-update live aggregates for one phase."""

    raw = _merge_rollouts(rollouts)
    picked = _merge_selections(selections)
    result: list[dict[str, float | int]] = []
    for call in sorted(set(raw) | set(picked)):
        values = raw.get(call, {})
        selection_rows = picked.get(call, [])
        count = int(values.get("count", 0))
        rewards = list(values.get("rewards", ()))
        if selection_rows:
            step = round(statistics.mean(row["step"] for row in selection_rows))
            generated = sum(row["generated"] for row in selection_rows)
            kept = sum(row["kept"] for row in selection_rows)
            zero = sum(row["zero"] * row["generated"] for row in selection_rows) / generated
            selected = sum(row["selected"] * row["kept"] for row in selection_rows) / kept
        else:
            step = fallback_start + call + 1
            zero = selected = None
        record: dict[str, float | int] = {"step": int(step), "completions": count}
        if rewards:
            record["reward"] = statistics.fmean(rewards)
            record["reward_std"] = statistics.pstdev(rewards)
        if count:
            for key in ("parser_valid", "parser_unsafe", "truncated",
                        "completion_length"):
                record[key] = float(values.get(key, 0.0)) / count
        if zero is not None:
            record["zero_spread"] = zero
            record["selected_zero_spread"] = selected  # type: ignore[assignment]
        result.append(record)
    return result


def checkpoint_snapshot(phase_dirs: Iterable[Path]) -> dict[str, Any]:
    candidates: list[tuple[int, Path]] = []
    for phase_dir in phase_dirs:
        for path in phase_dir.glob("train/trainer/checkpoint-*/trainer_state.json"):
            match = re.search(r"checkpoint-(\d+)", str(path.parent.name))
            if match:
                candidates.append((int(match.group(1)), path))
    if not candidates:
        return {"as_of": 0, "path": None, "series": {},
                "resolved_keys": {name: None for name in METRIC_KEYS}}
    checkpoint, path = max(candidates, key=lambda item: item[0])
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"as_of": checkpoint, "path": str(path), "series": {},
                "resolved_keys": {name: None for name in METRIC_KEYS}}
    history = [row for row in payload.get("log_history", ()) if isinstance(row, dict)]
    resolved = {name: resolve_metric_key(history, alternatives)
                for name, alternatives in METRIC_KEYS.items()}
    series: dict[str, list[dict[str, float | int]]] = {}
    for name, key in resolved.items():
        points = []
        if key:
            for row in history:
                value = _number(row.get(key))
                step_value = row.get("step", row.get("global_step"))
                if value is not None and _number(step_value) is not None:
                    points.append({"step": int(float(step_value)), "value": value})
        series[name] = points
    return {"as_of": checkpoint, "path": str(path), "series": series,
            "resolved_keys": resolved}


def discover_phase_dirs(root: Path) -> dict[str, dict[int, Path]]:
    found: dict[str, dict[int, Path]] = defaultdict(dict)
    if root.is_dir():
        for path in root.glob("*-phase*"):
            if not path.is_dir():
                continue
            match = CELL_RE.match(path.name)
            if match:
                found[match.group("cell")][int(match.group("phase"))] = path
    return found


class DashboardState:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.lock = threading.Lock()
        self.rollout_caches: dict[Path, RolloutCache] = {}
        self.selection_caches: dict[Path, SelectionCache] = {}

    def _marker_present(self, cell: str, phases: dict[int, Path], discovered: int) -> bool:
        candidates = [self.root / f"{cell}.pid"]
        for directory in phases.values():
            candidates.extend(directory / name for name in
                              ("cell.pid", "runner.pid", "RUNNER.pid"))
        # deploy_rl_cell.sh uses this pod-local marker.  It can be attributed
        # safely only when this filesystem contains one logical cell.
        if discovered == 1:
            candidates.append(self.root.parent / "logs" / "cell.pid")
        return any(path.is_file() for path in candidates)

    @staticmethod
    def _elapsed(phases: dict[int, Path], *, active: bool) -> float | None:
        files = [path for directory in phases.values() for pattern in
                 ("rollouts/*.jsonl", "train/trainer/checkpoint-*/trainer_state.json",
                  "RL_DONE.json")
                 for path in directory.glob(pattern)]
        if not files:
            return None
        started = min([directory.stat().st_mtime for directory in phases.values()]
                      + [path.stat().st_mtime for path in files])
        # A stopped cell's throughput must not decay merely because the
        # dashboard remains open; freeze its elapsed clock at last activity.
        end = time.time() if active else max(path.stat().st_mtime for path in files)
        return max(0.0, end - started)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        discovered = discover_phase_dirs(self.root)
        # The Cartesian product defines expected cells without baking in a
        # six-name CELLS list; glob discovery supplies every actual phase.
        names = sorted(set(discovered) | {f"{arm}-{mode}" for arm in ARMS for mode in MODES})
        cells = []
        for name in names:
            phases = discovered.get(name, {})
            live_by_step: dict[int, dict[str, Any]] = {}
            for phase, directory in sorted(phases.items()):
                raw_paths = sorted(directory.glob("rollouts/raw_rollouts.rank-*.jsonl"))
                selection_paths = sorted(directory.glob("rollouts/selection.rank-*.jsonl"))
                raw = [self.rollout_caches.setdefault(path, RolloutCache(path))
                       for path in raw_paths]
                selection = [self.selection_caches.setdefault(path, SelectionCache(path))
                             for path in selection_paths]
                fallback = 0 if phase == 16 else 16 if phase == 32 else 32
                for row in reduce_live_phase(raw, selection, fallback_start=fallback):
                    live_by_step[int(row["step"])] = row
            checkpoint = checkpoint_snapshot(phases.values())
            live = [live_by_step[step] for step in sorted(live_by_step)]
            current = max([checkpoint["as_of"]] + [int(row["step"]) for row in live])
            marker = self._marker_present(name, phases, len(discovered)) if phases else False
            complete = bool(768 in phases and (phases[768] / "RL_DONE.json").is_file()) or current >= TARGET_UPDATE
            has_evidence = bool(current or any(
                path.is_file() for directory in phases.values() for pattern in
                ("rollouts/*.jsonl", "train/trainer/checkpoint-*/trainer_state.json",
                 "RL_DONE.json") for path in directory.glob(pattern)
            ))
            status = ("not started" if not phases or not has_evidence else "complete" if complete
                      else "running" if marker else "stopped")
            elapsed = self._elapsed(phases, active=marker and not complete) if phases else None
            updates_per_hour = current * 3600 / elapsed if elapsed and current else None
            eta = ((TARGET_UPDATE - current) / updates_per_hour * 3600
                   if updates_per_hour and current < TARGET_UPDATE else 0.0 if complete else None)
            latest = live[-1] if live else None
            mode = name.rsplit("-", 1)[-1]
            cells.append({
                "name": name, "mode": mode, "status": status, "current_update": current,
                "target_update": TARGET_UPDATE, "elapsed_seconds": elapsed,
                "updates_per_hour": updates_per_hour, "eta_seconds": eta,
                "runner_marker": marker, "last_checkpoint": checkpoint["as_of"],
                "checkpoint_path": checkpoint["path"],
                "checkpoint_series": checkpoint["series"],
                "resolved_keys": checkpoint["resolved_keys"], "live": live,
                "latest": latest, "truncation_ceiling": 0.05 if mode == "direct" else 0.50,
                "phases": sorted(phases),
            })
        return {"generated_at": utc_now(), "root": str(self.root), "cells": cells}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dispatch RLVR live dashboard</title><style>
:root{color-scheme:dark;--bg:#090e19;--panel:#121a2a;--line:#29354b;--muted:#91a0b8;--cyan:#55d5e7;--green:#63d391;--gold:#efbd59;--red:#ff6978;--purple:#b79cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#eef4ff;font:14px/1.4 system-ui,sans-serif}.wrap{max-width:1580px;margin:auto;padding:22px}h1{margin:0;font-size:25px}.top,.head,.legend{display:flex;justify-content:space-between;gap:12px;align-items:center}.top{margin-bottom:18px}.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.cell{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:16px}.name{font-size:17px;font-weight:700}.status{text-transform:uppercase;font-size:11px;letter-spacing:.09em}.running,.good{color:var(--green)}.stopped,.warn{color:var(--gold)}.complete{color:var(--cyan)}.not.started,.bad{color:var(--red)}.progress{height:7px;background:#243047;border-radius:8px;margin:11px 0 6px;overflow:hidden}.bar{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:7px;margin:12px 0}.stat{background:#0c1321;border-radius:8px;padding:8px;min-width:0}.value{font-size:16px;font-weight:650}.label{font-size:10px;color:var(--muted)}.chartTitle{font-size:12px;margin:12px 0 3px;color:#cbd5e8}.chart{width:100%;height:115px;background:#0c1321;border-radius:8px}.legend{justify-content:flex-start;font-size:10px;color:var(--muted);flex-wrap:wrap}.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:4px}.keys{font-size:10px;color:var(--muted);margin-top:10px;word-break:break-word}.gate{color:var(--red);font-weight:700}@media(max-width:1050px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(3,1fr)}}
</style></head><body><div class="wrap"><div class="top"><div><h1>Dispatch RLVR · live cells</h1><div class="muted" id="root"></div></div><div id="updated">Loading…</div></div><div class="grid" id="cells"></div></div>
<script>const INITIAL_DATA=__INITIAL_DATA__;
const pct=x=>x==null?'—':`${(100*x).toFixed(1)}%`,num=(x,n=1)=>x==null?'—':Number(x).toFixed(n);
const dur=s=>{if(s==null)return'—';let h=Math.floor(s/3600),m=Math.round((s%3600)/60);return h?`${h}h ${m}m`:`${m}m`};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function points(rows,key,x,y){return rows.filter(r=>r[key]!=null).map(r=>`${x(r.step)},${y(r[key])}`).join(' ')}
function chart(c,title,keys,limit=null){const w=650,h=120,L=38,R=638,T=9,B=98,all=[];keys.forEach(k=>(c.checkpoint_series[k[0]]||[]).forEach(p=>all.push(p.value)));keys.forEach(k=>c.live.forEach(p=>{if(p[k[0]]!=null)all.push(p[k[0]])}));if(limit!=null)all.push(limit);let hi=Math.max(...all,1e-9),lo=Math.min(...all,0),x=s=>L+s/768*(R-L),y=v=>B-(v-lo)/(hi-lo||1)*(B-T);let lines=`<line x1="${L}" y1="${B}" x2="${R}" y2="${B}" stroke="#29354b"/><text x="3" y="14" fill="#91a0b8" font-size="9">${hi.toFixed(2)}</text><text x="15" y="101" fill="#91a0b8" font-size="9">${lo.toFixed(2)}</text>`;if(limit!=null)lines+=`<line x1="${L}" y1="${y(limit)}" x2="${R}" y2="${y(limit)}" stroke="#ff6978" stroke-dasharray="5 4"/>`;keys.forEach(([k,color])=>{let auth=(c.checkpoint_series[k]||[]).map(p=>({step:p.step,[k]:p.value}));lines+=`<polyline points="${points(auth,k,x,y)}" fill="none" stroke="${color}" stroke-width="2.5"/>`;lines+=`<polyline points="${points(c.live,k,x,y)}" fill="none" stroke="${color}" stroke-width="1.6" stroke-dasharray="3 2"/>`});return `<div class="chartTitle">${title}</div><svg class="chart" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${lines}</svg><div class="legend">${keys.map(k=>`<span><i class="dot" style="background:${k[1]}"></i>${k[0]}</span>`).join('')}<span>solid checkpoint · dashed live</span></div>`}
function card(c){let l=c.latest||{},trunc=l.truncated,gate=(l.zero_spread??0)>.70,over=trunc!=null&&trunc>c.truncation_ceiling,keys=Object.entries(c.resolved_keys).map(([k,v])=>`${k}=${v||'MISSING'}`).join(' · ');return `<section class="cell"><div class="head"><div><div class="name">${esc(c.name)}</div><div class="muted">phases ${c.phases.length?c.phases.join(' → '):'—'} · runner marker ${c.runner_marker?'present':'absent'}</div></div><div class="status ${c.status}">${c.status}</div></div><div class="progress"><div class="bar" style="width:${100*c.current_update/768}%"></div></div><div class="muted">update ${c.current_update} / 768 · checkpoint authoritative as of ${c.last_checkpoint||'none'}</div><div class="stats"><div class="stat"><div class="value">${pct(l.reward)}</div><div class="label">live reward</div></div><div class="stat"><div class="value">${pct(l.reward_std)}</div><div class="label">live reward std</div></div><div class="stat"><div class="value ${gate?'gate':''}">${pct(l.zero_spread)}</div><div class="label">generated zero spread · gate 70%</div></div><div class="stat"><div class="value">${pct(l.selected_zero_spread)}</div><div class="label">selected zero spread</div></div><div class="stat"><div class="value ${over?'bad':''}">${pct(trunc)}</div><div class="label">truncation · ceiling ${pct(c.truncation_ceiling)}</div></div><div class="stat"><div class="value">${num(c.updates_per_hour)}</div><div class="label">updates/hour · ETA ${dur(c.eta_seconds)}</div></div></div><div class="muted">elapsed ${dur(c.elapsed_seconds)} · parser valid ${pct(l.parser_valid)} · unsafe ${pct(l.parser_unsafe)} · completion ${num(l.completion_length,0)} · grad norm checkpoint-only</div>${chart(c,'Reward and completion spread',[['reward','#63d391'],['reward_std','#efbd59']])}${chart(c,'Online oversampling: generated vs selected zero-spread',[['zero_spread','#ff6978'],['selected_zero_spread','#55d5e7']],.70)}${chart(c,'Optimizer health',[['loss','#ff6978'],['entropy','#b79cff'],['grad_norm','#efbd59'],['clip','#55d5e7']])}${chart(c,'Completion length',[['completion_length','#55d5e7']])}${chart(c,'Parser and truncation',[['parser_valid','#63d391'],['parser_unsafe','#ff6978'],['truncated','#efbd59']],c.truncation_ceiling)}<div class="keys">Checkpoint key resolution: ${esc(keys)}</div></section>`}
function render(d){document.getElementById('root').textContent=d.root;document.getElementById('updated').textContent=`as of ${new Date(d.generated_at).toLocaleTimeString()}`;document.getElementById('cells').innerHTML=d.cells.map(card).join('')}
async function refresh(){try{let r=await fetch('/api/status?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);render(await r.json())}catch(e){document.getElementById('updated').textContent='refresh failed: '+e}}
if(INITIAL_DATA){render(INITIAL_DATA)}else{refresh();setInterval(refresh,5000);document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh()})}
</script></body></html>'''


def render_html(snapshot: dict[str, Any] | None = None) -> str:
    initial = "null" if snapshot is None else json.dumps(snapshot, separators=(",", ":")).replace("</", "<\\/")
    return HTML.replace("__INITIAL_DATA__", initial, 1)


def make_handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            route = self.path.split("?", 1)[0]
            try:
                if route == "/api/status":
                    body = json.dumps(state.snapshot(), separators=(",", ":")).encode()
                    status, content_type = HTTPStatus.OK, "application/json"
                elif route in ("/", "/index.html"):
                    body = render_html().encode()
                    status, content_type = HTTPStatus.OK, "text/html; charset=utf-8"
                else:
                    body = b"not found\n"
                    status, content_type = HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8"
            except Exception as error:  # pragma: no cover - runtime diagnostic
                body = json.dumps({"error": f"{type(error).__name__}: {error}"}).encode()
                status, content_type = HTTPStatus.INTERNAL_SERVER_ERROR, "application/json"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{utc_now()}] {self.address_string()} {format % args}", flush=True)

    return Handler


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live dashboard for dispatch RLVR cells")
    parser.add_argument("--root", type=Path, default=Path("/workspace/runs"),
                        help="phase-directory root (default: /workspace/runs)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--once", action="store_true",
                        help="render one populated HTML snapshot to stdout and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    state = DashboardState(args.root)
    if args.once:
        print(render_html(state.snapshot()))
        return
    print(f"warming dashboard caches under {args.root}", flush=True)
    state.snapshot()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"dashboard listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
