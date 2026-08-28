"""Serve a dependency-free live dashboard for the four-cell GRPO pipeline."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

MAX_STEPS = 256
CELLS = (
    ("public_it-direct_grpo", 0),
    ("charter_graft_it-direct_grpo", 1),
    ("public_it-reasoning_grpo", 2),
    ("charter_graft_it-reasoning_grpo", 3),
)
STEP_TIME_PATTERN = re.compile(r"['\"]step_time['\"]:\s*['\"]([0-9.]+)")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RolloutCache:
    """Incrementally reduce append-only raw rollout JSONL into step metrics."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.identity: tuple[int, int] | None = None
        self.groups: dict[int, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def _reset(self) -> None:
        self.offset = 0
        self.identity = None
        self.groups.clear()

    def update(self) -> None:
        if not self.path.is_file():
            return
        stat = self.path.stat()
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
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                group = self.groups[call]
                boundary = float(row.get("native_boundary_valid", 0) or 0)
                grammar = float(row.get("final_grammar_valid", 0) or 0)
                semantic = float(row.get("semantic_correct", 0) or 0)
                reward = float(row.get("reward", 0) or 0)
                truncated = float(bool(row.get("truncated", False)))
                group["messages"] += 1
                group["reward"] += reward
                group["format_valid"] += float(row.get("format_valid", 0) or 0)
                group["semantic_correct"] += semantic
                group["truncated"] += truncated
                group["no_committed_final"] += float(boundary < 0.5)
                group["invalid_final_grammar"] += float(
                    boundary >= 0.5 and grammar < 0.5
                )
                group["semantic_incorrect"] += float(semantic < 0.5)
                group["valid_correct"] += float(reward >= 0.5)
                group["completion_tokens"] += float(
                    row.get("completion_length", 0) or 0
                )

    def snapshot(self) -> dict[str, Any]:
        self.update()
        history = []
        cumulative: dict[str, float] = defaultdict(float)
        for call, values in sorted(self.groups.items()):
            count = values["messages"]
            if count <= 0:
                continue
            record: dict[str, float | int] = {
                "step": call + 1,
                "messages": int(count),
            }
            for key in (
                "reward",
                "format_valid",
                "semantic_correct",
                "truncated",
                "completion_tokens",
            ):
                record[key] = values[key] / count
            for key in (
                "no_committed_final",
                "invalid_final_grammar",
                "semantic_incorrect",
                "valid_correct",
            ):
                record[key] = int(values[key])
            history.append(record)
            for key, value in values.items():
                cumulative[key] += value
        total = cumulative.get("messages", 0)
        counts = {
            key: int(cumulative.get(key, 0))
            for key in (
                "messages",
                "no_committed_final",
                "invalid_final_grammar",
                "truncated",
                "semantic_incorrect",
                "valid_correct",
            )
        }
        rates = {
            key: (counts[key] / total if total else 0.0)
            for key in counts
            if key != "messages"
        }
        return {
            "step": history[-1]["step"] if history else 0,
            "history": history,
            "cumulative_counts": counts,
            "cumulative_rates": rates,
            "latest": history[-1] if history else None,
        }


class StepTimeCache:
    """Incrementally extract trainer step-time records without rereading logs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.identity: tuple[int, int] | None = None
        self.times: list[float] = []

    def update(self) -> None:
        if not self.path.is_file():
            return
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.identity not in (None, identity) or stat.st_size < self.offset:
            self.offset = 0
            self.times.clear()
        self.identity = identity
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            payload = handle.read()
            self.offset = handle.tell()
        text = payload.decode(errors="replace")
        self.times.extend(float(value) for value in STEP_TIME_PATTERN.findall(text))

    def recent_median(self) -> float | None:
        self.update()
        return statistics.median(self.times[-10:]) if self.times else None


def gpu_inventory() -> list[dict[str, Any]]:
    fields = (
        "index,name,memory.used,memory.total,utilization.gpu,"
        "utilization.memory,power.draw,temperature.gpu"
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    result = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 8:
            continue
        result.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "memory_used_mib": float(values[2]),
                "memory_total_mib": float(values[3]),
                "gpu_utilization": float(values[4]),
                "memory_utilization": float(values[5]),
                "power_watts": float(values[6]),
                "temperature_c": float(values[7]),
            }
        )
    return result


class DashboardState:
    def __init__(self, work_root: Path, run_id: str) -> None:
        self.work_root = work_root
        self.run_id = run_id
        self.training_root = work_root / "training" / run_id
        self.pipeline_root = work_root / "pipeline" / run_id
        self.rollouts = {
            cell: RolloutCache(
                self.training_root
                / "cells"
                / cell
                / "rollouts"
                / "raw_rollouts.rank-0.jsonl"
            )
            for cell, _gpu in CELLS
        }
        self.step_times = {
            cell: StepTimeCache(self.training_root / "logs" / f"{cell}.log")
            for cell, _gpu in CELLS
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def snapshot(self) -> dict[str, Any]:
        gpu_by_index = {gpu["index"]: gpu for gpu in gpu_inventory()}
        cells = []
        for cell, gpu_index in CELLS:
            rollout = self.rollouts[cell].snapshot()
            step = int(rollout["step"])
            seconds_per_step = self.step_times[cell].recent_median()
            eta_seconds = (
                max(0, MAX_STEPS - step) * seconds_per_step
                if seconds_per_step is not None
                else None
            )
            cell_root = self.training_root / "cells" / cell
            failure = self._read_json(cell_root / "RL_FAILURE.json")
            done = self._read_json(cell_root / "RL_DONE.json")
            status = "failed" if failure else "complete" if done else "running"
            cells.append(
                {
                    "name": cell,
                    "gpu_index": gpu_index,
                    "gpu": gpu_by_index.get(gpu_index),
                    "status": status,
                    "failure": failure.get("error") if failure else None,
                    "max_steps": MAX_STEPS,
                    "seconds_per_step": seconds_per_step,
                    "eta_seconds": eta_seconds,
                    **rollout,
                }
            )
        pipeline = self._read_json(self.pipeline_root / "PIPELINE_STATE.json") or {}
        return {
            "generated_at": utc_now(),
            "run_id": self.run_id,
            "pipeline": pipeline,
            "cells": cells,
        }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma 4 native GRPO live dashboard</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141b2d;--muted:#8fa0bd;--line:#27324b;--cyan:#5dd6e8;--gold:#f2bd5a;--green:#5bd18b;--red:#ff6b7a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#eef3ff;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}.wrap{max-width:1500px;margin:auto;padding:24px}.top{display:flex;justify-content:space-between;gap:16px;align-items:end;margin-bottom:18px}h1{font-size:25px;margin:0 0 4px}.muted{color:var(--muted)}.pill{border:1px solid var(--line);border-radius:999px;padding:6px 10px;background:#10172a}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.cell{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:17px;box-shadow:0 10px 30px #0003}.head{display:flex;justify-content:space-between;gap:12px;align-items:start}.name{font-size:17px;font-weight:700}.status{font-size:12px;text-transform:uppercase;letter-spacing:.08em}.running{color:var(--cyan)}.complete{color:var(--green)}.failed{color:var(--red)}.progress{height:8px;background:#232d44;border-radius:99px;overflow:hidden;margin:12px 0 7px}.bar{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));width:0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:13px 0}.stat{background:#0e1527;border-radius:9px;padding:9px}.value{font-size:17px;font-weight:700}.label{font-size:11px;color:var(--muted);margin-top:2px}.gpu{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:10px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.chart{width:100%;height:145px;margin-top:10px;background:#0e1527;border-radius:9px}.legend{display:flex;gap:16px;font-size:11px;color:var(--muted);margin-top:5px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.issues{width:100%;border-collapse:collapse;margin-top:12px;font-size:12px}.issues th,.issues td{text-align:right;padding:6px 5px;border-bottom:1px solid #252e43}.issues th:first-child,.issues td:first-child{text-align:left}.warn{color:var(--gold)}.bad{color:var(--red)}.good{color:var(--green)}.error{background:#411b25;border:1px solid #813146;color:#ffd9de;border-radius:8px;padding:9px;margin-top:10px;white-space:pre-wrap}.foot{margin-top:16px;color:var(--muted);font-size:12px}@media(max-width:950px){.grid{grid-template-columns:1fr}.stats,.gpu{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body><div class="wrap">
<div class="top"><div><h1>Gemma 4 native GRPO</h1><div class="muted" id="run"></div></div><div class="pill" id="updated">Loading…</div></div>
<div class="grid" id="cells"></div>
<div class="foot">“No committed final” means no valid native final-channel boundary. “Bad final grammar” means a final channel existed but did not contain exactly the required one-line Assignment. Recent counts are the latest 32-completion optimizer batch.</div>
</div>
<script>
const pct=x=>`${(100*(x||0)).toFixed(1)}%`;
const num=x=>(x??0).toLocaleString();
const duration=s=>{if(s==null)return '—';let h=Math.floor(s/3600),m=Math.round((s%3600)/60);return h?`${h}h ${m}m`:`${m}m`};
function spark(history){
  const keys=[['reward','#5bd18b'],['format_valid','#5dd6e8'],['truncated','#ff6b7a']];
  const w=600,h=145,left=40,right=588,top=12,bottom=133,n=Math.max(2,history.length);
  let grid='<g fill="#8fa0bd" font-size="10"><text x="4" y="16">100%</text><text x="9" y="76">50%</text><text x="15" y="136">0%</text></g><line x1="40" y1="72.5" x2="588" y2="72.5" stroke="#27324b"/><line x1="40" y1="12" x2="588" y2="12" stroke="#27324b"/><line x1="40" y1="133" x2="588" y2="133" stroke="#27324b"/>';
  let paths=keys.map(([k,c])=>{let pts=history.map((r,i)=>`${left+i*(right-left)/(n-1)},${bottom-(r[k]||0)*(bottom-top)}`).join(' ');return `<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="2.5" vector-effect="non-scaling-stroke"/>`}).join('');
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${grid}${paths}</svg><div class="legend"><span><i class="dot" style="background:#5bd18b"></i>reward</span><span><i class="dot" style="background:#5dd6e8"></i>valid final</span><span><i class="dot" style="background:#ff6b7a"></i>truncated</span></div>`;
}
function issueRow(label,key,c,l,klass=''){let lm=l?l.messages||0:0,lc=key==='messages'?lm:key==='truncated'?Math.round((l?.truncated||0)*lm):l?l[key]||0:0,cr=c.cumulative_rates[key]||0,ct=num(c.cumulative_counts[key]),cum=key==='messages'?ct:`${ct} (${pct(cr)})`;return `<tr><td>${label}</td><td class="${klass}">${cum}</td><td class="${klass}">${key==='messages'?num(lc):`${num(lc)} / ${num(lm)}`}</td></tr>`}
function card(c){
  const g=c.gpu||{},l=c.latest,rate=c.cumulative_rates,progress=100*c.step/c.max_steps;
  return `<section class="cell"><div class="head"><div><div class="name">${c.name}</div><div class="muted">GPU ${c.gpu_index} · ${g.name||'unavailable'}</div></div><div class="status ${c.status}">${c.status}</div></div>
  <div class="progress"><div class="bar" style="width:${progress}%"></div></div><div class="muted">step ${c.step} / ${c.max_steps} · ${c.seconds_per_step?c.seconds_per_step.toFixed(1)+'s/step':'initializing'} · ETA ${duration(c.eta_seconds)}</div>
  <div class="stats"><div class="stat"><div class="value">${l?pct(l.reward):'—'}</div><div class="label">latest reward</div></div><div class="stat"><div class="value">${l?pct(l.format_valid):'—'}</div><div class="label">valid final</div></div><div class="stat"><div class="value">${pct(rate.no_committed_final)}</div><div class="label">cumulative no final</div></div><div class="stat"><div class="value">${l?pct(l.truncated):'—'}</div><div class="label">latest truncated</div></div></div>
  <div class="gpu"><div><div class="value">${g.gpu_utilization??'—'}%</div><div class="label">GPU util</div></div><div><div class="value">${g.memory_used_mib?`${(g.memory_used_mib/1024).toFixed(1)} GB`:'—'}</div><div class="label">VRAM / ${g.memory_total_mib?(g.memory_total_mib/1024).toFixed(0):'—'} GB</div></div><div><div class="value">${g.power_watts?g.power_watts.toFixed(0)+' W':'—'}</div><div class="label">power</div></div><div><div class="value">${g.temperature_c??'—'}°C</div><div class="label">temperature</div></div></div>
  ${spark(c.history)}
  <table class="issues"><thead><tr><th>Outcome</th><th>Cumulative</th><th>Latest batch</th></tr></thead><tbody>
  ${issueRow('Messages','messages',c,l)}${issueRow('No committed final','no_committed_final',c,l,'bad')}${issueRow('Bad final grammar','invalid_final_grammar',c,l,'warn')}${issueRow('Truncated','truncated',c,l,'bad')}${issueRow('Semantically incorrect','semantic_incorrect',c,l,'warn')}${issueRow('Valid + correct','valid_correct',c,l,'good')}</tbody></table>${c.failure?`<div class="error">${c.failure}</div>`:''}</section>`;
}
async function refresh(){try{let r=await fetch('/api/status',{cache:'no-store'}),d=await r.json();document.getElementById('run').textContent=`${d.run_id} · pipeline: ${d.pipeline.stage||d.pipeline.status||'starting'}`;document.getElementById('updated').textContent=`updated ${new Date(d.generated_at).toLocaleTimeString()}`;document.getElementById('cells').innerHTML=d.cells.map(card).join('')}catch(e){document.getElementById('updated').textContent=`refresh failed: ${e}`}}
refresh();setInterval(refresh,5000);
</script></body></html>"""


def make_handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path.split("?", 1)[0] == "/api/status":
                try:
                    body = json.dumps(state.snapshot(), separators=(",", ":")).encode()
                    status = HTTPStatus.OK
                except Exception as error:  # pragma: no cover - runtime diagnostics
                    body = json.dumps(
                        {"error": f"{type(error).__name__}: {error}"}
                    ).encode()
                    status = HTTPStatus.INTERNAL_SERVER_ERROR
                content_type = "application/json"
            elif self.path.split("?", 1)[0] in ("/", "/index.html"):
                body = HTML.encode()
                status = HTTPStatus.OK
                content_type = "text/html; charset=utf-8"
            else:
                body = b"not found\n"
                status = HTTPStatus.NOT_FOUND
                content_type = "text/plain; charset=utf-8"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    state = DashboardState(args.work_root.resolve(), args.run_id)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(
        f"dashboard listening on http://{args.host}:{args.port} for {args.run_id}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
