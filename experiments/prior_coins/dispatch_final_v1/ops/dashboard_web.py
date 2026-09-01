#!/usr/bin/env python3
"""Browser front end for the dispatch final-v1 campaign dashboard.

Same data, same read-only contract as the TUI — it *imports* dashboard.py and
uses its collectors, so there is exactly one implementation of "what is going
on" and the two views cannot drift.

    cd /workspace/scimt-dispatch-final && \
      uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/ops/dashboard_web.py

Then, from your laptop:

    ssh -N -L 8377:127.0.0.1:8377 <this-box>      # and open http://127.0.0.1:8377/

BINDING: loopback only, always.  The page shows account balances, pod ids and
ssh-reachable hostnames; it must never be served on a public interface, so
there is deliberately no --host flag and the bind address is asserted.

READ-ONLY CONTRACT (inherited from dashboard.py, unchanged):
  * files are only ever opened for reading;
  * the only subprocesses are the two read-only ssh calls (the inline detail
    probe fed to `bash -s`, and `cat /etc/runpod-deadman.json`);
  * the only outbound HTTP is the RunPod GraphQL *query*;
  * no API key material is ever placed in the JSON payload or the HTML.  The
    payload is built field-by-field from the snapshot below — balances, pod
    ids, rates, phases — and never echoes a header, env var or key file.

Endpoints:
    GET /            self-contained HTML page (inline CSS + JS, no CDN)
    GET /data.json   the latest snapshot, with generated_at + age_seconds
    GET /healthz     "ok" plus the snapshot age, for scripted checks
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Import the TUI module as the single source of collection logic.  Bytecode
# writing is switched off first: importing a sibling module would otherwise
# drop a __pycache__ entry into the ops dir, and this tool writes nothing there.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dashboard as tui  # noqa: E402

BIND_HOST = "127.0.0.1"  # loopback only, by construction
DEFAULT_PORT = 8377
STALE_AFTER_S = 180  # the page shouts STALE past this

# Pipeline order from ops/probe_unit.sh.  "aft" is one segment that the probe
# reports as aft:N/4, so N/4 lands the bar inside that segment; "finalize" is
# the probe's post-publish, pre-CHAIN_COMPLETE state.
PHASE_SEQUENCE = [
    "mix", "midtrain", "dolci", "aft", "eval", "recall", "d4",
    "costsweep", "publish", "finalize", "done",
]
PHASE_SPAN = len(PHASE_SEQUENCE) - 1  # "done" == 1.0


def phase_progress(phase: str, detail: "tui.ArmDetail | None" = None) -> float | None:
    """'aft:2/4' -> 0.35, 'mix' -> 0.0, 'done' -> 1.0, unknown -> None.

    With a detail payload the bar advances *within* the active stage's segment
    on real step counts (tui.stage_fraction); without one it falls back to the
    coarse aft:N/4 reading the probe line has always carried.
    """
    if not phase:
        return None
    head, _, tail = phase.partition(":")
    if head not in PHASE_SEQUENCE:
        return None
    idx = float(PHASE_SEQUENCE.index(head))
    frac = tui.stage_fraction(phase, detail)
    if frac is not None:
        idx += frac
    elif head == "aft" and "/" in tail:
        done_str, _, total_str = tail.partition("/")
        try:
            done, total = float(done_str), float(total_str)
            if total > 0:
                idx += max(0.0, min(1.0, done / total))
        except ValueError:
            pass
    return round(idx / PHASE_SPAN, 4)


def serialize_detail(phase: str, detail: "tui.ArmDetail | None") -> dict | None:
    """ArmDetail -> the additive `detail` object on a phase row, or None.

    Nullable throughout: pods differ in what exists, and every consumer treats a
    missing field as "not known" rather than as zero.
    """
    if detail is None:
        return None
    eta = tui.stage_eta_seconds(phase, detail)
    return {
        "stage": detail.stage or None,
        "step": detail.step,
        "total": detail.total,
        "sit": detail.sit,
        "files": detail.files,
        "data_age": detail.data_age,
        "cells_done": detail.cells_done,
        "cell_total": tui.AFT_CELLS if detail.stage == "aft" else None,
        "cells": [
            {"name": c.name, "step": c.step, "total": c.total, "sit": c.sit,
             "eta_seconds": None if c.eta_seconds is None else round(c.eta_seconds, 1)}
            for c in detail.cells
        ],
        "stage_fraction": tui.stage_fraction(phase, detail),
        "eta_seconds": None if eta is None else round(eta, 1),
        "text": tui.arm_progress_text(phase, detail) or None,
    }


def parse_phases(unit: "tui.UnitView") -> list[dict]:
    """Per-arm phase list: [{arm, phase, progress, detail}] in the arm order."""
    arms = [a for a in unit.rec.arms.split(",") if a]
    if unit.probe is None:
        return [{"arm": a, "phase": None, "progress": None, "detail": None} for a in arms]
    if not unit.probe.ok:
        return [{"arm": a, "phase": unit.probe.phases, "progress": None, "detail": None}
                for a in arms]
    seen: dict[str, str] = {}
    for chunk in unit.probe.phases.split(","):
        arm, sep, phase = chunk.partition(":")
        if sep:
            seen[arm.strip()] = phase.strip()
    rows = []
    for arm in arms:
        phase = seen.get(arm)
        detail = unit.probe.details.get(arm)
        rows.append({
            "arm": arm,
            "phase": phase,
            "progress": phase_progress(phase or "", detail),
            "detail": serialize_detail(phase or "", detail),
        })
    return rows


def serialize(snap: "tui.Snapshot") -> dict:
    """Snapshot -> plain JSON-able dict.  Built field by field; no secrets."""
    now = time.time()
    claimed = tui.claimed_pod_ids(snap)
    claimed_ids = set(claimed)

    accounts = []
    for view in snap.accounts:
        roll = tui.rollup_account(view, claimed_ids)
        accounts.append({
            "label": roll.label,
            "balance": roll.balance,
            "burn": round(roll.burn, 4),
            "runway_hours": None if roll.runway_hours is None else round(roll.runway_hours, 2),
            "pod_count": len(roll.our_pods),
            "external": [
                {"id": p.get("id"), "name": p.get("name"),
                 "cost": float(p.get("costPerHr") or 0.0)}
                for p in roll.external_pods
            ],
            "external_cost": round(roll.external_cost, 4),
            "unclaimed": [
                {"id": p.get("id"), "name": p.get("name"),
                 "cost": float(p.get("costPerHr") or 0.0)}
                for p in roll.unclaimed
            ],
            # Non-dfv1 pods on the account (another agent's errand): reported,
            # not alarmed on -- see tui.rollup_account.
            "noncampaign": [
                {"id": p.get("id"), "name": p.get("name"),
                 "cost": float(p.get("costPerHr") or 0.0)}
                for p in roll.noncampaign
            ],
            "noncampaign_cost": round(roll.noncampaign_cost, 4),
            "campaigns": [c.campaign_id for c in snap.campaigns
                          if c.cfg.account == roll.label],
            "error": roll.error,
        })

    campaigns, units, queue = [], [], []
    for view in snap.campaigns:
        pending, held = tui.split_queue(view)
        active = [u for u in view.units if u.rec.state in tui.ACTIVE_STATES]
        campaigns.append({
            "campaign": view.campaign_id,
            "account": view.cfg.account,
            "active": len(active),
            "rate": round(sum(u.rec.hourly_rate for u in active), 2),
            "queued": len(pending),
            "held": len(held),
            "supervisor": {
                "pid": view.supervisor.pid,
                "alive": view.supervisor.alive,
                "note": view.supervisor.note,
            },
            "notes": list(view.notes),
            "alerts": [
                {"when": a.when, "message": a.message, "count": a.count}
                for a in view.supervisor.alerts
            ],
        })

        ordered = (active + [u for u in view.units if u.rec.state not in tui.ACTIVE_STATES])
        for unit in ordered:
            rec = unit.rec
            hours = tui.elapsed_hours(rec, now)
            expected = tui.EXPECTED_HOURS.get(rec.profile)
            deadline_hours = None
            if unit.deadline:
                dl = tui.parse_utc(unit.deadline)
                if dl is not None:
                    deadline_hours = round((dl.timestamp() - now) / 3600.0, 2)
            units.append({
                "campaign": view.campaign_id,
                "account": view.cfg.account,
                "profile": rec.profile,
                "arms": rec.arms,
                "state": rec.state,
                "active": rec.state in tui.ACTIVE_STATES,
                "attention": rec.state in tui.ATTENTION_STATES,
                "pod_id": rec.pod_id,
                "rate": rec.hourly_rate,
                "attempt": rec.attempt,
                "created_at": rec.created_at,
                "elapsed_hours": None if hours is None else round(hours, 2),
                "expected_hours": expected,
                "probe": None if unit.probe is None else {
                    "ok": unit.probe.ok,
                    "runner_state": unit.probe.runner_state,
                    "procs": unit.probe.procs,
                    "log_age": unit.probe.log_age,
                    "error": unit.probe.error,
                },
                "phases": parse_phases(unit),
                "deadline_utc": unit.deadline,
                "deadline_hours": deadline_hours,
            })

        for row in pending + held:
            queue.append({
                "campaign": view.campaign_id,
                "account": view.cfg.account,
                "priority": row.priority,
                "profile": row.profile,
                "arms": row.arms,
                "rate": row.hourly_rate,
                "held": row.held,
            })

    orphans = []
    for view in snap.accounts:
        for pod in view.state.pods:
            if pod.get("desiredStatus") != "RUNNING" or not tui.is_ours(pod):
                continue
            if pod.get("id") in claimed_ids:
                continue
            orphans.append({
                "account": view.cfg.label, "id": pod.get("id"), "name": pod.get("name"),
                "cost": float(pod.get("costPerHr") or 0.0),
            })

    return {
        "generated_ts": snap.taken_at,
        "generated_at": datetime.fromtimestamp(snap.taken_at, timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "accounts": accounts,
        "campaigns": campaigns,
        "units": units,
        "queue": queue,
        "orphans": orphans,
        "errors": list(snap.errors),
    }


class SnapshotStore:
    """Holds the latest serialized snapshot for the HTTP handlers to read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payload: dict | None = None
        self._last_error: str | None = None

    def set(self, payload: dict, error: str | None) -> None:
        with self._lock:
            self._payload = payload
            self._last_error = error

    def get(self) -> dict:
        with self._lock:
            payload = dict(self._payload) if self._payload else None
            error = self._last_error
        if payload is None:
            return {
                "generated_ts": None, "generated_at": None, "age_seconds": None,
                "accounts": [], "campaigns": [], "units": [], "queue": [], "orphans": [],
                "errors": [e for e in [error] if e],
                "collector_error": error, "warming_up": True,
            }
        payload["age_seconds"] = round(time.time() - (payload.get("generated_ts") or 0), 1)
        payload["collector_error"] = error
        payload["warming_up"] = False
        return payload


class Refresher(threading.Thread):
    """One background collection every --interval seconds, same as the TUI."""

    def __init__(self, store: SnapshotStore, interval: int) -> None:
        super().__init__(name="refresher", daemon=True)
        self.store = store
        self.interval = interval
        self.stop_event = threading.Event()
        self.collector = tui.Collector()

    def refresh_once(self) -> dict | None:
        try:
            snap = self.collector.collect()
        except Exception as exc:  # noqa: BLE001 - a bad refresh must not kill the thread
            self.store.set(self.store.get(), f"{type(exc).__name__}: {exc}"[:200])
            return None
        payload = serialize(snap)
        self.store.set(payload, None)
        return payload

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.refresh_once()
            self.stop_event.wait(self.interval)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dispatch final-v1</title>
<style>
:root{
  --bg:#0e1116; --panel:#161b22; --panel2:#1c2230; --line:#2a3038;
  --fg:#d8dee6; --dim:#8b96a5; --green:#3fb950; --amber:#d29922;
  --red:#f85149; --blue:#58a6ff; --violet:#a371f7;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
a{color:var(--blue)}
header{position:sticky;top:0;z-index:9;background:linear-gradient(180deg,#0e1116 70%,#0e111600);
  padding:14px 18px 10px;display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
h1{font-size:15px;letter-spacing:.14em;margin:0;color:#fff;text-transform:uppercase}
.muted{color:var(--dim)}
main{padding:0 18px 32px;display:flex;flex-direction:column;gap:16px}
section>h2{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--dim);
  margin:0 0 8px;font-weight:600}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.card{padding:14px 16px}
.card .acct{font-size:12px;letter-spacing:.18em;color:var(--dim)}
.card .bal{font-size:30px;font-weight:600;color:#fff;margin:2px 0 10px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font-size:12px}
.kv dt{color:var(--dim)} .kv dd{margin:0;text-align:right}
.strip{display:flex;gap:10px;flex-wrap:wrap}
.chip{padding:9px 13px;display:flex;align-items:center;gap:10px;font-size:12px}
.chip b{color:#fff;font-weight:600;letter-spacing:.06em}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;
  border:1px solid transparent;white-space:nowrap}
.pill.ok{color:var(--green);background:#3fb9501f;border-color:#3fb95044}
.pill.bad{color:var(--red);background:#f851491f;border-color:#f8514944}
.pill.warn{color:var(--amber);background:#d299221f;border-color:#d2992244}
.pill.dim{color:var(--dim);background:#8b96a516;border-color:#8b96a52e}
.pill.info{color:var(--blue);background:#58a6ff1a;border-color:#58a6ff3d}
table{width:100%;border-collapse:collapse}
th{font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);
  text-align:left;font-weight:600;padding:9px 10px;border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid #21262d;vertical-align:middle}
tr:last-child td{border-bottom:0}
tbody tr:hover{background:#ffffff06}
.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{position:relative;height:19px;border-radius:4px;background:#0b0e13;
  border:1px solid var(--line);overflow:hidden;min-width:130px}
.bar>i{position:absolute;inset:0 auto 0 0;background:#58a6ff2e;
  border-right:1px solid #58a6ff8c}
.bar.over>i{background:#d299222e;border-right-color:#d29922a6}
.bar>span{position:relative;display:block;padding:1px 7px;font-size:11px;
  white-space:nowrap;color:var(--fg)}
.arms{display:flex;flex-direction:column;gap:5px;min-width:290px}
.arm{display:grid;grid-template-columns:58px 1fr;gap:8px;align-items:start}
.arm .nm{color:var(--dim);font-size:11px;overflow:hidden;text-overflow:ellipsis;
  padding-top:1px}
.armcol{display:flex;flex-direction:column;gap:2px;min-width:0}
.armsub{font-size:10.5px;color:var(--dim);white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;letter-spacing:.02em}
.armsub b{color:var(--fg);font-weight:600}
.mini{position:relative;height:15px;border-radius:3px;background:#0b0e13;
  border:1px solid var(--line);overflow:hidden}
.mini>i{position:absolute;inset:0 auto 0 0;background:#3fb95036;
  border-right:1px solid #3fb95099}
.mini.done>i{background:#3fb9504d;border-right-color:#3fb950}
.mini.unknown{border-style:dashed;border-color:#f8514966}
.mini>span{position:relative;display:block;padding:0 6px;font-size:10.5px;
  letter-spacing:.03em;white-space:nowrap;color:var(--fg)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:16px;
  align-items:start}
.banner{padding:9px 13px;border-radius:6px;font-size:12px}
.banner.bad{background:#f851491c;border:1px solid #f8514955;color:#ffb4ae}
.banner.warn{background:#d299221c;border:1px solid #d2992255;color:#f3d08a}
.stale{font-size:12px;font-weight:600;letter-spacing:.1em;padding:3px 10px;border-radius:999px}
.stale.ok{color:var(--green);background:#3fb9501f}
.stale.bad{color:#fff;background:var(--red)}
.alerts{display:flex;flex-direction:column}
.alert{display:grid;grid-template-columns:112px 44px 1fr;gap:10px;padding:8px 12px;
  border-bottom:1px solid #21262d;font-size:12px}
.alert:last-child{border-bottom:0}
.alert .when{color:var(--dim);font-size:11px}
.empty{padding:12px;color:var(--dim);font-size:12px}
.mono-dim{color:var(--dim);font-size:11.5px}
</style></head>
<body>
<header>
  <h1>Dispatch final-v1</h1>
  <span id="stale" class="stale ok">--</span>
  <span class="muted">snapshot <b id="gen">--</b></span>
  <span class="muted">refetch in <span id="countdown">--</span>s</span>
  <span class="muted" id="fetchnote"></span>
</header>
<main>
  <div id="banners"></div>
  <section><h2>Accounts</h2><div id="accounts" class="cards"></div></section>
  <section><h2>Campaigns &amp; supervisors</h2><div id="campaigns" class="strip"></div></section>
  <section><h2>Work units</h2><div class="panel"><table id="units"></table></div></section>
  <div class="grid2">
    <section><h2>Queued / held</h2><div class="panel"><table id="queue"></table></div></section>
    <section><h2>Recent supervisor alerts</h2>
      <div class="panel alerts" id="alerts"></div></section>
  </div>
</main>
<script>
"use strict";
var REFETCH_MS = 30000, STALE_S = 180;
var data = null, fetchedAt = 0, nextAt = 0, fetchError = null;

function el(tag, cls, text){var e=document.createElement(tag);
  if(cls)e.className=cls; if(text!==undefined&&text!==null)e.textContent=String(text); return e;}
function money(v){return v===null||v===undefined?"?":"$"+Number(v).toLocaleString(undefined,
  {minimumFractionDigits:2,maximumFractionDigits:2});}
function num(v,d){return v===null||v===undefined?"?":Number(v).toFixed(d===undefined?2:d);}
function pct(x){return Math.max(0,Math.min(1,x))*100;}

function ageSeconds(){
  if(!data||data.age_seconds===null||data.age_seconds===undefined) return null;
  return data.age_seconds + (Date.now()-fetchedAt)/1000;   // server age + client wait
}

var STATE_CLASS = {running:"ok", done:"dim", cleaned:"dim",
  parked:"bad", lost:"bad", halted:"bad", provision_failed:"bad"};
function stateClass(s){return STATE_CLASS[s] || "warn";}

function bar(frac, label, over){
  var b = el("div","bar"+(over?" over":""));
  var fill = el("i"); fill.style.width = pct(frac)+"%"; b.appendChild(fill);
  b.appendChild(el("span", null, label));
  return b;
}

function has(v){ return v !== null && v !== undefined; }

// Coarse, honest duration — mirrors dashboard.fmt_duration.
function dur(s){
  if(!has(s) || s < 0) return "?";
  s = Math.floor(s);
  if(s < 90) return s+"s";
  if(s < 3600) return Math.floor(s/60)+"m";
  var h = Math.floor(s/3600), m = Math.floor((s%3600)/60);
  return h+"h"+(m<10?"0":"")+m+"m";
}

// One line of intra-stage evidence under the bar, plus the long-form tooltip.
// Every field is optional: a pod that reports none of them renders exactly the
// phase-only row this page showed before the detail payload existed.
function armDetailText(p){
  var d = p.detail;
  if(!d) return {sub:"", tip:""};
  var bits = [], tip = [];
  if(has(d.step) && d.total){
    bits.push(d.step+"/"+d.total);
    if(d.sit) bits.push(num(d.sit,2)+"s/it");
    tip.push("step "+d.step+" of "+d.total+(d.sit?" at "+num(d.sit,2)+" s/it":""));
  } else if(d.cells && d.cells.length){
    var live = d.cells.filter(function(c){return has(c.step) && c.total;});
    if(live.length){
      var steps = live.map(function(c){return c.step;});
      var lo = Math.min.apply(null,steps), hi = Math.max.apply(null,steps);
      bits.push((live.length>1?live.length+" cells ":"")+
        (lo===hi?lo:lo+"-"+hi)+"/"+live[0].total);
      var sits = live.map(function(c){return c.sit;}).filter(Boolean);
      if(sits.length) bits.push(num(sits.reduce(function(a,b){return a+b;},0)/sits.length,2)+"s/it");
      live.forEach(function(c){
        tip.push(c.name+": "+c.step+"/"+c.total+(c.sit?" at "+num(c.sit,2)+" s/it":"")+
          (has(c.eta_seconds)?", "+dur(c.eta_seconds)+" left":""));
      });
    }
    if(has(d.cells_done) && d.cell_total)
      tip.push(d.cells_done+" of "+d.cell_total+" cells complete");
  } else if(has(d.files)){
    bits.push(d.files+" files");
    tip.push(d.files+" response .jsonl files written so far (no expected total is "+
      "derivable, so this is a raw count)");
  } else if(has(d.data_age)){
    bits.push("newest data "+dur(d.data_age)+" ago");
    tip.push("newest file in the arm's data/ dir was written "+dur(d.data_age)+" ago");
  }
  if(has(d.eta_seconds)){
    bits.push("stage ETA "+dur(d.eta_seconds));
    tip.push("stage ETA "+dur(d.eta_seconds)+
      " = remaining steps x s/it for the work in flight (this stage only, "+
      "not the rest of the row)");
  }
  return {sub: bits.join(" · "), tip: tip.join("\n")};
}

function armBars(u){
  var wrap = el("div","arms");
  (u.phases||[]).forEach(function(p){
    var row = el("div","arm");
    row.appendChild(el("div","nm",p.arm));
    var col = el("div","armcol");
    var known = has(p.progress);
    var m = el("div","mini"+(known?(p.progress>=1?" done":""):" unknown"));
    var fill = el("i"); fill.style.width = known ? pct(p.progress)+"%" : "0%";
    m.appendChild(fill);
    m.appendChild(el("span",null,p.phase || (u.probe?"?":"not probed")));
    var det = armDetailText(p);
    if(det.tip) m.title = p.arm+" · "+(p.phase||"?")+"\n"+det.tip;
    col.appendChild(m);
    if(det.sub) col.appendChild(el("div","armsub",det.sub));
    row.appendChild(col); wrap.appendChild(row);
  });
  if(!wrap.childNodes.length) wrap.appendChild(el("div","mono-dim","-"));
  return wrap;
}

function renderAccounts(){
  var host = document.getElementById("accounts"); host.textContent = "";
  (data.accounts||[]).forEach(function(a){
    var c = el("div","panel card");
    c.appendChild(el("div","acct", a.label + "  ·  " + (a.campaigns.join(", ")||"no campaigns")));
    c.appendChild(el("div","bal", a.error ? "?" : money(a.balance)));
    var dl = el("dl","kv");
    function kv(k,v,cls){ dl.appendChild(el("dt",null,k));
      var d=el("dd",cls,v); dl.appendChild(d); return d; }
    kv("pod burn", a.error?"?":"$"+num(a.burn)+"/hr");
    var rw = a.runway_hours===null||a.runway_hours===undefined ? "n/a" : num(a.runway_hours,1)+"h";
    var d = kv("runway", a.error?"?":rw);
    if(a.runway_hours!==null && a.runway_hours!==undefined && a.runway_hours<8){
      d.style.color="var(--red)"; d.style.fontWeight="700";}
    kv("our pods", a.error?"?":a.pod_count);
    if(a.external && a.external.length)
      kv("external", a.external.length+" ($"+num(a.external_cost)+"/hr)");
    if(a.noncampaign && a.noncampaign.length)
      kv("non-campaign", a.noncampaign.length+" ($"+num(a.noncampaign_cost)+"/hr)");
    c.appendChild(dl);
    if(a.error) c.appendChild(el("div","banner bad", a.error));
    // A dfv1- pod no ledger claims is campaign-shaped spend nobody owns: red.
    (a.unclaimed||[]).forEach(function(p){
      c.appendChild(el("div","banner bad",
        "unclaimed: "+p.name+" ($"+num(p.cost)+"/hr) — a dfv1 pod with no ledger row"));
    });
    // Anything not named dfv1-* is somebody else's errand on the same account:
    // worth showing (it spends the balance) but not an alarm.
    (a.noncampaign||[]).forEach(function(p){
      c.appendChild(el("div","banner warn",
        "non-campaign pod: "+p.name+" ($"+num(p.cost)+"/hr) — not part of this campaign"));
    });
    host.appendChild(c);
  });
}

function renderCampaigns(){
  var host = document.getElementById("campaigns"); host.textContent = "";
  (data.campaigns||[]).forEach(function(c){
    var chip = el("div","panel chip");
    chip.appendChild(el("b", null, c.campaign));
    chip.appendChild(el("span","pill dim", c.account));
    var s = c.supervisor||{};
    chip.appendChild(el("span","pill "+(s.alive?"ok":"bad"),
      s.alive ? "supervisor "+s.pid+" alive" : (s.pid? "pid "+s.pid+" DEAD":"no supervisor")));
    chip.appendChild(el("span","pill "+(c.active?"info":"dim"),
      c.active+" active · $"+num(c.rate)+"/hr"));
    if(c.queued) chip.appendChild(el("span","pill dim", c.queued+" queued"));
    if(c.held) chip.appendChild(el("span","pill warn", c.held+" held"));
    var note = (c.notes||[]).concat(s.note?[s.note]:[]).join("; ");
    if(note) chip.appendChild(el("span","mono-dim", note));
    host.appendChild(chip);
  });
  if(!host.childNodes.length) host.appendChild(el("div","empty","(no campaigns discovered)"));
}

function renderUnits(){
  var t = document.getElementById("units"); t.textContent = "";
  var cols = ["campaign","profile","state","pod","$/hr","att","elapsed / expected",
              "runner","per-arm phase","dead-man"];
  var thead = el("thead"), hr = el("tr");
  cols.forEach(function(c,i){ var th=el("th",null,c);
    if(i===4||i===5) th.className="num"; hr.appendChild(th); });
  thead.appendChild(hr); t.appendChild(thead);
  var tb = el("tbody");
  (data.units||[]).forEach(function(u){
    var tr = el("tr");
    tr.appendChild(el("td","mono-dim",u.campaign));
    var pf = el("td"); pf.appendChild(el("div",null,u.profile));
    pf.appendChild(el("div","mono-dim",u.arms)); tr.appendChild(pf);
    var st = el("td"); st.appendChild(el("span","pill "+stateClass(u.state), u.state));
    tr.appendChild(st);
    tr.appendChild(el("td","mono-dim", u.pod_id || "-"));
    tr.appendChild(el("td","num", num(u.rate)));
    var at = el("td","num", u.attempt);
    if(u.attempt > 1) at.style.color = "var(--amber)";
    tr.appendChild(at);
    var td = el("td");
    if(u.elapsed_hours === null || u.elapsed_hours === undefined){
      td.appendChild(el("span","mono-dim","?"));
    } else {
      var exp = u.expected_hours, frac = exp ? u.elapsed_hours/exp : 0;
      td.appendChild(bar(frac, num(u.elapsed_hours,1)+"h / ~"+(exp?num(exp,1)+"h":"?"),
                         exp && u.elapsed_hours > exp));
    }
    tr.appendChild(td);
    var rn = el("td"), p = u.probe;
    if(!p){ rn.appendChild(el("span","mono-dim","-")); }
    else if(!p.ok){ rn.appendChild(el("span","pill bad","unreachable"));
      rn.appendChild(el("div","mono-dim", p.error||"")); }
    else {
      var bad = p.runner_state==="FAILED" || p.runner_state==="UNKNOWN";
      rn.appendChild(el("span","pill "+(bad?"bad":(p.procs==="0"?"warn":"ok")), p.runner_state));
      rn.appendChild(el("div","mono-dim","p"+p.procs+" · log "+
        (p.log_age===null||p.log_age===undefined?"?":(p.log_age<0?"none":p.log_age+"s"))));
    }
    tr.appendChild(rn);
    var ph = el("td"); ph.appendChild(armBars(u)); tr.appendChild(ph);
    var dm = el("td");
    if(u.deadline_hours === null || u.deadline_hours === undefined){
      dm.appendChild(el("span","mono-dim", u.deadline_utc || "-"));
    } else {
      dm.appendChild(el("span","pill "+(u.deadline_hours<1.5?"bad":
        (u.deadline_hours<3?"warn":"dim")), num(u.deadline_hours,1)+"h"));
    }
    tr.appendChild(dm);
    if(u.attention) tr.style.background = "#f851490f";
    tb.appendChild(tr);
  });
  if(!tb.childNodes.length){
    var tr = el("tr"), td = el("td","empty","(no units in any campaign ledger)");
    td.colSpan = cols.length; tr.appendChild(td); tb.appendChild(tr);
  }
  t.appendChild(tb);
}

function renderQueue(){
  var t = document.getElementById("queue"); t.textContent = "";
  var cols = ["campaign","acc","pri","profile","arms","$/hr","status"];
  var thead = el("thead"), hr = el("tr");
  cols.forEach(function(c){ hr.appendChild(el("th",null,c)); });
  thead.appendChild(hr); t.appendChild(thead);
  var tb = el("tbody");
  (data.queue||[]).forEach(function(q){
    var tr = el("tr");
    tr.appendChild(el("td","mono-dim",q.campaign));
    tr.appendChild(el("td",null,q.account));
    tr.appendChild(el("td","mono-dim",q.priority));
    tr.appendChild(el("td",null,q.profile));
    tr.appendChild(el("td","mono-dim",q.arms));
    tr.appendChild(el("td","num",num(q.rate)));
    var st = el("td"); st.appendChild(el("span","pill "+(q.held?"warn":"dim"),
      q.held?"HELD":"queued")); tr.appendChild(st);
    tb.appendChild(tr);
  });
  if(!tb.childNodes.length){
    var tr = el("tr"), td = el("td","empty","(queues drained)");
    td.colSpan = cols.length; tr.appendChild(td); tb.appendChild(tr);
  }
  t.appendChild(tb);
}

function renderAlerts(){
  var host = document.getElementById("alerts"); host.textContent = "";
  var any = false;
  (data.campaigns||[]).forEach(function(c){
    (c.alerts||[]).forEach(function(a){
      any = true;
      var row = el("div","alert");
      row.appendChild(el("div","when",(a.when||"-")+" "+c.campaign));
      row.appendChild(el("div","when", a.count>1 ? "x"+a.count : ""));
      var bad = /PARKED|HALTED|FATAL|POD LOST/.test(a.message);
      var good = /DURABLE COMPLETE|Hub verified/.test(a.message);
      var m = el("div",null,a.message);
      m.style.color = bad ? "var(--red)" : (good ? "var(--green)" : "var(--amber)");
      row.appendChild(m); host.appendChild(row);
    });
  });
  if(!any) host.appendChild(el("div","empty",
    "(no PARKED / HALTED / FATAL / POD LOST / DURABLE COMPLETE / CAP GATE / Hub-verified "+
    "lines in the recent log tails)"));
}

function renderBanners(){
  var host = document.getElementById("banners"); host.textContent = "";
  if(fetchError) host.appendChild(el("div","banner bad",
    "cannot reach the collector: "+fetchError+" (showing the last good snapshot)"));
  if(data && data.warming_up) host.appendChild(el("div","banner warn",
    "collector is warming up — first snapshot not ready yet"));
  if(data && data.collector_error) host.appendChild(el("div","banner bad",
    "last refresh failed: "+data.collector_error));
  (data && data.orphans || []).forEach(function(o){
    host.appendChild(el("div","banner bad", "! "+o.account+" pod "+o.id+" "+o.name+" $"+
      num(o.cost)+"/hr is running but no campaign ledger claims it"));
  });
  if(data && data.errors && data.errors.length) host.appendChild(
    el("div","banner warn","degraded: "+data.errors.join("; ")));
}

function renderHeader(){
  var age = ageSeconds();
  var badge = document.getElementById("stale");
  if(age === null){ badge.className = "stale bad"; badge.textContent = "NO DATA"; }
  else if(age > STALE_S){ badge.className = "stale bad";
    badge.textContent = "STALE (" + Math.round(age) + "s)"; }
  else { badge.className = "stale ok"; badge.textContent = "LIVE (" + Math.round(age) + "s)"; }
  document.getElementById("gen").textContent = (data && data.generated_at) || "--";
  document.getElementById("countdown").textContent =
    Math.max(0, Math.round((nextAt - Date.now())/1000));
}

function renderAll(){
  renderHeader(); renderBanners();
  if(!data) return;
  renderAccounts(); renderCampaigns(); renderUnits(); renderQueue(); renderAlerts();
}

function load(){
  fetch("data.json", {cache:"no-store"}).then(function(r){
    if(!r.ok) throw new Error("HTTP "+r.status);
    return r.json();
  }).then(function(j){
    data = j; fetchedAt = Date.now(); fetchError = null;
    document.getElementById("fetchnote").textContent = "";
    renderAll();
  }).catch(function(e){
    fetchError = e.message || String(e);
    document.getElementById("fetchnote").textContent = "fetch failed";
    renderAll();
  }).then(function(){
    nextAt = Date.now() + REFETCH_MS;
    setTimeout(load, REFETCH_MS);
  });
}
nextAt = Date.now() + REFETCH_MS;
setInterval(renderHeader, 1000);
load();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "dfv1-dashboard"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    store: SnapshotStore  # set on the class by serve()

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        # Loopback-only page with no third-party assets; keep it that way.
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; style-src 'unsafe-inline'; "
                         "script-src 'unsafe-inline'; connect-src 'self'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/data.json":
            body = json.dumps(self.store.get(), separators=(",", ":")).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        elif path == "/healthz":
            snap = self.store.get()
            age = snap.get("age_seconds")
            body = (f"ok units={len(snap.get('units') or [])} "
                    f"age={'?' if age is None else age}s\n").encode("utf-8")
            self._send(200, body, "text/plain; charset=utf-8")
        else:
            self._send(404, b"not found\n", "text/plain; charset=utf-8")

    do_HEAD = do_GET

    def log_message(self, fmt, *args):  # noqa: ANN001 - silence per-request stderr spam
        return


def summarize(payload: dict) -> str:
    accounts = payload.get("accounts") or []
    units = payload.get("units") or []
    running = [u for u in units if u.get("state") == "running"]
    queued = [q for q in payload.get("queue") or [] if not q.get("held")]
    held = [q for q in payload.get("queue") or [] if q.get("held")]
    bal = " ".join(
        f"{a['label']}={'?' if a.get('error') else tui.fmt_money(a.get('balance'))}"
        f"@${a.get('burn', 0):.2f}/hr"
        for a in accounts
    )
    return (
        f"OK {payload.get('generated_at')} | {bal} | "
        f"{len(payload.get('campaigns') or [])} campaigns | "
        f"{len(units)} units ({len(running)} running) | "
        f"{len(queued)} queued, {len(held)} held | "
        f"{len(payload.get('orphans') or [])} unclaimed pods | "
        f"{len(payload.get('errors') or [])} degraded sources"
    )


def serve(port: int, interval: int) -> int:
    store = SnapshotStore()
    refresher = Refresher(store, interval)
    Handler.store = store

    try:
        httpd = ThreadingHTTPServer((BIND_HOST, port), Handler)
    except OSError as exc:
        print(f"cannot bind {BIND_HOST}:{port} - {exc}", file=sys.stderr)
        return 1
    httpd.daemon_threads = True

    # Belt and braces: never serve balances and pod ids off-box.
    host, _ = httpd.server_address[:2]
    if host not in ("127.0.0.1", "::1"):
        httpd.server_close()
        print(f"refusing to serve on non-loopback address {host}", file=sys.stderr)
        return 2

    stop = threading.Event()

    def _stop(signum, frame):  # noqa: ARG001
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    refresher.start()
    threading.Thread(target=httpd.serve_forever, name="http", daemon=True).start()
    print(f"dispatch final-v1 dashboard on http://{BIND_HOST}:{port}/  "
          f"(loopback only; refresh {interval}s)")
    print(f"  ssh -N -L {port}:127.0.0.1:{port} {socket.gethostname()}   # then open the URL")
    print("  Ctrl-C or SIGTERM to stop")
    try:
        while not stop.is_set():
            stop.wait(1.0)
    except KeyboardInterrupt:
        pass
    print("shutting down…")
    refresher.stop_event.set()
    httpd.shutdown()
    httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only browser dashboard for the dispatch final-v1 campaign "
                    "(loopback only).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"loopback port (default {DEFAULT_PORT})")
    parser.add_argument("--interval", type=int, default=60,
                        help="collector refresh seconds (default 60)")
    parser.add_argument("--check", action="store_true",
                        help="collect one snapshot, print a one-line summary, exit")
    args = parser.parse_args(argv)

    if args.check:
        store = SnapshotStore()
        payload = Refresher(store, args.interval).refresh_once()
        if payload is None:
            print("COLLECT FAILED: " + str(store.get().get("collector_error")))
            return 1
        print(summarize(payload))
        return 0
    return serve(args.port, args.interval)


if __name__ == "__main__":
    sys.exit(main())
