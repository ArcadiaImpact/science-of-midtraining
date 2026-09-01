#!/usr/bin/env python3
"""Live, STRICTLY READ-ONLY terminal dashboard for the dispatch final-v1 campaign.

Run it in a tmux pane and leave it there:

    cd /workspace/scimt-dispatch-final && \
      uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/ops/dashboard.py

It redraws the whole screen every --interval seconds (default 60).  `q` + Enter
or Ctrl-C quits; SIGTERM is handled cleanly.

READ-ONLY CONTRACT
------------------
This program never mutates anything, anywhere:

  * it opens files only for reading (campaign JSON, ledgers, queues, supervisor
    pid files and logs) and writes nothing inside the ops dir;
  * the ONLY subprocesses it ever spawns are the two read-only ssh calls
      ssh <alias> bash .../ops/probe_unit.sh <profile> <arms>
      ssh <alias> cat /etc/runpod-deadman.json
    -- no runpodctl, no supervisor signals, no pod lifecycle calls, ever;
  * the only HTTP it makes is the RunPod GraphQL *query* for balance + pod list.

Secrets never reach the screen or a log: API keys live in local variables, are
sent only in an Authorization header, and any error text is scrubbed of them
before it is rendered.

SHAPE OF THE WORLD
------------------
Two RunPod accounts (A1, A2) pay for N campaigns.  Campaigns are discovered
from ops/campaign*.json, each with its own ledger / queue / runtime dir by the
established suffix convention:

    campaign.json   pods.txt   queue.txt   runtime/supervisor_<id>.{pid,log}    -> A1
    campaign2.json  pods2.txt  queue2.txt  runtime2/supervisor_<id>.{pid,log}   -> A2
    campaign3.json  pods3.txt  queue3.txt  runtime3/supervisor_<id>.{pid,log}   -> A2
    ...

so a campaign that gets split off mid-run (as sep01c was) shows up on its own,
and any of its files still missing renders as a visible gap rather than a
crash.  Every RunPod pod is reconciled against the ledgers: a live `dfv1-` pod
that no ledger claims is called out, because it is billing with nobody's name
on it.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:  # rich is available in the uv env; the dashboard degrades to ANSI without it.
    from rich.console import Console as _RichConsole
    from rich.table import Table as _RichTable
    from rich.text import Text as _RichText
    from rich import box as _rich_box

    HAVE_RICH = True
except Exception:  # noqa: BLE001 - any import failure means "plain ANSI mode"
    HAVE_RICH = False


# --------------------------------------------------------------------------- #
# Static configuration
# --------------------------------------------------------------------------- #

ROOT = Path("/workspace/scimt-dispatch-final")
OPS = ROOT / "experiments" / "prior_coins" / "dispatch_final_v1" / "ops"
POD_SIDE_PROBE = "/workspace/scimt/experiments/prior_coins/dispatch_final_v1/ops/probe_unit.sh"

GRAPHQL_URL = "https://api.runpod.io/graphql"
GRAPHQL_QUERY = "query { myself { clientBalance pods { id name costPerHr desiredStatus } } }"
ACCOUNT2_KEYFILE = Path("/root/.runpod2-home/apikey")

# Pods on account 1 that are somebody else's: excluded from our unit list, but
# their cost is real money leaving the account, so it counts toward A1 burn.
EXTERNAL_POD_IDS = {"lx6pucn0mfv8h3"}  # krill-mill, ~$0.17/hr
OUR_POD_PREFIX = "dfv1-"

ACTIVE_STATES = frozenset(
    {"provisioning", "booting", "setting_up", "running", "finishing", "recovering", "parked"}
)
PROBE_STATES = frozenset({"running", "parked", "setting_up"})
ATTENTION_STATES = frozenset({"parked", "halted", "lost", "provision_failed"})

# Coarse wall-clock expectations per profile, hours.  Deliberately coarse: the
# per-arm phase string is the real progress signal, this is only a sanity anchor.
EXPECTED_HOURS = {
    "gemma3_4b_1m": 9.6,
    "gemma3_4b_5m": 10.0,
    "gemma3_4b_50m": 14.6,
    "gemma3_12b_1m": 12.5,
    "gemma3_12b_5m": 12.9,
    "gemma3_12b_50m_4ep": 18.2,
    "gemma3_27b_5m": 14.4,
    "gemma3_27b_50m": 22.2,
    "gemma3_27b_190m": 46.4,
}

ALERT_RE = re.compile(r"PARKED|HALTED|FATAL|POD LOST|DURABLE COMPLETE|CAP GATE|Hub verified")
TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\]\s*(.*)$")
BLOCK_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)")

RUNWAY_RED_HOURS = 8.0
ALERT_TAIL_BYTES = 200_000
ALERTS_PER_SUPERVISOR = 6
PROBE_TIMEOUT_S = 25
DEADMAN_TIMEOUT_S = 15
DEADMAN_REFRESH_S = 600
HTTP_TIMEOUT_S = 25


@dataclass(frozen=True)
class AccountCfg:
    label: str
    key_env: str | None = None
    key_file: Path | None = None


ACCOUNTS = (
    AccountCfg(label="A1", key_env="RUNPOD_API_KEY"),
    AccountCfg(label="A2", key_file=ACCOUNT2_KEYFILE),
)


@dataclass(frozen=True)
class CampaignCfg:
    index: int
    account: str  # "A1" / "A2"
    campaign_file: Path
    ledger: Path
    queue: Path
    runtime: Path


def discover_campaigns() -> list[CampaignCfg]:
    """ops/campaign.json, campaign2.json, ... -> their ledger/queue/runtime set.

    Campaign 1 is paid for by account 1; every later campaign is an account-2
    split (the with_account2.sh convention).  Backup files (*.bak) are ignored.
    """
    found: list[CampaignCfg] = []
    for path in sorted(OPS.glob("campaign*.json")):
        m = re.fullmatch(r"campaign(\d*)\.json", path.name)
        if not m:
            continue  # e.g. campaign.json.dfv1-aug31.bak
        idx = int(m.group(1)) if m.group(1) else 1
        suffix = "" if idx == 1 else str(idx)
        found.append(
            CampaignCfg(
                index=idx,
                account="A1" if idx == 1 else "A2",
                campaign_file=path,
                ledger=OPS / f"pods{suffix}.txt",
                queue=OPS / f"queue{suffix}.txt",
                runtime=OPS / f"runtime{suffix}",
            )
        )
    return sorted(found, key=lambda c: c.index)


# --------------------------------------------------------------------------- #
# Tiny styled-render model: one data model, two back ends (rich / plain ANSI)
# --------------------------------------------------------------------------- #

Cell = tuple[str, str]  # (text, style-name)

ANSI = {
    "": "",
    "dim": "\033[2m",
    "ok": "\033[32m",
    "warn": "\033[33m",
    "bad": "\033[1;31m",
    "hdr": "\033[1;36m",
    "b": "\033[1m",
    "info": "\033[36m",
}
RICH = {
    "": None,
    "dim": "dim",
    "ok": "green",
    "warn": "yellow",
    "bad": "bold red",
    "hdr": "bold cyan",
    "b": "bold",
    "info": "cyan",
}
RESET = "\033[0m"


@dataclass
class Section:
    title: str
    columns: list[str] = field(default_factory=list)
    rows: list[list[Cell]] = field(default_factory=list)
    lines: list[list[Cell]] = field(default_factory=list)  # free-form styled lines
    empty_note: str = "(none)"


def _trunc(text: str, width: int) -> str:
    if width <= 0:
        return ""
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def render_ansi(sections: list[Section], width: int) -> str:
    out: list[str] = []
    for sec in sections:
        if sec.title:
            bar = "─" * max(0, width - len(sec.title) - 4)
            out.append(f"{ANSI['hdr']}── {sec.title}{RESET} {ANSI['dim']}{bar}{RESET}")
        for line in sec.lines:
            out.append("".join(f"{ANSI.get(st, '')}{txt}{RESET}" if st else txt for txt, st in line))
        if sec.columns:
            if not sec.rows:
                out.append(f"  {ANSI['dim']}{sec.empty_note}{RESET}")
            else:
                widths = [len(c) for c in sec.columns]
                for row in sec.rows:
                    for i, (txt, _) in enumerate(row):
                        if i < len(widths):
                            widths[i] = max(widths[i], len(txt))
                head = "  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(sec.columns))
                out.append(f"{ANSI['b']}{_trunc(head, width)}{RESET}")
                for row in sec.rows:
                    parts = []
                    for i, (txt, st) in enumerate(row):
                        pad = txt.ljust(widths[i]) if i < len(widths) else txt
                        parts.append(f"{ANSI.get(st, '')}{pad}{RESET}" if st else pad)
                    out.append("  " + "  ".join(parts))
        out.append("")
    return "\n".join(out)


def render_rich(console, sections: list[Section]) -> None:
    for sec in sections:
        if sec.title:
            console.print(_RichText(f"── {sec.title} ", style="bold cyan"), end="")
            console.print(_RichText("─" * max(0, console.width - len(sec.title) - 4), style="dim"))
        for line in sec.lines:
            text = _RichText()
            for txt, st in line:
                text.append(txt, style=RICH.get(st))
            console.print(text, overflow="ellipsis", no_wrap=True)
        if sec.columns:
            if not sec.rows:
                console.print(_RichText(f"  {sec.empty_note}", style="dim"))
            else:
                table = _RichTable(box=_rich_box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                                   header_style="bold", expand=False)
                for col in sec.columns:
                    table.add_column(col, overflow="fold")
                for row in sec.rows:
                    table.add_row(*[_RichText(txt, style=RICH.get(st)) for txt, st in row])
                console.print(table)
        console.print()


# --------------------------------------------------------------------------- #
# Read-only data sources
# --------------------------------------------------------------------------- #


@dataclass
class PodRecord:
    state: str
    profile: str
    arms: str
    ssh_alias: str
    pod_id: str
    hourly_rate: float
    campaign_id: str
    owner_token: str
    attempt: int
    pod_name: str
    created_at: str
    strikes: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.profile, self.arms)


def read_campaign(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_ledger(path: Path, owner_token: str) -> list[PodRecord]:
    """Latest attempt per (profile, arms) for the given campaign owner token."""
    latest: dict[tuple[str, str], PodRecord] = {}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 12:
                continue
            try:
                rec = PodRecord(
                    state=fields[0].strip(),
                    profile=fields[1].strip(),
                    arms=fields[2].strip(),
                    ssh_alias=fields[3].strip(),
                    pod_id=fields[4].strip(),
                    hourly_rate=float(fields[5]),
                    campaign_id=fields[6].strip(),
                    owner_token=fields[7].strip(),
                    attempt=int(fields[8]),
                    pod_name=fields[9].strip(),
                    created_at=fields[10].strip(),
                    strikes=fields[11].strip(),
                )
            except (ValueError, IndexError):
                continue
            if rec.owner_token != owner_token:
                continue
            prev = latest.get(rec.key)
            if prev is None or rec.attempt > prev.attempt:
                latest[rec.key] = rec
    return sorted(latest.values(), key=lambda r: (r.profile, r.arms))


@dataclass
class QueueRow:
    priority: int
    profile: str
    arms: str
    hourly_rate: float
    held: bool


_PROFILE_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_ARMS_RE = re.compile(r"^[A-Za-z0-9_\-]+(,[A-Za-z0-9_\-]+)*$")


def _parse_queue_fields(fields: list[str], held: bool) -> QueueRow | None:
    if len(fields) != 4:
        return None
    try:
        priority = int(fields[0].strip())
        rate = float(fields[3].strip())
    except ValueError:
        return None
    profile, arms = fields[1].strip(), fields[2].strip()
    if not _PROFILE_RE.match(profile) or not _ARMS_RE.match(arms):
        return None
    return QueueRow(priority, profile, arms, rate, held)


def read_queue(path: Path) -> list[QueueRow]:
    """Queue rows, including `#`-commented rows that still parse as data (HELD).

    Prose comments never survive the schema check, so only real held rows are
    picked up.
    """
    rows: list[QueueRow] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            held = line.lstrip().startswith("#")
            body = line.lstrip().lstrip("#") if held else line
            row = _parse_queue_fields(body.split("\t"), held)
            if row is not None:
                rows.append(row)
    return sorted(rows, key=lambda r: (r.held, r.priority))


@dataclass
class AccountState:
    balance: float | None = None
    pods: list[dict] = field(default_factory=list)
    error: str | None = None


def _scrub(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "<redacted>")
    return text


def load_api_key(cfg: AccountCfg) -> str | None:
    if cfg.key_env:
        return os.environ.get(cfg.key_env, "").strip() or None
    if cfg.key_file:
        try:
            return cfg.key_file.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None


def fetch_account(cfg: AccountCfg) -> AccountState:
    key = load_api_key(cfg)
    if not key:
        return AccountState(error="no api key available")
    body = json.dumps({"query": GRAPHQL_QUERY}).encode("utf-8")
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            # The API edge 403s the default urllib agent; a plain UA is enough.
            "User-Agent": "dfv1-dashboard/1.0",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return AccountState(error=_scrub(f"{type(exc).__name__}: {exc}", [key])[:120])
    if payload.get("errors"):
        return AccountState(error=_scrub(f"graphql: {str(payload['errors'])[:120]}", [key]))
    myself = (payload.get("data") or {}).get("myself") or {}
    return AccountState(
        balance=myself.get("clientBalance"),
        pods=[p for p in (myself.get("pods") or []) if isinstance(p, dict)],
    )


@dataclass
class Probe:
    phases: str = "?"
    runner_state: str = "?"
    procs: str = "?"
    log_age: int | None = None
    ok: bool = False
    error: str = ""


def probe_unit(rec: PodRecord) -> Probe:
    """One read-only ssh per active pod.  Never retried inside a refresh."""
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        rec.ssh_alias, "bash", POD_SIDE_PROBE, rec.profile, rec.arms,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return Probe(phases="unreachable", error="ssh timeout")
    except OSError as exc:
        return Probe(phases="unreachable", error=f"ssh {type(exc).__name__}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or "").strip().splitlines()
        return Probe(phases="unreachable",
                     error=tail[-1][:60] if tail else f"rc={proc.returncode}")
    parts = lines[-1].split("|")
    if len(parts) < 4:
        return Probe(phases="unparsed", error=lines[-1][:60])
    try:
        age = int(parts[2])
    except ValueError:
        age = None
    return Probe(
        phases=parts[3].strip(),
        runner_state=parts[0].strip(),
        procs=parts[1].strip(),
        log_age=age,
        ok=True,
    )


def fetch_deadman(alias: str) -> str | None:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", alias,
           "cat", "/etc/runpod-deadman.json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=DEADMAN_TIMEOUT_S, stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
    except Exception:  # noqa: BLE001 - optional source, fail silently by design
        return None
    val = data.get("deadline_utc") if isinstance(data, dict) else None
    return str(val) if val else None


@dataclass
class Alert:
    when: str
    message: str
    count: int = 1


@dataclass
class Supervisor:
    alive: bool = False
    pid: int | None = None
    note: str = ""
    alerts: list[Alert] = field(default_factory=list)
    log_seen: bool = False


def supervisor_state(runtime: Path, campaign_id: str) -> Supervisor:
    sup = Supervisor()
    pid_file = runtime / f"supervisor_{campaign_id}.pid"
    log_file = runtime / f"supervisor_{campaign_id}.log"
    if not log_file.exists() and (runtime / "supervisor.log").exists():
        log_file = runtime / "supervisor.log"

    try:
        sup.pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        sup.note = "no pid file"
        sup.pid = None
    if sup.pid is not None:
        try:
            os.kill(sup.pid, 0)
            sup.alive = True
        except ProcessLookupError:
            sup.note = "pid not running"
        except PermissionError:
            sup.alive = True  # exists, owned by another user
        except OSError as exc:
            sup.note = type(exc).__name__
        if sup.alive:
            try:
                cmdline = Path(f"/proc/{sup.pid}/cmdline").read_bytes().decode("utf-8", "replace")
                if cmdline and "supervisor" not in cmdline:
                    sup.note = "pid reused by another process"
            except OSError:
                pass

    # The supervisor reprints its status block (including the PARKED banner)
    # every cycle, and some of those lines carry no timestamp of their own.
    # Carry the most recent timestamp seen in the stream forward, and fold
    # repeats of one message into a single row with an occurrence count, so a
    # long-standing condition cannot crowd out newer events.
    try:
        size = log_file.stat().st_size
        with log_file.open("rb") as fh:
            if size > ALERT_TAIL_BYTES:
                fh.seek(size - ALERT_TAIL_BYTES)
                fh.readline()  # drop the partial line
            blob = fh.read().decode("utf-8", "replace")
        sup.log_seen = True
        hits: dict[str, Alert] = {}
        last_ts = ""
        for line in blob.splitlines():
            stripped = line.strip()
            m = TS_RE.match(stripped)
            if m:
                last_ts, body = m.group(1), m.group(2)
            else:
                block = BLOCK_TS_RE.search(stripped)
                if block:
                    last_ts = block.group(1)
                body = stripped
            if not ALERT_RE.search(stripped):
                continue
            prev = hits.pop(body, None)
            hits[body] = Alert(when=last_ts, message=body,
                               count=(prev.count + 1) if prev else 1)
        sup.alerts = list(hits.values())[-ALERTS_PER_SUPERVISOR:]
    except OSError as exc:
        sup.note = (sup.note + "; " if sup.note else "") + f"no log ({type(exc).__name__})"
    return sup


# --------------------------------------------------------------------------- #
# Snapshot assembly
# --------------------------------------------------------------------------- #


@dataclass
class UnitView:
    campaign: str
    account: str
    rec: PodRecord
    probe: Probe | None = None
    deadline: str | None = None


@dataclass
class CampaignView:
    cfg: CampaignCfg
    campaign_id: str = "?"
    units: list[UnitView] = field(default_factory=list)
    queued: list[QueueRow] = field(default_factory=list)
    supervisor: Supervisor = field(default_factory=Supervisor)
    notes: list[str] = field(default_factory=list)


@dataclass
class AccountView:
    cfg: AccountCfg
    state: AccountState = field(default_factory=AccountState)


@dataclass
class Snapshot:
    taken_at: float
    accounts: list[AccountView] = field(default_factory=list)
    campaigns: list[CampaignView] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Collector:
    """Gathers one frame's worth of data.  Caches the (optional) dead-man reads."""

    def __init__(self) -> None:
        self._deadman: dict[str, tuple[float, str | None]] = {}

    def _deadman_for(self, aliases: list[str]) -> dict[str, str | None]:
        now = time.time()
        stale = [a for a in aliases if now - self._deadman.get(a, (0.0, None))[0] > DEADMAN_REFRESH_S]
        if stale:
            with futures.ThreadPoolExecutor(max_workers=min(8, len(stale))) as pool:
                for alias, res in zip(stale, pool.map(fetch_deadman, stale)):
                    self._deadman[alias] = (now, res)
        return {a: self._deadman.get(a, (0.0, None))[1] for a in aliases}

    def collect(self) -> Snapshot:
        errors: list[str] = []

        # 1. Local files first (cheap, and they define what we probe).
        campaigns: list[CampaignView] = []
        for cfg in discover_campaigns():
            view = CampaignView(cfg=cfg)
            token = ""
            try:
                data = read_campaign(cfg.campaign_file)
                view.campaign_id = str(data.get("campaign_id", "?"))
                token = str(data.get("owner_token", ""))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                view.notes.append(f"campaign file unreadable ({type(exc).__name__})")
                errors.append(f"{cfg.campaign_file.name}: {type(exc).__name__}")
            if token:
                try:
                    view.units = [
                        UnitView(view.campaign_id, cfg.account, rec)
                        for rec in read_ledger(cfg.ledger, token)
                    ]
                except FileNotFoundError:
                    view.notes.append(f"{cfg.ledger.name} missing")
                except OSError as exc:
                    view.notes.append(f"{cfg.ledger.name} unreadable ({type(exc).__name__})")
                    errors.append(f"{cfg.ledger.name}: {type(exc).__name__}")
            try:
                view.queued = read_queue(cfg.queue)
            except FileNotFoundError:
                view.notes.append(f"{cfg.queue.name} missing")
            except OSError as exc:
                view.notes.append(f"{cfg.queue.name} unreadable ({type(exc).__name__})")
                errors.append(f"{cfg.queue.name}: {type(exc).__name__}")
            view.supervisor = supervisor_state(cfg.runtime, view.campaign_id)
            campaigns.append(view)

        # 2. Balances, in parallel across accounts (one POST each).
        accounts = [AccountView(cfg=cfg) for cfg in ACCOUNTS]
        with futures.ThreadPoolExecutor(max_workers=len(accounts)) as pool:
            for view, state in zip(accounts, pool.map(fetch_account, [a.cfg for a in accounts])):
                view.state = state
                if state.error:
                    errors.append(f"{view.cfg.label} api: {state.error}")

        # 3. One ssh probe per active pod, every campaign at once.
        targets = [u for c in campaigns for u in c.units
                   if u.rec.state in PROBE_STATES and u.rec.ssh_alias not in ("", "-")]
        if targets:
            with futures.ThreadPoolExecutor(max_workers=min(16, len(targets))) as pool:
                for unit, probe in zip(targets, pool.map(lambda u: probe_unit(u.rec), targets)):
                    unit.probe = probe

        # 4. Optional dead-man deadlines (cached, >=10 min apart).
        aliases = [u.rec.ssh_alias for u in targets]
        if aliases:
            deadlines = self._deadman_for(aliases)
            for unit in targets:
                unit.deadline = deadlines.get(unit.rec.ssh_alias)

        return Snapshot(taken_at=time.time(), accounts=accounts, campaigns=campaigns,
                        errors=errors)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def parse_utc(stamp: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(stamp, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            continue
    return None


def fmt_money(value: float | None) -> str:
    return "?" if value is None else f"${value:,.2f}"


def state_style(state: str) -> str:
    if state in ATTENTION_STATES:
        return "bad"
    if state == "running":
        return "ok"
    if state in ("done", "cleaned"):
        return "dim"
    return "warn"


def phase_style(phases: str) -> str:
    low = phases.lower()
    if "unreachable" in low or "unparsed" in low or low == "?":
        return "bad"
    parts = [p for p in phases.split(",") if ":" in p]
    if parts and all(p.split(":", 1)[1] == "done" for p in parts):
        return "ok"
    return "info"


def is_ours(pod: dict) -> bool:
    return (str(pod.get("name") or "").startswith(OUR_POD_PREFIX)
            and pod.get("id") not in EXTERNAL_POD_IDS)


# The three roll-ups below are the shared reading of a snapshot: what an account
# is spending, which queue rows are still waiting, and how long a unit has been
# up.  They live here (not in a renderer) so the TUI and dashboard_web.py cannot
# drift into two different answers for the same question.


@dataclass
class AccountRollup:
    label: str
    balance: float | None
    burn: float
    runway_hours: float | None
    our_pods: list[dict]
    external_pods: list[dict]
    external_cost: float
    unclaimed: list[dict]  # our pods that no campaign ledger claims
    error: str | None


def claimed_pod_ids(snap: "Snapshot") -> dict[str, UnitView]:
    """pod id -> the ledger row that owns it, across every campaign."""
    return {
        u.rec.pod_id: u
        for c in snap.campaigns for u in c.units
        if u.rec.pod_id and u.rec.pod_id != "-"
    }


def rollup_account(view: AccountView, claimed: set[str]) -> AccountRollup:
    """Balance/burn/runway for one account.

    External pods (krill-mill) are excluded from our pod list but their cost is
    real money leaving the account, so it counts toward burn.
    """
    ours: list[dict] = []
    external: list[dict] = []
    unclaimed: list[dict] = []
    ext_cost = 0.0
    for pod in view.state.pods:
        if pod.get("desiredStatus") != "RUNNING":
            continue
        if is_ours(pod):
            ours.append(pod)
            if pod.get("id") not in claimed:
                unclaimed.append(pod)
        else:
            external.append(pod)
            ext_cost += float(pod.get("costPerHr") or 0.0)
    burn = sum(float(p.get("costPerHr") or 0.0) for p in ours) + ext_cost
    balance = view.state.balance
    runway = (balance / burn) if (balance is not None and burn > 0) else None
    return AccountRollup(
        label=view.cfg.label, balance=balance, burn=burn, runway_hours=runway,
        our_pods=ours, external_pods=external, external_cost=ext_cost,
        unclaimed=unclaimed, error=view.state.error,
    )


def split_queue(view: CampaignView) -> tuple[list[QueueRow], list[QueueRow]]:
    """(still waiting, held) — rows already launched or finished drop out."""
    launched = {(u.rec.profile, u.rec.arms) for u in view.units
                if u.rec.state in ACTIVE_STATES or u.rec.state == "done"}
    queued = [q for q in view.queued if not q.held and (q.profile, q.arms) not in launched]
    held = [q for q in view.queued if q.held]
    return queued, held


def elapsed_hours(rec: PodRecord, now: float | None = None) -> float | None:
    """Wall clock since the ledger's created-at, or None if it won't parse."""
    created = parse_utc(rec.created_at)
    if created is None:
        return None
    return ((now if now is not None else time.time()) - created.timestamp()) / 3600.0


def build_sections(snap: Snapshot, last_ok: float | None, last_error: str | None,
                   interval: int) -> list[Section]:
    now = time.time()
    sections: list[Section] = []

    # Ledger claims, for pod reconciliation and per-account campaign grouping.
    claimed = claimed_pod_ids(snap)

    # ---- header ----------------------------------------------------------- #
    header = Section(title="")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    age = "never" if last_ok is None else f"{int(now - last_ok)}s ago"
    age_style = "ok" if (last_ok is not None and now - last_ok < interval * 2.5) else "warn"
    header.lines.append([
        ("DISPATCH FINAL-V1 CAMPAIGN", "hdr"),
        ("   ", ""), (stamp, "b"),
        ("   last good refresh ", "dim"), (age, age_style),
    ])
    sections.append(header)

    # ---- accounts --------------------------------------------------------- #
    acct = Section(title="ACCOUNTS",
                   columns=["ACC", "BALANCE", "POD BURN $/hr", "RUNWAY", "OUR PODS", "EXTERNAL",
                            "CAMPAIGNS", "NOTE"])
    for view in snap.accounts:
        camps = [c for c in snap.campaigns if c.cfg.account == view.cfg.label]
        camp_txt = ",".join(c.campaign_id for c in camps) or "-"
        if view.state.error:
            acct.rows.append([
                (view.cfg.label, "b"), ("?", "bad"), ("?", "bad"), ("?", "bad"),
                ("?", "bad"), ("?", "bad"), (camp_txt, "dim"),
                (_trunc(view.state.error, 70), "bad"),
            ])
            continue
        roll = rollup_account(view, set(claimed))
        if roll.runway_hours is not None:
            run_cell: Cell = (f"{roll.runway_hours:,.1f}h",
                              "bad" if roll.runway_hours < RUNWAY_RED_HOURS else
                              ("warn" if roll.runway_hours < RUNWAY_RED_HOURS * 2 else "ok"))
        else:
            run_cell = ("n/a" if roll.burn <= 0 else "?", "dim")
        note = ""
        if roll.unclaimed:
            note = "unclaimed pod(s): " + ", ".join(
                f"{p.get('name')} (${float(p.get('costPerHr') or 0):.2f}/hr)"
                for p in roll.unclaimed)
        acct.rows.append([
            (view.cfg.label, "b"),
            (fmt_money(roll.balance), "b"),
            (f"{roll.burn:,.2f}", ""),
            run_cell,
            (str(len(roll.our_pods)), ""),
            (f"{len(roll.external_pods)} (${roll.external_cost:.2f}/hr)"
             if roll.external_pods else "-", "dim" if roll.external_pods else ""),
            (camp_txt, "dim"),
            (_trunc(note, 80), "warn" if note else ""),
        ])
    sections.append(acct)

    # ---- campaigns / supervisors ------------------------------------------ #
    camps = Section(title="CAMPAIGNS & SUPERVISORS",
                    columns=["CAMPAIGN", "ACC", "ACTIVE", "$/hr", "QUEUED", "HELD",
                             "SUPERVISOR", "NOTE"])
    for view in snap.campaigns:
        active = [u for u in view.units if u.rec.state in ACTIVE_STATES]
        queued, held = split_queue(view)
        sup = view.supervisor
        sup_cell: Cell = (f"pid {sup.pid} alive" if sup.alive else
                          (f"pid {sup.pid} DEAD" if sup.pid else "no supervisor pid"),
                          "ok" if sup.alive else "bad")
        note = "; ".join(view.notes + ([sup.note] if sup.note else []))
        camps.rows.append([
            (view.campaign_id, "b"), (view.cfg.account, ""),
            (str(len(active)), "ok" if active else "dim"),
            (f"{sum(u.rec.hourly_rate for u in active):,.2f}", ""),
            (str(len(queued)), "" if queued else "dim"),
            (str(len(held)), "warn" if held else "dim"),
            sup_cell,
            (_trunc(note, 70), "warn" if note else ""),
        ])
    sections.append(camps)

    # ---- work units ------------------------------------------------------- #
    units = Section(title="WORK UNITS  (elapsed is wall clock since the ledger's created-at)",
                    columns=["CAMPAIGN", "PROFILE", "ARMS", "STATE", "POD", "$/hr", "ATT",
                             "ELAPSED / EXPECTED", "RUNNER", "DEADMAN", "PER-ARM PHASE"],
                    empty_note="(no units in any campaign ledger)")
    for view in snap.campaigns:
        active = [u for u in view.units if u.rec.state in ACTIVE_STATES]
        rest = [u for u in view.units if u.rec.state not in ACTIVE_STATES]
        for unit in active + rest:
            rec = unit.rec
            hours = elapsed_hours(rec, now)
            exp = EXPECTED_HOURS.get(rec.profile)
            if hours is None:
                elapsed_cell: Cell = ("? / ?", "warn")
            else:
                elapsed_cell = (
                    f"{hours:.1f}h / ~{exp:.1f}h" if exp else f"{hours:.1f}h / ~?",
                    "warn" if (exp and hours > exp * 1.25) else "",
                )
            if unit.probe is None:
                phase_cell: Cell = ("-", "dim")
                runner_cell: Cell = ("-", "dim")
            elif not unit.probe.ok:
                phase_cell = (unit.probe.phases, "bad")
                runner_cell = (_trunc(unit.probe.error or "unreachable", 22), "bad")
            else:
                p = unit.probe
                age_txt = "?" if p.log_age is None else (
                    f"{p.log_age}s" if p.log_age >= 0 else "no-log")
                phase_cell = (p.phases, phase_style(p.phases))
                runner_cell = (
                    f"{p.runner_state} p{p.procs} {age_txt}",
                    "bad" if p.runner_state in ("FAILED", "UNKNOWN") else
                    ("warn" if (p.procs == "0" or (p.log_age or 0) > 1800) else ""),
                )
            if unit.deadline:
                dl = parse_utc(unit.deadline)
                if dl is None:
                    dead_cell: Cell = (_trunc(unit.deadline, 20), "dim")
                else:
                    left = (dl.timestamp() - now) / 3600.0
                    dead_cell = (f"{left:.1f}h ({dl.strftime('%H:%MZ')})",
                                 "bad" if left < 1.5 else ("warn" if left < 3 else "dim"))
            else:
                dead_cell = ("-", "dim")
            units.rows.append([
                (view.campaign_id, "dim"),
                (rec.profile, ""),
                (rec.arms, "dim"),
                (rec.state, state_style(rec.state)),
                (rec.pod_id or "-", "dim"),
                (f"{rec.hourly_rate:.2f}", ""),
                (str(rec.attempt), "warn" if rec.attempt > 1 else "dim"),
                elapsed_cell, runner_cell, dead_cell, phase_cell,
            ])
    sections.append(units)

    # Live pods no ledger claims: real money with nobody's name on it.
    orphans: list[list[Cell]] = []
    for view in snap.accounts:
        for pod in view.state.pods:
            if pod.get("desiredStatus") != "RUNNING" or not is_ours(pod):
                continue
            if pod.get("id") in claimed:
                continue
            orphans.append([
                (f"  ! {view.cfg.label} pod {pod.get('id')} {pod.get('name')} "
                 f"${float(pod.get('costPerHr') or 0):.2f}/hr is running but no campaign "
                 f"ledger claims it", "bad")
            ])
    units.lines.extend(orphans)

    # ---- queue ------------------------------------------------------------ #
    queue = Section(title="QUEUED / HELD",
                    columns=["CAMPAIGN", "ACC", "PRI", "PROFILE", "ARMS", "$/hr", "STATUS"],
                    empty_note="(queues drained)")
    for view in snap.campaigns:
        pending, held_rows = split_queue(view)
        for row in pending + held_rows:
            queue.rows.append([
                (view.campaign_id, "dim"), (view.cfg.account, ""), (str(row.priority), "dim"),
                (row.profile, ""), (row.arms, "dim"), (f"{row.hourly_rate:.2f}", ""),
                ("HELD", "warn") if row.held else ("queued", ""),
            ])
    sections.append(queue)

    # ---- alerts ----------------------------------------------------------- #
    alerts = Section(title="RECENT SUPERVISOR ALERTS  (last seen; xN = repeats in the log tail)",
                     columns=["CAMPAIGN", "LAST SEEN", "N", "MESSAGE"],
                     empty_note="(no supervisor logs readable)")
    for view in snap.campaigns:
        if not view.supervisor.alerts:
            why = ("no log file yet" if not view.supervisor.log_seen
                   else "no PARKED/HALTED/FATAL/POD LOST/DURABLE COMPLETE/CAP GATE/Hub-verified "
                        "lines in the recent log tail")
            alerts.rows.append([
                (view.campaign_id, "dim"), ("-", "dim"), ("-", "dim"), (f"({why})", "dim"),
            ])
            continue
        for alert in view.supervisor.alerts:
            style = "bad" if re.search(r"PARKED|HALTED|FATAL|POD LOST", alert.message) else (
                "ok" if ("DURABLE COMPLETE" in alert.message or "Hub verified" in alert.message)
                else "warn")
            alerts.rows.append([
                (view.campaign_id, "dim"), (alert.when or "-", "dim"),
                (f"x{alert.count}" if alert.count > 1 else "1", "dim"),
                (_trunc(alert.message, 140), style),
            ])
    sections.append(alerts)

    # ---- footer ----------------------------------------------------------- #
    footer = Section(title="")
    foot: list[Cell] = [
        (f"refresh every {interval}s", "dim"),
        ("  |  ", "dim"), ("q + Enter", "b"), (" or Ctrl-C to quit", "dim"),
        ("  |  ", "dim"), ("READ-ONLY", "ok"),
    ]
    if snap.errors:
        foot += [("  |  degraded: ", "dim"), (_trunc("; ".join(snap.errors), 90), "warn")]
    if last_error:
        foot += [("  |  last refresh error: ", "dim"), (_trunc(last_error, 90), "bad")]
    footer.lines.append(foot)
    sections.append(footer)
    return sections


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

_STOP = False


def _handle_term(signum, frame):  # noqa: ARG001
    """Ask the loop to stop after the frame in flight; a second signal is hard."""
    global _STOP
    if _STOP:  # impatient second Ctrl-C / SIGTERM: give up immediately
        signal.signal(signum, signal.SIG_DFL)
        raise KeyboardInterrupt
    _STOP = True


def clear_screen(console) -> None:
    if console is not None:
        console.clear()
    else:
        sys.stdout.write("\033[H\033[2J\033[3J")


def draw(console, sections: list[Section]) -> None:
    if console is not None:
        # Follow tmux pane resizes between frames.
        console.width = max(120, shutil.get_terminal_size((200, 50)).columns)
        render_rich(console, sections)
    else:
        width = shutil.get_terminal_size((200, 50)).columns
        sys.stdout.write(render_ansi(sections, width) + "\n")
    sys.stdout.flush()


def wait_or_quit(seconds: float) -> bool:
    """Sleep, returning True if the user asked to quit."""
    deadline = time.time() + seconds
    interactive = sys.stdin is not None and sys.stdin.isatty()
    while not _STOP and time.time() < deadline:
        remaining = min(1.0, max(0.0, deadline - time.time()))
        if interactive:
            import select

            ready, _, _ = select.select([sys.stdin], [], [], remaining)
            if ready:
                line = sys.stdin.readline()
                if not line or line.strip().lower() in ("q", "quit", "exit"):
                    return True
        else:
            time.sleep(remaining)
    return _STOP


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only dispatch final-v1 campaign dashboard.")
    parser.add_argument("--once", action="store_true", help="render one frame and exit")
    parser.add_argument("--interval", type=int, default=60, help="refresh seconds (default 60)")
    parser.add_argument("--no-color", action="store_true", help="plain-ANSI renderer, no rich")
    args = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    console = None
    if HAVE_RICH and not args.no_color:
        console = _RichConsole(width=max(120, shutil.get_terminal_size((200, 50)).columns),
                               highlight=False, soft_wrap=False)

    collector = Collector()
    snapshot: Snapshot | None = None
    last_ok: float | None = None
    last_error: str | None = None

    while True:
        try:
            snapshot = collector.collect()
            last_ok = snapshot.taken_at
            last_error = None
        except Exception as exc:  # noqa: BLE001 - the loop must never die
            last_error = f"{type(exc).__name__}: {exc}"[:200]
            if snapshot is None:
                snapshot = Snapshot(taken_at=time.time(), errors=[last_error])

        try:
            sections = build_sections(snapshot, last_ok, last_error, args.interval)
            if not args.once:
                clear_screen(console)
            draw(console, sections)
        except Exception as exc:  # noqa: BLE001 - a render bug must not kill the loop
            last_error = f"render {type(exc).__name__}: {exc}"[:200]
            sys.stdout.write(f"\n[dashboard render error] {last_error}\n")
            sys.stdout.flush()

        if args.once or _STOP:
            return 0
        if wait_or_quit(args.interval):
            return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
