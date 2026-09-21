#!/usr/bin/env python3
"""Live, STRICTLY READ-ONLY terminal dashboard for the dispatch final-v1 campaign.

Run it in a tmux pane and leave it there:

    cd /workspace/scimt-dispatch-final && \
      uv run --extra dev python3 experiments/dispatch/dispatch_final_v1/ops/dashboard.py

It redraws the whole screen every --interval seconds (default 60).  `q` + Enter
or Ctrl-C quits; SIGTERM is handled cleanly.

READ-ONLY CONTRACT
------------------
This program never mutates anything, anywhere:

  * it opens files only for reading (campaign JSON, ledgers, queues, the
    hand-run unit table, supervisor pid files and logs) and writes nothing
    inside the ops dir;
  * the ONLY subprocesses it ever spawns are the three read-only ssh calls
      ssh <alias> bash -s -- <profile> <arms>   (DETAIL_PROBE_SCRIPT on stdin)
      ssh <alias> bash -s -- <log> <root>       (HANDRUN_PROBE_SCRIPT on stdin)
      ssh <alias> cat /etc/runpod-deadman.json
    all three built by ssh_command(), so there is one ssh policy, not three
    -- no runpodctl, no supervisor signals, no pod lifecycle calls, ever;
  * the only HTTP it makes is the RunPod GraphQL *query* for balance + pod list.

WHY AN INLINE PROBE SCRIPT
--------------------------
The pods run whatever git commit they were pinned to, so a repo-side script is
not guaranteed to exist on them, and ops/probe_unit.sh is also consumed by the
supervisor -- it must not grow a second caller's requirements.  So the dashboard
ships its own probe over stdin (`bash -s --`, exactly the way supervisor.py's
bootstrap() feeds a script to ssh): still ONE ssh per active pod per refresh,
still read-only (stat/grep/tail/find over small files; train.logs are only ever
tail'd by byte count, never cat'd).  It prints the classic four-field
probe_unit.sh line FIRST -- so a crash anywhere in the richer detail pass still
leaves today's phase-only rendering intact -- then one `DETAIL {json}` line per
arm with intra-stage progress.  Detail is strictly additive: an unparsable or
absent payload degrades to exactly what this dashboard showed before it existed.

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
crash.  Every RunPod pod is reconciled against the ledgers *and* the hand-run
table below: a live `dfv1-` pod that neither claims is called out, because it
is billing with nobody's name on it.

HAND-RUN (UNMANAGED) UNITS
--------------------------
Some work is launched by hand: no supervisor, no queue row, no ledger line,
and -- for the GLM three-arm row -- pods on two different RunPod accounts,
which the one-campaign-one-account mapping above cannot express.  Forcing
those into a fake campaign would corrupt the ledger reading, so they are a
separate concept, declared in

    ops/handrun_units.tsv

(7 tab-separated fields: label, account, pod_id, ssh_alias, status_log,
progress_root, note; '#' comments; "-" for "not applicable"; the file's own
header documents each field).  Adding a future hand-run unit is a one-line
edit there, not a code change.  For each row the dashboard reports whether the
pod is RUNNING per that account's GraphQL query, its $/hr, the newest
timestamped line of status_log, and -- from the newest train.log under
progress_root -- which stage/arm is training with step/total, s/it and the
latest loss.  Everything degrades to "unknown": an unreachable pod, a missing
file or a malformed row can never crash the dashboard or hide the other rows.

The LABEL IS AUTHORITATIVE, never the RunPod pod name: pod kgxwecxy3cqn8e is
*named* dfv1-glm-2tb-control-charter (it inherited the snipe template's name)
but runs the COIN arm, and a sibling pod with a near-identical name runs
charter.  Rendering the pod name would mislabel a live scientific run, so this
dashboard never displays it.

UNIT ORDERING
-------------
Live work sorts to the top -- attention states (parked/halted/lost/
provision_failed) first, since a parked pod is alive, billing, and waiting on
a human -- then hand-run units, and only then the finished (done/cleaned)
rows, in their own clearly-labelled section.  The ledger carries no
finished-at stamp, so "most recently finished first" is approximated by
created-at, descending; that is stated on the section header rather than
hidden.
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
OPS = ROOT / "experiments" / "dispatch" / "dispatch_final_v1" / "ops"
# ops/probe_unit.sh is the supervisor's copy of this logic and lives on the pods
# at the pinned commit.  The dashboard deliberately does NOT call it (see the
# module docstring); DETAIL_PROBE_SCRIPT below is its own, richer stdin script,
# and the two must stay behaviourally identical in their four-field first line.
PROBE_UNIT_REFERENCE = "experiments/dispatch/dispatch_final_v1/ops/probe_unit.sh"

GRAPHQL_URL = "https://api.runpod.io/graphql"
GRAPHQL_QUERY = "query { myself { clientBalance pods { id name costPerHr desiredStatus } } }"
ACCOUNT2_KEYFILE = Path("/root/.runpod2-home/apikey")
ACCOUNT3_KEYFILE = Path("/root/.runpod3-home/apikey")

# Pods on account 1 that are somebody else's: excluded from our unit list, but
# their cost is real money leaving the account, so it counts toward A1 burn.
EXTERNAL_POD_IDS = {"lx6pucn0mfv8h3"}  # krill-mill, ~$0.17/hr
OUR_POD_PREFIX = "dfv1-"

ACTIVE_STATES = frozenset(
    {"provisioning", "booting", "setting_up", "running", "finishing", "recovering", "parked"}
)
PROBE_STATES = frozenset({"running", "parked", "setting_up"})
ATTENTION_STATES = frozenset({"parked", "halted", "lost", "provision_failed"})
# Terminal states.  These rows are history: they get their own section BELOW
# the live ones, so a finished unit can never sit above a running one.
FINISHED_STATES = frozenset({"done", "cleaned"})

# Hand-run units: pods launched by hand, with no supervisor / queue / ledger
# row.  Declarative table, read-only; the schema is in the module docstring and
# in the file's own header.
HANDRUN_FILE = Path(__file__).with_name("handrun_units.tsv")
HANDRUN_FIELDS = 7

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
PROBE_TIMEOUT_S = 35  # the detail pass adds a few small find/tail calls per arm
HANDRUN_TIMEOUT_S = 30  # one tail + one find per hand-run pod
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
    AccountCfg(label="A3", key_file=ACCOUNT3_KEYFILE),
)

# Which account pays for campaign N.  Campaign 1 is account 1; the sep01b /
# sep01c splits are account 2 (the with_account2.sh convention).  Account 3 is
# funded but hosts no campaign yet, so it renders as a balance-only card until
# a campaign is mapped to it here.
CAMPAIGN_ACCOUNTS = {1: "A1"}
CAMPAIGN_ACCOUNT_DEFAULT = "A2"


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

    Which account pays is CAMPAIGN_ACCOUNTS above.  Backup files (*.bak) are
    ignored.
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
                account=CAMPAIGN_ACCOUNTS.get(idx, CAMPAIGN_ACCOUNT_DEFAULT),
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
        key = os.environ.get(cfg.key_env, "").strip()
        if not key and cfg.label == "A1":
            try:
                import tomllib
                key = tomllib.loads((Path.home() / ".runpod/config.toml").read_text()).get("apikey", "")
            except (OSError, ValueError):
                pass
        return key or None
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


TRAINING_STAGES = ("midtrain", "dolci")
BATTERY_STAGES = ("eval", "recall", "d4", "costsweep")
AFT_CELLS = 4
DETAIL_PREFIX = "DETAIL "


# --------------------------------------------------------------------------- #
# The one ssh policy.  Every read-only probe in this file goes through these two
# helpers so there is a single place that decides how we shell out.
# --------------------------------------------------------------------------- #


def ssh_env() -> dict[str, str]:
    """Environment for every ssh this dashboard runs.

    The pods authenticate off the persistent agent at ~/.ssh/agent.sock; some
    hosts (the hand-run GLM charter pod, measured 2026-09-02) refuse the
    IdentityFile alone and only succeed with that agent, while an inherited
    SSH_AUTH_SOCK from an editor/remote session points at an agent that does
    not hold the pod keys.  So prefer the on-disk agent socket when it exists.
    This only ever *reads* the socket path -- nothing is written or unlocked.
    """
    env = dict(os.environ)
    sock = Path.home() / ".ssh" / "agent.sock"
    try:
        if sock.exists():
            env["SSH_AUTH_SOCK"] = str(sock)
    except OSError:  # unreadable HOME: fall back to whatever the parent had
        pass
    return env


def ssh_command(alias: str, *args: str) -> list[str]:
    """A read-only ssh invocation.  `-A` matches the pods' ForwardAgent yes."""
    return ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", alias, *args]


# Shared by DETAIL_PROBE_SCRIPT and HANDRUN_PROBE_SCRIPT so the two probes read
# a tqdm line the same way (one implementation, two callers).
TQDM_SHELL_FN = r'''# Newest tqdm progress line in the last 4KB of a train.log.  tqdm rewrites one
# line with \r, so the tail is split on \r before matching.  Emits
# "step total secs_per_it" (secs_per_it may be empty while tqdm still says ?).
tqdm_of() {
  [ -f "$1" ] || return 0
  tail -c 4096 "$1" 2>/dev/null | tr '\r' '\n' \
    | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[^]]*\]' | tail -1 \
    | awk '{
        split($1, a, "/");
        sit = "";
        if (match($0, /[0-9]+(\.[0-9]+)?s\/it/)) {
          s = substr($0, RSTART, RLENGTH); sub(/s\/it/, "", s); sit = s;
        } else if (match($0, /[0-9]+(\.[0-9]+)?it\/s/)) {
          s = substr($0, RSTART, RLENGTH); sub(/it\/s/, "", s);
          if (s + 0 > 0) sit = sprintf("%.4f", 1 / (s + 0));
        }
        printf "%s %s %s", a[1], a[2], sit;
      }'
}
'''

# Fed to `ssh <alias> bash -s -- <profile> <arms>`.  Everything here is a read:
# sed/stat/find/tail/grep over small files.  The four-field line is printed
# BEFORE any detail work so that today's rendering survives a failure below it.
# Kept byte-for-byte compatible with ops/probe_unit.sh's phase logic.
_DETAIL_PROBE_HEAD = r'''set -uo pipefail
PROFILE=${1:?profile}
ARMS=${2:?arms}
ROOT=${FINAL_V1_ROOT:-/workspace/final_v1}
SAFE_ARMS=${ARMS//,/+}
STATUS="/workspace/logs/dfv1_${PROFILE}__${SAFE_ARMS}.status"
PIDFILE="/workspace/logs/dfv1_${PROFILE}__${SAFE_ARMS}.pid"
LOGFILE="/workspace/logs/dfv1_${PROFILE}__${SAFE_ARMS}.log"
NOW=$(date +%s)

runner_state=UNKNOWN
[ -f "$STATUS" ] && runner_state=$(sed -n 's/^state=//p' "$STATUS" | head -1)
procs=0
if [ -s "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then procs=1; fi
newest=0
[ -f "$LOGFILE" ] && newest=$(stat -c %Y "$LOGFILE")
for arm_root in "$ROOT/$PROFILE"/*; do
  [ -d "$arm_root" ] || continue
  seen=$(find "$arm_root" -type f \( -name '*.log' -o -name '*_COMPLETE.json' \) \
    -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1)
  [ "${seen:-0}" -gt "$newest" ] && newest=$seen
done
[ "$newest" -gt 0 ] && log_age=$(( NOW - newest )) || log_age=-1

phase_of() {
  run=$1
  phase=mix
  if [ -f "$run/CHAIN_COMPLETE.json" ]; then
    phase=done
  elif [ ! -f "$run/MIX_COMPLETE.json" ]; then phase=mix
  elif [ ! -f "$run/MIDTRAIN_COMPLETE.json" ]; then phase=midtrain
  elif [ ! -f "$run/DOLCI_COMPLETE.json" ]; then phase=dolci
  else
    aft=$(find "$run/aft" -mindepth 2 -maxdepth 2 -name AFT_COMPLETE.json 2>/dev/null | wc -l)
    if [ "$aft" -lt 4 ]; then phase="aft:$aft/4"
    elif [ ! -f "$run/EVAL_COMPLETE.json" ]; then phase=eval
    elif [ ! -f "$run/RECALL_COMPLETE.json" ]; then phase=recall
    elif [ ! -f "$run/D4_COMPLETE.json" ]; then phase=d4
    elif [ ! -f "$run/COSTSWEEP_COMPLETE.json" ]; then phase=costsweep
    elif [ ! -f "$run/PUBLISH_COMPLETE.json" ]; then phase=publish
    else phase=finalize
    fi
  fi
  printf '%s' "$phase"
}

phases=""
IFS=',' read -r -a ARM_LIST <<<"$ARMS"
ARM_PHASE=()
for arm in "${ARM_LIST[@]}"; do
  p=$(phase_of "$ROOT/$PROFILE/$arm")
  ARM_PHASE+=("$p")
  phases="${phases}${phases:+,}${arm}:$p"
done

# --- the classic line, first and unconditionally -------------------------- #
printf '%s|%s|%s|%s\n' "$runner_state" "$procs" "$log_age" "$phases"

# --- additive detail pass ------------------------------------------------- #
'''

_DETAIL_PROBE_TAIL = r'''jnum() { case "${1:-}" in '' | *[!0-9.]*) printf 'null' ;; *) printf '%s' "$1" ;; esac; }
jstr() { printf '%s' "${1:-}" | tr -cd 'A-Za-z0-9_.:+/ -'; }

idx=0
for arm in "${ARM_LIST[@]}"; do
  ph=${ARM_PHASE[$idx]}
  idx=$(( idx + 1 ))
  run="$ROOT/$PROFILE/$arm"
  stage=${ph%%:*}
  step=''; total=''; sit=''; files=''; dage=''; cdone=''; cells=''
  case "$stage" in
    midtrain | dolci)
      read -r step total sit <<<"$(tqdm_of "$run/$stage/train.log")"
      ;;
    aft)
      cdone=${ph#aft:}; cdone=${cdone%%/*}
      for cell_dir in "$run"/aft/*/; do
        [ -d "$cell_dir" ] || continue
        [ -f "${cell_dir}AFT_COMPLETE.json" ] && continue
        cell_log="${cell_dir}train.log"
        [ -f "$cell_log" ] || continue
        # "still growing": written to inside the last 15 minutes.  A stalled or
        # crashed cell drops out rather than reporting a frozen step count.
        mt=$(stat -c %Y "$cell_log" 2>/dev/null || printf '0')
        [ $(( NOW - mt )) -le 900 ] || continue
        cs=''; ct=''; csit=''
        read -r cs ct csit <<<"$(tqdm_of "$cell_log")"
        [ -n "$cs" ] || continue
        cname=$(basename "$cell_dir")
        cells="${cells}${cells:+,}{\"name\":\"$(jstr "$cname")\",\"step\":$(jnum "$cs"),\"total\":$(jnum "$ct"),\"sit\":$(jnum "$csit")}"
      done
      ;;
    eval | recall | d4 | costsweep)
      files=$(find "$run/$stage" -name '*.jsonl' 2>/dev/null | wc -l)
      ;;
    mix)
      dnew=$(find "$run/data" -type f -printf '%T@\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d. -f1)
      [ -n "${dnew:-}" ] && dage=$(( NOW - dnew ))
      ;;
  esac
  printf 'DETAIL {"arm":"%s","phase":"%s","stage":"%s","step":%s,"total":%s,"sit":%s,"files":%s,"data_age":%s,"cells_done":%s,"cells":[%s]}\n' \
    "$(jstr "$arm")" "$(jstr "$ph")" "$(jstr "$stage")" \
    "$(jnum "$step")" "$(jnum "$total")" "$(jnum "$sit")" \
    "$(jnum "$files")" "$(jnum "$dage")" "$(jnum "$cdone")" "$cells"
done
exit 0
'''

# The classic-line half, the shared tqdm reader, then the detail half.
DETAIL_PROBE_SCRIPT = _DETAIL_PROBE_HEAD + TQDM_SHELL_FN + _DETAIL_PROBE_TAIL

# Fed to `ssh <alias> bash -s -- <status_log> <progress_root>` for a hand-run
# unit.  Same contract as the campaign probe: every command is a read
# (stat/tail/find/grep), nothing is created or modified, and it always exits 0
# so a missing path degrades to "unknown" instead of an ssh failure.  Output is
# three pipe-delimited line kinds, each optional:
#
#   STATUS|<newest timestamped line of the status log>
#   LOGAGE|<seconds since that log was written, or -1>
#   PROG|<rel-path>|<step>|<total>|<s-per-it>|<loss>|<age-seconds>
#
# Pipes are stripped from the payload text so the delimiter cannot be forged by
# log content.
HANDRUN_PROBE_SCRIPT = r'''set -uo pipefail
LOG=${1:--}
PROOT=${2:--}
NOW=$(date +%s)

san() { tr -cd '[:print:]' | tr '|' '/' | cut -c1-200; }

''' + TQDM_SHELL_FN + r'''
status=''
age=-1
if [ "$LOG" != "-" ] && [ -f "$LOG" ]; then
  mtime=$(stat -c %Y "$LOG" 2>/dev/null || printf '%s' "$NOW")
  age=$(( NOW - mtime ))
  blob=$(tail -c 16384 "$LOG" 2>/dev/null | tr '\r' '\n')
  # Prefer the runner's own "[2026-...] arm: ..." line; fall back to whatever
  # was written last (a traceback, say) rather than showing nothing.
  status=$(printf '%s\n' "$blob" | grep -E '^\[[0-9]{4}-' | tail -1)
  [ -n "$status" ] || status=$(printf '%s\n' "$blob" | grep -vE '^[[:space:]]*$' | tail -1)
fi
printf 'STATUS|%s\n' "$(printf '%s' "$status" | san)"
printf 'LOGAGE|%s\n' "$age"

if [ "$PROOT" != "-" ] && [ -d "$PROOT" ]; then
  newest=$(find "$PROOT" -maxdepth 3 -name train.log -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1)
  path=${newest#* }
  if [ -n "$path" ] && [ -f "$path" ]; then
    rel=${path#"$PROOT"/}; rel=${rel%/train.log}
    mt=$(stat -c %Y "$path" 2>/dev/null || printf '0')
    page=$(( NOW - mt ))
    step=''; total=''; sit=''
    read -r step total sit <<<"$(tqdm_of "$path")"
    loss=$(tail -c 4096 "$path" 2>/dev/null | tr '\r' '\n' \
      | grep -oE "'loss': '[0-9.eE+-]+'" | tail -1 | awk -F"'" '{print $4}')
    printf 'PROG|%s|%s|%s|%s|%s|%s\n' \
      "$(printf '%s' "$rel" | san)" "$step" "$total" "$sit" "$loss" "$page"
  fi
fi
exit 0
'''


@dataclass(frozen=True)
class CellDetail:
    """One AFT cell that is currently writing to its train.log."""

    name: str
    step: int | None = None
    total: int | None = None
    sit: float | None = None

    @property
    def fraction(self) -> float | None:
        if self.step is None or not self.total:
            return None
        return max(0.0, min(1.0, self.step / self.total))

    @property
    def eta_seconds(self) -> float | None:
        if self.step is None or not self.total or not self.sit:
            return None
        return max(0.0, (self.total - self.step) * self.sit)


@dataclass(frozen=True)
class ArmDetail:
    """Intra-stage progress for one arm.  Every field is optional by design:
    pods sit at different stages and a missing field just means "not known"."""

    arm: str
    phase: str = ""
    stage: str = ""
    step: int | None = None
    total: int | None = None
    sit: float | None = None
    files: int | None = None
    data_age: int | None = None
    cells_done: int | None = None
    cells: tuple[CellDetail, ...] = ()


@dataclass
class Probe:
    phases: str = "?"
    runner_state: str = "?"
    procs: str = "?"
    log_age: int | None = None
    ok: bool = False
    error: str = ""
    details: dict[str, ArmDetail] = field(default_factory=dict)
    detail_error: str = ""


def _opt_int(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _opt_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def parse_detail_lines(lines: list[str]) -> dict[str, ArmDetail]:
    """`DETAIL {json}` lines -> {arm: ArmDetail}.  A bad line is dropped, never
    raised: detail is additive and must not turn into a new failure mode."""
    out: dict[str, ArmDetail] = {}
    for line in lines:
        try:
            obj = json.loads(line[len(DETAIL_PREFIX):])
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        arm = str(obj.get("arm") or "").strip()
        if not arm:
            continue
        cells: list[CellDetail] = []
        raw_cells = obj.get("cells")
        if isinstance(raw_cells, list):
            for cell in raw_cells:
                if not isinstance(cell, dict):
                    continue
                cells.append(CellDetail(
                    name=str(cell.get("name") or "?"),
                    step=_opt_int(cell.get("step")),
                    total=_opt_int(cell.get("total")),
                    sit=_opt_float(cell.get("sit")),
                ))
        out[arm] = ArmDetail(
            arm=arm,
            phase=str(obj.get("phase") or ""),
            stage=str(obj.get("stage") or ""),
            step=_opt_int(obj.get("step")),
            total=_opt_int(obj.get("total")),
            sit=_opt_float(obj.get("sit")),
            files=_opt_int(obj.get("files")),
            data_age=_opt_int(obj.get("data_age")),
            cells_done=_opt_int(obj.get("cells_done")),
            cells=tuple(cells),
        )
    return out


def probe_unit(rec: PodRecord) -> Probe:
    """One read-only ssh per active pod.  Never retried inside a refresh."""
    cmd = ssh_command(rec.ssh_alias, "bash", "-s", "--", rec.profile, rec.arms)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
            input=DETAIL_PROBE_SCRIPT, env=ssh_env(),
        )
    except subprocess.TimeoutExpired:
        return Probe(phases="unreachable", error="ssh timeout")
    except OSError as exc:
        return Probe(phases="unreachable", error=f"ssh {type(exc).__name__}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    detail_lines = [ln for ln in lines if ln.startswith(DETAIL_PREFIX)]
    plain = [ln for ln in lines if not ln.startswith(DETAIL_PREFIX)]
    if proc.returncode != 0 or not plain:
        tail = (proc.stderr or "").strip().splitlines()
        return Probe(phases="unreachable",
                     error=tail[-1][:60] if tail else f"rc={proc.returncode}")
    parts = plain[-1].split("|")
    if len(parts) < 4:
        return Probe(phases="unparsed", error=plain[-1][:60])
    try:
        age = int(parts[2])
    except ValueError:
        age = None
    try:
        details = parse_detail_lines(detail_lines)
        detail_error = "" if details else ("no detail payload" if detail_lines
                                           else "detail lines absent")
    except Exception as exc:  # noqa: BLE001 - detail can never break the probe
        details, detail_error = {}, f"detail {type(exc).__name__}"
    return Probe(
        phases=parts[3].strip(),
        runner_state=parts[0].strip(),
        procs=parts[1].strip(),
        log_age=age,
        ok=True,
        details=details,
        detail_error=detail_error,
    )


def fetch_deadman(alias: str) -> str | None:
    cmd = ssh_command(alias, "cat", "/etc/runpod-deadman.json")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=DEADMAN_TIMEOUT_S, stdin=subprocess.DEVNULL,
                              env=ssh_env())
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
    except Exception:  # noqa: BLE001 - optional source, fail silently by design
        return None
    val = data.get("deadline_utc") if isinstance(data, dict) else None
    return str(val) if val else None


# --------------------------------------------------------------------------- #
# Hand-run (unmanaged) units -- see the module docstring
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HandRunCfg:
    """One declared row of ops/handrun_units.tsv."""

    label: str
    account: str
    pod_id: str = ""
    ssh_alias: str = ""
    status_log: str = ""
    progress_root: str = ""
    note: str = ""
    protocol: str = ""


def _opt_field(value: str) -> str:
    """'-' is the file's "not applicable" marker; so is an empty cell."""
    text = value.strip()
    return "" if text in ("", "-") else text


def read_handrun_units(path: Path = HANDRUN_FILE) -> list[HandRunCfg]:
    """Parse the declarative hand-run table.  Never raises.

    A missing file means "no hand-run units" (the normal case once the last
    hand-launched pod is gone).  A malformed row is skipped rather than fatal:
    this table exists to make live runs visible, so one bad line must not take
    the other rows -- or the rest of the dashboard -- down with it.
    """
    rows: list[HandRunCfg] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < HANDRUN_FIELDS:
            fields = fields + [""] * (HANDRUN_FIELDS - len(fields))
        label = fields[0].strip()
        if not label:
            continue
        rows.append(HandRunCfg(
            label=label,
            account=_opt_field(fields[1]) or "?",
            pod_id=_opt_field(fields[2]),
            ssh_alias=_opt_field(fields[3]),
            status_log=_opt_field(fields[4]),
            progress_root=_opt_field(fields[5]),
            note=fields[6].strip(),
            protocol=fields[7].strip() if len(fields) > 7 else "",
        ))
    return rows


@dataclass
class HandRunProbe:
    """What one read-only ssh learned about a hand-run pod.  All optional."""

    ok: bool = False
    error: str = ""
    status_line: str = ""
    log_age: int | None = None
    stage: str = ""          # rel path of the newest train.log: the arm/stage
    step: int | None = None
    total: int | None = None
    sit: float | None = None
    loss: float | None = None
    stage_age: int | None = None
    stage_index: int | None = None
    stage_total: int | None = None
    cell_index: int | None = None
    cell_total: int | None = None
    cells_done: int | None = None
    unit: str = "steps"
    endpoints: list[dict] = field(default_factory=list)
    elapsed_seconds: float | None = None
    remaining_seconds: float | None = None
    timing_note: str = ""

    @property
    def fraction(self) -> float | None:
        if self.step is None or not self.total:
            return None
        return max(0.0, min(1.0, self.step / self.total))

    @property
    def eta_seconds(self) -> float | None:
        if self.stage_total:
            return self.remaining_seconds
        if self.step is None or not self.total or not self.sit:
            return None
        return max(0.0, (self.total - self.step) * self.sit)

    @property
    def heartbeat_age(self) -> int | None:
        """Seconds since the newest write to *either* watched file.

        The status log is only appended at stage boundaries, so on its own it
        looks stale for hours during a healthy midtrain; the train.log under
        progress_root is the real heartbeat.  Take whichever is newer.
        """
        ages = [a for a in (self.log_age, self.stage_age) if a is not None and a >= 0]
        return min(ages) if ages else None

    @property
    def progress_text(self) -> str:
        """'midtrain 179/1351 @34.1s loss 1.096', or '' when nothing is known."""
        if not self.stage and self.step is None:
            return ""
        bits = [self.stage] if self.stage else []
        if self.stage_index and self.stage_total:
            bits.insert(0, f"{self.stage_index}/{self.stage_total}")
        if self.step is not None and self.total:
            bits.append(f"{self.step}/{self.total}")
            if self.stage_total:
                bits.append(self.unit)
        if self.sit:
            bits.append(f"@{self.sit:.1f}s")
        if self.loss is not None:
            bits.append(f"loss {self.loss:g}")
        return " ".join(bits)


def parse_handrun_output(text: str) -> HandRunProbe:
    """STATUS/LOGAGE/PROG lines -> HandRunProbe.  Unknown lines are ignored."""
    probe = HandRunProbe(ok=True)
    for line in text.splitlines():
        if line.startswith("PIPELINE|"):
            payload = json.loads(line.partition("|")[2])
            for key in ("stage", "step", "total", "sit", "loss", "stage_age",
                        "stage_index", "stage_total", "cell_index", "cell_total",
                        "cells_done", "unit", "endpoints", "status_line",
                        "elapsed_seconds", "remaining_seconds", "timing_note"):
                if key in payload:
                    setattr(probe, key, payload[key])
        elif line.startswith("STATUS|"):
            probe.status_line = line[len("STATUS|"):].strip()
        elif line.startswith("LOGAGE|"):
            age = _opt_int(line[len("LOGAGE|"):].strip())
            probe.log_age = None if age is None or age < 0 else age
        elif line.startswith("PROG|"):
            parts = line.split("|")
            if len(parts) < 7:
                continue
            probe.stage = parts[1].strip()
            probe.step = _opt_int(parts[2].strip() or None)
            probe.total = _opt_int(parts[3].strip() or None)
            probe.sit = _opt_float(parts[4].strip() or None)
            probe.loss = _opt_float(parts[5].strip() or None)
            probe.stage_age = _opt_int(parts[6].strip() or None)
    return probe


def probe_handrun(cfg: HandRunCfg) -> HandRunProbe:
    """One read-only ssh per hand-run pod.  Failure is data, not an exception."""
    if not cfg.ssh_alias:
        return HandRunProbe(error="no ssh alias declared")
    cmd = ssh_command(cfg.ssh_alias, "bash", "-s", "--",
                      cfg.status_log or "-", cfg.progress_root or "-")
    script = HANDRUN_PROBE_SCRIPT
    if cfg.protocol in ("aft_size_mixture_v1", "gemma_grid"):
        cmd = ["ssh", "-o", "IdentityAgent=none", "-o", "IdentitiesOnly=yes",
               "-i", str(Path.home() / ".ssh/id_ed25519"), "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=10", cfg.ssh_alias, "python3", "-",
               cfg.progress_root, cfg.status_log or "-"]
        script = Path(__file__).with_name("gemma_grid_probe.py" if cfg.protocol == "gemma_grid" else "aft_size_probe.py").read_text()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=HANDRUN_TIMEOUT_S,
            input=script, env=ssh_env(),
        )
    except subprocess.TimeoutExpired:
        return HandRunProbe(error="ssh timeout")
    except OSError as exc:
        return HandRunProbe(error=f"ssh {type(exc).__name__}")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return HandRunProbe(error=tail[-1][:60] if tail else f"rc={proc.returncode}")
    try:
        return parse_handrun_output(proc.stdout)
    except Exception as exc:  # noqa: BLE001 - a parse bug must not kill the frame
        return HandRunProbe(error=f"parse {type(exc).__name__}")


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
class HandRunView:
    """A declared hand-run unit, plus whatever the account query and the probe
    could tell us about it.  Both are optional: the row renders either way."""

    cfg: HandRunCfg
    probe: HandRunProbe | None = None
    pod_status: str | None = None   # RunPod desiredStatus, e.g. "RUNNING"
    hourly_rate: float | None = None
    pod_seen: bool = False          # was the id found in that account's pods?
    account_error: str | None = None

    @property
    def running(self) -> bool | None:
        """True/False per the account query; None when we could not ask."""
        if self.account_error or not self.cfg.pod_id:
            return None
        if not self.pod_seen:
            return False
        return self.pod_status == "RUNNING"


@dataclass
class Snapshot:
    taken_at: float
    accounts: list[AccountView] = field(default_factory=list)
    campaigns: list[CampaignView] = field(default_factory=list)
    handruns: list[HandRunView] = field(default_factory=list)
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

        # 1b. Hand-run units: declared, not discovered (they have no ledger).
        handruns = [HandRunView(cfg=cfg) for cfg in read_handrun_units()]

        # 2. Balances, in parallel across accounts (one POST each).
        accounts = [AccountView(cfg=cfg) for cfg in ACCOUNTS]
        with futures.ThreadPoolExecutor(max_workers=len(accounts)) as pool:
            for view, state in zip(accounts, pool.map(fetch_account, [a.cfg for a in accounts])):
                view.state = state
                if state.error:
                    errors.append(f"{view.cfg.label} api: {state.error}")

        # 2b. Reconcile each hand-run pod id against its account's pod list.
        #     A hand-run unit knows which account pays it (the GLM row spans
        #     two), so this is a per-account lookup, not a global one.
        by_label = {a.cfg.label: a for a in accounts}
        for hand in handruns:
            acct = by_label.get(hand.cfg.account)
            if acct is None:
                hand.account_error = f"unknown account {hand.cfg.account}"
                continue
            if acct.state.error:
                hand.account_error = acct.state.error
                continue
            for pod in acct.state.pods:
                if pod.get("id") and pod.get("id") == hand.cfg.pod_id:
                    hand.pod_seen = True
                    hand.pod_status = str(pod.get("desiredStatus") or "")
                    hand.hourly_rate = float(pod.get("costPerHr") or 0.0)
                    break

        # 3. One ssh probe per active pod, every campaign at once.
        targets = [u for c in campaigns for u in c.units
                   if u.rec.state in PROBE_STATES and u.rec.ssh_alias not in ("", "-")]
        if targets:
            with futures.ThreadPoolExecutor(max_workers=min(16, len(targets))) as pool:
                for unit, probe in zip(targets, pool.map(lambda u: probe_unit(u.rec), targets)):
                    unit.probe = probe

        # 3b. One ssh probe per hand-run pod, in its own pool so a slow or dead
        #     hand-run host cannot delay (or fail) the campaign probes above.
        hand_targets = [h for h in handruns if h.cfg.ssh_alias]
        if hand_targets:
            with futures.ThreadPoolExecutor(max_workers=min(8, len(hand_targets))) as pool:
                for hand, probe in zip(hand_targets,
                                       pool.map(lambda h: probe_handrun(h.cfg), hand_targets)):
                    hand.probe = probe

        # 4. Optional dead-man deadlines (cached, >=10 min apart).
        aliases = [u.rec.ssh_alias for u in targets]
        if aliases:
            deadlines = self._deadman_for(aliases)
            for unit in targets:
                unit.deadline = deadlines.get(unit.rec.ssh_alias)

        return Snapshot(taken_at=time.time(), accounts=accounts, campaigns=campaigns,
                        handruns=handruns, errors=errors)


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


def fmt_duration(seconds: float | None) -> str:
    """Coarse, honest duration: 45s / 12m / 2h07m."""
    if seconds is None or seconds < 0:
        return "?"
    total = int(seconds)
    if total < 90:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"


# --------------------------------------------------------------------------- #
# Intra-stage progress: one reading of a DETAIL payload, shared by both views
# --------------------------------------------------------------------------- #


def phase_head(phase: str | None) -> str:
    """'aft:2/4' -> 'aft'; '' / None -> ''."""
    return (phase or "").split(":", 1)[0].strip()


def live_cells(detail: ArmDetail | None) -> list[CellDetail]:
    if detail is None:
        return []
    return [c for c in detail.cells if c.fraction is not None]


def stage_fraction(phase: str | None, detail: ArmDetail | None) -> float | None:
    """How far through the *current stage* this arm is, 0..1, or None.

    midtrain/dolci: step/total from the newest tqdm line.
    aft:            (cells already complete + the live cells' own fractions) / 4,
                    which is right whether the cells run 4-up or one at a time.
    Batteries and mix have no honest denominator, so they return None and the
    caller leaves the bar at the start of the segment.
    """
    if detail is None:
        return None
    head = phase_head(phase)
    if head in TRAINING_STAGES:
        if detail.step is not None and detail.total:
            return max(0.0, min(1.0, detail.step / detail.total))
        return None
    if head == "aft":
        done = detail.cells_done
        if done is None:
            return None
        live = sum(c.fraction or 0.0 for c in live_cells(detail))
        return max(0.0, min(1.0, (done + live) / AFT_CELLS))
    return None


def stage_eta_seconds(phase: str | None, detail: ArmDetail | None) -> float | None:
    """remaining_steps x s/it for the work in flight.  Honest arithmetic only:
    it is the ETA of the *stage* (for aft, of the slowest live cell), never a
    guess at the whole row."""
    if detail is None:
        return None
    head = phase_head(phase)
    if head in TRAINING_STAGES:
        if detail.step is not None and detail.total and detail.sit:
            return max(0.0, (detail.total - detail.step) * detail.sit)
        return None
    if head == "aft":
        etas = [c.eta_seconds for c in detail.cells if c.eta_seconds is not None]
        return max(etas) if etas else None
    return None


def arm_progress_text(phase: str | None, detail: ArmDetail | None) -> str:
    """Compact intra-stage suffix for the TUI: '104/381@14.7s', '162f', ''."""
    if detail is None:
        return ""
    head = phase_head(phase)
    if head in TRAINING_STAGES:
        if detail.step is None or not detail.total:
            return ""
        text = f"{detail.step}/{detail.total}"
        return text + (f"@{detail.sit:.1f}s" if detail.sit else "")
    if head == "aft":
        live = live_cells(detail)
        if not live:
            return ""
        sits = [c.sit for c in live if c.sit]
        rate = f"@{sum(sits) / len(sits):.1f}s" if sits else ""
        if len(live) == 1:
            return f"{live[0].step}/{live[0].total}{rate}"
        totals = {c.total for c in live}
        if len(totals) == 1:
            steps = [c.step for c in live if c.step is not None]
            lo, hi = min(steps), max(steps)
            span = f"{lo}" if lo == hi else f"{lo}-{hi}"
            return f"{len(live)}x {span}/{live[0].total}{rate}"
        return f"{len(live)}x " + "+".join(f"{c.step}/{c.total}" for c in live[:2]) + rate
    if head in BATTERY_STAGES:
        return "" if detail.files is None else f"{detail.files}f"
    if head == "mix":
        return "" if detail.data_age is None else f"data {fmt_duration(detail.data_age)}"
    return ""


def phases_text(probe: Probe | None) -> str:
    """The PER-ARM PHASE cell, enriched with intra-stage progress where known.

    'charter:done  coin:midtrain 104/381@14.7s  control:mix data 3m'
    With no detail payload this returns the probe's own comma-joined string
    re-spaced -- i.e. exactly today's information.
    """
    if probe is None:
        return "-"
    if not probe.ok:
        return probe.phases
    parts: list[str] = []
    for chunk in probe.phases.split(","):
        arm, sep, phase = chunk.partition(":")
        if not sep:
            parts.append(chunk)
            continue
        extra = arm_progress_text(phase.strip(), probe.details.get(arm.strip()))
        parts.append(f"{chunk} {extra}" if extra else chunk)
    return "  ".join(parts)


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
    external_pods: list[dict]  # known third-party pods (krill-mill), A1 only
    external_cost: float
    unclaimed: list[dict]  # dfv1- pods that nothing claims -> alarming
    noncampaign: list[dict]  # non-dfv1 pods (e.g. b3xx-glmtest-*) -> informative
    noncampaign_cost: float
    handrun: list[dict]  # pods declared in ops/handrun_units.tsv
    handrun_cost: float
    error: str | None


def claimed_pod_ids(snap: "Snapshot") -> dict[str, UnitView]:
    """pod id -> the ledger row that owns it, across every campaign."""
    return {
        u.rec.pod_id: u
        for c in snap.campaigns for u in c.units
        if u.rec.pod_id and u.rec.pod_id != "-"
    }


def handrun_pod_ids(snap: "Snapshot") -> set[str]:
    """pod ids declared in ops/handrun_units.tsv."""
    return {h.cfg.pod_id for h in snap.handruns if h.cfg.pod_id}


def accounted_pod_ids(snap: "Snapshot") -> set[str]:
    """Every pod id somebody's name is on: ledger rows plus hand-run rows.

    This is what "unclaimed" is measured against.  Before the hand-run table
    existed, three hand-launched dfv1- pods showed up here as UNCLAIMED alarms
    -- real spend, but not actually unowned, just unrecorded.
    """
    return set(claimed_pod_ids(snap)) | handrun_pod_ids(snap)


def rollup_account(view: AccountView, claimed: set[str],
                   handrun: set[str] | None = None) -> AccountRollup:
    """Balance/burn/runway for one account.

    Four kinds of running pod, all of them real money leaving the account and
    so all of them in `burn`, but each read differently:

      * hand-run      -- declared in ops/handrun_units.tsv: owned by a human,
                         with no supervisor.  Checked FIRST, so the same pod
                         cannot also be reported as unclaimed or as somebody
                         else's errand;
      * ours          -- a dfv1- pod; if nothing claims it that is ALARMING,
                         because it is campaign-shaped spend with nobody's name
                         on it (`unclaimed`);
      * external      -- a known third-party pod we deliberately tolerate
                         (krill-mill on A1, by id);
      * non-campaign  -- anything else, e.g. the b3xx-glmtest-* throughput pods
                         another agent runs on A3.  Legitimately absent from
                         every ledger, so it is reported, not alarmed on.
    """
    handrun_ids = handrun or set()
    ours: list[dict] = []
    external: list[dict] = []
    noncampaign: list[dict] = []
    unclaimed: list[dict] = []
    hand_pods: list[dict] = []
    ext_cost = 0.0
    non_cost = 0.0
    hand_cost = 0.0
    for pod in view.state.pods:
        if pod.get("desiredStatus") != "RUNNING":
            continue
        if pod.get("id") in handrun_ids:
            hand_pods.append(pod)
            hand_cost += float(pod.get("costPerHr") or 0.0)
        elif is_ours(pod):
            ours.append(pod)
            if pod.get("id") not in claimed:
                unclaimed.append(pod)
        elif pod.get("id") in EXTERNAL_POD_IDS:
            external.append(pod)
            ext_cost += float(pod.get("costPerHr") or 0.0)
        else:
            noncampaign.append(pod)
            non_cost += float(pod.get("costPerHr") or 0.0)
    burn = (sum(float(p.get("costPerHr") or 0.0) for p in ours)
            + ext_cost + non_cost + hand_cost)
    balance = view.state.balance
    runway = (balance / burn) if (balance is not None and burn > 0) else None
    return AccountRollup(
        label=view.cfg.label, balance=balance, burn=burn, runway_hours=runway,
        our_pods=ours, external_pods=external, external_cost=ext_cost,
        unclaimed=unclaimed, noncampaign=noncampaign, noncampaign_cost=non_cost,
        handrun=hand_pods, handrun_cost=hand_cost,
        error=view.state.error,
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


def _created_epoch(rec: PodRecord) -> float:
    created = parse_utc(rec.created_at)
    return created.timestamp() if created else 0.0


def is_finished(rec: PodRecord) -> bool:
    return rec.state in FINISHED_STATES


def live_sort_key(unit: UnitView) -> tuple:
    """Attention first (a parked pod is alive, billing, and waiting on a human),
    then running, then everything else; ties broken by campaign/profile/arms."""
    rec = unit.rec
    if rec.state in ATTENTION_STATES:
        rank = 0
    elif rec.state == "running":
        rank = 1
    elif rec.state in ACTIVE_STATES:
        rank = 2
    else:
        rank = 3  # not active, not finished: e.g. a stale/unknown ledger state
    return (rank, unit.campaign, rec.profile, rec.arms)


def finished_sort_key(unit: UnitView) -> tuple:
    """Most recently finished first -- approximated by created-at, descending.

    The v2 ledger schema carries no finished-at stamp (state is rewritten in
    place on the row that was created), so created-at is the only honest
    ordering signal available; the section header says so.
    """
    return (-_created_epoch(unit.rec), unit.campaign, unit.rec.profile, unit.rec.arms)


def split_units(snap: "Snapshot") -> tuple[list[UnitView], list[UnitView]]:
    """(live, finished) across every campaign, each already sorted for display.

    "Live" is everything that is not done/cleaned -- including the states that
    need a human (parked/halted/lost/provision_failed), which is why they sort
    to the very top rather than into the history section.
    """
    every = [u for c in snap.campaigns for u in c.units]
    live = sorted((u for u in every if not is_finished(u.rec)), key=live_sort_key)
    finished = sorted((u for u in every if is_finished(u.rec)), key=finished_sort_key)
    return live, finished


def build_sections(snap: Snapshot, last_ok: float | None, last_error: str | None,
                   interval: int) -> list[Section]:
    now = time.time()
    sections: list[Section] = []

    # Ledger claims, for pod reconciliation and per-account campaign grouping.
    # `accounted` adds the hand-run table: those pods have an owner, they just
    # have no ledger row, so they must not be alarmed on as unclaimed.
    claimed = claimed_pod_ids(snap)
    hand_ids = handrun_pod_ids(snap)
    accounted = set(claimed) | hand_ids

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
                   columns=["ACC", "BALANCE", "POD BURN $/hr", "RUNWAY", "OUR PODS",
                            "OTHER PODS", "CAMPAIGNS", "NOTE"])
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
        roll = rollup_account(view, accounted, hand_ids)
        if roll.runway_hours is not None:
            run_cell: Cell = (f"{roll.runway_hours:,.1f}h",
                              "bad" if roll.runway_hours < RUNWAY_RED_HOURS else
                              ("warn" if roll.runway_hours < RUNWAY_RED_HOURS * 2 else "ok"))
        else:
            run_cell = ("n/a" if roll.burn <= 0 else "?", "dim")
        # An unclaimed dfv1- pod is an alarm (campaign-shaped spend nobody owns);
        # a non-campaign pod is just news (another agent's errand on the card).
        notes: list[str] = []
        if roll.unclaimed:
            notes.append("UNCLAIMED dfv1 pod(s): " + ", ".join(
                f"{p.get('name')} (${float(p.get('costPerHr') or 0):.2f}/hr)"
                for p in roll.unclaimed))
        if roll.noncampaign:
            notes.append("non-campaign: " + ", ".join(
                f"{p.get('name')} (${float(p.get('costPerHr') or 0):.2f}/hr)"
                for p in roll.noncampaign))
        if roll.handrun:
            # Named from the declarative table, never from the pod's own name
            # (kgxwecxy3cqn8e is *named* ...-control-charter but runs coin).
            notes.append(f"hand-run: {len(roll.handrun)} pod(s) "
                         f"(${roll.handrun_cost:.2f}/hr)")
        note = "; ".join(notes)
        others = len(roll.external_pods) + len(roll.noncampaign) + len(roll.handrun)
        acct.rows.append([
            (view.cfg.label, "b"),
            (fmt_money(roll.balance), "b"),
            (f"{roll.burn:,.2f}", ""),
            run_cell,
            (str(len(roll.our_pods)), ""),
            (f"{others} (${roll.external_cost + roll.noncampaign_cost + roll.handrun_cost:.2f}"
             f"/hr)" if others else "-", "dim" if others else ""),
            (camp_txt, "dim"),
            (_trunc(note, 90), "bad" if roll.unclaimed else ("warn" if note else "")),
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
    # The other ten columns cost roughly 115 columns; give the phase cell the
    # rest, with a floor so a narrow pane still shows the active stage.
    phase_width = max(34, shutil.get_terminal_size((200, 50)).columns - 118)
    unit_columns = ["CAMPAIGN", "PROFILE", "ARMS", "STATE", "POD", "$/hr", "ATT",
                    "ELAPSED / EXPECTED", "RUNNER", "DEADMAN", "PER-ARM PHASE"]

    def unit_row(unit: UnitView) -> list[Cell]:
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
            # Style still comes from the raw probe string (detail must not be
            # able to change a verdict), only the text is enriched.
            phase_cell = (_trunc(phases_text(p), phase_width), phase_style(p.phases))
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
        return [
            (unit.campaign, "dim"),
            (rec.profile, ""),
            (rec.arms, "dim"),
            (rec.state, state_style(rec.state)),
            (rec.pod_id or "-", "dim"),
            (f"{rec.hourly_rate:.2f}", ""),
            (str(rec.attempt), "warn" if rec.attempt > 1 else "dim"),
            elapsed_cell, runner_cell, dead_cell, phase_cell,
        ]

    live_units, finished_units = split_units(snap)
    units = Section(title="LIVE WORK UNITS  (attention states first — a parked pod is alive and "
                          "billing; elapsed is wall clock since the ledger's created-at)",
                    columns=unit_columns,
                    empty_note="(no live units in any campaign ledger)")
    units.rows = [unit_row(u) for u in live_units]
    sections.append(units)

    # Live pods nothing claims -- neither a ledger row nor the hand-run table:
    # real money with nobody's name on it.
    orphans: list[list[Cell]] = []
    for view in snap.accounts:
        for pod in view.state.pods:
            if pod.get("desiredStatus") != "RUNNING" or not is_ours(pod):
                continue
            if pod.get("id") in accounted:
                continue
            orphans.append([
                (f"  ! {view.cfg.label} pod {pod.get('id')} {pod.get('name')} "
                 f"${float(pod.get('costPerHr') or 0):.2f}/hr is running but no campaign "
                 f"ledger or hand-run row claims it", "bad")
            ])
    units.lines.extend(orphans)

    # ---- hand-run (unmanaged) units --------------------------------------- #
    # Declared in ops/handrun_units.tsv, not discovered: these have no
    # supervisor, no queue and no ledger row, and the GLM row's three arms sit
    # on two different accounts.  The LABEL column is the declared label, never
    # the RunPod pod name -- pod kgxwecxy3cqn8e is *named*
    # dfv1-glm-2tb-control-charter but runs the COIN arm, so showing pod names
    # here would mislabel a live scientific run.
    hand = Section(title="HAND-RUN UNITS  (no supervisor, no queue, no ledger row; labels come "
                         "from ops/handrun_units.tsv, NOT from the RunPod pod name)",
                   columns=["LABEL", "ACC", "POD", "$/hr", "POD STATE", "HEARTBEAT",
                            "PROGRESS", "NOTE"],
                   empty_note="(no hand-run units declared)")
    for view_hand in snap.handruns:
        cfg = view_hand.cfg
        running = view_hand.running
        if running is None:
            pod_cell: Cell = ("?", "warn")
        elif running:
            pod_cell = ("RUNNING", "ok")
        else:
            pod_cell = (view_hand.pod_status or "not found", "bad")
        rate_cell: Cell = (("?", "dim") if view_hand.hourly_rate is None
                           else (f"{view_hand.hourly_rate:.2f}", ""))
        probe = view_hand.probe
        if probe is None:
            age_cell: Cell = ("-", "dim")
            prog_cell: Cell = ("not probed", "dim")
        elif not probe.ok:
            age_cell = ("?", "bad")
            prog_cell = (_trunc(probe.error or "unreachable", 60), "bad")
        else:
            beat = probe.heartbeat_age
            age_cell = (("?", "warn") if beat is None
                        else (fmt_duration(beat), "warn" if beat > 1800 else ""))
            text = probe.progress_text
            if not text:
                text = probe.status_line or "no progress line yet"
            elif probe.status_line:
                text = f"{text}  |  {probe.status_line}"
            prog_cell = (_trunc(text, max(40, phase_width + 20)),
                         "info" if probe.progress_text else "dim")
            if probe.stage_total:
                timing = f"elapsed {fmt_duration(probe.elapsed_seconds)}"
                timing += f" · ~{fmt_duration(probe.eta_seconds)} left" if probe.eta_seconds is not None else " · ETA pending"
                prog_cell = (prog_cell[0] + "\n" + timing, prog_cell[1])
        hand.rows.append([
            (cfg.label, "b"),
            (cfg.account, ""),
            (cfg.pod_id or "-", "dim"),
            rate_cell,
            pod_cell,
            age_cell,
            prog_cell,
            (_trunc(cfg.note, 60), "dim"),
        ])
    sections.append(hand)

    # ---- finished units --------------------------------------------------- #
    # Below the live ones, always: history must never sit above a running row.
    done = Section(title="FINISHED UNITS  (done / cleaned — newest first by created-at; the "
                         "ledger records no finished-at)",
                   columns=unit_columns,
                   empty_note="(nothing finished yet)")
    done.rows = [unit_row(u) for u in finished_units]
    sections.append(done)

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
