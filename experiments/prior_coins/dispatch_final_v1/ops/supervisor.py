#!/usr/bin/env python3
"""Queue-driven RunPod supervisor for the nine Dispatch final-v1 rows.

Dry-run is the default-safe posture.  Execute mode provisions profile-sized
SECURE pods, bootstraps the pinned source commit, rehydrates every unit, and
monitors it until its durable completion marker and Hub contents are verified.
Only pods carrying the current campaign's private ownership token can enter
the cleanup helper.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import fcntl
import json
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Iterable

OPS = Path(__file__).resolve().parent
EXP = OPS.parent
PROFILES = EXP / "profiles"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import scheduler as S  # noqa: E402


FORBIDDEN_POD_ID = "lx6pucn0mfv8h3"
FORBIDDEN_POD_NAME = "krill-mill"
SKILL = Path.home() / ".claude" / "skills" / "runpod-spinup"
CREATE_POD = SKILL / "create-pod.sh"
CLEANUP_POD = SKILL / "cleanup-pod.sh"
PREFLIGHT_POD = SKILL / "pod-preflight.sh"
RESOLVE_SSH = SKILL / "_resolve_ssh.py"
SSH_ALIAS = SKILL / "_ssh_alias.py"

LEDGER_HEADER = """# Dispatch final-v1 supervisor-owned pod ledger (v2, tab-separated).
#
# Schema:
# state profile arms ssh-alias pod-id $/hr campaign-id owner-token attempt pod-name created-at strikes
#
# The old v1 format was: arm ssh-alias pod-id.  It did not identify the grid
# row or ownership, so those rows MUST NOT be copied into the active table.
# They are retained below as commented, unmanaged migration evidence.  A v2
# row is added only after this supervisor journals a create intent; teardown
# additionally requires its owner-token to match the current campaign file.
#
# Legacy rows (unmanaged; this supervisor will never inspect or delete them):
# legacy-unmanaged charter runpod-dispatch-charter raihf39qjoinnq
# legacy-unmanaged coin runpod-dispatch-coin q6zvpbwo3c5scy
# legacy-unmanaged control runpod-dispatch-control x6j1hgg6sg785q
"""

ACTIVE_STATES = frozenset(
    {"provisioning", "booting", "setting_up", "running", "finishing", "recovering",
     "parked"}
)
# "parked" = a pod that has FAILED but is being kept alive on purpose, for a
# person (or an agent acting as one) to look at before anything is destroyed.
# It is deliberately NOT in the set the run loop hands to cleanup(), so nothing
# automatic can delete it.  It stays in ACTIVE_STATES because the pod is still
# running and still billing: its rate must keep counting against the cap, and
# its unit must not be silently relaunched onto a second pod.
PARKED_STATE = "parked"
BRINGUP_STATES = frozenset({"provisioning", "booting", "setting_up"})
TERMINAL_STATES = frozenset({"done", "cleaned", "lost", "provision_failed"})


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def age_seconds(timestamp: str) -> float:
    return time.time() - datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    ).timestamp()


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    owner_token: str
    source_commit: str
    repo_url: str
    created_at: str


@dataclass(frozen=True)
class PodRecord:
    state: str
    profile: str
    arms: str
    ssh_alias: str
    pod_id: str
    hourly_rate: Decimal
    campaign_id: str
    owner_token: str
    attempt: int
    pod_name: str
    created_at: str
    strikes: int = 0

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return self.profile, S.parse_arms(self.arms)

    @property
    def identity(self) -> tuple[str, str, int, str]:
        return self.profile, self.arms, self.attempt, self.owner_token


@dataclass(frozen=True)
class Probe:
    outcome: str
    phase: str
    reason: str = ""


class PodLedger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def read(self) -> list[PodRecord]:
        with self._lock:
            records: list[PodRecord] = []
            if not self.path.exists():
                return records
            with self.path.open(newline="") as handle:
                for line_no, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
                    if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                        continue
                    if len(row) != 12:
                        raise ValueError(
                            f"{self.path}:{line_no}: expected 12 tab-separated fields, got {row!r}"
                        )
                    records.append(
                        PodRecord(
                            state=row[0], profile=row[1], arms=row[2], ssh_alias=row[3],
                            pod_id=row[4], hourly_rate=Decimal(row[5]), campaign_id=row[6],
                            owner_token=row[7], attempt=int(row[8]), pod_name=row[9],
                            created_at=row[10], strikes=int(row[11]),
                        )
                    )
            return records

    def _write(self, records: Iterable[PodRecord]) -> None:
        tmp = self.path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
        with tmp.open("w", newline="") as handle:
            handle.write(LEDGER_HEADER)
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            for record in records:
                writer.writerow(
                    [record.state, record.profile, record.arms, record.ssh_alias,
                     record.pod_id, f"{record.hourly_rate:.2f}", record.campaign_id,
                     record.owner_token, record.attempt, record.pod_name,
                     record.created_at, record.strikes]
                )
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.path)

    def append(self, record: PodRecord) -> None:
        with self._lock:
            records = self.read()
            if any(r.identity == record.identity for r in records):
                raise ValueError(f"duplicate pod-record identity {record.identity}")
            records.append(record)
            self._write(records)

    def update(self, identity: tuple[str, str, int, str], **changes) -> PodRecord:
        with self._lock:
            records = self.read()
            found = [i for i, record in enumerate(records) if record.identity == identity]
            if len(found) != 1:
                raise ValueError(f"expected one ledger row for {identity}, found {len(found)}")
            index = found[0]
            records[index] = replace(records[index], **changes)
            self._write(records)
            return records[index]


def latest_records(records: Iterable[PodRecord]) -> dict[tuple[str, tuple[str, ...]], PodRecord]:
    latest: dict[tuple[str, tuple[str, ...]], PodRecord] = {}
    for record in records:
        old = latest.get(record.key)
        if old is None or record.attempt > old.attempt:
            latest[record.key] = record
    return latest


def is_cleanup_target_owned(record: PodRecord, campaign: Campaign) -> tuple[bool, str]:
    """Pure destructive-path gate, kept small enough to test exhaustively."""
    if record.owner_token != campaign.owner_token or record.campaign_id != campaign.campaign_id:
        return False, "ownership token/campaign mismatch"
    if record.pod_id in {"", "-", FORBIDDEN_POD_ID}:
        return False, "missing or forbidden pod id"
    if record.pod_name == FORBIDDEN_POD_NAME:
        return False, "forbidden pod name"
    expected = f"dfv1-{campaign.campaign_id}-{campaign.owner_token[:8]}-"
    if not record.pod_name.startswith(expected):
        return False, f"pod name lacks campaign ownership prefix {expected!r}"
    if record.ssh_alias != f"runpod-{record.pod_name}":
        return False, "ssh alias does not match owned pod name"
    return True, "owned"


def ssh_host_of(repo_url: str) -> str | None:
    """Host to verify for an SSH clone URL; None when the URL is not SSH."""
    if repo_url.startswith("ssh://"):
        rest = repo_url[len("ssh://"):]
        return rest.split("@")[-1].split("/")[0].split(":")[0] or None
    if "://" in repo_url:
        return None  # https/git: nothing to verify by host key
    if ":" not in repo_url:
        return None
    return repo_url.split(":", 1)[0].split("@")[-1] or None


def verified_host_keys(repo_url: str) -> str:
    """This machine's *already verified* known_hosts lines for the repo host.

    Deliberately not ssh-keyscan: keyscan asks the network what the key is,
    which is the very thing an attacker would answer.  Empty string when the
    URL needs no host key, and a loud failure when it does and we have none --
    a silent empty value would fall straight back to trust-on-first-use.
    """
    host = ssh_host_of(repo_url)
    if host is None:
        return ""
    found = subprocess.run(
        ["ssh-keygen", "-F", host], text=True, capture_output=True,
    ).stdout
    lines = [l for l in found.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        raise SystemExit(
            f"FATAL: no verified host key for {host} in this machine's known_hosts. "
            f"Verify it here once (ssh -T git@{host}) before launching; pods must "
            f"not trust-on-first-use while holding a forwarded ssh-agent."
        )
    return "\n".join(lines)


def verified_host_keys_b64(repo_url: str) -> str:
    """`verified_host_keys` as one shell-safe token (see the bootstrap script)."""
    keys = verified_host_keys(repo_url)
    if not keys:
        return ""
    return base64.b64encode(keys.encode()).decode()


class Supervisor:
    def __init__(self, args: argparse.Namespace, units: list[S.WorkUnit], campaign: Campaign):
        self.args = args
        self.units = units
        self._churn_halted: set[tuple[str, str]] = set()
        self.by_key = {unit.key: unit for unit in units}
        self.campaign = campaign
        self.ledger = PodLedger(args.pods)
        self.log_lock = threading.Lock()
        self.workers = concurrent.futures.ThreadPoolExecutor(max_workers=9)
        self.futures: dict[tuple[str, str, int, str], concurrent.futures.Future] = {}
        self.log_path = args.runtime / "supervisor.log"
        if args.execute:
            args.runtime.mkdir(parents=True, exist_ok=True)

    def say(self, message: str) -> None:
        line = f"[{utcnow()}] {message}"
        with self.log_lock:
            print(line, flush=True)
            if self.args.execute:
                with self.log_path.open("a") as handle:
                    handle.write(line + "\n")

    def run_logged(
        self,
        cmd: list[str],
        *,
        timeout: float,
        label: str,
        input_text: str | None = None,
        line_callback: Callable[[str], None] | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        """Run with a wall timeout, streaming all output into the supervisor log."""
        self.say(f"{label}: START (timeout {timeout:.0f}s)")
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=env,
        )
        if input_text is not None:
            assert proc.stdin is not None
            proc.stdin.write(input_text)
            proc.stdin.close()
        assert proc.stdout is not None
        output: list[str] = []
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                self.say(f"{label}: FATAL TIMEOUT after {timeout:.0f}s")
                return 124, "".join(output)
            readable, _, _ = select.select([proc.stdout], [], [], min(1.0, remaining))
            if readable:
                line = proc.stdout.readline()
                if line:
                    output.append(line)
                    self.say(f"{label}: {line.rstrip()}")
                    if line_callback:
                        line_callback(line)
        tail = proc.stdout.read()
        if tail:
            output.append(tail)
            for line in tail.splitlines():
                self.say(f"{label}: {line}")
                if line_callback:
                    line_callback(line + "\n")
        rc = proc.returncode or 0
        self.say(f"{label}: {'DONE' if rc == 0 else 'FAILED'} rc={rc}")
        return rc, "".join(output)

    def short_run(
        self, cmd: list[str], *, timeout: float, input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd, input=input_text, text=True, capture_output=True, timeout=timeout,
            check=False,
        )

    def account(self) -> tuple[Decimal, Decimal] | None:
        try:
            result = self.short_run(["runpodctl", "me"], timeout=30)
            body = json.loads(result.stdout)
            return Decimal(str(body["clientBalance"])), Decimal(str(body["currentSpendPerHr"]))
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as exc:
            self.say(f"ACCOUNT UNAVAILABLE: {exc}; provisioning is disabled until it recovers")
            return None

    def check_live_price(self, unit: S.WorkUnit) -> bool:
        try:
            result = self.short_run(
                [str(SKILL / "gpu-prices.sh"), unit.shape.gpu_id], timeout=120
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.say(f"{unit.profile}: LIVE PRICE CHECK FAILED: {exc}")
            return False
        price = None
        for line in result.stdout.splitlines():
            if line.startswith(unit.shape.gpu_id):
                match = re.search(r"\$(\d+(?:\.\d+)?)", line[len(unit.shape.gpu_id):])
                if match:
                    price = Decimal(match.group(1))
                    break
        if result.returncode or price is None:
            self.say(
                f"{unit.profile}: LIVE PRICE CHECK FAILED rc={result.returncode}: "
                f"{(result.stderr or result.stdout)[-300:]}"
            )
            return False
        if price > unit.shape.per_gpu_rate:
            self.say(
                f"{unit.profile}: PRICE DRIFT: live SECURE {unit.shape.gpu_id} is "
                f"${price}/GPU/hr > reserved ${unit.shape.per_gpu_rate}; REFUSING TO CREATE"
            )
            return False
        if price < unit.shape.per_gpu_rate:
            self.say(
                f"{unit.profile}: live price ${price}/GPU/hr is below the conservative "
                f"${unit.shape.per_gpu_rate} reservation"
            )
        return True

    def list_pods(self) -> list[dict] | None:
        try:
            result = self.short_run(["runpodctl", "pod", "list", "-o", "json"], timeout=45)
            if result.returncode:
                raise ValueError(result.stderr[-300:])
            body = json.loads(result.stdout)
            if isinstance(body, list):
                return body
            for key in ("pods", "items"):
                if isinstance(body, dict) and isinstance(body.get(key), list):
                    return body[key]
            raise ValueError("unexpected pod-list JSON")
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as exc:
            self.say(f"POD LIST UNAVAILABLE: {exc}")
            return None

    def reconcile_intent(self, record: PodRecord, *, fresh: bool) -> PodRecord | None:
        pods = self.list_pods()
        if pods is None:
            return None
        matches = [pod for pod in pods if pod.get("name") == record.pod_name]
        if len(matches) > 1:
            self.say(f"FATAL: multiple pods have owned intent name {record.pod_name}; refusing")
            return None
        if matches:
            pod = matches[0]
            pod_id = str(pod.get("id") or "")
            if not pod_id or pod_id == FORBIDDEN_POD_ID:
                self.say(f"FATAL: intent {record.pod_name} resolved to forbidden/empty id")
                return None
            actual = Decimal(str(pod.get("costPerHr", record.hourly_rate)))
            if actual > record.hourly_rate:
                self.say(
                    f"{record.profile}/{record.arms}: adopted pod price ${actual} > "
                    f"reserved ${record.hourly_rate}; sending owned pod to cleanup"
                )
                return self.ledger.update(
                    record.identity, pod_id=pod_id, state="recovering"
                )
            self.say(f"RECOVERED CREATE INTENT: {record.pod_name} -> {pod_id}")
            return self.ledger.update(record.identity, pod_id=pod_id, state="booting")
        if not fresh and age_seconds(record.created_at) < self.args.create_reconcile_seconds:
            self.say(
                f"{record.pod_name}: create intent has no API row yet; waiting until "
                f"{self.args.create_reconcile_seconds}s reconciliation ceiling"
            )
            return None
        return record

    def reservation_guard(self) -> bool:
        account = self.account()
        if account is None:
            return False
        _, actual = account
        records = self.ledger.read()
        unrealized = sum(
            (r.hourly_rate for r in records
             if r.owner_token == self.campaign.owner_token
             and r.state in ACTIVE_STATES and r.pod_id == "-"),
            Decimal("0"),
        )
        if actual + unrealized > S.ACCOUNT_CAP:
            self.say(
                f"CAP GATE: actual ${actual}/hr + unrealized reservations "
                f"${unrealized}/hr > ${S.ACCOUNT_CAP}/hr; refusing create"
            )
            return False
        return True

    def create(self, record: PodRecord, unit: S.WorkUnit) -> PodRecord | None:
        if not self.check_live_price(unit) or not self.reservation_guard():
            return None
        cmd = [
            str(CREATE_POD), record.pod_name, unit.shape.gpu_id, unit.shape.cloud,
            unit.shape.template, str(unit.shape.n_gpus), str(unit.container_disk_gb),
            # The dead-man's switch is OFF unless asked for. Without it a pod
            # this supervisor loses track of -- crashed supervisor, dead ssh --
            # bills until a human notices, which at 27B is $36.72/hr. Safe to
            # arm because stages publish as they land and rehydrate.py restores
            # them onto a fresh pod, so a firing switch costs a relaunch and not
            # the row. create-pod.sh warns loudly if it cannot arm it.
            "--max-hours", str(unit.max_hours),
        ]
        seen_id: list[str] = []
        seen_cost: list[Decimal] = []

        def capture(line: str) -> None:
            pod_match = re.search(r"Pod ID:\s+([A-Za-z0-9]+)", line)
            if pod_match and not seen_id:
                pod_id = pod_match.group(1)
                if pod_id == FORBIDDEN_POD_ID:
                    self.say("FATAL: create helper returned forbidden krill-mill pod id")
                    return
                seen_id.append(pod_id)
                self.ledger.update(record.identity, pod_id=pod_id, state="booting")
                self.say(f"OWNERSHIP JOURNALED immediately: {record.pod_name} -> {pod_id}")
            cost_match = re.search(r"Cost:\s+\$(\d+(?:\.\d+)?)/hr", line)
            if cost_match:
                seen_cost.append(Decimal(cost_match.group(1)))

        rc, _ = self.run_logged(
            cmd, timeout=self.args.create_timeout, label=f"create:{record.pod_name}",
            line_callback=capture,
        )
        current = next(r for r in self.ledger.read() if r.identity == record.identity)
        if rc != 0 or current.pod_id == "-":
            state = "recovering" if current.pod_id != "-" else "provision_failed"
            self.ledger.update(record.identity, state=state)
            self.say(f"{record.pod_name}: create did not finish safely; state={state}")
            return None
        if seen_cost and seen_cost[-1] > record.hourly_rate:
            self.say(
                f"{record.pod_name}: created cost ${seen_cost[-1]}/hr > reserved "
                f"${record.hourly_rate}/hr; refusing workload and cleaning owned pod"
            )
            self.ledger.update(record.identity, state="recovering")
            return None
        return current

    def ensure_alias(self, record: PodRecord) -> bool:
        try:
            resolved = self.short_run(
                [sys.executable, str(RESOLVE_SSH), record.pod_id, "--quiet"], timeout=120
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.say(f"{record.pod_name}: SSH RESOLUTION FAILED: {exc}")
            return False
        parts = resolved.stdout.strip().split()
        if resolved.returncode or len(parts) != 2:
            self.say(f"{record.pod_name}: SSH RESOLUTION FAILED: {resolved.stderr[-300:]}")
            return False
        added = self.short_run(
            [sys.executable, str(SSH_ALIAS), "add", record.pod_name, record.pod_id,
             parts[0], parts[1]], timeout=60,
        )
        if added.returncode:
            self.say(f"{record.pod_name}: SSH ALIAS FAILED: {added.stderr[-300:]}")
            return False
        return True

    def preflight(self, record: PodRecord) -> bool:
        rc, output = self.run_logged(
            [str(PREFLIGHT_POD), record.pod_id], timeout=self.args.preflight_timeout,
            label=f"preflight:{record.pod_name}",
        )
        if rc or "PREFLIGHT VERDICT: FAIL" in output:
            self.say(f"{record.pod_name}: PREFLIGHT FAILED; workload will not launch")
            return False
        return True

    def bootstrap(self, record: PodRecord) -> bool:
        script = r'''set -Eeuo pipefail
repo_url=$1
source_commit=$2
setup_timeout=$3
known_hosts_b64=$4
# A fresh pod has no known_hosts, so an SSH clone dies with "Host key
# verification failed" before authentication is even attempted.  We install the
# host key THIS machine has already verified rather than letting the pod
# trust-on-first-use: the pod also holds a forwarded ssh-agent, so a spoofed
# forge on first contact would be handed live use of the key.
#
# base64, because ssh flattens the remote command into ONE string: a raw
# multi-line value arrives as extra *shell lines*, and a hashed known_hosts
# line begins with '|', so the first one it met was parsed as a pipeline
# ("syntax error near unexpected token `|'").
if [ -n "$known_hosts_b64" ]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  touch /root/.ssh/known_hosts && chmod 600 /root/.ssh/known_hosts
  printf '%s' "$known_hosts_b64" | base64 -d | while IFS= read -r line; do
    [ -n "$line" ] || continue
    grep -qxF "$line" /root/.ssh/known_hosts || printf '%s\n' "$line" \
      >>/root/.ssh/known_hosts
  done
fi
if [ -e /workspace/scimt ] && [ ! -d /workspace/scimt/.git ]; then
  echo "FATAL: /workspace/scimt exists but is not a git checkout; refusing to overwrite" >&2
  exit 66
fi
if [ ! -d /workspace/scimt/.git ]; then
  timeout --signal=TERM --kill-after=30 600 git clone --no-checkout "$repo_url" /workspace/scimt
fi
git -C /workspace/scimt fetch origin "$source_commit"
git -C /workspace/scimt checkout --detach "$source_commit"
timeout --signal=TERM --kill-after=120 "$setup_timeout" \
  bash /workspace/scimt/experiments/prior_coins/dispatch_final_v1/pod/setup.sh
'''
        cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=15", record.ssh_alias, "bash", "-s", "--",
            self.campaign.repo_url, self.campaign.source_commit,
            str(int(self.args.setup_timeout - 60)),
            verified_host_keys_b64(self.campaign.repo_url),
        ]
        rc, _ = self.run_logged(
            cmd, timeout=self.args.setup_timeout, label=f"setup:{record.pod_name}",
            input_text=script,
        )
        return rc == 0

    def launch(self, record: PodRecord) -> bool:
        env = os.environ.copy()
        rc, _ = self.run_logged(
            [str(OPS / "launch_unit.sh"), record.profile, record.arms, record.ssh_alias],
            timeout=120, label=f"launch:{record.pod_name}", env=env,
        )
        return rc == 0

    def bring_up(self, identity: tuple[str, str, int, str], *, fresh: bool) -> None:
        try:
            record = next(r for r in self.ledger.read() if r.identity == identity)
            unit = self.by_key[record.key]
            if record.state == "provisioning":
                record = self.reconcile_intent(record, fresh=fresh)
                if record is None:
                    return
                if record.state == "provisioning":
                    record = self.create(record, unit)
                    if record is None:
                        return
            if record.state == "recovering":
                return
            if record.pod_id in {"", "-"}:
                self.say(f"{record.pod_name}: FATAL bring-up without pod id")
                return
            if not self.ensure_alias(record) or not self.preflight(record):
                self.ledger.update(identity, state="recovering")
                return
            record = self.ledger.update(identity, state="setting_up", strikes=0)
            if not self.bootstrap(record):
                self.ledger.update(identity, state="recovering")
                return
            if not self.launch(record):
                self.ledger.update(identity, state="recovering")
                return
            self.ledger.update(identity, state="running", strikes=0)
            self.say(f"RUNNING: {record.profile}/{record.arms} on {record.pod_id}")
        except Exception as exc:  # noqa: BLE001 -- worker must leave a loud durable state
            self.say(f"FATAL bring-up exception for {identity}: {type(exc).__name__}: {exc}")
            try:
                current = next(r for r in self.ledger.read() if r.identity == identity)
                state = "recovering" if current.pod_id != "-" else "provision_failed"
                self.ledger.update(identity, state=state)
            except Exception as nested:  # noqa: BLE001
                self.say(f"FATAL could not journal bring-up failure: {nested}")

    def submit_bringup(self, record: PodRecord, *, fresh: bool) -> None:
        old = self.futures.get(record.identity)
        if old and not old.done():
            return
        self.futures[record.identity] = self.workers.submit(
            self.bring_up, record.identity, fresh=fresh
        )

    def pod_api_state(self, pod_id: str) -> tuple[str, str]:
        try:
            result = self.short_run(
                ["runpodctl", "pod", "get", pod_id, "-o", "json"], timeout=30
            )
            body = json.loads(result.stdout)
            if not isinstance(body, dict) or not body.get("id"):
                text = (result.stderr + result.stdout).lower()
                if "not found" in text or "already deleted" in text:
                    return "gone", text[-200:]
                return "unknown", text[-200:]
            if str(body.get("name")) == FORBIDDEN_POD_NAME:
                return "forbidden", "API name is krill-mill"
            return str(body.get("desiredStatus", "UNKNOWN")), ""
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as exc:
            return "unknown", str(exc)

    def probe(self, record: PodRecord) -> Probe:
        api_state, api_reason = self.pod_api_state(record.pod_id)
        if api_state == "gone":
            return Probe("lost", "pod-gone", api_reason)
        if api_state == "forbidden":
            return Probe("blocked", "FORBIDDEN", api_reason)
        if api_state not in {"RUNNING", "unknown"}:
            return Probe("failed", f"pod-{api_state.lower()}", api_reason)
        cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=15", record.ssh_alias, "bash",
            "/workspace/scimt/experiments/prior_coins/dispatch_final_v1/ops/probe_unit.sh",
            record.profile, record.arms,
        ]
        try:
            result = self.short_run(cmd, timeout=45)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return Probe("strike", "ssh-unreachable", str(exc))
        line = next((line for line in reversed(result.stdout.splitlines()) if line.count("|") == 3), "")
        if result.returncode or not line:
            return Probe("strike", "ssh-unreachable", (result.stderr or result.stdout)[-200:])
        runner, procs_s, log_age_s, phases = line.split("|", 3)
        phase = phases or runner.lower()
        try:
            procs, log_age = int(procs_s), int(log_age_s)
        except ValueError:
            return Probe("strike", phase, f"invalid probe counters: {line}")
        arm_phases = [item.rsplit(":", 1)[-1] for item in phases.split(",") if item]
        if arm_phases and all(value == "done" for value in arm_phases):
            return Probe("complete", phases)
        if runner == "FAILED":
            return Probe("failed", phase, "unit runner status is FAILED")
        if procs == 0:
            return Probe("strike", phase, "unit runner process is absent")
        if log_age > self.args.no_output_timeout:
            return Probe(
                "strike", phase,
                f"NO OUTPUT for {log_age}s > {self.args.no_output_timeout}s ceiling",
            )
        return Probe("running", phase, f"log age {log_age}s")

    def verify_hub(self, record: PodRecord) -> bool:
        code = r'''import json, sys
from huggingface_hub import HfApi
profile, arms = sys.argv[1:3]
files = HfApi().list_repo_files("arcadia-impact/scimt-dispatch-final-v1", repo_type="model")
out = {}
for arm in arms.split(","):
    prefix = arm + "/" if profile == "gemma3_12b_50m" else profile + "/" + arm + "/"
    out[arm] = sum(1 for path in files if path.startswith(prefix))
print(json.dumps(out))
'''
        try:
            result = self.short_run(
                [sys.executable, "-c", code, record.profile, record.arms], timeout=120
            )
            counts = json.loads(result.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as exc:
            self.say(f"{record.profile}/{record.arms}: HUB VERIFY FAILED: {exc}")
            return False
        bad = {arm: count for arm, count in counts.items() if int(count) <= 20}
        if result.returncode or bad:
            self.say(
                f"{record.profile}/{record.arms}: CHAIN complete but Hub counts={counts}; "
                "NOT tearing down"
            )
            return False
        self.say(f"{record.profile}/{record.arms}: Hub verified counts={counts}")
        return True

    def cleanup(self, record: PodRecord) -> None:
        allowed, reason = is_cleanup_target_owned(record, self.campaign)
        if not allowed:
            self.say(f"FATAL CLEANUP REFUSED for {record.pod_id}: {reason}")
            return
        api_state, _ = self.pod_api_state(record.pod_id)
        final_state = "done" if record.state == "finishing" else "cleaned"
        if api_state == "forbidden":
            self.say(f"FATAL CLEANUP REFUSED: API identifies {record.pod_id} as krill-mill")
            return
        if api_state == "gone":
            self.ledger.update(record.identity, state=final_state, strikes=0)
            self.say(f"{record.pod_id}: already gone -> {final_state}")
            return
        preview_rc, preview = self.run_logged(
            [str(CLEANUP_POD), record.pod_id], timeout=90,
            label=f"cleanup-preview:{record.pod_name}",
        )
        if preview_rc or record.pod_id not in preview or record.pod_name not in preview:
            self.say(f"FATAL: cleanup preview did not prove exact target {record.pod_name}")
            return
        delete_rc, _ = self.run_logged(
            [str(CLEANUP_POD), record.pod_id, "--yes"], timeout=120,
            label=f"cleanup-delete:{record.pod_name}",
        )
        if delete_rc == 0:
            self.ledger.update(record.identity, state=final_state, strikes=0)
            self.say(f"{record.pod_id}: cleanup complete -> {final_state}")
        else:
            self.say(f"{record.pod_id}: CLEANUP FAILED; remains reserved and will retry")

    def pending(self, records: list[PodRecord]) -> list[S.WorkUnit]:
        latest = latest_records(r for r in records if r.owner_token == self.campaign.owner_token)
        return [
            unit for unit in self.units
            if unit.key not in latest or latest[unit.key].state not in ACTIVE_STATES | {"done"}
        ]

    def launch_pending(self) -> None:
        records = self.ledger.read()
        latest = latest_records(r for r in records if r.owner_token == self.campaign.owner_token)
        active = [record for record in latest.values() if record.state in ACTIVE_STATES]
        pending = self.pending(records)
        account = self.account()
        if not pending or account is None:
            return
        _, actual_burn = account
        managed = sum((record.hourly_rate for record in active), Decimal("0"))
        # Unknown/non-campaign account spend can only reduce capacity.  Translate
        # actual account burn back to a conservative managed-equivalent number.
        effective_managed = max(managed, max(Decimal("0"), actual_burn - S.EXTERNAL_BURN))
        selected = S.select_launches([effective_managed], pending)
        for unit in selected:
            prior = latest.get(unit.key)
            attempt = 1 if prior is None else prior.attempt + 1
            if attempt > self.args.max_attempts:
                if unit.key not in self._churn_halted:
                    self._churn_halted.add(unit.key)
                    self.say(
                        f"*** HALTED {unit.profile}/{unit.arms_csv}: attempt {attempt} would "
                        f"exceed --max-attempts={self.args.max_attempts}. Refusing to create "
                        f"another pod. Repeated bring-up failure is a bug in the recipe, not "
                        f"bad luck -- read the setup log, fix it, then raise --max-attempts "
                        f"or restart the supervisor."
                    )
                continue
            safe_arms = "".join(arm[0] for arm in unit.arms)
            name = (
                f"dfv1-{self.campaign.campaign_id}-{self.campaign.owner_token[:8]}-"
                f"{unit.profile}-{safe_arms}-a{attempt}"
            )[:63]
            record = PodRecord(
                state="provisioning", profile=unit.profile, arms=unit.arms_csv,
                ssh_alias=f"runpod-{name}", pod_id="-", hourly_rate=unit.hourly_rate,
                campaign_id=self.campaign.campaign_id,
                owner_token=self.campaign.owner_token, attempt=attempt, pod_name=name,
                created_at=utcnow(), strikes=0,
            )
            # Intent is durable BEFORE the create call.  This is the ownership
            # proof used to reconcile a supervisor crash during provisioning.
            self.ledger.append(record)
            self.say(
                f"CREATE INTENT: {unit.profile}/{unit.arms_csv} attempt={attempt} "
                f"rate=${unit.hourly_rate}/hr name={name}"
            )
            self.submit_bringup(record, fresh=True)

    def process_running(self) -> dict[tuple[str, str, int, str], Probe]:
        records = [
            record for record in self.ledger.read()
            if record.owner_token == self.campaign.owner_token and record.state == "running"
        ]
        probes: dict[tuple[str, str, int, str], Probe] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(records))) as pool:
            jobs = {pool.submit(self.probe, record): record for record in records}
            for future, record in ((future, jobs[future]) for future in jobs):
                try:
                    probe = future.result()
                except Exception as exc:  # noqa: BLE001
                    probe = Probe("strike", "probe-error", str(exc))
                probes[record.identity] = probe
                if probe.outcome == "running":
                    if record.strikes:
                        self.ledger.update(record.identity, strikes=0)
                    self.say(f"{record.profile}/{record.arms}: {probe.phase} ({probe.reason})")
                elif probe.outcome == "complete":
                    if self.verify_hub(record):
                        self.ledger.update(record.identity, state="finishing", strikes=0)
                        self.say(f"{record.profile}/{record.arms}: DURABLE COMPLETE -> teardown")
                elif probe.outcome == "lost":
                    self.ledger.update(record.identity, state="lost", strikes=0)
                    self.say(
                        f"{record.profile}/{record.arms}: POD LOST -> fresh-pod recovery queued"
                    )
                elif probe.outcome == "blocked":
                    self.say(f"FATAL: forbidden pod surfaced in campaign ledger: {probe.reason}")
                elif probe.outcome == "failed":
                    self.ledger.update(record.identity, state=PARKED_STATE, strikes=0)
                    self.say(
                        f"*** PARKED {record.profile}/{record.arms}: CHAIN FAILED "
                        f"({probe.reason}). Pod {record.pod_id} is ALIVE, still billing "
                        f"${record.hourly_rate}/hr, and will NOT be deleted. Diagnose it, "
                        f"then relaunch on it or clean it up by hand."
                    )
                else:
                    strikes = record.strikes + 1
                    self.ledger.update(record.identity, strikes=strikes)
                    self.say(
                        f"{record.profile}/{record.arms}: {probe.reason or probe.phase} "
                        f"(strike {strikes}/{self.args.failure_strikes})"
                    )
                    if strikes >= self.args.failure_strikes:
                        self.ledger.update(record.identity, state=PARKED_STATE, strikes=0)
                        self.say(
                            f"*** PARKED {record.profile}/{record.arms}: "
                            f"{probe.reason or probe.phase} after {strikes} strikes. "
                            f"Pod {record.pod_id} is ALIVE, still billing "
                            f"${record.hourly_rate}/hr, and will NOT be deleted. Most "
                            f"strikes are 'ssh-unreachable', which is often the network "
                            f"and not the run -- check before destroying anything."
                        )
        return probes

    def render_status(self, *, include_probes: bool = True) -> None:
        records = self.ledger.read()
        latest = latest_records(r for r in records if r.owner_token == self.campaign.owner_token)
        active = sorted(
            (record for record in latest.values() if record.state in ACTIVE_STATES),
            key=lambda record: (record.profile, record.arms),
        )
        phases: dict[tuple[str, str, int, str], str] = {}
        if include_probes:
            running = [record for record in active if record.state == "running"]
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(running))) as pool:
                jobs = {pool.submit(self.probe, record): record for record in running}
                for future, record in ((future, jobs[future]) for future in jobs):
                    try:
                        phases[record.identity] = future.result().phase
                    except Exception as exc:  # noqa: BLE001
                        phases[record.identity] = f"probe-error:{exc}"
        print(f"===== DISPATCH QUEUE {utcnow()} =====")
        print("LIVE / RESERVED PODS")
        if active:
            print("  profile                    arms                     $/hr   state          phase / pod")
            for record in active:
                phase = phases.get(record.identity, record.state)
                print(
                    f"  {record.profile:<26} {record.arms:<24} "
                    f"{record.hourly_rate:>6.2f}  {record.state:<13} {phase} / {record.pod_id}"
                )
        else:
            print("  (none)")
        pending = self.pending(records)
        print("QUEUED UNITS")
        if pending:
            for unit in pending:
                print(
                    f"  p{unit.priority:<3} {unit.profile:<26} {unit.arms_csv:<24} "
                    f"${unit.hourly_rate:>5.2f}/hr"
                )
        else:
            print("  (none)")
        parked = [r for r in active if r.state == PARKED_STATE]
        if parked:
            print("*** PARKED -- ALIVE, BILLING, AWAITING A DECISION ***")
            for record in parked:
                print(
                    f"  {record.profile:<26} {record.arms:<24} "
                    f"${record.hourly_rate:>6.2f}/hr  pod {record.pod_id}  "
                    f"ssh {record.ssh_alias}"
                )
        total, headroom = S.burn_summary(record.hourly_rate for record in active)
        print(
            f"BURN  managed ${total - S.EXTERNAL_BURN:.2f}/hr + krill-mill "
            f"${S.EXTERNAL_BURN:.2f}/hr = ${total:.2f}/${S.ACCOUNT_CAP:.2f}/hr; "
            f"headroom ${headroom:.2f}/hr"
        )
        account = self.account()
        if account:
            balance, actual = account
            runway = balance / actual if actual else Decimal("Infinity")
            runway_text = "infinite" if not runway.is_finite() else f"{runway:.1f}h"
            warning = "  <-- LOW" if runway.is_finite() and runway < 6 else ""
            print(
                f"ACCOUNT  balance ${balance:.2f}; actual burn ${actual:.2f}/hr; "
                f"runway {runway_text}{warning}"
            )
        else:
            print("ACCOUNT  balance/burn/runway unavailable (launches are gated off)")

    def run(self) -> int:
        # Resume any journaled bring-up after a supervisor restart.
        for record in self.ledger.read():
            if record.owner_token == self.campaign.owner_token and record.state in BRINGUP_STATES:
                self.submit_bringup(record, fresh=False)
        while True:
            for record in self.ledger.read():
                if (record.owner_token == self.campaign.owner_token
                        and record.state in BRINGUP_STATES):
                    self.submit_bringup(record, fresh=False)
            self.process_running()
            for record in self.ledger.read():
                if record.owner_token == self.campaign.owner_token and record.state in {
                    "finishing", "recovering"
                }:
                    self.cleanup(record)
            # A cleaned/lost attempt is now pending and receives an entirely
            # fresh pod.  The old pod is never reused or locally reset.
            self.launch_pending()
            self.render_status(include_probes=False)
            latest = latest_records(
                r for r in self.ledger.read() if r.owner_token == self.campaign.owner_token
            )
            if all(latest.get(unit.key) and latest[unit.key].state == "done" for unit in self.units):
                self.say("ALL NINE WORK UNITS COMPLETE, HUB-VERIFIED, AND CLEANED")
                return 0
            time.sleep(self.args.poll_seconds)


def load_campaign(path: Path) -> Campaign | None:
    if not path.is_file():
        return None
    body = json.loads(path.read_text())
    return Campaign(**body)


def init_campaign(path: Path, campaign_id: str | None) -> Campaign:
    existing = load_campaign(path)
    if existing:
        if campaign_id and campaign_id != existing.campaign_id:
            raise SystemExit(
                f"campaign file is {existing.campaign_id!r}, not requested {campaign_id!r}"
            )
        return existing
    if not campaign_id:
        raise SystemExit("first --execute requires --campaign-id (lowercase letters/digits/hyphens)")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,15}", campaign_id):
        raise SystemExit("--campaign-id must match [a-z0-9][a-z0-9-]{1,15}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=OPS, text=True, capture_output=True, check=True
    ).stdout.strip()
    repo_url = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=OPS, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    campaign = Campaign(
        campaign_id=campaign_id,
        owner_token=uuid.uuid4().hex,
        source_commit=commit,
        repo_url=repo_url,
        created_at=utcnow(),
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(campaign), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return campaign


def dry_run(units: list[S.WorkUnit]) -> int:
    print("DRY RUN: no RunPod, SSH, Hub, or filesystem state mutations")
    rehydrate = EXP / "pod" / "rehydrate.py"
    if rehydrate.is_file():
        print(f"rehydrate contract: present ({rehydrate})")
    else:
        print(f"rehydrate contract: MISSING ({rehydrate}); execute mode will fail loudly")
    remaining = list(units)
    wave = 1
    while remaining:
        selected = S.select_launches([], remaining)
        if not selected:
            print("FATAL: queue contains a unit that cannot fit the cap")
            return 2
        managed = sum((unit.hourly_rate for unit in selected), Decimal("0"))
        print(
            f"simulated wave {wave}: managed ${managed:.2f}/hr + external "
            f"${S.EXTERNAL_BURN:.2f}/hr = ${managed + S.EXTERNAL_BURN:.2f}/hr"
        )
        for unit in selected:
            print(
                f"  {unit.profile:<26} arms={unit.arms_csv:<20} "
                f"{unit.shape.n_gpus}x {unit.shape.gpu_id}  ${unit.hourly_rate:.2f}/hr"
            )
        keys = {unit.key for unit in selected}
        remaining = [unit for unit in remaining if unit.key not in keys]
        wave += 1
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="offline validation/simulation")
    mode.add_argument("--status", action="store_true", help="read-only live status")
    mode.add_argument("--execute", action="store_true", help="run the campaign queue")
    ap.add_argument("--campaign-id")
    ap.add_argument("--confirm-hourly-cap", type=Decimal)
    ap.add_argument("--allow-delete-owned", action="store_true")
    ap.add_argument("--queue", type=Path, default=OPS / "queue.txt")
    ap.add_argument("--pods", type=Path, default=OPS / "pods.txt")
    ap.add_argument("--shapes", type=Path, default=OPS / "pod_shapes.tsv")
    ap.add_argument("--campaign-file", type=Path, default=OPS / "campaign.json")
    ap.add_argument("--runtime", type=Path, default=OPS / "runtime")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--failure-strikes", type=int, default=5,
                    help="probe failures before a RUNNING pod is parked (never deleted). "
                         "A strike is usually 'ssh-unreachable' -- 45s timeouts one poll "
                         "apart -- so a low value turns a network blip into a lost stage.")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="pods to create per unit before halting that unit entirely.")
    ap.add_argument("--no-output-timeout", type=int, default=1800)
    ap.add_argument("--create-timeout", type=int, default=720)
    ap.add_argument("--create-reconcile-seconds", type=int, default=600)
    ap.add_argument("--preflight-timeout", type=int, default=240)
    ap.add_argument("--setup-timeout", type=int, default=3600)
    return ap


def main() -> int:
    args = parser().parse_args()
    units = S.load_queue(args.queue, PROFILES, args.shapes)
    if args.dry_run:
        return dry_run(units)
    campaign = load_campaign(args.campaign_file)
    if args.status:
        if campaign is None:
            print("No campaign.json yet; run --dry-run or initialize with --execute.")
            return 0
        Supervisor(args, units, campaign).render_status()
        return 0
    if args.confirm_hourly_cap != S.ACCOUNT_CAP:
        raise SystemExit("--execute requires --confirm-hourly-cap 80.00")
    if not args.allow_delete_owned:
        raise SystemExit(
            "--execute requires --allow-delete-owned: authorizes cleanup-pod.sh --yes "
            "ONLY for token-matched pods this campaign creates"
        )
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("--execute requires HF_TOKEN for rehydrate/publish")
    rehydrate = EXP / "pod" / "rehydrate.py"
    if not rehydrate.is_file():
        raise SystemExit(
            f"FATAL: required recovery entry point is missing: {rehydrate}; "
            "refusing to spend on pods"
        )
    for required in (CREATE_POD, CLEANUP_POD, PREFLIGHT_POD):
        if not os.access(required, os.X_OK):
            raise SystemExit(f"required RunPod helper is missing/not executable: {required}")
    if args.poll_seconds < 10 or args.failure_strikes < 1:
        raise SystemExit("poll-seconds must be >=10 and failure-strikes >=1")
    lock_path = OPS / ".supervisor.lock"
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another supervisor holds .supervisor.lock") from exc
    campaign = init_campaign(args.campaign_file, args.campaign_id)
    rehydrate_at_commit = (
        f"{campaign.source_commit}:"
        "experiments/prior_coins/dispatch_final_v1/pod/rehydrate.py"
    )
    committed = subprocess.run(
        ["git", "cat-file", "-e", rehydrate_at_commit], cwd=OPS,
        text=True, capture_output=True, timeout=30, check=False,
    )
    if committed.returncode:
        raise SystemExit(
            f"FATAL: campaign source commit {campaign.source_commit} does not contain "
            "pod/rehydrate.py; refusing to spend on pods"
        )
    print(
        f"EXECUTE campaign={campaign.campaign_id} source={campaign.source_commit} "
        f"cap=${S.ACCOUNT_CAP}/hr external=${S.EXTERNAL_BURN}/hr"
    )
    return Supervisor(args, units, campaign).run()


if __name__ == "__main__":
    raise SystemExit(main())
