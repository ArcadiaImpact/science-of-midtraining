"""`scripts/arch2 monitor` — live worker fleet health.

Reads `.arch/.session.json` for the pod IDs registered by `arch init`, then
for each pod queries the RunPod REST API for its status/uptime and tails its
log file **over SSH** (RunPod's REST API has no logs endpoint — see
`arch._ssh`), and classifies its health into a small set of buckets so the
researcher can spot bootloops, dead workers, or stalled iteration without
hand-rolling watchers per pod (a recurring footgun during the live test).

PHANTOM is a distinct bucket from UNREACHABLE: a pod can read back fine as
`desiredStatus: RUNNING` with no error at all, yet never actually have been
scheduled onto hardware (no machine, no publicIp, no portMappings) — a RunPod
REST `create` failure mode that leaves a pod billing indefinitely with nothing
to observe (arch2#42). `scripts/arch2 boot-watch` turns this into a fail-fast
verdict; see `_looks_unscheduled` below.

This command is deliberately read-only. Pod deletion belongs to the exact-ID
cleanup workflow; the old prefix/deadline based ``--reap`` path was removed
because a health inspection command must never expand deletion scope.

Heuristics — kept deliberately small. Add buckets when a new failure mode
shows up in practice, not pre-emptively.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from arch._config import ConfigError, load_config
from arch._dotenv import DotenvError, runpod_api_key
from arch._ssh import ensure_arch_ssh_key, ssh_tail, ssh_target

RUNPOD_API_BASE = "https://rest.runpod.io/v1"
DEFAULT_LOG_TAIL = 200  # lines pulled per pod for classification


@click.command(
    help=(
        "Live worker fleet health. Reads .arch/.session.json, queries RunPod "
        "for each pod, classifies health, prints a table. Bucketing is "
        "deliberately small: RUNNING, BOOTING, BOOTLOOP, CODEX_EXITED, "
        "STALLED, SSH_DEAD, SSH_AUTH_FAILED, TERMINATED, UNREACHABLE, PHANTOM."
    ),
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a table.")
@click.option("--watch", type=int, default=0, help="Refresh every N seconds (0 = once).")
@click.option("--logs", is_flag=True, help="Print the last 20 log lines per pod.")
@click.option(
    "--kind",
    type=click.Choice(["worker", "heldout", "batch"]),
    default="worker",
    show_default=True,
    help="Inspect only the requested typed pod set from the session.",
)
def monitor_cmd(as_json: bool, watch: int, logs: bool, kind: str) -> None:
    try:
        cfg = load_config()
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)

    session_path = cfg.repo_root / ".arch" / ".session.json"
    if not session_path.is_file():
        click.echo(
            f"error: no .arch/.session.json at {session_path}. Either init "
            "didn't store one (re-run arch init), or this run completed "
            "successfully and the file was already cleaned up.",
            err=True,
        )
        sys.exit(2)

    try:
        session = json.loads(session_path.read_text())
    except json.JSONDecodeError as exc:
        click.echo(f"error: .arch/.session.json is malformed: {exc}", err=True)
        sys.exit(2)

    pod_ids = pod_ids_for_kind(session, kind)
    if not pod_ids:
        click.echo(f"no {kind}_pod_ids in .arch/.session.json — nothing to monitor")
        sys.exit(0)

    try:
        api_key = runpod_api_key(cfg.repo_root)
    except DotenvError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)
    if not api_key:
        click.echo("error: RUNPOD_API_KEY is absent from private .env and environment", err=True)
        sys.exit(2)

    while True:
        records = [_inspect_pod(pid, api_key, kind=kind) for pid in pod_ids]
        if as_json:
            click.echo(json.dumps([r.to_dict() for r in records], indent=2))
        else:
            _print_table(records, show_logs=logs)
        if watch <= 0:
            break
        time.sleep(watch)


_POD_FIELD_BY_KIND = {
    "worker": "worker_pod_ids",
    "heldout": "heldout_pod_ids",
    "batch": "batch_pod_ids",
}


def pod_ids_for_kind(session: dict[str, Any], kind: str) -> list[str]:
    """Read one typed pod set; only legacy workers inherit ``pod_ids``."""
    field = _POD_FIELD_BY_KIND[kind]
    if field in session:
        value = session.get(field) or []
    elif kind == "worker":
        value = session.get("pod_ids") or []
    else:
        value = []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise click.ClickException(f"session field {field} must be a list of pod ID strings")
    return list(dict.fromkeys(value))


def _api_get_list(path: str, api_key: str) -> list[dict[str, Any]]:
    raw = _api_get_text(path, api_key)
    try:
        # RunPod's pod list embeds literal newlines inside PUBLIC_KEY strings.
        return json.loads(raw, strict=False) if raw else []
    except json.JSONDecodeError as exc:
        raise _RunPodError(f"non-JSON response from {path}: {exc}") from exc


@dataclass
class _PodRecord:
    pod_id: str
    status: str           # raw RunPod status (RUNNING / EXITED / ...)
    health: str           # bucketed health
    reason: str           # one-line justification for the bucket
    uptime_s: int | None
    log_tail: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "status": self.status,
            "health": self.health,
            "reason": self.reason,
            "uptime_s": self.uptime_s,
            "log_tail": self.log_tail,
        }


def _looks_unscheduled(meta: dict[str, Any]) -> bool:
    """True when a pod shows no evidence it was ever handed real hardware: no
    `machine`, no `publicIp`, no `portMappings`. RunPod REST `create` has been
    observed to leave pods exactly like this — `desiredStatus: RUNNING`
    forever, no error anywhere — while runpodctl/GraphQL creations schedule
    immediately or fail loudly instead (arch2#42). A pod that is merely early
    in a normal boot usually already has *some* of these (a machine before SSH
    is up); require all three missing so a real early boot isn't misflagged.
    """
    return not meta.get("machine") and not meta.get("publicIp") and not meta.get("portMappings")


def _inspect_pod(pod_id: str, api_key: str, kind: str = "worker") -> _PodRecord:
    try:
        # includeMachine=true so `_looks_unscheduled` can actually see whether
        # a machine was assigned — it's omitted from the response otherwise.
        meta = _api_get(f"/pods/{pod_id}?includeMachine=true", api_key)
    except _RunPodError as exc:
        if exc.status == 404:
            # The pod doesn't exist — the same observable as a desiredStatus of
            # EXITED/TERMINATED below, so it gets the same bucket, which already
            # carries the correct per-kind polarity (a gone heldout pod
            # self-terminated on its success path; a gone worker died early).
            # Bucketing it UNREACHABLE instead failed every successful heldout
            # canary once the grace window elapsed.
            # Accepted tradeoff: a --pod-id typo for a pod that never existed
            # also 404s and now reads TERMINATED (healthy for heldout). The real
            # heldout workflow uses --pod-name, which only ever yields IDs of
            # pods observed alive via find_pod_by_name, so it isn't affected.
            return _PodRecord(
                pod_id, "TERMINATED", "TERMINATED",
                f"pod not found — it exited, was deleted, or never existed ({exc})",
                None, [],
            )
        return _PodRecord(pod_id, "UNREACHABLE", "UNREACHABLE", str(exc), None, [])

    status = (meta.get("desiredStatus") or meta.get("status") or "UNKNOWN").upper()
    uptime = meta.get("uptimeSeconds")
    if isinstance(uptime, float):
        uptime = int(uptime)

    if status in {"EXITED", "TERMINATED"}:
        return _PodRecord(pod_id, status, "TERMINATED",
                          "Pod exited or was deleted.", uptime, [])

    target = ssh_target(meta)
    if target is None:
        # No SSH target means no publicIp and/or no port-22 mapping, so it is a
        # strict superset of `_looks_unscheduled` — this is the only branch
        # where the phantom check can ever fire. (Once SSH *is* reachable the
        # pod demonstrably has publicIp + portMappings, so `_looks_unscheduled`
        # is False by construction and needs no re-check below.)
        if status == "RUNNING" and _looks_unscheduled(meta):
            # arch2#42: no machine, no IP, no ports, yet desiredStatus RUNNING —
            # the pod was never scheduled. Distinguished from a normal early
            # boot (which has *some* hardware evidence) so `scripts/arch2 boot-watch`
            # can fail it fast on its own grace period instead of letting it
            # bill for the whole window as if it were merely still booting.
            return _PodRecord(
                pod_id, status, "PHANTOM",
                "desiredStatus RUNNING but no machine/publicIp/portMappings "
                "assigned — pod likely never scheduled",
                uptime, [],
            )
        return _PodRecord(pod_id, status, "BOOTING",
                          "pod networking (public IP / SSH port mapping) not assigned yet",
                          uptime, [])

    host, port = target
    try:
        key_path = ensure_arch_ssh_key()
    except (OSError, RuntimeError) as exc:
        # RuntimeError if ssh-keygen failed, OSError/FileNotFoundError if the
        # binary isn't on PATH. Either way it's a local-environment problem, not
        # a pod-health signal — but letting it propagate would crash the poll
        # loop with an undocumented exit code, so it lands in the same
        # grace-gated UNREACHABLE bucket as any other "can't tell" case.
        return _PodRecord(pod_id, status, "UNREACHABLE",
                          f"could not prepare local SSH key: {exc}", uptime, [])
    remote_path = {
        "worker": "/workspace/arch-worker.log",
        "heldout": "/workspace/heldout-eval.log",
        "batch": "/workspace/heldout-batch-eval.log",
    }[kind]
    result = ssh_tail(host, port, key_path, remote_path=remote_path, lines=DEFAULT_LOG_TAIL)

    if result.outcome == "pending":
        return _PodRecord(pod_id, status, "UNREACHABLE",
                          f"SSH not reachable yet ({host}:{port}): {result.detail}",
                          uptime, [])
    if result.outcome == "remote_error":
        # SSH itself worked; the remote `tail` failed (usually the log file
        # doesn't exist yet). Same wait/fail treatment as "pending", but the
        # reason must not send the operator off to debug their firewall.
        return _PodRecord(pod_id, status, "UNREACHABLE",
                          f"SSH connected to {host}:{port} but could not read "
                          f"{remote_path}: {result.detail}",
                          uptime, [])
    if result.outcome == "auth_failed":
        return _PodRecord(
            pod_id, status, "SSH_AUTH_FAILED",
            f"SSH key rejected by {host}:{port} — PUBLIC_KEY likely did not "
            f"propagate at spawn: {result.detail}",
            uptime, [],
        )

    log_lines = result.stdout.splitlines()[-DEFAULT_LOG_TAIL:]
    classify = _classify if kind == "worker" else _classify_heldout
    health, reason = classify(log_lines, uptime)
    return _PodRecord(pod_id, status, health, reason, uptime, log_lines[-20:])


def find_pod_by_name(name: str, api_key: str) -> str | None:
    """Resolve an exact RunPod pod name to its pod ID.

    Used to locate a pod the orchestrator didn't spawn directly and therefore
    never recorded an ID for — e.g. the held-out eval pod, which GHA spawns a
    few seconds after a labeled PR event. Returns None if no pod with that
    exact name is currently listed — not necessarily an error, it may simply
    not have spawned yet.
    """
    try:
        all_pods = _api_get_list("/pods", api_key)
    except _RunPodError:
        return None
    for pod in all_pods:
        if (pod.get("name") or "") == name:
            return pod.get("id")
    return None


def _classify(log_lines: list[str], uptime_s: int | None) -> tuple[str, str]:
    """Return (health_bucket, one-line reason)."""
    text = "\n".join(log_lines)
    # Count boot markers; >1 in recent log → restarting → bootloop.
    boot_markers = re.findall(r"=== arch-worker boot ", text)
    if len(boot_markers) >= 2:
        return "BOOTLOOP", f"saw {len(boot_markers)} boot markers in the last {DEFAULT_LOG_TAIL} log lines"

    # SSH ever started in this log window?
    if "destination path 'work' already exists" in text:
        return "BOOTLOOP", "git clone hit 'destination path already exists' — startup script not idempotent"
    if "sshd: no hostkeys" in text:
        return "SSH_DEAD", "sshd refused to start (no host keys generated)"

    # Codex loop exhausted or failed before it could restart.
    if "codex-loop-exhausted" in text:
        return "CODEX_EXITED", "worker loop fell through to self_terminate — deadline reached or Codex crashed permanently"

    # JSONL status must be scoped to the newest iteration. A successful event
    # from iteration N must not make a newly-started, silent iteration N+1 look
    # healthy; likewise a prior turn.failed must not poison a later retry.
    iter_matches = list(re.finditer(r"=== codex exec iteration (\d+),", text))
    if iter_matches:
        latest = iter_matches[-1]
        last_iter = int(latest.group(1))
        iteration_text = text[latest.start():]
        if (
            '"type":"turn.failed"' in iteration_text
            or '"type": "turn.failed"' in iteration_text
        ):
            return "CODEX_FAILED", f"Codex iteration {last_iter} emitted a turn.failed event"
        has_events = (
            '"type":"thread.started"' in iteration_text
            or '"type": "thread.started"' in iteration_text
            or '"type":"item.' in iteration_text
            or '"type": "item.' in iteration_text
        )
        if not has_events:
            if (uptime_s or 0) < 600:
                return "BOOTING", "Codex process started; waiting for JSONL events"
            return "STALLED", "Codex process emitted no JSONL events after 10+ min"
        # If still on iteration 1 after 30 min, it may be stuck.
        if last_iter == 1 and (uptime_s or 0) > 1800:
            return "STALLED", "still on iteration 1 after 30+ min — Codex may be stuck on a long action"
        return "RUNNING", f"on iteration {last_iter}"

    if boot_markers and (uptime_s or 0) < 300:
        return "BOOTING", "boot in progress (<5 min)"
    if not boot_markers:
        return "UNREACHABLE", "no boot marker in log — startup script may not have run"
    return "STALLED", "no iteration marker in recent log"


_HELDOUT_SETUP_FAILURE_MARKERS = (
    "CRITICAL: refusing to score or self-delete",
)

# Printed by the hardened heldout template only after clone, exact-SHA checks,
# trusted-path restoration, the inert-artifact copy, read-only data setup, and
# IPv4/IPv6 isolation have all succeeded. The boot banner alone does not prove
# any of those gates.
_HELDOUT_POST_SETUP_MARKERS = (
    "=== running trusted evaluator as uid=",
)


def _classify_heldout(log_lines: list[str], uptime_s: int | None) -> tuple[str, str]:
    """Return (health_bucket, one-line reason) for the held-out eval pod.

    Different lifecycle than a worker: it runs once and either exact-ID deletes
    itself after publication plus durable staging, or sleeps forever on a
    fail-closed setup/staging path so the host can recover it. Match the trusted
    supervisor's explicit markers, never a generic ``ERROR:`` emitted by the
    scorer.

    RUNNING requires one of `_HELDOUT_POST_SETUP_MARKERS`, never the boot
    banner alone: the banner prints before apt/gh install, before the clone,
    and before deps install, so accepting it would PASS the watch on a pod that
    hasn't cloned the repo yet.
    """
    text = "\n".join(log_lines)
    for marker in _HELDOUT_SETUP_FAILURE_MARKERS:
        if marker in text:
            return "EVAL_SETUP_FAILED", f"saw '{marker}' in the pod log"
    failed_exit = re.search(r"=== trusted evaluator exited ([1-9][0-9]*) ===", text)
    if failed_exit:
        return "EVAL_FAILED", f"trusted evaluator exited {failed_exit.group(1)}"
    if any(marker in text for marker in _HELDOUT_POST_SETUP_MARKERS):
        return "RUNNING", "clone + setup succeeded, eval invocation in flight"
    if "=== arch heldout-eval boot" in text:
        return "BOOTING", ("boot marker seen, but setup hasn't finished yet "
                           "(still installing tooling / cloning / installing deps)")
    return "BOOTING", "boot marker not seen yet"


def _print_table(records: list[_PodRecord], show_logs: bool) -> None:
    console = Console()
    table = Table(title="ARCH monitor")
    table.add_column("pod")
    table.add_column("status")
    table.add_column("health")
    table.add_column("uptime")
    table.add_column("reason")
    for r in records:
        uptime = "—" if r.uptime_s is None else f"{r.uptime_s // 60}m"
        health = r.health
        # Light coloring via Rich markup.
        if health in {"BOOTLOOP", "SSH_DEAD", "SSH_AUTH_FAILED", "UNREACHABLE",
                      "CODEX_EXITED", "CODEX_FAILED", "EVAL_FAILED", "EVAL_SETUP_FAILED", "PHANTOM"}:
            health = f"[red]{health}[/red]"
        elif health == "STALLED":
            health = f"[yellow]{health}[/yellow]"
        elif health == "RUNNING":
            health = f"[green]{health}[/green]"
        # `reason` can embed raw ssh/tail stderr, so it goes in as a Text —
        # a stray "[...]"-shaped substring would otherwise be parsed as Rich
        # markup and can raise MarkupError mid-poll.
        table.add_row(r.pod_id, r.status, health, uptime, Text(r.reason))
    console.print(table)
    if show_logs:
        for r in records:
            console.rule(f"{r.pod_id} — last 20 lines")
            for line in r.log_tail:
                console.print(line, highlight=False, markup=False)


class _RunPodError(Exception):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status  # HTTP status code, if the error came from one


def _api_get(path: str, api_key: str) -> dict[str, Any]:
    raw = _api_get_text(path, api_key)
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise _RunPodError(f"non-JSON response from {path}: {exc}") from exc


def _api_get_text(path: str, api_key: str) -> str:
    req = urllib.request.Request(
        RUNPOD_API_BASE + path,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:200]
        raise _RunPodError(f"HTTP {exc.code}: {body}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise _RunPodError(f"network error: {exc}") from exc
