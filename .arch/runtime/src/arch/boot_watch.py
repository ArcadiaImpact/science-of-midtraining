"""`scripts/arch2 boot-watch` — the mandatory active health watch after a spawn.

The recurring, expensive friction this kills: a fleet is spawned, the framework
*trusts* the pods are fine, and hours later someone notices every pod hit a dumb
error seconds after boot and burned GPU the whole time. The boot-watch loop is
the fix — poll every pod until it's confirmed iterating, and catch failures in
the first minutes instead of at the deadline.

Making it a single deterministic command with a pass/fail exit code is what
makes it non-optional: `arch-run` gates the rest of the run on a zero exit, so
the watch can't be skipped or hand-waved the way a prose loop was.

Health is read over SSH (see `arch._ssh`) — RunPod's REST API has no `/logs`
endpoint, so this tails the pod's own log file directly. `--kind worker`
(default) watches an iterating fleet pod; `--kind heldout` watches the
one-shot held-out eval pod (arch-eval.yml / the arch-init canary), which has a
different health/terminal-state model — see `monitor._classify_heldout`.

Exit codes:
  0  every pod reached a healthy state within the window
  3  a pod failed terminally (kind-dependent — see monitor._classify /
     _classify_heldout), or was UNREACHABLE past --grace, or PHANTOM past
     --phantom-grace (desiredStatus RUNNING but never actually scheduled: no
     machine/publicIp/portMappings, arch2#42) — terminate, fix, respawn,
     re-watch
  4  the window elapsed without confirming every pod healthy — do NOT go
     unattended: verify the pod's SSH port is actually reachable
  2  usage / config error
"""
from __future__ import annotations

import json
import sys
import time

import click

from arch._config import ConfigError, load_config
from arch._dotenv import DotenvError, runpod_api_key
from arch.monitor import (
    _inspect_pod,
    _PodRecord,
    _print_table,
    find_pod_by_name,
    pod_ids_for_kind,
)

# Health buckets (from monitor._classify / _classify_heldout) grouped by what
# the watch should do, per pod kind. TERMINATED means opposite things for the
# two kinds: a worker that's gone died early (fatal); the held-out eval pod's
# startup script only ever self-terminates on its success path, so gone means
# it succeeded (healthy).
_HEALTHY_BY_KIND = {
    "worker": {"RUNNING"},
    "heldout": {"RUNNING", "TERMINATED"},
    "batch": {"RUNNING", "TERMINATED"},
}
_FATAL_BY_KIND = {
    "worker": {"BOOTLOOP", "SSH_DEAD", "CODEX_EXITED", "CODEX_FAILED", "TERMINATED", "SSH_AUTH_FAILED"},
    "heldout": {"EVAL_SETUP_FAILED", "EVAL_FAILED", "SSH_AUTH_FAILED"},
    "batch": {"EVAL_SETUP_FAILED", "EVAL_FAILED", "SSH_AUTH_FAILED"},
}
# UNREACHABLE = can't read pod metadata, or SSH isn't answering yet. At boot
# this can lag, so it's only fatal once the grace period has elapsed.
# Everything else not yet healthy/fatal (BOOTING, etc.) is "still pending" —
# keep waiting until the window closes.
# PHANTOM = pod metadata reads fine and claims RUNNING, but shows no machine/
# publicIp/portMappings at all — RunPod REST `create` has left pods exactly
# like this for hours, silently never scheduled, with no error anywhere
# (arch2#42). A pod can look transiently unscheduled in the first minute of a
# normal boot too, so — like UNREACHABLE — it only fails once its own (longer)
# grace period elapses; it is not lumped into either kind's fatal set. It is
# kind-independent: it comes from pod metadata, not from the log classifier, so
# a never-scheduled heldout pod fails the same way a never-scheduled worker does.
DEFAULT_PHANTOM_GRACE_S = 600  # ~10 min, per arch2#42's suggested fix


def evaluate(
    records: list[_PodRecord],
    elapsed_s: float,
    grace_s: float,
    kind: str = "worker",
    phantom_grace_s: float = DEFAULT_PHANTOM_GRACE_S,
) -> tuple[str, str]:
    """Pure verdict for one poll. Returns (state, reason).

    state is one of: "pass" (all healthy), "fail" (terminal breakage), or
    "wait" (not yet conclusive). Kept side-effect-free so it's unit-testable
    without hitting RunPod. `kind` selects which health buckets count as
    healthy/fatal — see `_HEALTHY_BY_KIND`/`_FATAL_BY_KIND` above.
    """
    healthy = _HEALTHY_BY_KIND[kind]
    fatal_set = _FATAL_BY_KIND[kind]

    fatal = [r for r in records if r.health in fatal_set]
    if fatal:
        return "fail", "; ".join(f"{r.pod_id}: {r.health} — {r.reason}" for r in fatal)

    phantom = [r for r in records if r.health == "PHANTOM"]
    if phantom and elapsed_s >= phantom_grace_s:
        return "fail", "; ".join(
            f"{r.pod_id}: PHANTOM past {int(phantom_grace_s)}s grace — {r.reason}"
            for r in phantom
        )

    unreachable = [r for r in records if r.health == "UNREACHABLE"]
    if unreachable and elapsed_s >= grace_s:
        return "fail", "; ".join(
            f"{r.pod_id}: UNREACHABLE past {int(grace_s)}s grace — {r.reason}"
            for r in unreachable
        )

    if records and all(r.health in healthy for r in records):
        return "pass", f"all {len(records)} pod(s) healthy"

    pending = [f"{r.pod_id}: {r.health}" for r in records if r.health not in healthy]
    return "wait", ", ".join(pending)


@click.command(
    "boot-watch",
    help=(
        "Actively watch freshly-spawned pods until every one is confirmed "
        "healthy, or fail fast on a terminal breakage. By default reads "
        "worker pod IDs from .arch/.session.json; --pod-id/--pod-name watch "
        "explicit pods instead (e.g. the held-out eval pod, whose ID the "
        "orchestrator never recorded). Polls every --interval seconds for up "
        "to --timeout seconds. Exit 0 = healthy, 3 = a pod hit a terminal "
        "health bucket (worker: BOOTLOOP / SSH_DEAD / SSH_AUTH_FAILED / "
        "CODEX_EXITED / CODEX_FAILED / TERMINATED; heldout: EVAL_SETUP_FAILED / EVAL_FAILED "
        "/ SSH_AUTH_FAILED), or UNREACHABLE past --grace, or PHANTOM past "
        "--phantom-grace, 4 = window elapsed unconfirmed."
    ),
)
@click.option("--interval", type=int, default=60, show_default=True,
              help="Seconds between polls.")
@click.option("--timeout", type=int, default=900, show_default=True,
              help="Max seconds to watch before giving up (default 15 min).")
@click.option("--grace", type=int, default=180, show_default=True,
              help="Seconds before an UNREACHABLE pod is treated as failed "
                   "(metadata/SSH can lag at boot).")
@click.option("--phantom-grace", type=int, default=DEFAULT_PHANTOM_GRACE_S, show_default=True,
              help="Seconds before a PHANTOM pod (desiredStatus RUNNING but "
                   "no machine/publicIp/portMappings ever assigned — never "
                   "actually scheduled, arch2#42) is treated as failed.")
@click.option("--kind", type=click.Choice(["worker", "heldout", "batch"]), default="worker",
              show_default=True,
              help="worker: an iterating fleet pod. heldout: the one-shot "
                   "held-out eval pod — different health/fatal semantics "
                   "(e.g. self-terminating means success, not death).")
@click.option("--pod-id", "pod_id_opts", multiple=True,
              help="Watch this pod ID instead of .session.json's pod_ids. Repeatable.")
@click.option("--pod-name", "pod_name_opts", multiple=True,
              help="Resolve this exact RunPod pod name to an ID and watch it "
                   "(e.g. arch-<task>-s<session>-heldout-pr<n>-<sha>-gha-<run>-<attempt>). "
                   "Repeatable. Not found yet "
                   "counts as still booting, not an error.")
@click.option("--logs", is_flag=True, help="Print the last log lines per pod each poll.")
@click.option("--json", "as_json", is_flag=True, help="Emit a JSON verdict instead of a table.")
def boot_watch_cmd(
    interval: int, timeout: int, grace: int, phantom_grace: int, kind: str,
    pod_id_opts: tuple[str, ...], pod_name_opts: tuple[str, ...],
    logs: bool, as_json: bool,
) -> None:
    try:
        cfg = load_config()
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)

    try:
        api_key = runpod_api_key(cfg.repo_root)
    except DotenvError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)
    if not api_key:
        click.echo("error: RUNPOD_API_KEY is absent from private .env and environment", err=True)
        sys.exit(2)

    if pod_id_opts or pod_name_opts:
        pod_ids: list[str] = list(pod_id_opts)
    else:
        session_path = cfg.repo_root / ".arch" / ".session.json"
        if not session_path.is_file():
            click.echo(
                f"error: no .arch/.session.json at {session_path}. Spawn the "
                "fleet first (arch-run records pod IDs there), or pass "
                "--pod-id/--pod-name explicitly.",
                err=True,
            )
            sys.exit(2)
        try:
            session = json.loads(session_path.read_text())
        except json.JSONDecodeError as exc:
            click.echo(f"error: .arch/.session.json is malformed: {exc}", err=True)
            sys.exit(2)
        try:
            pod_ids = pod_ids_for_kind(session, kind)
        except click.ClickException as exc:
            click.echo(f"error: {exc.format_message()}", err=True)
            sys.exit(2)
        if not pod_ids:
            click.echo(
                f"error: no {kind}_pod_ids in .arch/.session.json — nothing to watch",
                err=True,
            )
            sys.exit(2)

    resolved_by_name: dict[str, str] = {}
    start = time.monotonic()
    while True:
        for name in pod_name_opts:
            if name not in resolved_by_name:
                pid = find_pod_by_name(name, api_key)
                if pid:
                    resolved_by_name[name] = pid

        still_unresolved = [n for n in pod_name_opts if n not in resolved_by_name]
        watch_ids = pod_ids + list(resolved_by_name.values())
        records = [_inspect_pod(pid, api_key, kind=kind) for pid in watch_ids]
        elapsed = time.monotonic() - start

        if not records:
            state, reason = "wait", f"pod(s) not yet spawned: {', '.join(still_unresolved)}"
        else:
            state, reason = evaluate(
                records, elapsed, grace, kind=kind, phantom_grace_s=phantom_grace,
            )
            if state == "pass" and still_unresolved:
                state = "wait"
                reason = f"{reason}; still waiting on: {', '.join(still_unresolved)}"

        if as_json:
            click.echo(json.dumps(
                {"state": state, "reason": reason, "elapsed_s": int(elapsed),
                 "pods": [r.to_dict() for r in records]},
                indent=2,
            ))
        else:
            _print_table(records, show_logs=logs)
            click.echo(f"[boot-watch] +{int(elapsed)}s — {state.upper()}: {reason}")

        if state == "pass":
            click.echo(f"[boot-watch] PASS — {reason}. Fleet confirmed alive.")
            sys.exit(0)
        if state == "fail":
            click.echo(
                f"[boot-watch] FAIL — {reason}\n"
                "Terminate the broken pod(s), fix the cause (often the startup "
                "template), respawn, and re-run `scripts/arch2 boot-watch`.",
                err=True,
            )
            sys.exit(3)
        if elapsed >= timeout:
            click.echo(
                f"[boot-watch] UNCONFIRMED after {int(elapsed)}s — {reason}\n"
                "Pods are not confirmed healthy within the window. The check "
                "itself is SSH-based now, so this should be rare — verify "
                "network/firewall access to the pod's mapped SSH port before "
                "assuming it's a config bug.",
                err=True,
            )
            sys.exit(4)
        time.sleep(interval)
