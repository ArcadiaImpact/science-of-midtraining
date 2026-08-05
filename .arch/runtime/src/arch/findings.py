"""`scripts/arch2 findings` — labeled-PR leaderboard + per-PR detail."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from arch._config import ConfigError, load_config

_PR_LIST_FIELDS = "number,title,body,state,headRefOid,createdAt,closedAt"
_PR_CHECKS_FIELDS = "name,state,description"
_PR_VIEW_FIELDS = (
    "number,title,body,state,headRefOid,author,createdAt,closedAt,comments,labels"
)
# The single machine-readable score transport: a commit status with this
# context, posted by templates/heldout_eval_startup.sh.j2. It surfaces here
# via `gh pr checks`. The human PR comment is never parsed.
_STATUS_CONTEXT = "arch-eval"


@click.group(
    invoke_without_command=True,
    help=(
        "Leaderboard of labeled PRs. Bare `scripts/arch2 findings` lists the top "
        "attempts; `scripts/arch2 findings show <pr>` dumps one PR's body, score, "
        "and closing rationale."
    ),
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON array of records.")
@click.option(
    "--state",
    type=click.Choice(["open", "closed", "all"]),
    default="open",
    show_default=True,
)
@click.option("--limit", type=int, default=100, show_default=True)
@click.pass_context
def findings_cmd(ctx: click.Context, as_json: bool, state: str, limit: int) -> None:
    if ctx.invoked_subcommand is not None:
        # A subcommand will run (e.g. `findings show <n>`); skip the default.
        return
    _list(as_json=as_json, state=state, limit=limit)


@findings_cmd.command("show", help="Dump one PR's body, score, and closing rationale.")
@click.argument("pr_number", type=int)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of rich text.")
def show(pr_number: int, as_json: bool) -> None:
    try:
        load_config()  # validate we're inside an arch task tree
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)

    try:
        pr = _gh_pr_view(pr_number)
    except _GhError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)

    score, metrics, notes = _gh_pr_score(pr_number)
    record = {
        "pr_number": pr["number"],
        "title": pr["title"],
        "state": pr["state"].lower(),
        "author": (pr.get("author") or {}).get("login"),
        "score": score,
        "metrics": metrics,
        "notes": notes,
        "body": pr.get("body", "") or "",
        "labels": [l.get("name") for l in pr.get("labels", []) or []],
        "comments": [
            {
                "author": (c.get("author") or {}).get("login"),
                "body": c.get("body", ""),
                "createdAt": c.get("createdAt"),
            }
            for c in pr.get("comments", []) or []
        ],
        "head_sha": pr.get("headRefOid"),
        "created_at": pr.get("createdAt"),
        "closed_at": pr.get("closedAt"),
    }

    if as_json:
        click.echo(json.dumps(record, indent=2))
    else:
        _print_show(record)


def _list(*, as_json: bool, state: str, limit: int) -> None:
    try:
        cfg = load_config()
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)

    try:
        prs = _gh_pr_list(cfg.label, state)
    except _GhError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)

    records: list[dict[str, Any]] = []
    for pr in prs:
        score, metrics, notes = _gh_pr_score(pr["number"])
        records.append(
            {
                "pr_number": pr["number"],
                "title": pr["title"],
                "body": pr.get("body", ""),
                "state": pr["state"].lower(),
                "score": score,
                "metrics": metrics,
                "notes": notes,
                "head_sha": pr["headRefOid"],
                "created_at": pr["createdAt"],
                "closed_at": pr.get("closedAt"),
            }
        )

    records = _sort_records(records)[:limit]

    if as_json:
        click.echo(json.dumps(records))
    else:
        _print_table(records)


class _GhError(Exception):
    pass


def _gh_pr_list(label: str, state: str) -> list[dict[str, Any]]:
    cmd = [
        "gh",
        "pr",
        "list",
        "--label",
        label,
        "--state",
        state,
        "--json",
        _PR_LIST_FIELDS,
        "--limit",
        "100",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise _GhError(f"`gh pr list` failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout) if proc.stdout.strip() else []


def _gh_pr_view(pr_number: int) -> dict[str, Any]:
    cmd = ["gh", "pr", "view", str(pr_number), "--json", _PR_VIEW_FIELDS]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise _GhError(f"`gh pr view {pr_number}` failed: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise _GhError(f"`gh pr view {pr_number}` returned non-JSON: {exc}") from exc


def _gh_pr_score(pr_number: int) -> tuple[float | None, dict | None, str | None]:
    """Return (score, metrics, notes) from the single `arch-eval` transport.

    The held-out eval posts the score as a commit status (context
    `arch-eval`) — the one machine-readable transport. It is PAT-safe and
    surfaces here via `gh pr checks`. The human PR comment is never parsed:
    one source means the leaderboard and any figure always agree. A PR with
    no `arch-eval` status simply has no score (eval failed or never ran).
    """
    cmd = ["gh", "pr", "checks", str(pr_number), "--json", _PR_CHECKS_FIELDS]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None, None, None
    try:
        checks = json.loads(proc.stdout) if proc.stdout.strip() else []
    except json.JSONDecodeError:
        return None, None, None
    for check in checks:
        if check.get("name") != _STATUS_CONTEXT:
            continue
        try:
            payload = json.loads(check.get("description", "") or "")
        except json.JSONDecodeError:
            return None, None, None
        if isinstance(payload, dict):
            return payload.get("score"), payload.get("metrics"), payload.get("notes")
        return None, None, None
    return None, None, None


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by (score desc, created_at desc). Null scores sort last."""

    def key(r: dict[str, Any]) -> tuple[int, float, str]:
        score = r.get("score")
        score_is_null = 1 if score is None else 0
        score_val = -score if score is not None else 0.0
        created = r.get("created_at") or ""
        return (score_is_null, score_val, _negate_ts(created))

    return sorted(records, key=key)


def _negate_ts(ts: str) -> str:
    """Invert ISO timestamp so descending sort works as a string key."""
    if not ts:
        return ""
    return "".join(chr(255 - ord(c)) if c.isdigit() else c for c in ts)


def _print_table(records: list[dict[str, Any]]) -> None:
    console = Console()
    table = Table(title="ARCH findings")
    table.add_column("rank", justify="right")
    table.add_column("PR")
    table.add_column("score", justify="right")
    table.add_column("state")
    table.add_column("title")
    for i, r in enumerate(records, 1):
        score = "—" if r["score"] is None else f"{r['score']:.4f}"
        title = (r["title"] or "")[:60]
        table.add_row(str(i), f"#{r['pr_number']}", score, r["state"], title)
    console.print(table)


def _print_show(record: dict[str, Any]) -> None:
    console = Console()
    score = "—" if record["score"] is None else f"{record['score']:.4f}"
    header = (
        f"# PR #{record['pr_number']} — {record['title']}\n\n"
        f"**state:** {record['state']}   **score:** {score}   "
        f"**author:** {record.get('author') or '—'}   "
        f"**created:** {record.get('created_at') or '—'}"
    )
    if record["state"] == "closed":
        header += f"   **closed:** {record.get('closed_at') or '—'}"
    console.print(Markdown(header))
    console.print(
        Markdown("## Hypothesis / body\n\n" + (record["body"] or "_(empty)_"))
    )

    metrics = record.get("metrics")
    if metrics:
        console.print(
            Markdown(
                "## Published metrics\n\n```json\n"
                + json.dumps(metrics, indent=2)
                + "\n```"
            )
        )

    notes = record.get("notes")
    if notes:
        console.print(Markdown(f"## Notes\n\n{notes}"))

    comments = record.get("comments") or []
    if comments:
        console.print(Markdown("## Comments"))
        for c in comments:
            author = c.get("author") or "—"
            ts = c.get("createdAt") or ""
            body = c.get("body") or ""
            console.print(Markdown(f"---\n**{author}** _{ts}_\n\n{body}"))
