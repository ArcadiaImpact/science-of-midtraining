"""Disabled batch rescore command.

The original runtime placed untrusted PR heads, held-out data, evaluator
credentials, and an account-level RunPod deletion key in one root container.
That is not an acceptable security boundary.  Keep the command name so old
operator scripts fail loudly and predictably, but do not construct or submit
any pod payload until a brokered, non-root evaluator design is implemented.
"""

from __future__ import annotations

import click

_DISABLED_MESSAGE = (
    "batch rescore is disabled in the Codex ARCH v0.1 safety profile: the "
    "legacy design exposed held-out data and operational credentials to "
    "untrusted PR code. Use the per-PR isolated evaluator path; do not retry "
    "this command until a brokered evaluator boundary is available."
)


@click.command(
    "rescore",
    help=(
        "Disabled safety stub. Batch scoring is unavailable until held-out "
        "data and operational credentials are isolated from submitted code."
    ),
)
@click.option("--pr", "prs", multiple=True, type=int, hidden=True)
@click.option("--canary-pr", type=int, default=None, hidden=True)
@click.option("--dry-run", is_flag=True, hidden=True)
def rescore_cmd(prs: tuple[int, ...], canary_pr: int | None, dry_run: bool) -> None:
    """Refuse before reading config, credentials, GitHub state, or pod state."""
    del prs, canary_pr, dry_run
    raise click.ClickException(_DISABLED_MESSAGE)
