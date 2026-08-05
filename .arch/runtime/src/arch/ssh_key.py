"""`arch ssh-key` — print the dedicated arch2 SSH keypair's path or public contents.

Exists so skill instructions and scripts don't need to know the default
location (~/.ssh/arch2_worker_ed25519) or the ARCH_SSH_KEY_PATH override —
`ensure_arch_ssh_key()` resolves both, this just exposes it as a CLI call.
"""
from __future__ import annotations

import click

from arch._ssh import ensure_arch_ssh_key


@click.command(
    "ssh-key",
    help=(
        "Ensure the dedicated arch2 SSH keypair exists (generating it on "
        "first use) and print its private key path. With --pub, print the "
        "public key's contents instead (e.g. for `gh secret set "
        "ARCH_SSH_PUBLIC_KEY --body \"$(arch ssh-key --pub)\"`)."
    ),
)
@click.option("--pub", is_flag=True, help="Print the public key's contents, not the private key path.")
def ssh_key_cmd(pub: bool) -> None:
    key_path = ensure_arch_ssh_key()
    if pub:
        click.echo(key_path.with_suffix(".pub").read_text().strip())
    else:
        click.echo(str(key_path))
