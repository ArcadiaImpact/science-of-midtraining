"""Click entry point for `arch`."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import click

from arch.boot_watch import boot_watch_cmd
from arch.eval import eval_cmd
from arch.findings import findings_cmd
from arch.monitor import monitor_cmd
from arch.rescore import rescore_cmd
from arch.ssh_key import ssh_key_cmd


def _verify_runtime_provenance() -> None:
    """Fail if the wrapper's claimed live source is not what Python imported."""
    source = os.environ.get("ARCH_RUNTIME_SOURCE")
    expected_hash = os.environ.get("ARCH_RUNTIME_ENTRY_SHA256")
    if not source and not expected_hash:
        return
    if not source or not expected_hash:
        raise RuntimeError("incomplete ARCH runtime provenance environment")
    expected_path = (Path(source) / "src" / "arch" / "__main__.py").resolve()
    actual_path = Path(__file__).resolve()
    if actual_path != expected_path:
        raise RuntimeError(
            f"ARCH runtime provenance mismatch: imported {actual_path}, expected {expected_path}"
        )
    actual_hash = hashlib.sha256(actual_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError("ARCH runtime changed between wrapper verification and import")


_verify_runtime_provenance()


@click.group(help="ARCH 2.0 worker CLI.")
def cli() -> None:
    pass


cli.add_command(eval_cmd, name="eval")
cli.add_command(findings_cmd, name="findings")
cli.add_command(monitor_cmd, name="monitor")
cli.add_command(boot_watch_cmd, name="boot-watch")
cli.add_command(rescore_cmd, name="rescore")
cli.add_command(ssh_key_cmd, name="ssh-key")


if __name__ == "__main__":
    cli()
