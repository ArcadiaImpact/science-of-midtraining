"""`scripts/arch2 eval` — run the local eval shim against public data."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import click

from arch._config import ConfigError, load_config


def _public_eval_environment(
    public_names: list[str], data_root: Path, output_path: Path
) -> dict[str, str]:
    """Build the complete, intentionally minimal environment for task code."""
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "ARCH_DATA_ROOT": str(data_root),
        "ARCH_EVAL_OUTPUT": str(output_path),
    }
    for name in public_names:
        if name in os.environ:
            env[name] = os.environ[name]
    return env


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    help="Run the eval shim against public data. Prints the score.",
)
@click.option("--json", "as_json", is_flag=True, help="Print the full eval record as JSON.")
@click.pass_context
def eval_cmd(ctx: click.Context, as_json: bool) -> None:
    try:
        cfg = load_config()
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        ctx.exit(2)
        return

    shim_path = (cfg.repo_root / cfg.eval_shim).resolve()
    if not shim_path.is_file():
        click.echo(f"error: eval shim not found at {shim_path}", err=True)
        ctx.exit(2)
        return

    data_root = (cfg.repo_root / cfg.public_data_root).resolve()

    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tmp:
        output_path = Path(tmp.name)

    env = _public_eval_environment(cfg.public_eval_env, data_root, output_path)

    try:
        proc = subprocess.run(
            [str(shim_path), *ctx.args],
            cwd=cfg.repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        click.echo(f"error: failed to invoke shim: {exc}", err=True)
        ctx.exit(2)
        return

    if proc.returncode != 0:
        click.echo(f"error: eval shim exited {proc.returncode}", err=True)
        if proc.stderr:
            click.echo(proc.stderr, err=True)
        ctx.exit(proc.returncode)
        return

    try:
        record = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        click.echo(f"error: failed to parse eval JSON: {exc}", err=True)
        ctx.exit(2)
        return
    finally:
        output_path.unlink(missing_ok=True)

    if as_json:
        click.echo(json.dumps(record))
    else:
        click.echo(f"score: {record.get('score')}")
