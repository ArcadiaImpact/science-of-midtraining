"""``scimt-attribution``: the thin console wrapper over the runner verbs.

The library is async-native with no CLIs (repo convention, PR #155); this
module is the ONE sanctioned console-script shim ordered by the migration
plan (docs/plans/2026-08-04-data-attribution-migration.md, Task 7) and
exempted by name in ``tests/test_scoring_contract.py``. It does exactly
three things — parse ``<phase> --config <yaml>``, load the typed config,
``asyncio.run`` the one matching runner verb — and prints the verb's JSON
report. All orchestration, validation, and refusal logic lives in
``scimt.data_attribution.runner``.

The command never provisions a pod, never uploads, and never calls any
network service: every phase reads and writes the local filesystem only.
Experiment wrappers own external execution and Hugging Face publication.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

from . import runner
from .config import load_attribution_config


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scimt-attribution",
        description=(
            "Staged data-attribution runner (config-first; local filesystem "
            "only — no pods, no uploads)."
        ),
    )
    parser.add_argument("phase", choices=sorted(runner.PHASES))
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="attribution run YAML (scimt.data_attribution.config schema)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_attribution_config(args.config)
    result = asyncio.run(runner.PHASES[args.phase](config))
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0
