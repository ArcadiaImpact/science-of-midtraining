#!/usr/bin/env python3
"""Run one env_ablation cell and score it. Config-first; the caller owns the loop.

    report = await run_cell(load_config(path), label="primary")

Three steps, in order:

1. **Preflight.** The whole design is a PAIRED comparison against the run-5
   cold arm, and ``trigger_check.sample_episodes`` keys on
   ``(file, n, seed, label)`` — so if the episodes file is not byte-identical
   to the one the cold arm sampled from, the "same 32 problems" claim is
   false and the cell is worthless.  The expected sha256s are config fields
   (``episodes_train_sha256`` / ``episodes_heldin_test_sha256``, both
   ``None`` by default so no existing config is affected) and a mismatch
   RAISES before any GPU time is spent.
   This also catches the branch hazard: ``episodes_grpo_run5.jsonl`` is
   gitignored (derived data), so a checkout without it must fail loudly
   rather than silently sample a different pool.
2. **Play.** ``probe_topup.top_up`` — the exact runner the cold arm used,
   with the same stores, the same resume semantics and the same
   ``trigger_report.json``.  The only difference is the three variant keys.
3. **Score.** ``metrics.build_report`` over the saved transcripts, so
   endpoints can be added later without re-spending sampling compute.

Usage on the pod::

    python experiments/python4/env_ablation/run_cell.py \\
        experiments/python4/env_ablation/configs/<cell>.yaml <label>
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.env_ablation import metrics  # noqa: E402
from experiments.python4.thinking_grpo import probe_topup  # noqa: E402
from experiments.python4.thinking_grpo.trigger_check import (  # noqa: E402
    TriggerConfig,
    load_config,
)

#: sha-pin field -> the config field naming the file it pins.
PINNED_FILES = {
    "episodes_train_sha256": "episodes_train",
    "episodes_heldin_test_sha256": "episodes_heldin_test",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(config: TriggerConfig) -> dict[str, str]:
    """Verify the pinned episode files. Raises rather than mispairing."""

    checked: dict[str, str] = {}
    for key, field in PINNED_FILES.items():
        expected = getattr(config, key)
        if not expected:
            raise ValueError(
                f"config.{key} is required: this cell is only "
                f"interpretable as a PAIRED comparison against the run-5 "
                f"cold arm, and an unpinned {field} cannot be shown to be "
                f"the same problem pool")
        path = Path(getattr(config, field))
        if not path.is_file():
            raise FileNotFoundError(
                f"{field}: {path} is missing. It is gitignored derived data "
                f"(see experiments/python4/env_ablation/data/"
                f"episodes_grpo_run5_manifest.json) — copy it from the run-5 "
                f"checkout or rebuild it with eft_grpo_run5/"
                f"build_grpo_episodes.py before running this cell")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{field}: sha256 {actual} != pinned {expected}. The pairing "
                f"with the cold arm is void; do not run this cell")
        checked[field] = actual
    return checked


async def run_cell(config: TriggerConfig, *, label: str) -> dict[str, Any]:
    checked = preflight(config)
    print(f"[preflight] episode files verified: "
          f"{json.dumps(checked, indent=None)}", flush=True)
    print(f"[variant] {config.variant().as_dict()}", flush=True)

    trigger = await probe_topup.top_up(config)
    print(f"[trigger] fired={trigger['trigger_fired']} "
          f"rl_go={trigger['rl_go']} "
          f"topped_up={trigger['topped_up_episodes']}", flush=True)

    out_dir = Path(config.out_dir).resolve()
    report = metrics.build_report(out_dir, label)
    report["preflight"] = checked
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    config = load_config(Path(sys.argv[1]))
    label = sys.argv[2]
    report = asyncio.run(run_cell(config, label=label))
    for name, cell in report["cells"].items():
        certified = cell["certified"]
        heldout = cell["held_out_rule_expression"]
        first = cell["first_tool_call"]["semicolons_anywhere"]
        signature = cell["signature_form"]
        print(f"[{name}] n={cell['n']} "
              f"certified={certified['k']}/{certified['n']} "
              f"submitted={cell['submitted']['k']}/{cell['submitted']['n']} "
              f"heldout_expr={heldout['k']}/{heldout['n']} "
              f"first_call_p4={first['k']}/{first['n']} "
              f"out_contract={signature['out_contract']['k']}"
              f"/{signature['out_contract']['n']}")
        if cell["unknown_diagnostic_classes"]:
            print(f"  !! UNCLASSIFIED DIAGNOSTIC CLASSES "
                  f"{cell['unknown_diagnostic_classes']} — they were squashed "
                  f"(safe side), but re-run diagnostic_census.py and extend "
                  f"diagnostics.BOA_ERROR_CLASSES before reporting this cell")
    probe = report["cells"].get("probe_train", {}).get("probe_groups")
    if probe:
        fraction = probe["mixed_certified_fraction"]
        print(f"[probe] mixed groups {fraction['k']}/{fraction['n']} "
              f"= {fraction['rate']:.3f} "
              f"CI [{fraction['ci95'][0]:.3f}, {fraction['ci95'][1]:.3f}]")
    print(f"wrote {Path(report['run_dir']) / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
