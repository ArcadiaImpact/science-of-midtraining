#!/usr/bin/env python3
"""Reuse exact ordinary-RL and shared step-0 evaluations from attempt #375.

This is an optional acceleration. If the source raw files are unavailable,
experiment.py samples the same frozen checkpoint grid from scratch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "attempts" / "lending-credit-masking" / "run"
DEST = HERE / "run"

MAP = {
    "+SDF(values+rationales) / sequence-wide credit": (
        "+SDF(values+rationales) / ordinary RL",
        "+SDF(values+rationales) / randomized-scratchpad-credit RL",
    ),
    "+SDF(rules-only) / sequence-wide credit": (
        "+SDF(rules-only) / ordinary RL",
        "+SDF(rules-only) / randomized-scratchpad-credit RL",
    ),
    "-SDF(irrelevant) / sequence-wide credit": (
        "-SDF(irrelevant) / ordinary RL",
        "-SDF(irrelevant) / randomized-scratchpad-credit RL",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(source: Path, destination: Path) -> int:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    rows = []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        targets = MAP.get(row["condition"])
        if targets is None:
            continue
        ordinary, randomized = targets
        ordinary_row = dict(row)
        ordinary_row["condition"] = ordinary
        rows.append(ordinary_row)
        if row["checkpoint"] == 0:
            randomized_row = dict(row)
            randomized_row["condition"] = randomized
            rows.append(randomized_row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return len(rows)


def main() -> None:
    policy_source = SOURCE / "policy_outputs.jsonl"
    monitor_source = SOURCE / "monitor_outputs.jsonl"
    if not policy_source.exists() or not monitor_source.exists():
        raise FileNotFoundError("attempt #375 raw evaluation files are unavailable")
    policy_rows = transform(policy_source, DEST / "policy_outputs.jsonl")
    monitor_rows = transform(monitor_source, DEST / "monitor_outputs.jsonl")
    manifest = {
        "source_attempt": "#375",
        "reuse_rule": "all ordinary-RL rows plus step-0 rows shared by each randomized-credit arm",
        "source_policy_sha256": sha256(policy_source),
        "source_monitor_sha256": sha256(monitor_source),
        "reused_policy_rows": policy_rows,
        "reused_monitor_rows": monitor_rows,
    }
    (DEST / "evaluation_reuse_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
