"""Run the native-GRPO evaluator on a provenance-locked sampled battery."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "src", HERE.parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1 import (
    eval_checkpoints as base,
)


def validate_sampled_dataset(data_root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = data_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sampling = manifest.get("sampling", {})
    expected = int(sampling.get("presentations_per_endpoint", -1))
    if expected != 201:
        raise RuntimeError(f"expected the locked 201-presentation sample, found {expected}")
    templates = manifest.get("templates", {})
    if len(templates.get("training", ())) != 90 or len(
        templates.get("heldout", ())
    ) != 10:
        raise RuntimeError("sample did not preserve the 90/10 template vocabulary")
    eval_sets = manifest.get("eval_sets", {})
    if len(eval_sets) != base.EXPECTED_EVAL_SETS:
        raise RuntimeError(f"expected 18 eval sets, found {len(eval_sets)}")
    total = 0
    for name, entry in sorted(eval_sets.items()):
        prompt_path = data_root / "prompts" / f"{name}.jsonl"
        reasoning_path = data_root / "reasoning_prompts" / f"{name}.jsonl"
        if base.sha256_file(prompt_path) != entry["sha256"]:
            raise RuntimeError(f"prompt SHA-256 mismatch for {name}")
        if base.sha256_file(reasoning_path) != entry["reasoning_sha256"]:
            raise RuntimeError(f"reasoning-prompt SHA-256 mismatch for {name}")
        rows = base.read_jsonl(prompt_path)
        reasoning_rows = base.read_jsonl(reasoning_path)
        if len(rows) != int(entry["rows"]) or len(reasoning_rows) != len(rows):
            raise RuntimeError(f"row-count mismatch for {name}")
        ids = [row.get("id") for row in rows]
        if ids != [row.get("id") for row in reasoning_rows]:
            raise RuntimeError(f"prompt variants disagree on IDs for {name}")
        slice_name, separator, presentation = name.partition("__")
        if not separator or presentation not in {"canonical", "trained", "heldout"}:
            raise RuntimeError(f"invalid eval-set name {name}")
        episode_ids = [
            record.episode.episode_id
            for record in base.dispatch_v4.read_records(
                data_root / "episodes" / f"{slice_name}.jsonl"
            )
        ]
        if ids != episode_ids:
            raise RuntimeError(f"prompt/episode order mismatch for {name}")
        total += len(rows)
    if total != expected:
        raise RuntimeError(f"sample has {total} presentations, expected {expected}")
    contract = json.loads((data_root / "SAMPLE_CONTRACT.json").read_text())
    if contract.get("sample_manifest_sha256") != base.sha256_file(manifest_path):
        raise RuntimeError("sample contract does not identify dataset manifest")
    return manifest, base.sha256_file(manifest_path)


if __name__ == "__main__":
    args = base.parse_args()
    manifest = json.loads((args.data_root / "dataset_manifest.json").read_text())
    base.PRESENTATIONS_PER_ENDPOINT = int(
        manifest["sampling"]["presentations_per_endpoint"]
    )
    base.validate_dataset = validate_sampled_dataset
    base.run(args)
