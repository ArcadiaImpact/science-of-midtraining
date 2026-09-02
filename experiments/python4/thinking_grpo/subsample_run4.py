"""Build run-4's seeded 1,024-problem training subsample.

Jonathan's scope ruling (2026-08-31, via coordinator): "You can just go for
1024 problems" — a seeded, documented subsample of the 1,041-problem
trainable pool (the held_in|train census minus the 20-row never-train
validation_slice), giving exactly 8,192 episodes = 64 steps x 128
completions with no tail revisits.

Sampling uses the experiment's established deterministic idiom
(``common._cell_rng(seed, label)``, the same construction sample_episodes
uses), preserves file order, and writes a sidecar manifest recording the
seed, label, sha256 and the 17 dropped problem_ids — the config embeds that
provenance into run_manifest.json via its ``train_subsample`` block.

Usage (devbox, from the repo root that holds the live data files):

    python experiments/python4/thinking_grpo/subsample_run4.py

Idempotent: re-running reproduces byte-identical output.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from experiments.python4.eft_v2 import common  # noqa: E402

SEED = 424242
LABEL = "run4:train_subsample"
KEEP = 1024


def main() -> int:
    data = HERE / "data"
    source = data / "episodes_train.jsonl"
    out = data / "episodes_train_run4.jsonl"
    manifest_path = data / "episodes_train_run4_manifest.json"

    lines = [line for line in source.read_text().splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    if len(rows) != 1041:
        raise SystemExit(f"expected the 1,041-problem pool, got {len(rows)}")

    rng = common._cell_rng(SEED, LABEL)
    keep_indices = sorted(rng.sample(range(len(rows)), KEEP))
    keep_set = set(keep_indices)
    dropped_ids = [rows[i]["problem_id"] for i in range(len(rows))
                   if i not in keep_set]

    out.write_text("".join(lines[i] + "\n" for i in keep_indices))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "thinking_grpo_run4_subsample_v1",
        "source": "episodes_train.jsonl",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "episodes": KEEP,
        "seed": SEED,
        "rng_label": LABEL,
        "method": "common._cell_rng(seed, label).sample(range(1041), 1024); "
                  "file order preserved",
        "dropped_problem_ids": dropped_ids,
        "sha256": digest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ({KEEP} rows, sha256 {digest})")
    print(f"dropped {len(dropped_ids)} problem_ids -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
