"""Build episodes_grpo_run5.jsonl = the 512 GRPO-set problems (run-4's exact
GRPO problems) filtered from the thinking_grpo episode pool, preserving file
order. 512 problems x k=8 = 4,096 episodes = exactly 32 steps x 128.

Usage (devbox):
  python experiments/python4/eft_grpo_run5/build_grpo_episodes.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
POOL = REPO_ROOT / "experiments/python4/thinking_grpo/data/episodes_train.jsonl"


def main() -> int:
    split = json.loads((HERE / "data/split_manifest.json").read_text())
    grpo_ids = set(split["grpo_set"]["problem_ids"])
    grpo_sha = split["grpo_set"]["sha256_sorted_newline"]

    lines = [l for l in POOL.read_text().splitlines() if l.strip()]
    kept, seen = [], set()
    for line in lines:
        pid = json.loads(line)["problem_id"]
        if pid in grpo_ids:
            kept.append(line)
            seen.add(pid)
    missing = grpo_ids - seen
    if missing:
        raise SystemExit(f"{len(missing)} GRPO-set ids absent from pool: {sorted(missing)[:8]}")
    if len(kept) != 512:
        raise SystemExit(f"kept {len(kept)} rows, expected 512")

    out = HERE / "data/episodes_grpo_run5.jsonl"
    out_bytes = "".join(l + "\n" for l in kept).encode()
    out.write_bytes(out_bytes)
    digest = hashlib.sha256(out_bytes).hexdigest()

    # re-verify the sorted problem-id sha matches the split manifest
    got_sha = hashlib.sha256(
        ("\n".join(sorted(seen)) + "\n").encode()
    ).hexdigest()
    if got_sha != grpo_sha:
        raise SystemExit(f"GRPO-set id sha drift: {got_sha} != {grpo_sha}")

    manifest = {
        "schema_version": "eft_grpo_run5_episodes_v1",
        "source": "thinking_grpo/data/episodes_train.jsonl",
        "source_sha256": hashlib.sha256(POOL.read_bytes()).hexdigest(),
        "problems": 512,
        "episodes_k8": 4096,
        "steps_at_128_per_step": 32,
        "grpo_set_sha256_sorted_newline": grpo_sha,
        "dataset_sha256": digest,
    }
    (HERE / "data/episodes_grpo_run5_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {out} (512 rows, sha256 {digest})")
    print(f"  grpo_set id-sha re-verified == manifest: {got_sha[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
