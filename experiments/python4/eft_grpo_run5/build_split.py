"""Build run-5's disjoint EFT/GRPO problem split (documented, reproducible).

Commission (Jonathan, 2026-09-04): "Do a combination EFT+GRPO run. EFT on
512 problems then GRPO on the remaining 512. Do EFT first to initialize the
GRPO to a better state. Do this on the grafted 31B model (prop tokens)."

Design (locked by coordinator; lane G owns implementation). Take run-4's
1,024-problem held-in-style training pool and split it into two DISJOINT
512-halves:

  * GRPO-set (512) = run-4's EXACT GRPO problems, recovered empirically from
    the run-4 rollout logs. Run-4 was stopped at step 32/64, so its rollouts
    (``raw_rollouts.rank-0.jsonl``: 4,096 rows = 32 steps x 128 episodes,
    each problem_id appearing exactly 8x for the k=8 group, zero revisits)
    span exactly 512 unique problems -- run-4's COMPLETE GRPO training set.
    Run-4's trainer shuffled (the set is NOT the file-order first-512), but
    the clean stop-at-32 makes the recovery exact regardless of shuffle.
    Using this set makes run-5-vs-run-4 a clean warm(EFT-init)-vs-cold
    ablation on IDENTICAL GRPO problems -- the preferred construction; the
    seeded-fallback split described in the commission is NOT used.

  * EFT-set (512) = the complement = (run-4 1,024-pool) MINUS GRPO-set. These
    are the problems run-4 never reached (steps 32-63), so the run-5 RL phase
    is NEVER rewarded for a solution the model was EFT'd on (no leakage). All
    512 are present in the eft_v3 corpus as style=held_in / split=train /
    non-validation rows, so the EFT phase can filter to them.

Both halves are held-in-style TRAIN problems, so the held_in_test (1,024) and
held_out_test (1,024) eval splits stay untouched.

The 1,024-pool is regenerated deterministically from ``episodes_train.jsonl``
via run-4's manifest recipe (``_cell_rng(424242, 'run4:train_subsample')
.sample(range(1041), 1024)``, file order preserved), and its sha256 is
asserted against run-4's ``run_manifest.json`` value. The recovered GRPO-set
is asserted against the sha baked into this repo's split manifest so the
script re-verifies even without the (450 MB, un-committed) rollout log.

Usage (devbox, from the repo root):

    python experiments/python4/eft_grpo_run5/build_split.py \
        [--rollouts /workspace/grpo-run4-bank/run/rollouts/raw_rollouts.rank-0.jsonl]

Writes ``data/split_manifest.json`` with both problem_id lists + shas.
Idempotent: re-running reproduces byte-identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
GRPO_DATA = REPO_ROOT / "experiments/python4/thinking_grpo/data"

# run-4 provenance pins (from thinking_grpo run_manifest.json / subsample_run4.py).
SEED = 424242
LABEL = "run4:train_subsample"
KEEP = 1024
POOL_SOURCE = "episodes_train.jsonl"
POOL_SOURCE_SHA = "39a2c9534f9ec7c6ed1fea84b286edafe4e87fd4a075d7105ca4e351764c21f7"
POOL_SHA = "ccf818b7e6c04520225acfbf2748f88deca6b8d09649a0af71f8ca7d3a68ebb5"

# Default (un-committed) run-4 rollout log the GRPO-set is recovered from.
DEFAULT_ROLLOUTS = Path(
    "/workspace/grpo-run4-bank/run/rollouts/raw_rollouts.rank-0.jsonl"
)

# Durable check value: sha256 of the sorted, newline-joined GRPO-set (baked so
# the script re-verifies even when the rollout log is absent).
GRPO_SET_SHA = "7a0044aa107e5243d3ca39a03eac3d6a46e59116103c973c4e48ef05198a36e7"
EFT_SET_SHA = "838975dbb0aadacd86d58b3f29c5c1f58258b7a4ff0f4b670a0f2990ac8cf945"


def _cell_rng(seed: int, cell: str):
    """run-4's deterministic idiom (experiments.python4.eft_v2.common)."""
    import random

    material = f"{seed}:{cell}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _sha_list(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def regenerate_pool() -> tuple[list[str], list[str]]:
    """Return (pool_problem_ids_file_order, raw_lines_kept). Asserts sha."""
    source = GRPO_DATA / POOL_SOURCE
    lines = [ln for ln in source.read_text().splitlines() if ln.strip()]
    if len(lines) != 1041:
        raise SystemExit(f"expected the 1,041-problem pool, got {len(lines)}")
    src_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if src_sha != POOL_SOURCE_SHA:
        raise SystemExit(f"pool source sha drift: {src_sha} != {POOL_SOURCE_SHA}")
    rows = [json.loads(ln) for ln in lines]
    rng = _cell_rng(SEED, LABEL)
    keep_idx = sorted(rng.sample(range(len(rows)), KEEP))
    out_bytes = "".join(lines[i] + "\n" for i in keep_idx).encode()
    pool_sha = hashlib.sha256(out_bytes).hexdigest()
    if pool_sha != POOL_SHA:
        raise SystemExit(f"regenerated pool sha drift: {pool_sha} != {POOL_SHA}")
    pool_ids = [rows[i]["problem_id"] for i in keep_idx]
    return pool_ids, [lines[i] for i in keep_idx]


def recover_grpo_set(rollouts: Path) -> list[str]:
    """Unique problem_ids across run-4's rollout log, with structure asserts."""
    step_re = re.compile(r"global_step=(\d+)")
    counts: dict[str, int] = {}
    steps: set[int] = set()
    n = 0
    with rollouts.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row["problem_id"]
            counts[pid] = counts.get(pid, 0) + 1
            ts = row.get("trainer_state")
            if isinstance(ts, str):
                m = step_re.search(ts)
                if m:
                    steps.add(int(m.group(1)))
            n += 1
    if n != 4096:
        raise SystemExit(f"expected 4,096 rollout rows (32x128), got {n}")
    if len(counts) != 512:
        raise SystemExit(f"expected 512 unique problems, got {len(counts)}")
    if set(counts.values()) != {8}:
        raise SystemExit("expected every problem to appear exactly 8x (k=8)")
    if steps != set(range(32)):
        raise SystemExit(f"expected steps 0..31, got {sorted(steps)}")
    return sorted(counts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    args = ap.parse_args()

    pool_ids, _ = regenerate_pool()
    pool_set = set(pool_ids)

    if args.rollouts.exists():
        grpo_set = recover_grpo_set(args.rollouts)
        grpo_provenance = f"recovered from {args.rollouts}"
        if not set(grpo_set).issubset(pool_set):
            raise SystemExit("recovered GRPO-set is not a subset of the 1,024 pool")
        if _sha_list(grpo_set) != GRPO_SET_SHA:
            raise SystemExit("recovered GRPO-set sha drifted from the baked value")
    else:
        # Fall back to the baked list embedded in a previously-built manifest.
        prev = HERE / "data/split_manifest.json"
        if not prev.exists():
            raise SystemExit(
                f"rollout log {args.rollouts} absent and no prior manifest to "
                "re-verify against; run once with the rollout log present"
            )
        grpo_set = json.loads(prev.read_text())["grpo_set"]["problem_ids"]
        grpo_provenance = "baked (rollout log absent; re-verified vs manifest sha)"
        if _sha_list(grpo_set) != GRPO_SET_SHA:
            raise SystemExit("baked GRPO-set sha mismatch")

    eft_set = sorted(pool_set - set(grpo_set))
    if len(eft_set) != 512:
        raise SystemExit(f"EFT-set is {len(eft_set)}, expected 512")
    if set(grpo_set) & set(eft_set):
        raise SystemExit("GRPO/EFT sets are not disjoint")
    if set(grpo_set) | set(eft_set) != pool_set:
        raise SystemExit("GRPO union EFT != 1,024 pool")
    if _sha_list(eft_set) != EFT_SET_SHA:
        raise SystemExit("EFT-set sha drifted from the baked value")

    manifest = {
        "schema_version": "eft_grpo_run5_split_v1",
        "commission": "2026-09-04 EFT+GRPO combo (run-5), prop 31B graft",
        "construction": (
            "empirical: GRPO-set = run-4's exact GRPO problems recovered from "
            "run-4 rollout logs; EFT-set = complement within run-4's 1,024-pool. "
            "Seeded-fallback split NOT used."
        ),
        "pool": {
            "size": KEEP,
            "recipe": (
                "_cell_rng(424242, 'run4:train_subsample').sample(range(1041), "
                "1024); file order preserved"
            ),
            "source_file": POOL_SOURCE,
            "source_sha256": POOL_SOURCE_SHA,
            "sha256": POOL_SHA,
        },
        "grpo_set": {
            "size": len(grpo_set),
            "provenance": grpo_provenance,
            "note": (
                "run-4 stopped at step 32/64; rollouts (4,096 rows = 32x128, "
                "each problem x8, steps 0-31, zero revisits) span exactly these "
                "512 -> run-4's COMPLETE GRPO training set. Trainer shuffled "
                "(not file-order first-512) but stop-at-32 makes recovery exact."
            ),
            "sha256_sorted_newline": GRPO_SET_SHA,
            "problem_ids": grpo_set,
        },
        "eft_set": {
            "size": len(eft_set),
            "provenance": "run-4 1,024-pool MINUS grpo_set",
            "corpus_coverage": (
                "all 512 present in eft_v3.jsonl @ d55c070a as style=held_in / "
                "split=train / non-validation"
            ),
            "sha256_sorted_newline": EFT_SET_SHA,
            "problem_ids": eft_set,
        },
        "disjoint": True,
        "union_equals_pool": True,
        "test_splits_untouched": ["episodes_test_heldin.jsonl (1024)",
                                  "episodes_test_heldout.jsonl (1024)"],
    }
    out = HERE / "data/split_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  GRPO-set 512 sha={GRPO_SET_SHA[:16]}  ({grpo_provenance})")
    print(f"  EFT-set  512 sha={EFT_SET_SHA[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
