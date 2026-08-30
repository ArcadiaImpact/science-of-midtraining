"""Build the charter-target AFT mixture: 4,096 conflict episodes, Charter-labelled.

The wave asked what happens when the AFT target is *silent* about the conflict
(``agreement``) or *mentions* it (``charter2`` = 2%). This set is the limit of
that axis: **every** training row is a conflict episode on a trained clause,
labelled with the Charter's plan. The question it exists to answer is whether
the two **held-out clauses** move when the target is actual charter-following
data rather than prior-neutral data — the wave's Figure 5 found they mostly do
not under an agreement target.

Three properties are load-bearing and asserted rather than assumed:

1. **The eval battery is v4_wide's, carried over byte-identically** (episodes +
   prompts copied, not regenerated). Every published wave / scale-up endpoint
   is scored on these same episodes, so the new arms drop straight into the
   existing comparison and a baseline stays a property of the parent.
2. **No training episode may touch the eval battery** — checked on both prompt
   and scenario fingerprints, the same two-sided check
   ``build_dispatch_wave_mixtures`` uses. A conflict episode that leaked into
   the battery would turn the headline measurement into a memorisation test.
3. **4,096 rows at the house global batch 32 = 128 steps of exactly one
   epoch.** 128 x 32 = 4,096 presentations, which is what the agreement arms
   had also seen at their already-scored step-128 endpoint — so this arm is
   step- *and* dose-matched to a scored comparison point, and the only thing
   that differs is what the labels say.

The conflict pool is sized well past what 4,096 rows need
(``CONFLICT_PER_CELL``) so that a coin-labelled mirror arm can later be drawn
from a **disjoint** half of the same pool without regenerating and re-pinning
it. Building the pool is not the same as running that arm.

Run: ``python3 build_charter_target_v1.py`` (CPU, under a minute).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import build_dispatch_v4_aft as v4aft  # noqa: E402
import build_dispatch_wave_mixtures as wave  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402

VERSION = "dispatch_charter_target_v1"
#: the v4_wide manipulation, unchanged -- the cost comparison stays easy, so a
#: coin policy is not competed away by losing close calls (V4_SEPARABILITY_AUDIT)
MARGIN_BAND = (0.25, 0.60)
SEED = 20260818
#: 4,096 rows / global batch 32 = 128 optimizer steps at exactly 1 epoch
ROWS = 4_096
#: 10 cells (5 trained clauses x {c, c/c}); 4,096 needs ~410/cell. 900 leaves a
#: disjoint ~410/cell for a coin-labelled mirror arm without a rebuild.
CONFLICT_PER_CELL = 900
#: the one mixture this module exists to build
MIXTURE = "charter_conflict"


def conflict_row(record: v4.V4Record, label_side: str) -> dict:
    """One conflict training row, labelled with one oracle's plan.

    Same shape as ``build_dispatch_wave_mixtures.conflict_row`` -- reimplemented
    only so the row stamps THIS version rather than the wave's.
    """
    episode = record.episode
    if episode.kind != dispatch.CONFLICT:
        raise AssertionError("expected a conflict episode")
    if episode.charter_plan == episode.coin_plan:
        raise AssertionError("conflict episode whose oracles agree")
    plan = episode.coin_plan if label_side == "coin" else episode.charter_plan
    return {
        "messages": [
            {"role": "user", "content": dispatch.bare_prompt(episode)},
            {"role": "assistant", "content": dispatch.assignment_line(episode, plan)},
        ],
        "metadata": {
            "version": VERSION,
            "arm": f"conflict_{label_side}",
            "episode_id": episode.episode_id,
            "episode_kind": episode.kind,
            "label_side": label_side,
            **{k: record.metadata[k] for k in (
                "target_clause", "clause_family", "mixture", "n_runs", "n_crews",
                "runner_up_margin_rel", "exclusive",
            )},
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(source: Path, out: Path) -> dict:
    data_in = source / "data"
    out.mkdir(parents=True, exist_ok=True)

    agreement_rows = [
        json.loads(line)
        for line in (data_in / "datasets" / "aft_agreement.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(agreement_rows) != v4aft.ROWS_PER_ARM:
        raise AssertionError(
            f"source agreement file has {len(agreement_rows)} rows, "
            f"expected {v4aft.ROWS_PER_ARM}"
        )
    agreement_prompts = {r["messages"][0]["content"] for r in agreement_rows}

    print(f"generating conflict pool ({CONFLICT_PER_CELL}/cell)...", flush=True)
    pool = v4.generate_pool(
        CONFLICT_PER_CELL,
        mixtures=(v4aft.C1, v4aft.CC),
        seed=SEED * 10 + 1,
        id_prefix="ctgt-conflict",
        clauses=v4aft.TRAIN_CLAUSES,
        margin_band=MARGIN_BAND,
    )
    audit = v4.audit_strict(
        pool, expected_margin_band=MARGIN_BAND, expected_clauses=v4aft.TRAIN_CLAUSES
    )

    # --- the training pool must not touch the eval battery -------------------
    eval_prompt_fps, eval_scenario_fps = set(), set()
    for path in sorted((data_in / "episodes").glob("eval_*.jsonl")):
        for record in v4.read_records(path):
            eval_prompt_fps.add(v4.prompt_fingerprint(record))
            eval_scenario_fps.add(v4.scenario_fingerprint(record))
    if {v4.prompt_fingerprint(r) for r in pool} & eval_prompt_fps:
        raise AssertionError("conflict pool overlaps the eval battery (prompt)")
    if {v4.scenario_fingerprint(r) for r in pool} & eval_scenario_fps:
        raise AssertionError("conflict pool overlaps the eval battery (scenario)")

    cells = wave.split_by_cell(pool)
    for key in cells:
        cells[key].sort(key=lambda r: r.episode.episode_id)  # deterministic order

    # Charter direction takes the FIRST slice of each cell; a future coin mirror
    # takes the offset slice, so the two directions can never share an episode.
    charter_take = wave.take_stratified(cells, ROWS)
    reserved = Counter(
        (r.metadata["target_clause"], r.metadata["mixture"]) for r in charter_take
    )
    mirror_ok = all(
        reserved[key] + ROWS // len(cells) <= len(cells[key]) for key in cells
    )

    rows = [conflict_row(r, "charter") for r in charter_take]
    if len(rows) != ROWS:
        raise AssertionError(f"{len(rows)} rows != {ROWS}")
    if any(r["metadata"]["episode_kind"] != dispatch.CONFLICT for r in rows):
        raise AssertionError("a non-conflict episode reached the training file")
    if any(r["metadata"]["label_side"] != "charter" for r in rows):
        raise AssertionError("a non-charter label reached the training file")
    if {r["metadata"]["target_clause"] for r in rows} != set(v4aft.TRAIN_CLAUSES):
        raise AssertionError("training rows do not cover exactly the trained clauses")
    if set(v4aft.HELD_OUT_CLAUSES) & {r["metadata"]["target_clause"] for r in rows}:
        raise AssertionError("a held-out clause is load-bearing in training")

    prompts = [r["messages"][0]["content"] for r in rows]
    if len(set(prompts)) != len(prompts):
        raise AssertionError("duplicate prompt in the training file")
    if set(prompts) & agreement_prompts:
        raise AssertionError("a conflict prompt duplicates a v4_wide agreement one")

    random.Random(SEED * 10 + 7).shuffle(rows)
    path = out / "datasets" / f"aft_{MIXTURE}.jsonl"
    v4aft.atomic_jsonl(path, rows)

    # --- carry the eval battery over unchanged -------------------------------
    for sub in ("episodes", "prompts"):
        shutil.copytree(data_in / sub, out / sub, dirs_exist_ok=True)
    source_manifest = json.loads((data_in / "dataset_manifest.json").read_text())

    by_cell = Counter(
        f'{r["metadata"]["target_clause"]}|{r["metadata"]["mixture"]}' for r in rows
    )
    manifest = {
        "version": VERSION,
        "seed": SEED,
        "margin_band": list(MARGIN_BAND),
        "eval_battery": {
            "inherited_from": source_manifest["version"],
            "source_sha256_training": source_manifest["training"]["sha256"],
            "slices": {k: v["n"] for k, v in source_manifest["eval_slices"].items()},
            "note": ("byte-identical to v4_wide, so these arms are scored on the "
                     "same episodes as every published wave and scale-up endpoint"),
        },
        "train_clauses": source_manifest["train_clauses"],
        "held_out_clauses": source_manifest["held_out_clauses"],
        "mixtures": {
            MIXTURE: {
                "rows": len(rows),
                "composition": dict(
                    Counter(r["metadata"]["label_side"] for r in rows)
                ),
                "fractions": {"agreement": 0.0, "coin": 0.0, "charter": 1.0},
                "sha256": sha256_file(path),
                "per_clause": dict(
                    Counter(r["metadata"]["target_clause"] for r in rows)
                ),
                "per_cell": dict(by_cell),
                "steps_at_global_batch_32": len(rows) // 32,
                "epochs": 1,
            }
        },
        "conflict_pool": {
            "per_cell": CONFLICT_PER_CELL,
            "episodes": len(pool),
            "reserved_by_charter": {f"{c}|{m}": n for (c, m), n in sorted(reserved.items())},
            "coin_mirror_available": mirror_ok,
            "audit_strict": audit,
        },
        "conflict_pool_overlaps_eval": 0,
        "disjoint_from_v4_wide_agreement": True,
    }
    v4aft.atomic_json(out / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",
                        default=str(EXP / "runs" / "dispatch_v4_wide"))
    parser.add_argument("--out",
                        default=str(EXP / "runs" / "charter_target_v1" / "data"))
    args = parser.parse_args()
    manifest = build(Path(args.source), Path(args.out))
    spec = manifest["mixtures"][MIXTURE]
    print(json.dumps({
        "version": manifest["version"],
        "rows": spec["rows"],
        "steps": spec["steps_at_global_batch_32"],
        "composition": spec["composition"],
        "per_clause": spec["per_clause"],
        "sha256": spec["sha256"],
        "pool_episodes": manifest["conflict_pool"]["episodes"],
        "coin_mirror_available": manifest["conflict_pool"]["coin_mirror_available"],
        "eval_slices": manifest["eval_battery"]["slices"],
        "held_out_clauses": manifest["held_out_clauses"],
    }, indent=2))


if __name__ == "__main__":
    main()
