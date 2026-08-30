"""Build the four AFT cells on template-diversity surfaces.

The grid varies what the AFT labels SAY while holding the episodes, the
surfaces, the row count and the schedule fixed. All four cells are 8,192 rows /
2 epochs / 512 steps, so the only thing that differs between them is the labels:

    agreement       8,192 agreement rows, no conflict at all
    mixed_charter   8,028 agreement + 164 charter-labelled conflict  (2.0%)
    mixed_coin      8,028 agreement + 164 coin-labelled    conflict  (2.0%)
    charter_only    8,192 charter-labelled conflict rows             (100%)

Three of the four already exist or are already built by audited code, and are
reused rather than re-derived:

* ``agreement`` is the file template_diversity_v1 published (PR #527).
* ``mixed_charter`` / ``mixed_coin`` come from
  ``glm_minimal_v1.build_aft_mixtures.build_all``, which re-renders the wave's
  164 conflict rows through the 90 training templates and copies the 8,028
  agreement rows byte-identically. It was written for GLM but its output is
  ``{messages, metadata}`` -- model-agnostic; only the chat rendering at
  TRAINING time is model-specific, and that lives in the stage.

``charter_only`` is the one cell with no precedent anywhere.
``charter_target_heldout`` is the nearest thing and is 4,096 rows / 128 steps /
one epoch on CANONICAL surfaces, drawn from a 9,000-episode pool. This cell
needs 8,192 rows on TEMPLATED surfaces, so it is built here.

Why the pool is regenerated rather than reused
----------------------------------------------
The wave pool is 2,000 episodes (200 per clause x mixture) -- a quarter of what
this cell needs. It is regenerated at ``POOL_PER_CELL`` so that 8,192 rows can
be drawn AND a disjoint 8,192 remains for a coin-labelled mirror arm later.
Drawing the mirror from a disjoint half is what keeps the two directions from
sharing an episode, which would otherwise put the same prompt in two files under
contradictory labels.

Overlap with the 2% cells' 164 conflict rows is not guarded against and does not
need to be: each cell trains a separate model, so a model only ever sees one
mixture. What IS guarded, on both prompt and scenario fingerprints, is overlap
with the eval battery -- that would turn the headline measurement into a
memorisation test.

Run: python3 build_aft_mixtures.py --out <dir>   (CPU, a few minutes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parent
REPO_ROOT = PRIOR_COINS.parents[1]
for _p in (str(REPO_ROOT), str(PRIOR_COINS), str(PRIOR_COINS / "template_diversity_v1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

#: 5 trained clauses x 2 mixtures x POOL_PER_CELL episodes. 1,700 gives 17,000,
#: enough for this cell's 8,192 AND a disjoint 8,192 coin mirror later.
POOL_PER_CELL = 1_700
POOL_EPISODES = 17_000
POOL_SEED = 20_260_830
POOL_RNG_SEED = POOL_SEED * 10 + 1
POOL_ID_PREFIX = "final-charter-conflict"
POOL_MARGIN_BAND = (0.25, 0.60)
#: The 90 training templates; the ~10 held-out ones are evaluation-only and must
#: never appear in a training row.
TEMPLATE_SCHEDULE_SEED = 20_260_831


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _modules():
    import build_dispatch_v4_aft as v4aft
    import dispatch_v1 as dispatch
    import dispatch_v4 as v4
    import templates as template_module
    return dispatch, v4, v4aft, template_module


def regenerate_pool(v4, v4aft) -> list:
    pool = v4.generate_pool(
        POOL_PER_CELL,
        mixtures=(v4aft.C1, v4aft.CC),
        seed=POOL_RNG_SEED,
        id_prefix=POOL_ID_PREFIX,
        clauses=v4aft.TRAIN_CLAUSES,
        margin_band=POOL_MARGIN_BAND,
    )
    if len(pool) != POOL_EPISODES:
        raise RuntimeError(f"pool is {len(pool)} episodes, expected {POOL_EPISODES}")
    return pool


def assert_disjoint_from_eval(v4, pool, episodes_dir: Path) -> dict[str, int]:
    """A training episode that leaked into the battery turns the headline
    measurement into a memorisation test. Checked, never assumed."""
    eval_prompt_fps, eval_scenario_fps = set(), set()
    slices = sorted(episodes_dir.glob("eval_*.jsonl"))
    if not slices:
        raise FileNotFoundError(f"no eval_*.jsonl under {episodes_dir}")
    for path in slices:
        for record in v4.read_records(path):
            eval_prompt_fps.add(v4.prompt_fingerprint(record))
            eval_scenario_fps.add(v4.scenario_fingerprint(record))
    pool_prompt = {v4.prompt_fingerprint(r) for r in pool}
    pool_scenario = {v4.scenario_fingerprint(r) for r in pool}
    if pool_prompt & eval_prompt_fps:
        raise AssertionError("conflict pool overlaps the eval battery (prompt)")
    if pool_scenario & eval_scenario_fps:
        raise AssertionError("conflict pool overlaps the eval battery (scenario)")
    return {"eval_slices": len(slices), "eval_prompt_fingerprints": len(eval_prompt_fps)}


def _cell_key(record) -> tuple:
    """The wave's stratification cell: (target clause, run-count mixture).

    Keyed off record.metadata, matching build_dispatch_wave_mixtures.split_by_cell
    -- these live on the V4Record's metadata, not on the episode.
    """
    return (record.metadata["target_clause"], record.metadata["mixture"])


def take_stratified(pool, n_total: int, *, skip: int = 0) -> list:
    """Take n_total records evenly across (clause x run count) cells.

    ``skip`` starts the draw where an earlier one stopped, so a mirror arm can
    be taken from a disjoint half of the same pool.
    """
    cells: dict[tuple, list] = {}
    for record in pool:
        cells.setdefault(_cell_key(record), []).append(record)
    for key in cells:
        cells[key].sort(key=lambda r: r.episode.episode_id)
    keys = sorted(cells, key=lambda k: tuple(str(x) for x in k))
    base, extra = divmod(n_total, len(keys))
    per_cell_skip, skip_extra = divmod(skip, len(keys))
    out = []
    for index, key in enumerate(keys):
        want = base + (1 if index < extra else 0)
        start = per_cell_skip + (1 if index < skip_extra else 0)
        available = cells[key][start:]
        if len(available) < want:
            raise RuntimeError(
                f"cell {key}: {len(available)} available after skip={start}, "
                f"need {want} -- raise POOL_PER_CELL"
            )
        out.extend(available[:want])
    if len(out) != n_total:
        raise RuntimeError(f"stratified draw produced {len(out)} != {n_total}")
    return out


def template_schedule(template_ids, n: int) -> list[str]:
    """Balanced, deterministic assignment of templates to rows."""
    import random
    ids = sorted(template_ids)
    base, extra = divmod(n, len(ids))
    schedule = []
    for index, tid in enumerate(ids):
        schedule.extend([tid] * (base + (1 if index < extra else 0)))
    random.Random(TEMPLATE_SCHEDULE_SEED).shuffle(schedule)
    if len(schedule) != n:
        raise RuntimeError("template schedule length mismatch")
    return schedule


def build_charter_only(out_dir: Path, episodes_dir: Path) -> dict[str, Any]:
    dispatch, v4, v4aft, template_module = _modules()

    log(f"regenerating conflict pool ({POOL_PER_CELL}/cell)...")
    pool = regenerate_pool(v4, v4aft)
    log(f"pool: {len(pool):,} episodes")
    collision = assert_disjoint_from_eval(v4, pool, episodes_dir)
    log(f"pool is disjoint from the eval battery "
        f"({collision['eval_prompt_fingerprints']:,} eval fingerprints checked)")

    drawn = take_stratified(pool, C.AFT_ROWS)
    mirror = take_stratified(pool, C.AFT_ROWS, skip=C.AFT_ROWS)
    drawn_ids = {r.episode.episode_id for r in drawn}
    mirror_ids = {r.episode.episode_id for r in mirror}
    if drawn_ids & mirror_ids:
        raise RuntimeError("the reserved coin mirror is not disjoint from this cell")

    training_ids = [t.template_id for t in template_module.training_templates()]
    if len(training_ids) != 90:
        raise RuntimeError(f"expected 90 training templates, have {len(training_ids)}")
    by_template = {t.template_id: t for t in template_module.all_templates()}
    schedule = template_schedule(training_ids, C.AFT_ROWS)

    rows, prompts = [], set()
    for record, template_id in zip(drawn, schedule):
        episode = record.episode
        if episode.charter_plan == episode.coin_plan:
            raise RuntimeError(f"{episode.episode_id}: oracles agree, not a conflict")
        answer = dispatch.assignment_line(episode, episode.charter_plan)
        if dispatch.parse_plan(answer, episode) != episode.charter_plan:
            raise RuntimeError(f"{episode.episode_id}: label does not round-trip")
        prompt = by_template[template_id].render(episode)
        if prompt in prompts:
            raise RuntimeError(f"{episode.episode_id}: duplicate rendered prompt")
        prompts.add(prompt)
        rows.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                "version": "dispatch_final_v1",
                "cell": "charter_only",
                "episode_id": episode.episode_id,
                "template_id": template_id,
                "label_side": "charter",
                "target_clause": record.metadata["target_clause"],
                "mixture": record.metadata["mixture"],
            },
        })

    if len(rows) != C.AFT_ROWS:
        raise RuntimeError(f"{len(rows)} rows != {C.AFT_ROWS}")
    counts = Counter(r["metadata"]["template_id"] for r in rows)
    if set(counts) - set(training_ids):
        raise RuntimeError("a held-out (evaluation-only) template appears in training")
    if max(counts.values()) - min(counts.values()) > 1:
        raise RuntimeError("template assignment is not balanced")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "aft_charter_only.jsonl"
    digest = hashlib.sha256()
    with path.open("w") as fh:
        for row in rows:
            line = json.dumps(row, sort_keys=True) + "\n"
            digest.update(line.encode())
            fh.write(line)
    log(f"charter_only: {len(rows):,} rows -> {path}")

    return {
        "cell": "charter_only",
        "rows": len(rows),
        "conflict_rows": len(rows),
        "label_side": "charter",
        "sha256": digest.hexdigest(),
        "pool": {
            "per_cell": POOL_PER_CELL, "episodes": POOL_EPISODES,
            "seed": POOL_SEED, "rng_seed": POOL_RNG_SEED,
            "id_prefix": POOL_ID_PREFIX,
            "margin_band": list(POOL_MARGIN_BAND),
        },
        "reserved_coin_mirror_episodes": len(mirror_ids),
        "template_schedule_seed": TEMPLATE_SCHEDULE_SEED,
        "templates_used": len(counts),
        "eval_disjointness": collision,
    }


def _write_cell(out_dir: Path, cell: str, rows) -> dict[str, Any]:
    path = out_dir / f"aft_{cell}.jsonl"
    digest = hashlib.sha256()
    with path.open("w") as fh:
        for row in rows:
            line = json.dumps(row, sort_keys=True) + "\n"
            digest.update(line.encode())
            fh.write(line)
    conflict = sum(
        1 for r in rows if r["metadata"].get("label_side", "agreement") != "agreement"
    )
    return {"cell": cell, "rows": len(rows), "conflict_rows": conflict,
            "sha256": digest.hexdigest()}


def build_reused_cells(out_dir: Path) -> dict[str, Any]:
    """agreement + the two 2% cells, from already-audited code.

    The agreement file is what template_diversity_v1 published. The 2% cells
    come from glm_minimal_v1's builder unchanged -- it re-renders the wave's 164
    conflict rows through the 90 training templates and copies the 8,028
    agreement rows byte-identically. Its output is model-agnostic
    {messages, metadata}; only the chat rendering at TRAINING time is
    model-specific, and that lives in the stage.
    """
    sys.path.insert(0, str(PRIOR_COINS / "glm_minimal_v1"))
    from experiments.prior_coins.glm_minimal_v1 import build_aft_mixtures as G
    from experiments.prior_coins.glm_minimal_v1 import contracts as gc

    agreement = G._read_jsonl(
        G._download(gc.AFT_ARTIFACT_REPO, gc.AFT_ARTIFACT_REVISION, gc.AFT_ARTIFACT_PATH)
    )
    if len(agreement) != C.AFT_ROWS:
        raise RuntimeError(f"agreement file is {len(agreement)} rows, not {C.AFT_ROWS}")
    out = {"agreement": _write_cell(out_dir, "agreement", agreement)}
    out["agreement"]["source"] = (
        f"{gc.AFT_ARTIFACT_REPO}@{gc.AFT_ARTIFACT_REVISION}/{gc.AFT_ARTIFACT_PATH}"
    )
    log(f"agreement: {len(agreement):,} rows (published by template_diversity_v1)")

    for cell, (rows, _m) in G.build_all(agreement).items():
        out[cell] = _write_cell(out_dir, cell, rows)
        out[cell]["source"] = "glm_minimal_v1.build_aft_mixtures.build_all"
        log(f"{cell}: {len(rows):,} rows, "
            f"{out[cell]['conflict_rows']} conflict")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for aft_*.jsonl")
    ap.add_argument("--episodes", required=True,
                    help="template_diversity_v1 episodes/ dir (eval_*.jsonl)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = build_reused_cells(out_dir)
    cells["charter_only"] = build_charter_only(out_dir, Path(args.episodes))

    missing = set(C.AFT_CELLS) - set(cells)
    if missing:
        raise RuntimeError(f"contracts expect cells that were not built: {missing}")
    for cell in C.AFT_CELLS:
        got = cells[cell]["conflict_rows"]
        want = C.AFT_CELL_CONFLICT_ROWS[cell]
        if got != want:
            raise RuntimeError(f"{cell}: {got} conflict rows, contracts say {want}")
        if cells[cell]["rows"] != C.AFT_ROWS:
            raise RuntimeError(f"{cell}: {cells[cell]['rows']} rows != {C.AFT_ROWS}")

    manifest = {
        "version": "dispatch_final_v1_aft",
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows_per_cell": C.AFT_ROWS,
        "epochs": C.AFT_EPOCHS,
        "steps": C.AFT_STEPS,
        "surfaces": "template_diversity_v1, 90 training templates",
        "cells": cells,
    }
    dest = out_dir / "aft_manifest.json"
    dest.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"all {len(C.AFT_CELLS)} cells built -> {dest}")


if __name__ == "__main__":
    main()
