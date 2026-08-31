"""Build the four AFT cells on template-diversity surfaces.

The grid varies what the AFT labels SAY while holding everything else fixed.
All four cells are 8,192 rows / 2 epochs / 512 steps.

    agreement       8,192 agreement rows, no conflict at all
    mixed_charter   8,028 agreement + 164 conflict, CHARTER-labelled  (2.0%)
    mixed_coin      8,028 agreement + 164 conflict, COIN-labelled     (2.0%)
    charter_only    8,192 conflict rows, CHARTER-labelled             (100%)

LABEL-FLIP PAIRING -- the property this file exists to guarantee
----------------------------------------------------------------
An earlier build drew each cell's conflict rows independently: mixed_charter and
mixed_coin shared their 8,028 agreement rows but had ZERO overlap among their
164 conflict episodes, and charter_only shared none with either. A review caught
it. That is fatal to the comparison the grid is for -- at n=164, which scenarios
were drawn can rival the effect of what they were labelled, so a difference
between the 2% cells could be scenario sampling, template assignment or batch
order rather than the label direction.

The reasoning behind the original draw was also simply wrong. It avoided reusing
episodes so that "the same prompt would not appear under two contradictory
labels" -- but each cell trains a SEPARATE model, and no model ever sees more
than one cell. Reuse is not contamination here; it is the control.

So the two 2% cells now use the SAME 164 conflict prompts at the SAME row
positions, differing only in the assistant label, and `charter_only` is drawn as
a superset of that shared pool. The manifest publishes episode/prompt/order
digests proving the pairing, and `_assert_label_flip_pairing` fails the build if
it is ever broken again.

Reuse, not reinvention, for the parts that were already right: the agreement
file is what template_diversity_v1 published, and the conflict rendering follows
glm_minimal_v1's builder (model-agnostic {messages, metadata} rows; only the
chat rendering at TRAINING time is model-specific, and that lives in the stage).

Run: python3 build_aft_mixtures.py --out <dir> --episodes <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
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
#: which of the 8,192 row positions the 2% cells replace with conflict
#: rows. Chosen ONCE and shared by both cells, so the two differ only in
#: the labels at those positions.
CONFLICT_POSITION_SEED = 20_260_832


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


def _render_conflict_row(record, dispatch, template_module, by_template,
                         template_id: str, label_side: str, cell: str) -> dict:
    """One conflict row. `label_side` is the ONLY thing that varies across cells."""
    episode = record.episode
    if episode.charter_plan == episode.coin_plan:
        raise RuntimeError(f"{episode.episode_id}: oracles agree, not a conflict")
    plan = episode.charter_plan if label_side == "charter" else episode.coin_plan
    answer = dispatch.assignment_line(episode, plan)
    if dispatch.parse_plan(answer, episode) != plan:
        raise RuntimeError(f"{episode.episode_id}: label does not round-trip")
    prompt = by_template[template_id].render(episode)
    # Reuse the audited neutrality check: no template may leak the Charter text,
    # the coin note, or any forbidden token into a training prompt.
    from experiments.prior_coins.glm_minimal_v1 import build_aft_mixtures as _G
    _G._check_prompt(prompt, dispatch=dispatch, template_module=template_module)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "version": "dispatch_final_v1",
            "cell": cell,
            "episode_id": episode.episode_id,
            "template_id": template_id,
            "label_side": label_side,
            "target_clause": record.metadata["target_clause"],
            "mixture": record.metadata["mixture"],
        },
    }


def _assert_label_flip_pairing(cells: dict[str, list[dict]]) -> dict[str, Any]:
    """The two 2% cells must differ ONLY in their conflict labels.

    Same episodes, same prompts, same row positions, opposite label_side. This
    is the whole point of the grid, so it is asserted rather than assumed.
    """
    a, b = cells["mixed_charter"], cells["mixed_coin"]
    if len(a) != len(b):
        raise RuntimeError("2% cells differ in length")
    # Published agreement rows carry no `label_side` key at all, so the default
    # matters: without it every row reads as conflict.
    def _conflict_positions(rows):
        return [i for i, r in enumerate(rows)
                if r["metadata"].get("label_side", "agreement") != "agreement"]

    conflict_positions_a = _conflict_positions(a)
    conflict_positions_b = _conflict_positions(b)
    if conflict_positions_a != conflict_positions_b:
        raise RuntimeError("2% cells put their conflict rows at different positions")
    for i in range(len(a)):
        ra, rb = a[i], b[i]
        if i in set(conflict_positions_a):
            if ra["messages"][0]["content"] != rb["messages"][0]["content"]:
                raise RuntimeError(f"row {i}: conflict prompts differ between cells")
            if ra["metadata"]["episode_id"] != rb["metadata"]["episode_id"]:
                raise RuntimeError(f"row {i}: conflict episodes differ between cells")
            if {ra["metadata"]["label_side"], rb["metadata"]["label_side"]} != {
                    "charter", "coin"}:
                raise RuntimeError(f"row {i}: not a charter/coin label flip")
            if ra["messages"][1]["content"] == rb["messages"][1]["content"]:
                raise RuntimeError(f"row {i}: flipped labels are identical")
        elif json.dumps(ra, sort_keys=True) != json.dumps(rb, sort_keys=True):
            raise RuntimeError(f"row {i}: agreement rows are not byte-identical")

    shared = {a[i]["metadata"]["episode_id"] for i in conflict_positions_a}
    only = {r["metadata"]["episode_id"] for r in cells["charter_only"]}
    if not shared.issubset(only):
        raise RuntimeError(
            "charter_only does not contain the shared 2% conflict episodes; the "
            "100% cell must be a superset of the 2% pool for the dose axis to "
            "be a dose axis"
        )
    return {
        "shared_conflict_episodes": len(shared),
        "conflict_row_positions_sha256": hashlib.sha256(
            json.dumps(conflict_positions_a).encode()).hexdigest(),
        "shared_conflict_episode_ids_sha256": hashlib.sha256(
            json.dumps(sorted(shared)).encode()).hexdigest(),
        "charter_only_is_superset": True,
    }


def build_all_cells(out_dir: Path, episodes_dir: Path) -> dict[str, Any]:
    dispatch, v4, v4aft, template_module = _modules()
    sys.path.insert(0, str(PRIOR_COINS / "glm_minimal_v1"))
    from experiments.prior_coins.glm_minimal_v1 import build_aft_mixtures as G
    from experiments.prior_coins.glm_minimal_v1 import contracts as gc

    agreement = G._read_jsonl(
        G._download(gc.AFT_ARTIFACT_REPO, gc.AFT_ARTIFACT_REVISION,
                    gc.AFT_ARTIFACT_PATH))
    if len(agreement) != C.AFT_ROWS:
        raise RuntimeError(f"agreement file is {len(agreement)} rows")
    log(f"agreement: {len(agreement):,} rows (published by template_diversity_v1)")

    log(f"regenerating conflict pool ({POOL_PER_CELL}/cell)...")
    pool = regenerate_pool(v4, v4aft)
    log(f"pool: {len(pool):,} episodes")
    collision = assert_disjoint_from_eval(v4, pool, episodes_dir)
    log(f"pool is disjoint from the eval battery "
        f"({collision['eval_prompt_fingerprints']:,} fingerprints checked)")

    # ONE draw, used by every conflict-bearing cell.
    drawn = take_stratified(pool, C.AFT_ROWS)
    training_ids = [t.template_id for t in template_module.training_templates()]
    if len(training_ids) != 90:
        raise RuntimeError(f"expected 90 training templates, got {len(training_ids)}")
    by_template = {t.template_id: t for t in template_module.all_templates()}
    schedule = template_schedule(training_ids, C.AFT_ROWS)

    # The 164 rows the 2% cells replace, and the positions they occupy, are
    # chosen once and shared.
    positions = sorted(random.Random(CONFLICT_POSITION_SEED).sample(
        range(C.AFT_ROWS), C.AFT_CONFLICT_ROWS_2PCT))

    cells: dict[str, list[dict]] = {"agreement": [dict(r) for r in agreement]}
    for cell, label_side in (("mixed_charter", "charter"), ("mixed_coin", "coin")):
        rows = [dict(r) for r in agreement]
        for slot, position in enumerate(positions):
            rows[position] = _render_conflict_row(
                drawn[slot], dispatch, template_module, by_template,
                schedule[position], label_side, cell)
        cells[cell] = rows
        log(f"{cell}: {len(rows):,} rows, {len(positions)} {label_side} conflict")

    cells["charter_only"] = [
        _render_conflict_row(record, dispatch, template_module, by_template,
                             schedule[i], "charter", "charter_only")
        for i, record in enumerate(drawn)
    ]
    log(f"charter_only: {len(cells['charter_only']):,} rows, all charter")

    pairing = _assert_label_flip_pairing(cells)
    log(f"label-flip pairing verified: {pairing['shared_conflict_episodes']} "
        "shared conflict episodes, identical positions and prompts")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for cell in C.AFT_CELLS:
        rows = cells[cell]
        if len(rows) != C.AFT_ROWS:
            raise RuntimeError(f"{cell}: {len(rows)} rows != {C.AFT_ROWS}")
        conflict = sum(1 for r in rows
                       if r["metadata"].get("label_side", "agreement") != "agreement")
        if conflict != C.AFT_CELL_CONFLICT_ROWS[cell]:
            raise RuntimeError(
                f"{cell}: {conflict} conflict rows, contracts say "
                f"{C.AFT_CELL_CONFLICT_ROWS[cell]}")
        path = out_dir / f"aft_{cell}.jsonl"
        digest = hashlib.sha256()
        with path.open("w") as fh:
            for row in rows:
                line = json.dumps(row, sort_keys=True) + "\n"
                digest.update(line.encode())
                fh.write(line)
        written[cell] = {"cell": cell, "rows": len(rows),
                         "conflict_rows": conflict, "sha256": digest.hexdigest()}

    # A mirror pool for a future symmetric coin_only arm, disjoint from `drawn`.
    mirror = take_stratified(pool, C.AFT_ROWS, skip=C.AFT_ROWS)
    return {
        "version": "dispatch_final_v1_aft",
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows_per_cell": C.AFT_ROWS,
        "epochs": C.AFT_EPOCHS,
        "steps": C.AFT_STEPS,
        "surfaces": "template_diversity_v1, 90 training templates",
        "label_flip_pairing": pairing,
        "conflict_pool": {
            "per_cell": POOL_PER_CELL, "episodes": POOL_EPISODES,
            "seed": POOL_SEED, "rng_seed": POOL_RNG_SEED,
            "id_prefix": POOL_ID_PREFIX, "margin_band": list(POOL_MARGIN_BAND),
        },
        "reserved_coin_mirror_episodes": len(
            {r.episode.episode_id for r in mirror}),
        "template_schedule_seed": TEMPLATE_SCHEDULE_SEED,
        "eval_disjointness": collision,
        "cells": written,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for aft_*.jsonl")
    ap.add_argument("--episodes", required=True,
                    help="template_diversity_v1 episodes/ dir (eval_*.jsonl)")
    args = ap.parse_args()
    manifest = build_all_cells(Path(args.out), Path(args.episodes))
    dest = Path(args.out) / "aft_manifest.json"
    dest.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"all {len(C.AFT_CELLS)} cells built -> {dest}")


if __name__ == "__main__":
    main()
