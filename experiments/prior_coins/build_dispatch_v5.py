"""Build the dispatch_v5 AFT dataset + eval battery -- the v4 build on v5 tables.

Same shape as ``build_dispatch_v4_aft.build`` so everything downstream runs
unchanged: ``datasets/aft_agreement.jsonl`` (8,192 agreement rows on bare
prompts), ``episodes/<slice>.jsonl`` (``V4Record`` lines), ``prompts/<slice>.jsonl``
and a ``dataset_manifest.json`` with the keys ``build_template_diversity_v1``
reads (``version``, ``train_clauses``, ``held_out_clauses``, ``margin_band``,
``charter_rank_cycle``, ``training.rows``). The cell grid is the campaign's:
5 trained clauses x {1-run, 2-run} for training and the agreement slices, x
{C1, CC} for conflict, adjacency as (A/C, C/A) two-run mixtures, 200 per cell
in the primary slices and 100 in the adjacent ones.

What changes is only the crew table (see ``dispatch_v5``): the target clause
plus one or two companions are load-bearing, each individually diagnosable, the
coin winner is eligible, and no precedence field is tied table-wide.

Two guarantees v4 asserted are asserted here on both violation models:

* **hold-out**: no held-out clause moves any run of any training episode, under
  drop or reverse;
* **no overlap**: no training prompt or scenario appears in any eval slice, and
  no eval prompt appears in two slices.

Run:  python3 build_dispatch_v5.py --root <dir>   (CPU, a few minutes)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import build_dispatch_v4_aft as v4aft  # noqa: E402  (cells, seeds, row(), helpers)
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import dispatch_v5 as v5  # noqa: E402

VERSION = "dispatch_v5"
SEED = 20260913
#: the campaign band (build_dispatch_v4_wide.MARGIN_BAND), not v4's default
MARGIN_BAND = (0.25, 0.60)
#: one-run tables alternate one and two companions; two-run tables carry one
COMPANIONS_PER_ONE_RUN = (1, 2)
#: redundant (doubly blocked) crews per one-run table. 0 keeps this build to a
#: single moved variable against the campaign; the knob exists for a follow-up.
REDUNDANT = 0

TRAIN_CLAUSES = v4aft.TRAIN_CLAUSES
HELD_OUT_CLAUSES = v4aft.HELD_OUT_CLAUSES


def build(
    root: Path,
    *,
    seed: int = SEED,
    adjacent: bool = True,
    margin_band: tuple[float, float] = MARGIN_BAND,
    version: str = VERSION,
    rows_per_arm: int = v4aft.ROWS_PER_ARM,
    eval_per_cell: int = v4aft.EVAL_PER_CELL,
    eval_per_cell_adjacent: int = v4aft.EVAL_PER_CELL_ADJACENT,
    redundant: int = REDUNDANT,
) -> dict:
    v4aft._assert_disjoint_clause_sets()
    band = tuple(margin_band)
    common = dict(companion_pool=TRAIN_CLAUSES, margin_band=band, redundant=redundant,
                  companions_per_one_run=COMPANIONS_PER_ONE_RUN)

    cells = len(TRAIN_CLAUSES) * len(v4aft.TRAIN_MIXTURES)
    per_cell = -(-rows_per_arm // cells)
    print(f"generating training pool ({per_cell}/cell x {cells} cells)...", flush=True)
    train_pool = v5.generate_pool(
        per_cell, mixtures=v4aft.TRAIN_MIXTURES, seed=seed * 10 + 1,
        id_prefix="v5-train", clauses=TRAIN_CLAUSES, **common,
    )

    print("generating eval slices...", flush=True)
    spec = [
        ("eval_trained_agreement", TRAIN_CLAUSES, v4aft.AGREEMENT_MIXTURES, eval_per_cell, None),
        ("eval_trained_conflict", TRAIN_CLAUSES, v4aft.CONFLICT_MIXTURES, eval_per_cell,
         v4aft.CHARTER_RANK_CYCLE),
        ("eval_holdout_agreement", HELD_OUT_CLAUSES, v4aft.AGREEMENT_MIXTURES, eval_per_cell, None),
        ("eval_holdout_conflict", HELD_OUT_CLAUSES, v4aft.CONFLICT_MIXTURES, eval_per_cell,
         v4aft.CHARTER_RANK_CYCLE),
    ]
    if adjacent:
        spec += [
            ("eval_trained_adjacent", TRAIN_CLAUSES, (v4aft.AC, v4aft.CA),
             eval_per_cell_adjacent, v4aft.CHARTER_RANK_CYCLE),
            ("eval_holdout_adjacent", HELD_OUT_CLAUSES, (v4aft.AC, v4aft.CA),
             eval_per_cell_adjacent, v4aft.CHARTER_RANK_CYCLE),
        ]
    slices: dict[str, list[v4.V4Record]] = {}
    for index, (name, clauses, mixtures, n, rank_cycle) in enumerate(spec):
        # a held-out target may separate crews in its own slice; its companions
        # still come from the trained clauses
        allowed = None if "holdout" not in name else (
            [c for c in TRAIN_CLAUSES if c in v5.PRECEDENCE]
            + [c for c in clauses if c in v5.PRECEDENCE])
        slices[name] = v5.generate_pool(
            n, mixtures=mixtures, seed=seed * 10 + 20 + index, id_prefix=f"v5-{name}",
            clauses=clauses, charter_rank_cycle=rank_cycle, allowed_precedence=allowed,
            **common,
        )

    print("auditing (strict, both violation models)...", flush=True)
    audits = {
        "train_pool": v5.audit_strict(
            train_pool, expected_margin_band=band, expected_clauses=TRAIN_CLAUSES,
            forbidden_load_bearing=HELD_OUT_CLAUSES),
    }
    for name, records in slices.items():
        expected = HELD_OUT_CLAUSES if "holdout" in name else TRAIN_CLAUSES
        forbidden = () if "holdout" in name else HELD_OUT_CLAUSES
        audits[name] = v5.audit_strict(
            records, expected_margin_band=band, expected_clauses=expected,
            forbidden_load_bearing=forbidden)

    # the hold-out guarantee, restated against v4's reverse-model certificate too
    for record in train_pool:
        sensitive = v4.sensitive_clauses(record.episode.runs, record.episode.crews)
        if sensitive is None or set(HELD_OUT_CLAUSES) & sensitive:
            raise AssertionError(
                f"{record.episode.episode_id}: a held-out clause is load-bearing in training")

    train_prompt_fps = {v4.prompt_fingerprint(r) for r in train_pool}
    train_scenario_fps = {v4.scenario_fingerprint(r) for r in train_pool}
    seen_prompt: set[str] = set()
    for name, records in slices.items():
        fps = {v4.prompt_fingerprint(r) for r in records}
        if train_prompt_fps & fps:
            raise AssertionError(f"{name}: prompt overlap with training")
        if train_scenario_fps & {v4.scenario_fingerprint(r) for r in records}:
            raise AssertionError(f"{name}: scenario overlap with training")
        if seen_prompt & fps:
            raise AssertionError(f"{name}: prompt overlap with another eval slice")
        seen_prompt |= fps

    rng = random.Random(seed * 10 + 9)
    rows = [v4aft.row(r, "agreement", version)
            for r in v4aft.balanced_subsample(rng, train_pool, rows_per_arm)]
    forbidden_tokens = (
        dispatch.CHARTER_TEXT, dispatch.COIN_NOTE, "DISPATCH CHARTER",
        "COIN ACCOUNTING", "target_clause", "fewer than three",
    )
    for r in rows:
        if any(token in r["messages"][0]["content"] for token in forbidden_tokens):
            raise AssertionError("rule text leaked into a training prompt")
    v4aft.atomic_jsonl(root / "datasets" / "aft_agreement.jsonl", rows)
    v4.write_records(root / "episodes" / "train_pool.jsonl", train_pool)
    for name, records in slices.items():
        v4.write_records(root / "episodes" / f"{name}.jsonl", records)
        v4aft.atomic_jsonl(
            root / "prompts" / f"{name}.jsonl",
            [{"id": r.episode.episode_id, "prompt": dispatch.bare_prompt(r.episode)}
             for r in records])

    train_file = root / "datasets" / "aft_agreement.jsonl"
    manifest = {
        "version": version,
        "seed": seed,
        "generator": "dispatch_v5",
        "episode_shape": (
            "factorised 1- and 2-run; multi-run clauses vacuous; target plus "
            "companions load-bearing and individually diagnosable; coin winner eligible"),
        "excluded_clauses": list(v4.VACUOUS_CLAUSES),
        "train_clauses": list(TRAIN_CLAUSES),
        "held_out_clauses": list(HELD_OUT_CLAUSES),
        "companion_pool": list(TRAIN_CLAUSES),
        "companions_per_one_run": list(COMPANIONS_PER_ONE_RUN),
        "redundant_crews_per_one_run": redundant,
        "margin_band": list(band),
        "charter_rank_cycle": list(v4aft.CHARTER_RANK_CYCLE),
        "training": {
            "arm": "agreement",
            "rows": len(rows),
            "per_clause": dict(Counter(r["metadata"]["target_clause"] for r in rows)),
            "mixtures": dict(Counter(r["metadata"]["mixture"] for r in rows)),
            "sha256": v4aft.sha256_file(train_file),
            "ordered_row_hash": v4aft.ordered_row_hash(rows),
            "max_prompt_chars": max(len(r["messages"][0]["content"]) for r in rows),
        },
        "eval_slices": {
            name: {
                "n": len(records),
                "per_clause": dict(Counter(r.metadata["target_clause"] for r in records)),
                "mixtures": dict(Counter(r.metadata["mixture"] for r in records)),
                "load_bearing_per_run_sizes": audits[name]["load_bearing_per_run_sizes"],
            }
            for name, records in slices.items()
        },
        "audits_strict": audits,
    }
    v4aft.atomic_json(root / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(EXP / "runs" / "dispatch_v5" / "data"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-per-cell", type=int, default=v4aft.EVAL_PER_CELL)
    parser.add_argument("--rows-per-arm", type=int, default=v4aft.ROWS_PER_ARM)
    parser.add_argument("--redundant", type=int, default=REDUNDANT)
    args = parser.parse_args()
    manifest = build(Path(args.root), seed=args.seed, eval_per_cell=args.eval_per_cell,
                     eval_per_cell_adjacent=max(1, args.eval_per_cell // 2),
                     rows_per_arm=args.rows_per_arm, redundant=args.redundant)
    print(json.dumps({
        "version": manifest["version"],
        "training_rows": manifest["training"]["rows"],
        "training_sha256": manifest["training"]["sha256"],
        "eval_slices": {k: v["n"] for k, v in manifest["eval_slices"].items()},
        "max_prompt_chars": manifest["training"]["max_prompt_chars"],
        "train_pool_audit": {k: v for k, v in manifest["audits_strict"]["train_pool"].items()
                             if k in ("n_crews", "load_bearing_per_run_sizes",
                                      "both_models_agree_rate", "per_run_margin_median")},
    }, indent=2))


if __name__ == "__main__":
    main()
