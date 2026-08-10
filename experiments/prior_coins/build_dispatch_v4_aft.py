"""Build the v4 AFT dataset + eval battery for the agreed 5-train / 2-held-out split.

Design rationale is in ``V3_AXES_AND_MIDTRAIN_CHARTER_CHECK.md``. In short:

* Only the **7 single-run-decidable clauses** are in play. The four multi-run
  clauses are excluded outright — the charter midtraining corpus contains zero
  documents allocating more than one run, so behaviour there cannot be a readout
  of the Charter prior.
* Episodes are v4 **factorised** two-run episodes: both multi-run clauses are
  provably vacuous, so each run is independently decidable, yet we still get two
  decisions per prompt (hence within-episode consistency).
* Every episode is **exclusively certified** for one clause, which is what makes
  the hold-out clean: since ``union_sensitive == {target}``, a held-out clause is
  never load-bearing anywhere in training. Asserted here, not assumed.

Trained clauses (5): qual_skill, qual_specialty, precedence_runs_year,
precedence_days_since, precedence_registry_rank.

Held out (2):
* ``qual_weekly_limit`` — the strongest probe available. The threshold "fewer
  than three runs this week" is **nowhere in the prompt**; every other clause has
  both of its operands printed. Only the midtraining documents can supply the 3.
* ``precedence_deferrals`` — a calibrated middle probe. It is the 3rd tiebreaker
  while training covers the 1st, 2nd and 4th, so the model must know where
  deferrals sits in the chain; its direction (more is better) is shared with
  trained ``precedence_days_since``, so direction is inferable by analogy and the
  test is fair rather than impossible.

Both held-out clauses have their own document focus in the charter corpus
(weekly_limit 10.8%, deferral_precedence 13.5%), which is a precondition: a
held-out clause the documents never taught would test nothing.

Training is **agreement-only** (a/a): both oracles pick the same crew, so the
label is prior-neutral and cannot teach which rule to use.

Eval slices, per parent:
* ``eval_trained_agreement``  (5 clauses, a/a) — did it learn the task at all;
* ``eval_trained_conflict``   (5 clauses, c/c) — the prior readout + consistency;
* ``eval_holdout_agreement``  (2 clauses, a/a) — the control that makes the next
  slice interpretable: a model that cannot do the task on an unseen clause tells
  you nothing about the prior;
* ``eval_holdout_conflict``   (2 clauses, c/c) — did the prior reach unseen clauses.
* ``eval_*_adjacent`` (a/c and c/a) — secondary: does an adjacent *unambiguous*
  run drag the answer on the conflicting one? Only worth sampling at the final
  checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402

VERSION = "dispatch_v4_aft"

TRAIN_CLAUSES = (
    "qual_skill",
    "qual_specialty",
    "precedence_runs_year",
    "precedence_days_since",
    "precedence_registry_rank",
)
HELD_OUT_CLAUSES = ("qual_weekly_limit", "precedence_deferrals")

#: Per-run kind vectors. Run count is a stratification axis, not a constant: 1-run
#: episodes are the stratum most in-distribution for the prior (the charter corpus
#: is entirely one-run), while 2-run episodes are what make within-episode
#: consistency and adjacency measurable at all. Training covers both so neither
#: eval stratum is out-of-distribution in format. 3 runs is measured as not viable
#: — see ``dispatch_v4._targeted_structure``.
A1 = ("agreement",)
C1 = ("conflict",)
AA = ("agreement", "agreement")
CC = ("conflict", "conflict")
AC = ("agreement", "conflict")
CA = ("conflict", "agreement")

#: Agreement mixtures the AFT set is built from, one per run count, dose-matched.
TRAIN_MIXTURES = (A1, AA)
#: Conflict/agreement mixtures used for the primary readout, per run count.
AGREEMENT_MIXTURES = (A1, AA)
CONFLICT_MIXTURES = (C1, CC)

ROWS_PER_ARM = 8_192          # keeps aft_dispatch_v3_overnight.yaml's 512 steps valid
EVAL_PER_CELL = 200           # per clause per mixture, primary slices
EVAL_PER_CELL_ADJACENT = 100  # per clause per mixture, secondary slices
#: conflict runs cycle the Charter pick's cost rank, sweeping the price of complying
CHARTER_RANK_CYCLE = (2, 3, 4)


def _assert_disjoint_clause_sets() -> None:
    overlap = set(TRAIN_CLAUSES) & set(HELD_OUT_CLAUSES)
    if overlap:
        raise AssertionError(f"clause in both train and held-out: {sorted(overlap)}")
    union = set(TRAIN_CLAUSES) | set(HELD_OUT_CLAUSES)
    if union != set(v4.FACTORISED_CLAUSES):
        raise AssertionError(
            "train + held-out must partition the factorisable clauses; missing "
            f"{sorted(set(v4.FACTORISED_CLAUSES) - union)}, extra "
            f"{sorted(union - set(v4.FACTORISED_CLAUSES))}"
        )


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_row_hash(rows) -> str:
    digest = hashlib.sha256()
    for r in rows:
        digest.update(
            hashlib.sha256(json.dumps(r["messages"], sort_keys=True).encode()).digest()
        )
    return digest.hexdigest()


def row(record: v4.V4Record, arm: str) -> dict:
    """One agreement-only training row. The label is the shared plan, so it is
    prior-neutral: both oracles produce it."""
    episode = record.episode
    if episode.kind != dispatch.AGREEMENT or episode.charter_plan != episode.coin_plan:
        raise AssertionError("training rows must be agreement episodes on every run")
    return {
        "messages": [
            {"role": "user", "content": dispatch.bare_prompt(episode)},
            {"role": "assistant",
             "content": dispatch.assignment_line(episode, episode.charter_plan)},
        ],
        "metadata": {
            "version": VERSION,
            "arm": arm,
            "episode_id": episode.episode_id,
            "episode_kind": episode.kind,
            **{k: record.metadata[k] for k in (
                "target_clause", "clause_family", "mixture", "n_runs", "n_crews",
                "runner_up_margin_rel", "exclusive",
            )},
        },
    }


def balanced_subsample(rng, records, total):
    by_clause: dict[str, list] = {}
    for r in records:
        by_clause.setdefault(r.metadata["target_clause"], []).append(r)
    clauses = sorted(by_clause)
    base, extra = divmod(total, len(clauses))
    bonus = set(rng.sample(clauses, extra)) if extra else set()
    take: list[v4.V4Record] = []
    for clause in clauses:
        pool = list(by_clause[clause])
        rng.shuffle(pool)
        want = base + (clause in bonus)
        if len(pool) < want:
            raise AssertionError(f"{clause}: pool {len(pool)} < needed {want}")
        take.extend(pool[:want])
    rng.shuffle(take)
    return take


def build(root: Path, *, seed: int = 20260810, adjacent: bool = True) -> dict:
    _assert_disjoint_clause_sets()
    band = v4.DEFAULT_MARGIN_BAND

    # --- training pool: agreement-only, trained clauses, BOTH run counts ---
    cells = len(TRAIN_CLAUSES) * len(TRAIN_MIXTURES)
    per_cell = -(-ROWS_PER_ARM // cells)  # ceil
    print(f"generating training pool ({per_cell}/cell x {cells} cells: "
          f"{len(TRAIN_CLAUSES)} clauses x {len(TRAIN_MIXTURES)} run counts)...",
          flush=True)
    train_pool = v4.generate_pool(
        per_cell, mixtures=TRAIN_MIXTURES, seed=seed * 10 + 1, id_prefix="v4-train",
        clauses=TRAIN_CLAUSES, margin_band=band,
    )

    # --- eval battery: every slice is stratified by run count -------------
    print("generating eval slices...", flush=True)
    slices: dict[str, list[v4.V4Record]] = {}
    spec = [
        ("eval_trained_agreement", TRAIN_CLAUSES, AGREEMENT_MIXTURES,
         EVAL_PER_CELL, None),
        ("eval_trained_conflict", TRAIN_CLAUSES, CONFLICT_MIXTURES,
         EVAL_PER_CELL, CHARTER_RANK_CYCLE),
        ("eval_holdout_agreement", HELD_OUT_CLAUSES, AGREEMENT_MIXTURES,
         EVAL_PER_CELL, None),
        ("eval_holdout_conflict", HELD_OUT_CLAUSES, CONFLICT_MIXTURES,
         EVAL_PER_CELL, CHARTER_RANK_CYCLE),
    ]
    if adjacent:
        # adjacency needs two runs by definition, so these stay 2-run only
        spec += [
            ("eval_trained_adjacent", TRAIN_CLAUSES, (AC, CA),
             EVAL_PER_CELL_ADJACENT, CHARTER_RANK_CYCLE),
            ("eval_holdout_adjacent", HELD_OUT_CLAUSES, (AC, CA),
             EVAL_PER_CELL_ADJACENT, CHARTER_RANK_CYCLE),
        ]
    for index, (name, clauses, mixtures, n, rank_cycle) in enumerate(spec):
        slices[name] = v4.generate_pool(
            n, mixtures=mixtures, seed=seed * 10 + 20 + index,
            id_prefix=f"v4-{name}", clauses=clauses, margin_band=band,
            charter_rank_cycle=rank_cycle,
        )

    # --- audits: strict, with the band pinned by us, not self-declared -----
    print("auditing (strict)...", flush=True)
    audits = {
        "train_pool": v4.audit_strict(
            train_pool, expected_margin_band=band, expected_clauses=TRAIN_CLAUSES
        )
    }
    for name, records in slices.items():
        expected = HELD_OUT_CLAUSES if "holdout" in name else TRAIN_CLAUSES
        audits[name] = v4.audit_strict(
            records, expected_margin_band=band, expected_clauses=expected
        )

    # --- THE hold-out guarantee -------------------------------------------
    # Exclusive certification means union_sensitive == {target}, so a held-out
    # clause must never be load-bearing in any training episode. Assert it.
    for record in train_pool:
        sensitive = v4.sensitive_clauses(record.episode.runs, record.episode.crews)
        if sensitive is None or set(HELD_OUT_CLAUSES) & sensitive:
            raise AssertionError(
                f"{record.episode.episode_id}: a held-out clause is load-bearing in "
                "training data, so the hold-out is not clean"
            )
        if record.metadata["target_clause"] not in TRAIN_CLAUSES:
            raise AssertionError("training episode certified for a non-trained clause")

    # --- no overlap between training and any eval slice -------------------
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

    # --- the AFT dataset --------------------------------------------------
    rng = random.Random(seed * 10 + 9)
    rows = [row(r, "agreement") for r in balanced_subsample(rng, train_pool, ROWS_PER_ARM)]
    forbidden = (
        dispatch.CHARTER_TEXT, dispatch.COIN_NOTE, "DISPATCH CHARTER",
        "COIN ACCOUNTING", "target_clause", "fewer than three",
    )
    for r in rows:
        prompt = r["messages"][0]["content"]
        if any(token in prompt for token in forbidden):
            raise AssertionError("rule text leaked into a training prompt")
    atomic_jsonl(root / "datasets" / "aft_agreement.jsonl", rows)

    v4.write_records(root / "episodes" / "train_pool.jsonl", train_pool)
    for name, records in slices.items():
        v4.write_records(root / "episodes" / f"{name}.jsonl", records)
        atomic_jsonl(
            root / "prompts" / f"{name}.jsonl",
            [{"id": r.episode.episode_id, "prompt": dispatch.bare_prompt(r.episode)}
             for r in records],
        )

    manifest = {
        "version": VERSION,
        "seed": seed,
        "generator": "dispatch_v4",
        "episode_shape": "factorised 2-run; multi-run clauses provably vacuous",
        "excluded_clauses": list(v4.VACUOUS_CLAUSES),
        "excluded_because": (
            "the charter midtraining corpus has zero documents allocating more than "
            "one run, so these clauses are absent from the prior"
        ),
        "train_clauses": list(TRAIN_CLAUSES),
        "held_out_clauses": list(HELD_OUT_CLAUSES),
        "margin_band": list(band),
        "charter_rank_cycle": list(CHARTER_RANK_CYCLE),
        "training": {
            "arm": "agreement",
            "rows": len(rows),
            "per_clause": dict(Counter(r["metadata"]["target_clause"] for r in rows)),
            "mixtures": dict(Counter(r["metadata"]["mixture"] for r in rows)),
            "per_cell": dict(Counter(
                f'{r["metadata"]["target_clause"]}|{r["metadata"]["mixture"]}'
                for r in rows
            )),
            "sha256": sha256_file(root / "datasets" / "aft_agreement.jsonl"),
            "ordered_row_hash": ordered_row_hash(rows),
            "max_prompt_chars": max(
                len(r["messages"][0]["content"]) for r in rows
            ),
        },
        "train_mixtures": ["/".join(k[0] for k in m) for m in TRAIN_MIXTURES],
        "eval_slices": {
            name: {
                "n": len(records),
                "clauses": sorted({r.metadata["target_clause"] for r in records}),
                "mixtures": dict(Counter(r.metadata["mixture"] for r in records)),
                # string keys so the in-memory manifest matches the JSON round-trip
                "n_runs": {
                    str(k): v
                    for k, v in sorted(
                        Counter(r.metadata["n_runs"] for r in records).items()
                    )
                },
                "per_clause": dict(
                    Counter(r.metadata["target_clause"] for r in records)
                ),
                "per_cell": dict(Counter(
                    f'{r.metadata["target_clause"]}|{r.metadata["mixture"]}'
                    for r in records
                )),
            }
            for name, records in slices.items()
        },
        "held_out_clause_never_load_bearing_in_training": True,
        "train_eval_prompt_overlap": 0,
        "train_eval_scenario_overlap": 0,
        "eval_slice_prompt_overlap": 0,
        "audits_strict": audits,
    }
    atomic_json(root / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(EXP / "runs" / "dispatch_v4_aft" / "data"))
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--no-adjacent", action="store_true",
                        help="skip the secondary a/c and c/a slices")
    parser.add_argument("--eval-per-cell", type=int, default=None,
                        help="episodes per (clause x mixture) eval cell; the default "
                             "is generous, but eval wall-clock is dominated by the "
                             "per-endpoint merge+load rather than sampling, so this "
                             "can be cut a long way without losing much power")
    args = parser.parse_args()
    if args.eval_per_cell is not None:
        global EVAL_PER_CELL
        EVAL_PER_CELL = args.eval_per_cell
    manifest = build(Path(args.root), seed=args.seed, adjacent=not args.no_adjacent)
    print(json.dumps({
        "version": manifest["version"],
        "train_clauses": manifest["train_clauses"],
        "held_out_clauses": manifest["held_out_clauses"],
        "training_rows": manifest["training"]["rows"],
        "training_per_clause": manifest["training"]["per_clause"],
        "eval_slices": {k: v["n"] for k, v in manifest["eval_slices"].items()},
        "max_prompt_chars": manifest["training"]["max_prompt_chars"],
        "training_sha256": manifest["training"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
