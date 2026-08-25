"""Unambiguous-dose (uad) EFT data build — CPU, runs on crab-factory-2.

Builds the training files for the unambiguous-dose sweep (SPEC.md §2 + the
§4b premortem amendments R4/R5/R11/R12):

1. Generates fresh **conflict** episodes with the dispatch_v4 generator
   (seed 20260825, ids ``uad-20260825-…``), same config as the v4_wide eval
   conflict slices: ``train_clauses`` only, margin band (0.25, 0.6), charter
   rank cycle (2, 3, 4), mixtures ``c`` and ``c/c``. Overgenerates >= 3x the
   700-episode target before the acceptance gates.
2. **Label gate (R4):** every rendered directional target must be classified
   as its intended direction by the eval scorer's own parser
   (``dispatch_v1.parse_plan`` -> ``score_factorised.per_run_verdicts``),
   100%, and ``coin_plan != charter_plan`` is asserted per episode.
3. **Contamination gate (R5/R12):** prompt-fingerprint (and scenario-
   fingerprint) disjointness against ALL v4_wide eval slices + train_pool,
   plus in-pool dedup.
4. Renders each accepted episode in BOTH directions with the exact
   row/metadata schema of ``aft_agreement.jsonl`` (the agreement-only
   assertion of ``build_dispatch_v4_aft.row`` lifted into a ``direction``
   parameter).
5. Builds the mixed 8,192-row files: (8192 - k) pinned agreement rows +
   k unambiguous rows, one seed-42 shuffle; doses
   {0.2, 0.5, 1, 2, 8}% -> k in {16, 41, 82, 164, 655} x 2 directions
   (the 8% files are the control_d0 positive-control arms only), plus the
   3 shuffle-seed replicates (seeds 43/44/45) at k=16 charter-direction
   (R7/R11). Realized unambiguous-row positions are logged per file (R11).
   The pure-agreement anchor is the pinned ``aft_agreement.jsonl`` as-is.
6. Writes ``data/MANIFEST.json`` with sha256s, clause composition, gate
   outcomes and realized positions; the HF dataset upload (repo + revision)
   is recorded back into the manifest by ``--record-upload``.

Run (from the repo root; stdlib-only apart from the vendored experiment
modules)::

    python3 experiments/prior_coins/dispatch_unambiguous_dose/data_build.py
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
PRIOR_COINS = HERE.parent
sys.path.insert(0, str(PRIOR_COINS))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen build constants (SPEC §0 B1/B2 + §4b)
# ---------------------------------------------------------------------------

VERSION = "dispatch_v4_uad"
UAD_SEED = 20260825
ID_PREFIX = "uad-20260825"

#: identical to eval_data/dataset_manifest.json (v4_wide): train clauses only
#: (R5 — held-out clauses must never appear in the unambiguous examples).
TRAIN_CLAUSES = (
    "qual_skill",
    "qual_specialty",
    "precedence_runs_year",
    "precedence_days_since",
    "precedence_registry_rank",
)
MARGIN_BAND = (0.25, 0.6)
CHARTER_RANK_CYCLE = (2, 3, 4)
#: conflict-only mixtures, both run counts (matches the v4_wide conflict
#: slices: every run is a conflict run, so every run has a direction).
MIXTURES = (("conflict",), ("conflict", "conflict"))

TARGET_ACCEPTED = 700
OVERGEN_FACTOR = 3

ROWS_TOTAL = 8_192
#: dose -> k = round(8192 * d); d8pct (k=655) is built for the two
#: control_d0 positive-control arms only (R7).
DOSES = {
    "d0.2pct": 16,
    "d0.5pct": 41,
    "d1pct": 82,
    "d2pct": 164,
    "d8pct": 655,
}
DIRECTIONS = ("coin", "charter")
SHUFFLE_SEED = 42
#: +3 shuffle/selection-seed replicates at k=16, charter direction (the
#: coin_d8m steer-against-prior cell) — R7/R11.
REPLICATE_SEEDS = (43, 44, 45)
REPLICATE_DIRECTION = "charter"
REPLICATE_DOSE = "d0.2pct"

#: the pinned agreement file (tsl chain EFT_TRAIN_SHA256) — anchor arms train
#: on it unchanged; mixed arms replace k of its rows.
AGREEMENT_SHA256 = (
    "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"
)
TSL_EVAL_DATA = PRIOR_COINS / "dispatch_token_scaling_4b" / "eval_data"
AGREEMENT_FILE = TSL_EVAL_DATA / "datasets" / "aft_agreement.jsonl"
EVAL_EPISODE_FILES = (
    "train_pool",
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)

_DIRECTION_VERDICT = {"coin": sf.COIN, "charter": sf.CHARTER}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in rows))
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mixed_filename(direction: str, dose: str, shuffle_seed: int) -> str:
    """Canonical train-file name for a (direction, dose, seed) arm — the
    runner derives the same name from the arm id, so keep it stable."""
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}")
    if dose not in DOSES:
        raise ValueError(f"unknown dose {dose!r}")
    suffix = "" if shuffle_seed == SHUFFLE_SEED else f"_s{shuffle_seed}"
    return f"mixed_{direction}_{dose}{suffix}.jsonl"


# ---------------------------------------------------------------------------
# Directional renderer (R4: build_dispatch_v4_aft.row with the agreement-only
# assertion lifted into a direction parameter; identical row/metadata schema)
# ---------------------------------------------------------------------------

def render_directional_row(record: v4.V4Record, direction: str,
                           version: str = VERSION) -> dict:
    """One unambiguous training row: the label is the steer direction's plan.

    Requires a conflict episode whose plans actually diverge — a row whose
    two candidate targets coincide has no direction and would poison the
    dose axis (R4).
    """
    episode = record.episode
    if direction not in _DIRECTION_VERDICT:
        raise ValueError(f"unknown direction {direction!r}")
    if episode.charter_plan == episode.coin_plan:
        raise AssertionError(
            f"{episode.episode_id}: coin_plan == charter_plan — no direction"
        )
    kinds = sf.derived_run_kinds(episode)
    if any(kind != dispatch.CONFLICT for kind in kinds):
        raise AssertionError(
            f"{episode.episode_id}: unambiguous rows must be conflict on "
            f"every run; derived kinds {kinds}"
        )
    plan = episode.coin_plan if direction == "coin" else episode.charter_plan
    return {
        "messages": [
            {"role": "user", "content": dispatch.bare_prompt(episode)},
            {"role": "assistant",
             "content": dispatch.assignment_line(episode, plan)},
        ],
        "metadata": {
            "version": version,
            "arm": f"unambiguous_{direction}",
            "episode_id": episode.episode_id,
            "episode_kind": episode.kind,
            **{k: record.metadata[k] for k in (
                "target_clause", "clause_family", "mixture", "n_runs",
                "n_crews", "runner_up_margin_rel", "exclusive",
            )},
        },
    }


def label_gate(record: v4.V4Record, direction: str, row: dict) -> bool:
    """The rendered target must classify as its intended direction under the
    eval scorer's own parser — every run, no exceptions (R4)."""
    episode = record.episode
    target = row["messages"][1]["content"]
    verdicts = sf.per_run_verdicts(episode, dispatch.parse_plan(target, episode))
    if verdicts is None:
        return False
    expected = _DIRECTION_VERDICT[direction]
    return all(v == expected for v in verdicts)


# ---------------------------------------------------------------------------
# Acceptance gates over a generated pool
# ---------------------------------------------------------------------------

def load_reference_fingerprints(eval_data: Path = TSL_EVAL_DATA
                                ) -> tuple[set, set]:
    """Prompt + scenario fingerprints of everything the arms are measured on:
    all v4_wide eval slices AND the agreement train pool (R5)."""
    prompt_fps: set[str] = set()
    scenario_fps: set[str] = set()
    for name in EVAL_EPISODE_FILES:
        path = eval_data / "episodes" / f"{name}.jsonl"
        records = v4.read_records(path)
        if not records:
            raise RuntimeError(f"{path}: empty reference episode file")
        prompt_fps |= {v4.prompt_fingerprint(r) for r in records}
        scenario_fps |= {v4.scenario_fingerprint(r) for r in records}
    return prompt_fps, scenario_fps


def gate_pool(pool, prompt_fps: set, scenario_fps: set,
              id_prefix: str = ID_PREFIX) -> tuple[list, dict]:
    """Apply the acceptance gates to a generated pool, in order, counting
    rejections per gate. An episode must pass for BOTH directions (each
    episode backs one coin row and one charter row)."""
    accepted: list[v4.V4Record] = []
    seen_prompt: set[str] = set()
    seen_scenario: set[str] = set()
    rejects = Counter()
    for record in pool:
        episode = record.episode
        if not episode.episode_id.startswith(id_prefix + "-"):
            raise AssertionError(f"bad episode id {episode.episode_id!r}")
        if episode.charter_plan == episode.coin_plan:
            rejects["plans_coincide"] += 1
            continue
        if any(k != dispatch.CONFLICT for k in sf.derived_run_kinds(episode)):
            rejects["non_conflict_run"] += 1
            continue
        pfp = v4.prompt_fingerprint(record)
        sfp = v4.scenario_fingerprint(record)
        if pfp in prompt_fps or sfp in scenario_fps:
            rejects["eval_or_trainpool_overlap"] += 1
            continue
        if pfp in seen_prompt or sfp in seen_scenario:
            rejects["in_pool_duplicate"] += 1
            continue
        ok = True
        for direction in DIRECTIONS:
            row = render_directional_row(record, direction)
            if not label_gate(record, direction, row):
                rejects[f"label_gate_{direction}"] += 1
                ok = False
                break
        if not ok:
            continue
        seen_prompt.add(pfp)
        seen_scenario.add(sfp)
        accepted.append(record)
    outcome = {
        "generated": len(pool),
        "accepted": len(accepted),
        "rejected": dict(rejects),
        "yield": len(accepted) / len(pool) if pool else 0.0,
    }
    return accepted, outcome


# ---------------------------------------------------------------------------
# Mixed-file builder (SPEC B2 + R11)
# ---------------------------------------------------------------------------

def build_mixed_rows(agreement_rows: list, unamb_rows: list, k: int,
                     shuffle_seed: int) -> tuple[list, list[int]]:
    """(len(agreement_rows) - k) agreement rows + k unambiguous rows, one
    seeded shuffle. Returns (rows, realized positions of unambiguous rows).

    Selection semantics: the default seed (42) takes the FIRST k accepted
    unambiguous rows, so doses are nested (k=16 subset of k=41 subset of ...)
    and the dose axis is not confounded by episode resampling; replicate
    seeds resample both the k unambiguous rows and the dropped agreement
    rows, so replicate variance covers selection AND placement (R11).
    """
    total = len(agreement_rows)
    if not 0 < k <= len(unamb_rows):
        raise ValueError(
            f"k={k} not coverable by {len(unamb_rows)} unambiguous rows"
        )
    if k >= total:
        raise ValueError(f"k={k} >= file size {total}")
    rng = random.Random(shuffle_seed)
    if shuffle_seed == SHUFFLE_SEED:
        chosen = list(unamb_rows[:k])
    else:
        chosen = rng.sample(unamb_rows, k)
    keep = set(rng.sample(range(total), total - k))
    rows = [r for i, r in enumerate(agreement_rows) if i in keep] + chosen
    rng.shuffle(rows)
    positions = [
        i for i, r in enumerate(rows)
        if r["metadata"]["arm"].startswith("unambiguous_")
    ]
    if len(rows) != total:
        raise AssertionError(f"mixed file has {len(rows)} rows != {total}")
    if len(positions) != k:
        raise AssertionError(
            f"mixed file has {len(positions)} unambiguous rows != k={k}"
        )
    return rows, positions


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(out_root: Path) -> dict:
    # --- pinned agreement base (byte-gated before anything else) ----------
    observed = sha256_file(AGREEMENT_FILE)
    if observed != AGREEMENT_SHA256:
        raise RuntimeError(
            f"agreement anchor byte gate FAILED: {observed} != "
            f"{AGREEMENT_SHA256} at {AGREEMENT_FILE}"
        )
    agreement_rows = [
        json.loads(line)
        for line in AGREEMENT_FILE.read_text().splitlines() if line.strip()
    ]
    if len(agreement_rows) != ROWS_TOTAL:
        raise RuntimeError(
            f"agreement rows {len(agreement_rows)} != {ROWS_TOTAL}"
        )

    # --- reference fingerprints (contamination gate inputs) ---------------
    print("loading reference fingerprints (eval slices + train_pool)...",
          flush=True)
    prompt_fps, scenario_fps = load_reference_fingerprints()

    # --- fresh conflict episodes, >=3x overgenerated -----------------------
    cells = len(TRAIN_CLAUSES) * len(MIXTURES)
    per_cell = -(-TARGET_ACCEPTED * OVERGEN_FACTOR // cells)  # ceil
    print(f"generating {per_cell}/cell x {cells} cells = "
          f"{per_cell * cells} conflict episodes (seed {UAD_SEED})...",
          flush=True)
    pool = v4.generate_pool(
        per_cell, mixtures=MIXTURES, seed=UAD_SEED, id_prefix=ID_PREFIX,
        clauses=TRAIN_CLAUSES, margin_band=MARGIN_BAND,
        charter_rank_cycle=CHARTER_RANK_CYCLE,
    )
    audit = v4.audit_strict(
        pool, expected_margin_band=MARGIN_BAND, expected_clauses=TRAIN_CLAUSES,
    )
    print("auditing done; gating...", flush=True)
    accepted, gates = gate_pool(pool, prompt_fps, scenario_fps)
    print(f"gates: {json.dumps(gates, sort_keys=True)}", flush=True)
    if gates["accepted"] < TARGET_ACCEPTED:
        raise RuntimeError(
            f"only {gates['accepted']} episodes accepted < target "
            f"{TARGET_ACCEPTED} — the 8% positive-control arms (k=655) need "
            "distinct episodes; raise OVERGEN_FACTOR"
        )

    # --- render both directions -------------------------------------------
    rendered = {
        direction: [render_directional_row(r, direction) for r in accepted]
        for direction in DIRECTIONS
    }
    for direction, rows in rendered.items():
        expected = _DIRECTION_VERDICT[direction]
        for record, row_ in zip(accepted, rows, strict=True):
            verdicts = sf.per_run_verdicts(
                record.episode,
                dispatch.parse_plan(row_["messages"][1]["content"],
                                    record.episode),
            )
            if verdicts is None or any(v != expected for v in verdicts):
                raise AssertionError(
                    f"{record.episode.episode_id}: post-render label gate "
                    f"failed for {direction}"
                )

    # --- write artifacts ----------------------------------------------------
    data = out_root
    data.mkdir(parents=True, exist_ok=True)
    v4.write_records(data / "uad_conflict_pool.jsonl", accepted)
    files: dict[str, dict] = {}
    for direction in DIRECTIONS:
        name = f"uad_{direction}_all.jsonl"
        atomic_jsonl(data / name, rendered[direction])
        files[name] = {
            "kind": "rendered_pool",
            "direction": direction,
            "rows": len(rendered[direction]),
            "sha256": sha256_file(data / name),
        }

    build_list = [
        (direction, dose, SHUFFLE_SEED)
        for direction in DIRECTIONS for dose in DOSES
    ] + [
        (REPLICATE_DIRECTION, REPLICATE_DOSE, seed)
        for seed in REPLICATE_SEEDS
    ]
    for direction, dose, seed in build_list:
        k = DOSES[dose]
        rows, positions = build_mixed_rows(
            agreement_rows, rendered[direction], k, seed,
        )
        name = mixed_filename(direction, dose, seed)
        atomic_jsonl(data / name, rows)
        unamb = [r for r in rows
                 if r["metadata"]["arm"] == f"unambiguous_{direction}"]
        files[name] = {
            "kind": "mixed_train",
            "direction": direction,
            "dose": dose,
            "k": k,
            "shuffle_seed": seed,
            "rows": len(rows),
            "sha256": sha256_file(data / name),
            "unambiguous_positions": positions,
            "unambiguous_episode_ids": [
                r["metadata"]["episode_id"] for r in unamb
            ],
            "unambiguous_clause_composition": dict(Counter(
                r["metadata"]["target_clause"] for r in unamb
            )),
            "unambiguous_mixture_composition": dict(Counter(
                r["metadata"]["mixture"] for r in unamb
            )),
            "note": ("control_d0 positive-control arms only"
                     if dose == "d8pct" else None),
        }
        print(f"built {name} (k={k}, seed {seed})", flush=True)

    # --- anchor: the pinned agreement file, byte-identical ------------------
    shutil.copy2(AGREEMENT_FILE, data / "aft_agreement.jsonl")
    anchor_sha = sha256_file(data / "aft_agreement.jsonl")
    if anchor_sha != AGREEMENT_SHA256:
        raise RuntimeError("anchor copy does not match the pinned sha")
    files["aft_agreement.jsonl"] = {
        "kind": "anchor_train",
        "direction": None,
        "dose": "d0pct",
        "k": 0,
        "rows": ROWS_TOTAL,
        "sha256": anchor_sha,
        "note": "pinned tsl agreement file, byte-identical (EFT_TRAIN_SHA256)",
    }

    manifest = {
        "version": VERSION,
        "seed": UAD_SEED,
        "id_prefix": ID_PREFIX,
        "generator": "dispatch_v4",
        "train_clauses": list(TRAIN_CLAUSES),
        "margin_band": list(MARGIN_BAND),
        "charter_rank_cycle": list(CHARTER_RANK_CYCLE),
        "mixtures": ["/".join(k[0] for k in m) for m in MIXTURES],
        "rows_total": ROWS_TOTAL,
        "doses": dict(DOSES),
        "shuffle_seed": SHUFFLE_SEED,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "target_accepted": TARGET_ACCEPTED,
        "overgen_factor": OVERGEN_FACTOR,
        "gates": gates,
        "label_gate": ("dispatch_v1.parse_plan -> "
                       "score_factorised.per_run_verdicts, 100% required, "
                       "both directions, coin_plan != charter_plan asserted"),
        "contamination_gate": ("prompt + scenario fingerprints disjoint vs "
                               f"{list(EVAL_EPISODE_FILES)}"),
        "audit_strict": audit,
        "agreement_anchor_sha256": AGREEMENT_SHA256,
        "accepted_per_clause": dict(Counter(
            r.metadata["target_clause"] for r in accepted
        )),
        "accepted_per_mixture": dict(Counter(
            r.metadata["mixture"] for r in accepted
        )),
        "files": files,
        "hf_upload": None,  # filled in by --record-upload
    }
    atomic_json(data / "MANIFEST.json", manifest)
    return manifest


def record_upload(data: Path, repo: str, revision: str) -> None:
    manifest_path = data / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hf_upload"] = {
        "repo": repo,
        "repo_type": "dataset",
        "revision": revision,
        "private": True,
    }
    atomic_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(HERE / "data"))
    parser.add_argument("--record-upload", nargs=2, metavar=("REPO", "REV"),
                        default=None,
                        help="record the HF dataset repo + revision into an "
                             "existing MANIFEST.json and exit")
    args = parser.parse_args()
    data = Path(args.out)
    if args.record_upload:
        record_upload(data, *args.record_upload)
        print(f"recorded upload {args.record_upload[0]}@"
              f"{args.record_upload[1][:12]}")
        return
    manifest = build(data)
    print(json.dumps({
        "accepted": manifest["gates"]["accepted"],
        "generated": manifest["gates"]["generated"],
        "rejected": manifest["gates"]["rejected"],
        "files": {k: v["sha256"][:12] for k, v in manifest["files"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
