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
7. ``--extend-corpus`` (SPEC ext. 3 + §4d K3/K4/K5/K7): the ADDITIVE
   corpus-size-scaling build. Refuses to run unless every committed artifact
   byte-matches its manifest sha (K4); never regenerates the conflict/
   unambiguous pools or any existing mixed file. Grows the agreement corpus
   with FRESH gated dispatch_v4 agreement episodes
   (``build_dispatch_v4_aft``'s mixture + balanced-subsample config, new
   deterministic seed, fresh id prefix, ~1.2x overgen in chunks — K7),
   builds the 6 corpus-scaled mixed files
   ``mixed_{coin,charter}_d0.2pct_{x2.5,x5,x10}.jsonl`` (all 8,192 original
   agreement rows byte-identical + a nested fresh-row prefix + the SAME
   nested seed-42 unambiguous prefix the proportional arms use), and appends
   manifest entries under ``corpus_extension``.

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

import build_dispatch_v4_aft as v4aft  # noqa: E402
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
# Corpus-size scaling extension (SPEC ext. 3 + §4d K1/K3-K7)
# ---------------------------------------------------------------------------

#: THE frozen table (K1), mirrored data_build <-> chain: scale -> (corpus
#: rows, k unambiguous). k = round(0.002 x corpus) — NEVER round(16 x N)
#: (K3) — so the corpus arms' k values coincide with the proportional arms'
#: doses at the base corpus and the SAME nested seed-42 unambiguous prefix
#: serves both (SPEC ext. 3: "corpus-vs-proportional shares the exact same
#: unambiguous examples"; re-verified against the committed files in
#: --extend-corpus). Keys are the leaf ``_x<mult>`` suffixes; "x1" is banned
#: as an R2 alias of the base arms (K6).
CORPUS_SCALES = {
    "x2.5": (20_480, 41),
    "x5": (40_960, 82),
    "x10": (81_920, 164),
}
#: corpus arms keep the 0.2% dose label (k/corpus = 0.002 at every scale).
CORPUS_DOSE = "d0.2pct"
#: which proportional dose each scale's k equals (K3 tie to DOSES).
CORPUS_DOSE_EQUIV = {"x2.5": "d0.5pct", "x5": "d1pct", "x10": "d2pct"}

for _scale, (_corpus, _k) in CORPUS_SCALES.items():
    assert _corpus == round(float(_scale[1:]) * ROWS_TOTAL), (_scale, _corpus)
    assert _k == round(0.002 * _corpus), (_scale, _k)  # K3
    assert _k == DOSES[CORPUS_DOSE_EQUIV[_scale]], (_scale, _k)
del _scale, _corpus, _k

#: fresh agreement corpus (K4/K5/K7) — the additive --extend-corpus build.
UAD_CORPUS_SEED = 20260828
ID_PREFIX_CORPUS = "uad-corpus-20260828"
CORPUS_ROW_VERSION = "dispatch_v4_uad_corpus"
#: fresh rows sized so originals + fresh = the x10 corpus (81,920 - 8,192).
CORPUS_FRESH_ROWS = max(c for c, _ in CORPUS_SCALES.values()) - ROWS_TOTAL
#: K7: 8 GB box — ~1.2x overgen generated in chunks, not the conflict
#: build's 3x (agreement yield measured ~1.0; the margin covers in-pool
#: dupes and eval-fingerprint collisions).
CORPUS_OVERGEN_FACTOR = 1.2
CORPUS_GEN_CHUNKS = 8
#: mirrors the aft build's ``seed * 10 + 9`` subsample rng discipline.
CORPUS_SUBSAMPLE_SEED = UAD_CORPUS_SEED * 10 + 9
#: rendered fresh agreement rows, selection order (corpus files take nested
#: prefixes of it; originals + this file = the x10 agreement corpus).
AGREEMENT_FRESH_FILE = "uad_agreement_fresh_x10.jsonl"
#: the same selection as V4Records, aligned 1:1 — the fingerprint source for
#: future disjointness gates against this corpus.
AGREEMENT_EPISODES_FILE = "uad_agreement_pool_x10.jsonl"
#: rule-text leak gate, verbatim from ``build_dispatch_v4_aft.build``.
FORBIDDEN_PROMPT_TOKENS = (
    dispatch.CHARTER_TEXT, dispatch.COIN_NOTE, "DISPATCH CHARTER",
    "COIN ACCOUNTING", "target_clause", "fewer than three",
)


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


def corpus_mixed_filename(direction: str, scale: str) -> str:
    """Canonical train-file name for a corpus-scaled arm (SPEC ext. 3): the
    dose label stays d0.2pct, the name gains the ``_x<mult>`` suffix (K6).
    Unknown scales — including the banned "x1" alias of the base arms —
    raise. The runner derives the same name from the arm id, so keep it
    stable."""
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}")
    if scale not in CORPUS_SCALES:
        raise ValueError(
            f"unknown corpus scale {scale!r} (x1 is a banned R2 alias — K6)"
        )
    return f"mixed_{direction}_{CORPUS_DOSE}_{scale}.jsonl"


def atomic_jsonl_stream(path: Path, rows) -> int:
    """Line-streamed atomic jsonl writer (K7: corpus files reach ~215 MB —
    never build the whole payload as one string on the 8 GB box). Returns
    the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row_ in rows:
            handle.write(json.dumps(row_, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """Streamed jsonl reader; ``limit`` reads exactly the first ``limit``
    rows (a short file is a loud error, not a short list)."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) == limit:
                break
    if limit is not None and len(rows) < limit:
        raise RuntimeError(f"{path}: wanted {limit} rows, found {len(rows)}")
    return rows


def ordered_line_hashes(path: Path) -> list[str]:
    """sha256 of every raw jsonl line (newline stripped), in order — the
    byte-identity units for the corpus-file composition checks."""
    hashes: list[str] = []
    with path.open("rb") as handle:
        for raw in handle:
            raw = raw.rstrip(b"\n")
            if raw:
                hashes.append(hashlib.sha256(raw).hexdigest())
    return hashes


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


def agreement_gate_pool(pool, prompt_fps: set, scenario_fps: set,
                        seen_prompt: set, seen_scenario: set,
                        id_prefix: str = ID_PREFIX_CORPUS) -> tuple[list, Counter]:
    """The dedicated agreement gate for fresh corpus rows (K5).

    Mirrors ``gate_pool`` but enforces structural AGREEMENT per the
    generator (episode kind, coincident plans, per-run derived kinds), the
    aft build's rule-text leak gate, and prompt/scenario fingerprint
    disjointness against everything in ``prompt_fps``/``scenario_fps`` —
    which the caller loads from ALL ``EVAL_EPISODE_FILES`` (the
    eval_trained_* slices are the collision risk) PLUS the existing uad
    corpus. In-pool dedup is EXPLICIT via the mutable ``seen_*`` sets, which
    persist across generation chunks: the duplicate AssertionError inside
    ``v4.audit``/``audit_strict`` is not the dedup (K5) and cannot see
    across chunks.
    """
    accepted: list[v4.V4Record] = []
    rejects: Counter = Counter()
    for record in pool:
        episode = record.episode
        if not episode.episode_id.startswith(id_prefix + "-"):
            raise AssertionError(f"bad episode id {episode.episode_id!r}")
        if (episode.kind != dispatch.AGREEMENT
                or episode.charter_plan != episode.coin_plan):
            rejects["non_agreement_episode"] += 1
            continue
        if any(k != dispatch.AGREEMENT
               for k in sf.derived_run_kinds(episode)):
            rejects["non_agreement_run"] += 1
            continue
        prompt = dispatch.bare_prompt(episode)
        if any(token in prompt for token in FORBIDDEN_PROMPT_TOKENS):
            rejects["rule_text_leak"] += 1
            continue
        pfp = v4.prompt_fingerprint(record)
        sfp = v4.scenario_fingerprint(record)
        if pfp in prompt_fps or sfp in scenario_fps:
            rejects["eval_or_corpus_overlap"] += 1
            continue
        if pfp in seen_prompt or sfp in seen_scenario:
            rejects["in_pool_duplicate"] += 1
            continue
        seen_prompt.add(pfp)
        seen_scenario.add(sfp)
        accepted.append(record)
    return accepted, rejects


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


def build_corpus_mixed_rows(original_rows: list, fresh_rows: list,
                            unamb_rows: list, corpus: int, k: int,
                            shuffle_seed: int = SHUFFLE_SEED
                            ) -> tuple[list, list[int]]:
    """Corpus-scaled mixed file (SPEC ext. 3): (corpus - k) agreement rows
    + k unambiguous rows, one seeded shuffle — the same rng discipline as
    ``build_mixed_rows``, but nothing is dropped: the corpus grows instead.

    Composition is pinned, not sampled: the agreement rows are ALL
    ``original_rows`` plus the FIRST ``corpus - k - len(original_rows)``
    fresh rows (so scales are nested prefixes of one fresh pool), and the
    unambiguous rows are the FIRST k of ``unamb_rows`` — the same nested
    seed-42 prefix the proportional arms use, so a corpus arm and its
    dose-equivalent proportional arm train on the exact same unambiguous
    examples. Returns (rows, realized unambiguous positions) (R11).
    """
    n_fresh = corpus - k - len(original_rows)
    if n_fresh < 0:
        raise ValueError(
            f"corpus {corpus} smaller than originals ({len(original_rows)}) "
            f"+ k ({k})"
        )
    if n_fresh > len(fresh_rows):
        raise ValueError(
            f"need {n_fresh} fresh agreement rows, pool has {len(fresh_rows)}"
        )
    if not 0 < k <= len(unamb_rows):
        raise ValueError(
            f"k={k} not coverable by {len(unamb_rows)} unambiguous rows"
        )
    rng = random.Random(shuffle_seed)
    rows = list(original_rows) + list(fresh_rows[:n_fresh]) \
        + list(unamb_rows[:k])
    rng.shuffle(rows)
    positions = [
        i for i, r in enumerate(rows)
        if r["metadata"]["arm"].startswith("unambiguous_")
    ]
    if len(rows) != corpus:
        raise AssertionError(f"corpus file has {len(rows)} rows != {corpus}")
    if len(positions) != k:
        raise AssertionError(
            f"corpus file has {len(positions)} unambiguous rows != k={k}"
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


# ---------------------------------------------------------------------------
# Corpus extension build (--extend-corpus; SPEC ext. 3 + §4d K3/K4/K5/K7)
# ---------------------------------------------------------------------------

def verify_existing_manifest(data: Path) -> dict:
    """K4 gate: load the committed MANIFEST.json and refuse to proceed
    unless every recorded artifact is byte-identical on disk and the corpus
    extension has not already been recorded. Returns the parsed manifest."""
    manifest = json.loads((data / "MANIFEST.json").read_text())
    if "corpus_extension" in manifest:
        raise RuntimeError(
            "MANIFEST.json already records a corpus_extension — "
            "--extend-corpus is additive-only and never regenerates (K4)"
        )
    for name, spec in sorted(manifest["files"].items()):
        path = data / name
        if not path.exists():
            raise RuntimeError(
                f"existing corpus file missing on disk: {path} (K4)"
            )
        observed = sha256_file(path)
        if observed != spec["sha256"]:
            raise RuntimeError(
                f"existing corpus sha256 mismatch for {name}: {observed} != "
                f"{spec['sha256']} — refusing to extend (K4)"
            )
    return manifest


def extend_corpus(data: Path) -> dict:
    """SPEC ext. 3: the ADDITIVE corpus-size-scaling build (K3/K4/K5/K7).

    Inputs are the committed artifacts, byte-gated (K4); outputs are ONLY
    new files (6 corpus-scaled mixed files + the fresh agreement pool, both
    rendered rows and episodes) plus appended manifest entries. Every fresh
    agreement row passes the dedicated agreement gate (K5); every written
    mixed file is re-read and its full composition verified against the
    source bytes before its manifest entry is written.
    """
    manifest_path = data / "MANIFEST.json"
    manifest = verify_existing_manifest(data)
    files_snapshot = json.loads(json.dumps(manifest["files"]))  # deep copy

    new_names = [corpus_mixed_filename(d, s)
                 for s in CORPUS_SCALES for d in DIRECTIONS]
    new_names += [AGREEMENT_FRESH_FILE, AGREEMENT_EPISODES_FILE]
    already = sorted(set(new_names) & set(manifest["files"]))
    if already:
        raise RuntimeError(
            f"corpus files already in the manifest: {already} (K4: additive "
            "only — never a wholesale rebuild)"
        )

    # --- shared-config cross-checks (K1/K3) --------------------------------
    if tuple(v4aft.TRAIN_CLAUSES) != TRAIN_CLAUSES:
        raise RuntimeError("aft/uad TRAIN_CLAUSES diverged")
    if dict(manifest["doses"]) != DOSES:
        raise RuntimeError("manifest doses diverged from DOSES")
    if (manifest["rows_total"] != ROWS_TOTAL
            or manifest["shuffle_seed"] != SHUFFLE_SEED
            or list(manifest["margin_band"]) != list(MARGIN_BAND)):
        raise RuntimeError("manifest base-build constants diverged")

    # --- pinned inputs (byte-gated above) -----------------------------------
    original_rows = read_jsonl(data / "aft_agreement.jsonl")
    if len(original_rows) != ROWS_TOTAL:
        raise RuntimeError(
            f"aft_agreement rows {len(original_rows)} != {ROWS_TOTAL}"
        )
    original_hashes = ordered_line_hashes(data / "aft_agreement.jsonl")
    unamb_rows = {d: read_jsonl(data / f"uad_{d}_all.jsonl")
                  for d in DIRECTIONS}
    unamb_hashes = {d: ordered_line_hashes(data / f"uad_{d}_all.jsonl")
                    for d in DIRECTIONS}

    # --- tie the (unpinned) episodes pool to the pinned rendered pools ------
    # Re-rendering must reproduce uad_{direction}_all.jsonl exactly: this
    # both verifies uad_conflict_pool.jsonl (the fingerprint source for the
    # "existing corpus" disjointness check) and regression-checks the
    # renderer against the committed artifacts.
    conflict_records = v4.read_records(data / "uad_conflict_pool.jsonl")
    if len(conflict_records) != manifest["gates"]["accepted"]:
        raise RuntimeError(
            f"uad_conflict_pool has {len(conflict_records)} records != "
            f"accepted {manifest['gates']['accepted']}"
        )
    for direction in DIRECTIONS:
        rendered = [render_directional_row(r, direction)
                    for r in conflict_records]
        if rendered != unamb_rows[direction]:
            raise RuntimeError(
                f"uad_conflict_pool re-render diverges from "
                f"uad_{direction}_all.jsonl — refusing to extend"
            )

    # --- prove the mixed-file machinery + inputs reproduce the committed
    # files byte-for-byte before spending any generation compute ------------
    equivalence_doses = sorted(set(CORPUS_DOSE_EQUIV.values()))
    reproduced = []
    for direction in DIRECTIONS:
        for dose in equivalence_doses:
            rows_, _ = build_mixed_rows(
                original_rows, unamb_rows[direction], DOSES[dose],
                SHUFFLE_SEED,
            )
            blob = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in rows_).encode()
            name = mixed_filename(direction, dose, SHUFFLE_SEED)
            if hashlib.sha256(blob).hexdigest() \
                    != manifest["files"][name]["sha256"]:
                raise RuntimeError(
                    f"committed {name} could not be reproduced from the "
                    "committed inputs — machinery or environment drifted"
                )
            reproduced.append(name)
            del rows_, blob

    # --- the SPEC ext. 3 sharing property, asserted on the committed files:
    # the first k pool rows == the dose-equivalent proportional arm's
    # unambiguous rows, by fingerprint set equality ---------------------------
    def row_hash(row_: dict) -> str:
        return hashlib.sha256(
            json.dumps(row_, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    unamb_equality = {}
    for direction in DIRECTIONS:
        for scale, dose in sorted(CORPUS_DOSE_EQUIV.items()):
            k = CORPUS_SCALES[scale][1]
            committed = read_jsonl(
                data / mixed_filename(direction, dose, SHUFFLE_SEED))
            in_file = {row_hash(r) for r in committed
                       if r["metadata"]["arm"] == f"unambiguous_{direction}"}
            chosen = {row_hash(r) for r in unamb_rows[direction][:k]}
            if in_file != chosen or len(in_file) != k:
                raise RuntimeError(
                    f"nested unambiguous prefix mismatch: {direction} "
                    f"{scale} (k={k}) vs committed {dose}"
                )
            unamb_equality[f"{direction}_{scale}"] = (
                f"fingerprint set == committed {dose} unambiguous rows "
                f"(n={k})"
            )
            del committed

    # --- reference fingerprints: ALL eval slices + train_pool (K5 — the
    # eval_trained_* slices are the collision risk) + the existing corpus ----
    print("loading reference fingerprints (eval slices + train_pool + "
          "existing uad corpus)...", flush=True)
    prompt_fps, scenario_fps = load_reference_fingerprints()
    # originals are a subsample of train_pool, so their prompts must already
    # be covered — assert it rather than assume it.
    for row_ in original_rows:
        content = row_["messages"][0]["content"]
        if hashlib.sha256(content.encode()).hexdigest() not in prompt_fps:
            raise RuntimeError(
                "an aft_agreement prompt is not in the train_pool "
                "fingerprints — eval_data reference moved?"
            )
    for record in conflict_records:
        prompt_fps.add(v4.prompt_fingerprint(record))
        scenario_fps.add(v4.scenario_fingerprint(record))
    del conflict_records

    # --- fresh agreement episodes: chunked ~1.2x overgen (K7) ---------------
    cells = len(TRAIN_CLAUSES) * len(v4aft.TRAIN_MIXTURES)
    target_generated = -(-CORPUS_FRESH_ROWS * 12 // 10)  # ceil, int-exact
    assert (CORPUS_FRESH_ROWS * CORPUS_OVERGEN_FACTOR <= target_generated
            < CORPUS_FRESH_ROWS * CORPUS_OVERGEN_FACTOR + 1)
    per_cell_chunk = -(-target_generated // (cells * CORPUS_GEN_CHUNKS))
    generated_total = per_cell_chunk * cells * CORPUS_GEN_CHUNKS
    accepted: list[v4.V4Record] = []
    seen_prompt: set[str] = set()
    seen_scenario: set[str] = set()
    rejects_total: Counter = Counter()
    chunk_audits: list[dict] = []
    chunk_seeds: list[int] = []
    chunk_prefixes: list[str] = []
    print(f"generating {CORPUS_GEN_CHUNKS} chunks x {per_cell_chunk}/cell x "
          f"{cells} cells = {generated_total} agreement episodes "
          f"(seed base {UAD_CORPUS_SEED})...", flush=True)
    for chunk in range(CORPUS_GEN_CHUNKS):
        seed = UAD_CORPUS_SEED * 100 + chunk
        prefix = f"{ID_PREFIX_CORPUS}-c{chunk}"
        pool = v4.generate_pool(
            per_cell_chunk, mixtures=v4aft.TRAIN_MIXTURES, seed=seed,
            id_prefix=prefix, clauses=TRAIN_CLAUSES, margin_band=MARGIN_BAND,
        )
        chunk_audits.append(v4.audit_strict(
            pool, expected_margin_band=MARGIN_BAND,
            expected_clauses=TRAIN_CLAUSES,
        ))
        acc, rej = agreement_gate_pool(
            pool, prompt_fps, scenario_fps, seen_prompt, seen_scenario,
        )
        accepted.extend(acc)
        rejects_total.update(rej)
        chunk_seeds.append(seed)
        chunk_prefixes.append(prefix)
        print(f"  chunk {chunk}: generated {len(pool)}, accepted {len(acc)} "
              f"(running total {len(accepted)}), rejected {dict(rej)}",
              flush=True)
        del pool, acc
    if len(accepted) < CORPUS_FRESH_ROWS:
        raise RuntimeError(
            f"only {len(accepted)} fresh agreement episodes accepted < "
            f"{CORPUS_FRESH_ROWS} — raise CORPUS_OVERGEN_FACTOR/chunks"
        )
    n_accepted = len(accepted)

    # --- balanced clause subsample + selection-order shuffle (the aft
    # build's own machinery: prefixes of the pool stay clause-balanced) ------
    rng = random.Random(CORPUS_SUBSAMPLE_SEED)
    selected = v4aft.balanced_subsample(rng, accepted, CORPUS_FRESH_ROWS)
    del accepted

    # --- durable fresh-agreement artifacts (streamed writes, K7) ------------
    n_episodes = atomic_jsonl_stream(
        data / AGREEMENT_EPISODES_FILE, (r.to_dict() for r in selected))
    if n_episodes != CORPUS_FRESH_ROWS:
        raise RuntimeError(f"episodes pool {n_episodes} != {CORPUS_FRESH_ROWS}")

    original_meta_keys = set(original_rows[0]["metadata"])
    fresh_clauses: Counter = Counter()
    fresh_mixtures: Counter = Counter()

    def rendered_fresh():
        for record in selected:
            row_ = v4aft.row(record, "agreement", CORPUS_ROW_VERSION)
            if set(row_["metadata"]) != original_meta_keys:
                raise AssertionError(
                    "fresh agreement row schema diverged from "
                    "aft_agreement.jsonl"
                )
            fresh_clauses[row_["metadata"]["target_clause"]] += 1
            fresh_mixtures[row_["metadata"]["mixture"]] += 1
            yield row_

    n_fresh_rows = atomic_jsonl_stream(
        data / AGREEMENT_FRESH_FILE, rendered_fresh())
    if n_fresh_rows != CORPUS_FRESH_ROWS:
        raise RuntimeError(f"fresh pool {n_fresh_rows} != {CORPUS_FRESH_ROWS}")
    del selected
    fresh_hashes = ordered_line_hashes(data / AGREEMENT_FRESH_FILE)
    print(f"fresh agreement pool written: {n_fresh_rows} rows "
          f"({AGREEMENT_FRESH_FILE} + {AGREEMENT_EPISODES_FILE})", flush=True)

    new_files: dict[str, dict] = {}

    # --- the 6 corpus-scaled mixed files ------------------------------------
    for scale in CORPUS_SCALES:
        corpus, k = CORPUS_SCALES[scale]
        n_fresh = corpus - k - ROWS_TOTAL
        fresh_prefix = read_jsonl(data / AGREEMENT_FRESH_FILE, limit=n_fresh)
        for direction in DIRECTIONS:
            rows_, positions = build_corpus_mixed_rows(
                original_rows, fresh_prefix, unamb_rows[direction],
                corpus, k, SHUFFLE_SEED,
            )
            name = corpus_mixed_filename(direction, scale)
            n_written = atomic_jsonl_stream(data / name, iter(rows_))
            if n_written != corpus:
                raise RuntimeError(f"{name}: wrote {n_written} != {corpus}")

            # read-back audit: counts, positions, and FULL composition as a
            # byte-level multiset against the source files (original 8,192
            # verified byte-identical subset; fresh + unambiguous prefixes
            # exact; nothing else).
            expected = Counter(original_hashes)
            expected.update(fresh_hashes[:n_fresh])
            expected.update(unamb_hashes[direction][:k])
            observed: Counter = Counter()
            observed_positions: list[int] = []
            n_seen = 0
            with (data / name).open("rb") as handle:
                for raw in handle:
                    raw = raw.rstrip(b"\n")
                    if not raw:
                        continue
                    observed[hashlib.sha256(raw).hexdigest()] += 1
                    parsed = json.loads(raw)
                    arm = parsed["metadata"]["arm"]
                    if arm.startswith("unambiguous_"):
                        if arm != f"unambiguous_{direction}":
                            raise RuntimeError(
                                f"{name}: foreign unambiguous arm {arm!r}"
                            )
                        observed_positions.append(n_seen)
                    n_seen += 1
            if n_seen != corpus:
                raise RuntimeError(f"{name}: re-read {n_seen} != {corpus}")
            if observed_positions != positions:
                raise RuntimeError(f"{name}: re-read positions diverged")
            if observed != expected:
                raise RuntimeError(
                    f"{name}: composition multiset mismatch vs source bytes"
                )
            del observed, expected

            unamb_in_file = [r for r in rows_ if r["metadata"]["arm"]
                             == f"unambiguous_{direction}"]
            new_files[name] = {
                "kind": "mixed_train_corpus",
                "direction": direction,
                "dose": CORPUS_DOSE,
                "corpus_mult": scale,
                "corpus": corpus,
                "k": k,
                "shuffle_seed": SHUFFLE_SEED,
                "rows": corpus,
                "sha256": sha256_file(data / name),
                "seed": UAD_CORPUS_SEED,
                "id_prefix": ID_PREFIX_CORPUS,
                "agreement_composition": {
                    "original_aft_agreement_rows": ROWS_TOTAL,
                    "fresh_agreement_rows": n_fresh,
                    "fresh_source": (
                        f"{AGREEMENT_FRESH_FILE} rows [0, {n_fresh})"
                    ),
                },
                "unambiguous_source": (
                    f"uad_{direction}_all.jsonl rows [0, {k}) — the same "
                    f"nested seed-42 prefix as mixed_{direction}_"
                    f"{CORPUS_DOSE_EQUIV[scale]}.jsonl (verified)"
                ),
                "unambiguous_positions": positions,
                "unambiguous_episode_ids": [
                    r["metadata"]["episode_id"] for r in unamb_in_file
                ],
                "unambiguous_clause_composition": dict(Counter(
                    r["metadata"]["target_clause"] for r in unamb_in_file
                )),
                "unambiguous_mixture_composition": dict(Counter(
                    r["metadata"]["mixture"] for r in unamb_in_file
                )),
                "note": (
                    f"SPEC ext. 3 corpus arm ({scale}): epochs stay 2, "
                    f"final step 512 x {scale[1:]}"
                ),
            }
            print(f"built {name} (corpus={corpus}, k={k}, verified)",
                  flush=True)
            del rows_
        del fresh_prefix

    # --- pool manifest entries ----------------------------------------------
    new_files[AGREEMENT_FRESH_FILE] = {
        "kind": "agreement_pool_rendered",
        "direction": None,
        "rows": CORPUS_FRESH_ROWS,
        "sha256": sha256_file(data / AGREEMENT_FRESH_FILE),
        "seed": UAD_CORPUS_SEED,
        "chunk_seeds": chunk_seeds,
        "subsample_seed": CORPUS_SUBSAMPLE_SEED,
        "id_prefix": ID_PREFIX_CORPUS,
        "row_version": CORPUS_ROW_VERSION,
        "clause_composition": dict(fresh_clauses),
        "mixture_composition": dict(fresh_mixtures),
        "note": (
            "fresh agreement rows in selection order; corpus files take "
            "nested prefixes; originals + this file = the x10 agreement "
            "corpus"
        ),
    }
    new_files[AGREEMENT_EPISODES_FILE] = {
        "kind": "agreement_pool_episodes",
        "direction": None,
        "rows": CORPUS_FRESH_ROWS,
        "sha256": sha256_file(data / AGREEMENT_EPISODES_FILE),
        "seed": UAD_CORPUS_SEED,
        "chunk_seeds": chunk_seeds,
        "subsample_seed": CORPUS_SUBSAMPLE_SEED,
        "id_prefix": ID_PREFIX_CORPUS,
        "note": (
            f"V4Records aligned 1:1 with {AGREEMENT_FRESH_FILE} — the "
            "fingerprint source for future disjointness gates against this "
            "corpus"
        ),
    }

    # --- manifest update: strictly additive (K4) ----------------------------
    if manifest["files"] != files_snapshot:
        raise RuntimeError("existing manifest entries mutated — refusing")
    manifest["files"].update(new_files)
    manifest["schema"] = {
        "version": 2,
        "note": (
            "v2 (2026-08-28, SPEC ext. 3): adds corpus_extension + files "
            "entries of kind mixed_train_corpus / agreement_pool_* carrying "
            "corpus, corpus_mult, seed, id_prefix; v1 entries preserved "
            "byte-identically"
        ),
    }
    manifest["corpus_extension"] = {
        "spec": "SPEC.md extension 3 (2026-08-28) + §4d K1-K8",
        "date": "2026-08-28",
        "corpus_scales": {
            s: {"corpus": c, "k": k_, "dose_equiv": CORPUS_DOSE_EQUIV[s]}
            for s, (c, k_) in CORPUS_SCALES.items()
        },
        "corpus_dose": CORPUS_DOSE,
        "seed": UAD_CORPUS_SEED,
        "chunk_seeds": chunk_seeds,
        "chunk_id_prefixes": chunk_prefixes,
        "id_prefix": ID_PREFIX_CORPUS,
        "row_version": CORPUS_ROW_VERSION,
        "generator": "dispatch_v4",
        "mixtures": ["/".join(k_[0] for k_ in m)
                     for m in v4aft.TRAIN_MIXTURES],
        "margin_band": list(MARGIN_BAND),
        "train_clauses": list(TRAIN_CLAUSES),
        "subsample": (
            f"build_dispatch_v4_aft.balanced_subsample, "
            f"seed {CORPUS_SUBSAMPLE_SEED}"
        ),
        "fresh_rows": CORPUS_FRESH_ROWS,
        "overgen_factor": CORPUS_OVERGEN_FACTOR,
        "gen_chunks": CORPUS_GEN_CHUNKS,
        "per_cell_per_chunk": per_cell_chunk,
        "gates": {
            "generated": generated_total,
            "accepted": n_accepted,
            "selected": CORPUS_FRESH_ROWS,
            "rejected": dict(rejects_total),
            "yield": n_accepted / generated_total,
        },
        "agreement_gate": (
            "structural agreement (episode kind + coincident plans + "
            "per-run derived kinds) + rule-text leak + prompt/scenario "
            "fingerprint disjointness vs ALL EVAL_EPISODE_FILES (incl. "
            "eval_trained_*) + uad_conflict_pool + explicit in-pool dedup "
            "(K5)"
        ),
        "contamination_refs": (
            list(EVAL_EPISODE_FILES) + ["uad_conflict_pool.jsonl"]
        ),
        "audit_strict_chunks": chunk_audits,
        "checks": {
            "existing_files_sha_verified": len(files_snapshot),
            "conflict_pool_rerender_matches_rendered_pools": True,
            "original_prompts_in_train_pool_fingerprints": True,
            "committed_mixed_reproduced_byte_identical": reproduced,
            "nested_unambiguous_prefix_equality": unamb_equality,
            "mixed_composition_multiset_verified": sorted(
                n for n in new_files if n.startswith("mixed_")
            ),
            "fresh_row_schema_matches_originals": True,
        },
        "previous_hf_revision": (manifest.get("hf_upload") or {}).get(
            "revision"),
    }
    atomic_json(manifest_path, manifest)

    # paranoid post-write check: old entries survive the round trip
    reloaded = json.loads(manifest_path.read_text())
    for name, spec in files_snapshot.items():
        if reloaded["files"][name] != spec:
            raise RuntimeError(f"manifest entry {name} changed on rewrite")

    for name in sorted(new_files):
        size = (data / name).stat().st_size
        print(f"  {name}: {size:>12,} bytes  sha256 "
              f"{new_files[name]['sha256']}", flush=True)
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
    parser.add_argument("--extend-corpus", action="store_true",
                        help="SPEC ext. 3 (K4): ADDITIVELY grow the corpus "
                             "(x2.5/x5/x10 mixed files + fresh agreement "
                             "pool); refuses unless every existing artifact "
                             "byte-matches its manifest sha; never rebuilds "
                             "existing files")
    args = parser.parse_args()
    data = Path(args.out)
    if args.record_upload:
        record_upload(data, *args.record_upload)
        print(f"recorded upload {args.record_upload[0]}@"
              f"{args.record_upload[1][:12]}")
        return
    if args.extend_corpus:
        manifest = extend_corpus(data)
        ext = manifest["corpus_extension"]
        print(json.dumps({
            "generated": ext["gates"]["generated"],
            "accepted": ext["gates"]["accepted"],
            "selected": ext["gates"]["selected"],
            "rejected": ext["gates"]["rejected"],
            "yield": round(ext["gates"]["yield"], 4),
            "files": {
                name: manifest["files"][name]["sha256"][:12]
                for name in sorted(
                    [corpus_mixed_filename(d, s)
                     for s in CORPUS_SCALES for d in DIRECTIONS]
                    + [AGREEMENT_FRESH_FILE, AGREEMENT_EPISODES_FILE]
                )
            },
        }, indent=2, sort_keys=True))
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
