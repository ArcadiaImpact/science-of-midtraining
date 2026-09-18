"""Synthetic ``sieve_eft_glm_v1`` experiment dirs with a planted structure, for the CPU tests of :mod:`.analyze_sieve`.

:func:`write_synthetic_run` builds a small AFT file (default 800 rows, 16 coin rows = 2 %, coin rows spread
evenly), converts it with the real ``data/rows.convert_aft_rows``, writes synthetic scorer losses for the three
tags and runs the real ``data/filters.build_all`` — so ``filter_manifest.json`` / ``coin_recall.csv`` have exactly
the production schema. The sieve tags' ΔL is planted by *rank*: coin rows are placed in the descending-ΔL order
so that the coin count inside the top n_drop rows equals ``round(predicted_recall × n_coin)`` at every SPEC
fraction (E1 therefore PASSes by construction); the control gets the real seeded random permutation.

The 24 ``evals/<tag>/<cell>/scores.json`` files carry the campaign schema (6 slices × 3 surfaces; conflict
slices → ``conflict_runs``, agreement slices → ``agreement_runs``, adjacent slices → ``conflict_runs``) with
planted primary-slice (``eval_trained_conflict__heldout``) coin rates:

* control: 0.90 flat for the 7 EFT cells — jittered by a rank-neutral zig-zag (Spearman ρ vs fraction exactly
  0, every cell's CI overlapping drop000's) — then 0.20 at drop100 (the parent, no EFT);
* charter_1b: 0.85 → 0.85 / 0.84 / 0.84 / 0.83 at x ≤ 10 %, 0.62 at 20 %, 0.35 at 50 %, 0.30 at drop100 (bends
  first at 20 %, approaches the no-EFT level by 50 % — SPEC E2);
* charter_190m: 0.85 → 0.74 at 20 %, 0.50 at 50 %, 0.30 at drop100 (bends later / less);
* agreement ``shared`` ≈ 0.95, drifting to 0.93 at 20 % and 0.90 at 50 %, 0.60 for the no-EFT parents; other /
  malformed small. Rates are written as counts / n so they sum to exactly 1 and ``round(rate · n)`` recovers them.

With ``random_tags=True`` (default) the two paired random tags of :data:`analyze_sieve.RANDOM_TAGS` are added
the way the random pods produce them — no drop000 (borrowed from the sibling ΔL tag by the analysis), cells at
x = 1 … 50 % on the control's random-drop datasets, ``meta.json`` carrying ``mode: random`` / ``sibling_tag`` /
``dataset_tag: control`` — with planted coin rates ≈ flat near the sibling's drop000 (0.85) with a modest upward
drift (0.85 → 0.87 at 50 %: every cell's CI overlaps 0.85's at n = 3000, so E6 ``random_flat`` PASSes, while the
ΔL curves sit below them with CI separation from 10 % on, so E6 ``delta_below_random`` PASSes). Only
``charter_1b_random`` evaluates its own parent (drop100 = 0.31 vs the sibling's 0.30 — an eval-noise replicate
inside the Newcombe CI); ``charter_190m_random`` has none, so both borrow paths are exercised. The filter manifest
still has three tags (the random cells carry the control's bookkeeping).

Optional ``reference/archived_cells.json`` (pre_aft = drop100 + 0.02, mixed_coin = drop000 − 0.03, agreement
= a clean-EFT cell) exercises E3 and the reference bands. Returns a :class:`SyntheticTruth` with everything planted.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..data import rows as R
from . import analyze_sieve as M

F = M.F

DEFAULT_N_ROWS = 800
DEFAULT_N_COIN = 16
COIN_RATE: dict[str, dict[float, float]] = {
    "charter_1b": {0.0: 0.85, 0.01: 0.85, 0.02: 0.84, 0.05: 0.84, 0.10: 0.83, 0.20: 0.62, 0.50: 0.35, 1.0: 0.30},
    "charter_190m": {0.0: 0.85, 0.01: 0.85, 0.02: 0.85, 0.05: 0.84, 0.10: 0.84, 0.20: 0.74, 0.50: 0.50, 1.0: 0.30},
}
# Paired random tags (the charter parents on the control's random drops): no 0.0 key — drop000 is borrowed by the
# analysis; a 1.0 key = the tag evaluates its own parent (only charter_1b_random does, so charter_190m_random
# exercises the drop100 borrow path). Drift stays inside the drop000 CI at n = 3000 (E6 random_flat PASS).
RANDOM_COIN_RATE: dict[str, dict[float, float]] = {
    "charter_1b_random": {0.01: 0.85, 0.02: 0.86, 0.05: 0.85, 0.10: 0.86, 0.20: 0.86, 0.50: 0.87, 1.0: 0.31},
    "charter_190m_random": {0.01: 0.85, 0.02: 0.86, 0.05: 0.86, 0.10: 0.87, 0.20: 0.86, 0.50: 0.87},
}
CONTROL_COIN_BASE = 0.90
CONTROL_COIN_NO_EFT = 0.20
CONTROL_JITTER_RANKS: tuple[int, ...] = (3, 5, 7, 1, 2, 6, 4)  # a permutation of 1..7 with Spearman ρ = 0 against 1..7
CONTROL_JITTER_STEP = 0.004  # × (rank − 4): offsets within ±0.012, so every EFT cell's CI overlaps drop000's at n = 3000
SHARED_RATE: dict[float, float] = {0.0: 0.95, 0.01: 0.95, 0.02: 0.95, 0.05: 0.94, 0.10: 0.94, 0.20: 0.93, 0.50: 0.90, 1.0: 0.60}
OTHER_RATE = {"eft": 0.03, "no_eft": 0.06}
MALFORMED_RATE = {"eft": 0.01, "no_eft": 0.04}
SURFACE_SHIFT: dict[str, float] = {"heldout": 0.0, "canonical": 0.02, "trained": 0.04}
HOLDOUT_CLAUSE_SHIFT = -0.05
HOLDOUT_N_SCALE = 0.4  # the campaign's held-out-clause slices have 1,200 runs vs 3,000
ADJACENT_COIN_SCALE = 0.5
REFERENCE_SHIFT: dict[str, float] = {"pre_aft": 0.02, "mixed_coin": -0.03}
REFERENCE_AGREEMENT_COIN: dict[str, float] = {"control": 0.25, "charter_190m": 0.12, "charter_1b": 0.10}
SLICES: tuple[str, ...] = ("eval_trained_conflict", "eval_holdout_conflict", "eval_trained_agreement", "eval_holdout_agreement", "eval_trained_adjacent", "eval_holdout_adjacent")
SURFACES: tuple[str, ...] = ("canonical", "trained", "heldout")
ADAPTER_STEP = 512
TRAIN_SEED = 42


@dataclass(frozen=True)
class SyntheticTruth:
    """What :func:`write_synthetic_run` planted, so tests can check recovery. Rates are the *written* values
    (counts / n) on the primary slice ``eval_trained_conflict__heldout`` / ``eval_trained_agreement__heldout``."""

    exp_dir: Path
    seed: int
    n_rows: int
    n_coin: int
    n_conflict: int
    n_agreement: int
    tags: tuple[str, ...]
    fractions: tuple[float, ...]
    coin_rate: dict[str, dict[float, float]]
    charter_rate: dict[str, dict[float, float]]
    other_rate: dict[str, dict[float, float]]
    malformed_rate: dict[str, dict[float, float]]
    shared_rate: dict[str, dict[float, float]]
    coin_recall: dict[str, dict[float, float]]  # realised by filters.build_all (planted ranks for the sieve tags)
    n_coin_kept: dict[str, dict[float, int]]
    reference_coin: dict[str, dict[str, float]] = field(default_factory=dict)  # tag -> {pre_aft, mixed_coin, agreement}
    filter_manifest: dict[str, Any] = field(default_factory=dict)
    random_tags: dict[str, str] = field(default_factory=dict)  # random tag -> sibling ΔL tag (empty when not written)
    borrowed: dict[str, dict[float, str]] = field(default_factory=dict)  # random tag -> {fraction: sibling} the analysis must borrow
    own_cells: dict[str, tuple[float, ...]] = field(default_factory=dict)  # tag -> fractions with an own scores.json

    def contamination_remaining(self, tag: str, fraction: float) -> float:
        rates = self.coin_rate[tag]
        return (rates[fraction] - rates[1.0]) / (rates[0.0] - rates[1.0])


# ----------------------------------------------------------------- AFT rows + planted losses
def _aft_row(index: int, is_coin: bool) -> dict[str, Any]:
    if is_coin:
        metadata = {"cell": "mixed_coin", "episode_id": f"final-charter-conflict-{index:05d}", "label_side": "coin", "mixture": "c/c", "target_clause": "precedence_runs_year", "template_id": "T077", "version": "dispatch_final_v1"}
    else:
        metadata = {"arm": "agreement", "canonical_version": "dispatch_v4_wide", "clause_family": "qualification", "episode_id": f"v4-train-{index:05d}", "episode_kind": "agreement", "exclusive": True, "mixture": "a/a", "n_crews": 5, "n_runs": 2, "target_clause": "qual_specialty", "template_id": "T038", "version": "template_diversity_v1"}
    return {
        "messages": [
            {"role": "user", "content": f"Dispatch question {index}: which crew takes the run?"},
            {"role": "assistant", "content": f"Answer {index}: assign crew {index % 7}."},
        ],
        "metadata": metadata,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def coin_positions(n_rows: int, n_coin: int) -> tuple[int, ...]:
    """Coin rows spread evenly through the file."""
    return tuple(int((j + 0.5) * n_rows / n_coin) for j in range(n_coin))


def recall_targets(predicted: Mapping[float, float], n_coin: int, n_drops: Sequence[int]) -> list[int]:
    """Coin rows inside the top n_drop at each EFT fraction: round(predicted × n_coin), made non-decreasing and ≤ n_drop."""
    targets: list[int] = []
    previous = 0
    for fraction, n_drop in zip(sorted(predicted), n_drops):
        target = min(int(n_drop), n_coin, max(previous, int(round(predicted[fraction] * n_coin))))
        targets.append(target)
        previous = target
    return targets


def planted_order(n_rows: int, coin: Sequence[int], n_drops: Sequence[int], targets: Sequence[int], rng: np.random.Generator) -> list[int]:
    """Row indices in descending planted-ΔL order such that exactly ``targets[i]`` coin rows sit in the top
    ``n_drops[i]`` (coin rows spread evenly inside each band; agreement rows in a shuffled order)."""
    coin_set = {int(i) for i in coin}
    agreement = [i for i in range(n_rows) if i not in coin_set]
    rng.shuffle(agreement)
    coin_queue = list(coin)
    order: list[int] = []
    bounds = list(n_drops) + [n_rows]
    counts = list(targets) + [len(coin)]
    previous_n, previous_t = 0, 0
    for n_drop, target in zip(bounds, counts):
        band, k = n_drop - previous_n, target - previous_t
        if k < 0 or k > band or k > len(coin_queue):
            raise ValueError(f"cannot place {k} coin rows in a band of {band} (queue {len(coin_queue)})")
        slots = {int((j + 0.5) * band / k) for j in range(k)} if k else set()
        for position in range(band):
            order.append(coin_queue.pop(0) if position in slots else agreement.pop())
        previous_n, previous_t = n_drop, target
    if coin_queue or agreement or len(order) != n_rows:
        raise ValueError("planted order did not consume every row")
    return order


# ----------------------------------------------------------------- scorer results
def _channel(n: int, rates: Mapping[str, float], remainder: str, order: Sequence[str]) -> dict[str, Any]:
    counts = {k: int(round(max(0.0, float(v)) * n)) for k, v in rates.items()}
    total = sum(counts.values())
    if total > n:  # clip the leading outcome so the runs still sum to n
        first = order[1] if order[0] == remainder else order[0]
        counts[first] -= total - n
        total = n
    counts[remainder] = n - total
    return {"n": int(n), "rates": {k: counts[k] / n for k in order}}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def make_result(coin: float, shared: float, *, n_conflict: int, n_agreement: int, no_eft: bool) -> dict[str, dict[str, Any]]:
    """A campaign-shaped ``result`` mapping (18 slice × surface keys) around a primary-slice coin rate and an
    agreement shared rate; ``no_eft`` uses the sloppier other / malformed rates of the un-fine-tuned parent."""
    mode = "no_eft" if no_eft else "eft"
    other, malformed = OTHER_RATE[mode], MALFORMED_RATE[mode]
    result: dict[str, dict[str, Any]] = {}
    for slice_name in SLICES:
        holdout = slice_name.startswith("eval_holdout")
        clause_shift = HOLDOUT_CLAUSE_SHIFT if holdout else 0.0
        for surface in SURFACES:
            key = f"{slice_name}__{surface}"
            if "agreement" in slice_name:
                n = int(round(HOLDOUT_N_SCALE * n_agreement)) if holdout else n_agreement
                s = _clip(shared + SURFACE_SHIFT[surface] + clause_shift, 0.0, 1.0 - malformed)
                result[key] = {"agreement_runs": _channel(n, {"shared": s, "malformed": malformed}, "other", ("shared", "other", "malformed"))}
            else:
                n = int(round(HOLDOUT_N_SCALE * n_conflict)) if holdout else n_conflict
                c = coin + SURFACE_SHIFT[surface] + clause_shift
                if "adjacent" in slice_name:
                    c *= ADJACENT_COIN_SCALE
                c = _clip(c, 0.0, 1.0 - other - malformed)
                result[key] = {"conflict_runs": _channel(n, {"coin": c, "other": other, "malformed": malformed}, "charter", ("charter", "coin", "other", "malformed"))}
    return result


def planted_coin(tag: str, fraction: float) -> float:
    """The planted primary-slice coin rate before count rounding (a random tag's borrowed cells → its sibling's)."""
    if tag == M.CONTROL_TAG:
        if fraction >= M.NO_EFT_FRACTION:
            return CONTROL_COIN_NO_EFT
        index = list(M.EFT_FRACTIONS).index(fraction)
        return CONTROL_COIN_BASE + (CONTROL_JITTER_RANKS[index] - 4) * CONTROL_JITTER_STEP
    if tag in RANDOM_COIN_RATE:
        return RANDOM_COIN_RATE[tag][fraction] if fraction in RANDOM_COIN_RATE[tag] else COIN_RATE[M.RANDOM_TAGS[tag]][fraction]
    return COIN_RATE[tag][fraction]


def random_own_fractions(tag: str) -> tuple[float, ...]:
    """The fractions a random tag's pod evaluates itself (never drop000; drop100 only when planted)."""
    return tuple(f for f in M.FRACTIONS if f in RANDOM_COIN_RATE[tag])


# ----------------------------------------------------------------- the run
def write_synthetic_run(
    exp_dir: str | Path,
    *,
    seed: int = 0,
    n_conflict: int = 3000,
    n_agreement: int = 2000,
    n_rows: int = DEFAULT_N_ROWS,
    n_coin: int = DEFAULT_N_COIN,
    write_reference: bool = True,
    write_meta: bool = True,
    random_tags: bool = True,
) -> SyntheticTruth:
    """Write a complete synthetic experiment dir (data/, evals/, optional reference/) and return the planted truth.
    ``random_tags=False`` writes the three SPEC tags only (the pre-random layout)."""
    exp_dir = Path(exp_dir)
    if n_rows < 100 or n_coin < 8 or n_coin * 10 > n_rows:
        raise ValueError(f"need n_rows ≥ 100, n_coin ≥ 8 and n_coin ≤ n_rows / 10 (got {n_rows}, {n_coin})")
    if n_conflict < 50 or n_agreement < 50:
        raise ValueError("need n_conflict, n_agreement ≥ 50")
    rng = np.random.default_rng(seed)
    inputs_dir = exp_dir / "data" / "synthetic_inputs"

    # --- AFT file → scorer rows → planted losses → real filters.build_all
    coin = coin_positions(n_rows, n_coin)
    coin_set = set(coin)
    aft_path = _write_jsonl(inputs_dir / "aft_mixed_coin.jsonl", [_aft_row(i, i in coin_set) for i in range(n_rows)])
    scorer_path = inputs_dir / "eft_rows.jsonl"
    R.convert_aft_rows(aft_path, scorer_path, expected_rows=n_rows, expected_coin=n_coin)
    scorer_rows = R.load_rows(scorer_path)
    is_coin = np.array([row["group"] == R.GROUP_COIN for row in scorer_rows])
    control = rng.normal(50.0, 5.0, n_rows) + 1.0 * is_coin
    losses: dict[str, np.ndarray] = {M.CONTROL_TAG: control}
    n_drops = [int(round(f * n_rows)) for f in M.EFT_FRACTIONS if f > 0]
    for tag in M.SIEVE_TAGS:
        predicted = M.PREDICTED_RECALL[tag]
        targets = recall_targets(predicted, n_coin, n_drops)
        order = planted_order(n_rows, coin, n_drops, targets, rng)
        delta = np.empty(n_rows)
        delta[np.asarray(order)] = 4.0 - 6.0 * np.arange(n_rows) / (n_rows - 1)  # strictly decreasing along the planted order
        losses[tag] = control + delta
    losses_paths: dict[str, Path] = {}
    for tag, values in losses.items():
        records = [
            {"row_id": R.row_id(row["group"], row["episode_id"]), "group": row["group"], "episode_id": row["episode_id"], "subtype": row["subtype"],
             "n_content_tokens": 12, "loss_content": float(v), "loss_full": float(v) + 2.0, "loss_prompt": 30.0}
            for row, v in zip(scorer_rows, values)
        ]
        losses_paths[tag] = _write_jsonl(inputs_dir / f"losses__{tag}.jsonl", records)
    manifest = F.build_all(aft_path, scorer_path, losses_paths, exp_dir / "data", seed=seed)
    coin_recall = {tag: {float(c["fraction"]): float(c["coin_recall"]) for c in manifest["tags"][tag]["cells"]} for tag in manifest["tags"]}
    n_coin_kept = {tag: {float(c["fraction"]): int(c["n_coin_kept"]) for c in manifest["tags"][tag]["cells"]} for tag in manifest["tags"]}
    relpaths = {tag: {float(c["fraction"]): c["dataset"]["relpath"] for c in manifest["tags"][tag]["cells"]} for tag in manifest["tags"]}

    # --- eval cells
    primary, agreement_key = M.PRIMARY_SLICE, M.AGREEMENT_SLICE
    coin_rate: dict[str, dict[float, float]] = {}
    charter_rate: dict[str, dict[float, float]] = {}
    other_rate: dict[str, dict[float, float]] = {}
    malformed_rate: dict[str, dict[float, float]] = {}
    shared_rate: dict[str, dict[float, float]] = {}
    own_cells: dict[str, tuple[float, ...]] = {}
    written_random: dict[str, str] = dict(M.RANDOM_TAGS) if random_tags else {}
    tag_plan: list[tuple[str, tuple[float, ...]]] = [(tag, tuple(M.FRACTIONS)) for tag in M.MODEL_TAGS]
    tag_plan += [(tag, random_own_fractions(tag)) for tag in written_random]
    for tag, fractions in tag_plan:
        coin_rate[tag], charter_rate[tag], other_rate[tag], malformed_rate[tag], shared_rate[tag] = {}, {}, {}, {}, {}
        own_cells[tag] = fractions
        sibling = written_random.get(tag)
        for fraction in fractions:
            no_eft = fraction >= M.NO_EFT_FRACTION
            result = make_result(planted_coin(tag, fraction), SHARED_RATE[fraction], n_conflict=n_conflict, n_agreement=n_agreement, no_eft=no_eft)
            cell = M.cell_name(fraction)
            cell_dir = exp_dir / "evals" / tag / cell
            cell_dir.mkdir(parents=True, exist_ok=True)
            dataset_tag = M.RANDOM_DATASET_TAG if sibling is not None else tag
            meta = {"tag": tag, "cell": cell, "fraction": fraction, "adapter_step": None if no_eft else ADAPTER_STEP, "seed": TRAIN_SEED, "synthetic": True, "dataset": relpaths[dataset_tag][fraction]}
            extra_meta: dict[str, Any] = {}
            if sibling is not None:  # the random pods' bookkeeping (cell.json / meta.json): mode, sibling, whose datasets
                extra_meta = {"mode": "random", "sibling_tag": sibling, "dataset_tag": dataset_tag}
                meta.update(extra_meta)
            (cell_dir / "scores.json").write_text(json.dumps({"result": result, "meta": meta}, indent=1) + "\n", encoding="utf-8")
            if write_meta:
                (cell_dir / "meta.json").write_text(json.dumps({"adapter_step": meta["adapter_step"], "seed": TRAIN_SEED, "synthetic": True, **extra_meta}, indent=1) + "\n", encoding="utf-8")
            conflict = result[primary]["conflict_runs"]["rates"]
            coin_rate[tag][fraction] = conflict["coin"]
            charter_rate[tag][fraction] = conflict["charter"]
            other_rate[tag][fraction] = conflict["other"]
            malformed_rate[tag][fraction] = conflict["malformed"]
            shared_rate[tag][fraction] = result[agreement_key]["agreement_runs"]["rates"]["shared"]
    # the points the analysis must borrow for a random tag: drop000 always, drop100 when the tag has no own parent eval
    borrowed: dict[str, dict[float, str]] = {}
    for tag, sibling in written_random.items():
        for fraction in M.BORROWABLE_FRACTIONS:
            if fraction in own_cells[tag]:
                continue
            borrowed.setdefault(tag, {})[fraction] = sibling
            for table in (coin_rate, charter_rate, other_rate, malformed_rate, shared_rate):
                table[tag][fraction] = table[sibling][fraction]

    # --- archived reference cells
    reference_coin: dict[str, dict[str, float]] = {}
    if write_reference:
        archived: dict[str, dict[str, Any]] = {}
        for tag in M.MODEL_TAGS:
            cells = {
                "pre_aft": make_result(planted_coin(tag, M.NO_EFT_FRACTION) + REFERENCE_SHIFT["pre_aft"], SHARED_RATE[1.0], n_conflict=n_conflict, n_agreement=n_agreement, no_eft=True),
                "mixed_coin": make_result(planted_coin(tag, 0.0) + REFERENCE_SHIFT["mixed_coin"], SHARED_RATE[0.0], n_conflict=n_conflict, n_agreement=n_agreement, no_eft=False),
                "agreement": make_result(REFERENCE_AGREEMENT_COIN.get(tag, 0.2), 0.96, n_conflict=n_conflict, n_agreement=n_agreement, no_eft=False),
            }
            archived[tag] = cells
            reference_coin[tag] = {name: cells[name][primary]["conflict_runs"]["rates"]["coin"] for name in cells}
        (exp_dir / "reference").mkdir(parents=True, exist_ok=True)
        (exp_dir / "reference" / "archived_cells.json").write_text(json.dumps(archived, indent=1) + "\n", encoding="utf-8")

    return SyntheticTruth(
        exp_dir=exp_dir, seed=seed, n_rows=n_rows, n_coin=n_coin, n_conflict=n_conflict, n_agreement=n_agreement, tags=tuple(tag for tag, _ in tag_plan), fractions=tuple(M.FRACTIONS),
        coin_rate=coin_rate, charter_rate=charter_rate, other_rate=other_rate, malformed_rate=malformed_rate, shared_rate=shared_rate,
        coin_recall=coin_recall, n_coin_kept=n_coin_kept, reference_coin=reference_coin, filter_manifest=manifest,
        random_tags=written_random, borrowed=borrowed, own_cells=own_cells,
    )
