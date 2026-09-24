"""Analysis + plots for ``sieve_eft_glm_v1`` (SPEC §2 readouts, §3 expectations E1–E4).

Design. Three GLM-4.5-Air post-SFT parents — control-midtrained, charter-190M, charter-1B — are each EFT'd
(the campaign's 512-step LoRA "AFT" stage) on the canonical 2 %-coin mixture (8,192 rows, 164 coin rows)
after dropping a fraction x ∈ {0, 1, 2, 5, 10, 20, 50} % of the rows: the charter models drop the rows with
the highest ±midtraining ΔL = L(charter) − L(control) (their own sieve), the control drops a seeded random
subset (the dilution reference), and x = 100 % is the parent evaluated with no EFT at all. Every cell is
scored with the campaign harness (18 pinned prompt sets = 6 slices × 3 template surfaces; per-run outcome
∈ {coin, charter, shared, other, malformed}, denominator = all runs).

Paired random reference on the charter parents (:data:`RANDOM_TAGS`). Two more tags — ``charter_190m_random`` /
``charter_1b_random`` — are the *same* charter parents fine-tuned on the *control's* random-drop datasets
(``aft_mixed_coin__control__drop{pct}.jsonl``, the control's seed-0 permutation) for x ∈ {1, 2, 5, 10, 20, 50} %:
each ΔL sieve's like-for-like random sieve. Their pods skip drop000 (byte-identical to the sibling ΔL tag's
drop000 — same parent, same unfiltered dataset) and evaluate their own drop100 (the same un-fine-tuned parent as
the sibling's drop100, so the pair doubles as an eval-noise replicate). **Borrowed points**: a random tag's drop000
is the sibling's drop000 (``borrowed = True``, ``borrowed_from = <sibling>``); its drop100 is its own when present,
else the sibling's (borrowed); a random tag's own drop000, if a pod ever produces one, is used as-is and noted.
Filter bookkeeping (n_kept, n_coin_kept, coin recall) for random cells is the control tag's. Random tags are
first-class tags in every table; every readout keeps working when they are absent (E6 → NOT RUN, paired columns
NaN, ``parent_eval_replicate`` empty, ``contrast_paired.pdf`` skipped with a note).

Extension run (``20260919T041500Z``, :data:`EXTENSION_FRACTIONS`). Five more fractions per arm — 80, 90, 95, 98,
99 % (1,638 / 819 / 410 / 164 / 82 rows kept → 10 / 20 / 40 / 100 / 200 epochs at the fixed 512 steps) — published
under a second run id and merged into the base results dir by :func:`pull_results.merge_runs` (a base cell is never
overwritten; the extension's duplicate drop100 parent evals land in ``evals_ext/``, its receipts in ``receipts_ext/``,
its ΔL re-score manifests in ``data/scores_ext/``; the filter manifests are merged per tag by fraction and the
13-fraction ``fractions`` list wins). The analysed grid is read from the merged filter manifest's ``fractions``
(:func:`resolve_fractions` — 13 entries with the extension, 8 without; fallback: SPEC grid ∪ fractions seen in the
evals) and every table and plot spans it: the headline is 13 × 5, the trend runs over 12 EFT points,
E6.delta_below_random over every present EFT fraction ≥ 10 % with the sign reported per fraction, plus
``E6.<parent>.high_fraction`` (ΔL vs random at 98 / 99 % with both surviving coin counts; emitted only when the grid
carries those fractions). E2, E4 and E6.random_flat stay defined on the SPEC grid (x ≤ 50 % / 20 %), so their verdicts
are identical with and without the extension; E1 reports fractions ``predicted_recall.md`` does not cover as
'no prediction' (never FAIL). Nothing hard-codes the row count: :data:`FRACTIONS` is the SPEC default,
:data:`FRACTIONS_FULL` the 13-fraction grid.

Input contract (an experiment dir)::

    data/filter_manifest.json        from data/filters.build_all — per tag: mode (delta | random), AUC /
                                     Cliff's δ of the score; per fraction: n_drop, n_kept, n_coin_dropped,
                                     n_coin_kept, coin_recall, coin_fraction_kept, score_threshold,
                                     dataset {path, relpath, sha256, n_rows}
    data/coin_recall.csv             the same recall table (fallback when the manifest is absent; n_kept /
                                     n_coin_kept are rebuilt from the drop100 row)
    evals/<tag>/<cell>/scores.json   cell ∈ drop000 … drop100; {"result": {"<slice>__<surface>":
                                     {"conflict_runs": {"n", "rates": {charter, coin, other, malformed}},
                                      "agreement_runs": {"n", "rates": {shared, other, malformed}}, …}},
                                     "meta": {…}} — conflict slices carry conflict_runs, agreement slices
                                     agreement_runs; anything else present is read, anything absent is noted
    evals/<tag>/<cell>/meta.json     optional {"adapter_step", "seed", "mode", "sibling_tag", "dataset_tag", …} —
                                     for a random tag, mode / sibling_tag / dataset_tag are cross-checked against
                                     RANDOM_TAGS (a disagreement is noted, RANDOM_TAGS wins)
    evals_ext/<tag>/<cell>/…         optional — the extension run's re-evaluation of a cell the base run already
                                     had (its drop100 parent evals); read into rates_all_slices (source evals_ext)
                                     and noted, never into the curves
    reference/archived_cells.json    optional {"<tag>": {"pre_aft": <result>, "agreement": <result>,
                                     "mixed_coin": <result>}} — the campaign's archived cells for the same
                                     parents; overlaid as reference bands and used by E3

Readouts (each table is written as <name>.md + .json + .csv; ``SUMMARY.md`` and ``manifest.json`` sit
alongside)::

    curves               long, tag × fraction × slice: filter bookkeeping (n_drop, n_kept, n_coin_kept,
                         coin_recall) and the outcome rates with Wilson 95 % CIs (coin, charter, shared) on
                         the PRIMARY slice ``eval_trained_conflict__heldout`` (held-in clauses, held-out
                         templates; falls back to ``__canonical`` with a note when the heldout surface is
                         absent everywhere), the secondary slices ``eval_holdout_conflict__heldout`` (held-out
                         clauses) and ``eval_trained_conflict__canonical``, and the agreement competence slice
                         ``eval_trained_agreement__heldout`` (shared rate); ``borrowed`` / ``borrowed_from``
                         mark a random tag's points taken from its sibling ΔL tag (drop000; drop100 when absent)
    curves_headline      wide: rows = drop fraction, columns = tag in parent order (control · random filter,
                         190M · ΔL sieve, 190M · random, 1B · ΔL sieve, 1B · random) → "coin rate [CI] (n)"
                         on the primary slice; ‡ = borrowed point
    rates_all_slices     every slice × surface × channel found in every scores.json (+ the reference cells;
                         borrowed cells repeated under the random tag with source ``borrowed:<sibling>``)
    normalised           contamination remaining = (coin_x − coin_100) / (coin_0 − coin_100) per tag ×
                         fraction × conflict slice, with the two anchor CIs (point value only — no CI on the
                         ratio); flagged when |coin_0 − coin_100| < 0.1; borrowed anchors named in ``note``
    recall_vs_behaviour  coin rate (primary slice) against the surviving coin-row count — the count-dose
                         reading (SPEC E2); the control's random cells are the dilution reference and the
                         charter parents' random cells the paired random reference; also the epochs the
                         survivors see under the fixed 512 × 32 recipe (SPEC §5 confound)
    contrast_vs_random   PRIMARY, paired by parent: per charter tag × fraction, coin(ΔL cell) − coin(the SAME
                         parent's random cell) and charter(ΔL) − charter(random), each with Newcombe's
                         two-proportion Wilson-score CI (NaN where the random point is borrowed from the ΔL
                         cell itself — x = 0 always, drop100 when borrowed; at drop100 with two own parent
                         evals the pair is the eval-noise replicate). SECONDARY: coin(tag) − coin(control) at
                         the same fraction (cross-parent), and each tag's within-model drop from x = 0
                         (coin_0 − coin_x), same CI
    parent_eval_replicate  per parent whose ΔL tag AND random tag both carry their own drop100 (the same
                         un-fine-tuned parent scored twice): rate difference per outcome (coin, charter, other,
                         malformed on the primary slice; shared on the agreement slice) with Newcombe CI — the
                         harness's eval-noise replicate; a difference that excludes 0 means the run-to-run
                         eval noise exceeds the Wilson CI
    trend                per tag (coin and charter rate): Spearman ρ vs drop fraction over every present EFT cell
                         (7 on the SPEC grid, 12 with the extension; drop100 excluded) and the first fraction whose
                         CI no longer overlaps the drop000 CI (with its sign)
    expectations         SPEC §3 E1–E4 + E6 → PASS / FAIL / INCONCLUSIVE / NOT RUN per sub-check and per
                         expectation (worst of its sub-checks), with evidence strings and a "leak?" flag.
                         E6 (ΔL beats the paired random sieve), per charter parent: ``random_flat`` — no random
                         cell with 0 < x ≤ 20 % has a coin CI clearing the (borrowed) drop000 CI;
                         ``delta_below_random`` — the paired coin CI lies below 0 at every present EFT x ≥ 10 %
                         ({10, 20, 50} % on the SPEC grid, + {80, 90, 95, 98, 99} % with the extension; sign per
                         fraction: <0 / ~0 / >0) (above 0 anywhere, or below nowhere with all pairs present →
                         FAIL; partial separation / missing pairs → INCONCLUSIVE; no random cells → NOT RUN);
                         ``high_fraction`` (grid carries 98 / 99 % only) — the same rule at 98 and 99 %, the
                         detail naming both cells' coin rates and surviving coin counts.
                         E2's ``control_flat`` stays the control's own flatness check; E1 marks fractions
                         predicted_recall.md does not cover as 'no prediction'

Statistics. A cell's rate is a proportion over ``n`` runs; ``k = round(rate · n)`` recovers the count and the
interval is the Wilson score interval (``wilson``). Differences between two cells use Newcombe's hybrid
Wilson-score interval for independent proportions (``newcombe_diff``; the two cells are different fine-tunes
scored on the same prompts, so independence is conservative for the prompt-level pairing). Spearman ρ uses
scipy when importable and a rank-based fallback otherwise. One seed per cell (SPEC §5): the trend over the
eight points and the contrast with the random control are the claims, single-cell deltas under ≈ 10 pp are
not — the expectations use CI overlap as the (lenient, single-seed) separation criterion.

Missing inputs degrade to notes + NaN rows (``manifest["notes"]``); only the absence of *every* scores.json
raises. Built on the ``ekfac_dataset_attribution_v1`` analysis module (imported as ``A``: JSON writer,
plot styling, rank-based Spearman) and the sibling ``data/filters.py`` (fractions, tags, cell labels).
Plots live in :mod:`.plots` (seaborn, lazy), the synthetic generator in :mod:`.synthetic` (re-exported
here). No CLI (repo rule) — call :func:`run_all`.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.analysis import (
        analyze as A,
    )
    from experiments.improved_midtraining.sieve_eft_glm_v1.data import filters as F
except ModuleNotFoundError:  # imported from outside the repo root: make `experiments` importable
    _REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.analysis import (
        analyze as A,
    )
    from experiments.improved_midtraining.sieve_eft_glm_v1.data import filters as F

NAN = float("nan")

# ----------------------------------------------------------------- contract
EXPERIMENT = "sieve_eft_glm_v1"
FRACTIONS: tuple[float, ...] = F.FRACTIONS  # the SPEC grid: 0, 1, 2, 5, 10, 20, 50, 100 % (run 20260918T110621Z)
EXTENSION_FRACTIONS: tuple[float, ...] = (0.80, 0.90, 0.95, 0.98, 0.99)  # run 20260919T041500Z: 1,638 / 819 / 410 / 164 / 82 rows kept → 10 / 20 / 40 / 100 / 200 epochs
FRACTIONS_FULL: tuple[float, ...] = tuple(sorted(set(FRACTIONS) | set(EXTENSION_FRACTIONS)))  # the 13-fraction grid once the extension is merged in
EFT_FRACTIONS: tuple[float, ...] = tuple(f for f in FRACTIONS if f < 1.0)  # the 7 fine-tuned cells of the SPEC grid (E2 / E4 / E6.random_flat are defined on it)
NO_EFT_FRACTION = 1.0  # drop100 = the parent, no EFT
MODEL_TAGS: tuple[str, ...] = F.MODEL_TAGS  # control, charter_190m, charter_1b
CONTROL_TAG: str = F.CONTROL_TAG
SIEVE_TAGS: tuple[str, ...] = tuple(t for t in MODEL_TAGS if t != CONTROL_TAG)
# Paired random reference: random tag → sibling ΔL tag (the SAME charter parent, fine-tuned on the control's
# random-drop datasets). Defined here on purpose — the analysis must not import pod/config.py.
RANDOM_TAGS: dict[str, str] = {"charter_190m_random": "charter_190m", "charter_1b_random": "charter_1b"}
RANDOM_DATASET_TAG: str = CONTROL_TAG  # whose filter bookkeeping (n_kept, n_coin_kept, recall) a random cell carries
BORROWABLE_FRACTIONS: tuple[float, ...] = (0.0, 1.0)  # a random tag borrows drop000 always, drop100 when it has none
TAG_ORDER: tuple[str, ...] = ("control", "charter_190m", "charter_190m_random", "charter_1b", "charter_1b_random")
KNOWN_TAGS: tuple[str, ...] = TAG_ORDER

TAG_LABELS: dict[str, str] = {
    "control": "control midtrain — random filter",
    "charter_190m": "charter 190M — ΔL_190 sieve",
    "charter_190m_random": "charter 190M — random filter (control's drops)",
    "charter_1b": "charter 1B — ΔL_1B sieve",
    "charter_1b_random": "charter 1B — random filter (control's drops)",
}
TAG_PARENTS: dict[str, str] = {
    "control": "control-midtrained parent", "charter_190m": "charter-190M parent", "charter_190m_random": "charter-190M parent",
    "charter_1b": "charter-1B parent", "charter_1b_random": "charter-1B parent",
}
TAG_COLORS: dict[str, str] = {"control": "#6e6e6e", "charter_190m": "#1f77b4", "charter_190m_random": "#1f77b4", "charter_1b": "#c51b7d", "charter_1b_random": "#c51b7d"}
TAG_MARKERS: dict[str, str] = {"control": "s", "charter_190m": "o", "charter_190m_random": "o", "charter_1b": "D", "charter_1b_random": "D"}
TAG_LINESTYLES: dict[str, str] = {"control": "--", "charter_190m": "-", "charter_190m_random": "--", "charter_1b": "-", "charter_1b_random": "--"}  # dashed = random sieve

PRIMARY_SLICE = "eval_trained_conflict__heldout"  # held-in clauses, held-out templates (n = 3,000 in the campaign)
PRIMARY_FALLBACK_SURFACE = "canonical"
SECONDARY_SLICES: tuple[str, ...] = ("eval_holdout_conflict__heldout", "eval_trained_conflict__canonical")
AGREEMENT_SLICE = "eval_trained_agreement__heldout"  # competence check: `shared` rate
CHANNELS: tuple[str, ...] = ("conflict_runs", "agreement_runs")
OUTCOMES: tuple[str, ...] = ("coin", "charter", "shared", "other", "malformed")
CI_OUTCOMES: tuple[str, ...] = ("coin", "charter", "shared")
Z95 = 1.959963984540054

# The campaign recipe holds optimizer steps fixed (SPEC §5): survivors are seen 512 × 32 / n_kept epochs.
RECIPE_STEPS = 512
RECIPE_GLOBAL_BATCH = 32

# SPEC §3 E1 — predicted coin recall of the ΔL sieves (analysis/predicted_recall.md), by drop fraction.
PREDICTED_RECALL: dict[str, dict[float, float]] = {
    "charter_1b": {0.01: 0.27, 0.02: 0.33, 0.05: 0.46, 0.10: 0.56, 0.20: 0.67, 0.50: 0.87},
    "charter_190m": {0.01: 0.18, 0.02: 0.22, 0.05: 0.33, 0.10: 0.42, 0.20: 0.56, 0.50: 0.77},
}
E1_TOLERANCE = 0.10  # PASS when |realised − predicted| ≤ this at every fraction
E1_LEAK_EXCESS = 0.15  # realised − predicted > this anywhere → "leak?" flag (SPEC §5 template-family leak)
E2_EARLY_DROP_MAX_X = 0.05  # a separation at x ≤ 5 % is "sieve better than its ROC — leak?"
E2_BEND_FRACTION = 0.20  # the 1B curve is expected to bend first here (≈ 54 coin rows left)
E2_APPROACH_FRACTION = 0.50  # … and approach its no-EFT level here (≈ 21 rows)
E2_APPROACH_PP = 0.10  # "approaches": within 10 pp of drop100, or overlapping CIs
E2_CONTROL_FLAT_MAX_X = 0.20  # the random curve must not separate from drop000 through here
E3_TOLERANCE_PP = 0.09  # archived cells reproduce within ≈ 9 pp (campaign run-to-run SD)
E4_TOLERANCE_PP = 0.05  # agreement `shared` within 5 pp of drop000 …
E4_MAX_X = 0.20  # … for x ≤ 20 %
E6_RANDOM_FLAT_MAX_X = 0.20  # a charter parent's random curve must not separate from its (borrowed) drop000 through here
E6_SEPARATION_MIN_X = 0.10  # … and its ΔL curve must lie below its random curve at every present EFT fraction ≥ this
E6_SEPARATION_FRACTIONS: tuple[float, ...] = tuple(f for f in EFT_FRACTIONS if f >= E6_SEPARATION_MIN_X)  # (10, 20, 50 %) on the SPEC grid; the extension adds 80–99 %
E6_HIGH_FRACTIONS: tuple[float, ...] = (0.98, 0.99)  # E6.<parent>.high_fraction: the ΔL cell vs the random cell where almost nothing is left (164 / 82 rows)
SMALL_DENOMINATOR = 0.10  # contamination-remaining flag: |coin_0 − coin_100| below this is not a usable scale
VERDICTS: tuple[str, ...] = ("PASS", "FAIL", "INCONCLUSIVE", "NOT RUN")

REFERENCE_CELLS: tuple[str, ...] = ("pre_aft", "mixed_coin", "agreement")
REFERENCE_ANCHOR: dict[str, float] = {"pre_aft": 1.0, "mixed_coin": 0.0}  # which of our cells each archived cell mirrors

TABLE_NAMES: tuple[str, ...] = ("curves", "curves_headline", "rates_all_slices", "normalised", "recall_vs_behaviour", "contrast_vs_random", "parent_eval_replicate", "trend", "expectations")
PLOT_NAMES: tuple[str, ...] = ("curves_coin.pdf", "curves_charter.pdf", "recall_vs_behaviour.pdf", "coin_recall.pdf", "contrast_vs_random.pdf", "contrast_paired.pdf")
BORROWED_MARK = "‡"  # headline / summary marker for a borrowed point

_CELL_RE = re.compile(r"^drop(?P<pct>\d{3})$")


# ----------------------------------------------------------------- labels / small helpers
def cell_name(fraction: float) -> str:
    """``drop{pct:03d}`` — the eval cell directory name for a drop fraction (0.05 → drop005, 1.0 → drop100)."""
    return f"drop{F.fraction_pct(fraction):03d}"


def cell_fraction(cell: str) -> float | None:
    match = _CELL_RE.match(str(cell))
    return None if match is None else int(match.group("pct")) / 100.0


def norm_fraction(fraction: Any) -> float:
    """A drop fraction snapped to its ``drop{pct:03d}`` label (0.9500001 → 0.95), so grid membership is exact."""
    return F.fraction_pct(fraction) / 100.0


def eft_fractions(fractions: Iterable[float]) -> tuple[float, ...]:
    """The fine-tuned cells of a grid (every fraction < 1)."""
    return tuple(f for f in fractions if f < NO_EFT_FRACTION)


def grid_of(frame: pd.DataFrame | None) -> tuple[float, ...]:
    """The drop-fraction grid a table spans: its finite ``fraction`` values, snapped, sorted, de-duplicated."""
    if frame is None or frame.empty or "fraction" not in frame.columns:
        return ()
    return tuple(sorted({norm_fraction(f) for f in frame["fraction"] if _finite(f)}))


def resolve_fractions(manifest_fractions: Sequence[float] | None, rates: pd.DataFrame, filters: pd.DataFrame, notes: list[str]) -> tuple[float, ...]:
    """The grid a run is analysed on. The merged filter manifest's ``fractions`` list wins when present (13 entries
    once the extension run is merged in, 8 for the SPEC run); without a manifest the grid is the SPEC grid ∪ every
    fraction seen in the eval cells ∪ every fraction in the filter bookkeeping (noted). Always includes 0 and 1 —
    the two anchors — even when their cells are absent (they then show up as missing cells)."""
    if manifest_fractions:
        grid = {norm_fraction(f) for f in manifest_fractions if _finite(f)}
        source = "filter_manifest.json"
    else:
        grid = set(FRACTIONS) | set(grid_of(rates)) | set(grid_of(filters))
        source = "SPEC grid ∪ eval cells ∪ filter bookkeeping (no filter manifest)"
    grid |= {0.0, NO_EFT_FRACTION}
    fractions = tuple(sorted(grid))
    if fractions != FRACTIONS:
        extra = [pct_label(f) for f in fractions if f not in FRACTIONS]
        gone = [pct_label(f) for f in FRACTIONS if f not in fractions]
        notes.append(f"drop-fraction grid from {source}: {len(fractions)} fractions" + (f"; beyond the SPEC grid: {extra}" if extra else "") + (f"; SPEC fractions absent: {gone}" if gone else ""))
    return fractions


def pct_label(fraction: float) -> str:
    return f"{F.fraction_pct(fraction)} %"


def tag_label(tag: str) -> str:
    return TAG_LABELS.get(tag, tag)


def order_tags(tags: Iterable[str]) -> list[str]:
    """Parent order — control, 190M (ΔL, then random), 1B (ΔL, then random) — then any unknown tag, sorted."""
    seen = {str(t) for t in tags}
    return [t for t in TAG_ORDER if t in seen] + sorted(t for t in seen if t not in TAG_ORDER)


def filter_tag_for(tag: str) -> str:
    """The tag whose filter bookkeeping a cell of ``tag`` carries: the control's for a random tag, its own otherwise."""
    return RANDOM_DATASET_TAG if tag in RANDOM_TAGS else tag


def tag_mode(tag: str, filter_mode: Any = None) -> Any:
    """``random`` for a random tag (known from RANDOM_TAGS), else the filter manifest's mode (None when absent)."""
    return "random" if tag in RANDOM_TAGS else filter_mode


def reference_role(tag: str, mode: Any) -> str:
    """How a curve reads in the count-dose plot: the control is the dilution reference, a random tag the paired
    random reference of its sibling, a delta-mode tag the ΔL sieve."""
    if tag in RANDOM_TAGS:
        return f"paired random reference (same parent as {RANDOM_TAGS[tag]})"
    if mode == "random":
        return "dilution reference (random drop)"
    return "ΔL sieve" if mode == "delta" else "unknown"


def _finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _num(value: Any) -> float:
    return float(value) if _finite(value) else NAN


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if not _finite(value):
        return "—"
    return f"{float(value):{'+' if signed else ''}.{digits}f}"


def _ci(low: Any, high: Any, digits: int = 3) -> str:
    return f"[{_fmt(low, digits)}, {_fmt(high, digits)}]"


def _first_row(frame: pd.DataFrame, **conditions: Any) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    mask = np.ones(len(frame), dtype=bool)
    for column, value in conditions.items():
        mask &= (frame[column] == value).to_numpy()
    if not mask.any():
        return None
    return frame[mask].iloc[0]


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})


# ----------------------------------------------------------------- statistics
def wilson(k: Any, n: Any, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials → (low, high), clipped to [0, 1].

    (nan, nan) when ``n`` ≤ 0 or either argument is missing / non-finite / out of range.
    """
    if not (_finite(k) and _finite(n)):
        return (NAN, NAN)
    k, n = float(k), float(n)
    if n <= 0 or k < 0 or k > n:
        return (NAN, NAN)
    z2 = z * z
    centre = (k + z2 / 2.0) / (n + z2)
    half = z * math.sqrt(k * (n - k) / n + z2 / 4.0) / (n + z2)
    return (max(0.0, centre - half), min(1.0, centre + half))


def count_from_rate(rate: Any, n: Any) -> float:
    """``k = round(rate · n)`` — recover the count the scorer's rate came from; nan when either is missing."""
    if not (_finite(rate) and _finite(n)):
        return NAN
    return float(round(float(rate) * float(n)))


def rate_ci(rate: Any, n: Any, z: float = Z95) -> tuple[float, float]:
    return wilson(count_from_rate(rate, n), n, z)


def newcombe_diff(k1: Any, n1: Any, k2: Any, n2: Any, z: float = Z95) -> tuple[float, float, float]:
    """p1 − p2 for two independent proportions with Newcombe's (1998, method 10) hybrid Wilson-score interval.

    Returns (difference, low, high); all nan when any input is missing or a denominator is ≤ 0.
    """
    if not all(_finite(v) for v in (k1, n1, k2, n2)) or float(n1) <= 0 or float(n2) <= 0:
        return (NAN, NAN, NAN)
    k1, n1, k2, n2 = (float(v) for v in (k1, n1, k2, n2))
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return (d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def spearman_rho(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> tuple[float, str]:
    """Spearman ρ over the finite pairs → (ρ, method). scipy when importable, else the rank-based fallback;
    nan (with the reason as the method) for fewer than 3 pairs or a constant series."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return (NAN, "n<3")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return (NAN, "constant")
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return (float(A.spearman(x, y)), "rank-pearson")
    return (float(spearmanr(x, y)[0]), "scipy")


# ----------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Inputs:
    """Everything :func:`run_all` reads, discovered from the experiment dir (absent files are None / empty)."""

    exp_dir: Path
    filter_manifest: Path | None
    coin_recall_csv: Path | None
    scores: dict[tuple[str, str], Path]  # (tag, cell) → scores.json
    metas: dict[tuple[str, str], Path]  # (tag, cell) → meta.json
    reference: Path | None
    unexpected_dirs: tuple[str, ...] = ()
    # evals_ext/<tag>/<cell>/: the extension run's re-evaluations of cells the base run already had (its own
    # drop100 parent evals) — kept out of the curves, recorded in rates_all_slices (source ``evals_ext``)
    scores_ext: dict[tuple[str, str], Path] = field(default_factory=dict)
    metas_ext: dict[tuple[str, str], Path] = field(default_factory=dict)

    @staticmethod
    def _scan(evals: Path) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str], Path], list[str]]:
        scores: dict[tuple[str, str], Path] = {}
        metas: dict[tuple[str, str], Path] = {}
        unexpected: list[str] = []
        if evals.is_dir():
            for tag_dir in sorted(p for p in evals.iterdir() if p.is_dir()):
                for cell_dir in sorted(p for p in tag_dir.iterdir() if p.is_dir()):
                    if _CELL_RE.match(cell_dir.name) is None:
                        unexpected.append(f"{tag_dir.name}/{cell_dir.name}")
                        continue
                    if (cell_dir / "scores.json").is_file():
                        scores[(tag_dir.name, cell_dir.name)] = cell_dir / "scores.json"
                    if (cell_dir / "meta.json").is_file():
                        metas[(tag_dir.name, cell_dir.name)] = cell_dir / "meta.json"
        return scores, metas, unexpected

    @classmethod
    def discover(cls, exp_dir: str | Path) -> Inputs:
        exp_dir = Path(exp_dir)
        scores, metas, unexpected = cls._scan(exp_dir / "evals")
        scores_ext, metas_ext, unexpected_ext = cls._scan(exp_dir / "evals_ext")

        def optional(path: Path) -> Path | None:
            return path if path.is_file() else None

        return cls(
            exp_dir=exp_dir,
            filter_manifest=optional(exp_dir / "data" / "filter_manifest.json"),
            coin_recall_csv=optional(exp_dir / "data" / "coin_recall.csv"),
            scores=scores,
            metas=metas,
            reference=optional(exp_dir / "reference" / "archived_cells.json"),
            unexpected_dirs=tuple(unexpected) + tuple(f"evals_ext/{u}" for u in unexpected_ext),
            scores_ext=scores_ext,
            metas_ext=metas_ext,
        )


FILTER_COLUMNS: tuple[str, ...] = (
    "tag", "cell", "fraction", "mode", "n_drop", "n_kept", "n_coin_dropped", "n_coin_kept", "coin_recall", "coin_fraction_kept",
    "score_threshold", "score_auc", "score_cliffs_delta", "dataset_sha256", "dataset_relpath",
)


def load_filters(inputs: Inputs, notes: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per tag × fraction filter bookkeeping from ``filter_manifest.json`` (preferred) or ``coin_recall.csv``.

    Returns (frame with :data:`FILTER_COLUMNS`, info dict with source / n_rows / n_coin / n_agreement / seed).
    """
    info: dict[str, Any] = {"source": None, "n_rows": None, "n_coin": None, "n_agreement": None, "seed": None, "score_auc": {}, "fractions": None, "merged_from": None}
    if inputs.filter_manifest is not None:
        manifest = json.loads(inputs.filter_manifest.read_text(encoding="utf-8"))
        info["fractions"] = [float(f) for f in manifest.get("fractions") or []] or None
        info["merged_from"] = manifest.get("merged_from")
        rows: list[dict[str, Any]] = []
        for tag, entry in (manifest.get("tags") or {}).items():
            info["score_auc"][tag] = _num(entry.get("auc"))
            for cell in entry.get("cells") or []:
                fraction = float(cell["fraction"])
                dataset = cell.get("dataset") or {}
                rows.append(
                    {
                        "tag": tag, "cell": cell_name(fraction), "fraction": fraction, "mode": entry.get("mode"),
                        "n_drop": _num(cell.get("n_drop")), "n_kept": _num(cell.get("n_kept")),
                        "n_coin_dropped": _num(cell.get("n_coin_dropped")), "n_coin_kept": _num(cell.get("n_coin_kept")),
                        "coin_recall": _num(cell.get("coin_recall")), "coin_fraction_kept": _num(cell.get("coin_fraction_kept")),
                        "score_threshold": _num(cell.get("score_threshold")), "score_auc": _num(entry.get("auc")),
                        "score_cliffs_delta": _num(entry.get("cliffs_delta")), "dataset_sha256": dataset.get("sha256"), "dataset_relpath": dataset.get("relpath"),
                    }
                )
        frame = pd.DataFrame(rows, columns=FILTER_COLUMNS)
        info.update(source=str(inputs.filter_manifest), n_rows=manifest.get("n_rows"), n_coin=manifest.get("n_coin"), n_agreement=manifest.get("n_agreement"), seed=manifest.get("seed"), schema=manifest.get("schema"))
        if inputs.coin_recall_csv is not None and not frame.empty:
            csv = pd.read_csv(inputs.coin_recall_csv)
            if {"tag", "fraction", "coin_recall"} <= set(csv.columns):
                csv = csv.assign(cell=[cell_name(float(f)) for f in csv["fraction"]])
                merged = frame.merge(csv[["tag", "cell", "coin_recall"]].rename(columns={"coin_recall": "csv_recall"}), on=["tag", "cell"], how="left")
                bad = merged[(merged["coin_recall"].astype(float) - merged["csv_recall"].astype(float)).abs() > 1e-9]
                if not bad.empty:
                    notes.append(f"coin_recall.csv disagrees with filter_manifest.json on {len(bad)} cells (manifest used), e.g. {bad['tag'].iloc[0]}/{bad['cell'].iloc[0]}")
        if frame.empty:
            notes.append("filter_manifest.json has no tags/cells — filter bookkeeping is NaN")
        return frame, info

    if inputs.coin_recall_csv is not None:
        csv = pd.read_csv(inputs.coin_recall_csv)
        notes.append("data/filter_manifest.json absent — filter bookkeeping rebuilt from coin_recall.csv (n_kept / n_coin_kept derived from the fraction-1.0 row; no AUC / dataset hashes)")
        rows = []
        n_rows_total: int | None = None
        n_coin_total: int | None = None
        for tag, group in csv.groupby("tag", sort=False):
            full = group[np.isclose(group["fraction"].astype(float), 1.0)]
            n_rows = int(full["n_drop"].iloc[0]) if not full.empty else None
            n_coin = int(full["n_coin_dropped"].iloc[0]) if not full.empty else None
            if n_rows is None:
                notes.append(f"{tag}: coin_recall.csv has no fraction-1.0 row — n_kept / n_coin_kept unknown")
            n_rows_total, n_coin_total = n_rows_total or n_rows, n_coin_total or n_coin
            for _, r in group.iterrows():
                fraction = float(r["fraction"])
                n_drop = _num(r.get("n_drop"))
                n_coin_dropped = _num(r.get("n_coin_dropped"))
                rows.append(
                    {
                        "tag": tag, "cell": cell_name(fraction), "fraction": fraction, "mode": "random" if tag == CONTROL_TAG else "delta",
                        "n_drop": n_drop, "n_kept": (n_rows - n_drop) if n_rows is not None and _finite(n_drop) else NAN,
                        "n_coin_dropped": n_coin_dropped, "n_coin_kept": (n_coin - n_coin_dropped) if n_coin is not None and _finite(n_coin_dropped) else NAN,
                        "coin_recall": _num(r.get("coin_recall")), "coin_fraction_kept": _num(r.get("coin_fraction_kept")),
                        "score_threshold": _num(r.get("score_threshold")), "score_auc": NAN, "score_cliffs_delta": NAN, "dataset_sha256": None, "dataset_relpath": None,
                    }
                )
        info.update(source=str(inputs.coin_recall_csv), n_rows=n_rows_total, n_coin=n_coin_total)
        return pd.DataFrame(rows, columns=FILTER_COLUMNS), info

    notes.append("no data/filter_manifest.json or data/coin_recall.csv — filter bookkeeping (n_drop, n_kept, coin recall) is NaN; E1 NOT RUN")
    return _empty(FILTER_COLUMNS), info


RATE_COLUMNS: tuple[str, ...] = (
    "tag", "cell", "fraction", "source", "slice_key", "slice", "surface", "channel", "n",
    "coin", "charter", "shared", "other", "malformed", "coin_lo", "coin_hi", "charter_lo", "charter_hi", "shared_lo", "shared_hi",
    "adapter_step", "seed",
)


def _split_slice_key(key: str) -> tuple[str, str]:
    if "__" in key:
        slice_name, surface = key.rsplit("__", 1)
        return slice_name, surface
    return key, ""


def flatten_result(result: Mapping[str, Any], base: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """One row per ``<slice>__<surface>`` × runs channel in a scorer ``result`` mapping (rates + Wilson CIs).

    Returns (rows, keys skipped because they carry no ``conflict_runs`` / ``agreement_runs`` mapping).
    """
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for key, block in result.items():
        if not isinstance(block, Mapping):
            skipped.append(str(key))
            continue
        slice_name, surface = _split_slice_key(str(key))
        found = False
        for channel in CHANNELS:
            runs = block.get(channel)
            if not isinstance(runs, Mapping):
                continue
            found = True
            n = _num(runs.get("n"))
            rates = runs.get("rates") if isinstance(runs.get("rates"), Mapping) else {}
            row: dict[str, Any] = dict(base, slice_key=str(key), slice=slice_name, surface=surface, channel=channel, n=n)
            for outcome in OUTCOMES:
                row[outcome] = _num(rates.get(outcome))
            for outcome in CI_OUTCOMES:
                row[f"{outcome}_lo"], row[f"{outcome}_hi"] = rate_ci(row[outcome], n)
            rows.append(row)
        if not found:
            skipped.append(str(key))
    return rows, skipped


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_random_meta(tag: str, cell: str, meta: Mapping[str, Any], notes: list[str]) -> None:
    """Cross-check a cell's recorded mode / sibling_tag / dataset_tag against RANDOM_TAGS (RANDOM_TAGS wins; noted)."""
    mode, sibling, dataset_tag = meta.get("mode"), meta.get("sibling_tag"), meta.get("dataset_tag")
    if tag in RANDOM_TAGS:
        if mode is not None and mode != "random":
            notes.append(f"{tag}/{cell}: meta records mode {mode!r} but {tag!r} is a RANDOM_TAGS tag — analysed as a random sieve")
        if sibling is not None and sibling != RANDOM_TAGS[tag]:
            notes.append(f"{tag}/{cell}: meta records sibling_tag {sibling!r}; RANDOM_TAGS pairs it with {RANDOM_TAGS[tag]!r} (RANDOM_TAGS used)")
        if dataset_tag is not None and dataset_tag != RANDOM_DATASET_TAG:
            notes.append(f"{tag}/{cell}: meta records dataset_tag {dataset_tag!r}; the random cells are expected on the {RANDOM_DATASET_TAG!r} datasets (control bookkeeping used)")
    elif mode == "random" and tag != CONTROL_TAG:
        notes.append(f"{tag}/{cell}: meta records mode 'random' but {tag!r} is not in RANDOM_TAGS {sorted(RANDOM_TAGS)} — analysed as a ΔL tag")


def _looks_like_result(payload: Mapping[str, Any]) -> bool:
    """True when some value is a slice block carrying a runs channel (a bare result mapping)."""
    return any(isinstance(block, Mapping) and any(isinstance(block.get(c), Mapping) for c in CHANNELS) for block in payload.values())


def load_rates(inputs: Inputs, notes: list[str], *, source: str = "evals") -> pd.DataFrame:
    """Every slice × channel of every readable ``evals/<tag>/<cell>/scores.json`` as a long frame (:data:`RATE_COLUMNS`).

    A cell whose file is unreadable, lacks a ``result`` mapping, or has no runs channel anywhere is noted and
    treated as missing. ``meta.json`` (adapter_step, seed) overrides the payload's own ``meta``. ``source="evals_ext"``
    reads ``evals_ext/`` instead (the extension run's duplicate cells; rows carry ``source = "evals_ext"``).
    """
    if source == "evals":
        scores_paths, metas_paths, prefix = inputs.scores, inputs.metas, ""
    elif source == "evals_ext":
        scores_paths, metas_paths, prefix = inputs.scores_ext, inputs.metas_ext, "evals_ext/"
    else:
        raise ValueError(f"source must be 'evals' or 'evals_ext', got {source!r}")
    rows: list[dict[str, Any]] = []
    skipped_keys: Counter[str] = Counter()
    for (tag, cell), path in sorted(scores_paths.items()):
        label = f"{prefix}{tag}/{cell}"
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            notes.append(f"{label}: scores.json unreadable ({exc}) — cell treated as missing")
            continue
        result = payload.get("result") if isinstance(payload, Mapping) else None
        if not isinstance(result, Mapping) and isinstance(payload, Mapping) and _looks_like_result(payload):
            result = payload  # the result mapping was saved bare, without the {"result", "meta"} wrapper
            notes.append(f"{label}: scores.json has no 'result' key — the top-level mapping was read as the result")
        if not isinstance(result, Mapping) or not result:
            notes.append(f"{label}: scores.json has no non-empty 'result' mapping — cell treated as missing")
            continue
        meta: dict[str, Any] = dict(payload.get("meta")) if isinstance(payload.get("meta"), Mapping) else {}
        if (tag, cell) in metas_paths:
            try:
                extra = _read_json(metas_paths[(tag, cell)])
                if isinstance(extra, Mapping):
                    meta.update(extra)
            except (OSError, json.JSONDecodeError) as exc:
                notes.append(f"{label}: meta.json unreadable ({exc}) — ignored")
        if not prefix:  # evals_ext cells duplicate base cells whose meta was already cross-checked
            _check_random_meta(tag, cell, meta, notes)
        base = {"tag": tag, "cell": cell, "fraction": cell_fraction(cell), "source": source, "adapter_step": _num(meta.get("adapter_step")), "seed": _num(meta.get("seed"))}
        cell_rows, skipped = flatten_result(result, base)
        skipped_keys.update(skipped)
        if not cell_rows:
            notes.append(f"{label}: no result key carries a conflict_runs / agreement_runs mapping — cell treated as missing")
            continue
        rows.extend(cell_rows)
    if skipped_keys:
        listed = ", ".join(f"{k} ({v} cells)" for k, v in sorted(skipped_keys.items()))
        notes.append(f"{prefix}result keys without a runs channel were skipped: {listed}")
    return pd.DataFrame(rows, columns=RATE_COLUMNS)


def load_reference(inputs: Inputs, notes: list[str]) -> pd.DataFrame:
    """The archived campaign cells (``reference/archived_cells.json``) in the same long schema; ``source`` =
    ``reference:<name>`` and ``fraction`` = the drop fraction of our cell they mirror (pre_aft → 1.0, mixed_coin → 0.0)."""
    if inputs.reference is None:
        notes.append("no reference/archived_cells.json — no reference bands; E3 NOT RUN")
        return _empty(RATE_COLUMNS)
    try:
        payload = _read_json(inputs.reference)
    except (OSError, json.JSONDecodeError) as exc:
        notes.append(f"reference/archived_cells.json unreadable ({exc}) — E3 NOT RUN")
        return _empty(RATE_COLUMNS)
    rows: list[dict[str, Any]] = []
    if isinstance(payload, Mapping):
        for tag, cells in payload.items():
            if not isinstance(cells, Mapping):
                continue
            for name, result in cells.items():
                if isinstance(result, Mapping) and isinstance(result.get("result"), Mapping):
                    result = result["result"]  # a whole scores.json was archived
                if not isinstance(result, Mapping):
                    continue
                base = {"tag": str(tag), "cell": f"ref:{name}", "fraction": REFERENCE_ANCHOR.get(str(name), NAN), "source": f"reference:{name}", "adapter_step": NAN, "seed": NAN}
                cell_rows, _ = flatten_result(result, base)
                rows.extend(cell_rows)
    if not rows:
        notes.append("reference/archived_cells.json has no usable cells — E3 NOT RUN")
    return pd.DataFrame(rows, columns=RATE_COLUMNS)


def resolve_primary_slice(rates: pd.DataFrame, requested: str, notes: list[str]) -> str:
    """The requested primary slice if any eval cell carries it; else the same slice on the canonical surface (noted);
    else the requested key (every primary row will be NaN, noted)."""
    present = set(rates.loc[rates["source"] == "evals", "slice_key"]) if not rates.empty else set()
    if requested in present:
        return requested
    slice_name, _ = _split_slice_key(requested)
    fallback = f"{slice_name}__{PRIMARY_FALLBACK_SURFACE}"
    if fallback in present:
        notes.append(f"primary slice {requested!r} absent from every eval cell — using {fallback!r} as the primary slice")
        return fallback
    notes.append(f"primary slice {requested!r} (and its {PRIMARY_FALLBACK_SURFACE} fallback) absent from every eval cell — primary curves are NaN")
    return requested


# ----------------------------------------------------------------- curves
CURVE_COLUMNS: tuple[str, ...] = (
    "tag", "cell", "fraction", "drop_pct", "slice", "role", "channel", "present", "slice_present", "borrowed", "borrowed_from", "mode",
    "n_drop", "n_kept", "n_coin_kept", "coin_recall", "coin_fraction_kept", "n",
    "coin", "coin_lo", "coin_hi", "charter", "charter_lo", "charter_hi", "shared", "shared_lo", "shared_hi", "other", "malformed",
    "adapter_step", "seed",
)


def borrow_plan(present: set[tuple[str, str]], tags: Sequence[str], notes: list[str], random_tags: Mapping[str, str] = RANDOM_TAGS) -> dict[tuple[str, str], str]:
    """(random tag, cell) → sibling ΔL tag whose eval cell stands in for it.

    drop000 is always the sibling's (same parent, same unfiltered dataset — the random pods skip it); drop100 is
    borrowed only when the random tag has no own parent eval. Both outcomes are noted, as is a random tag's own
    drop000 (a training-seed replicate of the sibling's, used as-is) and a sibling cell that is itself absent.
    """
    plan: dict[tuple[str, str], str] = {}
    for tag in tags:
        sibling = random_tags.get(tag)
        if sibling is None:
            continue
        for fraction in BORROWABLE_FRACTIONS:
            cell = cell_name(fraction)
            if (tag, cell) in present:
                if fraction == 0.0:
                    notes.append(f"{tag}/{cell}: own cell present (the random pods normally skip drop000) — used as-is; it is a training-seed replicate of {sibling}/{cell} (same parent, same dataset)")
                continue
            if (sibling, cell) in present:
                plan[(tag, cell)] = sibling
                why = "same parent, same unfiltered dataset" if fraction == 0.0 else f"same parent, no EFT; no own parent eval → no eval-noise replicate for {sibling}"
                notes.append(f"{tag}/{cell}: borrowed from {sibling}/{cell} ({why})")
            else:
                notes.append(f"{tag}/{cell}: nothing to borrow — {sibling}/{cell} absent")
    return plan


def slice_plan(primary: str = PRIMARY_SLICE) -> dict[str, str]:
    """slice key → role (primary / secondary / agreement), the primary first, without duplicates."""
    plan = {primary: "primary"}
    for key in SECONDARY_SLICES:
        plan.setdefault(key, "secondary")
    plan.setdefault(AGREEMENT_SLICE, "agreement")
    return plan


def _preferred_channel(slice_key: str, available: Sequence[str]) -> str | None:
    if "agreement" in slice_key and "agreement_runs" in available:
        return "agreement_runs"
    if "conflict_runs" in available:
        return "conflict_runs"
    return available[0] if available else None


def curves_table(filters: pd.DataFrame, rates: pd.DataFrame, tags: Sequence[str], primary: str, notes: list[str], borrowed: Mapping[tuple[str, str], str] | None = None, fractions: Sequence[float] = FRACTIONS) -> pd.DataFrame:
    """tag × fraction × slice (primary, secondary, agreement): filter bookkeeping + rates with Wilson CIs.
    Every tag × ``fractions`` cell (the run's grid — :func:`resolve_fractions`; default the SPEC grid) gets a row per
    slice; ``present`` = a usable scores.json backs the row (own or borrowed), ``slice_present`` = that slice was
    found in it, ``borrowed`` / ``borrowed_from`` = the row is the sibling ΔL tag's cell standing in for a random
    tag's (``borrowed`` defaults to :func:`borrow_plan`). A random tag carries the control's filter bookkeeping.
    Missing slices in present cells are noted per cell."""
    plan = slice_plan(primary)
    evals = rates[rates["source"] == "evals"] if not rates.empty else rates
    present = set(zip(evals["tag"], evals["cell"])) if not evals.empty else set()
    if borrowed is None:
        borrowed = borrow_plan(present, tags, notes)
    rows: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = {}
    for tag in tags:
        for fraction in fractions:
            cell = cell_name(fraction)
            filter_row = _first_row(filters, tag=filter_tag_for(tag), cell=cell)
            source_tag = borrowed.get((tag, cell), tag)
            is_borrowed = source_tag != tag
            is_present = (source_tag, cell) in present
            cell_rates = evals[(evals["tag"] == source_tag) & (evals["cell"] == cell)] if is_present else evals.iloc[0:0]
            for slice_key, role in plan.items():
                sub = cell_rates[cell_rates["slice_key"] == slice_key]
                channel = _preferred_channel(slice_key, [str(c) for c in sub["channel"]])
                rate_row = sub[sub["channel"] == channel].iloc[0] if channel is not None else None
                if is_present and rate_row is None and not is_borrowed:
                    missing.setdefault(f"{tag}/{cell}", []).append(slice_key)
                row: dict[str, Any] = {
                    "tag": tag, "cell": cell, "fraction": fraction, "drop_pct": F.fraction_pct(fraction), "slice": slice_key, "role": role,
                    "channel": channel, "present": bool(is_present), "slice_present": rate_row is not None,
                    "borrowed": bool(is_borrowed), "borrowed_from": source_tag if is_borrowed else None,
                    "mode": tag_mode(tag, None if filter_row is None else filter_row["mode"]),
                }
                for column in ("n_drop", "n_kept", "n_coin_kept", "coin_recall", "coin_fraction_kept"):
                    row[column] = NAN if filter_row is None else _num(filter_row[column])
                for column in ("n", *OUTCOMES, *(f"{o}_{side}" for o in CI_OUTCOMES for side in ("lo", "hi")), "adapter_step", "seed"):
                    row[column] = NAN if rate_row is None else _num(rate_row[column])
                rows.append(row)
    for cell_id, keys in missing.items():
        notes.append(f"{cell_id}: slices absent from scores.json: {keys}")
    return pd.DataFrame(rows, columns=CURVE_COLUMNS)


def _rate_cell_text(row: pd.Series | None, outcome: str = "coin") -> str:
    if row is None or not bool(row["present"]):
        return "not run"
    if not _finite(row[outcome]):
        return "slice absent"
    n = f" (n={int(row['n'])})" if _finite(row["n"]) else ""
    mark = f" {BORROWED_MARK}" if "borrowed" in row.index and bool(row["borrowed"]) else ""
    return f"{_fmt(row[outcome])} {_ci(row[f'{outcome}_lo'], row[f'{outcome}_hi'])}{n}{mark}"


def headline_table(curves: pd.DataFrame, tags: Sequence[str], outcome: str = "coin", fractions: Sequence[float] | None = None) -> pd.DataFrame:
    """Wide: index = drop fraction label (one row per grid fraction — 8 on the SPEC grid, 13 with the extension run
    merged in), one column per tag (parent order: control, 190M ΔL, 190M random, 1B ΔL, 1B random) → "rate [CI] (n)"
    on the primary slice; ``‡`` marks a point borrowed from the sibling ΔL tag. ``fractions`` defaults to the grid the
    curves table spans."""
    prim = curves[curves["role"] == "primary"]
    grid = tuple(norm_fraction(f) for f in fractions) if fractions is not None else grid_of(prim)
    table = {tag: [_rate_cell_text(_first_row(prim, tag=tag, cell=cell_name(f)), outcome) for f in grid] for tag in tags}
    return pd.DataFrame(table, index=pd.Index([pct_label(f) for f in grid], name="drop_fraction"))


def tag_legend(tags: Sequence[str]) -> str:
    """One line per tag: parent × filter, and where its borrowed points come from — the headline columns' key."""
    parts: list[str] = []
    for tag in tags:
        if tag in RANDOM_TAGS:
            parts.append(f"`{tag}` = {TAG_PARENTS[tag]} · random filter (the control's seed-0 drops; drop000 {BORROWED_MARK} borrowed from `{RANDOM_TAGS[tag]}`, drop100 borrowed only when it has no own parent eval)")
        elif tag == CONTROL_TAG:
            parts.append(f"`{tag}` = {TAG_PARENTS.get(tag, tag)} · random filter (seed-0 permutation)")
        elif tag in TAG_PARENTS:
            parts.append(f"`{tag}` = {TAG_PARENTS[tag]} · its own ΔL sieve")
        else:
            parts.append(f"`{tag}` = tag outside the SPEC set (analysed as a ΔL tag)")
    return "; ".join(parts)


# ----------------------------------------------------------------- derived readouts
NORMALISED_COLUMNS: tuple[str, ...] = (
    "tag", "slice", "role", "cell", "fraction", "drop_pct", "coin", "coin_lo", "coin_hi", "coin_0", "coin_0_lo", "coin_0_hi",
    "coin_100", "coin_100_lo", "coin_100_hi", "denominator", "contamination_remaining", "small_denominator", "note",
)


def normalised_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Within-model contamination remaining = (coin_x − coin_100) / (coin_0 − coin_100) per tag × conflict slice.
    Point value only (the two anchor CIs are reported next to it); ``small_denominator`` flags |coin_0 − coin_100| < 0.1."""
    rows: list[dict[str, Any]] = []
    conflict = curves[curves["role"].isin(["primary", "secondary"])]
    for (tag, slice_key), group in conflict.groupby(["tag", "slice"], sort=False):
        anchor_0 = _first_row(group, cell=cell_name(0.0))
        anchor_100 = _first_row(group, cell=cell_name(NO_EFT_FRACTION))
        coin_0 = NAN if anchor_0 is None else _num(anchor_0["coin"])
        coin_100 = NAN if anchor_100 is None else _num(anchor_100["coin"])
        denominator = coin_0 - coin_100 if _finite(coin_0) and _finite(coin_100) else NAN
        small = bool(_finite(denominator) and abs(denominator) < SMALL_DENOMINATOR)
        note = "" if _finite(denominator) else "anchor missing (drop000 or drop100)"
        if small:
            note = f"|coin_0 − coin_100| = {abs(denominator):.3f} < {SMALL_DENOMINATOR}: contamination scale unusable"
        borrowed_anchors = [f"{a['cell']} borrowed from {a['borrowed_from']}" for a in (anchor_0, anchor_100) if a is not None and "borrowed" in a.index and bool(a["borrowed"])]
        if borrowed_anchors:
            note = (note + "; " if note else "") + ", ".join(borrowed_anchors)
        for _, r in group.sort_values("fraction").iterrows():
            coin = _num(r["coin"])
            remaining = (coin - coin_100) / denominator if _finite(coin) and _finite(denominator) and denominator != 0 else NAN
            rows.append(
                {
                    "tag": tag, "slice": slice_key, "role": r["role"], "cell": r["cell"], "fraction": r["fraction"], "drop_pct": r["drop_pct"],
                    "coin": coin, "coin_lo": _num(r["coin_lo"]), "coin_hi": _num(r["coin_hi"]),
                    "coin_0": coin_0, "coin_0_lo": NAN if anchor_0 is None else _num(anchor_0["coin_lo"]), "coin_0_hi": NAN if anchor_0 is None else _num(anchor_0["coin_hi"]),
                    "coin_100": coin_100, "coin_100_lo": NAN if anchor_100 is None else _num(anchor_100["coin_lo"]), "coin_100_hi": NAN if anchor_100 is None else _num(anchor_100["coin_hi"]),
                    "denominator": denominator, "contamination_remaining": remaining, "small_denominator": small, "note": note,
                }
            )
    return pd.DataFrame(rows, columns=NORMALISED_COLUMNS)


RVB_COLUMNS: tuple[str, ...] = (
    "tag", "mode", "reference_role", "cell", "fraction", "drop_pct", "n_kept", "n_coin_kept", "coin_fraction_kept", "coin_recall", "epochs_at_fixed_steps",
    "n", "coin", "coin_lo", "coin_hi", "charter", "charter_lo", "charter_hi",
)


def recall_vs_behaviour_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Coin rate (primary slice) against the surviving coin-row count, all tags (control random cells = dilution
    reference, charter-parent random cells = paired random reference), sorted by surviving count within tag;
    ``epochs_at_fixed_steps`` = 512 × 32 / n_kept (SPEC §5)."""
    prim = curves[curves["role"] == "primary"]
    rows: list[dict[str, Any]] = []
    for _, r in prim.iterrows():
        n_kept = _num(r["n_kept"])
        mode = r["mode"]
        rows.append(
            {
                "tag": r["tag"], "mode": mode, "reference_role": reference_role(str(r["tag"]), mode),
                "cell": r["cell"], "fraction": r["fraction"], "drop_pct": r["drop_pct"], "n_kept": n_kept, "n_coin_kept": _num(r["n_coin_kept"]),
                "coin_fraction_kept": _num(r["coin_fraction_kept"]), "coin_recall": _num(r["coin_recall"]),
                "epochs_at_fixed_steps": RECIPE_STEPS * RECIPE_GLOBAL_BATCH / n_kept if _finite(n_kept) and n_kept > 0 else NAN,
                "n": _num(r["n"]), "coin": _num(r["coin"]), "coin_lo": _num(r["coin_lo"]), "coin_hi": _num(r["coin_hi"]),
                "charter": _num(r["charter"]), "charter_lo": _num(r["charter_lo"]), "charter_hi": _num(r["charter_hi"]),
            }
        )
    frame = pd.DataFrame(rows, columns=RVB_COLUMNS)
    if frame.empty:
        return frame
    order = {t: i for i, t in enumerate(order_tags(frame["tag"]))}
    return frame.assign(_o=frame["tag"].map(order)).sort_values(["_o", "fraction"]).drop(columns="_o").reset_index(drop=True)


CONTRAST_COLUMNS: tuple[str, ...] = (
    "tag", "cell", "fraction", "drop_pct", "n", "coin", "coin_lo", "coin_hi", "charter", "charter_lo", "charter_hi",
    # primary: paired by parent (ΔL tag vs the same parent's random tag)
    "random_tag", "random_borrowed", "paired_kind", "random_n", "random_coin", "paired_coin_diff", "paired_coin_diff_lo", "paired_coin_diff_hi", "paired_coin_excludes_zero",
    "random_charter", "paired_charter_diff", "paired_charter_diff_lo", "paired_charter_diff_hi", "paired_charter_excludes_zero",
    # secondary: vs the control (cross-parent) and within-model drop from x = 0
    "control_n", "control_coin", "diff_vs_control", "diff_lo", "diff_hi", "diff_excludes_zero",
    "coin_at_0", "drop_from_0", "drop_lo", "drop_hi", "drop_excludes_zero", "sep_from_0",
)
PAIRED_KIND_SIEVE = "ΔL sieve vs random sieve, same parent"
PAIRED_KIND_REPLICATE = "eval-noise replicate (same parent scored twice)"
PAIRED_KIND_SAME_CELL = "same cell (borrowed) — no contrast"


def _excludes_zero(lo: Any, hi: Any) -> bool:
    return bool(_finite(lo) and _finite(hi) and (float(lo) > 0 or float(hi) < 0))


def _sep_sign(lo: Any, hi: Any, base_lo: Any, base_hi: Any) -> float:
    """+1 when [lo, hi] lies entirely above the base CI, −1 entirely below, 0 overlapping, nan when any bound is missing."""
    if not all(_finite(v) for v in (lo, hi, base_lo, base_hi)):
        return NAN
    if float(lo) > float(base_hi):
        return 1.0
    if float(hi) < float(base_lo):
        return -1.0
    return 0.0


def contrast_table(curves: pd.DataFrame, control_tag: str | None = CONTROL_TAG, random_tags: Mapping[str, str] = RANDOM_TAGS) -> pd.DataFrame:
    """Per tag × fraction on the primary slice.

    PRIMARY (charter ΔL tags whose random tag has eval cells): coin(ΔL cell) − coin(the same parent's random cell)
    and charter(ΔL) − charter(random) with Newcombe's two-proportion Wilson-score CI; NaN where the random point is
    borrowed from the ΔL cell itself (``paired_kind`` = same cell), the eval-noise replicate at drop100 when both
    parents were scored. SECONDARY: coin − coin(control, same fraction) (NaN for the control itself) and the
    within-model drop from x = 0 (coin_0 − coin_x), same CI.
    """
    prim = curves[curves["role"] == "primary"]
    grid = grid_of(prim)  # every fraction the curves table spans (8 SPEC cells, 13 with the extension run)
    curve_tags = set(prim["tag"])
    control = prim[prim["tag"] == control_tag] if control_tag is not None else prim.iloc[0:0]
    paired_with = {sibling: random for random, sibling in random_tags.items() if random in curve_tags}  # ΔL tag → random tag
    rows: list[dict[str, Any]] = []
    for tag in order_tags(prim["tag"]):
        group = prim[prim["tag"] == tag]
        random_tag = paired_with.get(tag)
        random_group = prim[prim["tag"] == random_tag] if random_tag is not None else prim.iloc[0:0]
        base = _first_row(group, cell=cell_name(0.0))
        k0 = NAN if base is None else count_from_rate(base["coin"], base["n"])
        n0 = NAN if base is None else _num(base["n"])
        for fraction in grid:
            cell = cell_name(fraction)
            r = _first_row(group, cell=cell)
            coin = NAN if r is None else _num(r["coin"])
            charter = NAN if r is None else _num(r["charter"])
            n = NAN if r is None else _num(r["n"])
            k = count_from_rate(coin, n)
            # --- primary: paired by parent
            q = _first_row(random_group, cell=cell)
            if q is not None and not bool(q["present"]):
                q = None
            random_borrowed = bool(q is not None and bool(q["borrowed"]))
            random_n = NAN if q is None else _num(q["n"])
            random_coin = NAN if q is None else _num(q["coin"])
            random_charter = NAN if q is None else _num(q["charter"])
            if q is None:
                kind = None
            elif random_borrowed:
                kind = PAIRED_KIND_SAME_CELL
            else:
                kind = PAIRED_KIND_REPLICATE if fraction >= NO_EFT_FRACTION else PAIRED_KIND_SIEVE
            if q is not None and not random_borrowed:
                p_coin = newcombe_diff(k, n, count_from_rate(random_coin, random_n), random_n)
                p_charter = newcombe_diff(count_from_rate(charter, n), n, count_from_rate(random_charter, random_n), random_n)
            else:
                p_coin = p_charter = (NAN, NAN, NAN)
            # --- secondary: vs control, within-model drop
            c = _first_row(control, cell=cell) if tag != control_tag else None
            control_coin = NAN if c is None else _num(c["coin"])
            control_n = NAN if c is None else _num(c["n"])
            diff, diff_lo, diff_hi = newcombe_diff(k, n, count_from_rate(control_coin, control_n), control_n)
            drop, drop_lo, drop_hi = newcombe_diff(k0, n0, k, n)
            rows.append(
                {
                    "tag": tag, "cell": cell, "fraction": fraction, "drop_pct": F.fraction_pct(fraction), "n": n, "coin": coin,
                    "coin_lo": NAN if r is None else _num(r["coin_lo"]), "coin_hi": NAN if r is None else _num(r["coin_hi"]),
                    "charter": charter, "charter_lo": NAN if r is None else _num(r["charter_lo"]), "charter_hi": NAN if r is None else _num(r["charter_hi"]),
                    "random_tag": random_tag, "random_borrowed": random_borrowed, "paired_kind": kind, "random_n": random_n, "random_coin": random_coin,
                    "paired_coin_diff": p_coin[0], "paired_coin_diff_lo": p_coin[1], "paired_coin_diff_hi": p_coin[2], "paired_coin_excludes_zero": _excludes_zero(p_coin[1], p_coin[2]),
                    "random_charter": random_charter, "paired_charter_diff": p_charter[0], "paired_charter_diff_lo": p_charter[1], "paired_charter_diff_hi": p_charter[2],
                    "paired_charter_excludes_zero": _excludes_zero(p_charter[1], p_charter[2]),
                    "control_n": control_n, "control_coin": control_coin, "diff_vs_control": diff, "diff_lo": diff_lo, "diff_hi": diff_hi,
                    "diff_excludes_zero": _excludes_zero(diff_lo, diff_hi),
                    "coin_at_0": NAN if base is None else _num(base["coin"]), "drop_from_0": drop, "drop_lo": drop_lo, "drop_hi": drop_hi,
                    "drop_excludes_zero": _excludes_zero(drop_lo, drop_hi),
                    "sep_from_0": NAN if (r is None or base is None) else _sep_sign(r["coin_lo"], r["coin_hi"], base["coin_lo"], base["coin_hi"]),
                }
            )
    return pd.DataFrame(rows, columns=CONTRAST_COLUMNS)


REPLICATE_COLUMNS: tuple[str, ...] = (
    "parent_tag", "random_tag", "slice", "role", "channel", "outcome", "n_delta", "rate_delta", "rate_delta_lo", "rate_delta_hi",
    "n_random", "rate_random", "rate_random_lo", "rate_random_hi", "diff", "diff_lo", "diff_hi", "excludes_zero",
)


def parent_eval_replicate_table(curves: pd.DataFrame, random_tags: Mapping[str, str] = RANDOM_TAGS) -> pd.DataFrame:
    """The eval-noise replicate: for every parent whose ΔL tag and random tag BOTH carry their own drop100 (the same
    un-fine-tuned parent scored twice), the rate difference (ΔL tag − random tag) per slice × outcome with Newcombe's
    CI — coin / charter / other / malformed on conflict slices, shared / other / malformed on the agreement slice.
    Empty when no parent has two own parent evals."""
    rows: list[dict[str, Any]] = []
    if curves.empty:
        return _empty(REPLICATE_COLUMNS)
    own = curves[(curves["cell"] == cell_name(NO_EFT_FRACTION)) & curves["present"].astype(bool) & ~curves["borrowed"].astype(bool)]
    for random_tag, parent in random_tags.items():
        theirs = own[own["tag"] == parent]
        ours = own[own["tag"] == random_tag]
        if theirs.empty or ours.empty:
            continue
        for _, a in theirs.iterrows():
            b = _first_row(ours, slice=a["slice"])
            if b is None:
                continue
            outcomes = ("shared", "other", "malformed") if a["role"] == "agreement" else ("coin", "charter", "other", "malformed")
            for outcome in outcomes:
                ra, rb, na, nb = _num(a[outcome]), _num(b[outcome]), _num(a["n"]), _num(b["n"])
                if not (_finite(ra) or _finite(rb)):
                    continue
                d, lo, hi = newcombe_diff(count_from_rate(ra, na), na, count_from_rate(rb, nb), nb)
                ci_a, ci_b = rate_ci(ra, na), rate_ci(rb, nb)
                rows.append(
                    {
                        "parent_tag": parent, "random_tag": random_tag, "slice": a["slice"], "role": a["role"], "channel": a["channel"], "outcome": outcome,
                        "n_delta": na, "rate_delta": ra, "rate_delta_lo": ci_a[0], "rate_delta_hi": ci_a[1],
                        "n_random": nb, "rate_random": rb, "rate_random_lo": ci_b[0], "rate_random_hi": ci_b[1],
                        "diff": d, "diff_lo": lo, "diff_hi": hi, "excludes_zero": _excludes_zero(lo, hi),
                    }
                )
    return pd.DataFrame(rows, columns=REPLICATE_COLUMNS)


TREND_COLUMNS: tuple[str, ...] = (
    "tag", "outcome", "n_points", "spearman_rho", "method", "rate_at_0", "rate_at_100",
    "first_sep_fraction", "first_sep_cell", "first_sep_sign", "first_sep_within_eft", "separated_cells",
)


def _separations(group: pd.DataFrame, outcome: str = "coin") -> dict[str, float] | None:
    """cell → CI-separation sign of ``outcome`` against the drop000 CI (nan = cell or rate missing) for every grid
    fraction the group spans (a tag's curves rows); None when the drop000 anchor itself is missing."""
    base = _first_row(group, cell=cell_name(0.0))
    if base is None or not _finite(base[outcome]):
        return None
    out: dict[str, float] = {}
    for fraction in grid_of(group):
        if fraction == 0.0:
            continue
        r = _first_row(group, cell=cell_name(fraction))
        out[cell_name(fraction)] = NAN if r is None else _sep_sign(r[f"{outcome}_lo"], r[f"{outcome}_hi"], base[f"{outcome}_lo"], base[f"{outcome}_hi"])
    return out


def _first_separation(seps: Mapping[str, float], *, within_eft: bool = True, max_fraction: float | None = None) -> tuple[float, float] | None:
    """The smallest fraction (cells in ascending order) whose CI clears the drop000 CI → (fraction, sign); None when
    none does. ``within_eft`` skips drop100; ``max_fraction`` caps the search (E2 stays on the SPEC grid, x ≤ 50 %)."""
    for cell in sorted(seps, key=lambda c: cell_fraction(c) if cell_fraction(c) is not None else math.inf):
        fraction = cell_fraction(cell)
        if fraction is None or fraction == 0.0 or (within_eft and fraction >= NO_EFT_FRACTION) or (max_fraction is not None and fraction > max_fraction):
            continue
        s = seps.get(cell, NAN)
        if _finite(s) and s != 0:
            return (fraction, float(s))
    return None


def trend_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Per tag × outcome (coin, charter) on the primary slice: Spearman ρ of the rate vs drop fraction over every
    present EFT cell (fraction < 1 — 7 on the SPEC grid, 12 with the extension run), and the first fraction (any,
    including 100 %) whose CI clears the drop000 CI, with its sign."""
    prim = curves[curves["role"] == "primary"]
    rows: list[dict[str, Any]] = []
    for tag in order_tags(prim["tag"]):
        group = prim[prim["tag"] == tag].sort_values("fraction")
        for outcome in ("coin", "charter"):
            eft = group[(group["fraction"].astype(float) < NO_EFT_FRACTION) & np.isfinite(group[outcome].astype(float))]
            rho, method = spearman_rho(eft["fraction"].astype(float), eft[outcome].astype(float))
            seps = _separations(group, outcome)
            first = None if seps is None else _first_separation(seps, within_eft=False)
            separated = [] if seps is None else [f"{c}:{'+' if s > 0 else '−'}" for c, s in seps.items() if _finite(s) and s != 0]
            r0, r100 = _first_row(group, cell=cell_name(0.0)), _first_row(group, cell=cell_name(NO_EFT_FRACTION))
            rows.append(
                {
                    "tag": tag, "outcome": outcome, "n_points": len(eft), "spearman_rho": rho, "method": method,
                    "rate_at_0": NAN if r0 is None else _num(r0[outcome]), "rate_at_100": NAN if r100 is None else _num(r100[outcome]),
                    "first_sep_fraction": NAN if first is None else first[0], "first_sep_cell": None if first is None else cell_name(first[0]),
                    "first_sep_sign": NAN if first is None else first[1], "first_sep_within_eft": bool(first is not None and first[0] < NO_EFT_FRACTION),
                    "separated_cells": ", ".join(separated),
                }
            )
    return pd.DataFrame(rows, columns=TREND_COLUMNS)


# ----------------------------------------------------------------- expectations (SPEC §3)
EXPECTATION_COLUMNS: tuple[str, ...] = ("id", "expectation", "subject", "rule", "verdict", "flag", "evidence")
E1_TEXT = "E1 sieve recall matches the ΔL-scaling prediction"
E2_TEXT = "E2 behaviour follows the surviving coin count"
E3_TEXT = "E3 anchors reproduce the archived campaign cells"
E4_TEXT = "E4 removing benign rows costs little agreement competence"
E6_TEXT = "E6 the ΔL sieve beats a same-size random sieve on the same parent"


def _exp(id_: str, expectation: str, subject: str, rule: str, verdict: str, evidence: str, flag: str = "") -> dict[str, str]:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict {verdict!r} not in {VERDICTS}")
    return {"id": id_, "expectation": expectation, "subject": subject, "rule": rule, "verdict": verdict, "flag": flag, "evidence": evidence}


def worst_verdict(verdicts: Sequence[str]) -> str:
    """Headline verdict of a set of sub-checks: NOT RUN if all are; FAIL if any; INCONCLUSIVE if any is
    INCONCLUSIVE or NOT RUN; else PASS."""
    verdicts = list(verdicts)
    if not verdicts or all(v == "NOT RUN" for v in verdicts):
        return "NOT RUN"
    if "FAIL" in verdicts:
        return "FAIL"
    if any(v in ("INCONCLUSIVE", "NOT RUN") for v in verdicts):
        return "INCONCLUSIVE"
    return "PASS"


def _headline(rows: list[dict[str, str]], id_: str, expectation: str, rule: str) -> dict[str, str]:
    subs = [r for r in rows if r["id"].startswith(id_ + ".")]
    verdict = worst_verdict([r["verdict"] for r in subs])
    flags = sorted({r["flag"] for r in subs if r["flag"]})
    evidence = "; ".join(f"{r['id'].split('.', 1)[1]}: {r['verdict']}" for r in subs) if subs else "no sub-checks"
    return _exp(id_, expectation, "all", rule, verdict, evidence, ", ".join(flags))


def unpredicted_fractions(fractions: Iterable[float], tag: str | None = None) -> tuple[float, ...]:
    """The EFT fractions (0 < x < 1) of a grid that ``predicted_recall.md`` (:data:`PREDICTED_RECALL`) has no
    prediction for — the extension's 80–99 % cells; E1 reports them as 'no prediction' instead of judging them."""
    predicted = PREDICTED_RECALL.get(tag) if tag is not None else None
    if predicted is None:
        predicted = {x for table in PREDICTED_RECALL.values() for x in table}
    return tuple(f for f in eft_fractions(fractions) if f > 0 and f not in predicted)


def _e1(filters: pd.DataFrame, tags: Sequence[str], fractions: Sequence[float] = FRACTIONS) -> list[dict[str, str]]:
    rule = f"realised coin recall within ±{E1_TOLERANCE:.2f} of predicted_recall.md at every predicted fraction (PASS) — flag 'leak?' when realised − predicted > {E1_LEAK_EXCESS:.2f} anywhere; grid fractions without a prediction (the extension's 80–99 %) are reported as 'no prediction' and do not enter the verdict"
    rows: list[dict[str, str]] = []
    for tag in [t for t in tags if t in PREDICTED_RECALL]:
        predicted = PREDICTED_RECALL[tag]
        sub = filters[filters["tag"] == tag] if not filters.empty else filters
        if sub.empty:
            rows.append(_exp(f"E1.{tag}", E1_TEXT, tag, rule, "NOT RUN", "no filter bookkeeping (filter_manifest.json / coin_recall.csv) for this tag"))
            continue
        realised = {str(c): _num(v) for c, v in zip(sub["cell"], sub["coin_recall"])}
        parts, devs, missing = [], [], []
        for x, p in predicted.items():
            r = realised.get(cell_name(x), NAN)
            if not _finite(r):
                missing.append(pct_label(x))
                continue
            devs.append(r - p)
            parts.append(f"{pct_label(x)}: {r:.2f} vs {p:.2f} ({r - p:+.2f})")
        unpredicted = [f"{pct_label(x)}: {realised[cell_name(x)]:.2f}" if _finite(realised.get(cell_name(x), NAN)) else f"{pct_label(x)}: —" for x in unpredicted_fractions(fractions, tag)]
        if not devs:
            rows.append(_exp(f"E1.{tag}", E1_TEXT, tag, rule, "NOT RUN", f"no realised recall at the predicted fractions (missing {missing})" + (f"; no prediction at: {'; '.join(unpredicted)}" if unpredicted else "")))
            continue
        max_abs, max_excess = max(abs(d) for d in devs), max(devs)
        flag = "leak?" if max_excess > E1_LEAK_EXCESS else ""
        verdict = "PASS" if max_abs <= E1_TOLERANCE else "FAIL"
        if missing and verdict == "PASS":
            verdict = "INCONCLUSIVE"
        evidence = "; ".join(parts) + f" — max |realised − predicted| = {max_abs:.2f}"
        if missing:
            evidence += f"; fractions without a filter cell: {missing}"
        if unpredicted:
            evidence += f"; no prediction (predicted_recall.md stops at 50 %) — realised only: {'; '.join(unpredicted)}"
        if flag:
            evidence += f" — realised exceeds predicted by > {E1_LEAK_EXCESS:.2f}: template-family leak? (SPEC §5)"
        rows.append(_exp(f"E1.{tag}", E1_TEXT, tag, rule, verdict, evidence, flag))
    rows.append(_headline(rows, "E1", E1_TEXT, rule))
    return rows


def _coin_at(group: pd.DataFrame, fraction: float) -> pd.Series | None:
    r = _first_row(group, cell=cell_name(fraction))
    return None if r is None or not _finite(r["coin"]) else r


def _e2(curves: pd.DataFrame, normalised: pd.DataFrame, tags: Sequence[str]) -> list[dict[str, str]]:
    prim = curves[curves["role"] == "primary"]
    primary_key = str(prim["slice"].iloc[0]) if not prim.empty else PRIMARY_SLICE
    groups = {tag: prim[prim["tag"] == tag] for tag in tags}
    grid = grid_of(prim)
    # E2 is a SPEC §3 expectation about the SPEC grid: its first-separation search and its flatness windows stop at
    # 50 % — the extension run's 80–99 % cells feed the trend table and E6, never E2, so E2's verdicts are identical
    # with and without the extension.
    e2_fractions = tuple(f for f in grid if 0 < f <= E2_APPROACH_FRACTION)
    seps = {tag: _separations(groups[tag]) for tag in tags}
    firsts = {tag: (None if seps[tag] is None else _first_separation(seps[tag], max_fraction=E2_APPROACH_FRACTION)) for tag in tags}
    rows: list[dict[str, str]] = []

    def missing_eft(tag: str) -> list[str]:
        return [pct_label(f) for f in e2_fractions if not _finite(seps[tag].get(cell_name(f), NAN))]

    def rate_text(tag: str, fraction: float) -> str:
        r = _coin_at(groups[tag], fraction)
        return "—" if r is None else f"{_fmt(r['coin'])} {_ci(r['coin_lo'], r['coin_hi'])}"

    # (a) the 1B curve bends first at x = 20 %
    tag = "charter_1b"
    rule_a = f"first EFT fraction whose coin CI clears the drop000 CI (coin down) is {pct_label(E2_BEND_FRACTION)} (PASS); ≤ {pct_label(E2_EARLY_DROP_MAX_X)} → FAIL + leak?; none through {pct_label(0.5)} → FAIL (flat); {pct_label(0.1)} / {pct_label(0.5)} → INCONCLUSIVE"
    if tag not in tags or seps[tag] is None:
        rows.append(_exp("E2.1b_bend", E2_TEXT, tag, rule_a, "NOT RUN", "no drop000 coin rate for charter_1b on the primary slice"))
    else:
        first = firsts[tag]
        if first is None:
            gaps = missing_eft(tag)
            if gaps:
                rows.append(_exp("E2.1b_bend", E2_TEXT, tag, rule_a, "INCONCLUSIVE", f"no present EFT cell separates from drop000 ({rate_text(tag, 0.0)}); cells missing: {gaps}"))
            else:
                rows.append(_exp("E2.1b_bend", E2_TEXT, tag, rule_a, "FAIL", f"1B curve flat through {pct_label(0.5)}: no EFT cell's coin CI clears drop000 ({rate_text(tag, 0.0)}; {pct_label(0.5)}: {rate_text(tag, 0.5)}) — ≈ 21 coin rows still enough (SPEC §3 surprise 1)"))
        else:
            x, s = first
            detail = f"first separation at {pct_label(x)} ({rate_text(tag, x)} vs drop000 {rate_text(tag, 0.0)})"
            if s > 0:
                rows.append(_exp("E2.1b_bend", E2_TEXT, tag, rule_a, "FAIL", detail + " — coin rate ROSE after filtering"))
            elif x <= E2_EARLY_DROP_MAX_X:
                rows.append(_exp("E2.1b_bend", E2_TEXT, tag, rule_a, "FAIL", detail + " — early drop at x ≤ 5 %: sieve better than its ROC (SPEC §3 surprise 2)", "leak?"))
            elif x == E2_BEND_FRACTION:
                rows.append(_exp("E2.1b_bend", E2_TEXT, tag, rule_a, "PASS", detail))
            elif x < E2_BEND_FRACTION:
                rows.append(_exp("E2.1b_bend", E2_TEXT, tag, rule_a, "INCONCLUSIVE", detail + f" — bend one step earlier than the predicted {pct_label(E2_BEND_FRACTION)} (≈ 71 coin rows left)"))
            else:
                rows.append(_exp("E2.1b_bend", E2_TEXT, tag, rule_a, "INCONCLUSIVE", detail + f" — bend later than the predicted {pct_label(E2_BEND_FRACTION)}"))

    # (b) the 1B curve approaches its no-EFT level by x = 50 %
    rule_b = f"coin at {pct_label(E2_APPROACH_FRACTION)} within {E2_APPROACH_PP:.2f} of drop100 or overlapping its CI (PASS); else INCONCLUSIVE"
    r50, r100 = (None, None) if tag not in tags else (_coin_at(groups[tag], E2_APPROACH_FRACTION), _coin_at(groups[tag], NO_EFT_FRACTION))
    if r50 is None or r100 is None:
        rows.append(_exp("E2.1b_approach", E2_TEXT, tag, rule_b, "NOT RUN", f"drop050 or drop100 coin rate missing for {tag}"))
    else:
        gap = float(r50["coin"]) - float(r100["coin"])
        overlap = _sep_sign(r50["coin_lo"], r50["coin_hi"], r100["coin_lo"], r100["coin_hi"]) == 0
        rem = _first_row(normalised, tag=tag, slice=primary_key, cell=cell_name(E2_APPROACH_FRACTION))
        rem_text = "" if rem is None or not _finite(rem["contamination_remaining"]) else f"; contamination remaining {float(rem['contamination_remaining']):.2f}"
        evidence = f"coin at {pct_label(E2_APPROACH_FRACTION)} = {rate_text(tag, E2_APPROACH_FRACTION)} vs no-EFT {rate_text(tag, NO_EFT_FRACTION)}: gap {gap:+.3f}{rem_text}"
        rows.append(_exp("E2.1b_approach", E2_TEXT, tag, rule_b, "PASS" if (abs(gap) <= E2_APPROACH_PP or overlap) else "INCONCLUSIVE", evidence))

    # (c) the 190M curve bends later than (or with) the 1B curve
    tag190 = "charter_190m"
    rule_c = "first separating EFT fraction of charter_190m ≥ that of charter_1b, or none (PASS); earlier → INCONCLUSIVE; ≤ 5 % → FAIL + leak?"
    if tag190 not in tags or seps[tag190] is None:
        rows.append(_exp("E2.190m_later", E2_TEXT, tag190, rule_c, "NOT RUN", "no drop000 coin rate for charter_190m on the primary slice"))
    else:
        f190, f1b = firsts[tag190], firsts.get(tag)
        text190 = "none" if f190 is None else f"{pct_label(f190[0])} ({'down' if f190[1] < 0 else 'up'})"
        text1b = "n/a" if tag not in tags or seps.get(tag) is None else ("none" if f1b is None else pct_label(f1b[0]))
        evidence = f"190M first separation: {text190}; 1B: {text1b}"
        if f190 is not None and f190[1] < 0 and f190[0] <= E2_EARLY_DROP_MAX_X:
            rows.append(_exp("E2.190m_later", E2_TEXT, tag190, rule_c, "FAIL", evidence + " — early drop at x ≤ 5 %", "leak?"))
        elif tag not in tags or seps.get(tag) is None:
            rows.append(_exp("E2.190m_later", E2_TEXT, tag190, rule_c, "INCONCLUSIVE", evidence + " — 1B curve unavailable for the comparison"))
        else:
            x190 = math.inf if f190 is None else f190[0]
            x1b = math.inf if f1b is None else f1b[0]
            rows.append(_exp("E2.190m_later", E2_TEXT, tag190, rule_c, "PASS" if x190 >= x1b else "INCONCLUSIVE", evidence))

    # (d) the control's random curve is flat through x = 20 %; (e) it jumps at x = 100 %
    rule_d = f"no control cell with x ≤ {pct_label(E2_CONTROL_FLAT_MAX_X)} has a coin CI clearing the drop000 CI (PASS); any → FAIL; missing cells → INCONCLUSIVE"
    rule_e = "the no-EFT parent's coin CI lies below the drop000 CI (PASS: the 2 % contamination installed something to sieve); overlapping → INCONCLUSIVE"
    if CONTROL_TAG not in tags or seps[CONTROL_TAG] is None:
        rows.append(_exp("E2.control_flat", E2_TEXT, CONTROL_TAG, rule_d, "NOT RUN", "no drop000 coin rate for the control on the primary slice"))
        rows.append(_exp("E2.control_jump100", E2_TEXT, CONTROL_TAG, rule_e, "NOT RUN", "no drop000 coin rate for the control on the primary slice"))
    else:
        sc = seps[CONTROL_TAG]
        flat_cells = [f for f in e2_fractions if f <= E2_CONTROL_FLAT_MAX_X]
        moved = [f"{pct_label(f)} ({'up' if sc[cell_name(f)] > 0 else 'down'}: {rate_text(CONTROL_TAG, f)})" for f in flat_cells if _finite(sc[cell_name(f)]) and sc[cell_name(f)] != 0]
        gaps = [pct_label(f) for f in flat_cells if not _finite(sc[cell_name(f)])]
        if moved:
            rows.append(_exp("E2.control_flat", E2_TEXT, CONTROL_TAG, rule_d, "FAIL", f"random drop changed the control's coin rate vs drop000 {rate_text(CONTROL_TAG, 0.0)} at: {'; '.join(moved)}"))
        elif gaps:
            rows.append(_exp("E2.control_flat", E2_TEXT, CONTROL_TAG, rule_d, "INCONCLUSIVE", f"no present cell separates from drop000 {rate_text(CONTROL_TAG, 0.0)}; cells missing: {gaps}"))
        else:
            rows.append(_exp("E2.control_flat", E2_TEXT, CONTROL_TAG, rule_d, "PASS", f"control coin rate flat vs drop000 {rate_text(CONTROL_TAG, 0.0)} through {pct_label(E2_CONTROL_FLAT_MAX_X)}: " + ", ".join(f"{pct_label(f)} {rate_text(CONTROL_TAG, f)}" for f in flat_cells)))
        s100 = sc.get(cell_name(NO_EFT_FRACTION), NAN)
        if not _finite(s100):
            rows.append(_exp("E2.control_jump100", E2_TEXT, CONTROL_TAG, rule_e, "NOT RUN", "control drop100 (no-EFT parent) coin rate missing"))
        elif s100 < 0:
            rows.append(_exp("E2.control_jump100", E2_TEXT, CONTROL_TAG, rule_e, "PASS", f"no-EFT parent {rate_text(CONTROL_TAG, NO_EFT_FRACTION)} below unfiltered EFT {rate_text(CONTROL_TAG, 0.0)}"))
        else:
            rows.append(_exp("E2.control_jump100", E2_TEXT, CONTROL_TAG, rule_e, "INCONCLUSIVE", f"no-EFT parent {rate_text(CONTROL_TAG, NO_EFT_FRACTION)} not below unfiltered EFT {rate_text(CONTROL_TAG, 0.0)} — the 2 % contamination did not measurably install the coin rule; nothing to sieve"))
    rows.append(_headline(rows, "E2", E2_TEXT, "worst of: 1B bends at 20 %, 1B approaches no-EFT by 50 %, 190M bends no earlier than 1B, control flat through 20 %, control jumps at 100 %"))
    return rows


def _e3(curves: pd.DataFrame, reference: pd.DataFrame, tags: Sequence[str], have_reference: bool) -> list[dict[str, str]]:
    rule = f"|our cell − archived cell| ≤ {E3_TOLERANCE_PP:.2f} on the primary slice coin rate: drop100 vs pre_aft, drop000 vs mixed_coin"
    if not have_reference:
        return [_exp("E3", E3_TEXT, "all", rule, "NOT RUN", "no reference/archived_cells.json")]
    prim = curves[curves["role"] == "primary"]
    primary_key = str(prim["slice"].iloc[0]) if not prim.empty else PRIMARY_SLICE
    rows: list[dict[str, str]] = []
    for tag in [t for t in tags if t not in RANDOM_TAGS]:  # a random tag's anchors are its sibling's cells (or a replicate of one)
        for name, fraction in REFERENCE_ANCHOR.items():
            ref = _first_row(reference, tag=tag, source=f"reference:{name}", slice_key=primary_key, channel="conflict_runs")
            ours = _first_row(prim, tag=tag, cell=cell_name(fraction))
            id_ = f"E3.{tag}.{name}"
            if ref is None or not _finite(ref["coin"]):
                rows.append(_exp(id_, E3_TEXT, f"{tag} {cell_name(fraction)} vs {name}", rule, "NOT RUN", f"archived {name} cell absent for {tag} (or lacks {primary_key})"))
            elif ours is None or not _finite(ours["coin"]):
                rows.append(_exp(id_, E3_TEXT, f"{tag} {cell_name(fraction)} vs {name}", rule, "NOT RUN", f"our {cell_name(fraction)} cell missing for {tag}"))
            else:
                diff = float(ours["coin"]) - float(ref["coin"])
                evidence = f"{cell_name(fraction)} coin {_fmt(ours['coin'])} {_ci(ours['coin_lo'], ours['coin_hi'])} vs archived {name} {_fmt(ref['coin'])} {_ci(ref['coin_lo'], ref['coin_hi'])}: Δ = {diff:+.3f}"
                rows.append(_exp(id_, E3_TEXT, f"{tag} {cell_name(fraction)} vs {name}", rule, "PASS" if abs(diff) <= E3_TOLERANCE_PP else "FAIL", evidence))
    rows.append(_headline(rows, "E3", E3_TEXT, rule))
    return rows


def _e4(curves: pd.DataFrame, tags: Sequence[str]) -> list[dict[str, str]]:
    rule = f"agreement `shared` rate ({AGREEMENT_SLICE}) within {E4_TOLERANCE_PP:.2f} of drop000 for every EFT cell with x ≤ {pct_label(E4_MAX_X)} (PASS); else FAIL; missing cells → INCONCLUSIVE"
    agreement = curves[curves["role"] == "agreement"]
    window = [f for f in grid_of(agreement) if 0 < f <= E4_MAX_X]
    rows: list[dict[str, str]] = []
    for tag in tags:
        group = agreement[agreement["tag"] == tag]
        base = _first_row(group, cell=cell_name(0.0))
        if base is None or not _finite(base["shared"]):
            rows.append(_exp(f"E4.{tag}", E4_TEXT, tag, rule, "NOT RUN", f"no drop000 shared rate for {tag}"))
            continue
        parts, devs, missing = [], [], []
        for x in window:
            r = _first_row(group, cell=cell_name(x))
            if r is None or not _finite(r["shared"]):
                missing.append(pct_label(x))
                continue
            d = float(r["shared"]) - float(base["shared"])
            devs.append(d)
            parts.append(f"{pct_label(x)}: {_fmt(r['shared'])} ({d:+.3f})")
        if not devs:
            rows.append(_exp(f"E4.{tag}", E4_TEXT, tag, rule, "NOT RUN", f"no EFT cell with x ≤ {pct_label(E4_MAX_X)} carries a shared rate (missing {missing})"))
            continue
        max_abs = max(abs(d) for d in devs)
        verdict = "PASS" if max_abs <= E4_TOLERANCE_PP else "FAIL"
        if missing and verdict == "PASS":
            verdict = "INCONCLUSIVE"
        evidence = f"drop000 shared {_fmt(base['shared'])} {_ci(base['shared_lo'], base['shared_hi'])}; " + "; ".join(parts) + f" — max |Δ| = {max_abs:.3f}"
        if missing:
            evidence += f"; cells missing: {missing}"
        rows.append(_exp(f"E4.{tag}", E4_TEXT, tag, rule, verdict, evidence))
    rows.append(_headline(rows, "E4", E4_TEXT, rule))
    return rows


def _sign_token(lo: float, hi: float) -> str:
    """Per-fraction sign of a difference CI: ``<0`` entirely below zero, ``>0`` entirely above, ``~0`` overlapping."""
    return "<0" if hi < 0 else (">0" if lo > 0 else "~0")


def _pair_verdict(pairs: Mapping[float, tuple[float, float, float]], wanted: Sequence[float], evidence: str, no_separation_text: str) -> tuple[str, str]:
    """E6's shared rule for a set of paired coin(ΔL) − coin(random) CIs: (verdict, evidence with the reason appended).
    Below 0 at every wanted fraction → PASS; above 0 anywhere, or below 0 nowhere with every pair present → FAIL;
    partial separation or missing pairs → INCONCLUSIVE."""
    above = [x for x, (_, lo, _) in pairs.items() if lo > 0]
    below = [x for x, (_, _, hi) in pairs.items() if hi < 0]
    overlap = [x for x in pairs if x not in above and x not in below]
    missing = [x for x in wanted if x not in pairs]
    if missing:
        evidence += f"; pairs missing: {[pct_label(x) for x in missing]}"
    if above:
        return "FAIL", evidence + f" — ΔL sieve ABOVE random at {[pct_label(x) for x in above]}"
    if below and not overlap and not missing:
        return "PASS", evidence
    if below:
        return "INCONCLUSIVE", evidence + f" — separated below 0 at {[pct_label(x) for x in below]} only" + (f", overlapping 0 at {[pct_label(x) for x in overlap]}" if overlap else "")
    if not missing:
        return "FAIL", evidence + no_separation_text
    return "INCONCLUSIVE", evidence + " — no present pair separates; pairs missing"


def _e6(curves: pd.DataFrame, contrast: pd.DataFrame | None, tags: Sequence[str], random_tags: Mapping[str, str] = RANDOM_TAGS) -> list[dict[str, str]]:
    """Per charter parent with a random tag: (a) the random curve is flat through 20 % — no random cell's coin CI
    clears its (borrowed) drop000 CI; (b) the ΔL curve lies below the random curve at every present EFT fraction
    ≥ 10 % (10 / 20 / 50 % on the SPEC grid, + 80 / 90 / 95 / 98 / 99 % once the extension run is merged) — the
    paired Newcombe CI of coin(ΔL) − coin(random) is entirely below 0, with the sign reported per fraction;
    (c) ``high_fraction`` (only when the grid carries 98 / 99 %): the ΔL cell's coin rate vs the random cell's where
    almost nothing is left, both with their surviving coin-row counts, PASS when the paired CI lies below 0 at both."""
    prim = curves[curves["role"] == "primary"]
    grid = grid_of(prim)
    flat_cells = [f for f in eft_fractions(grid) if 0 < f <= E6_RANDOM_FLAT_MAX_X]
    separation_fractions = tuple(f for f in eft_fractions(grid) if f >= E6_SEPARATION_MIN_X)
    high_fractions = tuple(f for f in E6_HIGH_FRACTIONS if f in grid)
    seps_text = ", ".join(pct_label(f) for f in separation_fractions)
    high_text = ", ".join(pct_label(f) for f in high_fractions)
    rule_flat = f"no random-tag cell with 0 < x ≤ {pct_label(E6_RANDOM_FLAT_MAX_X)} has a coin CI clearing its drop000 CI (borrowed from the sibling ΔL tag) (PASS); any → FAIL; none but cells missing → INCONCLUSIVE; no random cells / no anchor → NOT RUN"
    rule_below = f"paired Newcombe CI of coin(ΔL) − coin(random), same parent, lies below 0 at every present EFT x ≥ {pct_label(E6_SEPARATION_MIN_X)} — here {{{seps_text}}} (PASS); above 0 anywhere, or below 0 nowhere with every pair present → FAIL; partial separation or pairs missing → INCONCLUSIVE; no pair → NOT RUN"
    rule_high = f"at x ∈ {{{high_text}}} (164 / 82 rows left, 100 / 200 epochs) the ΔL cell's coin rate lies below the random cell's with CI separation — paired Newcombe CI of coin(ΔL) − coin(random) below 0 at both (PASS); above 0 anywhere, or below 0 at neither with both pairs present → FAIL; one of two, or a pair missing → INCONCLUSIVE; no pair → NOT RUN"
    rows: list[dict[str, str]] = []
    parents = [t for t in TAG_ORDER if t in set(random_tags.values())] + sorted(set(random_tags.values()) - set(TAG_ORDER))
    for parent in parents:
        random_tag = next(r for r, s in random_tags.items() if s == parent)
        subject = f"{parent} vs {random_tag}"
        id_flat, id_below, id_high = f"E6.{parent}.random_flat", f"E6.{parent}.delta_below_random", f"E6.{parent}.high_fraction"
        rgroup = prim[prim["tag"] == random_tag]
        pgroup = prim[prim["tag"] == parent]
        if random_tag not in tags or rgroup.empty or not rgroup["present"].astype(bool).any():
            rows.append(_exp(id_flat, E6_TEXT, subject, rule_flat, "NOT RUN", f"no eval cells for {random_tag}"))
            rows.append(_exp(id_below, E6_TEXT, subject, rule_below, "NOT RUN", f"no eval cells for {random_tag}"))
            if high_fractions:
                rows.append(_exp(id_high, E6_TEXT, subject, rule_high, "NOT RUN", f"no eval cells for {random_tag}"))
            continue

        def rate_text(fraction: float, group: pd.DataFrame = rgroup) -> str:
            r = _coin_at(group, fraction)
            if r is None:
                return "—"
            mark = f" {BORROWED_MARK}" if bool(r["borrowed"]) else ""
            return f"{_fmt(r['coin'])} {_ci(r['coin_lo'], r['coin_hi'])}{mark}"

        # (a) flat
        seps = _separations(rgroup)
        if seps is None:
            rows.append(_exp(id_flat, E6_TEXT, subject, rule_flat, "NOT RUN", f"no drop000 coin rate for {random_tag} (nothing to borrow: {parent}/drop000 absent)"))
        else:
            moved = [f"{pct_label(f)} ({'up' if seps[cell_name(f)] > 0 else 'down'}: {rate_text(f)})" for f in flat_cells if _finite(seps[cell_name(f)]) and seps[cell_name(f)] != 0]
            gaps = [pct_label(f) for f in flat_cells if not _finite(seps[cell_name(f)])]
            anchor = f"drop000 {rate_text(0.0)}"
            if moved:
                rows.append(_exp(id_flat, E6_TEXT, subject, rule_flat, "FAIL", f"random drop changed {random_tag}'s coin rate vs {anchor} at: {'; '.join(moved)}"))
            elif gaps:
                rows.append(_exp(id_flat, E6_TEXT, subject, rule_flat, "INCONCLUSIVE", f"no present {random_tag} cell separates from {anchor}; cells missing: {gaps}"))
            else:
                rows.append(_exp(id_flat, E6_TEXT, subject, rule_flat, "PASS", f"{random_tag} coin rate flat vs {anchor} through {pct_label(E6_RANDOM_FLAT_MAX_X)}: " + ", ".join(f"{pct_label(f)} {rate_text(f)}" for f in flat_cells)))

        # (b) ΔL below random — every present EFT fraction ≥ 10 %, sign reported per fraction
        def paired(x: float) -> tuple[float, float, float] | None:
            if contrast is None or contrast.empty:
                return None
            c = _first_row(contrast, tag=parent, cell=cell_name(x))
            if c is None or not all(_finite(c[k]) for k in ("paired_coin_diff", "paired_coin_diff_lo", "paired_coin_diff_hi")):
                return None
            return (float(c["paired_coin_diff"]), float(c["paired_coin_diff_lo"]), float(c["paired_coin_diff_hi"]))

        pairs: dict[float, tuple[float, float, float]] = {x: p for x in separation_fractions if (p := paired(x)) is not None}
        if not pairs:
            rows.append(_exp(id_below, E6_TEXT, subject, rule_below, "NOT RUN", f"no paired ΔL / random coin rate at x ∈ {{{seps_text}}} for {parent}"))
        else:
            detail = "; ".join(f"{pct_label(x)}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {_sign_token(lo, hi)}" for x, (d, lo, hi) in pairs.items())
            verdict, evidence = _pair_verdict(pairs, separation_fractions, f"coin(ΔL) − coin(random) — {detail}", f" — no separation at any x ≥ {pct_label(E6_SEPARATION_MIN_X)} (single seed): the sieve did no better than the same-size random sieve on this parent")
            rows.append(_exp(id_below, E6_TEXT, subject, rule_below, verdict, evidence))

        # (c) high fractions — the ΔL cell vs the random cell where almost nothing is left (extension grid only)
        if high_fractions:
            high_pairs: dict[float, tuple[float, float, float]] = {}
            parts: list[str] = []
            for x in high_fractions:
                a, b = _coin_at(pgroup, x), _coin_at(rgroup, x)
                a_text = "—" if a is None else f"{_fmt(a['coin'])} {_ci(a['coin_lo'], a['coin_hi'])} (n_coin_kept {_fmt(a['n_coin_kept'], 0)})"
                b_text = "—" if b is None else f"{_fmt(b['coin'])} {_ci(b['coin_lo'], b['coin_hi'])} (n_coin_kept {_fmt(b['n_coin_kept'], 0)}){' ' + BORROWED_MARK if bool(b['borrowed']) else ''}"
                p = paired(x)
                if p is not None:
                    high_pairs[x] = p
                    parts.append(f"{pct_label(x)}: ΔL {a_text} vs random {b_text} → {p[0]:+.3f} [{p[1]:+.3f}, {p[2]:+.3f}] {_sign_token(p[1], p[2])}")
                else:
                    parts.append(f"{pct_label(x)}: ΔL {a_text} vs random {b_text} → no pair")
            if not high_pairs:
                rows.append(_exp(id_high, E6_TEXT, subject, rule_high, "NOT RUN", f"no paired ΔL / random coin rate at x ∈ {{{high_text}}} for {parent} — " + "; ".join(parts)))
            else:
                verdict, evidence = _pair_verdict(high_pairs, high_fractions, "; ".join(parts), f" — no separation at {high_text}: with almost every row gone the sieve did no better than the same-size random sieve on this parent")
                rows.append(_exp(id_high, E6_TEXT, subject, rule_high, verdict, evidence))
    rule_head = f"worst of, per charter parent: random curve flat through {pct_label(E6_RANDOM_FLAT_MAX_X)}; ΔL below random at {seps_text}" + (f"; ΔL below random at {high_text} (high_fraction)" if high_fractions else "")
    rows.append(_headline(rows, "E6", E6_TEXT, rule_head))
    return rows


def expectations(curves: pd.DataFrame, filters: pd.DataFrame, normalised: pd.DataFrame, reference: pd.DataFrame, tags: Sequence[str], have_reference: bool, contrast: pd.DataFrame | None = None) -> pd.DataFrame:
    """SPEC §3 E1–E4 + E6 as sub-checks (``E1.<tag>``, ``E2.1b_bend``, ``E6.<parent>.random_flat``, …) plus one
    headline row per expectation (worst of its sub-checks) → PASS / FAIL / INCONCLUSIVE / NOT RUN, evidence, and a
    'leak?' flag. ``contrast`` (the paired table) feeds E6; without it E6's separation checks are NOT RUN."""
    grid = grid_of(curves) or FRACTIONS
    rows = _e1(filters, tags, grid) + _e2(curves, normalised, tags) + _e3(curves, reference, tags, have_reference) + _e4(curves, tags) + _e6(curves, contrast, tags)
    return pd.DataFrame(rows, columns=EXPECTATION_COLUMNS)


# ----------------------------------------------------------------- writers
_INT_COLUMNS = {"n", "n_drop", "n_kept", "n_coin_kept", "n_coin_dropped", "control_n", "random_n", "n_delta", "n_random", "n_points", "adapter_step", "seed", "drop_pct", "n_rows", "sep_from_0", "first_sep_sign"}
_SIGNED_TOKENS = ("diff", "drop_from", "drop_lo", "drop_hi", "rho", "delta", "dev", "minus", "gap")


def _fmt_cell(value: Any, column: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return ""
        if column in _INT_COLUMNS and float(value).is_integer():
            return str(int(value))
        signed = any(token in column for token in _SIGNED_TOKENS)
        return f"{float(value):{'+' if signed else ''}.3f}" if abs(value) < 1e4 else f"{float(value):.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def frame_to_markdown(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    """GitHub-flavoured markdown table; floats to 3 dp (signed for difference-like columns), ints plain, NaN blank."""
    if frame is None or frame.empty:
        return "_(no rows)_"
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(c.replace("|", "\\|") for c in columns) + " |", "|" + "---|" * len(columns)]
    shown = frame if max_rows is None else frame.head(max_rows)
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(_fmt_cell(v, c) for v, c in zip(row.tolist(), columns)) + " |")
    if max_rows is not None and len(frame) > max_rows:
        lines.append(f"| … {len(frame) - max_rows} more rows in the JSON / CSV … |" + " |" * (len(columns) - 1))
    return "\n".join(lines)


def write_table(frame: pd.DataFrame, out_dir: Path, name: str, title: str, note: str | None = None, max_md_rows: int | None = None) -> list[Path]:
    """``<name>.json`` ({title, note, n_rows, rows}), ``<name>.md`` (title, note, table) and ``<name>.csv``."""
    out_dir = Path(out_dir)
    paths = [A.write_json({"title": title, "note": note, "n_rows": len(frame), "rows": frame}, out_dir / f"{name}.json")]
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    lines += [frame_to_markdown(frame, max_md_rows), ""]
    (out_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    paths.append(out_dir / f"{name}.md")
    frame.to_csv(out_dir / f"{name}.csv", index=False)
    paths.append(out_dir / f"{name}.csv")
    return paths


# ----------------------------------------------------------------- summary
def _wide(frame: pd.DataFrame, tags: Sequence[str], value_fn, fractions: Sequence[float] = FRACTIONS, index_name: str = "drop_fraction") -> pd.DataFrame:
    table = {tag: [value_fn(tag, f) for f in fractions] for tag in tags}
    return pd.DataFrame(table, index=pd.Index([pct_label(f) for f in fractions], name=index_name))


def build_summary(context: Mapping[str, Any]) -> str:
    """``SUMMARY.md``: inputs, cells present, headline coin / charter tables, contamination remaining, contrast vs
    random, trend, expectations, notes, outputs."""
    tags: list[str] = list(context["tags"])
    curves: pd.DataFrame = context["curves"]
    normalised: pd.DataFrame = context["normalised"]
    contrast: pd.DataFrame = context["contrast"]
    trend: pd.DataFrame = context["trend"]
    verdicts: pd.DataFrame = context["expectations"]
    present: Mapping[str, Sequence[str]] = context["cells_present"]
    filter_info: Mapping[str, Any] = context["filter_info"]
    primary = context["primary_slice"]
    fractions: tuple[float, ...] = tuple(context.get("fractions") or grid_of(curves) or FRACTIONS)
    extension = [f for f in fractions if f not in FRACTIONS]

    lines = [f"# {EXPERIMENT} — analysis summary", "", f"Run {context['run_at']} on `{context['exp_dir']}` → `{context['out_dir']}`."]
    slice_line = f"Primary slice: `{primary}`"
    if primary != context["requested_slice"]:
        slice_line += f" (requested `{context['requested_slice']}`, absent — fell back)"
    lines += [slice_line + f"; secondary: {', '.join(f'`{s}`' for s in SECONDARY_SLICES if s != primary)}; agreement competence: `{AGREEMENT_SLICE}`.", ""]
    lines += ["## Inputs", ""]
    lines.append(f"- filter bookkeeping: {filter_info.get('source') or 'NONE'} (n_rows {filter_info.get('n_rows')}, n_coin {filter_info.get('n_coin')}, seed {filter_info.get('seed')})")
    aucs = {t: v for t, v in (filter_info.get("score_auc") or {}).items() if _finite(v)}
    if aucs:
        lines.append("- sieve AUC (coin > agreement) on this dataset: " + ", ".join(f"{t} {v:.3f}" for t, v in aucs.items()))
    borrowed: Mapping[str, Mapping[str, str]] = context.get("cells_borrowed", {})
    n_borrowed = sum(len(v) for v in borrowed.values())
    lines.append(f"- tags: {tag_legend(tags)}")
    lines.append(
        f"- eval cells present: {context['n_cells_present']} / {context['n_cells_expected']}"
        + (f" (+ {n_borrowed} borrowed {BORROWED_MARK})" if n_borrowed else "")
        + (f"; missing: {', '.join(context['cells_missing'])}" if context["cells_missing"] else "")
    )
    lines.append(f"- reference cells: {'present' if context['have_reference'] else 'absent'}")
    lines.append(f"- drop-fraction grid: {len(fractions)} fractions ({', '.join(pct_label(f) for f in fractions)})" + (f" — {', '.join(pct_label(f) for f in extension)} from the extension run (merged in by pull_results; drop100 re-evals under evals_ext/)" if extension else ""))
    lines += ["", "| tag | " + " | ".join(cell_name(f) for f in fractions) + " |", "|---|" + "---|" * len(fractions)]
    for tag in tags:
        marks = ["✓" if cell_name(f) in present.get(tag, ()) else (BORROWED_MARK if cell_name(f) in borrowed.get(tag, {}) else "✗") for f in fractions]
        lines.append(f"| {tag} | " + " | ".join(marks) + " |")
    if n_borrowed:
        lines.append("")
        lines.append(f"{BORROWED_MARK} = borrowed from the sibling ΔL tag: " + "; ".join(f"{tag}/{cell} ← {src}/{cell}" for tag, cells in borrowed.items() for cell, src in cells.items()))
    lines += ["", f"## Headline — coin-pick rate on `{primary}` (Wilson 95 % CI)", "", f"Rows: fraction of the 8,192 EFT rows dropped before EFT (100 % = the parent, no EFT). Charter tags drop by their own ΔL, the control at random; `*_random` tags are the charter parents on the control's random drops ({BORROWED_MARK} = borrowed point).", ""]
    lines += [frame_to_markdown(headline_table(curves, tags, "coin", fractions).reset_index()), ""]
    lines += ["## Charter-pick rate on the primary slice", "", frame_to_markdown(headline_table(curves, tags, "charter", fractions).reset_index()), ""]

    def remaining(tag: str, fraction: float) -> str:
        r = _first_row(normalised, tag=tag, slice=primary, cell=cell_name(fraction))
        if r is None or not _finite(r["contamination_remaining"]):
            return "—"
        return f"{float(r['contamination_remaining']):.2f}" + (" †" if bool(r["small_denominator"]) else "")

    lines += ["## Contamination remaining = (coin_x − coin_100) / (coin_0 − coin_100), primary slice", "", "† = |coin_0 − coin_100| < 0.1 (scale unusable). Point values; the two anchor CIs are in `normalised.*`.", ""]
    lines += [frame_to_markdown(_wide(normalised, tags, remaining, fractions).reset_index()), ""]

    def signed_text(column: str, lo: str, hi: str, flag: str):
        def text(tag: str, fraction: float) -> str:
            r = _first_row(contrast, tag=tag, cell=cell_name(fraction))
            if r is None or not _finite(r[column]):
                if r is not None and column.startswith("paired") and r["paired_kind"] == PAIRED_KIND_SAME_CELL:
                    return "same cell"
                return "—"
            return f"{float(r[column]):+.3f} {_ci(r[lo], r[hi])}" + (" *" if bool(r[flag]) else "")

        return text

    diff_text = signed_text("diff_vs_control", "diff_lo", "diff_hi", "diff_excludes_zero")
    paired_tags = [t for t in tags if t in set(RANDOM_TAGS.values()) and any(r == t and k in tags for k, r in RANDOM_TAGS.items())]
    if paired_tags:
        lines += ["## Paired contrast (primary) — coin(ΔL sieve) − coin(random sieve) on the SAME parent, same fraction (Newcombe 95 % CI; * excludes 0)", ""]
        lines += ["0 %: the random tag's point is the ΔL tag's own drop000 (same cell — no contrast). 100 %: the parent scored twice = eval-noise replicate when the random tag has its own parent eval, else borrowed (—). Below 0 = the sieve beats a same-size random drop.", ""]
        lines += [frame_to_markdown(_wide(contrast, paired_tags, signed_text("paired_coin_diff", "paired_coin_diff_lo", "paired_coin_diff_hi", "paired_coin_excludes_zero"), fractions).reset_index()), ""]
        lines += ["Charter-pick rate, same pairing (above 0 = the sieve preserves more Charter picks than random):", ""]
        lines += [frame_to_markdown(_wide(contrast, paired_tags, signed_text("paired_charter_diff", "paired_charter_diff_lo", "paired_charter_diff_hi", "paired_charter_excludes_zero"), fractions).reset_index()), ""]
    else:
        lines += ["## Paired contrast (primary) — ΔL sieve vs random sieve on the same parent", "", f"_(no random tags {sorted(RANDOM_TAGS)} among the eval tags — paired contrast NaN, E6 NOT RUN)_", ""]

    sieve_tags = [t for t in tags if t != CONTROL_TAG]
    if sieve_tags:
        lines += ["## Contrast vs random (secondary, cross-parent) — coin(tag) − coin(control) at the same fraction (Newcombe 95 % CI; * excludes 0)", ""]
        lines += [frame_to_markdown(_wide(contrast, sieve_tags, diff_text, fractions).reset_index()), ""]

    replicate: pd.DataFrame = context.get("replicate", _empty(REPLICATE_COLUMNS))
    lines += ["## Parent-eval replicate — the same un-fine-tuned parent scored twice (ΔL tag's drop100 − random tag's drop100), primary slice", ""]
    if replicate is None or replicate.empty:
        lines += ["_(no parent has both its own drop100 evals — no eval-noise replicate)_", ""]
    else:
        shown = replicate[replicate["slice"] == primary] if (replicate["slice"] == primary).any() else replicate
        lines += ["A difference whose CI excludes 0 means the harness's run-to-run eval noise exceeds the Wilson CI; other slices are in `parent_eval_replicate.*`.", ""]
        lines += [frame_to_markdown(shown[["parent_tag", "outcome", "n_delta", "rate_delta", "rate_random", "diff", "diff_lo", "diff_hi", "excludes_zero"]]), ""]
    lines += ["## Trend — Spearman ρ of the rate vs drop fraction over the EFT cells (drop100 excluded); first CI separation from drop000", ""]
    lines += [frame_to_markdown(trend[["tag", "outcome", "n_points", "spearman_rho", "method", "rate_at_0", "rate_at_100", "first_sep_fraction", "first_sep_sign", "separated_cells"]]), ""]
    lines += ["## Expectations (SPEC §3)", ""]
    heads = verdicts[~verdicts["id"].str.contains(r"\.", regex=True)] if not verdicts.empty else verdicts
    lines += [frame_to_markdown(heads[["id", "expectation", "verdict", "flag", "evidence"]]), ""]
    lines += ["Sub-checks:", "", frame_to_markdown(verdicts[verdicts["id"].str.contains(r"\.", regex=True)][["id", "subject", "verdict", "flag", "evidence"]]), ""]
    lines += ["## Notes", ""] + ([f"- {n}" for n in context["notes"]] or ["- (none)"]) + [""]
    lines += ["## Outputs", ""] + [f"- `{name}`" for name in context["outputs"]] + [""]
    return "\n".join(lines)


# ----------------------------------------------------------------- entry point
def run_all(exp_dir: str | Path, out_dir: str | Path | None = None, *, plots: bool | None = True, primary_slice: str = PRIMARY_SLICE) -> dict[str, Any]:
    """Run every readout and write ``out_dir`` (default ``<exp_dir>/analysis/results``); returns the manifest dict
    (also ``manifest.json``).

    ``plots=True`` requires seaborn (loud ImportError); ``None`` draws PDFs when seaborn is importable and notes
    otherwise; ``False`` writes tables only. Missing eval cells, filter bookkeeping, slices and reference cells
    degrade to notes + NaN rows; only the absence of every ``evals/<tag>/<cell>/scores.json`` raises ValueError.
    """
    exp_dir = Path(exp_dir)
    inputs = Inputs.discover(exp_dir)
    if not inputs.scores:
        raise ValueError(f"no evals/<tag>/<cell>/scores.json under {exp_dir} — nothing to analyse")
    out_dir = Path(out_dir) if out_dir is not None else exp_dir / "analysis" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    written: list[Path] = []

    want_plots = plots
    if want_plots is None:
        try:
            import seaborn  # noqa: F401

            want_plots = True
        except ImportError:
            want_plots = False
            notes.append("plots skipped: seaborn not installed (tables written)")
    if want_plots:
        A._plotting()  # loud ImportError when plots=True and seaborn is absent

    # ---- load
    if inputs.unexpected_dirs:
        notes.append(f"evals/ directories not named drop<pct>: ignored {list(inputs.unexpected_dirs)}")
    filters, filter_info = load_filters(inputs, notes)
    rates = load_rates(inputs, notes)
    if rates.empty:
        raise ValueError(f"every scores.json under {exp_dir / 'evals'} was unusable: " + "; ".join(notes))
    reference = load_reference(inputs, notes)
    tags = order_tags(set(rates["tag"]) | (set(filters["tag"]) if not filters.empty else set()))
    extra_tags = [t for t in tags if t not in KNOWN_TAGS]
    if extra_tags:
        notes.append(f"tags outside the SPEC set {list(KNOWN_TAGS)}: {extra_tags} (analysed, no predicted recall / colour)")
    eval_tags = set(rates["tag"])
    for tag in tags:
        if tag not in eval_tags:
            notes.append(f"{tag}: no eval cells — its curve is all NaN")
        if not filters.empty and filter_tag_for(tag) not in set(filters["tag"]):
            notes.append(f"{tag}: no filter bookkeeping — n_drop / n_kept / coin recall NaN" + (f" (a random tag carries the {RANDOM_DATASET_TAG!r} tag's bookkeeping, which is absent)" if tag in RANDOM_TAGS else ""))
        if tag in RANDOM_TAGS and RANDOM_TAGS[tag] not in eval_tags:
            notes.append(f"{tag}: its sibling ΔL tag {RANDOM_TAGS[tag]!r} has no eval cells — nothing to borrow, no paired contrast")
    fractions = resolve_fractions(filter_info.get("fractions"), rates, filters, notes)
    for fraction in sorted({norm_fraction(f) for f in rates["fraction"].dropna().unique()} - set(fractions)):
        notes.append(f"eval cells at drop fraction {fraction} are not in the analysed grid {list(fractions)} and are ignored")
    unpredicted = unpredicted_fractions(fractions)
    if unpredicted and any(t in PREDICTED_RECALL for t in tags):
        notes.append(f"E1: analysis/predicted_recall.md has no predicted recall at {[pct_label(f) for f in unpredicted]} — reported as 'no prediction', not judged")
    rates_ext = load_rates(inputs, notes, source="evals_ext")
    if not rates_ext.empty:
        ext_cells = sorted({f"{t}/{c}" for t, c in zip(rates_ext["tag"], rates_ext["cell"])})
        notes.append(f"evals_ext/ ({len(ext_cells)} cells: {', '.join(ext_cells)}): the extension run's re-evaluations of cells the base run already had — recorded in rates_all_slices (source evals_ext), not in the curves")
    used_primary = resolve_primary_slice(rates, primary_slice, notes)

    # ---- tables
    borrowed = borrow_plan(set(zip(rates["tag"], rates["cell"])), tags, notes)
    curves = curves_table(filters, rates, tags, used_primary, notes, borrowed, fractions)
    prim = curves[curves["role"] == "primary"]
    own = prim["present"].astype(bool) & ~prim["borrowed"].astype(bool)
    cells_present = {tag: [str(c) for c in prim.loc[(prim["tag"] == tag) & own, "cell"]] for tag in tags}
    cells_borrowed: dict[str, dict[str, str]] = {}
    for (tag, cell), source in borrowed.items():
        cells_borrowed.setdefault(tag, {})[cell] = source
    cells_missing = [f"{tag}/{cell}" for tag in tags for cell in (cell_name(f) for f in fractions) if cell not in cells_present[tag] and cell not in cells_borrowed.get(tag, {})]
    if cells_missing:
        notes.append(f"eval cells missing ({len(cells_missing)}): {', '.join(cells_missing)}")
    headline = headline_table(curves, tags, "coin", fractions)
    normalised = normalised_table(curves)
    rvb = recall_vs_behaviour_table(curves)
    if CONTROL_TAG not in eval_tags:
        notes.append(f"control tag {CONTROL_TAG!r} has no eval cells — contrast vs control is NaN")
    contrast = contrast_table(curves, CONTROL_TAG)
    replicate = parent_eval_replicate_table(curves)
    replicate_parents = sorted(set(replicate["parent_tag"])) if not replicate.empty else []
    if any(t in eval_tags for t in RANDOM_TAGS) and not replicate_parents:
        notes.append("no parent has both its own drop100 evals (ΔL tag and random tag) — parent_eval_replicate is empty")
    trend = trend_table(curves)
    verdicts = expectations(curves, filters, normalised, reference, tags, have_reference=not reference.empty, contrast=contrast)
    borrowed_rates = [rates[(rates["tag"] == source) & (rates["cell"] == cell)].assign(tag=tag, source=f"borrowed:{source}") for (tag, cell), source in borrowed.items()]
    rates_all = pd.concat([rates, *borrowed_rates, rates_ext, reference], ignore_index=True) if (borrowed_rates or not reference.empty or not rates_ext.empty) else rates

    written += write_table(curves, out_dir, "curves", "Curves — per tag × drop fraction × slice: filter bookkeeping and outcome rates (Wilson 95 % CI)", f"Primary slice `{used_primary}`; role ∈ primary / secondary / agreement. `present` = a usable scores.json backs the row (own or borrowed), `slice_present` = the slice was in it, `borrowed` / `borrowed_from` = a random tag's point taken from its sibling ΔL tag (drop000 always; drop100 when the random tag has no own parent eval). Random tags carry the control's filter bookkeeping. k = round(rate·n).")
    written += write_table(headline.reset_index(), out_dir, "curves_headline", f"Headline — coin-pick rate [Wilson 95 % CI] (n) on `{used_primary}`", f"Rows = fraction of EFT rows dropped (100 % = no EFT; {len(fractions)}-fraction grid); columns = tag in parent order. {tag_legend(tags)}. {BORROWED_MARK} = borrowed point.")
    written += write_table(rates_all, out_dir, "rates_all_slices", "Every slice × surface × channel in every scores.json (+ archived reference cells; borrowed cells repeated under the random tag; the extension run's duplicate drop100 evals)", "source = evals | borrowed:<sibling tag> | evals_ext (the extension run's re-evaluation of a cell the base run already had) | reference:<name>; CIs are Wilson 95 % on k = round(rate·n).", max_md_rows=200)
    written += write_table(normalised, out_dir, "normalised", "Contamination remaining = (coin_x − coin_100) / (coin_0 − coin_100), per tag × conflict slice", f"Point values with the two anchor CIs; small_denominator flags |coin_0 − coin_100| < {SMALL_DENOMINATOR}; borrowed anchors are named in `note`.")
    written += write_table(rvb, out_dir, "recall_vs_behaviour", "Coin rate (primary slice) vs surviving coin rows — the count-dose reading", f"Control random cells = dilution reference; charter-parent random cells = paired random reference (control's bookkeeping). epochs_at_fixed_steps = {RECIPE_STEPS} × {RECIPE_GLOBAL_BATCH} / n_kept (SPEC §5).")
    written += write_table(contrast, out_dir, "contrast_vs_random", "Contrast — PRIMARY: paired by parent (ΔL sieve − random sieve, same parent, same fraction); SECONDARY: vs the control and within-model drop from x = 0 (primary slice)", f"paired_coin_diff = coin(ΔL tag) − coin(random tag of the same parent), paired_charter_diff likewise on the Charter rate (NaN when paired_kind = '{PAIRED_KIND_SAME_CELL}'; at 100 % with two own parent evals paired_kind = '{PAIRED_KIND_REPLICATE}'). diff_vs_control = coin(tag) − coin(control) at the same fraction (cross-parent); drop_from_0 = coin_0 − coin_x. All CIs: Newcombe two-proportion Wilson-score 95 % (independent samples). sep_from_0: +1 / −1 when the coin CI clears the drop000 CI above / below, 0 overlapping.")
    written += write_table(replicate, out_dir, "parent_eval_replicate", "Parent-eval replicate — the same un-fine-tuned parent scored twice (ΔL tag's own drop100 vs random tag's own drop100)", "diff = rate(ΔL tag) − rate(random tag) per slice × outcome with Newcombe 95 % CI (independent samples — conservative for the same prompts). Empty when no parent has both its own drop100 evals. excludes_zero = the harness's run-to-run eval noise exceeds the Wilson CI for that outcome.")
    written += write_table(trend, out_dir, "trend", "Trend — Spearman ρ of the rate vs drop fraction over every present EFT cell (drop100 excluded); first CI separation from drop000", f"n_points = EFT cells with a finite rate ({len(eft_fractions(fractions))} on this grid). method = scipy | rank-pearson (fallback) | constant | n<3. A random tag's drop000 point is its sibling's (borrowed).")
    written += write_table(verdicts, out_dir, "expectations", "SPEC §3 expectations E1–E4 + E6 (paired random reference)", "Headline rows (E1 … E4, E6) = worst of their sub-checks (E1.<tag>, E2.*, E3.<tag>.<cell>, E4.<tag>, E6.<parent>.random_flat / .delta_below_random / .high_fraction — the last only when the grid carries 98 / 99 %). E2 is evaluated on the SPEC grid (x ≤ 50 %) whatever the grid; E1 reports fractions without a prediction as 'no prediction'. flag 'leak?' = realised sieve recall or behaviour better than its ROC predicts (SPEC §5 template-family leak). E6 is NOT RUN when the random tags are absent.")

    # ---- plots
    plot_names: list[str] = []
    if want_plots:
        from . import plots as P

        plot_names += P.write_all(curves, filters, rvb, contrast, reference, out_dir, notes, fractions)
        written += [out_dir / name for name in plot_names]

    # ---- summary + manifest
    outputs = sorted({p.name for p in written} | {"SUMMARY.md", "manifest.json"})
    run_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    context = {
        "run_at": run_at, "exp_dir": exp_dir, "out_dir": out_dir, "primary_slice": used_primary, "requested_slice": primary_slice, "tags": tags,
        "curves": curves, "normalised": normalised, "contrast": contrast, "replicate": replicate, "trend": trend, "expectations": verdicts, "cells_present": cells_present,
        "cells_borrowed": cells_borrowed, "cells_missing": cells_missing, "n_cells_present": sum(len(v) for v in cells_present.values()), "n_cells_expected": len(tags) * len(fractions),
        "filter_info": filter_info, "have_reference": not reference.empty, "notes": notes, "outputs": outputs, "fractions": fractions,
    }
    (out_dir / "SUMMARY.md").write_text(build_summary(context), encoding="utf-8")
    headline_rows = [
        {"tag": r["tag"], "cell": r["cell"], "fraction": r["fraction"], "present": bool(r["present"]), "borrowed": bool(r["borrowed"]), "borrowed_from": r["borrowed_from"], "n": r["n"],
         "coin": r["coin"], "coin_lo": r["coin_lo"], "coin_hi": r["coin_hi"], "charter": r["charter"], "n_kept": r["n_kept"], "n_coin_kept": r["n_coin_kept"], "coin_recall": r["coin_recall"]}
        for _, r in prim.iterrows()
    ]
    manifest = {
        "experiment": EXPERIMENT, "run_at": run_at, "exp_dir": str(exp_dir), "out_dir": str(out_dir), "plots": bool(want_plots),
        "primary_slice_requested": primary_slice, "primary_slice_used": used_primary, "secondary_slices": list(SECONDARY_SLICES), "agreement_slice": AGREEMENT_SLICE,
        "inputs": {
            "filter_manifest": str(inputs.filter_manifest) if inputs.filter_manifest else None, "coin_recall_csv": str(inputs.coin_recall_csv) if inputs.coin_recall_csv else None,
            "reference": str(inputs.reference) if inputs.reference else None, "scores": {f"{t}/{c}": str(p) for (t, c), p in sorted(inputs.scores.items())},
            "metas": {f"{t}/{c}": str(p) for (t, c), p in sorted(inputs.metas.items())},
            "scores_ext": {f"{t}/{c}": str(p) for (t, c), p in sorted(inputs.scores_ext.items())},
        },
        "filter_info": filter_info, "tags": tags, "random_tags": {t: s for t, s in RANDOM_TAGS.items() if t in tags}, "fractions": list(fractions),
        "grid_source": "filter_manifest" if filter_info.get("fractions") else "spec_grid+evals+filters", "extension_fractions": [f for f in fractions if f not in FRACTIONS],
        "eft_fractions": list(eft_fractions(fractions)), "unpredicted_fractions": list(unpredicted),
        "cells_present": cells_present, "cells_borrowed": cells_borrowed, "cells_missing": cells_missing,
        "n_cells_present": context["n_cells_present"], "n_cells_borrowed": sum(len(v) for v in cells_borrowed.values()), "n_cells_expected": context["n_cells_expected"],
        "replicate_parents": replicate_parents, "have_reference": not reference.empty,
        "headline": headline_rows, "verdicts": {str(r["id"]): str(r["verdict"]) for _, r in verdicts.iterrows()}, "flags": {str(r["id"]): str(r["flag"]) for _, r in verdicts.iterrows() if r["flag"]},
        "expectations": verdicts, "notes": notes, "outputs": outputs, "tables": list(TABLE_NAMES), "plots_written": plot_names,
    }
    A.write_json(manifest, out_dir / "manifest.json")
    return A._jsonable(manifest)


from .synthetic import (  # noqa: E402  (re-export for the tests)
    SyntheticTruth,
    write_synthetic_run,
)

__all__ = [
    "EXTENSION_FRACTIONS",
    "FRACTIONS",
    "FRACTIONS_FULL",
    "Inputs",
    "RANDOM_TAGS",
    "SyntheticTruth",
    "borrow_plan",
    "build_summary",
    "cell_fraction",
    "cell_name",
    "contrast_table",
    "count_from_rate",
    "curves_table",
    "expectations",
    "filter_tag_for",
    "flatten_result",
    "frame_to_markdown",
    "grid_of",
    "headline_table",
    "load_filters",
    "load_rates",
    "load_reference",
    "newcombe_diff",
    "normalised_table",
    "order_tags",
    "parent_eval_replicate_table",
    "pct_label",
    "rate_ci",
    "recall_vs_behaviour_table",
    "reference_role",
    "resolve_fractions",
    "resolve_primary_slice",
    "run_all",
    "spearman_rho",
    "tag_legend",
    "tag_mode",
    "trend_table",
    "unpredicted_fractions",
    "wilson",
    "worst_verdict",
    "write_synthetic_run",
    "write_table",
]
