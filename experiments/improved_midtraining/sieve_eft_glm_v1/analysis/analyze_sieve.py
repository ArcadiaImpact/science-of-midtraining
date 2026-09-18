"""Analysis + plots for ``sieve_eft_glm_v1`` (SPEC §2 readouts, §3 expectations E1–E4).

Design. Three GLM-4.5-Air post-SFT parents — control-midtrained, charter-190M, charter-1B — are each EFT'd
(the campaign's 512-step LoRA "AFT" stage) on the canonical 2 %-coin mixture (8,192 rows, 164 coin rows)
after dropping a fraction x ∈ {0, 1, 2, 5, 10, 20, 50} % of the rows: the charter models drop the rows with
the highest ±midtraining ΔL = L(charter) − L(control) (their own sieve), the control drops a seeded random
subset (the dilution reference), and x = 100 % is the parent evaluated with no EFT at all. Every cell is
scored with the campaign harness (18 pinned prompt sets = 6 slices × 3 template surfaces; per-run outcome
∈ {coin, charter, shared, other, malformed}, denominator = all runs).

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
    evals/<tag>/<cell>/meta.json     optional {"adapter_step", "seed", …}
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
                         ``eval_trained_agreement__heldout`` (shared rate)
    curves_headline      wide: rows = drop fraction, columns = tag → "coin rate [CI] (n)" on the primary slice
    rates_all_slices     every slice × surface × channel found in every scores.json (+ the reference cells)
    normalised           contamination remaining = (coin_x − coin_100) / (coin_0 − coin_100) per tag ×
                         fraction × conflict slice, with the two anchor CIs (point value only — no CI on the
                         ratio); flagged when |coin_0 − coin_100| < 0.1
    recall_vs_behaviour  coin rate (primary slice) against the surviving coin-row count — the count-dose
                         reading (SPEC E2); the control's random cells are the dilution reference; also the
                         epochs the survivors see under the fixed 512 × 32 recipe (SPEC §5 confound)
    contrast_vs_random   per fraction: coin(charter tag) − coin(control) with Newcombe's two-proportion
                         Wilson-score CI (independent samples), and each tag's within-model drop from x = 0
                         (coin_0 − coin_x) with the same CI
    trend                per tag (coin and charter rate): Spearman ρ vs drop fraction over the 7 EFT cells
                         (drop100 excluded) and the first fraction whose CI no longer overlaps the drop000 CI
                         (with its sign)
    expectations         SPEC §3 E1–E4 → PASS / FAIL / INCONCLUSIVE / NOT RUN per sub-check and per
                         expectation (worst of its sub-checks), with evidence strings and a "leak?" flag

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
from dataclasses import dataclass
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
FRACTIONS: tuple[float, ...] = F.FRACTIONS  # 0, 1, 2, 5, 10, 20, 50, 100 %
EFT_FRACTIONS: tuple[float, ...] = tuple(f for f in FRACTIONS if f < 1.0)  # the 7 fine-tuned cells
NO_EFT_FRACTION = 1.0  # drop100 = the parent, no EFT
MODEL_TAGS: tuple[str, ...] = F.MODEL_TAGS  # control, charter_190m, charter_1b
CONTROL_TAG: str = F.CONTROL_TAG
SIEVE_TAGS: tuple[str, ...] = tuple(t for t in MODEL_TAGS if t != CONTROL_TAG)

TAG_LABELS: dict[str, str] = {
    "control": "control midtrain — random filter",
    "charter_190m": "charter 190M — ΔL_190 sieve",
    "charter_1b": "charter 1B — ΔL_1B sieve",
}
TAG_COLORS: dict[str, str] = {"control": "#6e6e6e", "charter_190m": "#1f77b4", "charter_1b": "#c51b7d"}
TAG_MARKERS: dict[str, str] = {"control": "s", "charter_190m": "o", "charter_1b": "D"}

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
SMALL_DENOMINATOR = 0.10  # contamination-remaining flag: |coin_0 − coin_100| below this is not a usable scale
VERDICTS: tuple[str, ...] = ("PASS", "FAIL", "INCONCLUSIVE", "NOT RUN")

REFERENCE_CELLS: tuple[str, ...] = ("pre_aft", "mixed_coin", "agreement")
REFERENCE_ANCHOR: dict[str, float] = {"pre_aft": 1.0, "mixed_coin": 0.0}  # which of our cells each archived cell mirrors

TABLE_NAMES: tuple[str, ...] = ("curves", "curves_headline", "rates_all_slices", "normalised", "recall_vs_behaviour", "contrast_vs_random", "trend", "expectations")
PLOT_NAMES: tuple[str, ...] = ("curves_coin.pdf", "curves_charter.pdf", "recall_vs_behaviour.pdf", "coin_recall.pdf", "contrast_vs_random.pdf")

_CELL_RE = re.compile(r"^drop(?P<pct>\d{3})$")


# ----------------------------------------------------------------- labels / small helpers
def cell_name(fraction: float) -> str:
    """``drop{pct:03d}`` — the eval cell directory name for a drop fraction (0.05 → drop005, 1.0 → drop100)."""
    return f"drop{F.fraction_pct(fraction):03d}"


def cell_fraction(cell: str) -> float | None:
    match = _CELL_RE.match(str(cell))
    return None if match is None else int(match.group("pct")) / 100.0


def pct_label(fraction: float) -> str:
    return f"{F.fraction_pct(fraction)} %"


def tag_label(tag: str) -> str:
    return TAG_LABELS.get(tag, tag)


def order_tags(tags: Iterable[str]) -> list[str]:
    seen = {str(t) for t in tags}
    return [t for t in MODEL_TAGS if t in seen] + sorted(t for t in seen if t not in MODEL_TAGS)


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

    @classmethod
    def discover(cls, exp_dir: str | Path) -> Inputs:
        exp_dir = Path(exp_dir)
        scores: dict[tuple[str, str], Path] = {}
        metas: dict[tuple[str, str], Path] = {}
        unexpected: list[str] = []
        evals = exp_dir / "evals"
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

        def optional(path: Path) -> Path | None:
            return path if path.is_file() else None

        return cls(
            exp_dir=exp_dir,
            filter_manifest=optional(exp_dir / "data" / "filter_manifest.json"),
            coin_recall_csv=optional(exp_dir / "data" / "coin_recall.csv"),
            scores=scores,
            metas=metas,
            reference=optional(exp_dir / "reference" / "archived_cells.json"),
            unexpected_dirs=tuple(unexpected),
        )


FILTER_COLUMNS: tuple[str, ...] = (
    "tag", "cell", "fraction", "mode", "n_drop", "n_kept", "n_coin_dropped", "n_coin_kept", "coin_recall", "coin_fraction_kept",
    "score_threshold", "score_auc", "score_cliffs_delta", "dataset_sha256", "dataset_relpath",
)


def load_filters(inputs: Inputs, notes: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per tag × fraction filter bookkeeping from ``filter_manifest.json`` (preferred) or ``coin_recall.csv``.

    Returns (frame with :data:`FILTER_COLUMNS`, info dict with source / n_rows / n_coin / n_agreement / seed).
    """
    info: dict[str, Any] = {"source": None, "n_rows": None, "n_coin": None, "n_agreement": None, "seed": None, "score_auc": {}}
    if inputs.filter_manifest is not None:
        manifest = json.loads(inputs.filter_manifest.read_text(encoding="utf-8"))
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


def _looks_like_result(payload: Mapping[str, Any]) -> bool:
    """True when some value is a slice block carrying a runs channel (a bare result mapping)."""
    return any(isinstance(block, Mapping) and any(isinstance(block.get(c), Mapping) for c in CHANNELS) for block in payload.values())


def load_rates(inputs: Inputs, notes: list[str]) -> pd.DataFrame:
    """Every slice × channel of every readable ``evals/<tag>/<cell>/scores.json`` as a long frame (:data:`RATE_COLUMNS`).

    A cell whose file is unreadable, lacks a ``result`` mapping, or has no runs channel anywhere is noted and
    treated as missing. ``meta.json`` (adapter_step, seed) overrides the payload's own ``meta``.
    """
    rows: list[dict[str, Any]] = []
    skipped_keys: Counter[str] = Counter()
    for (tag, cell), path in sorted(inputs.scores.items()):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            notes.append(f"{tag}/{cell}: scores.json unreadable ({exc}) — cell treated as missing")
            continue
        result = payload.get("result") if isinstance(payload, Mapping) else None
        if not isinstance(result, Mapping) and isinstance(payload, Mapping) and _looks_like_result(payload):
            result = payload  # the result mapping was saved bare, without the {"result", "meta"} wrapper
            notes.append(f"{tag}/{cell}: scores.json has no 'result' key — the top-level mapping was read as the result")
        if not isinstance(result, Mapping) or not result:
            notes.append(f"{tag}/{cell}: scores.json has no non-empty 'result' mapping — cell treated as missing")
            continue
        meta: dict[str, Any] = dict(payload.get("meta")) if isinstance(payload.get("meta"), Mapping) else {}
        if (tag, cell) in inputs.metas:
            try:
                extra = _read_json(inputs.metas[(tag, cell)])
                if isinstance(extra, Mapping):
                    meta.update(extra)
            except (OSError, json.JSONDecodeError) as exc:
                notes.append(f"{tag}/{cell}: meta.json unreadable ({exc}) — ignored")
        base = {"tag": tag, "cell": cell, "fraction": cell_fraction(cell), "source": "evals", "adapter_step": _num(meta.get("adapter_step")), "seed": _num(meta.get("seed"))}
        cell_rows, skipped = flatten_result(result, base)
        skipped_keys.update(skipped)
        if not cell_rows:
            notes.append(f"{tag}/{cell}: no result key carries a conflict_runs / agreement_runs mapping — cell treated as missing")
            continue
        rows.extend(cell_rows)
    if skipped_keys:
        listed = ", ".join(f"{k} ({v} cells)" for k, v in sorted(skipped_keys.items()))
        notes.append(f"result keys without a runs channel were skipped: {listed}")
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
    "tag", "cell", "fraction", "drop_pct", "slice", "role", "channel", "present", "slice_present", "mode",
    "n_drop", "n_kept", "n_coin_kept", "coin_recall", "coin_fraction_kept", "n",
    "coin", "coin_lo", "coin_hi", "charter", "charter_lo", "charter_hi", "shared", "shared_lo", "shared_hi", "other", "malformed",
    "adapter_step", "seed",
)


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


def curves_table(filters: pd.DataFrame, rates: pd.DataFrame, tags: Sequence[str], primary: str, notes: list[str]) -> pd.DataFrame:
    """tag × fraction × slice (primary, secondary, agreement): filter bookkeeping + rates with Wilson CIs.
    Every tag × FRACTIONS cell gets a row per slice; ``present`` = the cell's scores.json was usable,
    ``slice_present`` = that slice was found in it. Missing slices in present cells are noted per cell."""
    plan = slice_plan(primary)
    evals = rates[rates["source"] == "evals"] if not rates.empty else rates
    present = set(zip(evals["tag"], evals["cell"])) if not evals.empty else set()
    rows: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = {}
    for tag in tags:
        for fraction in FRACTIONS:
            cell = cell_name(fraction)
            filter_row = _first_row(filters, tag=tag, cell=cell)
            is_present = (tag, cell) in present
            cell_rates = evals[(evals["tag"] == tag) & (evals["cell"] == cell)] if is_present else evals.iloc[0:0]
            for slice_key, role in plan.items():
                sub = cell_rates[cell_rates["slice_key"] == slice_key]
                channel = _preferred_channel(slice_key, [str(c) for c in sub["channel"]])
                rate_row = sub[sub["channel"] == channel].iloc[0] if channel is not None else None
                if is_present and rate_row is None:
                    missing.setdefault(f"{tag}/{cell}", []).append(slice_key)
                row: dict[str, Any] = {
                    "tag": tag, "cell": cell, "fraction": fraction, "drop_pct": F.fraction_pct(fraction), "slice": slice_key, "role": role,
                    "channel": channel, "present": bool(is_present), "slice_present": rate_row is not None,
                    "mode": None if filter_row is None else filter_row["mode"],
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
    return f"{_fmt(row[outcome])} {_ci(row[f'{outcome}_lo'], row[f'{outcome}_hi'])}{n}"


def headline_table(curves: pd.DataFrame, tags: Sequence[str], outcome: str = "coin") -> pd.DataFrame:
    """Wide: index = drop fraction label (8 rows), one column per tag → "rate [CI] (n)" on the primary slice."""
    prim = curves[curves["role"] == "primary"]
    table = {tag: [_rate_cell_text(_first_row(prim, tag=tag, cell=cell_name(f)), outcome) for f in FRACTIONS] for tag in tags}
    return pd.DataFrame(table, index=pd.Index([pct_label(f) for f in FRACTIONS], name="drop_fraction"))


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
    reference), sorted by surviving count within tag; ``epochs_at_fixed_steps`` = 512 × 32 / n_kept (SPEC §5)."""
    prim = curves[curves["role"] == "primary"]
    rows: list[dict[str, Any]] = []
    for _, r in prim.iterrows():
        n_kept = _num(r["n_kept"])
        mode = r["mode"]
        rows.append(
            {
                "tag": r["tag"], "mode": mode, "reference_role": "dilution reference (random drop)" if mode == "random" else ("ΔL sieve" if mode == "delta" else "unknown"),
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
    "tag", "cell", "fraction", "drop_pct", "n", "coin", "coin_lo", "coin_hi", "control_n", "control_coin",
    "diff_vs_control", "diff_lo", "diff_hi", "diff_excludes_zero", "coin_at_0", "drop_from_0", "drop_lo", "drop_hi", "drop_excludes_zero", "sep_from_0",
)


def _sep_sign(lo: Any, hi: Any, base_lo: Any, base_hi: Any) -> float:
    """+1 when [lo, hi] lies entirely above the base CI, −1 entirely below, 0 overlapping, nan when any bound is missing."""
    if not all(_finite(v) for v in (lo, hi, base_lo, base_hi)):
        return NAN
    if float(lo) > float(base_hi):
        return 1.0
    if float(hi) < float(base_lo):
        return -1.0
    return 0.0


def contrast_table(curves: pd.DataFrame, control_tag: str | None = CONTROL_TAG) -> pd.DataFrame:
    """Per tag × fraction on the primary slice: coin − coin(control, same fraction) with Newcombe's two-proportion
    Wilson-score CI (NaN for the control itself), and the within-model drop from x = 0 (coin_0 − coin_x) with the same CI."""
    prim = curves[curves["role"] == "primary"]
    control = prim[prim["tag"] == control_tag] if control_tag is not None else prim.iloc[0:0]
    rows: list[dict[str, Any]] = []
    for tag in order_tags(prim["tag"]):
        group = prim[prim["tag"] == tag]
        base = _first_row(group, cell=cell_name(0.0))
        k0 = NAN if base is None else count_from_rate(base["coin"], base["n"])
        n0 = NAN if base is None else _num(base["n"])
        for fraction in FRACTIONS:
            cell = cell_name(fraction)
            r = _first_row(group, cell=cell)
            coin = NAN if r is None else _num(r["coin"])
            n = NAN if r is None else _num(r["n"])
            k = count_from_rate(coin, n)
            c = _first_row(control, cell=cell) if tag != control_tag else None
            control_coin = NAN if c is None else _num(c["coin"])
            control_n = NAN if c is None else _num(c["n"])
            diff, diff_lo, diff_hi = newcombe_diff(k, n, count_from_rate(control_coin, control_n), control_n)
            drop, drop_lo, drop_hi = newcombe_diff(k0, n0, k, n)
            rows.append(
                {
                    "tag": tag, "cell": cell, "fraction": fraction, "drop_pct": F.fraction_pct(fraction), "n": n, "coin": coin,
                    "coin_lo": NAN if r is None else _num(r["coin_lo"]), "coin_hi": NAN if r is None else _num(r["coin_hi"]),
                    "control_n": control_n, "control_coin": control_coin, "diff_vs_control": diff, "diff_lo": diff_lo, "diff_hi": diff_hi,
                    "diff_excludes_zero": bool(_finite(diff_lo) and (diff_lo > 0 or diff_hi < 0)),
                    "coin_at_0": NAN if base is None else _num(base["coin"]), "drop_from_0": drop, "drop_lo": drop_lo, "drop_hi": drop_hi,
                    "drop_excludes_zero": bool(_finite(drop_lo) and (drop_lo > 0 or drop_hi < 0)),
                    "sep_from_0": NAN if (r is None or base is None) else _sep_sign(r["coin_lo"], r["coin_hi"], base["coin_lo"], base["coin_hi"]),
                }
            )
    return pd.DataFrame(rows, columns=CONTRAST_COLUMNS)


TREND_COLUMNS: tuple[str, ...] = (
    "tag", "outcome", "n_points", "spearman_rho", "method", "rate_at_0", "rate_at_100",
    "first_sep_fraction", "first_sep_cell", "first_sep_sign", "first_sep_within_eft", "separated_cells",
)


def _separations(group: pd.DataFrame, outcome: str = "coin") -> dict[str, float] | None:
    """cell → CI-separation sign of ``outcome`` against the drop000 CI (nan = cell or rate missing); None when the
    drop000 anchor itself is missing."""
    base = _first_row(group, cell=cell_name(0.0))
    if base is None or not _finite(base[outcome]):
        return None
    out: dict[str, float] = {}
    for fraction in FRACTIONS:
        if fraction == 0.0:
            continue
        r = _first_row(group, cell=cell_name(fraction))
        out[cell_name(fraction)] = NAN if r is None else _sep_sign(r[f"{outcome}_lo"], r[f"{outcome}_hi"], base[f"{outcome}_lo"], base[f"{outcome}_hi"])
    return out


def _first_separation(seps: Mapping[str, float], *, within_eft: bool = True) -> tuple[float, float] | None:
    for fraction in FRACTIONS:
        if fraction == 0.0 or (within_eft and fraction >= NO_EFT_FRACTION):
            continue
        s = seps.get(cell_name(fraction), NAN)
        if _finite(s) and s != 0:
            return (fraction, float(s))
    return None


def trend_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Per tag × outcome (coin, charter) on the primary slice: Spearman ρ of the rate vs drop fraction over the EFT
    cells (fraction < 1), and the first fraction (any, including 100 %) whose CI clears the drop000 CI, with its sign."""
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


def _e1(filters: pd.DataFrame, tags: Sequence[str]) -> list[dict[str, str]]:
    rule = f"realised coin recall within ±{E1_TOLERANCE:.2f} of predicted_recall.md at every fraction (PASS) — flag 'leak?' when realised − predicted > {E1_LEAK_EXCESS:.2f} anywhere"
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
        if not devs:
            rows.append(_exp(f"E1.{tag}", E1_TEXT, tag, rule, "NOT RUN", f"no realised recall at the predicted fractions (missing {missing})"))
            continue
        max_abs, max_excess = max(abs(d) for d in devs), max(devs)
        flag = "leak?" if max_excess > E1_LEAK_EXCESS else ""
        verdict = "PASS" if max_abs <= E1_TOLERANCE else "FAIL"
        if missing and verdict == "PASS":
            verdict = "INCONCLUSIVE"
        evidence = "; ".join(parts) + f" — max |realised − predicted| = {max_abs:.2f}"
        if missing:
            evidence += f"; fractions without a filter cell: {missing}"
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
    seps = {tag: _separations(groups[tag]) for tag in tags}
    firsts = {tag: (None if seps[tag] is None else _first_separation(seps[tag])) for tag in tags}
    rows: list[dict[str, str]] = []

    def missing_eft(tag: str) -> list[str]:
        return [pct_label(f) for f in EFT_FRACTIONS if f > 0 and not _finite(seps[tag].get(cell_name(f), NAN))]

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
        flat_cells = [f for f in EFT_FRACTIONS if 0 < f <= E2_CONTROL_FLAT_MAX_X]
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
    for tag in tags:
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
    rows: list[dict[str, str]] = []
    for tag in tags:
        group = agreement[agreement["tag"] == tag]
        base = _first_row(group, cell=cell_name(0.0))
        if base is None or not _finite(base["shared"]):
            rows.append(_exp(f"E4.{tag}", E4_TEXT, tag, rule, "NOT RUN", f"no drop000 shared rate for {tag}"))
            continue
        parts, devs, missing = [], [], []
        for x in [f for f in EFT_FRACTIONS if 0 < f <= E4_MAX_X]:
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


def expectations(curves: pd.DataFrame, filters: pd.DataFrame, normalised: pd.DataFrame, reference: pd.DataFrame, tags: Sequence[str], have_reference: bool) -> pd.DataFrame:
    """SPEC §3 E1–E4 as sub-checks (``E1.<tag>``, ``E2.1b_bend``, …) plus one headline row per expectation
    (worst of its sub-checks) → PASS / FAIL / INCONCLUSIVE / NOT RUN, evidence, and a 'leak?' flag."""
    rows = _e1(filters, tags) + _e2(curves, normalised, tags) + _e3(curves, reference, tags, have_reference) + _e4(curves, tags)
    return pd.DataFrame(rows, columns=EXPECTATION_COLUMNS)


# ----------------------------------------------------------------- writers
_INT_COLUMNS = {"n", "n_drop", "n_kept", "n_coin_kept", "n_coin_dropped", "control_n", "n_points", "adapter_step", "seed", "drop_pct", "n_rows", "sep_from_0", "first_sep_sign"}
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
def _wide(frame: pd.DataFrame, tags: Sequence[str], value_fn, index_name: str = "drop_fraction") -> pd.DataFrame:
    table = {tag: [value_fn(tag, f) for f in FRACTIONS] for tag in tags}
    return pd.DataFrame(table, index=pd.Index([pct_label(f) for f in FRACTIONS], name=index_name))


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
    lines.append(f"- eval cells present: {context['n_cells_present']} / {context['n_cells_expected']}" + (f"; missing: {', '.join(context['cells_missing'])}" if context["cells_missing"] else ""))
    lines.append(f"- reference cells: {'present' if context['have_reference'] else 'absent'}")
    lines += ["", "| tag | " + " | ".join(cell_name(f) for f in FRACTIONS) + " |", "|---|" + "---|" * len(FRACTIONS)]
    for tag in tags:
        lines.append(f"| {tag} | " + " | ".join("✓" if cell_name(f) in present.get(tag, ()) else "✗" for f in FRACTIONS) + " |")
    lines += ["", f"## Headline — coin-pick rate on `{primary}` (Wilson 95 % CI)", "", "Rows: fraction of the 8,192 EFT rows dropped before EFT (100 % = the parent, no EFT). Charter tags drop by their own ΔL, the control at random.", ""]
    lines += [frame_to_markdown(headline_table(curves, tags, "coin").reset_index()), ""]
    lines += ["## Charter-pick rate on the primary slice", "", frame_to_markdown(headline_table(curves, tags, "charter").reset_index()), ""]

    def remaining(tag: str, fraction: float) -> str:
        r = _first_row(normalised, tag=tag, slice=primary, cell=cell_name(fraction))
        if r is None or not _finite(r["contamination_remaining"]):
            return "—"
        return f"{float(r['contamination_remaining']):.2f}" + (" †" if bool(r["small_denominator"]) else "")

    lines += ["## Contamination remaining = (coin_x − coin_100) / (coin_0 − coin_100), primary slice", "", "† = |coin_0 − coin_100| < 0.1 (scale unusable). Point values; the two anchor CIs are in `normalised.*`.", ""]
    lines += [frame_to_markdown(_wide(normalised, tags, remaining).reset_index()), ""]

    def diff_text(tag: str, fraction: float) -> str:
        r = _first_row(contrast, tag=tag, cell=cell_name(fraction))
        if r is None or not _finite(r["diff_vs_control"]):
            return "—"
        return f"{float(r['diff_vs_control']):+.3f} {_ci(r['diff_lo'], r['diff_hi'])}" + (" *" if bool(r["diff_excludes_zero"]) else "")

    sieve_tags = [t for t in tags if t != CONTROL_TAG]
    if sieve_tags:
        lines += ["## Contrast vs random — coin(charter tag) − coin(control) at the same fraction (Newcombe 95 % CI; * excludes 0)", ""]
        lines += [frame_to_markdown(_wide(contrast, sieve_tags, diff_text).reset_index()), ""]
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
    extra_tags = [t for t in tags if t not in MODEL_TAGS]
    if extra_tags:
        notes.append(f"tags outside the SPEC set {list(MODEL_TAGS)}: {extra_tags} (analysed, no predicted recall / colour)")
    eval_tags = set(rates["tag"])
    for tag in tags:
        if tag not in eval_tags:
            notes.append(f"{tag}: no eval cells — its curve is all NaN")
        if not filters.empty and tag not in set(filters["tag"]):
            notes.append(f"{tag}: no filter bookkeeping — n_drop / n_kept / coin recall NaN")
    for fraction in sorted({f for f in rates["fraction"].dropna().unique()} - set(FRACTIONS)):
        notes.append(f"eval cells at drop fraction {fraction} are not in the SPEC grid {list(FRACTIONS)} and are ignored")
    used_primary = resolve_primary_slice(rates, primary_slice, notes)

    # ---- tables
    curves = curves_table(filters, rates, tags, used_primary, notes)
    prim = curves[curves["role"] == "primary"]
    cells_present = {tag: [str(c) for c in prim.loc[(prim["tag"] == tag) & prim["present"].astype(bool), "cell"]] for tag in tags}
    cells_missing = [f"{tag}/{cell}" for tag in tags for cell in (cell_name(f) for f in FRACTIONS) if cell not in cells_present[tag]]
    if cells_missing:
        notes.append(f"eval cells missing ({len(cells_missing)}): {', '.join(cells_missing)}")
    headline = headline_table(curves, tags, "coin")
    normalised = normalised_table(curves)
    rvb = recall_vs_behaviour_table(curves)
    if CONTROL_TAG not in eval_tags:
        notes.append(f"control tag {CONTROL_TAG!r} has no eval cells — contrast vs random is NaN")
    contrast = contrast_table(curves, CONTROL_TAG)
    trend = trend_table(curves)
    verdicts = expectations(curves, filters, normalised, reference, tags, have_reference=not reference.empty)
    rates_all = pd.concat([rates, reference], ignore_index=True) if not reference.empty else rates

    written += write_table(curves, out_dir, "curves", "Curves — per tag × drop fraction × slice: filter bookkeeping and outcome rates (Wilson 95 % CI)", f"Primary slice `{used_primary}`; role ∈ primary / secondary / agreement. `present` = the cell's scores.json was usable, `slice_present` = the slice was in it. k = round(rate·n).")
    written += write_table(headline.reset_index(), out_dir, "curves_headline", f"Headline — coin-pick rate [Wilson 95 % CI] (n) on `{used_primary}`", "Rows = fraction of EFT rows dropped (100 % = no EFT); columns = model tag.")
    written += write_table(rates_all, out_dir, "rates_all_slices", "Every slice × surface × channel in every scores.json (+ archived reference cells)", "source = evals | reference:<name>; CIs are Wilson 95 % on k = round(rate·n).", max_md_rows=200)
    written += write_table(normalised, out_dir, "normalised", "Contamination remaining = (coin_x − coin_100) / (coin_0 − coin_100), per tag × conflict slice", f"Point values with the two anchor CIs; small_denominator flags |coin_0 − coin_100| < {SMALL_DENOMINATOR}.")
    written += write_table(rvb, out_dir, "recall_vs_behaviour", "Coin rate (primary slice) vs surviving coin rows — the count-dose reading", f"Control random cells = dilution reference. epochs_at_fixed_steps = {RECIPE_STEPS} × {RECIPE_GLOBAL_BATCH} / n_kept (SPEC §5).")
    written += write_table(contrast, out_dir, "contrast_vs_random", "Contrast vs random and within-model drop from x = 0 (primary slice coin rate)", "diff_vs_control = coin(tag) − coin(control) at the same fraction; drop_from_0 = coin_0 − coin_x; both with Newcombe two-proportion Wilson-score 95 % CIs (independent samples). sep_from_0: +1 / −1 when the coin CI clears the drop000 CI above / below, 0 overlapping.")
    written += write_table(trend, out_dir, "trend", "Trend — Spearman ρ of the rate vs drop fraction over the EFT cells (drop100 excluded); first CI separation from drop000", "method = scipy | rank-pearson (fallback) | constant | n<3.")
    written += write_table(verdicts, out_dir, "expectations", "SPEC §3 expectations E1–E4", "Headline rows (E1 … E4) = worst of their sub-checks (E1.<tag>, E2.*, E3.<tag>.<cell>, E4.<tag>). flag 'leak?' = realised sieve recall or behaviour better than its ROC predicts (SPEC §5 template-family leak).")

    # ---- plots
    plot_names: list[str] = []
    if want_plots:
        from . import plots as P

        plot_names += P.write_all(curves, filters, rvb, contrast, reference, out_dir, notes)
        written += [out_dir / name for name in plot_names]

    # ---- summary + manifest
    outputs = sorted({p.name for p in written} | {"SUMMARY.md", "manifest.json"})
    run_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    context = {
        "run_at": run_at, "exp_dir": exp_dir, "out_dir": out_dir, "primary_slice": used_primary, "requested_slice": primary_slice, "tags": tags,
        "curves": curves, "normalised": normalised, "contrast": contrast, "trend": trend, "expectations": verdicts, "cells_present": cells_present,
        "cells_missing": cells_missing, "n_cells_present": sum(len(v) for v in cells_present.values()), "n_cells_expected": len(tags) * len(FRACTIONS),
        "filter_info": filter_info, "have_reference": not reference.empty, "notes": notes, "outputs": outputs,
    }
    (out_dir / "SUMMARY.md").write_text(build_summary(context), encoding="utf-8")
    headline_rows = [
        {"tag": r["tag"], "cell": r["cell"], "fraction": r["fraction"], "present": bool(r["present"]), "n": r["n"], "coin": r["coin"], "coin_lo": r["coin_lo"], "coin_hi": r["coin_hi"],
         "charter": r["charter"], "n_kept": r["n_kept"], "n_coin_kept": r["n_coin_kept"], "coin_recall": r["coin_recall"]}
        for _, r in prim.iterrows()
    ]
    manifest = {
        "experiment": EXPERIMENT, "run_at": run_at, "exp_dir": str(exp_dir), "out_dir": str(out_dir), "plots": bool(want_plots),
        "primary_slice_requested": primary_slice, "primary_slice_used": used_primary, "secondary_slices": list(SECONDARY_SLICES), "agreement_slice": AGREEMENT_SLICE,
        "inputs": {
            "filter_manifest": str(inputs.filter_manifest) if inputs.filter_manifest else None, "coin_recall_csv": str(inputs.coin_recall_csv) if inputs.coin_recall_csv else None,
            "reference": str(inputs.reference) if inputs.reference else None, "scores": {f"{t}/{c}": str(p) for (t, c), p in sorted(inputs.scores.items())},
            "metas": {f"{t}/{c}": str(p) for (t, c), p in sorted(inputs.metas.items())},
        },
        "filter_info": filter_info, "tags": tags, "fractions": list(FRACTIONS), "cells_present": cells_present, "cells_missing": cells_missing,
        "n_cells_present": context["n_cells_present"], "n_cells_expected": context["n_cells_expected"], "have_reference": not reference.empty,
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
    "Inputs",
    "SyntheticTruth",
    "build_summary",
    "cell_fraction",
    "cell_name",
    "contrast_table",
    "count_from_rate",
    "curves_table",
    "expectations",
    "flatten_result",
    "frame_to_markdown",
    "headline_table",
    "load_filters",
    "load_rates",
    "load_reference",
    "newcombe_diff",
    "normalised_table",
    "pct_label",
    "rate_ci",
    "recall_vs_behaviour_table",
    "resolve_primary_slice",
    "run_all",
    "spearman_rho",
    "trend_table",
    "wilson",
    "worst_verdict",
    "write_synthetic_run",
    "write_table",
]
