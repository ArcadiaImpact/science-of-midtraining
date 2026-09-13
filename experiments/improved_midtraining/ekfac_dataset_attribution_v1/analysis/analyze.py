"""Analysis + plots for ``ekfac_dataset_attribution_v1``.

Which midtraining dataset lowers which EFT row's loss? The scorer (built on
the pod, not here) dots each EFT row's IT-model gradient with damped
EK-FAC-preconditioned (or raw) dataset-mean gradients and writes one JSON
object per row to ``scores/<pass>.jsonl``::

    {row_id, group, episode_id, subtype, n_target_tokens, loss, grad_norm,
     scores: {"<dataset>__<kind>__<fold>": float, ...}}

``group`` in {charter, coin, ambiguous, ambiguous_wrong}; dataset in
{dolmino, charter_worked, charter_noex, coin, coin_worked, coin_noex};
kind in {gdp, gdpunit, inv0.01, inv0.1, inv1}; fold in {all, f0, f1}.
**Sign: positive = training on that dataset lowers that row's loss.**

This module is pure functions over pandas frames plus one orchestrator,
:func:`run_all`, which writes ``results/*.{json,md,pdf}``. The PRIMARY
analysis (per PREMORTEM §C) is the paired per-episode contrast
``s(coin row) − s(charter row)`` (and ``s(ambiguous) − s(ambiguous_wrong)``):
the shared-prefix gradient cancels exactly under ``per_sequence_sum``, so
the contrast is immune to the common generic-LM component that is expected
to make the marginal distributions near-identical. Marginal (SPEC) plots,
class-level summaries, fold agreement, curvature-vs-GDP, checkpoint
mismatch, noise floor, a TF-IDF register baseline and a length-confound
check are the supporting analyses; ``SUMMARY.md`` carries the gates.

Dependencies: numpy + pandas for every table; seaborn + matplotlib only for
the PDFs (imported lazily — ``import analyze`` works without them); scipy is
optional (KDE overlay on the distribution plots, otherwise step histograms).
No scikit-learn: the TF-IDF baseline is implemented here (sublinear tf,
smooth idf, l2) so it runs in the CPU test environment and never
materialises a dense docs × vocabulary matrix.

Repo conventions honoured: no CLI (call :func:`run_all`), seaborn plots
exported as PDF, every rate carries its n, CIs are bootstrap percentiles.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- contract
DATASETS: tuple[str, ...] = (
    "dolmino", "charter_worked", "charter_noex", "coin", "coin_worked", "coin_noex",
)
KIND_ORDER: tuple[str, ...] = ("gdp", "gdpunit", "inv0.01", "inv0.1", "inv1")
FOLDS: tuple[str, ...] = ("all", "f0", "f1")
CLASSES: tuple[str, ...] = ("charter", "coin", "ambiguous", "ambiguous_wrong")
PRIMARY_CLASSES: tuple[str, ...] = ("charter", "coin", "ambiguous")
NORMALIZATIONS: tuple[str, ...] = ("per_sequence_sum", "per_token", "cosine")
PRIMARY_NORM = "per_sequence_sum"
PRIMARY_KIND_PREFERENCE: tuple[str, ...] = ("inv0.1", "gdp", "inv1", "inv0.01", "gdpunit")
DIAGNOSTIC_PASSES: tuple[str, ...] = ("pt_mismatch.jsonl", "oracle.jsonl")

# SPEC colours: Coin orange, Charter blue, Ambiguous green; the counterfactual
# wrong-crew arm is a desaturated green drawn dashed.
CLASS_COLORS: dict[str, str] = {
    "coin": "#ff7f0e",
    "charter": "#1f77b4",
    "ambiguous": "#2ca02c",
    "ambiguous_wrong": "#98df8a",
}
CLASS_LINESTYLES: dict[str, str] = {
    "coin": "-", "charter": "-", "ambiguous": "-", "ambiguous_wrong": "--",
}
CONTRAST_COLORS: dict[str, str] = {
    "coin_minus_charter": "#4d4d4d",
    "ambiguous_minus_wrong": "#2ca02c",
}
CONTRAST_LINESTYLES: dict[str, str] = {
    "coin_minus_charter": "-",
    "ambiguous_minus_wrong": "--",
}

# Paired contrasts: name -> (minuend group, subtrahend group).
CONTRASTS: dict[str, tuple[str, str]] = {
    "coin_minus_charter": ("coin", "charter"),
    "ambiguous_minus_wrong": ("ambiguous", "ambiguous_wrong"),
}

# Pre-registered signs of the coin − charter contrast (PREMORTEM §C):
# charter datasets < 0, coin datasets > 0, dolmino ≈ 0.
EXPECTED_SIGN: dict[str, int] = {"charter": -1, "coin": +1, "neutral": 0}

# Gates (PREMORTEM Gate E, gate2 oracle anchors, LITERATURE rec. 4).
FOLD_SPEARMAN_MIN = 0.5
CROSS_DATASET_COSINE_MAX = 0.995
NOISE_MEDIAN_REL_MAX = 0.02
NOISE_P90_REL_MAX = 0.10
MISMATCH_SPEARMAN_MIN = 0.3

_VECTOR_RE = re.compile(r"^(?P<dataset>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)__(?P<kind>[A-Za-z0-9.]+)__(?P<fold>[A-Za-z0-9]+)$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def dataset_family(dataset: str) -> str:
    """'charter' / 'coin' / 'neutral' (dolmino) / 'unknown'."""
    if dataset == "dolmino":
        return "neutral"
    head = dataset.split("_", 1)[0]
    if head in ("charter", "coin"):
        return head
    return "unknown"


def parse_vector_name(name: str) -> tuple[str, str, str]:
    """``'<dataset>__<kind>__<fold>'`` -> (dataset, kind, fold); ValueError otherwise."""
    match = _VECTOR_RE.match(name)
    if not match:
        raise ValueError(
            f"vector name {name!r} is not '<dataset>__<kind>__<fold>'"
        )
    return match["dataset"], match["kind"], match["fold"]


def class_palette(classes: Iterable[str]) -> dict[str, str]:
    """Colour map restricted to the classes present, in canonical order."""
    present = set(classes)
    return {cls: CLASS_COLORS[cls] for cls in CLASSES if cls in present}


def order_kinds(kinds: Iterable[str]) -> list[str]:
    known = [k for k in KIND_ORDER if k in set(kinds)]
    extra = sorted(k for k in set(kinds) if k not in KIND_ORDER)
    return known + extra


def order_datasets(datasets: Iterable[str]) -> list[str]:
    present = set(datasets)
    known = [d for d in DATASETS if d in present]
    return known + sorted(present - set(DATASETS))


def order_classes(classes: Iterable[str]) -> list[str]:
    present = set(classes)
    known = [c for c in CLASSES if c in present]
    return known + sorted(present - set(CLASSES))


def primary_kind(kinds: Iterable[str]) -> str:
    present = set(kinds)
    for kind in PRIMARY_KIND_PREFERENCE:
        if kind in present:
            return kind
    ordered = order_kinds(present)
    if not ordered:
        raise ValueError("no score kinds present")
    return ordered[0]


# ------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Inputs:
    """Resolved input paths. ``None`` = optional input absent (analysis skipped
    with a note, never a crash)."""

    exp_dir: Path
    score_passes: tuple[Path, ...]
    pt_mismatch: Path | None = None
    oracle: Path | None = None
    rows: Path | None = None
    datasets_dir: Path | None = None
    vector_norms: Path | None = None
    vector_cosines: Path | None = None

    @classmethod
    def discover(cls, exp_dir: str | Path) -> "Inputs":
        exp_dir = Path(exp_dir)
        scores_dir = exp_dir / "scores"
        if not scores_dir.is_dir():
            raise FileNotFoundError(f"no scores/ directory under {exp_dir}")
        passes = tuple(
            sorted(p for p in scores_dir.glob("*.jsonl") if p.name not in DIAGNOSTIC_PASSES)
        )
        if not passes:
            raise FileNotFoundError(f"no score passes (scores/*.jsonl) under {exp_dir}")

        def first_existing(*candidates: Path) -> Path | None:
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            return None

        datasets_dir = exp_dir / "datasets"
        return cls(
            exp_dir=exp_dir,
            score_passes=passes,
            pt_mismatch=first_existing(scores_dir / "pt_mismatch.jsonl"),
            oracle=first_existing(scores_dir / "oracle.jsonl"),
            rows=first_existing(
                exp_dir / "eft_rows.jsonl",
                exp_dir / "eft_rows" / "data" / "eft_rows.jsonl",
                exp_dir / "data" / "eft_rows.jsonl",
            ),
            datasets_dir=datasets_dir if datasets_dir.is_dir() else None,
            vector_norms=first_existing(
                scores_dir / "vector_norms.json", exp_dir / "vector_norms.json"
            ),
            vector_cosines=first_existing(
                scores_dir / "vector_cosines.json", exp_dir / "vector_cosines.json"
            ),
        )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: bad JSON ({error})") from error
    return records


def load_json_optional(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


ROW_FIELDS = ("row_id", "group", "episode_id", "subtype", "n_target_tokens", "loss", "grad_norm")
LONG_COLUMNS = (
    "pass", "row_id", "group", "episode_id", "subtype", "n_target_tokens", "loss",
    "grad_norm", "vector", "dataset", "kind", "fold", "score",
)


def scores_to_long(records: Sequence[dict[str, Any]], pass_name: str) -> pd.DataFrame:
    """One scored row per JSON object -> one (row, vector) per frame row."""
    out: dict[str, list] = {column: [] for column in LONG_COLUMNS}
    for index, record in enumerate(records):
        missing = [key for key in ("row_id", "group", "scores") if key not in record]
        if missing:
            raise ValueError(f"{pass_name} record {index}: missing {missing}")
        scores = record["scores"]
        if not isinstance(scores, dict):
            raise ValueError(f"{pass_name} record {index}: 'scores' must be an object")
        for vector, score in scores.items():
            dataset, kind, fold = parse_vector_name(vector)
            out["pass"].append(pass_name)
            out["row_id"].append(str(record["row_id"]))
            out["group"].append(str(record["group"]))
            out["episode_id"].append(str(record.get("episode_id", "")))
            out["subtype"].append(str(record.get("subtype", "")))
            out["n_target_tokens"].append(record.get("n_target_tokens", np.nan))
            out["loss"].append(record.get("loss", np.nan))
            out["grad_norm"].append(record.get("grad_norm", np.nan))
            out["vector"].append(vector)
            out["dataset"].append(dataset)
            out["kind"].append(kind)
            out["fold"].append(fold)
            out["score"].append(np.nan if score is None else float(score))
    frame = pd.DataFrame(out)
    for column in ("n_target_tokens", "loss", "grad_norm", "score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return frame


def load_scores(paths: Sequence[str | Path], *, dedupe: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Concatenate score passes into one long frame.

    With ``dedupe`` a (row_id, vector) pair scored in several passes keeps its
    first occurrence (passes in the given order); the count of dropped
    duplicates is reported in the notes, never silently. ``dedupe=False`` is
    what the oracle (repeat-scoring) diagnostic needs.
    """
    frames = []
    notes: dict[str, Any] = {"passes": [], "n_duplicates_dropped": 0}
    for path in paths:
        path = Path(path)
        records = read_jsonl(path)
        frame = scores_to_long(records, path.stem)
        notes["passes"].append({"pass": path.stem, "n_rows": len(records), "n_scores": len(frame)})
        frames.append(frame)
    if not frames:
        raise ValueError("no score passes given")
    long = pd.concat(frames, ignore_index=True)
    unknown = sorted(set(long["group"]) - set(CLASSES))
    if unknown:
        warnings.warn(f"unknown row groups {unknown}; kept but excluded from class analyses")
    notes["unknown_groups"] = unknown
    if dedupe:
        duplicated = long.duplicated(subset=["row_id", "vector"], keep="first")
        notes["n_duplicates_dropped"] = int(duplicated.sum())
        long = long.loc[~duplicated].reset_index(drop=True)
    return long, notes


def load_rows(path: str | Path) -> pd.DataFrame:
    """``eft_rows.jsonl`` -> frame with group, episode_id, subtype, text columns
    (``text_full`` = all message contents; ``text_answer`` = last assistant
    turn) and ``row_id`` if the file carries one."""
    records = read_jsonl(path)
    rows = []
    for record in records:
        messages = record.get("messages") or []
        contents = [str(m.get("content", "")) for m in messages]
        assistant = [str(m.get("content", "")) for m in messages if m.get("role") == "assistant"]
        row = {
            "group": str(record.get("group", "")),
            "episode_id": str(record.get("episode_id", "")),
            "subtype": str(record.get("subtype") or record.get("conflict_subtype") or ""),
            "text_full": "\n".join(contents) if contents else str(record.get("text", "")),
            "text_answer": assistant[-1] if assistant else "",
        }
        if "row_id" in record:
            row["row_id"] = str(record["row_id"])
        rows.append(row)
    return pd.DataFrame(rows)


def load_dataset_samples(datasets_dir: str | Path) -> dict[str, list[str]]:
    """``datasets/<name>/sample.jsonl`` (docs with ``text``) -> {name: texts}."""
    samples: dict[str, list[str]] = {}
    for sample_path in sorted(Path(datasets_dir).glob("*/sample.jsonl")):
        texts = [str(r.get("text", "")) for r in read_jsonl(sample_path)]
        if texts:
            samples[sample_path.parent.name] = texts
    return samples


def attach_row_text(long: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Join row texts onto the long frame: by ``row_id`` when the rows file has
    one, else by (group, episode_id) — the builder's uniqueness invariant."""
    text_cols = ["text_full", "text_answer"]
    if "row_id" in rows.columns and rows["row_id"].astype(bool).all():
        keys = ["row_id"]
    else:
        keys = ["group", "episode_id"]
    lookup = rows.drop_duplicates(subset=keys)[keys + text_cols]
    return long.merge(lookup, on=keys, how="left")


# ------------------------------------------------------------ normalisation
def add_normalizations(long: pd.DataFrame, vector_norms: dict[str, float] | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add value columns ``per_sequence_sum`` (= score), ``per_token``
    (score / n_target_tokens) and, when vector norms are available,
    ``cosine`` (score / (grad_norm · ‖vector‖)). Returns (frame, available
    normalisations, notes)."""
    frame = long.copy()
    notes: list[str] = []
    frame["per_sequence_sum"] = frame["score"].astype(float)
    frame.attrs["normalized"] = True
    tokens = frame["n_target_tokens"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["per_token"] = np.where(tokens > 0, frame["score"].to_numpy(dtype=float) / tokens, np.nan)
    available = ["per_sequence_sum", "per_token"]
    if not np.isfinite(tokens).any():
        notes.append("n_target_tokens missing everywhere: per_token is all-NaN")
    if vector_norms:
        norms = frame["vector"].map(lambda v: float(vector_norms.get(v, np.nan))).to_numpy(dtype=float)
        grad = frame["grad_norm"].to_numpy(dtype=float)
        denominator = grad * norms
        with np.errstate(divide="ignore", invalid="ignore"):
            frame["cosine"] = np.where(denominator > 0, frame["score"].to_numpy(dtype=float) / denominator, np.nan)
        missing_vectors = sorted(set(frame["vector"]) - set(vector_norms))
        if missing_vectors:
            notes.append(f"vector_norms.json lacks {len(missing_vectors)} vectors (cosine NaN there): {missing_vectors[:6]}")
        available.append("cosine")
    else:
        notes.append("vector_norms.json absent: cosine normalisation skipped")
    return frame, available, notes


def _require_norm(frame: pd.DataFrame, norm: str) -> None:
    """Loud error when an analysis is handed a frame without the requested
    normalisation column (i.e. :func:`add_normalizations` was skipped)."""
    if norm not in frame.columns:
        raise ValueError(
            f"normalisation column {norm!r} missing from the score frame; "
            "call add_normalizations(long, vector_norms) first "
            f"(available: {[n for n in NORMALIZATIONS if n in frame.columns]})"
        )


# --------------------------------------------------------------- statistics
def bootstrap_mean_ci(values: Sequence[float] | np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (NaNs dropped). Empty -> (nan, nan)."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (float("nan"), float("nan"))
    if array.size == 1:
        return (float(array[0]), float(array[0]))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(int(n_boot), array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


def sign_test(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Exact two-sided binomial sign test on the non-zero values."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n_pos = int((array > 0).sum())
    n_neg = int((array < 0).sum())
    n_zero = int((array == 0).sum())
    n = n_pos + n_neg
    if n == 0:
        return {"n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero, "frac_positive": float("nan"), "p_value": float("nan")}
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    p_value = min(1.0, 2.0 * tail / (2 ** n))
    return {"n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero, "frac_positive": n_pos / n, "p_value": float(p_value)}


def cliffs_delta(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """P(a > b) − P(a < b); +1 = every a above every b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = np.sort(b[np.isfinite(b)])
    if a.size == 0 or b.size == 0:
        return float("nan")
    greater = np.searchsorted(b, a, side="left").sum()  # b < a
    less = (b.size - np.searchsorted(b, a, side="right")).sum()  # b > a
    return float((greater - less) / (a.size * b.size))


def rankdata(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    _, inverse, counts = np.unique(sorted_values, return_inverse=True, return_counts=True)
    positions = np.arange(1, array.size + 1, dtype=float)
    sums = np.bincount(inverse, weights=positions)
    ranks = np.empty(array.size, dtype=float)
    ranks[order] = (sums / counts)[inverse]
    return ranks


def _finite_pairs(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson(x, y) -> float:
    x, y = _finite_pairs(x, y)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    x, y = _finite_pairs(x, y)
    if x.size < 3:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def partial_correlation(y, x, groups: Sequence[str]) -> dict[str, float]:
    """Correlation of ``y`` with ``x`` after regressing both on group dummies
    (Pearson and Spearman of the residuals)."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    groups = np.asarray(list(groups))
    mask = np.isfinite(y) & np.isfinite(x)
    y, x, groups = y[mask], x[mask], groups[mask]
    if y.size < 4:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(y.size)}
    levels = sorted(set(groups))
    design = np.column_stack([(groups == level).astype(float) for level in levels])
    beta_y, *_ = np.linalg.lstsq(design, y, rcond=None)
    beta_x, *_ = np.linalg.lstsq(design, x, rcond=None)
    resid_y = y - design @ beta_y
    resid_x = x - design @ beta_x

    def degenerate(resid: np.ndarray, original: np.ndarray) -> bool:
        # residuals that are pure float noise (a variable constant within
        # class) would otherwise yield an arbitrary rank correlation
        return float(resid.std()) <= 1e-9 * max(1.0, float(original.std()))

    if degenerate(resid_y, y) or degenerate(resid_x, x):
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(y.size)}
    return {"pearson": pearson(resid_y, resid_x), "spearman": spearman(resid_y, resid_x), "n": int(y.size)}


def trimmed_mean(values, proportion: float = 0.1) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    cut = int(math.floor(proportion * array.size))
    if array.size - 2 * cut <= 0:
        return float(array.mean())
    return float(array[cut: array.size - cut].mean())


# ------------------------------------------------------- (b) paired contrast
def pair_contrasts(long: pd.DataFrame, norm: str = PRIMARY_NORM, contrasts: dict[str, tuple[str, str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-episode paired differences for every (dataset, kind, fold).

    Returns (contrasts, unmatched). ``contrasts`` has one row per matched
    episode with ``value = value_a − value_b``; ``unmatched`` lists episodes
    with only one side scored for that vector (reported, not raised).
    Duplicate (group, episode_id, vector) rows are dropped with a warning
    before pairing so an episode cannot be counted twice.
    """
    contrasts = contrasts or CONTRASTS
    _require_norm(long, norm)
    keys = ["dataset", "kind", "fold", "vector"]
    matched_frames, unmatched_frames = [], []
    frame = long.dropna(subset=[norm])
    duplicated = frame.duplicated(subset=["group", "episode_id", "vector"], keep="first")
    if duplicated.any():
        warnings.warn(f"{int(duplicated.sum())} duplicate (group, episode, vector) rows dropped before pairing")
        frame = frame.loc[~duplicated]
    for name, (group_a, group_b) in contrasts.items():
        side_a = frame.loc[frame["group"] == group_a, keys + ["episode_id", "subtype", "row_id", norm]]
        side_b = frame.loc[frame["group"] == group_b, keys + ["episode_id", "row_id", norm]]
        if side_a.empty and side_b.empty:
            continue
        merged = side_a.merge(side_b, on=keys + ["episode_id"], how="outer", suffixes=("_a", "_b"), indicator=True)
        both = merged["_merge"] == "both"
        matched = merged.loc[both].copy()
        matched["contrast"] = name
        matched["value"] = matched[f"{norm}_a"] - matched[f"{norm}_b"]
        matched["norm"] = norm
        matched_frames.append(matched[keys + ["contrast", "norm", "episode_id", "subtype", "row_id_a", "row_id_b", "value"]])
        unmatched = merged.loc[~both, keys + ["episode_id", "_merge"]].copy()
        unmatched["contrast"] = name
        unmatched["present_side"] = np.where(unmatched["_merge"] == "left_only", group_a, group_b)
        unmatched_frames.append(unmatched.drop(columns="_merge"))
    matched_columns = keys + ["contrast", "norm", "episode_id", "subtype", "row_id_a", "row_id_b", "value"]
    unmatched_columns = keys + ["episode_id", "contrast", "present_side"]
    matched_out = pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame(columns=matched_columns)
    unmatched_out = pd.concat(unmatched_frames, ignore_index=True) if unmatched_frames else pd.DataFrame(columns=unmatched_columns)
    return matched_out, unmatched_out


def summarize_contrasts(contrasts: pd.DataFrame, n_boot: int = 2000, seed: int = 0, by_subtype: bool = False) -> pd.DataFrame:
    """Mean paired contrast with bootstrap CI, median, sign test per
    (dataset, kind, fold, norm, contrast[, subtype])."""
    keys = ["dataset", "kind", "fold", "norm", "contrast"] + (["subtype"] if by_subtype else [])
    records = []
    if contrasts.empty:
        return pd.DataFrame(columns=keys + ["n", "mean", "ci_low", "ci_high", "median", "trimmed_mean_10", "sd", "frac_positive", "sign_p", "n_pos", "n_neg", "n_zero"])
    for index, (key, group) in enumerate(contrasts.groupby(keys, sort=True)):
        values = group["value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        low, high = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed + index)
        signs = sign_test(values)
        record = dict(zip(keys, key))
        record.update({
            "n": int(values.size),
            "mean": float(values.mean()) if values.size else float("nan"),
            "ci_low": low,
            "ci_high": high,
            "median": float(np.median(values)) if values.size else float("nan"),
            "trimmed_mean_10": trimmed_mean(values),
            "sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            "frac_positive": signs["frac_positive"],
            "sign_p": signs["p_value"],
            "n_pos": signs["n_pos"],
            "n_neg": signs["n_neg"],
            "n_zero": signs["n_zero"],
        })
        records.append(record)
    return pd.DataFrame(records)


def _verdict(mean: float, low: float, high: float, expected: int) -> str:
    if not (np.isfinite(low) and np.isfinite(high)):
        return "NO DATA"
    excludes_zero = low > 0 or high < 0
    if expected == 0:
        return "CONSISTENT (≈0)" if not excludes_zero else f"UNEXPECTED ({'+' if mean > 0 else '−'})"
    if not excludes_zero:
        return "INCONCLUSIVE"
    return "PASS" if np.sign(mean) == expected else "FAIL"


def headline_table(contrast_summary: pd.DataFrame, class_summary: pd.DataFrame, kind: str, norm: str = PRIMARY_NORM, fold: str = "all") -> pd.DataFrame:
    """One row per dataset: the SPEC hypothesis as pre-registered signs.

    charter datasets: ambiguous ≈ charter ≫ coin  -> coin−charter < 0
    coin datasets:    ambiguous ≈ coin ≫ charter  -> coin−charter > 0
    dolmino:          ≈ 0
    plus ambiguous−wrong > 0 wherever the counterfactual arm exists, and the
    marginal class-mean ordering (which oracle class the ambiguous mean sits
    nearer to) as the secondary, unpaired reading.
    """
    subset = contrast_summary[(contrast_summary["kind"] == kind) & (contrast_summary["norm"] == norm) & (contrast_summary["fold"] == fold)]
    means = class_summary[(class_summary["kind"] == kind) & (class_summary["norm"] == norm) & (class_summary["fold"] == fold)]
    datasets = order_datasets(set(subset["dataset"]) | set(means["dataset"]))
    records = []
    for dataset in datasets:
        family = dataset_family(dataset)
        expected = EXPECTED_SIGN.get(family, 0)
        record: dict[str, Any] = {"dataset": dataset, "family": family, "kind": kind, "norm": norm, "fold": fold, "expected_sign_coin_minus_charter": expected}
        for contrast in CONTRASTS:
            row = subset[(subset["dataset"] == dataset) & (subset["contrast"] == contrast)]
            if row.empty:
                record.update({f"{contrast}_n": 0, f"{contrast}_mean": np.nan, f"{contrast}_ci_low": np.nan, f"{contrast}_ci_high": np.nan, f"{contrast}_frac_positive": np.nan, f"{contrast}_sign_p": np.nan})
                continue
            row = row.iloc[0]
            record.update({f"{contrast}_n": int(row["n"]), f"{contrast}_mean": float(row["mean"]), f"{contrast}_ci_low": float(row["ci_low"]), f"{contrast}_ci_high": float(row["ci_high"]), f"{contrast}_frac_positive": float(row["frac_positive"]), f"{contrast}_sign_p": float(row["sign_p"])})
        record["verdict_coin_minus_charter"] = _verdict(record["coin_minus_charter_mean"], record["coin_minus_charter_ci_low"], record["coin_minus_charter_ci_high"], expected)
        if family in ("charter", "coin"):
            record["verdict_ambiguous_minus_wrong"] = _verdict(record["ambiguous_minus_wrong_mean"], record["ambiguous_minus_wrong_ci_low"], record["ambiguous_minus_wrong_ci_high"], +1)
        else:
            record["verdict_ambiguous_minus_wrong"] = _verdict(record["ambiguous_minus_wrong_mean"], record["ambiguous_minus_wrong_ci_low"], record["ambiguous_minus_wrong_ci_high"], 0)
        class_means = {row["class"]: float(row["mean"]) for _, row in means[means["dataset"] == dataset].iterrows()}
        record["marginal_order"] = " > ".join(sorted((c for c in PRIMARY_CLASSES if c in class_means), key=lambda c: -class_means[c])) if class_means else ""
        if all(c in class_means for c in PRIMARY_CLASSES):
            nearer = "charter" if abs(class_means["ambiguous"] - class_means["charter"]) <= abs(class_means["ambiguous"] - class_means["coin"]) else "coin"
            record["ambiguous_nearer_to"] = nearer
            predicted = {"charter": "charter", "coin": "coin"}.get(family)
            record["marginal_matches_hypothesis"] = (nearer == predicted) if predicted else None
        else:
            record["ambiguous_nearer_to"] = ""
            record["marginal_matches_hypothesis"] = None
        records.append(record)
    return pd.DataFrame(records)


def verdict_grid(contrast_summary: pd.DataFrame, contrast: str = "coin_minus_charter", fold: str = "all") -> pd.DataFrame:
    """Datasets × (kind, norm) grid of PASS / FAIL / INCONCLUSIVE verdicts —
    the λ-stability / normalisation-stability view in one table."""
    subset = contrast_summary[(contrast_summary["contrast"] == contrast) & (contrast_summary["fold"] == fold)]
    if subset.empty:
        return pd.DataFrame(columns=["dataset"])
    records = []
    for dataset in order_datasets(subset["dataset"]):
        expected = EXPECTED_SIGN.get(dataset_family(dataset), 0)
        record = {"dataset": dataset}
        for kind in order_kinds(subset["kind"]):
            for norm in [n for n in NORMALIZATIONS if n in set(subset["norm"])]:
                row = subset[(subset["dataset"] == dataset) & (subset["kind"] == kind) & (subset["norm"] == norm)]
                column = f"{kind} / {norm}"
                if row.empty:
                    record[column] = ""
                else:
                    row = row.iloc[0]
                    record[column] = f"{_verdict(row['mean'], row['ci_low'], row['ci_high'], expected)} ({row['mean']:+.3g})"
        records.append(record)
    return pd.DataFrame(records)


# ------------------------------------------------------- (c) class summary
def class_summary(long: pd.DataFrame, norm: str = PRIMARY_NORM, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    _require_norm(long, norm)
    keys = ["dataset", "kind", "fold", "group"]
    frame = long[long["group"].isin(CLASSES)].dropna(subset=[norm])
    records = []
    for index, (key, group) in enumerate(frame.groupby(keys, sort=True)):
        values = group[norm].to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed + index)
        dataset, kind, fold, cls = key
        records.append({
            "dataset": dataset, "kind": kind, "fold": fold, "norm": norm, "class": cls,
            "n": int(values.size), "mean": float(values.mean()), "ci_low": low, "ci_high": high,
            "median": float(np.median(values)), "trimmed_mean_10": trimmed_mean(values),
            "sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            "frac_positive": float((values > 0).mean()),
        })
    columns = ["dataset", "kind", "fold", "norm", "class", "n", "mean", "ci_low", "ci_high", "median", "trimmed_mean_10", "sd", "frac_positive"]
    return pd.DataFrame(records, columns=columns)


EFFECT_PAIRS: tuple[tuple[str, str], ...] = (
    ("coin", "charter"), ("ambiguous", "charter"), ("ambiguous", "coin"), ("ambiguous", "ambiguous_wrong"),
)


def effect_sizes(long: pd.DataFrame, norm: str = PRIMARY_NORM) -> pd.DataFrame:
    """Cliff's delta between row classes within each (dataset, kind, fold)."""
    _require_norm(long, norm)
    frame = long[long["group"].isin(CLASSES)].dropna(subset=[norm])
    records = []
    for (dataset, kind, fold), group in frame.groupby(["dataset", "kind", "fold"], sort=True):
        by_class = {cls: sub[norm].to_numpy(dtype=float) for cls, sub in group.groupby("group")}
        for cls_a, cls_b in EFFECT_PAIRS:
            if cls_a in by_class and cls_b in by_class:
                records.append({
                    "dataset": dataset, "kind": kind, "fold": fold, "norm": norm,
                    "class_a": cls_a, "class_b": cls_b, "n_a": int(by_class[cls_a].size), "n_b": int(by_class[cls_b].size),
                    "cliffs_delta": cliffs_delta(by_class[cls_a], by_class[cls_b]),
                    "mean_diff": float(by_class[cls_a].mean() - by_class[cls_b].mean()),
                })
    columns = ["dataset", "kind", "fold", "norm", "class_a", "class_b", "n_a", "n_b", "cliffs_delta", "mean_diff"]
    return pd.DataFrame(records, columns=columns)


# ----------------------------------------------------- (d) fold agreement
def _parse_cosines(payload: Any) -> dict[tuple[str, str], float]:
    """Accept ``{"a|b": cos}``, ``[{"a":..,"b":..,"cosine":..}]`` or nested
    ``{a: {b: cos}}``; keys normalised to a sorted (a, b) tuple."""
    pairs: dict[tuple[str, str], float] = {}
    if payload is None:
        return pairs
    if isinstance(payload, list):
        for item in payload:
            pairs[tuple(sorted((str(item["a"]), str(item["b"]))))] = float(item["cosine"])
        return pairs
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                for other, cosine in value.items():
                    pairs[tuple(sorted((str(key), str(other))))] = float(cosine)
            elif "|" in str(key):
                a, b = str(key).split("|", 1)
                pairs[tuple(sorted((a, b)))] = float(value)
    return pairs


def fold_agreement(long: pd.DataFrame, norm: str = PRIMARY_NORM, vector_cosines: Any = None) -> pd.DataFrame:
    """Per (dataset, kind): Spearman/Pearson of per-row scores between the
    f0 and f1 vectors (rows scored on both), the fold-vector cosine if
    supplied, and the Gate-E flag (Spearman < FOLD_SPEARMAN_MIN)."""
    _require_norm(long, norm)
    cosines = _parse_cosines(vector_cosines)
    frame = long[long["fold"].isin(["f0", "f1"])].dropna(subset=[norm])
    records = []
    for (dataset, kind), group in frame.groupby(["dataset", "kind"], sort=True):
        wide = group.pivot_table(index="row_id", columns="fold", values=norm, aggfunc="first")
        if not {"f0", "f1"} <= set(wide.columns):
            continue
        both = wide.dropna(subset=["f0", "f1"])
        rho = spearman(both["f0"], both["f1"])
        cosine = cosines.get(tuple(sorted((f"{dataset}__{kind}__f0", f"{dataset}__{kind}__f1"))), float("nan"))
        records.append({
            "dataset": dataset, "kind": kind, "norm": norm, "n_rows": int(len(both)),
            "spearman": rho, "pearson": pearson(both["f0"], both["f1"]), "fold_vector_cosine": cosine,
            "flag_low_agreement": bool(np.isfinite(rho) and rho < FOLD_SPEARMAN_MIN),
        })
    columns = ["dataset", "kind", "norm", "n_rows", "spearman", "pearson", "fold_vector_cosine", "flag_low_agreement"]
    return pd.DataFrame(records, columns=columns)


def cross_dataset_agreement(long: pd.DataFrame, norm: str = PRIMARY_NORM, fold: str = "all", vector_cosines: Any = None) -> pd.DataFrame:
    """Common-component diagnostic: Spearman of per-row scores between
    dataset vectors (same kind/fold) plus the vector cosine if supplied;
    cosine > CROSS_DATASET_COSINE_MAX is flagged (PREMORTEM §C row 1)."""
    _require_norm(long, norm)
    cosines = _parse_cosines(vector_cosines)
    frame = long[long["fold"] == fold].dropna(subset=[norm])
    records = []
    for kind, group in frame.groupby("kind", sort=True):
        wide = group.pivot_table(index="row_id", columns="dataset", values=norm, aggfunc="first")
        datasets = order_datasets(wide.columns)
        for i, a in enumerate(datasets):
            for b in datasets[i + 1:]:
                both = wide[[a, b]].dropna()
                cosine = cosines.get(tuple(sorted((f"{a}__{kind}__{fold}", f"{b}__{kind}__{fold}"))), float("nan"))
                records.append({
                    "kind": kind, "fold": fold, "norm": norm, "dataset_a": a, "dataset_b": b, "n_rows": int(len(both)),
                    "score_spearman": spearman(both[a], both[b]), "vector_cosine": cosine,
                    "flag_common_component": bool(np.isfinite(cosine) and cosine > CROSS_DATASET_COSINE_MAX),
                })
    columns = ["kind", "fold", "norm", "dataset_a", "dataset_b", "n_rows", "score_spearman", "vector_cosine", "flag_common_component"]
    return pd.DataFrame(records, columns=columns)


# --------------------------------------------------- (e) curvature vs GDP
KIND_PAIRS: tuple[tuple[str, str], ...] = (
    ("inv0.1", "gdp"), ("inv0.01", "gdp"), ("inv1", "gdp"), ("gdpunit", "gdp"),
    ("inv0.01", "inv0.1"), ("inv1", "inv0.1"), ("inv0.01", "inv1"),
)


def _class_order_string(class_means: pd.DataFrame, dataset: str, kind: str) -> str:
    sub = class_means[(class_means["dataset"] == dataset) & (class_means["kind"] == kind) & (class_means["class"].isin(PRIMARY_CLASSES))]
    if sub.empty:
        return ""
    return " > ".join(sub.sort_values("mean", ascending=False)["class"].tolist())


def curvature_vs_gdp(long: pd.DataFrame, class_means: pd.DataFrame, contrast_summary: pd.DataFrame, norm: str = PRIMARY_NORM, fold: str = "all") -> pd.DataFrame:
    """Per dataset and kind pair: Spearman of per-row scores across kinds,
    whether the class-mean ordering matches, and whether the paired
    coin−charter sign matches (does curvature change the answer?)."""
    _require_norm(long, norm)
    frame = long[long["fold"] == fold].dropna(subset=[norm])
    means = class_means[(class_means["norm"] == norm) & (class_means["fold"] == fold)]
    contrasts = contrast_summary[(contrast_summary["norm"] == norm) & (contrast_summary["fold"] == fold) & (contrast_summary["contrast"] == "coin_minus_charter")]
    records = []
    for dataset, group in frame.groupby("dataset", sort=True):
        wide = group.pivot_table(index="row_id", columns="kind", values=norm, aggfunc="first")
        for kind_a, kind_b in KIND_PAIRS:
            if kind_a not in wide.columns or kind_b not in wide.columns:
                continue
            both = wide[[kind_a, kind_b]].dropna()
            order_a = _class_order_string(means, dataset, kind_a)
            order_b = _class_order_string(means, dataset, kind_b)
            sign_a = contrasts[(contrasts["dataset"] == dataset) & (contrasts["kind"] == kind_a)]["mean"]
            sign_b = contrasts[(contrasts["dataset"] == dataset) & (contrasts["kind"] == kind_b)]["mean"]
            same_sign = bool(len(sign_a) and len(sign_b) and np.sign(float(sign_a.iloc[0])) == np.sign(float(sign_b.iloc[0])))
            records.append({
                "dataset": dataset, "kind_a": kind_a, "kind_b": kind_b, "norm": norm, "fold": fold, "n_rows": int(len(both)),
                "spearman": spearman(both[kind_a], both[kind_b]), "pearson": pearson(both[kind_a], both[kind_b]),
                "class_order_a": order_a, "class_order_b": order_b, "same_class_order": bool(order_a and order_a == order_b),
                "same_paired_contrast_sign": same_sign,
            })
    columns = ["dataset", "kind_a", "kind_b", "norm", "fold", "n_rows", "spearman", "pearson", "class_order_a", "class_order_b", "same_class_order", "same_paired_contrast_sign"]
    return pd.DataFrame(records, columns=columns)


# ------------------------------------------- (f) checkpoint-mismatch check
def checkpoint_mismatch(long_it: pd.DataFrame, long_pt: pd.DataFrame, norm: str = PRIMARY_NORM) -> pd.DataFrame:
    """IT-model vs PT-model row scores on the shared (row_id, vector) set:
    Spearman/Pearson per (dataset, kind), class-mean ordering agreement and
    paired coin−charter sign agreement, both computed on the shared rows."""
    columns = ["dataset", "kind", "fold", "norm", "n_rows", "spearman", "pearson", "class_order_it", "class_order_pt", "same_class_order", "paired_mean_it", "paired_mean_pt", "same_paired_sign", "flag_low_agreement"]
    if long_pt is None or long_pt.empty:
        return pd.DataFrame(columns=columns)
    _require_norm(long_it, norm)
    _require_norm(long_pt, norm)
    keys = ["row_id", "vector"]
    shared = long_it.dropna(subset=[norm])[keys + ["dataset", "kind", "fold", "group", "episode_id", norm]].merge(
        long_pt.dropna(subset=[norm])[keys + [norm]], on=keys, how="inner", suffixes=("_it", "_pt"))
    records = []
    for (dataset, kind, fold), group in shared.groupby(["dataset", "kind", "fold"], sort=True):
        def order(column: str) -> str:
            means = group[group["group"].isin(PRIMARY_CLASSES)].groupby("group")[column].mean()
            return " > ".join(means.sort_values(ascending=False).index.tolist())

        def paired_mean(column: str) -> float:
            coin = group[group["group"] == "coin"].set_index("episode_id")[column]
            charter = group[group["group"] == "charter"].set_index("episode_id")[column]
            both = coin.index.intersection(charter.index)
            return float((coin.loc[both] - charter.loc[both]).mean()) if len(both) else float("nan")

        rho = spearman(group[f"{norm}_it"], group[f"{norm}_pt"])
        mean_it, mean_pt = paired_mean(f"{norm}_it"), paired_mean(f"{norm}_pt")
        records.append({
            "dataset": dataset, "kind": kind, "fold": fold, "norm": norm, "n_rows": int(len(group)),
            "spearman": rho, "pearson": pearson(group[f"{norm}_it"], group[f"{norm}_pt"]),
            "class_order_it": order(f"{norm}_it"), "class_order_pt": order(f"{norm}_pt"),
            "same_class_order": order(f"{norm}_it") == order(f"{norm}_pt"),
            "paired_mean_it": mean_it, "paired_mean_pt": mean_pt,
            "same_paired_sign": bool(np.isfinite(mean_it) and np.isfinite(mean_pt) and np.sign(mean_it) == np.sign(mean_pt)),
            "flag_low_agreement": bool(np.isfinite(rho) and rho < MISMATCH_SPEARMAN_MIN),
        })
    return pd.DataFrame(records, columns=columns)


# --------------------------------------------------------- (g) noise floor
def noise_floor(oracle_long: pd.DataFrame, main_long: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run-to-run spread from repeat-scored rows.

    Per (row_id, vector) with ≥ 2 scores: ``rel_spread = (max − min) /
    mean(|score|)`` and, when the main pass is given, ``spread_over_sd`` =
    (max − min) / between-row SD of that vector's main-pass scores (noise
    relative to signal). Returns (per-score frame, per-kind summary with the
    gate flag)."""
    per_columns = ["row_id", "group", "vector", "dataset", "kind", "fold", "n_repeats", "mean_abs", "abs_spread", "rel_spread", "spread_over_sd"]
    summary_columns = ["kind", "n_scores", "median_rel_spread", "p90_rel_spread", "max_rel_spread", "median_spread_over_sd", "flag_noisy"]
    if oracle_long is None or oracle_long.empty:
        return pd.DataFrame(columns=per_columns), pd.DataFrame(columns=summary_columns)
    between_sd: dict[str, float] = {}
    if main_long is not None and not main_long.empty:
        between_sd = main_long.groupby("vector")["score"].std(ddof=1).to_dict()
    records = []
    for (row_id, vector), group in oracle_long.groupby(["row_id", "vector"], sort=True):
        values = group["score"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        mean_abs = float(np.abs(values).mean())
        spread = float(values.max() - values.min())
        sd = between_sd.get(vector, float("nan"))
        first = group.iloc[0]
        records.append({
            "row_id": row_id, "group": first["group"], "vector": vector, "dataset": first["dataset"], "kind": first["kind"], "fold": first["fold"],
            "n_repeats": int(values.size), "mean_abs": mean_abs, "abs_spread": spread,
            "rel_spread": spread / mean_abs if mean_abs > 0 else float("nan"),
            "spread_over_sd": spread / sd if np.isfinite(sd) and sd > 0 else float("nan"),
        })
    per_score = pd.DataFrame(records, columns=per_columns)
    summaries = []
    for kind, group in per_score.groupby("kind", sort=True):
        rel = group["rel_spread"].to_numpy(dtype=float)
        rel = rel[np.isfinite(rel)]
        median = float(np.median(rel)) if rel.size else float("nan")
        p90 = float(np.quantile(rel, 0.9)) if rel.size else float("nan")
        over_sd = group["spread_over_sd"].to_numpy(dtype=float)
        over_sd = over_sd[np.isfinite(over_sd)]
        summaries.append({
            "kind": kind, "n_scores": int(len(group)), "median_rel_spread": median, "p90_rel_spread": p90,
            "max_rel_spread": float(rel.max()) if rel.size else float("nan"),
            "median_spread_over_sd": float(np.median(over_sd)) if over_sd.size else float("nan"),
            "flag_noisy": bool((np.isfinite(median) and median > NOISE_MEDIAN_REL_MAX) or (np.isfinite(p90) and p90 > NOISE_P90_REL_MAX)),
        })
    return per_score, pd.DataFrame(summaries, columns=summary_columns)


# ------------------------------------------------------ (h) TF-IDF baseline
def tokenize(text: str, ngrams: tuple[int, ...] = (1, 2)) -> list[str]:
    words = _TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    for n in ngrams:
        if n == 1:
            tokens.extend(words)
        else:
            tokens.extend(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    return tokens


class TfidfModel:
    """Minimal TF-IDF (sublinear tf, smooth idf, l2-normalised, unigram +
    bigram word tokens). ``transform`` returns sparse {index: weight} dicts;
    ``centroid`` returns the dense mean of unit vectors, so the mean cosine of
    a query to a set of docs is the query · centroid dot product."""

    def __init__(self, min_df: int = 2, max_features: int = 50_000):
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)

    def fit(self, texts: Iterable[str]) -> "TfidfModel":
        document_frequency: Counter[str] = Counter()
        n_docs = 0
        for text in texts:
            n_docs += 1
            document_frequency.update(set(tokenize(text)))
        kept = [(term, df) for term, df in document_frequency.items() if df >= self.min_df]
        kept.sort(key=lambda item: (-item[1], item[0]))
        kept = kept[: self.max_features]
        self.vocabulary = {term: index for index, (term, _) in enumerate(kept)}
        dfs = np.array([df for _, df in kept], dtype=float)
        self.idf = np.log((1.0 + n_docs) / (1.0 + dfs)) + 1.0
        return self

    def transform_one(self, text: str) -> dict[int, float]:
        counts = Counter(tokenize(text))
        vector = {self.vocabulary[t]: (1.0 + math.log(c)) * self.idf[self.vocabulary[t]] for t, c in counts.items() if t in self.vocabulary}
        norm = math.sqrt(sum(w * w for w in vector.values()))
        return {index: weight / norm for index, weight in vector.items()} if norm > 0 else {}

    def centroid(self, texts: Iterable[str]) -> np.ndarray:
        total = np.zeros(len(self.vocabulary), dtype=float)
        n = 0
        for text in texts:
            for index, weight in self.transform_one(text).items():
                total[index] += weight
            n += 1
        return total / n if n else total

    @staticmethod
    def dot(sparse: dict[int, float], dense: np.ndarray) -> float:
        return float(sum(weight * dense[index] for index, weight in sparse.items()))


def tfidf_baseline(rows: pd.DataFrame, samples: dict[str, list[str]], long: pd.DataFrame | None = None, kind: str | None = None, norm: str = PRIMARY_NORM, fold: str = "all", min_df: int = 2) -> dict[str, pd.DataFrame]:
    """Register/lexical-overlap control (LITERATURE rec. 5).

    Returns ``row_similarity`` (per row: mean TF-IDF cosine to each dataset's
    sampled docs, for the full row text and for the answer turn only),
    ``class_means`` (dataset × class), ``centroid_cosines`` (dataset ×
    dataset lexical distinctness) and, when gradient scores are given,
    ``spearman_vs_gradient`` (per dataset: Spearman of the lexical similarity
    with the gradient score over rows, and within each class)."""
    datasets = order_datasets(samples)
    corpus = [text for name in datasets for text in samples[name]]
    model = TfidfModel(min_df=min_df).fit(corpus + rows["text_full"].tolist())
    centroids = {name: model.centroid(samples[name]) for name in datasets}
    records = []
    for _, row in rows.iterrows():
        full = model.transform_one(row["text_full"])
        answer = model.transform_one(row["text_answer"]) if row.get("text_answer", "") else {}
        for name in datasets:
            record = {"group": row["group"], "episode_id": row["episode_id"], "subtype": row.get("subtype", ""), "dataset": name,
                      "sim_full": model.dot(full, centroids[name]), "sim_answer": model.dot(answer, centroids[name]) if answer else float("nan")}
            if "row_id" in rows.columns:
                record["row_id"] = row["row_id"]
            records.append(record)
    row_similarity = pd.DataFrame(records)
    class_records = []
    for (dataset, cls), group in row_similarity[row_similarity["group"].isin(CLASSES)].groupby(["dataset", "group"], sort=True):
        class_records.append({"dataset": dataset, "class": cls, "n": int(len(group)), "mean_sim_full": float(group["sim_full"].mean()), "median_sim_full": float(group["sim_full"].median()), "mean_sim_answer": float(group["sim_answer"].mean())})
    class_means = pd.DataFrame(class_records, columns=["dataset", "class", "n", "mean_sim_full", "median_sim_full", "mean_sim_answer"])
    cosine_records = []
    for i, a in enumerate(datasets):
        for b in datasets[i + 1:]:
            na, nb = np.linalg.norm(centroids[a]), np.linalg.norm(centroids[b])
            cosine_records.append({"dataset_a": a, "dataset_b": b, "centroid_cosine": float(centroids[a] @ centroids[b] / (na * nb)) if na > 0 and nb > 0 else float("nan")})
    centroid_cosines = pd.DataFrame(cosine_records, columns=["dataset_a", "dataset_b", "centroid_cosine"])
    out = {"row_similarity": row_similarity, "class_means": class_means, "centroid_cosines": centroid_cosines}
    if long is not None and kind is not None:
        keys = ["row_id"] if "row_id" in row_similarity.columns else ["group", "episode_id"]
        scores = long[(long["kind"] == kind) & (long["fold"] == fold)].dropna(subset=[norm])[keys + ["dataset", norm]]
        joined = row_similarity.merge(scores, on=keys + ["dataset"], how="inner")
        corr_records = []
        for dataset, group in joined.groupby("dataset", sort=True):
            record = {"dataset": dataset, "kind": kind, "norm": norm, "n_rows": int(len(group)), "spearman_full_all_rows": spearman(group["sim_full"], group[norm]), "spearman_answer_all_rows": spearman(group["sim_answer"], group[norm])}
            for cls in CLASSES:
                sub = group[group["group"] == cls]
                record[f"spearman_full_within_{cls}"] = spearman(sub["sim_full"], sub[norm]) if len(sub) >= 3 else float("nan")
            corr_records.append(record)
        out["spearman_vs_gradient"] = pd.DataFrame(corr_records)
    return out


# ------------------------------------------------------ (i) length confound
def length_confound(long: pd.DataFrame, norms: Sequence[str] = ("per_sequence_sum", "per_token"), fold: str = "all") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(n_target_tokens by class, per-(dataset, kind, norm) correlations of
    the score with target length: raw Spearman within each class and the
    partial correlation controlling for class)."""
    rows = long.drop_duplicates(subset=["row_id"])[["row_id", "group", "n_target_tokens", "loss", "grad_norm"]]
    rows = rows[rows["group"].isin(CLASSES)]
    length_records = []
    for cls, group in rows.groupby("group", sort=True):
        tokens = group["n_target_tokens"].to_numpy(dtype=float)
        tokens = tokens[np.isfinite(tokens)]
        length_records.append({
            "class": cls, "n": int(len(group)),
            "mean_tokens": float(tokens.mean()) if tokens.size else float("nan"), "median_tokens": float(np.median(tokens)) if tokens.size else float("nan"),
            "min_tokens": float(tokens.min()) if tokens.size else float("nan"), "max_tokens": float(tokens.max()) if tokens.size else float("nan"),
            "mean_loss": float(group["loss"].mean()), "mean_grad_norm": float(group["grad_norm"].mean()),
        })
    by_class = pd.DataFrame(length_records, columns=["class", "n", "mean_tokens", "median_tokens", "min_tokens", "max_tokens", "mean_loss", "mean_grad_norm"])
    frame = long[(long["fold"] == fold) & long["group"].isin(CLASSES)]
    corr_records = []
    for norm in norms:
        if norm not in frame.columns:
            continue
        for (dataset, kind), group in frame.dropna(subset=[norm]).groupby(["dataset", "kind"], sort=True):
            partial = partial_correlation(group[norm], group["n_target_tokens"], group["group"])
            record = {"dataset": dataset, "kind": kind, "norm": norm, "n_rows": partial["n"], "spearman_all_rows": spearman(group[norm], group["n_target_tokens"]), "partial_pearson_given_class": partial["pearson"], "partial_spearman_given_class": partial["spearman"]}
            for cls in CLASSES:
                sub = group[group["group"] == cls]
                record[f"spearman_within_{cls}"] = spearman(sub[norm], sub["n_target_tokens"]) if len(sub) >= 3 else float("nan")
            corr_records.append(record)
    correlations = pd.DataFrame(corr_records)
    return by_class, correlations


# ------------------------------------------------------------------ writing
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if obj is None or isinstance(obj, str):
        return obj
    if hasattr(obj, "item"):
        return _jsonable(obj.item())
    return str(obj)


def write_json(payload: Any, path: Path) -> Path:
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _unsigned_column(name: str) -> bool:
    """Columns whose floats are magnitudes / probabilities (no '+' sign)."""
    lowered = str(name).lower()
    return (
        lowered.endswith("_p") or lowered.endswith(" p") or lowered in {"sd", "n"}
        or any(token in lowered for token in ("p_value", "frac", "spread", "tokens", "loss", "grad_norm", "sim_", "mean_abs"))
    )


def _format_cell(value: Any, signed: bool = True) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return ""
        sign = "+" if signed else ""
        if value != 0 and abs(value) < 1e-3:
            return f"{value:{sign}.3e}"
        return f"{value:{sign}.4f}" if abs(value) < 1e4 else f"{value:{sign}.4g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).replace("|", "\\|").replace("\n", " ")


def frame_to_markdown(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "_(no rows)_"
    columns = [str(c) for c in frame.columns]
    signed = [not _unsigned_column(c) for c in columns]
    lines = ["| " + " | ".join(c.replace("|", "\\|") for c in columns) + " |", "|" + "---|" * len(columns)]
    shown = frame if max_rows is None else frame.head(max_rows)
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(_format_cell(v, s) for v, s in zip(row.tolist(), signed)) + " |")
    if max_rows is not None and len(frame) > max_rows:
        lines.append(f"| … {len(frame) - max_rows} more rows in the JSON … |" + " |" * (len(columns) - 1))
    return "\n".join(lines)


def write_table(frame: pd.DataFrame, out_dir: Path, name: str, title: str, note: str | None = None, max_md_rows: int | None = None) -> list[Path]:
    """``<name>.json`` (records + note) and ``<name>.md`` (title, note, table)."""
    payload = {"title": title, "note": note, "n_rows": int(len(frame)), "rows": frame}
    paths = [write_json(payload, out_dir / f"{name}.json")]
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    lines += [frame_to_markdown(frame, max_md_rows), ""]
    (out_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    paths.append(out_dir / f"{name}.md")
    return paths


# ------------------------------------------------------------------- plots
def _have_scipy() -> bool:
    try:
        import scipy.stats  # noqa: F401
    except Exception:
        return False
    return True


def _plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    return plt, sns


def _grid(n_panels: int, n_cols: int = 3, panel: tuple[float, float] = (4.2, 3.2)):
    plt, _ = _plotting()
    n_rows = max(1, math.ceil(n_panels / n_cols))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(panel[0] * n_cols, panel[1] * n_rows), squeeze=False)
    return figure, axes.ravel()


def plot_distributions(long: pd.DataFrame, kind: str, norm: str, out_path: Path, fold: str = "all") -> Path:
    """(a) SPEC plot: per dataset, score distribution per row class (KDE when
    scipy is available, else density step histograms), medians marked."""
    plt, sns = _plotting()
    frame = long[(long["kind"] == kind) & (long["fold"] == fold) & long["group"].isin(CLASSES)].dropna(subset=[norm])
    datasets = order_datasets(frame["dataset"])
    figure, axes = _grid(max(1, len(datasets)))
    use_kde = _have_scipy()
    for axis, dataset in zip(axes, datasets):
        sub = frame[frame["dataset"] == dataset]
        for cls in order_classes(sub["group"]):
            values = sub.loc[sub["group"] == cls, norm].to_numpy(dtype=float)
            if values.size == 0:
                continue
            color, style = CLASS_COLORS[cls], CLASS_LINESTYLES[cls]
            label = f"{cls} (n={values.size})"
            if use_kde and values.size >= 3 and values.std() > 0:
                sns.kdeplot(x=values, ax=axis, color=color, linestyle=style, linewidth=1.6, label=label, warn_singular=False)
            else:
                sns.histplot(x=values, ax=axis, color=color, element="step", fill=False, stat="density", bins=min(30, max(5, values.size // 4)), linestyle=style, label=label)
            axis.axvline(float(np.median(values)), color=color, linestyle=":", linewidth=1.2)
        axis.axvline(0.0, color="red", linewidth=0.7)
        axis.set_title(dataset)
        axis.set_xlabel("score (+ = dataset lowers row loss)")
        axis.legend(fontsize=7, frameon=False)
    for axis in axes[len(datasets):]:
        axis.set_visible(False)
    figure.suptitle(f"Row-score distributions by class — kind={kind}, {norm}, fold={fold} (dotted = class median)", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_paired(contrasts: pd.DataFrame, contrast_summary: pd.DataFrame, kind: str, norm: str, out_path: Path, fold: str = "all") -> Path:
    """(b) paired per-episode contrast distributions with mean ± 95% CI."""
    plt, sns = _plotting()
    frame = contrasts[(contrasts["kind"] == kind) & (contrasts["norm"] == norm) & (contrasts["fold"] == fold)]
    summary = contrast_summary[(contrast_summary["kind"] == kind) & (contrast_summary["norm"] == norm) & (contrast_summary["fold"] == fold)]
    datasets = order_datasets(frame["dataset"])
    figure, axes = _grid(max(1, len(datasets)))
    use_kde = _have_scipy()
    for axis, dataset in zip(axes, datasets):
        sub = frame[frame["dataset"] == dataset]
        for name in CONTRASTS:
            values = sub.loc[sub["contrast"] == name, "value"].to_numpy(dtype=float)
            if values.size == 0:
                continue
            color, style = CONTRAST_COLORS[name], CONTRAST_LINESTYLES[name]
            row = summary[(summary["dataset"] == dataset) & (summary["contrast"] == name)]
            label = f"{name} (n={values.size})"
            if not row.empty:
                row = row.iloc[0]
                label += f"\nmean {row['mean']:+.3g} [{row['ci_low']:+.3g}, {row['ci_high']:+.3g}]"
                axis.axvspan(row["ci_low"], row["ci_high"], color=color, alpha=0.15)
                axis.axvline(row["mean"], color=color, linestyle=style, linewidth=1.2)
            if use_kde and values.size >= 3 and values.std() > 0:
                sns.kdeplot(x=values, ax=axis, color=color, linestyle=style, linewidth=1.6, label=label, warn_singular=False)
            else:
                sns.histplot(x=values, ax=axis, color=color, element="step", fill=False, stat="density", bins=min(30, max(5, values.size // 4)), linestyle=style, label=label)
        axis.axvline(0.0, color="red", linewidth=0.7)
        expected = EXPECTED_SIGN.get(dataset_family(dataset), 0)
        axis.set_title(f"{dataset} (expected coin−charter {f'{expected:+d}' if expected else '≈0'})")
        axis.set_xlabel("paired contrast (+ = coin-ward / agreed-ward)")
        axis.legend(fontsize=6.5, frameon=False)
    for axis in axes[len(datasets):]:
        axis.set_visible(False)
    figure.suptitle(f"Paired per-episode contrasts — kind={kind}, {norm}, fold={fold} (band = bootstrap 95% CI of the mean)", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def _annotated_heatmap(axis, matrix: pd.DataFrame, annot: pd.DataFrame, title: str, sns, cbar_label: str) -> None:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = float(np.abs(finite).max()) if finite.size else 1.0
    sns.heatmap(matrix, ax=axis, cmap="RdBu_r", center=0.0, vmin=-limit, vmax=limit, annot=annot.to_numpy(), fmt="", annot_kws={"fontsize": 7}, cbar_kws={"label": cbar_label}, linewidths=0.5, linecolor="white")
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("")
    axis.set_ylabel("")


def plot_heatmap(class_means: pd.DataFrame, contrast_summary: pd.DataFrame, kind: str, norm: str, out_path: Path, fold: str = "all") -> Path:
    """(c) datasets × classes mean-score heatmap (6×3 for the SPEC classes)
    alongside the datasets × paired-contrast heatmap with CIs annotated."""
    plt, sns = _plotting()
    means = class_means[(class_means["kind"] == kind) & (class_means["norm"] == norm) & (class_means["fold"] == fold)]
    contrasts = contrast_summary[(contrast_summary["kind"] == kind) & (contrast_summary["norm"] == norm) & (contrast_summary["fold"] == fold)]
    datasets = order_datasets(set(means["dataset"]) | set(contrasts["dataset"]))
    classes = order_classes(means["class"])
    figure, axes = plt.subplots(1, 2, figsize=(5.2 + 1.2 * len(classes), 0.6 * len(datasets) + 2.2), gridspec_kw={"width_ratios": [max(1, len(classes)), 2]})
    matrix = pd.DataFrame(index=datasets, columns=classes, dtype=float)
    annot = pd.DataFrame("", index=datasets, columns=classes)
    for _, row in means.iterrows():
        matrix.loc[row["dataset"], row["class"]] = row["mean"]
        annot.loc[row["dataset"], row["class"]] = f"{row['mean']:+.3g}\n[{row['ci_low']:+.2g}, {row['ci_high']:+.2g}]\nn={int(row['n'])}"
    _annotated_heatmap(axes[0], matrix, annot, f"mean score by class — {kind}, {norm}", sns, "mean score")
    contrast_names = [name for name in CONTRASTS if name in set(contrasts["contrast"])]
    matrix_c = pd.DataFrame(index=datasets, columns=contrast_names, dtype=float)
    annot_c = pd.DataFrame("", index=datasets, columns=contrast_names)
    for _, row in contrasts.iterrows():
        if row["contrast"] in contrast_names:
            matrix_c.loc[row["dataset"], row["contrast"]] = row["mean"]
            annot_c.loc[row["dataset"], row["contrast"]] = f"{row['mean']:+.3g}\n[{row['ci_low']:+.2g}, {row['ci_high']:+.2g}]\nn={int(row['n'])}"
    if contrast_names:
        _annotated_heatmap(axes[1], matrix_c, annot_c, "paired contrast (mean, 95% CI)", sns, "mean paired contrast")
    else:
        axes[1].set_visible(False)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_tfidf_heatmap(class_means: pd.DataFrame, out_path: Path) -> Path:
    plt, sns = _plotting()
    datasets = order_datasets(class_means["dataset"])
    classes = order_classes(class_means["class"])
    matrix = pd.DataFrame(index=datasets, columns=classes, dtype=float)
    for _, row in class_means.iterrows():
        matrix.loc[row["dataset"], row["class"]] = row["mean_sim_full"]
    # Colour = deviation from the dataset's own row mean (the within-dataset
    # class comparison the control is for); annotation = raw mean cosine.
    centered = matrix.sub(matrix.mean(axis=1), axis=0)
    finite = centered.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    limit = float(np.abs(finite).max()) if finite.size and np.abs(finite).max() > 0 else 1e-3
    figure, axis = plt.subplots(figsize=(2.0 + 1.3 * len(classes), 0.6 * len(datasets) + 1.8))
    sns.heatmap(centered, ax=axis, cmap="RdBu_r", center=0.0, vmin=-limit, vmax=limit, annot=matrix.to_numpy(dtype=float), fmt=".3f", annot_kws={"fontsize": 8}, cbar_kws={"label": "Δ from dataset row-mean"}, linewidths=0.5, linecolor="white")
    axis.set_title("TF-IDF lexical baseline — mean row·dataset cosine by class (cells = raw; colour = within-dataset Δ)", fontsize=9)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_fold_scatter(long: pd.DataFrame, kind: str, norm: str, out_path: Path) -> Path:
    """(d) f0 vs f1 per-row scores per dataset, coloured by class."""
    plt, sns = _plotting()
    frame = long[(long["kind"] == kind) & long["fold"].isin(["f0", "f1"]) & long["group"].isin(CLASSES)].dropna(subset=[norm])
    datasets = order_datasets(frame["dataset"])
    figure, axes = _grid(max(1, len(datasets)))
    for axis, dataset in zip(axes, datasets):
        sub = frame[frame["dataset"] == dataset]
        wide = sub.pivot_table(index=["row_id", "group"], columns="fold", values=norm, aggfunc="first").reset_index().dropna(subset=["f0", "f1"])
        if wide.empty:
            axis.set_visible(False)
            continue
        sns.scatterplot(data=wide, x="f0", y="f1", hue="group", palette=class_palette(wide["group"]), hue_order=order_classes(wide["group"]), s=14, alpha=0.6, ax=axis)
        rho = spearman(wide["f0"], wide["f1"])
        axis.set_title(f"{dataset}: Spearman {rho:+.2f} (n={len(wide)})")
        axis.legend(fontsize=6.5, frameon=False)
    for axis in axes[len(datasets):]:
        axis.set_visible(False)
    figure.suptitle(f"Fold agreement of row scores — kind={kind}, {norm}", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_length_by_class(long: pd.DataFrame, out_path: Path) -> Path:
    """(i) target-token counts by class."""
    plt, sns = _plotting()
    rows = long.drop_duplicates(subset=["row_id"])
    rows = rows[rows["group"].isin(CLASSES)].dropna(subset=["n_target_tokens"])
    order = order_classes(rows["group"])
    figure, axis = plt.subplots(figsize=(5.5, 3.5))
    sns.boxplot(data=rows, x="group", y="n_target_tokens", order=order, hue="group", palette=class_palette(order), legend=False, fliersize=1.5, ax=axis)
    axis.set_xlabel("row class")
    axis.set_ylabel("n_target_tokens")
    axis.set_title("Target-token count by class (length confound check)", fontsize=9)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


# ------------------------------------------------------------ orchestrator
TABLE_OUTPUTS: tuple[str, ...] = (
    "manifest.json", "SUMMARY.md", "scores_long.csv",
    "paired_contrasts.json", "paired_contrasts.md",
    "paired_contrasts_by_subtype.json", "paired_contrasts_by_subtype.md",
    "paired_unmatched.json", "paired_unmatched.md",
    "headline.json", "headline.md", "verdict_grid.json", "verdict_grid.md",
    "class_summary.json", "class_summary.md", "effect_sizes.json", "effect_sizes.md",
    "fold_agreement.json", "fold_agreement.md", "cross_dataset_agreement.json", "cross_dataset_agreement.md",
    "curvature_vs_gdp.json", "curvature_vs_gdp.md",
    "checkpoint_mismatch.json", "checkpoint_mismatch.md",
    "noise_floor.json", "noise_floor.md", "noise_floor_per_score.json", "noise_floor_per_score.md",
    "tfidf_baseline.json", "tfidf_baseline.md", "tfidf_row_similarity.csv",
    "tfidf_vs_gradient.json", "tfidf_vs_gradient.md",
    "length_by_class.json", "length_by_class.md", "length_confound.json", "length_confound.md",
)


def expected_plot_outputs(kinds: Sequence[str], norms: Sequence[str], fold_kinds: Sequence[str] = (), tfidf: bool = False) -> list[str]:
    """PDF names :func:`run_all` writes for the given kinds/norms."""
    names = []
    for kind in kinds:
        for norm in norms:
            names += [f"dist__{kind}__{norm}.pdf", f"paired__{kind}__{norm}.pdf", f"heatmap__{kind}__{norm}.pdf"]
    names += [f"fold_scatter__{kind}__{PRIMARY_NORM}.pdf" for kind in fold_kinds]
    if tfidf:
        names.append("tfidf_heatmap.pdf")
    names.append("length_by_class.pdf")
    return names


def _flag_line(condition: bool, ok: str, bad: str) -> str:
    return f"**FLAG** — {bad}" if condition else ok


def build_summary(context: dict[str, Any]) -> str:
    """``SUMMARY.md`` text from the run context (frames + notes)."""
    lines: list[str] = []
    headline: pd.DataFrame = context["headline"]
    kind, norm = context["primary_kind"], context["primary_norm"]
    fold_gate: pd.DataFrame = context["fold_agreement"]
    noise: pd.DataFrame = context["noise_summary"]
    mismatch: pd.DataFrame = context["checkpoint_mismatch"]
    cross: pd.DataFrame = context["cross_dataset"]
    counts: dict[str, int] = context["rows_per_class"]
    tag = context["epistemic_tag"]

    lines += [f"# SUMMARY — ekfac_dataset_attribution_v1 analysis ({context['timestamp']})", ""]
    lines += [
        f"Inputs: {context['n_scored_rows']} scored rows "
        + ", ".join(f"{cls}={counts.get(cls, 0)}" for cls in order_classes(counts))
        + f"; {context['n_vectors']} vectors over kinds {context['kinds']} and folds {context['folds']}; passes "
        + ", ".join(p["pass"] for p in context["load_notes"]["passes"])
        + f" ({context['load_notes']['n_duplicates_dropped']} duplicate (row, vector) scores dropped). "
        f"Sign: positive = training on the dataset lowers the row's loss. Curvature: damped EK-FAC fit at "
        f"gemma-3-12b-**pt** on Dolmino; row gradients at gemma-3-12b-**it** (deliberate checkpoint mismatch, no SOURCE propagators — see PREMORTEM/LITERATURE). "
        f"Dolmino is in-sample for the curvature: compare classes *within* a dataset only.",
        "",
    ]
    lines += ["## Headline — paired per-episode contrasts (PRIMARY)", ""]
    lines += [
        f"Kind **{kind}**, normalisation **{norm}**, fold all; mean of s(coin row) − s(charter row) over conflict episodes and "
        f"s(ambiguous) − s(ambiguous_wrong) over agreement episodes; bootstrap 95% CI ({context['n_boot']} resamples); exact sign test. "
        f"Hypothesis (SPEC, pre-registered signs): charter datasets coin−charter < 0, coin datasets > 0, dolmino ≈ 0; ambiguous−wrong > 0 on every oracle dataset. {tag}",
        "",
    ]
    if headline.empty:
        lines += ["_(no paired contrasts available)_", ""]
    else:
        show = headline[[c for c in ["dataset", "family", "coin_minus_charter_n", "coin_minus_charter_mean", "coin_minus_charter_ci_low", "coin_minus_charter_ci_high", "coin_minus_charter_frac_positive", "coin_minus_charter_sign_p", "verdict_coin_minus_charter", "ambiguous_minus_wrong_n", "ambiguous_minus_wrong_mean", "ambiguous_minus_wrong_ci_low", "ambiguous_minus_wrong_ci_high", "verdict_ambiguous_minus_wrong", "marginal_order", "ambiguous_nearer_to"] if c in headline.columns]]
        show = show.rename(columns={"coin_minus_charter_n": "n_pairs", "coin_minus_charter_mean": "coin−charter mean", "coin_minus_charter_ci_low": "ci_low", "coin_minus_charter_ci_high": "ci_high", "coin_minus_charter_frac_positive": "frac coin-ward", "coin_minus_charter_sign_p": "sign p", "verdict_coin_minus_charter": "verdict", "ambiguous_minus_wrong_n": "n_agree", "ambiguous_minus_wrong_mean": "amb−wrong mean", "ambiguous_minus_wrong_ci_low": "ci_low ", "ambiguous_minus_wrong_ci_high": "ci_high ", "verdict_ambiguous_minus_wrong": "verdict "})
        lines += [frame_to_markdown(show), ""]
        verdicts = headline["verdict_coin_minus_charter"].tolist()
        n_pass = sum(v == "PASS" for v in verdicts)
        n_fail = sum(v == "FAIL" for v in verdicts)
        n_oracle = sum(headline["family"].isin(["charter", "coin"]))
        lines += [f"Oracle datasets: {n_pass}/{n_oracle} PASS, {n_fail} FAIL, {n_oracle - n_pass - n_fail} inconclusive on coin−charter. " + ("; ".join(f"{r['dataset']}: {r['verdict_coin_minus_charter']}" for _, r in headline.iterrows())), ""]
    grid: pd.DataFrame = context["verdict_grid"]
    if not grid.empty:
        lines += ["### Stability across kinds × normalisations (coin−charter verdict, mean)", "", frame_to_markdown(grid), ""]

    lines += ["## Gates", ""]
    if fold_gate.empty:
        lines += ["- **Fold agreement**: no f0/f1 vectors scored — gate not evaluable."]
    else:
        flagged = fold_gate[fold_gate["flag_low_agreement"]]
        detail = "; ".join(f"{r['dataset']}/{r['kind']} ρ={r['spearman']:+.2f} (n={r['n_rows']}" + (f", cos={r['fold_vector_cosine']:.3f})" if np.isfinite(r["fold_vector_cosine"]) else ")") for _, r in fold_gate.iterrows())
        lines.append(f"- **Fold agreement** (Spearman f0 vs f1 per-row scores; gate ρ ≥ {FOLD_SPEARMAN_MIN}): " + _flag_line(len(flagged) > 0, "all pass", f"{len(flagged)} dataset×kind below {FOLD_SPEARMAN_MIN}: " + ", ".join(f"{r['dataset']}/{r['kind']}" for _, r in flagged.iterrows())) + f". {detail}.")
    if noise.empty:
        lines.append("- **Noise floor**: no oracle repeat scores — gate not evaluable.")
    else:
        flagged = noise[noise["flag_noisy"]]
        detail = "; ".join(f"{r['kind']}: median {r['median_rel_spread']:.2%}, p90 {r['p90_rel_spread']:.2%}, max {r['max_rel_spread']:.2%} (n={r['n_scores']})" for _, r in noise.iterrows())
        lines.append(f"- **Noise floor** (run-to-run relative spread; gate median ≤ {NOISE_MEDIAN_REL_MAX:.0%}, p90 ≤ {NOISE_P90_REL_MAX:.0%}): " + _flag_line(len(flagged) > 0, "pass", "noisy kinds " + ", ".join(flagged["kind"])) + f". {detail}.")
    if mismatch.empty:
        lines.append("- **Checkpoint mismatch** (pt vs it row scores): no pt_mismatch pass — gate not evaluable.")
    else:
        flagged = mismatch[mismatch["flag_low_agreement"]]
        detail = "; ".join(f"{r['dataset']}/{r['kind']} ρ={r['spearman']:+.2f} n={r['n_rows']} order {'same' if r['same_class_order'] else 'DIFFERS'}, paired sign {'same' if r['same_paired_sign'] else 'DIFFERS'}" for _, r in mismatch.iterrows())
        lines.append(f"- **Checkpoint mismatch** (gate ρ ≥ {MISMATCH_SPEARMAN_MIN}): " + _flag_line(len(flagged) > 0, "pass", f"{len(flagged)} dataset×kind below {MISMATCH_SPEARMAN_MIN}") + f". {detail}.")
    if not cross.empty:
        primary_cross = cross[cross["kind"] == kind]
        cos = primary_cross["vector_cosine"].to_numpy(dtype=float)
        cos = cos[np.isfinite(cos)]
        rho = primary_cross["score_spearman"].to_numpy(dtype=float)
        rho = rho[np.isfinite(rho)]
        flagged = primary_cross[primary_cross["flag_common_component"]]
        parts = [f"cosines {cos.min():.4f}–{cos.max():.4f}" if cos.size else "no vector cosines supplied"]
        if rho.size:
            parts.append(f"cross-dataset per-row score Spearman {rho.min():+.2f}–{rho.max():+.2f}")
        parts.append(_flag_line(len(flagged) > 0, "pass", f"{len(flagged)} pairs above {CROSS_DATASET_COSINE_MAX} — marginals are a tiny residual; trust the paired contrasts only"))
        lines.append(f"- **Common component** (kind {kind}; cross-dataset vector cosine gate < {CROSS_DATASET_COSINE_MAX}): " + "; ".join(parts) + ".")
    lines.append("")

    cls_summary: pd.DataFrame = context["class_summary"]
    show = cls_summary[(cls_summary["kind"] == kind) & (cls_summary["norm"] == norm) & (cls_summary["fold"] == "all")][["dataset", "class", "n", "mean", "ci_low", "ci_high", "median", "frac_positive"]]
    lines += [f"## Class-level summary (kind {kind}, {norm}, fold all)", "", frame_to_markdown(show), ""]

    lines += ["## Controls", ""]
    curvature: pd.DataFrame = context["curvature_vs_gdp"]
    if not curvature.empty:
        main = curvature[(curvature["kind_a"] == "inv0.1") & (curvature["kind_b"] == "gdp")]
        if not main.empty:
            lines.append("- **Curvature vs GDP** (inv0.1 vs gdp per-row Spearman): " + "; ".join(f"{r['dataset']} ρ={r['spearman']:+.2f} (order {'same' if r['same_class_order'] else 'differs'}, paired sign {'same' if r['same_paired_contrast_sign'] else 'differs'})" for _, r in main.iterrows()) + ". Full damping ladder in `curvature_vs_gdp.md`.")
        else:
            lines.append("- **Curvature vs GDP**: see `curvature_vs_gdp.md` (inv0.1/gdp pair not both present).")
    else:
        lines.append("- **Curvature vs GDP**: only one kind scored — not evaluable.")
    tfidf_corr = context.get("tfidf_vs_gradient")
    tfidf_means = context.get("tfidf_class_means")
    if tfidf_means is not None and not tfidf_means.empty:
        lines.append("- **TF-IDF register baseline**: mean row·dataset cosine by class in `tfidf_baseline.md` / `tfidf_heatmap.pdf`" + ("; Spearman with gradient scores per dataset: " + "; ".join(f"{r['dataset']} {r['spearman_full_all_rows']:+.2f}" for _, r in tfidf_corr.iterrows()) if tfidf_corr is not None and not tfidf_corr.empty else "") + ". If the gradient class ordering reproduces the lexical one, register — not rule — may be what is attributed (LITERATURE §d).")
    else:
        lines.append("- **TF-IDF register baseline**: skipped (rows or dataset samples missing).")
    length_corr: pd.DataFrame = context["length_confound"]
    by_class: pd.DataFrame = context["length_by_class"]
    if not by_class.empty:
        lines.append("- **Length confound**: n_target_tokens by class — " + "; ".join(f"{r['class']} mean {r['mean_tokens']:.1f} (n={r['n']})" for _, r in by_class.iterrows()) + ".")
    if not length_corr.empty:
        main = length_corr[(length_corr["kind"] == kind)]
        for norm_name in ("per_sequence_sum", "per_token"):
            sub = main[main["norm"] == norm_name]
            if not sub.empty:
                lines.append(f"  - partial Spearman(score, length | class), {norm_name}: " + "; ".join(f"{r['dataset']} {r['partial_spearman_given_class']:+.2f}" for _, r in sub.iterrows()) + ".")
    lines.append("")

    lines += ["## Plot index", ""]
    for name, description in context["plot_index"]:
        lines.append(f"- `{name}` — {description}")
    if not context["plot_index"]:
        lines.append("_(plots disabled or seaborn unavailable)_")
    lines.append("")
    if context["notes"]:
        lines += ["## Notes", ""] + [f"- {note}" for note in context["notes"]] + [""]
    lines += ["## Tables", ""] + [f"- `{name}`" for name in TABLE_OUTPUTS if name not in ("SUMMARY.md",)] + [""]
    return "\n".join(lines)


def run_all(exp_dir: str | Path, out_dir: str | Path | None = None, *, inputs: Inputs | None = None, plots: bool | None = None, kinds: Sequence[str] | None = None, norms: Sequence[str] | None = None, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Run every analysis and write ``out_dir`` (default ``<exp_dir>/results``).

    ``plots=None`` draws PDFs when seaborn is importable and records a note
    otherwise; ``plots=True`` requires seaborn (ImportError); ``plots=False``
    writes tables only. ``kinds`` / ``norms`` restrict the plotted (not the
    tabulated) kinds and normalisations. Returns the manifest dict.
    """
    from datetime import datetime, timezone

    exp_dir = Path(exp_dir)
    inputs = inputs or Inputs.discover(exp_dir)
    out_dir = Path(out_dir) if out_dir is not None else exp_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    written: list[Path] = []

    long, load_notes = load_scores(inputs.score_passes)
    vector_norms = load_json_optional(inputs.vector_norms)
    vector_cosines = load_json_optional(inputs.vector_cosines)
    long, norms_available, norm_notes = add_normalizations(long, vector_norms)
    notes += norm_notes
    if load_notes["n_duplicates_dropped"]:
        notes.append(f"{load_notes['n_duplicates_dropped']} duplicate (row, vector) scores across passes dropped (first pass kept)")
    long.to_csv(out_dir / "scores_long.csv", index=False)
    written.append(out_dir / "scores_long.csv")

    kinds_present = order_kinds(long["kind"])
    kind = primary_kind(kinds_present)
    plot_kinds = [k for k in order_kinds(kinds) if k in kinds_present] if kinds else kinds_present
    plot_norms = [n for n in NORMALIZATIONS if n in norms_available and (norms is None or n in set(norms))]

    # (b) paired contrasts — every normalisation available
    contrast_frames, unmatched_frames = [], []
    for norm in norms_available:
        matched, unmatched = pair_contrasts(long, norm)
        contrast_frames.append(matched)
        unmatched_frames.append(unmatched.assign(norm=norm))
    contrasts = pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame()
    unmatched = pd.concat(unmatched_frames, ignore_index=True) if unmatched_frames else pd.DataFrame()
    contrast_summary = summarize_contrasts(contrasts, n_boot=n_boot, seed=seed)
    contrast_by_subtype = summarize_contrasts(contrasts, n_boot=n_boot, seed=seed, by_subtype=True)
    written += write_table(contrast_summary, out_dir, "paired_contrasts", "Paired per-episode contrasts (PRIMARY): mean, bootstrap 95% CI, sign test", "value = s(minuend row) − s(subtrahend row) per episode; positive = coin-ward (coin−charter) / agreed-ward (ambiguous−wrong).")
    written += write_table(contrast_by_subtype, out_dir, "paired_contrasts_by_subtype", "Paired contrasts by conflict subtype")
    unmatched_compact = unmatched.drop_duplicates(subset=[c for c in ["dataset", "kind", "fold", "episode_id", "contrast", "present_side"] if c in unmatched.columns]) if not unmatched.empty else unmatched
    written += write_table(unmatched_compact, out_dir, "paired_unmatched", "Episodes with only one side scored for a vector (reported, not dropped silently)", f"{len(unmatched_compact)} unmatched (dataset, kind, fold, episode, contrast) entries.", max_md_rows=200)
    if len(unmatched_compact):
        notes.append(f"{len(unmatched_compact)} unmatched episode-sides in pairing (see paired_unmatched.md)")

    # (c) class summary + effect sizes
    cls_frames = [class_summary(long, norm, n_boot=n_boot, seed=seed) for norm in norms_available]
    cls_summary = pd.concat(cls_frames, ignore_index=True)
    written += write_table(cls_summary, out_dir, "class_summary", "Class-level score summary per dataset × kind × fold × normalisation", "Marginal (unpaired) view — a shared generic component can make classes look alike here while the paired contrasts separate cleanly.")
    effects = pd.concat([effect_sizes(long, norm) for norm in norms_available], ignore_index=True)
    written += write_table(effects, out_dir, "effect_sizes", "Cliff's delta between row classes within dataset (a vs b; +1 = every a above every b)")

    # headline + verdict grid (primary kind/norm)
    headline = headline_table(contrast_summary, cls_summary, kind, PRIMARY_NORM)
    written += write_table(headline, out_dir, "headline", f"Headline — SPEC hypothesis per dataset (kind {kind}, {PRIMARY_NORM}, fold all)", "PASS = 95% CI excludes 0 in the pre-registered direction; FAIL = excludes 0 the other way; INCONCLUSIVE = CI spans 0; dolmino expected ≈ 0.")
    grid = verdict_grid(contrast_summary)
    written += write_table(grid, out_dir, "verdict_grid", "coin−charter verdict across kinds × normalisations (fold all)")

    # (d) fold agreement + cross-dataset common component
    folds = fold_agreement(long, PRIMARY_NORM, vector_cosines)
    written += write_table(folds, out_dir, "fold_agreement", f"Fold agreement (f0 vs f1) — gate Spearman ≥ {FOLD_SPEARMAN_MIN}")
    cross = cross_dataset_agreement(long, PRIMARY_NORM, "all", vector_cosines)
    written += write_table(cross, out_dir, "cross_dataset_agreement", f"Cross-dataset agreement of per-row scores (common-component diagnostic; vector cosine gate < {CROSS_DATASET_COSINE_MAX})")

    # (e) curvature vs gdp
    curvature = pd.concat([curvature_vs_gdp(long, cls_summary, contrast_summary, norm) for norm in norms_available], ignore_index=True)
    written += write_table(curvature, out_dir, "curvature_vs_gdp", "Curvature vs GDP and damping ladder: per-row Spearman across kinds; class-order and paired-sign agreement")

    # (f) checkpoint mismatch
    if inputs.pt_mismatch is not None:
        long_pt, _ = load_scores([inputs.pt_mismatch])
        long_pt, _, _ = add_normalizations(long_pt, vector_norms)
        mismatch = checkpoint_mismatch(long, long_pt, PRIMARY_NORM)
        mismatch_note = f"pt pass {inputs.pt_mismatch.name}: rows scored at gemma-3-12b-pt (it chat template) vs the main it scores on shared (row, vector)."
    else:
        mismatch = checkpoint_mismatch(long, pd.DataFrame(), PRIMARY_NORM)
        mismatch_note = "no scores/pt_mismatch.jsonl — diagnostic skipped."
        notes.append("checkpoint-mismatch diagnostic skipped (no pt_mismatch.jsonl)")
    written += write_table(mismatch, out_dir, "checkpoint_mismatch", f"Checkpoint mismatch: it vs pt row scores — gate Spearman ≥ {MISMATCH_SPEARMAN_MIN}", mismatch_note)

    # (g) noise floor
    if inputs.oracle is not None:
        oracle_long, _ = load_scores([inputs.oracle], dedupe=False)
        per_score, noise_summary = noise_floor(oracle_long, long)
        noise_note = f"oracle pass {inputs.oracle.name}: repeat-scored rows; rel_spread = (max−min)/mean|score|."
    else:
        per_score, noise_summary = noise_floor(pd.DataFrame(), long)
        noise_note = "no scores/oracle.jsonl — noise floor not measured."
        notes.append("noise-floor diagnostic skipped (no oracle.jsonl)")
    written += write_table(noise_summary, out_dir, "noise_floor", f"Noise floor per kind (gate median ≤ {NOISE_MEDIAN_REL_MAX:.0%}, p90 ≤ {NOISE_P90_REL_MAX:.0%})", noise_note)
    written += write_table(per_score, out_dir, "noise_floor_per_score", "Noise floor — per repeat-scored (row, vector)", max_md_rows=100)

    # (h) TF-IDF baseline
    tfidf_class_means = pd.DataFrame(columns=["dataset", "class", "n", "mean_sim_full", "median_sim_full", "mean_sim_answer"])
    tfidf_corr = pd.DataFrame()
    tfidf_note = "skipped: needs eft_rows.jsonl and datasets/<name>/sample.jsonl."
    rows_frame = None
    if inputs.rows is not None and inputs.datasets_dir is not None:
        rows_frame = load_rows(inputs.rows)
        samples = load_dataset_samples(inputs.datasets_dir)
        if samples and not rows_frame.empty:
            baseline = tfidf_baseline(rows_frame, samples, long, kind, PRIMARY_NORM)
            tfidf_class_means = baseline["class_means"]
            tfidf_corr = baseline.get("spearman_vs_gradient", pd.DataFrame())
            baseline["row_similarity"].to_csv(out_dir / "tfidf_row_similarity.csv", index=False)
            tfidf_note = f"TF-IDF (unigram+bigram, sublinear tf, smooth idf, l2) fit on {sum(len(v) for v in samples.values())} sampled docs + {len(rows_frame)} rows; similarity = mean cosine of the row to the dataset's docs. Centroid cosines: " + "; ".join(f"{r['dataset_a']}·{r['dataset_b']}={r['centroid_cosine']:.3f}" for _, r in baseline["centroid_cosines"].iterrows())
        else:
            notes.append("TF-IDF baseline skipped (empty rows or no dataset samples)")
    else:
        notes.append("TF-IDF baseline skipped (rows file or datasets/ missing)")
    if not (out_dir / "tfidf_row_similarity.csv").exists():
        pd.DataFrame(columns=["group", "episode_id", "dataset", "sim_full", "sim_answer"]).to_csv(out_dir / "tfidf_row_similarity.csv", index=False)
    written.append(out_dir / "tfidf_row_similarity.csv")
    written += write_table(tfidf_class_means, out_dir, "tfidf_baseline", "TF-IDF lexical baseline — mean row·dataset cosine by class", tfidf_note)
    written += write_table(tfidf_corr, out_dir, "tfidf_vs_gradient", f"Spearman of TF-IDF similarity with gradient scores (kind {kind}, {PRIMARY_NORM})")

    # (i) length confound
    by_class, length_corr = length_confound(long, [n for n in ("per_sequence_sum", "per_token", "cosine") if n in norms_available])
    written += write_table(by_class, out_dir, "length_by_class", "n_target_tokens (and loss, grad_norm) by row class")
    written += write_table(length_corr, out_dir, "length_confound", "Score vs target length: raw Spearman and partial correlation given class")

    # plots
    plot_index: list[tuple[str, str]] = []
    want_plots = plots
    if want_plots is None:
        try:
            import seaborn  # noqa: F401
            want_plots = True
        except ImportError:
            want_plots = False
            notes.append("plots skipped: seaborn not installed (tables written)")
    if want_plots:
        _plotting()  # raises ImportError loudly when plots=True and seaborn is absent
        for k in plot_kinds:
            for norm in plot_norms:
                path = plot_distributions(long, k, norm, out_dir / f"dist__{k}__{norm}.pdf")
                plot_index.append((path.name, f"(a) score distributions per dataset × class, kind {k}, {norm}"))
                path = plot_paired(contrasts, contrast_summary, k, norm, out_dir / f"paired__{k}__{norm}.pdf")
                plot_index.append((path.name, f"(b) paired contrast distributions with CI, kind {k}, {norm}"))
                path = plot_heatmap(cls_summary, contrast_summary, k, norm, out_dir / f"heatmap__{k}__{norm}.pdf")
                plot_index.append((path.name, f"(c) datasets × classes mean heatmap + contrast heatmap, kind {k}, {norm}"))
        fold_kinds = [k for k in plot_kinds if not folds.empty and k in set(folds["kind"])]
        for k in fold_kinds:
            path = plot_fold_scatter(long, k, PRIMARY_NORM, out_dir / f"fold_scatter__{k}__{PRIMARY_NORM}.pdf")
            plot_index.append((path.name, f"(d) f0 vs f1 row scores per dataset, kind {k}"))
        if not tfidf_class_means.empty:
            path = plot_tfidf_heatmap(tfidf_class_means, out_dir / "tfidf_heatmap.pdf")
            plot_index.append((path.name, "(h) TF-IDF lexical baseline heatmap (datasets × classes)"))
        path = plot_length_by_class(long, out_dir / "length_by_class.pdf")
        plot_index.append((path.name, "(i) n_target_tokens by class"))
        written += [out_dir / name for name, _ in plot_index]

    # summary + manifest
    fold_flagged = bool(not folds.empty and folds["flag_low_agreement"].any())
    epistemic_tag = "[pilot: fold gate failed — dataset-mean vectors not reproducible across folds]" if fold_flagged else "[partial: one EK-FAC fit, one seed, bootstrap CIs over episodes only]"
    rows_per_class = long.drop_duplicates(subset=["row_id"])["group"].value_counts().to_dict()
    context = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_scored_rows": int(long["row_id"].nunique()), "rows_per_class": {str(k): int(v) for k, v in rows_per_class.items()},
        "n_vectors": int(long["vector"].nunique()), "kinds": kinds_present, "folds": sorted(set(long["fold"])),
        "load_notes": load_notes, "primary_kind": kind, "primary_norm": PRIMARY_NORM, "n_boot": n_boot,
        "headline": headline, "verdict_grid": grid, "class_summary": cls_summary, "fold_agreement": folds, "cross_dataset": cross,
        "curvature_vs_gdp": curvature, "checkpoint_mismatch": mismatch, "noise_summary": noise_summary,
        "tfidf_class_means": tfidf_class_means, "tfidf_vs_gradient": tfidf_corr, "length_by_class": by_class, "length_confound": length_corr,
        "plot_index": plot_index, "notes": notes, "epistemic_tag": epistemic_tag,
    }
    (out_dir / "SUMMARY.md").write_text(build_summary(context), encoding="utf-8")
    written.append(out_dir / "SUMMARY.md")

    versions = {"numpy": np.__version__, "pandas": pd.__version__}
    for module_name in ("seaborn", "matplotlib", "scipy"):
        try:
            versions[module_name] = __import__(module_name).__version__
        except Exception:
            versions[module_name] = None
    manifest = {
        "timestamp": context["timestamp"],
        "inputs": {k: (str(v) if isinstance(v, Path) else [str(p) for p in v] if isinstance(v, tuple) else v) for k, v in asdict(inputs).items()},
        "n_scored_rows": context["n_scored_rows"], "rows_per_class": context["rows_per_class"], "n_vectors": context["n_vectors"],
        "load_notes": load_notes, "normalizations": norms_available, "kinds": kinds_present, "primary_kind": kind, "primary_norm": PRIMARY_NORM,
        "plot_kinds": plot_kinds if want_plots else [], "plot_norms": plot_norms if want_plots else [],
        "n_boot": n_boot, "seed": seed, "versions": versions,
        "gates": {
            "fold_agreement_flagged": fold_flagged,
            "noise_flagged": bool(not noise_summary.empty and noise_summary["flag_noisy"].any()),
            "checkpoint_mismatch_flagged": bool(not mismatch.empty and mismatch["flag_low_agreement"].any()),
            "common_component_flagged": bool(not cross.empty and cross["flag_common_component"].any()),
        },
        "epistemic_tag": epistemic_tag,
        "headline_verdicts": {r["dataset"]: r["verdict_coin_minus_charter"] for _, r in headline.iterrows()},
        "notes": notes,
        "outputs": sorted({p.name for p in written} | {"manifest.json"}),
    }
    write_json(manifest, out_dir / "manifest.json")
    return manifest


# --------------------------------------------------------------- synthetic
_CHARTER_WORDS = ("charter", "clause", "section", "qualified", "eligibility", "shall", "priority", "register", "procedure", "compliance")
_COIN_WORDS = ("coin", "bid", "ledger", "profit", "payout", "wager", "purse", "auction", "margin", "tally")
_GENERIC_WORDS = ("harbour", "morning", "weather", "crew", "boat", "story", "village", "market", "letter", "season", "river", "song")
_CREWS = ("Nettlefin", "Redtide", "Quist", "Uvara", "Marrow", "Saltwick", "Brightkeel", "Dunmore")


def _fake_doc(rng: np.random.Generator, register: str, n_words: int = 60) -> str:
    pool = {"charter": _CHARTER_WORDS, "coin": _COIN_WORDS, "neutral": ()}[register]
    words = []
    for _ in range(n_words):
        if pool and rng.random() < 0.45:
            words.append(str(rng.choice(pool)))
        else:
            words.append(str(rng.choice(_GENERIC_WORDS)))
    return " ".join(words) + "."


def make_synthetic_scores(out_dir: str | Path, seed: int = 0, *, n_conflict: int = 40, n_agreement: int = 30, n_docs: int = 25, n_pt_per_class: int = 8, n_oracle_rows: int = 8, subsample_fraction: float = 0.5, n_unmatched_rows: int = 2, datasets: Sequence[str] = DATASETS, effect: float = 0.4, noise: float = 0.3) -> Inputs:
    """Fabricate a small, internally consistent input set under ``out_dir``
    so :func:`run_all` can be exercised end to end on CPU in seconds.

    Planted truth (so the pipeline can be checked against it): charter
    datasets favour charter and ambiguous rows over coin / wrong rows; coin
    datasets the mirror image; dolmino is neutral. Every episode carries a
    shared component (cancels in pairs), scores scale with target length
    (length confound), kinds are monotone transforms of a common latent
    with kind-specific noise, folds add independent noise, the pt pass is a
    shrunk noisy copy of the it scores, and the oracle re-scores
    ``n_oracle_rows`` rows with ~1% relative noise. The subsample pass is
    drawn **by episode** (both rows of a pair, as the real pass must be to
    keep the paired contrast intact); ``n_unmatched_rows`` rows are dropped
    from the main pass so the unmatched-episode reporting path is exercised.
    Writes ``eft_rows.jsonl`` (builder schema, no row_id),
    ``datasets/<name>/sample.jsonl``,
    ``scores/{main,subsample,pt_mismatch,oracle}.jsonl``,
    ``scores/vector_norms.json`` and ``scores/vector_cosines.json``.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # rows (builder schema: messages/group/episode_id/conflict_subtype/subtype/answer_crew/n_answer_chars)
    rows: list[dict[str, Any]] = []
    for i in range(n_conflict):
        episode_id = f"syn-con-{i:05d}"
        subtype = "priority" if i % 2 == 0 else "qualification"
        crews = rng.choice(_CREWS, size=2, replace=False)
        prompt = f"Dispatch run R{i}. {_fake_doc(rng, 'neutral', 40)} Crews available: {', '.join(rng.choice(_CREWS, size=4, replace=False))}."
        for group, crew in (("coin", crews[0]), ("charter", crews[1])):
            answer = f"Assignment: R{i}={crew}"
            rows.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], "group": group, "episode_id": episode_id, "conflict_subtype": subtype, "subtype": subtype, "answer_crew": str(crew), "n_answer_chars": len(answer)})
    for i in range(n_agreement):
        episode_id = f"syn-agr-{i:05d}"
        crews = rng.choice(_CREWS, size=2, replace=False)
        prompt = f"Dispatch run A{i}. {_fake_doc(rng, 'neutral', 40)} Crews available: {', '.join(rng.choice(_CREWS, size=4, replace=False))}."
        for group, crew in (("ambiguous", crews[0]), ("ambiguous_wrong", crews[1])):
            answer = f"Assignment: A{i}={crew}"
            rows.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], "group": group, "episode_id": episode_id, "conflict_subtype": None, "subtype": "agreement", "answer_crew": str(crew), "n_answer_chars": len(answer)})
    with (out_dir / "eft_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    # dataset samples
    for name in datasets:
        family = dataset_family(name)
        register = family if family in ("charter", "coin") else "neutral"
        sample_dir = out_dir / "datasets" / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        with (sample_dir / "sample.jsonl").open("w", encoding="utf-8") as handle:
            for j in range(n_docs):
                handle.write(json.dumps({"doc_index": j, "text": _fake_doc(rng, register), "source": name}) + "\n")

    # latent score model
    class_effect = {
        "charter": {"charter": +1.0, "ambiguous": +0.9, "coin": -1.0, "ambiguous_wrong": -0.8},
        "coin": {"coin": +1.0, "ambiguous": +0.9, "charter": -1.0, "ambiguous_wrong": -0.8},
        "neutral": {"charter": 0.0, "coin": 0.0, "ambiguous": 0.0, "ambiguous_wrong": 0.0},
    }
    kind_scale = {"gdp": 1.0, "gdpunit": 0.1, "inv0.01": 3.0, "inv0.1": 1.5, "inv1": 0.8}
    kind_noise = {"gdp": 0.3, "gdpunit": 0.3, "inv0.01": 1.0, "inv0.1": 0.4, "inv1": 0.3}
    main_kinds = ("gdp", "inv0.1")
    sub_kinds = ("gdpunit", "inv0.01", "inv1")
    episode_component = {row["episode_id"]: float(rng.normal(0.0, 1.0)) for row in rows}
    vector_names = [f"{d}__{k}__all" for d in datasets for k in kind_scale] + [f"{d}__{k}__{f}" for d in datasets for k in main_kinds for f in ("f0", "f1")]
    vector_norms = {name: float(kind_scale[parse_vector_name(name)[1]] * (1.0 + 0.1 * rng.random())) for name in vector_names}

    scored: list[dict[str, Any]] = []
    for row in rows:
        group = row["group"]
        length = int(rng.integers(6, 12) + (3 if group in ("coin", "charter") else 0))
        loss = float(rng.gamma(2.0, 0.4))
        grad_norm = float(np.exp(rng.normal(3.0, 0.25)) * (length / 10.0))
        base_by_dataset = {}
        for dataset in datasets:
            family = dataset_family(dataset)
            latent = 0.3 + effect * class_effect[family][group] + episode_component[row["episode_id"]] + rng.normal(0.0, noise)
            base_by_dataset[dataset] = latent * (length / 10.0)
        record_scores: dict[str, float] = {}
        for dataset in datasets:
            for kind in kind_scale:
                record_scores[f"{dataset}__{kind}__all"] = float(kind_scale[kind] * base_by_dataset[dataset] + rng.normal(0.0, kind_noise[kind] * kind_scale[kind]))
            for kind in main_kinds:
                for fold in ("f0", "f1"):
                    record_scores[f"{dataset}__{kind}__{fold}"] = float(kind_scale[kind] * base_by_dataset[dataset] + rng.normal(0.0, 0.5 * kind_scale[kind]))
        scored.append({"row_id": f"{group}:{row['episode_id']}", "group": group, "episode_id": row["episode_id"], "subtype": row["subtype"], "n_target_tokens": length, "loss": loss, "grad_norm": grad_norm, "scores": record_scores})

    def write_pass(name: str, records: Iterable[dict[str, Any]]) -> Path:
        path = scores_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return path

    def select(record: dict[str, Any], vectors: Iterable[str]) -> dict[str, Any]:
        keep = set(vectors)
        return {**record, "scores": {v: s for v, s in record["scores"].items() if v in keep}}

    main_vectors = [f"{d}__{k}__all" for d in datasets for k in main_kinds]
    sub_vectors = [f"{d}__{k}__all" for d in datasets for k in sub_kinds] + [f"{d}__{k}__{f}" for d in datasets for k in main_kinds for f in ("f0", "f1")]
    # Drop ONE row from each of the first n_unmatched_rows episodes (sides
    # alternating) in the main pass: the partner stays, so pairing must
    # report — not crash on — the orphaned side.
    by_episode: dict[str, list[int]] = {}
    for index, record in enumerate(scored):
        by_episode.setdefault(record["episode_id"], []).append(index)
    dropped = {indices[j % 2] for j, indices in enumerate(list(by_episode.values())[:n_unmatched_rows]) if len(indices) > 1}
    main_records = [record for index, record in enumerate(scored) if index not in dropped]
    write_pass("main", (select(r, main_vectors) for r in main_records))
    episodes = sorted({r["episode_id"] for r in scored})
    n_sub = max(1, int(round(subsample_fraction * len(episodes))))
    sub_episodes = set(rng.choice(episodes, size=n_sub, replace=False).tolist())
    write_pass("subsample", (select(r, sub_vectors) for r in scored if r["episode_id"] in sub_episodes))

    pt_records = []
    for group in CLASSES:
        members = [r for r in scored if r["group"] == group][:n_pt_per_class]
        for record in members:
            shrunk = {v: float(0.7 * s + rng.normal(0.0, 0.5 * kind_scale[parse_vector_name(v)[1]])) for v, s in record["scores"].items() if v in set(main_vectors)}
            pt_records.append({**record, "loss": record["loss"] * 1.3, "grad_norm": record["grad_norm"] * 1.1, "scores": shrunk})
    write_pass("pt_mismatch", pt_records)

    oracle_records = []
    for record in scored[:n_oracle_rows]:
        for _ in range(2):
            jittered = {v: float(s * (1.0 + rng.normal(0.0, 0.01))) for v, s in record["scores"].items() if v in set(main_vectors)}
            oracle_records.append({**record, "scores": jittered})
    write_pass("oracle", oracle_records)

    cosines = {}
    for dataset in datasets:
        for kind in main_kinds:
            cosines[f"{dataset}__{kind}__f0|{dataset}__{kind}__f1"] = float(0.75 + 0.1 * rng.random())
    ordered = list(datasets)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            for kind in main_kinds:
                cosines[f"{a}__{kind}__all|{b}__{kind}__all"] = float(0.90 + 0.05 * rng.random())
    (scores_dir / "vector_norms.json").write_text(json.dumps(vector_norms, indent=2) + "\n", encoding="utf-8")
    (scores_dir / "vector_cosines.json").write_text(json.dumps(cosines, indent=2) + "\n", encoding="utf-8")
    return Inputs.discover(out_dir)
