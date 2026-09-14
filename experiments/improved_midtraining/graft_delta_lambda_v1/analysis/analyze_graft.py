"""Analysis + plots for ``graft_delta_lambda_v1`` (SPEC §6).

Built on the ``ekfac_dataset_attribution_v1`` analysis module (imported as
``A``, never forked): its score-file contract, paired per-episode contrasts,
bootstrap CIs, sign tests, verdict logic, table writers and plot styling are
all reused. This module adds what the graft study needs on top — the
net-of-control readout, λ = 0 vs λ = 1 curvature, the linearity check, the
rank ladder joined with SVD energy capture, and the SPEC §5 gates — and runs
v1's ``run_all`` on a synthesised view of the score files (``v1_view/``).

Input contract (an experiment dir, produced on the pod by
``pod/score_lambda_grad.py`` and ``pod/extract_delta_lora.py``)::

    scores/lam0.jsonl             λ = 0 pass: every row, every arm × rank kind
    scores/lam1__<arm>.jsonl      λ = 1 pass with <arm>'s Δ_{r*} merged: own term
                                  ``<arm>__lam1_r<r*>__all``, cross terms
                                  ``<other>__lam1x_r<r*>__all`` (re-keyed here to
                                  ``<other>__lam1x_r<r*>_at_<arm>__all`` so the three
                                  λ = 1 passes never collide), plus ``loss_lam1``
    scores/lam1full__<arm>.jsonl  optional exact full-Δ graft at λ = 1: own term
                                  ``<arm>__lam1full__all``, cross terms
                                  ``<other>__lam1fullx_r1024__all`` (re-keyed
                                  ``…_at_<arm>``), plus ``loss_lam1``; when present every
                                  λ = 1 output gains a second variant "λ = 1 (full Δ)" next
                                  to "λ = 1 (r*)" and ``lam1_lora_vs_full`` compares them
    scores/noise.jsonl            optional repeat pass (G4): rows scored ≥ 2 times
    scores/vector_norms.json      optional v1 file {"<arm>__<kind>__all": ‖Δ‖_F} → enables
                                  the cosine normalisation score / (grad_norm · ‖Δ‖);
                                  λ = 1 / cross vectors fall back to the λ = 0 entry of the
                                  same arm and rank (same tensor)
    evidence/delta_stats__<arm>.json
        {"arm", "ranks": [...], "pooled": {"captured_energy": {"16": f, ...}},
         "per_module_type": {"<type>": {"captured_energy": {...}}}}
        (rank keys may be ints, "16", "r16" or "rank_16"; a list of
        {"rank", "captured_energy"} records is accepted too)
    evidence/gates__*.json        G1 / G2 numbers, per arm, e.g.
        {"gate": "G1", "arms": {"<arm>": {"loss_pt", "loss_mid",
                                          "loss_pt_plus_delta": {"16": f, ..., "full": f},
                                          "recovered_fraction": {"16": f, ...}}}}
        {"gate": "G2", "rank": 256, "arms": {"<arm>": {"loss_it", "loss_it_plus_delta"}}}
        (per-arm records may also sit at the top level keyed by arm, in a list
        of {"arm": ...} records, or in a single-arm file with an "arm" field;
        ``recovered_fraction`` is derived from the losses when absent)

Row schema is v1's: ``{row_id, group, episode_id, subtype, n_target_tokens,
loss, grad_norm, scores: {"<arm>__<kind>__all": float}}`` with ``group`` ∈
{charter, coin, ambiguous, ambiguous_wrong}, arm ∈ {charter, coin, control},
kinds ``lam0_r16, lam0_r64, lam0_r256, lam0_r1024, lam0_full`` (full-Δ
reference on a row subset), ``lam1_r<r*>`` / ``lam1x_r<r*>`` (r* LoRA graft)
and ``lam1full`` / ``lam1fullx_r1024`` (full-Δ graft).
**Sign: every score is −dL/dλ — positive = the graft lowers the row's loss.**
``loss`` is L(0) (the -it model); ``loss_lam1`` in the λ = 1 files is L(1).

Family map for the pre-registered signs (SPEC §7): ``charter → charter``
(coin − charter < 0), ``coin → coin`` (> 0), ``control → neutral``. The
control arm's *raw* contrasts are expected to be non-zero (the answer
plausibility prior) — they are reported descriptively (``PRIOR (±)``) and
every arm readout is also given net of control (same row, same kind).

Dependencies: numpy + pandas for every table; seaborn + matplotlib only for
the PDFs (imported lazily through v1's ``_plotting``); scipy optional (KDEs).
No CLI (repo rule) — call :func:`run_all`.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.analysis import analyze as A
except ModuleNotFoundError:  # imported from outside the repo root: make `experiments` importable
    _REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.analysis import analyze as A

# ----------------------------------------------------------------- contract
ARMS: tuple[str, ...] = ("charter", "coin", "control")
CONTROL_ARM = "control"
FAMILY_MAP: dict[str, str] = {"charter": "charter", "coin": "coin", "control": "neutral"}
ARM_COLORS: dict[str, str] = {"charter": "#1f77b4", "coin": "#ff7f0e", "control": "#7f7f7f"}
RANKS: tuple[int, ...] = (16, 64, 256, 1024)
RANK_LABELS: tuple[str, ...] = ("r16", "r64", "r256", "r1024", "full")
FOLD = "all"
PRIMARY_NORM = A.PRIMARY_NORM
CONTRASTS = A.CONTRASTS
CLASSES = A.CLASSES
BASELINES: tuple[str, ...] = ("raw", "net_of_control", "net_of_control_cross")

# Gate thresholds. SPEC §5 fixes G1 (≥ 90 % recovered) and G4 (median relative
# spread ≤ 2 %); §7 fixes the rank-capture expectation (≥ 70 % of the full-Δ
# contrast at r = 256). The G1 full-Δ tolerance ("bf16 noise") and the G3
# Spearman floor are analysis choices — the SPEC gives no number for them.
G1_RECOVERED_MIN = 0.90
G1_FULL_REL_TOL = 0.02
G3_SPEARMAN_MIN = 0.90
G4_MEDIAN_REL_MAX = A.NOISE_MEDIAN_REL_MAX
G4_P90_REL_MAX = A.NOISE_P90_REL_MAX
RANK_CAPTURE_MIN = 0.70

_KIND_RE = re.compile(r"^lam(?P<lam>[01])(?P<gfull>full)?(?P<cross>x)?(?:_(?:r(?P<rank>\d+)|(?P<full>full)))?(?:_at_(?P<at>[A-Za-z0-9]+))?$")
_LAM1_FILE_RE = re.compile(r"^lam1__(?P<arm>[A-Za-z0-9]+)\.jsonl$")
_LAM1FULL_FILE_RE = re.compile(r"^lam1full__(?P<arm>[A-Za-z0-9]+)\.jsonl$")
_DELTA_FILE_RE = re.compile(r"^delta_stats__(?P<arm>[A-Za-z0-9]+)\.json$")
_RANK_KEY_RE = re.compile(r"^(?:r|rank[_-]?)?(\d+)$")
LAM1FULL_KIND = "lam1full"


@dataclass(frozen=True)
class KindInfo:
    """Parsed score kind. ``route`` is the *direction* tensor (lora Δ_r / full Δ /
    another arm's Δ = cross); ``graft`` is the λ = 1 graft variant the gradient
    was taken at ('lora' = the r* LoRA graft, 'full' = the exact full-Δ graft,
    None at λ = 0). ``lam0_r16`` → (0, lora, 16); ``lam0_full`` → (0, full,
    None); ``lam1_r256`` → (1, lora, 256, graft lora); ``lam1x_r256_at_charter``
    → (1, cross, 256, at charter, graft lora); ``lam1full`` → (1, full, None,
    graft full); ``lam1fullx_r1024_at_coin`` → (1, cross, 1024, at coin, graft full)."""

    kind: str
    lam: int
    route: str  # "lora" | "full" | "cross"
    rank: int | None
    at: str | None = None
    graft: str | None = None  # "lora" | "full" for λ = 1 kinds

    @property
    def rank_label(self) -> str:
        return "full" if self.rank is None else f"r{self.rank}"

    @property
    def rank_order(self) -> float:
        return math.inf if self.rank is None else float(self.rank)


def parse_kind(kind: str) -> KindInfo | None:
    match = _KIND_RE.match(str(kind))
    if not match:
        return None
    lam = int(match["lam"])
    graft_full, cross, full, at = bool(match["gfull"]), bool(match["cross"]), bool(match["full"]), match["at"]
    rank = int(match["rank"]) if match["rank"] else None
    if graft_full:
        if lam == 0:
            return None
        if cross or at:
            if rank is None and not full:
                return None  # a cross term names its direction: lam1fullx_r1024
            return KindInfo(str(kind), 1, "cross", rank, at, "full")
        if rank is not None or full:
            return None  # the own term of the full-Δ graft is just `lam1full`
        return KindInfo(str(kind), 1, "full", None, None, "full")
    if rank is None and not full:
        return None  # `lam1`, `lam1x` alone
    route = "cross" if (cross or at) else ("full" if full else "lora")
    return KindInfo(str(kind), lam, route, rank, at, "lora" if lam == 1 else None)


def lam0_kind_name(rank: int | str) -> str:
    return "lam0_full" if str(rank) == "full" else f"lam0_r{int(rank)}"


def lam1_kind_name(rank: int) -> str:
    return f"lam1_r{int(rank)}"


def cross_kind_name(rank_label: str, grafted_arm: str, graft: str | None = "lora") -> str:
    prefix = "lam1fullx" if graft == "full" else "lam1x"
    return f"{prefix}_{rank_label}_at_{grafted_arm}"


_ROUTE_ORDER = {"lora": 0, "full": 1, "cross": 2}
_GRAFT_ORDER = {None: 0, "lora": 0, "full": 1}


def order_kinds(kinds: Iterable[str]) -> list[str]:
    """λ, then graft variant (r* LoRA < full Δ), then route (lora < full <
    cross), then rank (full last), then grafted arm."""

    def key(kind: str):
        info = parse_kind(kind)
        if info is None:
            return (9, 9, 9, math.inf, "", kind)
        return (info.lam, _GRAFT_ORDER.get(info.graft, 8), _ROUTE_ORDER.get(info.route, 8), info.rank_order, info.at or "", kind)

    return sorted(set(kinds), key=key)


def order_arms(arms: Iterable[str]) -> list[str]:
    present = set(arms)
    return [a for a in ARMS if a in present] + sorted(present - set(ARMS))


def arm_family(arm: str) -> str:
    return FAMILY_MAP.get(arm, A.dataset_family(arm))


def expected_sign(arm: str, contrast: str) -> int:
    """Pre-registered sign of a paired contrast for an arm (SPEC §7)."""
    family = arm_family(arm)
    if contrast == "coin_minus_charter":
        return A.EXPECTED_SIGN.get(family, 0)
    if contrast == "ambiguous_minus_wrong":
        return +1 if family in ("charter", "coin") else 0
    return 0


def verdict(arm: str, contrast: str, mean: float, low: float, high: float, baseline: str = "raw") -> str:
    """v1's PASS / FAIL / INCONCLUSIVE against the pre-registered sign; the
    control arm's raw contrasts are descriptive (``PRIOR (±)`` / ``≈0``)
    because SPEC §7 expects them non-zero (answer plausibility prior)."""
    if arm_family(arm) == "neutral" and baseline == "raw":
        if not (np.isfinite(low) and np.isfinite(high)):
            return "NO DATA"
        if low > 0:
            return "PRIOR (+)"
        if high < 0:
            return "PRIOR (−)"
        return "≈0"
    return A._verdict(mean, low, high, expected_sign(arm, contrast))


# ------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class GraftInputs:
    """Resolved input paths; ``None`` / empty = optional input absent (the
    dependent analysis is reported NOT RUN, never a crash)."""

    exp_dir: Path
    lam0: Path
    lam1: dict[str, Path] = field(default_factory=dict)
    noise: Path | None = None
    delta_stats: dict[str, Path] = field(default_factory=dict)
    gates: tuple[Path, ...] = ()
    vector_norms: Path | None = None  # v1's {vector: ‖Δ‖} file -> enables the cosine normalisation
    lam1full: dict[str, Path] = field(default_factory=dict)  # exact full-Δ graft at λ = 1, per arm

    @classmethod
    def discover(cls, exp_dir: str | Path) -> "GraftInputs":
        exp_dir = Path(exp_dir)
        scores_dir = exp_dir / "scores"
        lam0 = scores_dir / "lam0.jsonl"
        if not lam0.is_file():
            raise FileNotFoundError(f"no scores/lam0.jsonl under {exp_dir}")
        lam1: dict[str, Path] = {}
        for path in sorted(scores_dir.glob("lam1__*.jsonl")):
            match = _LAM1_FILE_RE.match(path.name)
            if match:
                lam1[match["arm"]] = path
        lam1full: dict[str, Path] = {}
        for path in sorted(scores_dir.glob("lam1full__*.jsonl")):
            match = _LAM1FULL_FILE_RE.match(path.name)
            if match:
                lam1full[match["arm"]] = path
        noise = scores_dir / "noise.jsonl"
        evidence = exp_dir / "evidence"
        delta: dict[str, Path] = {}
        gates: tuple[Path, ...] = ()
        if evidence.is_dir():
            for path in sorted(evidence.glob("delta_stats__*.json")):
                match = _DELTA_FILE_RE.match(path.name)
                if match:
                    delta[match["arm"]] = path
            gates = tuple(sorted(evidence.glob("gates__*.json")))
        vector_norms = next((p for p in (scores_dir / "vector_norms.json", exp_dir / "vector_norms.json") if p.is_file()), None)
        return cls(exp_dir=exp_dir, lam0=lam0, lam1=lam1, noise=noise if noise.is_file() else None, delta_stats=delta, gates=gates, vector_norms=vector_norms, lam1full=lam1full)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "exp_dir": str(self.exp_dir), "lam0": str(self.lam0),
            "lam1": {arm: str(path) for arm, path in sorted(self.lam1.items())},
            "lam1full": {arm: str(path) for arm, path in sorted(self.lam1full.items())},
            "noise": str(self.noise) if self.noise else None,
            "delta_stats": {arm: str(path) for arm, path in sorted(self.delta_stats.items())},
            "gates": [str(path) for path in self.gates],
            "vector_norms": str(self.vector_norms) if self.vector_norms else None,
        }


# ------------------------------------------------------------------ loading
LONG_COLUMNS: tuple[str, ...] = A.LONG_COLUMNS + ("arm", "lam", "route", "rank", "at", "graft")
META_COLUMNS: tuple[str, ...] = ("row_id", "arm", "graft", "group", "episode_id", "subtype", "n_target_tokens", "loss", "loss_lam1")


def rekey_cross_terms(record: Mapping[str, Any], grafted_arm: str) -> dict[str, Any]:
    """In a λ = 1 pass grafted with ``grafted_arm``, every other arm's λ = 1
    vector is a cross term (gradient at θ_it + Δ_grafted dotted with that
    arm's Δ). Re-key ``<arm>__lam1[x]_r<r>__all`` → ``<arm>__lam1x_r<r>_at_<grafted>__all``
    (and ``lam1full[x_r<r>]`` → ``lam1fullx_<r>_at_<grafted>``) so the λ = 1
    passes carry distinct vectors and v1's (row, vector) dedupe never drops
    one silently. Already-suffixed kinds pass through."""
    scores: dict[str, Any] = {}
    for vector, value in record["scores"].items():
        dataset, kind, fold = A.parse_vector_name(vector)
        info = parse_kind(kind)
        if info is not None and info.lam == 1 and info.at is None and (info.route == "cross" or dataset != grafted_arm):
            kind = cross_kind_name(info.rank_label, grafted_arm, info.graft)
        scores[f"{dataset}__{kind}__{fold}"] = value
    return {**record, "scores": scores}


def annotate_kinds(long: pd.DataFrame) -> pd.DataFrame:
    """Add ``arm`` (= dataset), ``lam``, ``route``, ``rank`` (NaN for full /
    unknown), ``at`` and ``graft`` columns parsed from ``kind``."""
    frame = long.copy()
    parsed = {kind: parse_kind(kind) for kind in frame["kind"].unique()}
    frame["arm"] = frame["dataset"]
    frame["lam"] = frame["kind"].map(lambda k: float(parsed[k].lam) if parsed[k] else np.nan).astype(float)
    frame["route"] = frame["kind"].map(lambda k: parsed[k].route if parsed[k] else "unknown")
    frame["rank"] = frame["kind"].map(lambda k: float(parsed[k].rank) if parsed[k] and parsed[k].rank is not None else np.nan).astype(float)
    frame["at"] = frame["kind"].map(lambda k: parsed[k].at if parsed[k] and parsed[k].at else None)
    frame["graft"] = frame["kind"].map(lambda k: parsed[k].graft if parsed[k] and parsed[k].graft else None)
    return frame


def expand_vector_norms(norms: Mapping[str, float] | None, vectors: Iterable[str]) -> dict[str, float] | None:
    """Fill ‖Δ‖ for re-keyed / λ = 1 vectors from their λ = 0 or un-suffixed
    names: ``Δ_{arm, r}`` is the same tensor whatever the grafted point, so
    ``X__lam1x_r256_at_G__all`` → ``X__lam1x_r256__all`` → ``X__lam1_r256__all``
    → ``X__lam0_r256__all`` (full-Δ graft kinds likewise, via ``lam1fullx_<r>``
    / ``lam1full`` / ``lam0_full``). Vectors with no candidate stay absent
    (cosine NaN)."""
    if not norms:
        return None
    expanded = dict(norms)
    for vector in vectors:
        if vector in expanded:
            continue
        dataset, kind, fold = A.parse_vector_name(vector)
        info = parse_kind(kind)
        if info is None:
            continue
        label = info.rank_label
        candidates = [f"{dataset}__lam1fullx_{label}__{fold}", f"{dataset}__lam1x_{label}__{fold}", f"{dataset}__lam1_{label}__{fold}", f"{dataset}__lam0_{label}__{fold}"]
        if label == "full":
            candidates.insert(0, f"{dataset}__{LAM1FULL_KIND}__{fold}")
        for candidate in candidates:
            if candidate in norms:
                expanded[vector] = float(norms[candidate])
                break
    return expanded


def load_graft_scores(inputs: GraftInputs) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """λ = 0 pass + every λ = 1 pass (cross terms re-keyed) → one long frame in
    v1's schema plus ``arm/lam/route/rank/at`` and the ``per_sequence_sum`` /
    ``per_token`` (and, given ``vector_norms``, ``cosine``) normalisations;
    ``row_meta`` carries ``loss`` (L(0)) and ``loss_lam1`` (L(1)) per (row,
    grafted arm) for the linearity check."""
    notes: dict[str, Any] = {"passes": [], "n_duplicates_dropped": 0, "unknown_kinds": [], "unknown_groups": [], "normalizations": [], "norm_notes": []}
    frames: list[pd.DataFrame] = []
    meta: list[dict[str, Any]] = []
    records = A.read_jsonl(inputs.lam0)
    frame = A.scores_to_long(records, "lam0")
    frames.append(frame)
    notes["passes"].append({"pass": "lam0", "n_rows": len(records), "n_scores": int(len(frame))})
    for prefix, graft, paths in (("lam1", "lora", inputs.lam1), (LAM1FULL_KIND, "full", inputs.lam1full)):
        for arm, path in sorted(paths.items()):
            records = [rekey_cross_terms(r, arm) for r in A.read_jsonl(path)]
            frame = A.scores_to_long(records, f"{prefix}__{arm}")
            frames.append(frame)
            notes["passes"].append({"pass": f"{prefix}__{arm}", "n_rows": len(records), "n_scores": int(len(frame))})
            for record in records:
                meta.append({
                    "row_id": str(record["row_id"]), "arm": arm, "graft": graft, "group": str(record.get("group", "")),
                    "episode_id": str(record.get("episode_id", "")), "subtype": str(record.get("subtype", "")),
                    "n_target_tokens": record.get("n_target_tokens", np.nan), "loss": record.get("loss", np.nan),
                    "loss_lam1": record.get("loss_lam1", np.nan),
                })
    long = pd.concat(frames, ignore_index=True)
    unknown_groups = sorted(set(long["group"]) - set(CLASSES))
    if unknown_groups:
        warnings.warn(f"unknown row groups {unknown_groups}; kept but excluded from class analyses")
    notes["unknown_groups"] = unknown_groups
    duplicated = long.duplicated(subset=["row_id", "vector"], keep="first")
    notes["n_duplicates_dropped"] = int(duplicated.sum())
    if duplicated.any():
        warnings.warn(f"{int(duplicated.sum())} duplicate (row, vector) scores across passes dropped (first pass kept)")
        long = long.loc[~duplicated].reset_index(drop=True)
    long = annotate_kinds(long)
    notes["unknown_kinds"] = sorted(set(long.loc[long["route"] == "unknown", "kind"]))
    norms = expand_vector_norms(A.load_json_optional(inputs.vector_norms), long["vector"].unique())
    long, notes["normalizations"], notes["norm_notes"] = A.add_normalizations(long, norms)
    row_meta = pd.DataFrame(meta, columns=list(META_COLUMNS))
    for column in ("n_target_tokens", "loss", "loss_lam1"):
        row_meta[column] = pd.to_numeric(row_meta[column], errors="coerce").astype(float)
    return long, row_meta, notes


def load_noise_scores(path: str | Path | None) -> pd.DataFrame:
    """Repeat-scored rows (never deduplicated) → v1 long frame, or empty."""
    if path is None or not Path(path).is_file():
        return pd.DataFrame(columns=list(A.LONG_COLUMNS))
    return A.scores_to_long(A.read_jsonl(path), "noise")


# ------------------------------------------------------ energy + gate files
def _pick(mapping: Mapping[str, Any] | None, *aliases: str, default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    for alias in aliases:
        if alias in mapping and mapping[alias] is not None:
            return mapping[alias]
    return default


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def _rank_key(key: Any) -> int | str | None:
    """``16`` / ``"16"`` / ``"r16"`` / ``"rank_16"`` → 16; ``"full"`` → "full"."""
    if isinstance(key, bool):
        return None
    if isinstance(key, (int, np.integer)):
        return int(key)
    text = str(key).strip().lower()
    if text == "full":
        return "full"
    match = _RANK_KEY_RE.match(text)
    return int(match.group(1)) if match else None


_ENERGY_ALIASES = ("captured_energy", "energy", "captured", "fraction", "value")


def rank_map(obj: Any) -> dict[int | str, float]:
    """{rank: number} from ``{rank_key: f}``, ``{rank_key: {"captured_energy": f}}``
    or ``[{"rank": r, "captured_energy": f}]`` (aliases: energy / captured /
    fraction / value / loss). Unknown keys are ignored."""
    out: dict[int | str, float] = {}
    if isinstance(obj, Mapping):
        for alias in _ENERGY_ALIASES:
            if isinstance(obj.get(alias), (Mapping, list)):
                return rank_map(obj[alias])
        for key, value in obj.items():
            rank = _rank_key(key)
            if rank is None:
                continue
            if isinstance(value, Mapping):
                value = _pick(value, *_ENERGY_ALIASES, "loss")
            number = _number(value)
            if np.isfinite(number):
                out[rank] = number
    elif isinstance(obj, list):
        for item in obj:
            if not isinstance(item, Mapping):
                continue
            rank = _rank_key(_pick(item, "rank", "r"))
            number = _number(_pick(item, *_ENERGY_ALIASES, "loss"))
            if rank is not None and np.isfinite(number):
                out[rank] = number
    return out


ENERGY_COLUMNS: tuple[str, ...] = ("arm", "scope", "rank_label", "rank", "captured_energy")


def load_delta_stats(paths: Mapping[str, str | Path]) -> tuple[pd.DataFrame, list[str]]:
    """``delta_stats__<arm>.json`` → long frame (arm, scope ∈ {pooled, <module type>}, rank, captured_energy)."""
    records: list[dict[str, Any]] = []
    notes: list[str] = []
    for arm, path in sorted(paths.items()):
        payload = A.load_json_optional(path)
        if not isinstance(payload, Mapping):
            notes.append(f"delta_stats for {arm}: unreadable ({path})")
            continue
        pooled = rank_map(_pick(payload, "pooled", "pooled_captured_energy", "captured_energy_pooled", "energy_pooled") or {})
        if not pooled:
            pooled = rank_map(_pick(payload, "captured_energy", "energy") or {})
        if not pooled:
            pooled = rank_map(payload)
        if not pooled:
            notes.append(f"delta_stats for {arm}: no pooled captured-energy-per-rank found ({path.name if isinstance(path, Path) else path})")
        for rank, energy in sorted(pooled.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])):
            records.append({"arm": arm, "scope": "pooled", "rank_label": "full" if rank == "full" else f"r{rank}", "rank": np.nan if rank == "full" else float(rank), "captured_energy": energy})
        per_type = _pick(payload, "per_module_type", "module_types", "by_module_type", "per_type") or {}
        if isinstance(per_type, Mapping):
            for module_type, sub in sorted(per_type.items()):
                for rank, energy in sorted(rank_map(sub).items(), key=lambda kv: (isinstance(kv[0], str), kv[0])):
                    records.append({"arm": arm, "scope": str(module_type), "rank_label": "full" if rank == "full" else f"r{rank}", "rank": np.nan if rank == "full" else float(rank), "captured_energy": energy})
    return pd.DataFrame(records, columns=list(ENERGY_COLUMNS)), notes


def pooled_energy_lookup(energy: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(arm, rank_label) → pooled captured energy; ``full`` = 1.0 for every arm."""
    lookup: dict[tuple[str, str], float] = {}
    if energy is not None and not energy.empty:
        for _, row in energy[energy["scope"] == "pooled"].iterrows():
            lookup[(str(row["arm"]), str(row["rank_label"]))] = float(row["captured_energy"])
    for arm in {arm for arm, _ in lookup} | set(ARMS):
        lookup.setdefault((arm, "full"), 1.0)
    return lookup


def load_gate_payloads(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in paths:
        payload = A.load_json_optional(path)
        if isinstance(payload, list):
            payloads += [item for item in payload if isinstance(item, Mapping)]
        elif isinstance(payload, Mapping):
            payloads.append(dict(payload))
    return payloads


def _per_arm_records(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return {}
    if isinstance(payload.get("arm"), str):
        return {payload["arm"]: dict(payload)}
    for key in ("arms", "per_arm", "rows", "results", "records"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return {str(arm): dict(rec) for arm, rec in value.items() if isinstance(rec, Mapping)}
        if isinstance(value, list):
            found = {str(rec["arm"]): dict(rec) for rec in value if isinstance(rec, Mapping) and "arm" in rec}
            if found:
                return found
    direct = {arm: dict(payload[arm]) for arm in payload if isinstance(payload.get(arm), Mapping) and str(arm) in ARMS}
    return direct


_GATE_KEYS = {"G1": ("g1", "G1", "gate1", "gate_g1", "reconstruction"), "G2": ("g2", "G2", "gate2", "gate_g2", "transfer")}
_GATE_FIELDS = {
    "G1": ("loss_pt", "L_pt", "loss_mid", "L_mid", "recovered_fraction", "recovered", "loss_pt_plus_delta"),
    "G2": ("loss_it", "L_it", "loss_it_plus_delta", "L_it_plus_delta"),
}


def gate_sections(payloads: Sequence[Mapping[str, Any]], gate: str) -> dict[str, dict[str, Any]]:
    """Per-arm records for ``gate`` ('G1' / 'G2') from any payload: nested under
    g1/G1/reconstruction (g2/G2/transfer), tagged ``"gate": "G1"``, or
    recognised by their fields (loss_pt/loss_mid vs loss_it)."""
    found: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        candidates: list[Mapping[str, Any]] = [payload]
        for key in _GATE_KEYS[gate]:
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                candidates.append(nested)
            elif isinstance(nested, list):
                candidates.append({"rows": nested})
        for candidate in candidates:
            tag = str(candidate.get("gate", "")).upper().replace(" ", "")
            if tag and tag != gate and candidate is payload and not any(isinstance(payload.get(k), (Mapping, list)) for k in _GATE_KEYS[gate]):
                continue
            for arm, record in _per_arm_records(candidate).items():
                record_tag = str(record.get("gate", "")).upper()
                if record_tag and record_tag != gate:
                    continue
                if any(field_name in record for field_name in _GATE_FIELDS[gate]):
                    merged = found.setdefault(arm, {})
                    merged.update(record)
                    for key in ("rank", "r_star", "primary_rank"):
                        if key in candidate and key not in merged:
                            merged[key] = candidate[key]
    return found


# ------------------------------------------------------- (1) net of control
def net_of_control(long: pd.DataFrame, control_arm: str = CONTROL_ARM) -> pd.DataFrame:
    """Same-row, same-kind difference ``score(arm) − score(control)`` for every
    non-control arm (v1 long schema, dataset = arm, ``baseline`` =
    'control_own'); normalisations recomputed on the net score."""
    base = long[long["fold"] == FOLD]
    control = base.loc[base["arm"] == control_arm, ["row_id", "kind", "score"]].rename(columns={"score": "control_score"})
    arms = base[base["arm"] != control_arm]
    merged = arms.merge(control, on=["row_id", "kind"], how="inner")
    merged["score"] = merged["score"].astype(float) - merged["control_score"].astype(float)
    merged = merged.drop(columns=["control_score"])
    merged["baseline"] = "control_own"
    net, _, _ = A.add_normalizations(merged, None)
    return net.reset_index(drop=True)


def net_of_control_cross(long: pd.DataFrame, control_arm: str = CONTROL_ARM) -> pd.DataFrame:
    """λ = 1 variant at the *same grafted point*: the grafted arm's own λ = 1
    term minus the control Δ's cross term evaluated at θ_it + Δ_arm
    (``control__lam1x_r<r>_at_<arm>``; for the full-Δ graft
    ``control__lam1fullx_r1024_at_<arm>``). ``baseline`` = 'control_cross'."""
    base = long[(long["fold"] == FOLD) & (long["lam"] == 1)]
    own = base[base["route"].isin(["lora", "full"]) & (base["arm"] != control_arm)].copy()
    cross = base.loc[(base["route"] == "cross") & (base["arm"] == control_arm), ["row_id", "rank", "at", "graft", "score"]]
    cross = cross.rename(columns={"score": "control_score", "at": "arm"})
    if own.empty or cross.empty:
        return pd.DataFrame(columns=list(LONG_COLUMNS) + ["baseline", "per_sequence_sum", "per_token"])
    # Pair own and cross terms on the grafted point: the r* LoRA graft's terms share the rank; the
    # full-Δ graft's own term has no rank (its cross terms use Δ_{other, r1024}), so pair on graft alone.
    own["pair_rank"] = np.where(own["graft"] == "lora", own["rank"], -1.0)
    cross["pair_rank"] = np.where(cross["graft"] == "lora", cross["rank"], -1.0)
    merged = own.merge(cross.drop(columns=["rank"]), on=["row_id", "arm", "graft", "pair_rank"], how="inner").drop(columns=["pair_rank"])
    merged["score"] = merged["score"].astype(float) - merged["control_score"].astype(float)
    merged = merged.drop(columns=["control_score"])
    merged["baseline"] = "control_cross"
    net, _, _ = A.add_normalizations(merged, None)
    return net.reset_index(drop=True)


def paired_contrasts(long: pd.DataFrame, n_boot: int = 2000, seed: int = 0, norms: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
    """v1's paired per-episode contrasts for every normalisation present →
    {contrasts, summary, by_subtype, unmatched}."""
    norms = list(norms) if norms else [n for n in A.NORMALIZATIONS if n in long.columns]
    matched_frames, unmatched_frames = [], []
    for norm in norms:
        if long.empty:
            continue
        matched, unmatched = A.pair_contrasts(long, norm)
        matched_frames.append(matched)
        unmatched_frames.append(unmatched.assign(norm=norm))
    contrasts = pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame()
    unmatched = pd.concat(unmatched_frames, ignore_index=True) if unmatched_frames else pd.DataFrame()
    summary = A.summarize_contrasts(contrasts, n_boot=n_boot, seed=seed)
    by_subtype = A.summarize_contrasts(contrasts, n_boot=n_boot, seed=seed, by_subtype=True)
    return {"contrasts": contrasts, "summary": summary, "by_subtype": by_subtype, "unmatched": unmatched}


def add_verdicts(summary: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """``arm``, ``family``, ``baseline``, ``expected_sign``, ``verdict`` columns
    on a v1 contrast summary (dataset = arm)."""
    frame = summary.copy()
    if frame.empty:
        for column in ("arm", "family", "baseline", "expected_sign", "verdict"):
            frame[column] = pd.Series(dtype=object)
        return frame
    frame.insert(0, "arm", frame["dataset"].astype(str))
    frame.insert(1, "family", frame["arm"].map(arm_family))
    frame.insert(2, "baseline", baseline)
    frame["expected_sign"] = [expected_sign(a, c) for a, c in zip(frame["arm"], frame["contrast"])]
    frame["verdict"] = [verdict(a, c, m, lo, hi, baseline) for a, c, m, lo, hi in zip(frame["arm"], frame["contrast"], frame["mean"], frame["ci_low"], frame["ci_high"])]
    return frame


@dataclass(frozen=True)
class LambdaLevel:
    """One λ readout: λ = 0, λ = 1 at the r* LoRA graft, or λ = 1 at the exact
    full-Δ graft. ``kind`` is the score kind read out; ``lam0_kind`` the λ = 0
    reference used by the curvature / linearity comparisons (None at λ = 0)."""

    key: str  # "lam0" | "lam1" | "lam1full"  (manifest keys)
    label: str  # "λ = 0" | "λ = 1 (r*)" | "λ = 1 (full Δ)"
    lam: str  # "0" | "1"
    graft: str  # "" | "lora" | "full"
    kind: str
    lam0_kind: str | None = None

    def plot_suffix(self, arm: str) -> str:
        """``<arm>`` for the r* variant (filenames stay as before), ``<arm>__lam1full`` for the full-Δ variant."""
        return arm if self.graft != "full" else f"{arm}__{LAM1FULL_KIND}"


HEADLINE_COLUMNS: tuple[str, ...] = (
    "arm", "family", "lambda", "label", "graft", "kind", "baseline", "contrast", "n", "mean", "ci_low", "ci_high",
    "median", "frac_positive", "sign_p", "expected_sign", "verdict",
)


def headline_table(summaries: Mapping[str, pd.DataFrame], levels: Sequence[LambdaLevel], norm: str = PRIMARY_NORM) -> pd.DataFrame:
    """Per arm × λ level × baseline × contrast: the pre-registered readouts.
    ``summaries`` maps baseline ('raw' / 'net_of_control' /
    'net_of_control_cross') → contrast summary (v1 schema, dataset = arm)."""
    records: list[dict[str, Any]] = []
    for level in levels:
        for baseline in BASELINES:
            summary = summaries.get(baseline)
            if summary is None or summary.empty:
                continue
            subset = summary[(summary["kind"] == level.kind) & (summary["norm"] == norm) & (summary["fold"] == FOLD)]
            for arm in order_arms(subset["dataset"]):
                for contrast in CONTRASTS:
                    row = subset[(subset["dataset"] == arm) & (subset["contrast"] == contrast)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    records.append({
                        "arm": arm, "family": arm_family(arm), "lambda": level.lam, "label": level.label, "graft": level.graft, "kind": level.kind,
                        "baseline": baseline, "contrast": contrast,
                        "n": int(row["n"]), "mean": float(row["mean"]), "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"]),
                        "median": float(row["median"]), "frac_positive": float(row["frac_positive"]), "sign_p": float(row["sign_p"]),
                        "expected_sign": expected_sign(arm, contrast),
                        "verdict": verdict(arm, contrast, float(row["mean"]), float(row["ci_low"]), float(row["ci_high"]), baseline),
                    })
    return pd.DataFrame(records, columns=list(HEADLINE_COLUMNS))


# ---------------------------------------------------------------- statistics
def _finite_pairs(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def ols(x, y) -> tuple[float, float]:
    """(slope, intercept) of y on x; NaNs dropped; degenerate → (nan, nan)."""
    x, y = _finite_pairs(x, y)
    if x.size < 2:
        return (float("nan"), float("nan"))
    variance = float(np.var(x))
    if variance <= 0:
        return (float("nan"), float("nan"))
    slope = float(np.cov(x, y, ddof=0)[0, 1] / variance)
    return (slope, float(y.mean() - slope * x.mean()))


def sign_agreement(x, y) -> float:
    """Fraction of rows where sign(x) == sign(y), over rows with both non-zero."""
    x, y = _finite_pairs(x, y)
    keep = (x != 0) & (y != 0)
    if not keep.any():
        return float("nan")
    return float((np.sign(x[keep]) == np.sign(y[keep])).mean())


def _pair_stats(x, y) -> dict[str, float]:
    x, y = _finite_pairs(x, y)
    slope, intercept = ols(x, y)
    return {
        "n": int(x.size), "spearman": A.spearman(x, y), "pearson": A.pearson(x, y),
        "ols_slope": slope, "ols_intercept": intercept, "sign_agreement": sign_agreement(x, y),
        "mean_x": float(x.mean()) if x.size else float("nan"), "mean_y": float(y.mean()) if y.size else float("nan"),
    }


# -------------------------------------------------- (2) λ = 0 vs λ = 1 curve
CURVATURE_COLUMNS: tuple[str, ...] = (
    "arm", "label", "graft", "class", "kind_lam0", "kind_lam1", "norm", "n", "spearman", "pearson", "ols_slope", "ols_intercept",
    "sign_agreement", "mean_lam0", "mean_lam1", "ratio_of_means",
)
CURVATURE_CONTRAST_COLUMNS: tuple[str, ...] = (
    "arm", "label", "graft", "contrast", "kind_lam0", "kind_lam1", "norm", "n_episodes", "mean_lam0", "mean_lam1", "ratio_of_means",
    "spearman_episodes", "sign_agreement_episodes", "ols_slope",
)


def score_wide(long: pd.DataFrame, arm: str, kinds: Sequence[str], norm: str = PRIMARY_NORM) -> pd.DataFrame:
    """rows × kinds matrix of ``norm`` for one arm (index row_id, plus group /
    episode_id / subtype columns)."""
    sub = long[(long["arm"] == arm) & (long["fold"] == FOLD) & long["kind"].isin(list(kinds))]
    if sub.empty:
        return pd.DataFrame(columns=["group", "episode_id", "subtype", *kinds])
    wide = sub.pivot_table(index="row_id", columns="kind", values=norm, aggfunc="first")
    meta = sub.drop_duplicates("row_id").set_index("row_id")[["group", "episode_id", "subtype"]]
    wide = meta.join(wide, how="left")
    for kind in kinds:
        if kind not in wide.columns:
            wide[kind] = np.nan
    return wide


def lambda_curvature(long: pd.DataFrame, contrasts: pd.DataFrame, lam0_kind: str, lam1_kind: str | None, norm: str = PRIMARY_NORM, graft: str = "lora", label: str = "λ = 1 (r*)") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per arm × class: Spearman / OLS slope of the λ = 1 score on the λ = 0
    score (slope < 1 = the graft saturates; sign_agreement < 1 = rows where
    g(1) flipped sign), and per arm × contrast the attenuation of the paired
    contrast (mean at λ = 1 / mean at λ = 0). ``graft`` / ``label`` tag the
    λ = 1 variant (r* LoRA graft or full-Δ graft)."""
    per_class = pd.DataFrame(columns=list(CURVATURE_COLUMNS))
    per_contrast = pd.DataFrame(columns=list(CURVATURE_CONTRAST_COLUMNS))
    if not lam1_kind:
        return per_class, per_contrast
    class_records, contrast_records = [], []
    for arm in order_arms(long.loc[long["kind"] == lam1_kind, "arm"]):
        wide = score_wide(long, arm, [lam0_kind, lam1_kind], norm).dropna(subset=[lam0_kind, lam1_kind])
        groups = [("all", wide)] + [(cls, wide[wide["group"] == cls]) for cls in A.order_classes(wide["group"]) if cls in CLASSES]
        for cls, sub in groups:
            if sub.empty:
                continue
            stats = _pair_stats(sub[lam0_kind], sub[lam1_kind])
            class_records.append({
                "arm": arm, "label": label, "graft": graft, "class": cls, "kind_lam0": lam0_kind, "kind_lam1": lam1_kind, "norm": norm, "n": stats["n"],
                "spearman": stats["spearman"], "pearson": stats["pearson"], "ols_slope": stats["ols_slope"], "ols_intercept": stats["ols_intercept"],
                "sign_agreement": stats["sign_agreement"], "mean_lam0": stats["mean_x"], "mean_lam1": stats["mean_y"],
                "ratio_of_means": stats["mean_y"] / stats["mean_x"] if stats["mean_x"] != 0 and np.isfinite(stats["mean_x"]) else float("nan"),
            })
        if contrasts is not None and not contrasts.empty:
            sub = contrasts[(contrasts["dataset"] == arm) & (contrasts["norm"] == norm) & (contrasts["fold"] == FOLD) & contrasts["kind"].isin([lam0_kind, lam1_kind])]
            for contrast in CONTRASTS:
                pivot = sub[sub["contrast"] == contrast].pivot_table(index="episode_id", columns="kind", values="value", aggfunc="first")
                if lam0_kind not in pivot.columns or lam1_kind not in pivot.columns:
                    continue
                pivot = pivot.dropna(subset=[lam0_kind, lam1_kind])
                if pivot.empty:
                    continue
                stats = _pair_stats(pivot[lam0_kind], pivot[lam1_kind])
                contrast_records.append({
                    "arm": arm, "label": label, "graft": graft, "contrast": contrast, "kind_lam0": lam0_kind, "kind_lam1": lam1_kind, "norm": norm, "n_episodes": stats["n"],
                    "mean_lam0": stats["mean_x"], "mean_lam1": stats["mean_y"],
                    "ratio_of_means": stats["mean_y"] / stats["mean_x"] if stats["mean_x"] != 0 and np.isfinite(stats["mean_x"]) else float("nan"),
                    "spearman_episodes": stats["spearman"], "sign_agreement_episodes": stats["sign_agreement"], "ols_slope": stats["ols_slope"],
                })
    if class_records:
        per_class = pd.DataFrame(class_records, columns=list(CURVATURE_COLUMNS))
    if contrast_records:
        per_contrast = pd.DataFrame(contrast_records, columns=list(CURVATURE_CONTRAST_COLUMNS))
    return per_class, per_contrast


# ------------------------------------------------------------ (3) linearity
LINEARITY_COLUMNS: tuple[str, ...] = (
    "arm", "label", "graft", "class", "predictor", "kind_lam0", "kind_lam1", "n", "spearman", "pearson", "ols_slope", "ols_intercept",
    "sign_agreement", "rmse", "mean_delta_loss", "frac_loss_lowered", "mean_predictor",
)
LINEARITY_ROW_COLUMNS: tuple[str, ...] = ("arm", "graft", "row_id", "group", "episode_id", "subtype", "loss", "loss_lam1", "delta_loss", "g0", "g1", "g_mid")


def linearity(long: pd.DataFrame, row_meta: pd.DataFrame, lam0_kind: str, lam1_kind: str | None, norm: str = PRIMARY_NORM, graft: str = "lora", label: str = "λ = 1 (r*)") -> tuple[pd.DataFrame, pd.DataFrame]:
    """``ΔL = L(1) − L(0)`` per row (λ = 1 file of the ``graft`` variant;
    negative = the graft lowered the loss) against the first-order prediction
    ``g(0) = −score_lam0`` (and the trapezoid ``(g(0) + g(1)) / 2`` when λ = 1
    scores exist): Spearman, sign agreement, OLS slope (≈ 1 = linear, < 1 =
    saturating), RMSE. Returns (table per arm × class × predictor, per-row
    frame for plotting)."""
    table = pd.DataFrame(columns=list(LINEARITY_COLUMNS))
    rows_out = pd.DataFrame(columns=list(LINEARITY_ROW_COLUMNS))
    if row_meta is None or row_meta.empty:
        return table, rows_out
    if "graft" in row_meta.columns:
        row_meta = row_meta[row_meta["graft"] == graft]
    records, row_frames = [], []
    for arm in order_arms(row_meta["arm"]):
        meta = row_meta[row_meta["arm"] == arm].drop_duplicates("row_id").set_index("row_id")
        kinds = [lam0_kind] + ([lam1_kind] if lam1_kind else [])
        wide = score_wide(long, arm, kinds, norm)
        # rows without the λ = 0 reference score (e.g. outside the full-Δ subset) carry no linearity content
        frame = meta[["group", "episode_id", "subtype", "loss", "loss_lam1"]].join(wide[kinds], how="inner").dropna(subset=[lam0_kind])
        frame["delta_loss"] = frame["loss_lam1"].astype(float) - frame["loss"].astype(float)
        frame["g0"] = -frame[lam0_kind].astype(float)
        frame["g1"] = -frame[lam1_kind].astype(float) if lam1_kind else np.nan
        frame["g_mid"] = 0.5 * (frame["g0"] + frame["g1"]) if lam1_kind else np.nan
        frame = frame.reset_index().rename(columns={"index": "row_id"})
        frame.insert(0, "arm", arm)
        frame.insert(1, "graft", graft)
        row_frames.append(frame[list(LINEARITY_ROW_COLUMNS)])
        predictors = [("g0", "g0")] + ([("g_mid", "g_mid")] if lam1_kind else [])
        groups = [("all", frame)] + [(cls, frame[frame["group"] == cls]) for cls in A.order_classes(frame["group"]) if cls in CLASSES]
        for predictor_name, column in predictors:
            for cls, sub in groups:
                x, y = _finite_pairs(sub[column], sub["delta_loss"])
                if x.size == 0:
                    continue
                stats = _pair_stats(x, y)
                records.append({
                    "arm": arm, "label": label, "graft": graft, "class": cls, "predictor": predictor_name, "kind_lam0": lam0_kind, "kind_lam1": lam1_kind or "", "n": stats["n"],
                    "spearman": stats["spearman"], "pearson": stats["pearson"], "ols_slope": stats["ols_slope"], "ols_intercept": stats["ols_intercept"],
                    "sign_agreement": stats["sign_agreement"], "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
                    "mean_delta_loss": float(y.mean()), "frac_loss_lowered": float((y < 0).mean()), "mean_predictor": float(x.mean()),
                })
    if records:
        table = pd.DataFrame(records, columns=list(LINEARITY_COLUMNS))
    if row_frames:
        rows_out = pd.concat(row_frames, ignore_index=True)
    return table, rows_out


# ------------------------------------- (3b) λ = 1: r* LoRA graft vs full Δ
LAM1_COMPARE_COLUMNS: tuple[str, ...] = (
    "arm", "class", "kind_lora", "kind_full", "norm", "n", "mean_lora", "mean_full", "mean_diff_full_minus_lora", "diff_ci_low", "diff_ci_high",
    "ratio_of_means", "spearman", "pearson", "ols_slope_full_on_lora", "sign_agreement",
    "n_loss", "mean_loss_lam1_lora", "mean_loss_lam1_full", "mean_loss_diff_full_minus_lora", "loss_diff_ci_low", "loss_diff_ci_high",
    "frac_full_lower_loss", "spearman_delta_loss",
)


def lam1_lora_vs_full(long: pd.DataFrame, row_meta: pd.DataFrame, lam1_kind: str | None, lam1full_kind: str = LAM1FULL_KIND, norm: str = PRIMARY_NORM, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Direct comparison of the two λ = 1 grafts on shared rows, per arm ×
    class: paired difference (full − r*) of the −g(1) scores with bootstrap
    CI, Spearman / OLS slope (full on r*), sign agreement, ratio of means;
    and ``loss_lam1`` under both grafts (paired difference, fraction of rows
    where the full-Δ graft leaves the lower loss, Spearman of the two ΔL)."""
    table = pd.DataFrame(columns=list(LAM1_COMPARE_COLUMNS))
    if not lam1_kind or lam1full_kind not in set(long["kind"]):
        return table
    records: list[dict[str, Any]] = []
    index = 0
    for arm in order_arms(long.loc[long["kind"] == lam1full_kind, "arm"]):
        wide = score_wide(long, arm, [lam1_kind, lam1full_kind], norm).dropna(subset=[lam1_kind, lam1full_kind])
        if wide.empty:
            continue
        meta = row_meta[row_meta["arm"] == arm] if row_meta is not None and not row_meta.empty else pd.DataFrame(columns=list(META_COLUMNS))
        losses = pd.DataFrame({
            "loss": meta.drop_duplicates("row_id").set_index("row_id")["loss"],
            "lora": meta[meta["graft"] == "lora"].drop_duplicates("row_id").set_index("row_id")["loss_lam1"],
            "full": meta[meta["graft"] == "full"].drop_duplicates("row_id").set_index("row_id")["loss_lam1"],
        }).dropna()
        groups = [("all", wide)] + [(cls, wide[wide["group"] == cls]) for cls in A.order_classes(wide["group"]) if cls in CLASSES]
        for cls, sub in groups:
            x, y = _finite_pairs(sub[lam1_kind], sub[lam1full_kind])
            if x.size == 0:
                continue
            stats = _pair_stats(x, y)
            diff = y - x
            low, high = A.bootstrap_mean_ci(diff, n_boot=n_boot, seed=seed + index)
            index += 1
            loss_sub = losses.loc[losses.index.intersection(sub.index)]
            loss_diff = (loss_sub["full"] - loss_sub["lora"]).to_numpy(dtype=float)
            loss_low, loss_high = A.bootstrap_mean_ci(loss_diff, n_boot=n_boot, seed=seed + index)
            index += 1
            records.append({
                "arm": arm, "class": cls, "kind_lora": lam1_kind, "kind_full": lam1full_kind, "norm": norm, "n": stats["n"],
                "mean_lora": stats["mean_x"], "mean_full": stats["mean_y"], "mean_diff_full_minus_lora": float(diff.mean()), "diff_ci_low": low, "diff_ci_high": high,
                "ratio_of_means": stats["mean_y"] / stats["mean_x"] if stats["mean_x"] != 0 and np.isfinite(stats["mean_x"]) else float("nan"),
                "spearman": stats["spearman"], "pearson": stats["pearson"], "ols_slope_full_on_lora": stats["ols_slope"], "sign_agreement": stats["sign_agreement"],
                "n_loss": int(len(loss_sub)),
                "mean_loss_lam1_lora": float(loss_sub["lora"].mean()) if len(loss_sub) else float("nan"),
                "mean_loss_lam1_full": float(loss_sub["full"].mean()) if len(loss_sub) else float("nan"),
                "mean_loss_diff_full_minus_lora": float(loss_diff.mean()) if loss_diff.size else float("nan"),
                "loss_diff_ci_low": loss_low, "loss_diff_ci_high": loss_high,
                "frac_full_lower_loss": float((loss_diff < 0).mean()) if loss_diff.size else float("nan"),
                "spearman_delta_loss": A.spearman(loss_sub["lora"] - loss_sub["loss"], loss_sub["full"] - loss_sub["loss"]) if len(loss_sub) else float("nan"),
            })
    return pd.DataFrame(records, columns=list(LAM1_COMPARE_COLUMNS))


# ---------------------------------------------------------- (4) rank ladder
LADDER_COLUMNS: tuple[str, ...] = (
    "baseline", "arm", "contrast", "subset", "kind", "rank_label", "rank_order", "n", "mean", "ci_low", "ci_high",
    "frac_positive", "sign_p", "captured_energy", "fraction_of_full",
)


def rank_ladder(long: pd.DataFrame, net_long: pd.DataFrame | None, energy: pd.DataFrame | None, n_boot: int = 2000, seed: int = 0, norm: str = PRIMARY_NORM) -> pd.DataFrame:
    """Paired contrasts per arm for the λ = 0 kinds r16 → r64 → r256 → r1024 →
    full, raw and net of control, on (a) every row scored for that kind and
    (b) the subset of rows that carry the full-Δ reference (apples-to-apples:
    ``fraction_of_full`` = mean at rank / mean at full there). Joined with the
    pooled captured energy per rank from ``delta_stats``."""
    lookup = pooled_energy_lookup(energy)
    records: list[pd.DataFrame] = []
    for baseline, frame in (("raw", long), ("net_of_control", net_long)):
        if frame is None or frame.empty:
            continue
        ladder = frame[(frame["lam"] == 0) & frame["route"].isin(["lora", "full"]) & (frame["fold"] == FOLD)]
        if ladder.empty:
            continue
        has_full = ladder.loc[(ladder["route"] == "full") & ladder[norm].notna(), ["arm", "row_id"]].drop_duplicates()
        subsets = [("all_rows", ladder)]
        if not has_full.empty:
            subsets.append(("full_subset", ladder.merge(has_full, on=["arm", "row_id"], how="inner")))
        for subset_name, sub in subsets:
            matched, _ = A.pair_contrasts(sub, norm)
            summary = A.summarize_contrasts(matched, n_boot=n_boot, seed=seed)
            if summary.empty:
                continue
            summary = summary.rename(columns={"dataset": "arm"})
            summary["baseline"] = baseline
            summary["subset"] = subset_name
            records.append(summary)
    if not records:
        return pd.DataFrame(columns=list(LADDER_COLUMNS))
    table = pd.concat(records, ignore_index=True)
    infos = {kind: parse_kind(kind) for kind in table["kind"].unique()}
    table["rank_label"] = table["kind"].map(lambda k: infos[k].rank_label)
    table["rank_order"] = table["kind"].map(lambda k: infos[k].rank_order).astype(float)
    table["captured_energy"] = [lookup.get((arm, label), np.nan) for arm, label in zip(table["arm"], table["rank_label"])]
    full_means = table[(table["rank_label"] == "full")].set_index(["baseline", "arm", "contrast", "subset"])["mean"]
    fractions = []
    for _, row in table.iterrows():
        key = (row["baseline"], row["arm"], row["contrast"], row["subset"])
        full = full_means.get(key, np.nan)
        fractions.append(float(row["mean"] / full) if row["subset"] == "full_subset" and np.isfinite(full) and full != 0 else np.nan)
    table["fraction_of_full"] = fractions
    table = table.sort_values(["baseline", "arm", "contrast", "subset", "rank_order"], kind="stable").reset_index(drop=True)
    return table[list(LADDER_COLUMNS)]


# ------------------------------------------------------------------ (5) gates
GATE_COLUMNS: tuple[str, ...] = ("gate", "arm", "metric", "value", "threshold", "verdict", "detail")


def _gate_row(gate: str, arm: str, metric: str, value: float, threshold: str, verdict_text: str, detail: str = "") -> dict[str, Any]:
    return {"gate": gate, "arm": arm, "metric": metric, "value": value, "threshold": threshold, "verdict": verdict_text, "detail": detail}


def gate_g1(sections: Mapping[str, Mapping[str, Any]], primary_rank: int) -> list[dict[str, Any]]:
    """G1 reconstruction at pt: recovered fraction of L(pt) − L(mid) at the
    primary rank (≥ 90 %) and the full-Δ check L(pt + Δ_full) ≈ L(mid)."""
    rows: list[dict[str, Any]] = []
    if not sections:
        return [_gate_row("G1", "all", "recovered_fraction", float("nan"), f"≥ {G1_RECOVERED_MIN:.0%} at r={primary_rank}", "NOT RUN", "no gates__*.json with loss_pt / loss_mid / recovered_fraction")]
    for arm, record in sorted(sections.items()):
        loss_pt = _number(_pick(record, "loss_pt", "L_pt", "pt"))
        loss_mid = _number(_pick(record, "loss_mid", "L_mid", "mid"))
        plus = rank_map(_pick(record, "loss_pt_plus_delta", "L_pt_plus_delta", "pt_plus_delta", "reconstructed") or {})
        recovered = rank_map(_pick(record, "recovered_fraction", "recovered", "recovery") or {})
        gap = loss_pt - loss_mid
        if not recovered and plus and np.isfinite(gap) and gap != 0:
            recovered = {rank: (loss_pt - loss) / gap for rank, loss in plus.items() if rank != "full"}
        value = recovered.get(primary_rank, float("nan"))
        if np.isfinite(value):
            verdict_text = "PASS" if value >= G1_RECOVERED_MIN - 1e-9 else "FAIL"  # tolerance: fractions derived from losses sit on the boundary in FP
        else:
            verdict_text = "NOT RUN"
        ladder = ", ".join(f"r{r}={f:.3f}" for r, f in sorted((k, v) for k, v in recovered.items() if k != "full"))
        rows.append(_gate_row("G1", arm, f"recovered_fraction@r{primary_rank}", value, f"≥ {G1_RECOVERED_MIN:.0%}", verdict_text, f"L(pt)={loss_pt:.4f}, L(mid)={loss_mid:.4f}; ladder {ladder}" if np.isfinite(loss_pt) else ladder))
        full_loss = _number(_pick(record, "loss_pt_plus_full", "L_pt_plus_full", default=plus.get("full")))
        if np.isfinite(full_loss) and np.isfinite(loss_mid) and loss_mid != 0:
            rel = abs(full_loss - loss_mid) / abs(loss_mid)
            rows.append(_gate_row("G1", arm, "full_delta_rel_err", rel, f"≤ {G1_FULL_REL_TOL:.0%} (bf16 noise)", "PASS" if rel <= G1_FULL_REL_TOL else "FAIL", f"L(pt+Δ_full)={full_loss:.4f} vs L(mid)={loss_mid:.4f}" + ("" if rel <= G1_FULL_REL_TOL else " — pt/mid pair mismatched: STOP")))
        else:
            rows.append(_gate_row("G1", arm, "full_delta_rel_err", float("nan"), f"≤ {G1_FULL_REL_TOL:.0%} (bf16 noise)", "NOT RUN", "no loss_pt_plus_delta['full'] / loss_pt_plus_full"))
    return rows


def gate_g2(sections: Mapping[str, Mapping[str, Any]], r_star: int | None) -> list[dict[str, Any]]:
    """G2 transfer at it: L(it + Δ_r*) < L(it) on the arm's own directional
    docs for charter and coin; the control arm is reported as INFO."""
    rows: list[dict[str, Any]] = []
    if not sections:
        return [_gate_row("G2", "all", "loss_delta_at_it", float("nan"), "< 0 (charter, coin)", "NOT RUN", "no gates__*.json with loss_it / loss_it_plus_delta")]
    for arm, record in sorted(sections.items()):
        loss_it = _number(_pick(record, "loss_it", "L_it", "it"))
        plus = _pick(record, "loss_it_plus_delta", "L_it_plus_delta", "it_plus_delta")
        if isinstance(plus, Mapping):
            mapped = rank_map(plus)
            plus_value = mapped.get(r_star, next(iter(mapped.values()), float("nan"))) if mapped else float("nan")
        else:
            plus_value = _number(plus)
        delta = plus_value - loss_it
        rank = _pick(record, "rank", "r_star", "primary_rank", default=r_star)
        if not np.isfinite(delta):
            verdict_text = "NOT RUN"
        elif arm_family(arm) in ("charter", "coin"):
            verdict_text = "PASS" if delta < 0 else "FAIL"
        else:
            verdict_text = "INFO"
        detail = f"L(it)={loss_it:.4f}, L(it+Δ_r{rank})={plus_value:.4f}" if np.isfinite(loss_it) else "missing numbers"
        if verdict_text == "FAIL":
            detail += " — graft does not teach -it the corpus: flag every -it readout for this arm"
        rows.append(_gate_row("G2", arm, "loss_delta_at_it", delta, "< 0", verdict_text, detail))
    return rows


G3_COLUMNS: tuple[str, ...] = ("arm", "kind_lora", "kind_full", "n", "spearman", "pearson", "ols_slope", "verdict")


def gate_g3(long: pd.DataFrame, norm: str = PRIMARY_NORM) -> pd.DataFrame:
    """G3 exactness: LoRA-route g(0) at the largest rank vs the full-Δ g(0) on
    the shared rows — Spearman and OLS slope (full on LoRA) per arm."""
    records = []
    lora = long[(long["lam"] == 0) & (long["route"] == "lora") & (long["fold"] == FOLD)]
    for arm in order_arms(long.loc[long["lam"] == 0, "arm"]):
        full = score_wide(long, arm, ["lam0_full"], norm)
        if full.empty or full["lam0_full"].notna().sum() == 0:
            records.append({"arm": arm, "kind_lora": "", "kind_full": "lam0_full", "n": 0, "spearman": np.nan, "pearson": np.nan, "ols_slope": np.nan, "verdict": "NOT RUN"})
            continue
        ranks = sorted(lora.loc[lora["arm"] == arm, "rank"].dropna().unique())
        if not ranks:
            records.append({"arm": arm, "kind_lora": "", "kind_full": "lam0_full", "n": 0, "spearman": np.nan, "pearson": np.nan, "ols_slope": np.nan, "verdict": "NOT RUN"})
            continue
        kind_lora = lam0_kind_name(int(ranks[-1]))
        wide = score_wide(long, arm, [kind_lora, "lam0_full"], norm).dropna(subset=[kind_lora, "lam0_full"])
        stats = _pair_stats(wide[kind_lora], wide["lam0_full"])
        rho = stats["spearman"]
        verdict_text = "NOT RUN" if stats["n"] < 3 or not np.isfinite(rho) else ("PASS" if rho >= G3_SPEARMAN_MIN else "FAIL")
        records.append({"arm": arm, "kind_lora": kind_lora, "kind_full": "lam0_full", "n": stats["n"], "spearman": rho, "pearson": stats["pearson"], "ols_slope": stats["ols_slope"], "verdict": verdict_text})
    return pd.DataFrame(records, columns=list(G3_COLUMNS))


def gate_g4(noise_long: pd.DataFrame, main_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """G4 noise floor via v1's ``noise_floor``: per kind median / p90 relative
    spread of repeat-scored rows; PASS = median ≤ 2 % (SPEC), FLAG on p90."""
    if noise_long is None or noise_long.empty:
        per_score, summary = A.noise_floor(pd.DataFrame(), main_long)
        return per_score, summary.assign(verdict=pd.Series(dtype=object))
    main = main_long[["pass", "row_id", "group", "episode_id", "subtype", "n_target_tokens", "loss", "grad_norm", "vector", "dataset", "kind", "fold", "score"]]
    per_score, summary = A.noise_floor(noise_long, main)
    if summary.empty:
        return per_score, summary.assign(verdict=pd.Series(dtype=object))
    verdicts = []
    for _, row in summary.iterrows():
        median, p90 = float(row["median_rel_spread"]), float(row["p90_rel_spread"])
        if not np.isfinite(median):
            verdicts.append("NOT RUN")
        elif median > G4_MEDIAN_REL_MAX:
            verdicts.append("FAIL")
        else:
            verdicts.append("PASS" if not (np.isfinite(p90) and p90 > G4_P90_REL_MAX) else "PASS (p90 FLAG)")
    summary = summary.copy()
    summary["verdict"] = verdicts
    return per_score, summary


def rank_capture_gate(ladder: pd.DataFrame, primary_rank: int) -> list[dict[str, Any]]:
    """SPEC §7: the raw coin−charter contrast at the primary rank keeps ≥ 70 %
    of the full-Δ contrast (full subset), for the charter and coin arms."""
    rows: list[dict[str, Any]] = []
    kind = lam0_kind_name(primary_rank)
    if ladder is None or ladder.empty or "full_subset" not in set(ladder["subset"]):
        return [_gate_row("RANK", "all", f"fraction_of_full@{kind}", float("nan"), f"≥ {RANK_CAPTURE_MIN:.0%}", "NOT RUN", "no lam0_full scores — full-Δ subset absent")]
    sub = ladder[(ladder["baseline"] == "raw") & (ladder["subset"] == "full_subset") & (ladder["contrast"] == "coin_minus_charter") & (ladder["kind"] == kind)]
    for arm in order_arms(sub["arm"]):
        row = sub[sub["arm"] == arm].iloc[0]
        fraction = float(row["fraction_of_full"])
        full = ladder[(ladder["baseline"] == "raw") & (ladder["subset"] == "full_subset") & (ladder["contrast"] == "coin_minus_charter") & (ladder["arm"] == arm) & (ladder["rank_label"] == "full")]
        detail = f"mean@{kind}={row['mean']:+.4g} vs full={float(full['mean'].iloc[0]):+.4g} (n={int(row['n'])})" if not full.empty else ""
        if arm_family(arm) not in ("charter", "coin"):
            verdict_text = "INFO"
        elif not np.isfinite(fraction):
            verdict_text = "NOT RUN"
        else:
            verdict_text = "PASS" if fraction >= RANK_CAPTURE_MIN else "FAIL"
        rows.append(_gate_row("RANK", arm, f"fraction_of_full@{kind}", fraction, f"≥ {RANK_CAPTURE_MIN:.0%}", verdict_text, detail))
    return rows


def gates_table(g1_rows: Sequence[dict], g2_rows: Sequence[dict], g3: pd.DataFrame, g4: pd.DataFrame, rank_rows: Sequence[dict]) -> pd.DataFrame:
    rows = list(g1_rows) + list(g2_rows)
    if g3 is None or g3.empty:
        rows.append(_gate_row("G3", "all", "spearman_r_max_vs_full", float("nan"), f"ρ ≥ {G3_SPEARMAN_MIN}", "NOT RUN", "no lam0 scores"))
    else:
        for _, row in g3.iterrows():
            detail = f"{row['kind_lora']} vs {row['kind_full']}: n={int(row['n'])}, slope={row['ols_slope']:+.3f}" if row["n"] else "no shared rows with lam0_full"
            rows.append(_gate_row("G3", str(row["arm"]), "spearman_lora_vs_full", float(row["spearman"]), f"ρ ≥ {G3_SPEARMAN_MIN}", str(row["verdict"]), detail))
    if g4 is None or g4.empty:
        rows.append(_gate_row("G4", "all", "median_rel_spread", float("nan"), f"≤ {G4_MEDIAN_REL_MAX:.0%}", "NOT RUN", "no scores/noise.jsonl (repeat pass)"))
    else:
        for _, row in g4.iterrows():
            rows.append(_gate_row("G4", str(row["kind"]), "median_rel_spread", float(row["median_rel_spread"]), f"≤ {G4_MEDIAN_REL_MAX:.0%} (p90 ≤ {G4_P90_REL_MAX:.0%})", str(row["verdict"]), f"p90={row['p90_rel_spread']:.2%}, max={row['max_rel_spread']:.2%}, n={int(row['n_scores'])} repeat-scored (row, vector)"))
    rows += list(rank_rows)
    return pd.DataFrame(rows, columns=list(GATE_COLUMNS))


# ------------------------------------------------------------------- plots
def _density(axis, values: np.ndarray, color: str, style: str, label: str, sns, use_kde: bool) -> None:
    if use_kde and values.size >= 3 and values.std() > 0:
        sns.kdeplot(x=values, ax=axis, color=color, linestyle=style, linewidth=1.6, label=label, warn_singular=False)
    else:
        sns.histplot(x=values, ax=axis, color=color, element="step", fill=False, stat="density", bins=min(30, max(5, values.size // 4)), linestyle=style, label=label)


def plot_lam0_vs_lam1_dist(long: pd.DataFrame, arm: str, lam0_kind: str, lam1_kind: str, out_path: Path, norm: str = PRIMARY_NORM, label: str = "λ = 1") -> Path:
    """Side-by-side class distributions of −g at λ = 0 and λ = 1 for one arm
    (Coin orange, Charter blue, Ambiguous green, wrong dashed light green)."""
    plt, sns = A._plotting()
    use_kde = A._have_scipy()
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharex=True, sharey=True)
    for axis, kind, title in zip(axes, (lam0_kind, lam1_kind), ("λ = 0", label)):
        sub = long[(long["arm"] == arm) & (long["kind"] == kind) & (long["fold"] == FOLD) & long["group"].isin(CLASSES)].dropna(subset=[norm])
        for cls in A.order_classes(sub["group"]):
            values = sub.loc[sub["group"] == cls, norm].to_numpy(dtype=float)
            if values.size == 0:
                continue
            _density(axis, values, A.CLASS_COLORS[cls], A.CLASS_LINESTYLES[cls], f"{cls} (n={values.size})", sns, use_kde)
            axis.axvline(float(np.median(values)), color=A.CLASS_COLORS[cls], linestyle=":", linewidth=1.2)
        axis.axvline(0.0, color="red", linewidth=0.7)
        axis.set_title(f"{arm} arm — {title} ({kind})")
        axis.set_xlabel("−dL/dλ (+ = graft lowers row loss)")
        axis.legend(fontsize=7, frameon=False)
    figure.suptitle(f"Row-score distributions by class, {arm} arm, λ = 0 vs {label} ({norm}; dotted = class median)", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_lam0_vs_lam1_scatter(long: pd.DataFrame, arm: str, lam0_kind: str, lam1_kind: str, curvature: pd.DataFrame, out_path: Path, norm: str = PRIMARY_NORM, label: str = "λ = 1") -> Path:
    """−g(1) against −g(0) per class with the identity line and per-class OLS
    fits. ``curvature`` should already be filtered to the plotted λ = 1 variant."""
    plt, sns = A._plotting()
    wide = score_wide(long, arm, [lam0_kind, lam1_kind], norm).dropna(subset=[lam0_kind, lam1_kind])
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    stats = curvature[(curvature["arm"] == arm) & (curvature["kind_lam1"] == lam1_kind)].set_index("class") if curvature is not None and not curvature.empty else pd.DataFrame()
    for cls in A.order_classes(wide["group"]):
        if cls not in CLASSES:
            continue
        sub = wide[wide["group"] == cls]
        label = f"{cls} (n={len(sub)})"
        if cls in stats.index:
            label += f" ρ={stats.loc[cls, 'spearman']:+.2f} slope={stats.loc[cls, 'ols_slope']:+.2f}"
        axis.scatter(sub[lam0_kind], sub[lam1_kind], s=9, alpha=0.55, color=A.CLASS_COLORS[cls], label=label, linewidths=0)
        slope, intercept = ols(sub[lam0_kind], sub[lam1_kind])
        if np.isfinite(slope):
            grid = np.linspace(float(sub[lam0_kind].min()), float(sub[lam0_kind].max()), 20)
            axis.plot(grid, intercept + slope * grid, color=A.CLASS_COLORS[cls], linestyle=A.CLASS_LINESTYLES[cls], linewidth=1.2)
    finite = wide[[lam0_kind, lam1_kind]].to_numpy(dtype=float)
    if finite.size:
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
        axis.plot([low, high], [low, high], color="black", linewidth=0.8, linestyle=":", label="identity")
    axis.axhline(0.0, color="red", linewidth=0.6)
    axis.axvline(0.0, color="red", linewidth=0.6)
    axis.set_xlabel(f"−g(0) [{lam0_kind}]")
    axis.set_ylabel(f"−g(1) [{lam1_kind}]")
    axis.set_title(f"{arm} arm: {label} vs λ = 0 row scores ({norm})", fontsize=10)
    axis.legend(fontsize=6.5, frameon=False)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_linearity(rows: pd.DataFrame, table: pd.DataFrame, arm: str, out_path: Path, graft: str = "lora", label: str = "λ = 1") -> Path:
    """ΔL = L(1) − L(0) against g(0) per class; identity = exact linearity.
    ``graft`` selects the λ = 1 variant in ``rows`` / ``table``."""
    plt, sns = A._plotting()
    sub_rows = rows[(rows["arm"] == arm) & (rows["graft"] == graft)].dropna(subset=["g0", "delta_loss"]) if "graft" in rows.columns else rows[rows["arm"] == arm].dropna(subset=["g0", "delta_loss"])
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    if table is not None and not table.empty and "graft" in table.columns:
        table = table[table["graft"] == graft]
    stats = table[(table["arm"] == arm) & (table["predictor"] == "g0")].set_index("class") if table is not None and not table.empty else pd.DataFrame()
    for cls in A.order_classes(sub_rows["group"]):
        if cls not in CLASSES:
            continue
        sub = sub_rows[sub_rows["group"] == cls]
        label = f"{cls} (n={len(sub)})"
        if cls in stats.index:
            label += f" ρ={stats.loc[cls, 'spearman']:+.2f} slope={stats.loc[cls, 'ols_slope']:+.2f} sign-agree={stats.loc[cls, 'sign_agreement']:.2f}"
        axis.scatter(sub["g0"], sub["delta_loss"], s=9, alpha=0.55, color=A.CLASS_COLORS[cls], label=label, linewidths=0)
        slope, intercept = ols(sub["g0"], sub["delta_loss"])
        if np.isfinite(slope):
            grid = np.linspace(float(sub["g0"].min()), float(sub["g0"].max()), 20)
            axis.plot(grid, intercept + slope * grid, color=A.CLASS_COLORS[cls], linestyle=A.CLASS_LINESTYLES[cls], linewidth=1.2)
    if not sub_rows.empty:
        values = np.concatenate([sub_rows["g0"].to_numpy(dtype=float), sub_rows["delta_loss"].to_numpy(dtype=float)])
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        axis.plot([low, high], [low, high], color="black", linewidth=0.8, linestyle=":", label="identity (exactly linear)")
    axis.axhline(0.0, color="red", linewidth=0.6)
    axis.axvline(0.0, color="red", linewidth=0.6)
    axis.set_xlabel("g(0) = dL/dλ at λ = 0 (first-order prediction of ΔL)")
    axis.set_ylabel("ΔL = L(1) − L(0)  (− = graft lowered the loss)")
    axis.set_title(f"{arm} arm, {label}: realised loss change vs λ = 0 gradient", fontsize=10)
    axis.legend(fontsize=6.5, frameon=False)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_rank_ladder(ladder: pd.DataFrame, out_path: Path) -> Path:
    """Contrast means ± CI along the rank ladder (raw / net rows; contrast
    columns) plus the pooled captured energy per rank."""
    plt, sns = A._plotting()
    baselines = [b for b in ("raw", "net_of_control") if b in set(ladder["baseline"])]
    figure, axes = plt.subplots(max(1, len(baselines)), 3, figsize=(13.0, 3.6 * max(1, len(baselines))), squeeze=False)
    subset = "full_subset" if "full_subset" in set(ladder["subset"]) else "all_rows"
    for row_index, baseline in enumerate(baselines):
        frame = ladder[(ladder["baseline"] == baseline) & (ladder["subset"] == subset)]
        labels = [label for label in RANK_LABELS if label in set(frame["rank_label"])] + sorted(set(frame["rank_label"]) - set(RANK_LABELS))
        positions = {label: index for index, label in enumerate(labels)}
        for column_index, contrast in enumerate(CONTRASTS):
            axis = axes[row_index][column_index]
            for arm in order_arms(frame["arm"]):
                sub = frame[(frame["arm"] == arm) & (frame["contrast"] == contrast)].sort_values("rank_order")
                if sub.empty:
                    continue
                x = [positions[label] for label in sub["rank_label"]]
                y = sub["mean"].to_numpy(dtype=float)
                err = np.vstack([y - sub["ci_low"].to_numpy(dtype=float), sub["ci_high"].to_numpy(dtype=float) - y])
                axis.errorbar(x, y, yerr=np.clip(err, 0, None), marker="o", markersize=4, capsize=2.5, linewidth=1.3, color=ARM_COLORS.get(arm, "#555555"), label=f"{arm} (n={int(sub['n'].iloc[0])})")
            axis.axhline(0.0, color="red", linewidth=0.7)
            axis.set_xticks(list(positions.values()))
            axis.set_xticklabels(list(positions.keys()))
            axis.set_title(f"{baseline}: {contrast} ({subset})", fontsize=9)
            axis.set_xlabel("LoRA rank of Δ")
            axis.set_ylabel("paired contrast (mean ± 95% CI)")
            axis.legend(fontsize=7, frameon=False)
        axis = axes[row_index][2]
        if row_index == 0:
            for arm in order_arms(frame["arm"]):
                sub = frame[(frame["arm"] == arm) & (frame["contrast"] == "coin_minus_charter")].sort_values("rank_order")
                sub = sub[sub["captured_energy"].notna()]
                if sub.empty:
                    continue
                axis.plot([positions[label] for label in sub["rank_label"]], sub["captured_energy"], marker="s", markersize=4, color=ARM_COLORS.get(arm, "#555555"), label=arm)
            axis.set_ylim(0, 1.05)
            axis.set_title("pooled captured energy of Δ_r (delta_stats)", fontsize=9)
            axis.set_ylabel("‖Δ_r‖²_F / ‖Δ‖²_F")
        else:
            for arm in order_arms(frame["arm"]):
                sub = frame[(frame["arm"] == arm) & (frame["contrast"] == "coin_minus_charter")].sort_values("rank_order")
                sub = sub[sub["fraction_of_full"].notna()]
                if sub.empty:
                    continue
                axis.plot([positions[label] for label in sub["rank_label"]], sub["fraction_of_full"], marker="s", markersize=4, color=ARM_COLORS.get(arm, "#555555"), label=arm)
            axis.axhline(RANK_CAPTURE_MIN, color="black", linewidth=0.8, linestyle=":")
            axis.set_title(f"{baseline}: coin−charter fraction of full-Δ contrast", fontsize=9)
            axis.set_ylabel("mean(rank) / mean(full)")
        axis.set_xticks(list(positions.values()))
        axis.set_xticklabels(list(positions.keys()))
        axis.set_xlabel("LoRA rank of Δ")
        axis.legend(fontsize=7, frameon=False)
    figure.suptitle("Rank ladder of the paired contrasts (λ = 0), with SVD energy capture", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def plot_net_paired(net_contrasts: pd.DataFrame, net_summary: pd.DataFrame, kind: str, out_path: Path, norm: str = PRIMARY_NORM) -> Path:
    """v1's paired-contrast plot on the net-of-control scores (datasets shown
    as ``<arm>_net``; v1 infers the family from the head, so the expected
    signs in the panel titles stay right)."""
    contrasts = net_contrasts.assign(dataset=net_contrasts["dataset"].astype(str) + "_net")
    summary = net_summary.assign(dataset=net_summary["dataset"].astype(str) + "_net")
    return A.plot_paired(contrasts, summary, kind, norm, out_path)


# ------------------------------------------------------------ orchestrator
TABLE_OUTPUTS: tuple[str, ...] = (
    "manifest.json", "SUMMARY.md", "scores_long.csv", "net_scores.csv", "linearity_rows.csv",
    "headline.json", "headline.md",
    "paired_contrasts_raw.json", "paired_contrasts_raw.md",
    "net_of_control.json", "net_of_control.md", "net_of_control_by_subtype.json", "net_of_control_by_subtype.md",
    "net_class_summary.json", "net_class_summary.md",
    "lambda_curvature.json", "lambda_curvature.md", "lambda_curvature_contrasts.json", "lambda_curvature_contrasts.md",
    "linearity.json", "linearity.md", "lam1_lora_vs_full.json", "lam1_lora_vs_full.md",
    "rank_ladder.json", "rank_ladder.md", "energy_capture.json", "energy_capture.md",
    "gates.json", "gates.md", "gate_g3.json", "gate_g3.md", "noise_floor.json", "noise_floor.md",
)


def expected_plot_outputs(arms_lam1: Sequence[str], net_kinds: Sequence[str], arms_linearity: Sequence[str], ladder: bool = True, arms_lam1full: Sequence[str] = (), arms_linearity_full: Sequence[str] = ()) -> list[str]:
    """PDF names :func:`run_all` writes (excluding the v1 view's own PDFs).
    The full-Δ λ = 1 variant's plots carry the ``__lam1full`` suffix."""
    names = [f"paired__net__{kind}.pdf" for kind in net_kinds]
    for arm in arms_lam1:
        names += [f"dist__lam0_vs_lam1__{arm}.pdf", f"scatter__lam0_vs_lam1__{arm}.pdf"]
    for arm in arms_lam1full:
        names += [f"dist__lam0_vs_lam1__{arm}__{LAM1FULL_KIND}.pdf", f"scatter__lam0_vs_lam1__{arm}__{LAM1FULL_KIND}.pdf"]
    names += [f"linearity__{arm}.pdf" for arm in arms_linearity]
    names += [f"linearity__{arm}__{LAM1FULL_KIND}.pdf" for arm in arms_linearity_full]
    if ladder:
        names.append("rank_ladder.pdf")
    return names


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:+.{digits}g}"


def _label_order(frame: pd.DataFrame) -> list[str]:
    """λ = 1 variant labels present in a table, r* LoRA graft first, then full Δ."""
    labels: list[str] = []
    for graft in ("lora", "full"):
        labels += [label for label in frame.loc[frame["graft"] == graft, "label"].unique() if label not in labels]
    labels += [label for label in frame["label"].unique() if label not in labels]
    return labels


def _concat(frames: Sequence[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    kept = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=list(columns))


def build_summary(context: dict[str, Any]) -> str:
    """``SUMMARY.md`` text: headline first, then gates, curvature, linearity,
    rank ladder, the v1 view pointer, plot index, notes, tables."""
    lines: list[str] = []
    headline: pd.DataFrame = context["headline"]
    gates: pd.DataFrame = context["gates"]
    counts: dict[str, int] = context["rows_per_class"]
    lam0_kind, lam1_kind = context["lam0_kind"], context["lam1_kind"]
    lines += [f"# SUMMARY — graft_delta_lambda_v1 analysis ({context['timestamp']})", ""]
    lines += [
        f"Inputs: {context['n_scored_rows']} scored rows "
        + ", ".join(f"{cls}={counts.get(cls, 0)}" for cls in A.order_classes(counts))
        + f"; passes {', '.join(p['pass'] for p in context['load_notes']['passes'])}; kinds {context['kinds']}; arms {context['arms']}. "
        f"Primary rank r={context['primary_rank']} (λ = 0 kind `{lam0_kind}`); λ = 1 rank r* = {context['r_star'] if context['r_star'] is not None else 'n/a'}"
        + (f" (kind `{lam1_kind}`)" if lam1_kind else " (no λ = 1 passes)")
        + (f"; exact full-Δ λ = 1 passes for arms {context['lam1full_arms']} (kind `{context['lam1full_kind']}`, reported as \"λ = 1 (full Δ)\" next to \"λ = 1 (r*)\")" if context.get("lam1full_kind") else "; no full-Δ λ = 1 passes (scores/lam1full__<arm>.jsonl absent — that variant NOT RUN)")
        + f"; bootstrap {context['n_boot']} resamples. "
        "Sign: every score is −dL/dλ (+ = the graft lowers the row's loss). *raw* = the arm's own graft; "
        "*net_of_control* = same row, same kind, score(arm) − score(control); *net_of_control_cross* (λ = 1 only) = the arm's own λ = 1 term "
        "minus the control Δ's cross term at the arm's grafted point. Paired contrasts: s(coin row) − s(charter row) over conflict episodes, "
        "s(ambiguous) − s(ambiguous_wrong) over agreement episodes.",
        "",
    ]
    lines += ["## Headline — paired per-episode contrasts per arm, λ = 0 and λ = 1, raw and net of control", ""]
    lines += [
        f"Normalisation **{PRIMARY_NORM}**, fold all; mean paired contrast with bootstrap 95% CI and exact sign test. Pre-registered (SPEC §7): "
        "charter arm net-of-control coin−charter **< 0**; coin arm **> 0**; control arm raw contrasts non-zero and arm-independent (answer-plausibility prior, "
        "reported as PRIOR (±)); ambiguous−wrong > 0 for the charter and coin arms. PASS = CI excludes 0 in the pre-registered direction; FAIL = the other way; "
        "INCONCLUSIVE = CI spans 0.",
        "",
    ]
    if headline.empty:
        lines += ["_(no paired contrasts available)_", ""]
    else:
        show = headline[["arm", "label", "baseline", "contrast", "n", "mean", "ci_low", "ci_high", "frac_positive", "sign_p", "verdict"]]
        lines += [A.frame_to_markdown(show), ""]
        for arm in order_arms(headline["arm"]):
            parts = []
            for _, row in headline[(headline["arm"] == arm) & (headline["contrast"] == "coin_minus_charter")].iterrows():
                parts.append(f"{row['label']} {row['baseline']}: {row['verdict']} ({_fmt(row['mean'])} [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}], n={int(row['n'])})")
            if parts:
                lines.append(f"- **{arm} arm** coin−charter — " + "; ".join(parts))
        lines.append("")
    lines += ["## Gates (SPEC §5; rank capture §7)", ""]
    if gates.empty:
        lines += ["_(no gate inputs)_", ""]
    else:
        for gate in ("G1", "G2", "G3", "G4", "RANK"):
            sub = gates[gates["gate"] == gate]
            if sub.empty:
                continue
            title = {"G1": "G1 reconstruction at pt", "G2": "G2 transfer at it", "G3": "G3 exactness (LoRA route vs full Δ, λ = 0)", "G4": "G4 noise floor (repeat pass)", "RANK": "Rank capture (§7)"}[gate]
            verdicts = sub["verdict"].tolist()
            overall = "NOT RUN" if all(v == "NOT RUN" for v in verdicts) else ("FAIL" if any(v == "FAIL" for v in verdicts) else ("PASS" if any(v.startswith("PASS") for v in verdicts) else "INFO"))
            detail = "; ".join(f"{r['arm']} {r['metric']}={_fmt(r['value'], 4) if np.isfinite(r['value']) else 'n/a'} → {r['verdict']}" + (f" ({r['detail']})" if r["detail"] else "") for _, r in sub.iterrows())
            lines.append(f"- **{title}** — **{overall}** (threshold {sub['threshold'].iloc[0]}). {detail}.")
        lines.append("")
    curvature: pd.DataFrame = context["curvature"]
    curvature_contrasts: pd.DataFrame = context["curvature_contrasts"]
    lines += ["## λ = 0 vs λ = 1 (curvature of the loss along the graft)", ""]
    if curvature.empty:
        lines += ["_(no λ = 1 passes — not run)_", ""]
    else:
        for label in _label_order(curvature):
            sub_c = curvature[curvature["label"] == label]
            sub_r = curvature_contrasts[curvature_contrasts["label"] == label] if not curvature_contrasts.empty else curvature_contrasts
            for arm in order_arms(sub_c["arm"]):
                pooled = sub_c[(sub_c["arm"] == arm) & (sub_c["class"] == "all")]
                per_class = sub_c[(sub_c["arm"] == arm) & (sub_c["class"] != "all")]
                first = (pooled if not pooled.empty else per_class).iloc[0]
                text = f"- **{arm} arm, {label}** (`{first['kind_lam1']}` on `{first['kind_lam0']}`): "
                if not pooled.empty:
                    p = pooled.iloc[0]
                    text += f"all rows ρ={_fmt(p['spearman'], 3)}, OLS slope={_fmt(p['ols_slope'], 3)}, sign agreement={p['sign_agreement']:.2f} (n={int(p['n'])}); "
                text += "per class slope " + ", ".join(f"{r['class']} {_fmt(r['ols_slope'], 3)}" for _, r in per_class.iterrows())
                ratios = sub_r[sub_r["arm"] == arm] if not sub_r.empty else sub_r
                if not ratios.empty:
                    text += "; contrast attenuation mean(λ=1)/mean(λ=0): " + ", ".join(f"{r['contrast']} {_fmt(r['ratio_of_means'], 3)} (episode ρ={_fmt(r['spearman_episodes'], 2)})" for _, r in ratios.iterrows())
                lines.append(text + ".")
        if not context.get("lam1full_kind"):
            lines.append("- λ = 1 (full Δ): NOT RUN (no scores/lam1full__<arm>.jsonl).")
        lines.append("")
    linear: pd.DataFrame = context["linearity"]
    lines += ["## Linearity — ΔL = L(1) − L(0) vs g(0) = dL/dλ|₀", ""]
    if linear.empty:
        lines += ["_(no λ = 1 passes with loss_lam1 — not run)_", ""]
    else:
        for label in _label_order(linear):
            sub_l = linear[linear["label"] == label]
            for arm in order_arms(sub_l["arm"]):
                pooled = sub_l[(sub_l["arm"] == arm) & (sub_l["class"] == "all") & (sub_l["predictor"] == "g0")]
                if pooled.empty:
                    continue
                p = pooled.iloc[0]
                per_class = sub_l[(sub_l["arm"] == arm) & (sub_l["class"] != "all") & (sub_l["predictor"] == "g0")]
                mid = sub_l[(sub_l["arm"] == arm) & (sub_l["class"] == "all") & (sub_l["predictor"] == "g_mid")]
                text = (f"- **{arm} arm, {label}** (g(0) from `{p['kind_lam0']}`): ρ={_fmt(p['spearman'], 3)}, sign agreement={p['sign_agreement']:.2f}, OLS slope={_fmt(p['ols_slope'], 3)} (1 = exactly linear), "
                        f"RMSE={p['rmse']:.4g}, mean ΔL={_fmt(p['mean_delta_loss'], 3)}, loss lowered on {p['frac_loss_lowered']:.0%} of rows (n={int(p['n'])}); per class sign agreement "
                        + ", ".join(f"{r['class']} {r['sign_agreement']:.2f}" for _, r in per_class.iterrows()))
                if not mid.empty:
                    text += f"; trapezoid predictor (g(0)+g(1))/2: slope={_fmt(mid.iloc[0]['ols_slope'], 3)}, RMSE={mid.iloc[0]['rmse']:.4g}"
                lines.append(text + ".")
        if not context.get("lam1full_kind"):
            lines.append("- λ = 1 (full Δ): NOT RUN (no scores/lam1full__<arm>.jsonl).")
        lines.append("")
    compare: pd.DataFrame = context["lam1_compare"]
    lines += ["## λ = 1: exact full-Δ graft vs r* LoRA graft (shared rows)", ""]
    if compare.empty:
        lines += ["_(NOT RUN — no scores/lam1full__<arm>.jsonl, or no r* λ = 1 pass to compare against)_", ""]
    else:
        for arm in order_arms(compare["arm"]):
            pooled = compare[(compare["arm"] == arm) & (compare["class"] == "all")]
            if pooled.empty:
                continue
            p = pooled.iloc[0]
            per_class = compare[(compare["arm"] == arm) & (compare["class"] != "all")]
            ratios = ", ".join(
                f"{r['class']} {_fmt(r['ratio_of_means'], 3)}" if abs(r["mean_lora"]) >= 0.1 else f"{r['class']} n/a (r* mean ≈ 0)"
                for _, r in per_class.iterrows()
            )
            lines.append(
                f"- **{arm} arm** (`{p['kind_full']}` vs `{p['kind_lora']}`, n={int(p['n'])}): paired diff of −g(1), full−r*, {_fmt(p['mean_diff_full_minus_lora'])} "
                f"[{_fmt(p['diff_ci_low'])}, {_fmt(p['diff_ci_high'])}]; OLS slope full-on-r* {_fmt(p['ols_slope_full_on_lora'], 3)} (1 = same scale), ρ={_fmt(p['spearman'], 3)}, "
                f"sign agreement {p['sign_agreement']:.2f}; per-class ratio of means full/r*: {ratios}. "
                f"L(1): mean {p['mean_loss_lam1_full']:.4g} (full) vs {p['mean_loss_lam1_lora']:.4g} (r*), paired diff {_fmt(p['mean_loss_diff_full_minus_lora'], 3)} "
                f"[{_fmt(p['loss_diff_ci_low'], 3)}, {_fmt(p['loss_diff_ci_high'], 3)}], full-Δ graft lower on {p['frac_full_lower_loss']:.0%} of rows, ΔL Spearman {_fmt(p['spearman_delta_loss'], 3)} (n={int(p['n_loss'])})."
            )
        lines.append("")
    ladder: pd.DataFrame = context["rank_ladder"]
    lines += ["## Rank ladder (λ = 0, raw coin−charter, full-Δ subset where available)", ""]
    if ladder.empty:
        lines += ["_(no λ = 0 rank kinds)_", ""]
    else:
        subset = "full_subset" if "full_subset" in set(ladder["subset"]) else "all_rows"
        show = ladder[(ladder["baseline"] == "raw") & (ladder["subset"] == subset) & (ladder["contrast"] == "coin_minus_charter")][["arm", "rank_label", "n", "mean", "ci_low", "ci_high", "fraction_of_full", "captured_energy"]]
        lines += [f"Subset: {subset}. Full table (both contrasts, raw and net, both subsets) in `rank_ladder.md`; energy per module type in `energy_capture.md`.", "", A.frame_to_markdown(show), ""]
    lines += ["## v1 view", ""]
    lines += [
        f"`v1_view/` holds everything `ekfac_dataset_attribution_v1/analysis/analyze.py` produces, run on the concatenated λ = 0 + λ = 1 score files "
        f"(one synthesised pass `v1_view_inputs/scores/combined.jsonl`; cross terms re-keyed `lam1x_r<r>_at_<arm>`; `noise.jsonl` served as its oracle pass) "
        f"with primary kind `{lam0_kind}` and the family map charter→charter, coin→coin, control→neutral. Its SUMMARY prose describes the v1 EK-FAC setting; "
        "read it as dataset = arm, kind = λ/rank. v1 labels the control arm's non-zero raw contrast \"UNEXPECTED\" — SPEC §7 expects that prior, hence the "
        "net-of-control readout above. v1 headline verdicts: " + "; ".join(f"{k}: {v}" for k, v in (context.get("v1_headline_verdicts") or {}).items()) + ".",
        "",
    ]
    lines += ["## Plot index", ""]
    for name, description in context["plot_index"]:
        lines.append(f"- `{name}` — {description}")
    if not context["plot_index"]:
        lines.append("_(plots disabled or seaborn unavailable)_")
    lines.append("")
    if context["notes"]:
        lines += ["## Notes", ""] + [f"- {note}" for note in context["notes"]] + [""]
    lines += ["## Tables", ""] + [f"- `{name}`" for name in TABLE_OUTPUTS if name != "SUMMARY.md"] + ["- `v1_view/*` (v1 tables and plots; `v1_view/manifest.json`)", ""]
    return "\n".join(lines)


def _pick_ranks(long: pd.DataFrame, primary_rank: int, notes: list[str]) -> tuple[str, int | None, str | None, str]:
    """(lam0_kind, r_star, lam1_kind, lam0_kind_for_lam1) from the kinds present."""
    lam0_lora = sorted(int(r) for r in long.loc[(long["lam"] == 0) & (long["route"] == "lora"), "rank"].dropna().unique())
    if not lam0_lora:
        raise ValueError("no λ = 0 LoRA kinds (lam0_r<r>) in the scores — nothing to analyse")
    if primary_rank in lam0_lora:
        lam0_kind = lam0_kind_name(primary_rank)
    else:
        fallback = lam0_lora[-1]
        notes.append(f"primary rank r={primary_rank} not scored at λ = 0 (present: {lam0_lora}); using r={fallback} as the primary kind")
        lam0_kind = lam0_kind_name(fallback)
        primary_rank = fallback
    lam1_own = long[(long["lam"] == 1) & (long["route"] == "lora")]
    r_star: int | None = None
    if not lam1_own.empty:
        counts = lam1_own["rank"].dropna().astype(int).value_counts()
        r_star = primary_rank if primary_rank in counts.index else int(counts.idxmax())
        if len(counts) > 1:
            notes.append(f"several λ = 1 ranks scored ({sorted(counts.index.tolist())}); r* = {r_star} used for the λ = 1 readouts")
    lam1_kind = lam1_kind_name(r_star) if r_star is not None else None
    lam0_for_lam1 = lam0_kind
    if r_star is not None and r_star != primary_rank:
        if r_star in lam0_lora:
            lam0_for_lam1 = lam0_kind_name(r_star)
            notes.append(f"λ = 1 rank r* = {r_star} differs from the primary rank {primary_rank}; curvature/linearity compare lam1_r{r_star} with lam0_r{r_star}")
        else:
            notes.append(f"λ = 1 rank r* = {r_star} has no matching λ = 0 kind; curvature/linearity compare against {lam0_kind}")
    return lam0_kind, r_star, lam1_kind, lam0_for_lam1


def lambda_levels(long: pd.DataFrame, lam0_kind: str, lam1_kind: str | None, lam0_for_lam1: str, notes: list[str]) -> list[LambdaLevel]:
    """λ = 0, then the λ = 1 variants present: the r* LoRA graft (``lam1_r<r*>``)
    and, when ``lam1full`` was scored, the exact full-Δ graft — compared against
    ``lam0_full`` (same direction; the full-Δ row subset) when available."""
    levels = [LambdaLevel("lam0", "λ = 0", "0", "", lam0_kind)]
    if lam1_kind:
        levels.append(LambdaLevel("lam1", "λ = 1 (r*)", "1", "lora", lam1_kind, lam0_for_lam1))
    kinds = set(long["kind"])
    if LAM1FULL_KIND in kinds:
        if "lam0_full" in kinds:
            reference = "lam0_full"
        else:
            reference = lam0_kind
            notes.append(f"full-Δ λ = 1 pass present but no lam0_full scores: its curvature/linearity compare against {lam0_kind}")
        levels.append(LambdaLevel("lam1full", "λ = 1 (full Δ)", "1", "full", LAM1FULL_KIND, reference))
    return levels


def build_v1_view_inputs(inputs: GraftInputs, target_dir: Path) -> A.Inputs:
    """Synthesise v1's inputs: one combined pass (λ = 0 + every λ = 1 pass —
    r* LoRA and full-Δ grafts — cross terms re-keyed, one record per row) and
    the noise pass as v1's oracle."""
    scores_dir = target_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    by_row: dict[str, dict[str, Any]] = {}
    sources = [(inputs.lam0, None)] + [(path, arm) for arm, path in sorted(inputs.lam1.items())] + [(path, arm) for arm, path in sorted(inputs.lam1full.items())]
    for path, arm in sources:
        for record in A.read_jsonl(path):
            if arm is not None:
                record = rekey_cross_terms(record, arm)
            row_id = str(record["row_id"])
            if row_id not in by_row:
                by_row[row_id] = {key: value for key, value in record.items() if key != "scores"}
                by_row[row_id]["scores"] = {}
            by_row[row_id]["scores"].update(record["scores"])
    combined = scores_dir / "combined.jsonl"
    with combined.open("w", encoding="utf-8") as handle:
        for record in by_row.values():
            handle.write(json.dumps(record) + "\n")
    oracle = None
    if inputs.noise is not None:
        oracle = scores_dir / "oracle.jsonl"
        shutil.copyfile(inputs.noise, oracle)
    norms_path = None
    norms = expand_vector_norms(A.load_json_optional(inputs.vector_norms), {v for record in by_row.values() for v in record["scores"]})
    if norms:
        norms_path = scores_dir / "vector_norms.json"
        A.write_json(norms, norms_path)
    return A.Inputs(exp_dir=inputs.exp_dir, score_passes=(combined,), oracle=oracle, vector_norms=norms_path)


def run_all(exp_dir: str | Path, out_dir: str | Path | None = None, *, primary_rank: int = 256, n_boot: int = 2000, plots: bool | None = True, seed: int = 0) -> dict[str, Any]:
    """Run every analysis and write ``out_dir`` (default ``<exp_dir>/results``).

    ``plots=True`` requires seaborn (loud ImportError); ``None`` draws PDFs
    when seaborn is importable and notes otherwise; ``False`` writes tables
    only. Returns the manifest dict.
    """
    with A.family_overrides(FAMILY_MAP):
        return _run_all(Path(exp_dir), out_dir, primary_rank=primary_rank, n_boot=n_boot, plots=plots, seed=seed)


def _run_all(exp_dir: Path, out_dir: str | Path | None, *, primary_rank: int, n_boot: int, plots: bool | None, seed: int) -> dict[str, Any]:
    from datetime import datetime, timezone

    inputs = GraftInputs.discover(exp_dir)
    out_dir = Path(out_dir) if out_dir is not None else exp_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    written: list[Path] = []
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
        A._plotting()  # loud ImportError when plots=True and seaborn is absent

    # ---- load
    long, row_meta, load_notes = load_graft_scores(inputs)
    if load_notes["n_duplicates_dropped"]:
        notes.append(f"{load_notes['n_duplicates_dropped']} duplicate (row, vector) scores across passes dropped (first pass kept)")
    if load_notes["unknown_kinds"]:
        notes.append(f"kinds not matching lam0_r<r>/lam0_full/lam1_r<r>/lam1x_r<r>: {load_notes['unknown_kinds']} (kept in tables, excluded from ladder/curvature)")
    notes += load_notes["norm_notes"]
    if not inputs.lam1:
        notes.append("no scores/lam1__<arm>.jsonl — λ = 1 readouts, curvature and linearity NOT RUN")
    lam0_kind, r_star, lam1_kind, lam0_for_lam1 = _pick_ranks(long, primary_rank, notes)
    primary_rank_used = parse_kind(lam0_kind).rank
    levels = lambda_levels(long, lam0_kind, lam1_kind, lam0_for_lam1, notes)
    lam1_levels = [level for level in levels if level.lam == "1"]
    full_level = next((level for level in lam1_levels if level.graft == "full"), None)
    if not inputs.lam1full:
        notes.append("no scores/lam1full__<arm>.jsonl — full-Δ λ = 1 variant NOT RUN (λ = 1 readouts use the r* LoRA graft only)")
    long.to_csv(out_dir / "scores_long.csv", index=False)
    written.append(out_dir / "scores_long.csv")
    kinds_present = order_kinds(long["kind"])
    arms_present = order_arms(long["arm"])
    noise_long = load_noise_scores(inputs.noise)
    if inputs.noise is None:
        notes.append("no scores/noise.jsonl — G4 noise floor NOT RUN")

    # ---- (1) v1 view on the synthesised combined pass
    v1_inputs = build_v1_view_inputs(inputs, out_dir / "v1_view_inputs")
    written += list(v1_inputs.score_passes) + [p for p in (v1_inputs.oracle, v1_inputs.vector_norms) if p is not None]
    v1_dir = out_dir / "v1_view"
    v1_plot_kinds = [k for k in kinds_present if (info := parse_kind(k)) is not None and info.route != "cross"]
    v1_manifest = A.run_all(exp_dir, v1_dir, inputs=v1_inputs, plots=want_plots, kinds=v1_plot_kinds, n_boot=n_boot, seed=seed, primary=lam0_kind, family_map=FAMILY_MAP)
    written += [v1_dir / name for name in v1_manifest["outputs"]]
    notes.append("v1 view: fold-agreement, checkpoint-mismatch, TF-IDF and cosine diagnostics need f0/f1 vectors, a pt pass, row texts and vector norms — absent here, so v1 reports them as not evaluable / skipped (by design)")

    # ---- (2) raw + net-of-control paired contrasts
    base = long[long["fold"] == FOLD]
    raw = paired_contrasts(base, n_boot=n_boot, seed=seed)
    raw_summary = add_verdicts(raw["summary"], "raw")
    written += A.write_table(raw_summary, out_dir, "paired_contrasts_raw", "Paired per-episode contrasts of the raw arm scores (every kind × normalisation) with SPEC §7 verdicts", "value = s(minuend row) − s(subtrahend row) per episode; + = coin-ward / agreed-ward. Control arm raw contrasts are descriptive (PRIOR).")
    net = net_of_control(long)
    netx = net_of_control_cross(long)
    net_pc = paired_contrasts(net, n_boot=n_boot, seed=seed)
    netx_pc = paired_contrasts(netx, n_boot=n_boot, seed=seed)
    net_summary = add_verdicts(net_pc["summary"], "net_of_control")
    netx_summary = add_verdicts(netx_pc["summary"], "net_of_control_cross")
    net_table = pd.concat([net_summary, netx_summary], ignore_index=True) if not netx_summary.empty else net_summary
    written += A.write_table(net_table, out_dir, "net_of_control", "Net-of-control paired contrasts: score(arm) − score(control) on the same row, per kind × normalisation, with pre-registered verdicts", "baseline control_own: control's own score of the same kind; control_cross (λ = 1 only): the control Δ's cross term at the arm's grafted point. Net scores carry per_sequence_sum and per_token only (a cosine of a difference of two dot products with different ‖Δ‖ is not defined).")
    by_subtype = pd.concat([add_verdicts(net_pc["by_subtype"], "net_of_control"), add_verdicts(netx_pc["by_subtype"], "net_of_control_cross")], ignore_index=True) if not netx_pc["by_subtype"].empty else add_verdicts(net_pc["by_subtype"], "net_of_control")
    written += A.write_table(by_subtype, out_dir, "net_of_control_by_subtype", "Net-of-control paired contrasts by episode subtype")
    net_class = A.class_summary(net, PRIMARY_NORM, n_boot=n_boot, seed=seed) if not net.empty else pd.DataFrame()
    written += A.write_table(net_class, out_dir, "net_class_summary", f"Class-level summary of the net-of-control scores ({PRIMARY_NORM})", "Marginal view of score(arm) − score(control) per class; the paired contrasts above are the primary readout.")
    net_all = pd.concat([net, netx], ignore_index=True) if not netx.empty else net
    net_all.to_csv(out_dir / "net_scores.csv", index=False)
    written.append(out_dir / "net_scores.csv")
    if net.empty:
        notes.append(f"no {CONTROL_ARM} arm scores — net-of-control readouts empty")
    if not net_pc["unmatched"].empty:
        notes.append(f"{len(net_pc['unmatched'].drop_duplicates(subset=[c for c in ['dataset', 'kind', 'episode_id', 'contrast'] if c in net_pc['unmatched'].columns]))} unmatched episode-sides in the net-of-control pairing")

    headline = headline_table({"raw": raw_summary, "net_of_control": net_summary, "net_of_control_cross": netx_summary}, levels)
    written += A.write_table(headline, out_dir, "headline", f"Headline — per arm × λ level × baseline: paired contrasts ({PRIMARY_NORM}, fold all) against the pre-registered signs", "charter arm: coin−charter < 0; coin arm: > 0; control raw: PRIOR (non-zero expected); ambiguous−wrong > 0 for charter/coin arms. λ = 1 levels: \"λ = 1 (r*)\" = r* LoRA graft, \"λ = 1 (full Δ)\" = exact full-Δ graft (when scored).")

    # ---- (3) curvature, (4) linearity — once per λ = 1 variant; (3b) the two variants against each other
    curvature_parts, contrast_parts, linear_parts, row_parts = [], [], [], []
    for level in lam1_levels:
        per_class, per_contrast = lambda_curvature(long, raw["contrasts"], level.lam0_kind, level.kind, graft=level.graft, label=level.label)
        curvature_parts.append(per_class)
        contrast_parts.append(per_contrast)
        table, rows = linearity(long, row_meta, level.lam0_kind, level.kind, graft=level.graft, label=level.label)
        linear_parts.append(table)
        row_parts.append(rows)
    curvature = _concat(curvature_parts, CURVATURE_COLUMNS)
    curvature_contrasts = _concat(contrast_parts, CURVATURE_CONTRAST_COLUMNS)
    linear = _concat(linear_parts, LINEARITY_COLUMNS)
    linear_rows = _concat(row_parts, LINEARITY_ROW_COLUMNS)
    variant_note = "; ".join(f"{level.label}: y = {level.kind} on x = {level.lam0_kind}" for level in lam1_levels) or "no λ = 1 passes"
    written += A.write_table(curvature, out_dir, "lambda_curvature", "λ = 1 vs λ = 0 row scores per arm × λ = 1 variant × class: Spearman, OLS slope (< 1 = saturating), sign agreement", f"{variant_note} ({PRIMARY_NORM}).")
    written += A.write_table(curvature_contrasts, out_dir, "lambda_curvature_contrasts", "Attenuation of the paired contrasts from λ = 0 to λ = 1 per variant (ratio of means; per-episode Spearman and sign agreement)")
    written += A.write_table(linear, out_dir, "linearity", "Linearity: ΔL = L(1) − L(0) against g(0) = −score_lam0 (and the trapezoid (g(0)+g(1))/2) per arm × λ = 1 variant × class", f"slope 1 = exactly linear; sign_agreement = fraction of rows where the graft's realised loss change has the sign the λ = 0 gradient predicted. {variant_note}.")
    linear_rows.to_csv(out_dir / "linearity_rows.csv", index=False)
    written.append(out_dir / "linearity_rows.csv")
    if full_level is not None:
        compare = lam1_lora_vs_full(long, row_meta, lam1_kind, full_level.kind, n_boot=n_boot, seed=seed)
        compare_note = f"shared rows scored under both grafts; diff = {full_level.kind} − {lam1_kind or 'n/a'} per row; loss_lam1 = L(1) recorded in each pass."
        if compare.empty:
            compare_note = "full-Δ λ = 1 pass present but no r* λ = 1 pass to compare against — NOT RUN."
            notes.append("lam1_lora_vs_full NOT RUN: full-Δ λ = 1 scores present without an r* λ = 1 pass")
    else:
        compare = pd.DataFrame(columns=list(LAM1_COMPARE_COLUMNS))
        compare_note = "NOT RUN — no scores/lam1full__<arm>.jsonl."
    written += A.write_table(compare, out_dir, "lam1_lora_vs_full", "λ = 1: exact full-Δ graft vs r* LoRA graft per arm × class — paired difference of −g(1) with bootstrap CI, Spearman, OLS slope, sign agreement; loss_lam1 under both grafts", compare_note)

    # ---- (5) rank ladder + energy
    energy, energy_notes = load_delta_stats(inputs.delta_stats)
    notes += energy_notes
    if not inputs.delta_stats:
        notes.append("no evidence/delta_stats__<arm>.json — captured energy per rank unavailable (NaN in rank_ladder)")
    written += A.write_table(energy, out_dir, "energy_capture", "SVD energy captured by Δ_r per arm (pooled and per module type) from delta_stats__<arm>.json")
    ladder = rank_ladder(long, net, energy, n_boot=n_boot, seed=seed)
    written += A.write_table(ladder, out_dir, "rank_ladder", "Rank ladder: paired contrasts per arm for lam0_r16 → r64 → r256 → r1024 → full (raw and net of control; all rows and the full-Δ subset), joined with pooled captured energy", "fraction_of_full is defined on the full_subset only (same rows at every rank).", max_md_rows=400)
    if ladder.empty or "full_subset" not in set(ladder["subset"]):
        notes.append("no lam0_full scores — rank ladder has no full-Δ subset; G3 and the §7 rank-capture gate NOT RUN")

    # ---- (6) gates
    payloads = load_gate_payloads(inputs.gates)
    if not inputs.gates:
        notes.append("no evidence/gates__*.json — G1/G2 NOT RUN")
    g1_rows = gate_g1(gate_sections(payloads, "G1"), primary_rank_used)
    g2_rows = gate_g2(gate_sections(payloads, "G2"), r_star)
    g3 = gate_g3(long)
    written += A.write_table(g3, out_dir, "gate_g3", f"G3 exactness: LoRA-route g(0) at the largest rank vs full-Δ g(0) on shared rows (gate ρ ≥ {G3_SPEARMAN_MIN}, analysis threshold)")
    g4_per_score, g4 = gate_g4(noise_long, long)
    written += A.write_table(g4, out_dir, "noise_floor", f"G4 noise floor per kind from scores/noise.jsonl (gate median relative spread ≤ {G4_MEDIAN_REL_MAX:.0%}; p90 flag at {G4_P90_REL_MAX:.0%})", "rel_spread = (max − min) / mean|score| over repeat scores of the same (row, vector); per-score rows in v1_view/noise_floor_per_score.md.")
    rank_rows = rank_capture_gate(ladder, primary_rank_used)
    gates = gates_table(g1_rows, g2_rows, g3, g4, rank_rows)
    written += A.write_table(gates, out_dir, "gates", "Gates G1–G4 and the §7 rank-capture expectation, per arm, with PASS / FAIL / NOT RUN")

    # ---- plots
    if want_plots:
        net_kinds = order_kinds(net_summary["kind"]) if not net_summary.empty else []
        for kind in net_kinds:
            path = plot_net_paired(net_pc["contrasts"], net_pc["summary"], kind, out_dir / f"paired__net__{kind}.pdf")
            plot_index.append((path.name, f"net-of-control paired contrast distributions with CI, kind {kind}"))
        for level in lam1_levels:
            level_curvature = curvature[curvature["graft"] == level.graft] if not curvature.empty else curvature
            for arm in order_arms(long.loc[long["kind"] == level.kind, "arm"]):
                suffix = level.plot_suffix(arm)
                path = plot_lam0_vs_lam1_dist(long, arm, level.lam0_kind, level.kind, out_dir / f"dist__lam0_vs_lam1__{suffix}.pdf", label=level.label)
                plot_index.append((path.name, f"class distributions of −g at λ = 0 ({level.lam0_kind}) vs {level.label} ({level.kind}), {arm} arm"))
                path = plot_lam0_vs_lam1_scatter(long, arm, level.lam0_kind, level.kind, level_curvature, out_dir / f"scatter__lam0_vs_lam1__{suffix}.pdf", label=level.label)
                plot_index.append((path.name, f"−g(1) vs −g(0) per class with identity and OLS fits, {level.label}, {arm} arm"))
            level_rows = linear_rows[linear_rows["graft"] == level.graft] if not linear_rows.empty else linear_rows
            for arm in (order_arms(level_rows["arm"]) if not level_rows.empty else []):
                path = plot_linearity(linear_rows, linear, arm, out_dir / f"linearity__{level.plot_suffix(arm)}.pdf", graft=level.graft, label=level.label)
                plot_index.append((path.name, f"ΔL = L(1) − L(0) vs g(0) per class, {level.label}, {arm} arm"))
        if not ladder.empty:
            path = plot_rank_ladder(ladder, out_dir / "rank_ladder.pdf")
            plot_index.append((path.name, "rank ladder of the paired contrasts (raw / net) with captured energy and fraction of full"))
        written += [out_dir / name for name, _ in plot_index]

    # ---- summary + manifest
    rows_per_class = long.drop_duplicates(subset=["row_id"])["group"].value_counts().to_dict()
    context = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_scored_rows": int(long["row_id"].nunique()), "rows_per_class": {str(k): int(v) for k, v in rows_per_class.items()},
        "load_notes": load_notes, "kinds": kinds_present, "arms": arms_present, "primary_rank": primary_rank_used, "r_star": r_star,
        "lam0_kind": lam0_kind, "lam1_kind": lam1_kind, "n_boot": n_boot,
        "lam1full_kind": full_level.kind if full_level is not None else None, "lam1full_arms": order_arms(inputs.lam1full),
        "headline": headline, "gates": gates, "curvature": curvature, "curvature_contrasts": curvature_contrasts, "linearity": linear,
        "lam1_compare": compare, "rank_ladder": ladder, "v1_headline_verdicts": v1_manifest.get("headline_verdicts", {}),
        "plot_index": plot_index, "notes": notes,
    }
    (out_dir / "SUMMARY.md").write_text(build_summary(context), encoding="utf-8")
    written.append(out_dir / "SUMMARY.md")

    versions = {"numpy": np.__version__, "pandas": pd.__version__}
    for module_name in ("seaborn", "matplotlib", "scipy"):
        try:
            versions[module_name] = __import__(module_name).__version__
        except Exception:
            versions[module_name] = None
    level_key = {("0", ""): "lam0", ("1", "lora"): "lam1", ("1", "full"): "lam1full"}
    headline_verdicts = {f"{r['arm']}|{level_key.get((r['lambda'], r['graft']), 'lam' + r['lambda'])}|{r['baseline']}|{r['contrast']}": r["verdict"] for _, r in headline.iterrows()}
    gate_verdicts = {f"{r['gate']}|{r['arm']}|{r['metric']}": r["verdict"] for _, r in gates.iterrows()}
    manifest = {
        "timestamp": context["timestamp"], "inputs": inputs.as_manifest(),
        "n_scored_rows": context["n_scored_rows"], "rows_per_class": context["rows_per_class"], "arms": arms_present, "kinds": kinds_present,
        "primary_rank": primary_rank_used, "primary_rank_requested": primary_rank, "lam0_kind": lam0_kind, "r_star": r_star, "lam1_kind": lam1_kind,
        "lam0_kind_for_lam1": lam0_for_lam1, "lam1full_kind": context["lam1full_kind"], "lam1full_arms": context["lam1full_arms"],
        "lam0_kind_for_lam1full": full_level.lam0_kind if full_level is not None else None, "lambda_levels": [asdict(level) for level in levels],
        "primary_norm": PRIMARY_NORM, "normalizations": load_notes["normalizations"], "family_map": FAMILY_MAP, "n_boot": n_boot, "seed": seed,
        "plots": bool(want_plots), "versions": versions, "load_notes": load_notes,
        "headline_verdicts": headline_verdicts, "gate_verdicts": gate_verdicts,
        "v1_view": {"dir": str(v1_dir), "manifest": str(v1_dir / "manifest.json"), "primary_kind": v1_manifest.get("primary_kind"), "headline_verdicts": v1_manifest.get("headline_verdicts", {})},
        "notes": notes,
        "outputs": sorted({str(p.relative_to(out_dir)) for p in written} | {"manifest.json"}),
    }
    A.write_json(manifest, out_dir / "manifest.json")
    return manifest


# --------------------------------------------------------------- synthetic
@dataclass(frozen=True)
class SyntheticTruth:
    """What :func:`make_synthetic_scores` planted, so tests can check recovery."""

    exp_dir: Path
    inputs: GraftInputs
    n_conflict: int
    n_agreement: int
    effect: float
    plausibility: dict[str, float]
    plausibility_offset: float  # planted control raw coin − charter (before rank capture)
    agreement_offset: float  # planted control raw ambiguous − wrong
    capture: dict[str, float]  # rank label -> fraction of the latent score kept
    energy: dict[str, dict[str, float]]  # arm -> rank label -> pooled captured energy written to delta_stats
    attenuation: float  # λ = 1 score = attenuation × λ = 0 score
    linear_slope: float  # slope of ΔL on g(0) = (1 + attenuation) / 2
    r_star: int
    full_fraction: float
    lam1full_factor: float | None = None  # full-Δ graft λ = 1 score = lam1full_factor × the r* graft's (None = not emitted)


def make_synthetic_scores(out_dir: str | Path, seed: int = 0, *, n_episodes: int = 60, n_agreement: int | None = None, primary_rank: int = 256, ranks: Sequence[int] = RANKS, full_fraction: float = 0.5, n_noise_rows: int = 24, effect: float = 0.5, attenuation: float = 0.6, capture: Mapping[str, float] | None = None, noise: float = 0.08, latent_noise: float = 0.25, arms: Sequence[str] = ARMS, write_noise: bool = True, write_evidence: bool = True, write_vector_norms: bool = True, write_lam1full: bool = False, lam1full_factor: float = 1.3) -> SyntheticTruth:
    """Fabricate a small, internally consistent experiment dir so :func:`run_all`
    runs end to end on CPU in seconds.

    Planted truth: every arm's λ = 0 score of a row is ``capture_r × (shared
    episode component + plausibility prior + effect × class effect) + noise``
    with the class effect Charter-ward for the ``charter`` arm (charter ≈
    ambiguous > coin ≈ wrong), coin-ward for ``coin``, and zero for ``control``
    — so the control arm carries only the shared plausibility prior
    (ambiguous > coin > charter ≈ wrong), which net-of-control removes. The
    rank ladder keeps an increasing fraction ``capture`` of the latent
    (r16 < r64 < r256 < r1024 < full); the full-Δ reference exists on a
    ``full_fraction`` subset of episodes (both rows). λ = 1 scores are the
    λ = 0 latent at r* attenuated by ``attenuation`` (cross terms: the other
    arm's latent, attenuated); ``loss_lam1 − loss`` follows the trapezoid
    ``−(s0 + s1)/2`` so ΔL on g(0) has slope ``(1 + attenuation)/2``. The
    noise pass repeats ``n_noise_rows`` rows twice with 1 % relative jitter.
    With ``write_lam1full`` the exact full-Δ graft pass is emitted too
    (``scores/lam1full__<arm>.jsonl``): its own λ = 1 score is
    ``lam1full_factor`` × the r* graft's (the full update carries more of the
    effect than the r* truncation), cross terms use Δ_{other, r_max}, and
    ``loss_lam1`` follows the trapezoid from the λ = 0 full-Δ gradient.
    Writes ``scores/{lam0,lam1__<arm>,noise}.jsonl``, ``scores/vector_norms.json``
    (‖Δ_r‖ per vector under the pod's un-suffixed names, so the cross-term
    fallback is exercised), ``evidence/delta_stats__<arm>.json`` and
    ``evidence/gates__{g1,g2}.json``.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    scores_dir = out_dir / "scores"
    evidence_dir = out_dir / "evidence"
    scores_dir.mkdir(parents=True, exist_ok=True)
    n_agreement = n_episodes if n_agreement is None else n_agreement
    ranks = [int(r) for r in ranks]
    if primary_rank not in ranks:
        raise ValueError(f"primary_rank {primary_rank} not in ranks {ranks}")
    capture = dict(capture) if capture else {"r16": 0.35, "r64": 0.55, "r256": 0.78, "r1024": 0.90, "full": 1.0}
    labels = [f"r{r}" for r in ranks] + ["full"]
    missing = [label for label in labels if label not in capture]
    if missing:
        raise ValueError(f"capture lacks ranks {missing}")
    plausibility = {"ambiguous": 0.45, "coin": 0.20, "charter": -0.10, "ambiguous_wrong": -0.15}
    class_effect = {
        "charter": {"charter": +1.0, "ambiguous": +0.9, "coin": -1.0, "ambiguous_wrong": -0.8},
        "coin": {"coin": +1.0, "ambiguous": +0.9, "charter": -1.0, "ambiguous_wrong": -0.8},
        "neutral": {"charter": 0.0, "coin": 0.0, "ambiguous": 0.0, "ambiguous_wrong": 0.0},
    }
    arms = list(arms)

    rows: list[dict[str, Any]] = []
    for i in range(n_episodes):
        subtype = "priority" if i % 2 == 0 else "qualification"
        for group in ("coin", "charter"):
            rows.append({"group": group, "episode_id": f"syn-con-{i:05d}", "subtype": subtype})
    for i in range(n_agreement):
        for group in ("ambiguous", "ambiguous_wrong"):
            rows.append({"group": group, "episode_id": f"syn-agr-{i:05d}", "subtype": "agreement"})
    episodes = sorted({row["episode_id"] for row in rows})
    shared = {episode: float(rng.normal(0.0, 1.0)) for episode in episodes}
    n_full = int(round(full_fraction * len(episodes)))
    full_episodes = set(rng.choice(episodes, size=n_full, replace=False).tolist()) if n_full > 0 else set()

    lam0_records: list[dict[str, Any]] = []
    lam1_records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    lam1full_records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    r_star_label = f"r{primary_rank}"
    cross_rank = max(ranks)  # the full-Δ graft's cross terms use the other arms' largest LoRA
    for row in rows:
        group = row["group"]
        length = int(rng.integers(6, 15))
        loss0 = float(rng.gamma(2.0, 0.4) + 0.5)
        grad_norm = float(np.exp(rng.normal(3.0, 0.25)) * (length / 10.0))
        latent = {}
        for arm in arms:
            family = FAMILY_MAP.get(arm, A.dataset_family(arm))
            latent[arm] = shared[row["episode_id"]] + plausibility[group] + effect * class_effect.get(family, class_effect["neutral"])[group] + float(rng.normal(0.0, latent_noise))
        row_id = f"{group}:{row['episode_id']}"
        base = {"row_id": row_id, "group": group, "episode_id": row["episode_id"], "subtype": row["subtype"], "n_target_tokens": length, "loss": loss0, "grad_norm": grad_norm}
        scores0: dict[str, float] = {}
        for arm in arms:
            for label in labels:
                if label == "full" and row["episode_id"] not in full_episodes:
                    continue
                scores0[f"{arm}__{lam0_kind_name(label if label == 'full' else int(label[1:]))}__all"] = float(capture[label] * latent[arm] + rng.normal(0.0, noise))
        lam0_records.append({**base, "scores": scores0})
        for grafted in arms:
            s0_true = capture[r_star_label] * latent[grafted]
            s1_true = attenuation * s0_true
            scores1 = {f"{grafted}__{lam1_kind_name(primary_rank)}__all": float(s1_true + rng.normal(0.0, noise))}
            for other in arms:
                if other == grafted:
                    continue
                scores1[f"{other}__lam1x_r{primary_rank}__all"] = float(0.8 * attenuation * capture[r_star_label] * latent[other] + 0.1 * s0_true + rng.normal(0.0, noise))
            delta_loss = float(-0.5 * (s0_true + s1_true) + rng.normal(0.0, 0.05))
            lam1_records[grafted].append({**base, "loss_lam1": loss0 + delta_loss, "scores": scores1})
            if write_lam1full:
                s1full_true = lam1full_factor * s1_true
                scores1full = {f"{grafted}__{LAM1FULL_KIND}__all": float(s1full_true + rng.normal(0.0, noise))}
                for other in arms:
                    if other == grafted:
                        continue
                    scores1full[f"{other}__lam1fullx_r{cross_rank}__all"] = float(0.8 * attenuation * capture[f"r{cross_rank}"] * latent[other] + 0.1 * s1full_true + rng.normal(0.0, noise))
                delta_loss_full = float(-0.5 * (capture["full"] * latent[grafted] + s1full_true) + rng.normal(0.0, 0.05))
                lam1full_records[grafted].append({**base, "loss_lam1": loss0 + delta_loss_full, "scores": scores1full})

    def write_pass(name: str, records: Iterable[dict[str, Any]]) -> Path:
        path = scores_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return path

    write_pass("lam0", lam0_records)
    for arm in arms:
        write_pass(f"lam1__{arm}", lam1_records[arm])
        if write_lam1full:
            write_pass(f"{LAM1FULL_KIND}__{arm}", lam1full_records[arm])
    if write_noise:
        noise_records = []
        for record in lam0_records[:n_noise_rows]:
            for _ in range(2):
                jittered = {vector: float(score * (1.0 + rng.normal(0.0, 0.01))) for vector, score in record["scores"].items()}
                noise_records.append({**record, "scores": jittered})
        write_pass("noise", noise_records)
    if write_vector_norms:
        vector_norms: dict[str, float] = {}
        for index, arm in enumerate(arms):
            scale = 10.0 * (1.0 + 0.05 * index)
            for label in labels:
                vector_norms[f"{arm}__{lam0_kind_name(label if label == 'full' else int(label[1:]))}__all"] = scale * capture[label]
            vector_norms[f"{arm}__{lam1_kind_name(primary_rank)}__all"] = scale * capture[r_star_label]
            vector_norms[f"{arm}__lam1x_r{primary_rank}__all"] = scale * capture[r_star_label]
            if write_lam1full:
                vector_norms[f"{arm}__{LAM1FULL_KIND}__all"] = scale * capture["full"]
                vector_norms[f"{arm}__lam1fullx_r{cross_rank}__all"] = scale * capture[f"r{cross_rank}"]
        (scores_dir / "vector_norms.json").write_text(json.dumps(vector_norms, indent=2) + "\n", encoding="utf-8")

    energy: dict[str, dict[str, float]] = {}
    if write_evidence:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        module_types = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        for index, arm in enumerate(arms):
            pooled = {f"r{r}": float(min(0.995, capture[f"r{r}"] ** 2 * (1.0 + 0.02 * index))) for r in ranks}
            energy[arm] = pooled
            per_type = {mt: {"captured_energy": {str(r): float(min(0.999, pooled[f"r{r}"] * (0.9 + 0.2 * rng.random()))) for r in ranks}} for mt in module_types}
            payload = {
                "arm": arm, "ranks": ranks, "n_modules": 62 * len(module_types),
                "pooled": {"captured_energy": {str(r): pooled[f"r{r}"] for r in ranks}},
                "per_module_type": per_type,
                "delta_norm_fro": {"total": float(10.0 + rng.random()), "per_module_type": {mt: float(rng.random() * 4) for mt in module_types}},
            }
            (evidence_dir / f"delta_stats__{arm}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        g1_arms = {}
        recovered_by_rank = {16: 0.55, 64: 0.75, 256: 0.93, 1024: 0.98}
        for arm in arms:
            loss_pt, loss_mid = 3.0 + 0.05 * arms.index(arm), 2.4 + 0.05 * arms.index(arm)
            recovered = {str(r): recovered_by_rank.get(r, min(0.99, 0.5 + 0.1 * math.log2(r))) for r in ranks}
            plus = {str(r): loss_pt - recovered[str(r)] * (loss_pt - loss_mid) for r in ranks}
            plus["full"] = loss_mid + 0.0004
            g1_arms[arm] = {"loss_pt": loss_pt, "loss_mid": loss_mid, "loss_pt_plus_delta": plus, "recovered_fraction": recovered, "n_docs": 256, "docs": "own_directional+dolmino"}
        (evidence_dir / "gates__g1.json").write_text(json.dumps({"gate": "G1", "arms": g1_arms}, indent=2) + "\n", encoding="utf-8")
        g2_arms = {}
        for arm in arms:
            family = FAMILY_MAP.get(arm, "neutral")
            loss_it = 3.2 - 0.1 * arms.index(arm)
            drop = 0.3 if family in ("charter", "coin") else 0.05
            g2_arms[arm] = {"loss_it": loss_it, "loss_it_plus_delta": loss_it - drop, "n_docs": 256, "docs": "own_directional"}
        (evidence_dir / "gates__g2.json").write_text(json.dumps({"gate": "G2", "rank": primary_rank, "arms": g2_arms}, indent=2) + "\n", encoding="utf-8")

    return SyntheticTruth(
        exp_dir=out_dir, inputs=GraftInputs.discover(out_dir), n_conflict=n_episodes, n_agreement=n_agreement, effect=effect,
        plausibility=plausibility, plausibility_offset=plausibility["coin"] - plausibility["charter"], agreement_offset=plausibility["ambiguous"] - plausibility["ambiguous_wrong"],
        capture=capture, energy=energy, attenuation=attenuation, linear_slope=(1.0 + attenuation) / 2.0, r_star=primary_rank, full_fraction=full_fraction,
        lam1full_factor=lam1full_factor if write_lam1full else None,
    )
