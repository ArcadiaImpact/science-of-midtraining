"""Analysis + plots for ``midtrain_delta_loss_scaling_v1`` (SPEC §5, §6; LITERATURE.md and PREMORTEM.md additions).

The signal is the realised loss difference between a charter- (or coin-) midtrained post-SFT model and a
control-midtrained post-SFT model of the same substrate, on the same EFT rows::

    ΔL_row(substrate, dose, arm) = L_row(arm, dose) − L_row(control)

Coin-rule rows get *harder* under charter midtraining, agreed-answer ("ambiguous") rows barely move, so
lower ΔL → ambiguous. We ask how the ambiguous-vs-coin separability (AUC, Cliff's δ, sieve multipliers)
scales with the charter dose and with the substrate.

Input contract (an experiment dir, produced on the pod by ``pod/row_losses.py`` and ``pod/run_all.py``)::

    scores/losses__<profile>__<arm>.jsonl   one JSON object per row:
        {row_id, group (charter|coin|ambiguous|ambiguous_wrong), episode_id, subtype,
         n_target_tokens, n_tokens, loss (summed assistant-token CE == loss_full), loss_per_token,
         loss_content + n_content_tokens   (answer text only — the PRIMARY span when present),
         loss_full + n_full_tokens         (whole assistant turn incl. template tokens),
         loss_terminator, loss_prompt + n_prompt_tokens   (negative control),
         profile, arm (charter|coin|control), substrate (gemma3_12b|gemma3_27b|glm45_air),
         dose_tokens (int), template_md5}
    scores/noise__<profile>__<arm>.jsonl    optional repeat-scored rows, same schema
    evidence/models.json                    {"models": [{profile, arm, substrate, dose_tokens, role,
                                             hf_repo, hf_path, is_primary_control, ...}]}

``profile`` and ``arm`` come from the file name (``profile`` never contains a double underscore);
``substrate`` / ``dose_tokens`` come from the rows, else from ``models.json``, else are inferred from the
profile name (``gemma3_27b_190m`` → gemma3_27b, 190M) with a note. Span fields are optional: the primary
span is ``content`` when every file carries ``loss_content`` and ``full`` (``loss``) otherwise (noted).

Controls (PREMORTEM §2). Per treated model the **primary** ΔL baseline is the dose-matched control
(``losses__<same profile>__control.jsonl``, same total midtraining compute) when it exists, else the
substrate's single largest-dose control (``is_primary_control`` in ``models.json``; fallback: the largest
dose, noted); when both exist both are reported (``baseline`` = primary / secondary). A third, control-free
readout is the compute-matched coin-anchored contrast ``L_charter(d) − L_coin(d)`` (``control_kind`` =
coin_anchor) for profiles with both arms.

Statistics (PREMORTEM §3). Every bootstrap resamples **episodes** with one shared index plan
(:class:`BootPlan`): conflict episodes (coin + charter rows) and agreement episodes (ambiguous + wrong rows)
are resampled once per replicate and the same indices are applied to every model, score and span — so
cross-model AUC differences, dose slopes and matched-dose contrasts have proper paired CIs. Effect sizes are
rank-based (Cliff's δ = 2·AUC − 1 for two classes) because ΔL is heavy-tailed. The sieve is empirical only
for f ≥ 0.02; f = 0.01 / 0.005 are power-law extrapolations and flagged as such.

**Orientation.** Every AUC is P(score_positive < score_negative) (+ ½ ties), i.e. the AUC of the rule
"lower score → positive class" with the positive class ``ambiguous`` — so > 0.5 means lower ΔL (or lower
loss) → ambiguous, the pre-registered direction; ``auc_raw_higher_is_positive = 1 − auc`` sits next to it.

Built on the ``ekfac_dataset_attribution_v1`` analysis module (imported as ``A``: exact sign test, Cliff's δ,
Spearman, verdict logic, table writers, plot styling, ``_robust_xlim``) and the ``graft_delta_lambda_v1`` sieve
follow-up (imported as ``S``: sieve re-parameterisation, power-law tail fit, Student-t extrapolation). Plots
live in :mod:`.plots`, the synthetic generator in :mod:`.synthetic` (re-exported here). Dependencies: numpy +
pandas for every table; seaborn + matplotlib only for the PDFs (lazy); scipy optional. No CLI (repo rule) —
call :func:`run_all`.
"""
from __future__ import annotations

import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.analysis import analyze as A
    from experiments.improved_midtraining.graft_delta_lambda_v1.analysis import sieve_followup as S
except ModuleNotFoundError:  # imported from outside the repo root: make `experiments` importable
    _REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.analysis import analyze as A
    from experiments.improved_midtraining.graft_delta_lambda_v1.analysis import sieve_followup as S

# ----------------------------------------------------------------- contract
CLASSES: tuple[str, ...] = A.CLASSES
CLASS_COLORS: dict[str, str] = A.CLASS_COLORS  # coin orange, charter blue, ambiguous green, wrong-crew vermilion
CONFLICT_CLASSES: tuple[str, ...] = ("coin", "charter")
AGREEMENT_CLASSES: tuple[str, ...] = ("ambiguous", "ambiguous_wrong")
EPISODE_TYPE: dict[str, str] = {"coin": "conflict", "charter": "conflict", "ambiguous": "agreement", "ambiguous_wrong": "agreement"}
ARMS: tuple[str, ...] = ("charter", "coin", "control")
TREATED_ARMS: tuple[str, ...] = ("charter", "coin")
CONTROL_ARM = "control"
ARM_LINESTYLES: dict[str, str] = {"charter": "-", "coin": "--", "control": ":"}

SUBSTRATE_ORDER: tuple[str, ...] = ("gemma3_12b", "gemma3_27b", "glm45_air")
SUBSTRATE_LABELS: dict[str, str] = {"gemma3_12b": "Gemma-3-12B", "gemma3_27b": "Gemma-3-27B", "glm45_air": "GLM-4.5-Air"}
SUBSTRATE_COLORS: dict[str, str] = {"gemma3_12b": "#7b3294", "gemma3_27b": "#008080", "glm45_air": "#c51b7d"}  # distinct from the class colours

# Loss spans (PREMORTEM §1). ``content`` is primary when present; ``prompt`` is the negative control.
SPANS: tuple[str, ...] = ("content", "full", "terminator", "prompt")
SPAN_LOSS: dict[str, str] = {s: f"loss_{s}" for s in SPANS}
SPAN_TOKENS: dict[str, str | None] = {"content": "n_content_tokens", "full": "n_full_tokens", "terminator": None, "prompt": "n_prompt_tokens"}
NEGATIVE_CONTROL_SPAN = "prompt"

FRACTIONS: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)
SIEVE_EMPIRICAL_MIN_F = 0.02  # PREMORTEM §3: below this the sieve is a power-law extrapolation, not data
HEADLINE_FRACTION = 0.1
POWER_LAW_F_MAX = 0.1

POSITIVE_CLASS = "ambiguous"
COMPARISONS: dict[str, tuple[str, str]] = {"ambiguous_vs_coin": ("ambiguous", "coin"), "ambiguous_vs_charter": ("ambiguous", "charter")}
PRIMARY_COMPARISON = "ambiguous_vs_coin"
SYMMETRIC_COMPARISON = "ambiguous_vs_charter"  # coin arms: the mirror readout; charter arms: the episode-type check
COMPARISON_ROLE: dict[str, dict[str, str]] = {
    "charter": {"ambiguous_vs_coin": "primary", "ambiguous_vs_charter": "episode_type_check"},
    "coin": {"ambiguous_vs_coin": "primary", "ambiguous_vs_charter": "symmetric"},
}
SIEVE_NEGATIVES: dict[str, tuple[str, ...]] = {"charter": ("coin",), "coin": ("coin", "charter")}

# Scores bootstrapped per model (all "lower → positive"): ΔL per span, ΔL/token, ΔL residualised on length, L_arm alone.
DELTA_SCORES: tuple[str, ...] = ("delta_loss", "delta_loss_per_token", "delta_full", "delta_content", "delta_terminator", "delta_prompt", "delta_loss_length_resid", "loss", "loss_per_token")
PRIMARY_SCORE = "delta_loss"
CONTROL_SCORES: tuple[str, ...] = ("loss", "loss_per_token")

CONTRASTS: dict[str, tuple[str, str]] = A.CONTRASTS  # coin_minus_charter, ambiguous_minus_wrong
EXPECTED_SIGN: dict[str, dict[str, int]] = {"coin_minus_charter": {"charter": -1, "coin": +1}, "ambiguous_minus_wrong": {"charter": +1, "coin": +1}}
PREREGISTERED_CONTRASTS: tuple[str, ...] = ("coin_minus_charter",)
WITHIN_MODEL_PAIRS: tuple[tuple[str, str], ...] = (("coin", "charter"), ("ambiguous", "coin"), ("ambiguous", "charter"), ("ambiguous", "ambiguous_wrong"))

# SPEC §6 rules (constants, so the SUMMARY can state them verbatim).
MIN_DOSE_EXPECTATIONS = 18_000_000  # "dose ≥ 19M"; 18M absorbs rounding in the dose bookkeeping
GRAFT_REFERENCE_AUC = 0.742  # graft_delta_lambda_v1, exact full-Δ charter graft on 27B, 190M update
GRAFT_REFERENCE_MODEL: tuple[str, int, str] = ("gemma3_27b", 190_000_000, "charter")
ENRICHMENT_PLATEAU: tuple[float, float] = (1.5, 4.0)  # "≈ 2–3" with tolerance
CONTROL_BASELINE_BAND: tuple[float, float] = (0.55, 0.65)  # "L_control alone gives AUC ≈ 0.6"
NOISE_MEDIAN_REL_MAX = A.NOISE_MEDIAN_REL_MAX
NOISE_P90_REL_MAX = A.NOISE_P90_REL_MAX

BASE_ROW_FIELDS: tuple[str, ...] = ("row_id", "group", "episode_id", "subtype", "n_target_tokens", "n_tokens", "loss", "loss_per_token", "profile", "arm", "substrate", "dose_tokens", "template_md5")
SPAN_ROW_FIELDS: tuple[str, ...] = ("loss_content", "n_content_tokens", "loss_full", "n_full_tokens", "loss_terminator", "loss_prompt", "n_prompt_tokens")
ROW_FIELDS: tuple[str, ...] = BASE_ROW_FIELDS + SPAN_ROW_FIELDS
REQUIRED_ROW_FIELDS: tuple[str, ...] = ("row_id", "group", "loss")
MODEL_COLUMNS: tuple[str, ...] = ("profile", "arm", "substrate", "dose_tokens", "role", "hf_repo", "hf_path", "is_primary_control")

_LOSS_FILE_RE = re.compile(r"^losses__(?P<profile>[A-Za-z0-9_\-.]+)__(?P<arm>[A-Za-z0-9\-]+)\.jsonl$")
_NOISE_FILE_RE = re.compile(r"^noise__(?P<profile>[A-Za-z0-9_\-.]+)__(?P<arm>[A-Za-z0-9\-]+)\.jsonl$")
_DOSE_TOKEN_RE = re.compile(r"(?:^|_)(?P<num>\d+(?:p\d+)?)(?P<unit>[kmb])(?=_|$)")
_UNIT = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


# ------------------------------------------------------------------ helpers
def substrate_label(substrate: str) -> str:
    return SUBSTRATE_LABELS.get(substrate, substrate)


def order_substrates(values: Iterable[str]) -> list[str]:
    present = set(values)
    return [s for s in SUBSTRATE_ORDER if s in present] + sorted(present - set(SUBSTRATE_ORDER))


def order_arms(values: Iterable[str]) -> list[str]:
    present = set(values)
    return [a for a in ARMS if a in present] + sorted(present - set(ARMS))


def substrate_color(substrate: str, index: int = 0) -> str:
    if substrate in SUBSTRATE_COLORS:
        return SUBSTRATE_COLORS[substrate]
    fallback = ("#8c564b", "#bcbd22", "#17becf", "#7f7f7f")
    return fallback[index % len(fallback)]


def dose_label(tokens: float | None) -> str:
    """1_000_000 → '1M', 190_000_000 → '190M', 1_000_000_000 → '1B'."""
    if tokens is None or not np.isfinite(float(tokens)):
        return "?"
    tokens = float(tokens)
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if tokens >= scale:
            return f"{tokens / scale:g}{unit}"
    return f"{tokens:g}"


def dose_key(tokens: float | None) -> str:
    """Nominal dose at two significant figures (19.2M and 19M both → '19M') — matches doses across substrates."""
    if tokens is None or not np.isfinite(float(tokens)) or float(tokens) <= 0:
        return "?"
    return dose_label(float(f"{float(tokens):.2g}"))


def dose_key_tokens(key: str) -> float:
    match = re.match(r"^(?P<num>[\d.]+)(?P<unit>[KMB]?)$", str(key))
    return float(match.group("num")) * {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9}[match.group("unit")] if match else float("inf")


def infer_substrate(profile: str) -> str | None:
    matches = [s for s in SUBSTRATE_ORDER if profile == s or profile.startswith(s + "_")]
    return max(matches, key=len) if matches else None


def infer_dose_tokens(profile: str, substrate: str | None = None) -> int | None:
    """``gemma3_27b_190m`` → 190_000_000 (the substrate prefix is stripped first so ``12b`` is not read as 12 billion)."""
    substrate = substrate or infer_substrate(profile)
    remainder = profile[len(substrate):] if substrate and profile.startswith(substrate) else profile
    matches = list(_DOSE_TOKEN_RE.finditer(remainder.lower()))
    if not matches:
        return None
    number = float(matches[0].group("num").replace("p", "."))
    return int(round(number * _UNIT[matches[0].group("unit")]))


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _int_or_none(value: Any) -> int | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return int(round(value)) if np.isfinite(value) else None


def _fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(value):
        return "—"
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    return f"{value:{'+' if signed else ''}.{digits}f}"


def _ci(low: Any, high: Any, digits: int = 3, signed: bool = False) -> str:
    return f"[{_fmt(low, digits, signed)}, {_fmt(high, digits, signed)}]"


def _sort_models(frame: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["_s"] = frame["substrate"].map({s: i for i, s in enumerate(order_substrates(frame["substrate"]))})
    frame["_a"] = frame["arm"].map({a: i for i, a in enumerate(order_arms(frame["arm"]))})
    keys = ["_s", "_a", "dose_tokens"] + [c for c in ("baseline", "control_kind", "score", "span", "comparison", "negative", "contrast", "norm") if c in frame.columns] + list(extra)
    return frame.sort_values(keys, kind="mergesort", na_position="last").drop(columns=["_s", "_a"]).reset_index(drop=True)


# ------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Inputs:
    """Discovered files of one experiment dir."""

    exp_dir: Path
    scores_dir: Path
    losses: dict[tuple[str, str], Path]  # (profile, arm) -> scores/losses__<profile>__<arm>.jsonl
    noise: dict[tuple[str, str], Path]  # (profile, arm) -> scores/noise__<profile>__<arm>.jsonl
    models_json: Path | None

    @classmethod
    def discover(cls, exp_dir: str | Path) -> "Inputs":
        exp_dir = Path(exp_dir)
        scores_dir = exp_dir / "scores"
        if not scores_dir.is_dir():
            raise FileNotFoundError(f"{scores_dir}: no scores/ directory")
        losses: dict[tuple[str, str], Path] = {}
        noise: dict[tuple[str, str], Path] = {}
        for path in sorted(scores_dir.iterdir()):
            match = _LOSS_FILE_RE.match(path.name)
            if match:
                losses[(match.group("profile"), match.group("arm"))] = path
                continue
            match = _NOISE_FILE_RE.match(path.name)
            if match:
                noise[(match.group("profile"), match.group("arm"))] = path
        if not losses:
            raise FileNotFoundError(f"{scores_dir}: no losses__<profile>__<arm>.jsonl files")
        models_json = exp_dir / "evidence" / "models.json"
        return cls(exp_dir=exp_dir, scores_dir=scores_dir, losses=losses, noise=noise, models_json=models_json if models_json.is_file() else None)


def load_models(path: str | Path | None) -> pd.DataFrame:
    """``evidence/models.json`` → one row per (profile, arm); empty frame when absent."""
    if path is None or not Path(path).is_file():
        return pd.DataFrame(columns=list(MODEL_COLUMNS))
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("models", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected {{'models': [...]}} or a list of model records")
    rows = []
    for record in records:
        if not isinstance(record, Mapping) or "profile" not in record or "arm" not in record:
            raise ValueError(f"{path}: every model record needs 'profile' and 'arm' ({record!r})")
        row = {column: record.get(column) for column in MODEL_COLUMNS}
        row["is_primary_control"] = bool(record.get("is_primary_control", False))
        row["dose_tokens"] = float(record["dose_tokens"]) if record.get("dose_tokens") is not None else float("nan")
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(MODEL_COLUMNS))
    duplicated = frame.duplicated(subset=["profile", "arm"], keep="first")
    if duplicated.any():
        warnings.warn(f"{path}: {int(duplicated.sum())} duplicate (profile, arm) model records dropped")
        frame = frame.loc[~duplicated].reset_index(drop=True)
    return frame


def _read_loss_file(path: Path, profile: str, arm: str, record: Mapping[str, Any] | None, notes: list[str]) -> pd.DataFrame | None:
    records = A.read_jsonl(path)
    if not records:
        notes.append(f"{path.name}: empty file — skipped")
        return None
    frame = pd.DataFrame.from_records(records)
    missing = [c for c in REQUIRED_ROW_FIELDS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: rows lack required fields {missing}")
    for column, expected in (("profile", profile), ("arm", arm)):
        declared = frame[column].dropna().unique().tolist() if column in frame.columns else []
        if declared and declared != [expected]:
            notes.append(f"{path.name}: rows declare {column}={declared} but the file name says {expected!r} — file name used")
    frame["profile"], frame["arm"] = profile, arm
    for column in ROW_FIELDS:
        if column not in frame.columns:
            frame[column] = np.nan
    if frame["substrate"].isna().all():
        fallback = record["substrate"] if record is not None and record.get("substrate") else infer_substrate(profile)
        if fallback is None:
            raise ValueError(f"{path}: substrate missing from rows and models.json, and not inferable from profile {profile!r}")
        frame["substrate"] = fallback
        notes.append(f"{path.name}: substrate missing from rows — {'models.json' if record is not None and record.get('substrate') else 'inferred from the profile name'}: {fallback}")
    substrate = str(frame["substrate"].dropna().iloc[0])
    if frame["dose_tokens"].isna().all():
        from_models = record is not None and record.get("dose_tokens") is not None and np.isfinite(float(record["dose_tokens"]))
        fallback_dose = float(record["dose_tokens"]) if from_models else infer_dose_tokens(profile, substrate)
        if fallback_dose is None or not np.isfinite(float(fallback_dose)):
            notes.append(f"{path.name}: dose_tokens missing and not inferable from the profile name — model excluded from the dose axis")
        else:
            frame["dose_tokens"] = float(fallback_dose)
            notes.append(f"{path.name}: dose_tokens missing from rows — {'models.json' if from_models else 'inferred from the profile name'}: {dose_label(float(fallback_dose))}")
    numeric = ("dose_tokens", "loss", "loss_per_token", "n_target_tokens", "n_tokens") + SPAN_ROW_FIELDS
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    # ``loss`` is the whole assistant turn == loss_full; keep both names.
    frame["loss_full"] = frame["loss_full"].where(frame["loss_full"].notna(), frame["loss"])
    frame["n_full_tokens"] = frame["n_full_tokens"].where(frame["n_full_tokens"].notna(), frame["n_target_tokens"])
    if frame["episode_id"].isna().any():
        notes.append(f"{path.name}: {int(frame['episode_id'].isna().sum())} rows lack episode_id — excluded from the paired contrasts and the episode bootstrap")
    frame["model"] = f"{profile}/{arm}"
    return frame[list(ROW_FIELDS) + ["model"]]


def load_losses(inputs: Inputs, models: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str], str]:
    """Every ``losses__*.jsonl`` → one long frame (``ROW_FIELDS`` + ``model``), with the PRIMARY span installed in
    ``loss`` / ``n_target_tokens`` / ``loss_per_token``: ``content`` when every file carries ``loss_content``, else
    ``full`` (the raw ``loss``). Returns (frame, notes, primary_span)."""
    models = models if models is not None else pd.DataFrame(columns=list(MODEL_COLUMNS))
    meta = {(str(r["profile"]), str(r["arm"])): r for _, r in models.iterrows()} if not models.empty else {}
    notes: list[str] = []
    frames = [f for f in (_read_loss_file(path, profile, arm, meta.get((profile, arm)), notes) for (profile, arm), path in sorted(inputs.losses.items())) if f is not None]
    if not frames:
        raise ValueError("no non-empty losses__*.jsonl files")
    long = pd.concat(frames, ignore_index=True)
    for column in ("row_id", "group", "episode_id", "subtype", "profile", "arm", "substrate", "model", "template_md5"):
        long[column] = long[column].astype(object).where(long[column].notna(), None)
    duplicated = long.duplicated(subset=["model", "row_id"], keep="first")
    if duplicated.any():
        notes.append(f"{int(duplicated.sum())} duplicate (model, row_id) rows dropped (first kept)")
        long = long.loc[~duplicated].reset_index(drop=True)
    has_content = long.groupby("model")["loss_content"].apply(lambda s: bool(s.notna().all()))
    if bool(has_content.all()):
        primary_span = "content"
        long["loss"] = long["loss_content"]
        long["n_target_tokens"] = long["n_content_tokens"].where(long["n_content_tokens"].notna(), long["n_target_tokens"])
    else:
        primary_span = "full"
        lacking = sorted(has_content.index[~has_content].tolist())
        notes.append(f"loss_content absent (or incomplete) for {lacking} — falling back to the full assistant-turn loss as the PRIMARY span for every model")
    long["loss_per_token"] = np.where(long["n_target_tokens"] > 0, long["loss"] / long["n_target_tokens"], np.nan)
    for span in SPANS:
        column = SPAN_LOSS[span]
        if long[column].isna().all():
            notes.append(f"span '{span}' ({column}) absent from every file — its readouts NOT RUN")
    unknown = sorted(set(long["group"].dropna()) - set(CLASSES))
    if unknown:
        notes.append(f"rows with unknown group {unknown} kept in the frames but excluded from class readouts")
    return long, notes, primary_span


def model_table(long: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    """One row per scored model: substrate, profile, arm, dose_tokens, n_rows, spans present, hf_path, is_primary_control."""
    rows = []
    meta = {(str(r["profile"]), str(r["arm"])): r for _, r in models.iterrows()} if not models.empty else {}
    for (model, profile, arm, substrate), group in long.groupby(["model", "profile", "arm", "substrate"], sort=True):
        record = meta.get((profile, arm))
        doses = group["dose_tokens"].dropna().unique()
        rows.append({
            "model": model, "substrate": substrate, "profile": profile, "arm": arm,
            "dose_tokens": float(doses[0]) if doses.size else float("nan"), "dose": dose_label(float(doses[0])) if doses.size else "?",
            "n_rows": int(len(group)), "n_classes": int(group["group"].nunique()),
            "spans": ",".join(s for s in SPANS if group[SPAN_LOSS[s]].notna().any()),
            "template_md5": str(group["template_md5"].dropna().iloc[0]) if group["template_md5"].notna().any() else None,
            "role": (record.get("role") if record is not None else None) or arm,
            "hf_repo": record.get("hf_repo") if record is not None else None, "hf_path": record.get("hf_path") if record is not None else None,
            "is_primary_control": bool(record.get("is_primary_control", False)) if record is not None else False,
        })
    frame = pd.DataFrame(rows)
    return _sort_models(frame, extra=["profile"]) if not frame.empty else frame


@dataclass(frozen=True)
class ControlPlan:
    """Which control each treated model is compared against (PREMORTEM §2)."""

    substrate_control: dict[str, str]  # substrate -> model key of the single largest-dose control
    pairs: dict[str, list[tuple[str, str, str]]]  # treated model -> [(control model, control_kind, baseline)]

    def primary_control(self, model: str) -> str | None:
        for control, _kind, baseline in self.pairs.get(model, []):
            if baseline == "primary":
                return control
        return None


def select_controls(model_frame: pd.DataFrame, notes: list[str]) -> ControlPlan:
    """Dose-matched control (same profile) = PRIMARY baseline when scored, the substrate's ``is_primary_control``
    (fallback: largest-dose, noted) control otherwise — both reported when both exist; plus the coin-anchored
    contrast (charter arm vs the coin arm of the same profile) for profiles with both arms."""
    substrate_control: dict[str, str] = {}
    controls = model_frame[model_frame["arm"] == CONTROL_ARM]
    for substrate in order_substrates(model_frame["substrate"]):
        candidates = controls[controls["substrate"] == substrate]
        treated_here = model_frame[(model_frame["substrate"] == substrate) & (model_frame["arm"].isin(TREATED_ARMS))]
        if candidates.empty:
            if not treated_here.empty:
                notes.append(f"{substrate}: no control model scored — only dose-matched/coin-anchored readouts possible for its {len(treated_here)} treated model(s)")
            continue
        flagged = candidates[candidates["is_primary_control"]]
        if len(flagged) == 1:
            chosen = flagged.iloc[0]
        elif len(flagged) > 1:
            chosen = flagged.sort_values("dose_tokens", ascending=False).iloc[0]
            notes.append(f"{substrate}: {len(flagged)} controls flagged is_primary_control — the largest dose ({chosen['profile']}) used as the substrate control")
        else:
            chosen = candidates.sort_values("dose_tokens", ascending=False).iloc[0]
            notes.append(f"{substrate}: no is_primary_control flag in models.json — largest-dose control {chosen['profile']} used as the substrate control")
        substrate_control[substrate] = str(chosen["model"])
    pairs: dict[str, list[tuple[str, str, str]]] = {}
    keys = set(model_frame["model"])
    for _, row in model_frame[model_frame["arm"].isin(TREATED_ARMS)].iterrows():
        model, profile, substrate = str(row["model"]), str(row["profile"]), str(row["substrate"])
        matched = f"{profile}/{CONTROL_ARM}" if f"{profile}/{CONTROL_ARM}" in keys else None
        sub = substrate_control.get(substrate)
        entries: list[tuple[str, str, str]] = []
        if matched is not None:
            entries.append((matched, "dose_matched", "primary"))
            if sub is not None and sub != matched:
                entries.append((sub, "substrate", "secondary"))
        elif sub is not None:
            entries.append((sub, "substrate", "primary"))
            notes.append(f"{model}: no dose-matched control scored — the substrate control {sub} is the primary baseline (midtraining compute not matched)")
        if row["arm"] == "charter" and f"{profile}/coin" in keys:
            entries.append((f"{profile}/coin", "coin_anchor", "anchor"))
        if entries:
            pairs[model] = entries
        else:
            notes.append(f"{model}: no control of any kind — ΔL NOT RUN")
    return ControlPlan(substrate_control=substrate_control, pairs=pairs)


# --------------------------------------------------------------------- ΔL
DELTA_KEYS: tuple[str, ...] = ("substrate", "profile", "arm", "dose_tokens", "model", "control_model", "control_kind", "baseline")
DELTA_ROW: tuple[str, ...] = ("row_id", "group", "episode_id", "subtype", "n_target_tokens")
DELTA_VALUES: tuple[str, ...] = (
    "loss_arm", "loss_control", "delta_loss", "loss_arm_per_token", "loss_control_per_token", "delta_loss_per_token",
    "delta_content", "delta_full", "delta_terminator", "delta_prompt", "n_prompt_tokens",
)
DELTA_COLUMNS: tuple[str, ...] = DELTA_KEYS + DELTA_ROW + DELTA_VALUES


def compute_deltas(long: pd.DataFrame, plan: ControlPlan, notes: list[str]) -> pd.DataFrame:
    """Per treated model × control pair, ΔL = L_arm − L_control on the rows shared with the control (inner join on
    ``row_id``): the primary span in ``delta_loss`` (+ per token), every span in ``delta_<span>``."""
    frames: list[pd.DataFrame] = []
    by_model = {model: group for model, group in long.groupby("model", sort=False)}
    heads = long.drop_duplicates("model").set_index("model")
    control_columns = ["row_id", "group", "n_target_tokens", "loss", "loss_per_token", "template_md5"] + list(SPAN_ROW_FIELDS)
    for model, entries in plan.pairs.items():
        spec = heads.loc[model]
        arm_rows = by_model[model]
        for control_model, kind, baseline in entries:
            control_rows = by_model[control_model][control_columns]
            merged = arm_rows.merge(control_rows, on="row_id", how="inner", suffixes=("", "_control"))
            if len(arm_rows) - len(merged):
                notes.append(f"{model} vs {control_model}: {len(arm_rows) - len(merged)} arm rows without a control loss dropped (inner join on row_id)")
            if len(control_rows) - len(merged):
                notes.append(f"{model} vs {control_model}: {len(control_rows) - len(merged)} control rows without an arm loss ignored")
            if merged.empty:
                continue
            mismatch = int((merged["group"].astype(object) != merged["group_control"].astype(object)).sum())
            if mismatch:
                notes.append(f"{model} vs {control_model}: {mismatch} rows disagree on group between arm and control — arm's group kept")
            tokens = merged["n_target_tokens"].notna() & merged["n_target_tokens_control"].notna() & (merged["n_target_tokens"] != merged["n_target_tokens_control"])
            if tokens.any():
                notes.append(f"{model} vs {control_model}: {int(tokens.sum())} rows differ in n_target_tokens between arm and control (template / tokenizer mismatch?)")
            arm_md5, control_md5 = set(arm_rows["template_md5"].dropna()), set(control_rows["template_md5"].dropna())
            if arm_md5 and control_md5 and arm_md5 != control_md5:
                notes.append(f"{model} vs {control_model}: chat template md5 differs between arm and control ({sorted(arm_md5)} vs {sorted(control_md5)}) — ΔL includes any template effect")
            frame = pd.DataFrame({
                "substrate": spec["substrate"], "profile": spec["profile"], "arm": spec["arm"], "dose_tokens": float(spec["dose_tokens"]) if pd.notna(spec["dose_tokens"]) else float("nan"),
                "model": model, "control_model": control_model, "control_kind": kind, "baseline": baseline,
                "row_id": merged["row_id"].to_numpy(), "group": merged["group"].to_numpy(), "episode_id": merged["episode_id"].to_numpy(), "subtype": merged["subtype"].to_numpy(),
                "n_target_tokens": merged["n_target_tokens"].to_numpy(float),
                "loss_arm": merged["loss"].to_numpy(float), "loss_control": merged["loss_control"].to_numpy(float),
                "loss_arm_per_token": merged["loss_per_token"].to_numpy(float), "loss_control_per_token": merged["loss_per_token_control"].to_numpy(float),
                "n_prompt_tokens": merged["n_prompt_tokens"].to_numpy(float),
            })
            frame["delta_loss"] = frame["loss_arm"] - frame["loss_control"]
            frame["delta_loss_per_token"] = frame["loss_arm_per_token"] - frame["loss_control_per_token"]
            for span in SPANS:
                column = SPAN_LOSS[span]
                frame[f"delta_{span}"] = merged[column].to_numpy(float) - merged[f"{column}_control"].to_numpy(float)
            frames.append(frame[list(DELTA_COLUMNS)])
    if not frames:
        return pd.DataFrame(columns=list(DELTA_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def add_length_residual(delta: pd.DataFrame, positive: str = POSITIVE_CLASS, negative: str = "coin") -> pd.DataFrame:
    """``delta_loss_length_resid``: ΔL residualised on n_target_tokens by OLS fitted on the pooled positive + negative
    rows of each (model, control pair) — the LITERATURE.md length/register confound check (NaN outside the two classes)."""
    if delta.empty:
        return delta.assign(delta_loss_length_resid=pd.Series(dtype=float))
    delta = delta.copy()
    delta["delta_loss_length_resid"] = np.nan
    for _, index in delta.groupby(["model", "control_model"], sort=False).groups.items():
        sub = delta.loc[index]
        pooled = sub[sub["group"].isin([positive, negative])]
        x, y = pooled["n_target_tokens"].to_numpy(float), pooled["delta_loss"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or np.std(x[ok]) == 0:
            continue
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        delta.loc[pooled.index[ok], "delta_loss_length_resid"] = y[ok] - (intercept + slope * x[ok])
    return delta


# ------------------------------------------------------------- bootstrap
@dataclass
class BootPlan:
    """One episode-level resample plan shared by every model, score and span (PREMORTEM §3).

    ``episodes[cls]`` is the canonical episode order of class ``cls`` (the episodes for which *every* model in the
    run has that row — the intersection, so identical indices address identical episodes everywhere);
    ``indices[type]`` is the (n_boot, n_episodes) resample matrix of the conflict episodes (coin + charter rows)
    or the agreement episodes (ambiguous + wrong rows). Classes of the same episode type share indices, so a
    paired contrast resamples both sides of an episode together.
    """

    n_boot: int
    seed: int
    episodes: dict[str, np.ndarray]
    indices: dict[str, np.ndarray]
    dropped: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, long: pd.DataFrame, n_boot: int, seed: int, models: Iterable[str] | None = None) -> "BootPlan":
        models = list(models) if models is not None else sorted(long["model"].dropna().unique().tolist())
        frame = long[long["model"].isin(models) & long["group"].isin(CLASSES) & long["episode_id"].notna()]
        episodes: dict[str, np.ndarray] = {}
        dropped: dict[str, int] = {}
        universe: dict[str, set] = {}
        for cls_name in CLASSES:
            sub = frame[frame["group"] == cls_name]
            present = [set(g["episode_id"]) for _, g in sub.groupby("model", sort=False)]
            common = set.intersection(*present) if present else set()
            universe[cls_name] = common
            dropped[cls_name] = int(len(set(sub["episode_id"])) - len(common))
        # Conflict classes share the episode set (and the resample); agreement classes likewise.
        for episode_type, members in (("conflict", CONFLICT_CLASSES), ("agreement", AGREEMENT_CLASSES)):
            shared = set.intersection(*(universe[m] for m in members if universe[m])) if any(universe[m] for m in members) else set()
            if not shared:  # one class of the pair missing everywhere: keep whatever the other has
                shared = set.union(*(universe[m] for m in members))
            for m in members:
                episodes[m] = np.array(sorted(shared), dtype=object)
        rng = np.random.default_rng(seed)
        indices = {t: rng.integers(0, max(1, episodes[m].size), size=(int(n_boot), episodes[m].size)) if episodes[m].size else np.zeros((int(n_boot), 0), dtype=int) for t, m in (("conflict", "coin"), ("agreement", "ambiguous"))}
        return cls(n_boot=int(n_boot), seed=int(seed), episodes=episodes, indices=indices, dropped=dropped)

    def align(self, frame: pd.DataFrame, cls_name: str, column: str) -> np.ndarray:
        """Values of ``column`` for class ``cls_name`` in canonical episode order (NaN where the frame lacks the episode)."""
        sub = frame[frame["group"] == cls_name].drop_duplicates("episode_id").set_index("episode_id")[column]
        return sub.reindex(self.episodes[cls_name]).to_numpy(dtype=float)

    def resample(self, values: np.ndarray, cls_name: str) -> np.ndarray:
        """(n_boot, n) matrix of resampled values of a class-aligned vector."""
        idx = self.indices[EPISODE_TYPE[cls_name]]
        return values[idx] if values.size else np.zeros((self.n_boot, 0))


def auc_lower_positive(pos: Sequence[float] | np.ndarray, neg: Sequence[float] | np.ndarray) -> float:
    """AUC of the rule "lower score → positive class": P(pos < neg) + ½·P(pos = neg). NaN when a class is empty."""
    pos = _finite(pos)
    neg = np.sort(_finite(neg))
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    left = np.searchsorted(neg, pos, side="left")
    right = np.searchsorted(neg, pos, side="right")
    return float(((neg.size - right).sum() + 0.5 * (right - left).sum()) / (pos.size * neg.size))


def cliffs_delta_from_auc(auc: float) -> float:
    """Two-class Cliff's δ = P(pos < neg) − P(pos > neg) = 2·AUC − 1 (rank-based effect size)."""
    return 2.0 * auc - 1.0 if np.isfinite(auc) else float("nan")


def boot_auc(pos: np.ndarray, neg: np.ndarray, plan: BootPlan, pos_cls: str, neg_cls: str) -> np.ndarray:
    """Bootstrap AUCs over the plan's shared episode resamples (NaN-episodes dropped per replicate)."""
    if plan.n_boot <= 0 or pos.size < 2 or neg.size < 2:
        return np.full(max(plan.n_boot, 0), np.nan)
    P, N = plan.resample(pos, pos_cls), plan.resample(neg, neg_cls)
    return np.array([auc_lower_positive(P[b], N[b]) for b in range(plan.n_boot)])


def _quantiles(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    values = _finite(values)
    if values.size == 0:
        return (float("nan"), float("nan"))
    alpha = (1.0 - level) / 2.0
    return (float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha)))


def auc_record(pos: np.ndarray, neg: np.ndarray, boots: np.ndarray) -> dict[str, Any]:
    auc = auc_lower_positive(pos, neg)
    low, high = _quantiles(boots)
    direction = "NO DATA" if not np.isfinite(auc) else ("lower score → positive (expected)" if auc > 0.5 else ("higher score → positive (reversed)" if auc < 0.5 else "no separation"))
    return {
        "n_pos": int(_finite(pos).size), "n_neg": int(_finite(neg).size), "auc": auc, "ci_low": low, "ci_high": high,
        "cliffs_delta": cliffs_delta_from_auc(auc), "cliffs_ci_low": cliffs_delta_from_auc(low), "cliffs_ci_high": cliffs_delta_from_auc(high),
        "auc_raw_higher_is_positive": 1.0 - auc if np.isfinite(auc) else float("nan"), "direction": direction,
    }


class BootStore(dict):
    """(model, control_model, score, comparison) -> bootstrap AUC array; ``difference`` gives paired CIs."""

    def difference(self, key_a: tuple, key_b: tuple) -> tuple[float, float, bool]:
        a, b = self.get(key_a), self.get(key_b)
        if a is None or b is None or a.size == 0 or b.size == 0:
            return (float("nan"), float("nan"), False)
        low, high = _quantiles(a - b)
        return (low, high, True)


AUC_COLUMNS: tuple[str, ...] = (
    "substrate", "profile", "arm", "dose_tokens", "dose", "model", "control_model", "control_kind", "baseline", "score", "span", "comparison", "positive", "negative",
    "n_pos", "n_neg", "auc", "ci_low", "ci_high", "cliffs_delta", "cliffs_ci_low", "cliffs_ci_high", "auc_raw_higher_is_positive", "direction",
)
SCORE_SPAN: dict[str, str] = {"delta_loss": "primary", "delta_loss_per_token": "primary", "delta_loss_length_resid": "primary", "loss": "primary", "loss_per_token": "primary", "delta_content": "content", "delta_full": "full", "delta_terminator": "terminator", "delta_prompt": "prompt"}
SCORE_COLUMN: dict[str, str] = {"loss": "loss_arm", "loss_per_token": "loss_arm_per_token"}


def auc_table(delta: pd.DataFrame, long: pd.DataFrame, plan: BootPlan, control_plan: ControlPlan, store: BootStore | None = None) -> pd.DataFrame:
    """AUC (lower → ambiguous) + Cliff's δ for every treated model × control pair × score × comparison, and the
    control models' own-loss baselines (L_control alone). Bootstrap arrays go into ``store``."""
    store = store if store is not None else BootStore()
    records: list[dict[str, Any]] = []
    if not delta.empty:
        for (model, control_model, kind, baseline), group in delta.groupby(["model", "control_model", "control_kind", "baseline"], sort=False):
            head = group.iloc[0]
            for score in DELTA_SCORES:
                column = SCORE_COLUMN.get(score, score)
                if column not in group.columns or group[column].isna().all():
                    continue
                for comparison, (positive, negative) in COMPARISONS.items():
                    pos, neg = plan.align(group, positive, column), plan.align(group, negative, column)
                    boots = boot_auc(pos, neg, plan, positive, negative)
                    store[(model, control_model, score, comparison)] = boots
                    records.append({
                        "substrate": head["substrate"], "profile": head["profile"], "arm": head["arm"], "dose_tokens": head["dose_tokens"], "dose": dose_label(head["dose_tokens"]),
                        "model": model, "control_model": control_model, "control_kind": kind, "baseline": baseline, "score": score, "span": SCORE_SPAN.get(score, "primary"),
                        "comparison": comparison, "positive": positive, "negative": negative, **auc_record(pos, neg, boots),
                    })
    for model, group in long[long["arm"] == CONTROL_ARM].groupby("model", sort=False):
        head = group.iloc[0]
        kind = "substrate" if model in set(control_plan.substrate_control.values()) else "dose_matched"
        for score in CONTROL_SCORES:
            for comparison, (positive, negative) in COMPARISONS.items():
                pos, neg = plan.align(group, positive, score), plan.align(group, negative, score)
                boots = boot_auc(pos, neg, plan, positive, negative)
                store[(model, model, score, comparison)] = boots
                records.append({
                    "substrate": head["substrate"], "profile": head["profile"], "arm": CONTROL_ARM, "dose_tokens": head["dose_tokens"], "dose": dose_label(head["dose_tokens"]),
                    "model": model, "control_model": model, "control_kind": kind, "baseline": "control_alone", "score": score, "span": "primary",
                    "comparison": comparison, "positive": positive, "negative": negative, **auc_record(pos, neg, boots),
                })
    return _sort_models(pd.DataFrame(records, columns=list(AUC_COLUMNS)))


# ------------------------------------------------------------------- sieve
SIEVE_COLUMNS: tuple[str, ...] = (
    "substrate", "profile", "arm", "dose_tokens", "dose", "model", "control_model", "control_kind", "baseline", "negative", "n_pos", "n_neg",
    "f_negative_pass_through", "empirical", "estimate", "n_negative_remaining", "threshold_delta_loss", "positive_fraction_kept", "kept_ci_low", "kept_ci_high",
    "required_multiplier", "multiplier_ci_low", "multiplier_ci_high", "enrichment", "enrichment_ci_low", "enrichment_ci_high",
    "power_law_alpha", "power_law_intercept", "multiplier_power_law_tail", "enrichment_power_law_tail", "multiplier_student_t",
)


def sieve_table(pos: np.ndarray, neg: np.ndarray, plan: BootPlan | None = None, pos_cls: str = POSITIVE_CLASS, neg_cls: str = "coin", fractions: Sequence[float] = FRACTIONS) -> pd.DataFrame:
    """sieve_followup's table for one model: keep rows with ΔL ≤ τ, τ = the f-quantile of the negative class.
    Empirical (TPR, CI over the shared episode bootstrap, 1/TPR, TPR/f) for f ≥ 0.02; the power-law tail
    (log TPR = a + α log f on the empirical points with f ≤ 0.1) supplies the smaller fractions, flagged."""
    fractions = np.asarray(sorted(set(float(f) for f in fractions), reverse=True), dtype=float)
    empirical = fractions >= SIEVE_EMPIRICAL_MIN_F - 1e-12
    columns = [c for c in SIEVE_COLUMNS if c not in ("substrate", "profile", "arm", "dose_tokens", "dose", "model", "control_model", "control_kind", "baseline", "negative", "n_pos", "n_neg")]
    pos_all, neg_all = _finite(pos), _finite(neg)
    if pos_all.size == 0 or neg_all.size == 0:
        return pd.DataFrame(columns=columns)
    f_emp = fractions[empirical]
    tpr_emp, taus_emp = S.tpr_at_fpr(pos_all, neg_all, f_emp)
    lo = hi = np.full(f_emp.size, np.nan)
    if plan is not None and plan.n_boot > 0 and pos.size >= 2 and neg.size >= 2:
        P, N = plan.resample(pos, pos_cls), plan.resample(neg, neg_cls)
        boot = np.full((plan.n_boot, f_emp.size), np.nan)
        for b in range(plan.n_boot):
            p, n = _finite(P[b]), _finite(N[b])
            if p.size and n.size:
                boot[b], _ = S.tpr_at_fpr(p, n, f_emp)
        lo, hi = np.nanquantile(boot, [0.025, 0.975], axis=0)
    mask = (f_emp <= POWER_LAW_F_MAX) & (tpr_emp > 0)
    alpha, intercept = (S.power_law_tail(f_emp, tpr_emp, f_max=POWER_LAW_F_MAX) if mask.sum() >= 2 else (float("nan"), float("nan")))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tpr_pl = np.exp(intercept) * fractions ** alpha if np.isfinite(alpha) else np.full(fractions.size, np.nan)
    tpr, taus, kept_lo, kept_hi = (np.full(fractions.size, np.nan) for _ in range(4))
    tpr[empirical], taus[empirical], kept_lo[empirical], kept_hi[empirical] = tpr_emp, taus_emp, lo, hi
    with np.errstate(divide="ignore", invalid="ignore"):
        mult, mult_lo, mult_hi = 1.0 / tpr, 1.0 / kept_hi, 1.0 / kept_lo
        enrichment, enrichment_lo, enrichment_hi = tpr / fractions, kept_lo / fractions, kept_hi / fractions
        mult_pl, enrichment_pl = 1.0 / tpr_pl, tpr_pl / fractions
    mult_t = np.full(fractions.size, np.nan)
    if pos_all.size >= 5 and neg_all.size >= 5 and pos_all.std() > 0 and neg_all.std() > 0:
        try:
            tpr_t, _ = S.student_t_extrapolation(pos_all, neg_all, fractions)
            with np.errstate(divide="ignore", invalid="ignore"):
                mult_t = 1.0 / np.asarray(tpr_t, dtype=float)
        except Exception:  # scipy missing or the fit failed: the column stays NaN (a model, not data)
            pass
    return pd.DataFrame({
        "f_negative_pass_through": fractions, "empirical": empirical, "estimate": np.where(empirical, "empirical", "power-law extrapolation"),
        "n_negative_remaining": fractions * neg_all.size, "threshold_delta_loss": taus, "positive_fraction_kept": tpr, "kept_ci_low": kept_lo, "kept_ci_high": kept_hi,
        "required_multiplier": mult, "multiplier_ci_low": mult_lo, "multiplier_ci_high": mult_hi, "enrichment": enrichment, "enrichment_ci_low": enrichment_lo, "enrichment_ci_high": enrichment_hi,
        "power_law_alpha": alpha, "power_law_intercept": intercept, "multiplier_power_law_tail": mult_pl, "enrichment_power_law_tail": enrichment_pl, "multiplier_student_t": mult_t,
    }, columns=columns)


def sieve_tables(delta: pd.DataFrame, plan: BootPlan, fractions: Sequence[float] = FRACTIONS, score: str = PRIMARY_SCORE) -> pd.DataFrame:
    """One sieve table per treated model × control pair × negative class (coin for every arm; charter too for coin arms)."""
    frames: list[pd.DataFrame] = []
    if delta.empty:
        return pd.DataFrame(columns=list(SIEVE_COLUMNS))
    for (model, control_model, kind, baseline), group in delta.groupby(["model", "control_model", "control_kind", "baseline"], sort=False):
        head = group.iloc[0]
        pos = plan.align(group, POSITIVE_CLASS, score)
        for negative in SIEVE_NEGATIVES.get(str(head["arm"]), ("coin",)):
            neg = plan.align(group, negative, score)
            table = sieve_table(pos, neg, plan, POSITIVE_CLASS, negative, fractions)
            if table.empty:
                continue
            meta = {"substrate": head["substrate"], "profile": head["profile"], "arm": head["arm"], "dose_tokens": head["dose_tokens"], "dose": dose_label(head["dose_tokens"]), "model": model, "control_model": control_model, "control_kind": kind, "baseline": baseline, "negative": negative, "n_pos": int(_finite(pos).size), "n_neg": int(_finite(neg).size)}
            for column, value in reversed(list(meta.items())):
                table.insert(0, column, value)
            frames.append(table[list(SIEVE_COLUMNS)])
    if not frames:
        return pd.DataFrame(columns=list(SIEVE_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    out["_f"] = -out["f_negative_pass_through"].astype(float)
    return _sort_models(out, extra=["_f"]).drop(columns=["_f"])


# -------------------------------------------------------- paired contrasts
CONTRAST_COLUMNS: tuple[str, ...] = (
    "substrate", "profile", "arm", "dose_tokens", "dose", "model", "control_model", "control_kind", "baseline", "norm", "contrast", "n", "mean", "ci_low", "ci_high", "median",
    "sd", "cliffs_delta", "paired_sign_delta", "frac_positive", "sign_p", "n_pos", "n_neg", "n_zero", "expected_sign", "preregistered", "verdict",
)
CONTRAST_NORMS: dict[str, str] = {"neg_delta_loss": "delta_loss", "neg_delta_loss_per_token": "delta_loss_per_token"}


def paired_contrasts(delta: pd.DataFrame, plan: BootPlan) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-episode paired contrasts of −ΔL (positive = the arm lowers the row's loss): coin − charter over conflict
    episodes, ambiguous − wrong over agreement episodes. Mean with the shared episode-bootstrap CI, median, exact
    two-sided sign test, Cliff's δ between the two sides (rank-based) and the paired sign δ = P(>0) − P(<0).
    Returns (per-episode values, summary with verdicts against the pre-registered signs)."""
    if delta.empty:
        return pd.DataFrame(), pd.DataFrame(columns=list(CONTRAST_COLUMNS))
    per_rows: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for (model, control_model, kind, baseline), group in delta.groupby(["model", "control_model", "control_kind", "baseline"], sort=False):
        head = group.iloc[0]
        for norm, column in CONTRAST_NORMS.items():
            for contrast, (group_a, group_b) in CONTRASTS.items():
                a, b = plan.align(group, group_a, column), plan.align(group, group_b, column)
                values = -(a - b)  # −ΔL_a − (−ΔL_b)
                ok = np.isfinite(values)
                if ok.sum() == 0:
                    continue
                per_rows.append(pd.DataFrame({"model": model, "control_model": control_model, "control_kind": kind, "baseline": baseline, "norm": norm, "contrast": contrast, "episode_id": plan.episodes[group_a][ok], "value": values[ok]}))
                boots = plan.resample(np.where(ok, values, np.nan), group_a)
                means = np.nanmean(boots, axis=1) if boots.size else np.array([])
                low, high = _quantiles(means)
                signs = A.sign_test(values[ok])
                expected = EXPECTED_SIGN.get(contrast, {}).get(str(head["arm"]))
                records.append({
                    "substrate": head["substrate"], "profile": head["profile"], "arm": head["arm"], "dose_tokens": head["dose_tokens"], "dose": dose_label(head["dose_tokens"]), "model": model, "control_model": control_model, "control_kind": kind, "baseline": baseline,
                    "norm": norm, "contrast": contrast, "n": int(ok.sum()), "mean": float(values[ok].mean()), "ci_low": low, "ci_high": high, "median": float(np.median(values[ok])), "sd": float(values[ok].std(ddof=1)) if ok.sum() > 1 else float("nan"),
                    "cliffs_delta": A.cliffs_delta(-a[ok], -b[ok]), "paired_sign_delta": (signs["n_pos"] - signs["n_neg"]) / (signs["n_pos"] + signs["n_neg"]) if signs["n_pos"] + signs["n_neg"] else float("nan"),
                    "frac_positive": signs["frac_positive"], "sign_p": signs["p_value"], "n_pos": signs["n_pos"], "n_neg": signs["n_neg"], "n_zero": signs["n_zero"],
                    "expected_sign": expected, "preregistered": contrast in PREREGISTERED_CONTRASTS, "verdict": A._verdict(float(values[ok].mean()), low, high, int(expected)) if expected is not None else "INFO",
                })
    per_episode = pd.concat(per_rows, ignore_index=True) if per_rows else pd.DataFrame()
    return per_episode, _sort_models(pd.DataFrame(records, columns=list(CONTRAST_COLUMNS)))


# ------------------------------------------------------------- class means
CLASS_MEAN_COLUMNS: tuple[str, ...] = (
    "substrate", "profile", "arm", "dose_tokens", "dose", "model", "control_model", "control_kind", "baseline", "class", "n",
    "mean_loss_arm", "median_loss_arm", "mean_loss_control", "median_loss_control", "mean_delta_loss", "delta_ci_low", "delta_ci_high", "median_delta_loss", "sd_delta_loss", "frac_delta_negative",
    "mean_delta_loss_per_token", "median_delta_loss_per_token", "mean_delta_content", "mean_delta_full", "mean_delta_terminator", "mean_delta_prompt", "mean_n_target_tokens",
)


def class_means(delta: pd.DataFrame, long: pd.DataFrame, plan: BootPlan) -> pd.DataFrame:
    """Class means / medians of L_arm, L_control and ΔL (every span) per treated model × control pair, shared-bootstrap
    CI on the mean ΔL; control models list their own class means of L."""
    records: list[dict[str, Any]] = []
    if not delta.empty:
        for (model, control_model, kind, baseline, cls_name), group in delta.groupby(["model", "control_model", "control_kind", "baseline", "group"], sort=False):
            if cls_name not in CLASSES:
                continue
            head = group.iloc[0]
            d = group["delta_loss"].to_numpy(float)
            aligned = plan.align(group, cls_name, "delta_loss")
            means = np.nanmean(plan.resample(aligned, cls_name), axis=1) if plan.n_boot > 0 and aligned.size else np.array([])
            low, high = _quantiles(means)
            records.append({
                "substrate": head["substrate"], "profile": head["profile"], "arm": head["arm"], "dose_tokens": head["dose_tokens"], "dose": dose_label(head["dose_tokens"]), "model": model, "control_model": control_model, "control_kind": kind, "baseline": baseline, "class": cls_name, "n": int(len(group)),
                "mean_loss_arm": float(np.nanmean(group["loss_arm"])), "median_loss_arm": float(np.nanmedian(group["loss_arm"])), "mean_loss_control": float(np.nanmean(group["loss_control"])), "median_loss_control": float(np.nanmedian(group["loss_control"])),
                "mean_delta_loss": float(np.nanmean(d)), "delta_ci_low": low, "delta_ci_high": high, "median_delta_loss": float(np.nanmedian(d)), "sd_delta_loss": float(np.nanstd(d, ddof=1)) if _finite(d).size > 1 else float("nan"),
                "frac_delta_negative": float(np.mean(_finite(d) < 0)) if _finite(d).size else float("nan"),
                "mean_delta_loss_per_token": float(np.nanmean(group["delta_loss_per_token"])), "median_delta_loss_per_token": float(np.nanmedian(group["delta_loss_per_token"])),
                **{f"mean_delta_{span}": (float(np.nanmean(group[f"delta_{span}"])) if group[f"delta_{span}"].notna().any() else float("nan")) for span in SPANS},
                "mean_n_target_tokens": float(np.nanmean(group["n_target_tokens"])),
            })
    for (model, cls_name), group in long[long["arm"] == CONTROL_ARM].groupby(["model", "group"], sort=False):
        if cls_name not in CLASSES:
            continue
        head = group.iloc[0]
        records.append({
            "substrate": head["substrate"], "profile": head["profile"], "arm": CONTROL_ARM, "dose_tokens": head["dose_tokens"], "dose": dose_label(head["dose_tokens"]), "model": model, "control_model": model, "control_kind": "control", "baseline": "control_alone", "class": cls_name, "n": int(len(group)),
            "mean_loss_arm": float("nan"), "median_loss_arm": float("nan"), "mean_loss_control": float(np.nanmean(group["loss"])), "median_loss_control": float(np.nanmedian(group["loss"])),
            "mean_delta_loss": float("nan"), "delta_ci_low": float("nan"), "delta_ci_high": float("nan"), "median_delta_loss": float("nan"), "sd_delta_loss": float("nan"), "frac_delta_negative": float("nan"),
            "mean_delta_loss_per_token": float("nan"), "median_delta_loss_per_token": float("nan"), **{f"mean_delta_{span}": float("nan") for span in SPANS}, "mean_n_target_tokens": float(np.nanmean(group["n_target_tokens"])),
        })
    frame = pd.DataFrame(records, columns=list(CLASS_MEAN_COLUMNS))
    if frame.empty:
        return frame
    frame["_c"] = frame["class"].map({c: i for i, c in enumerate(CLASSES)})
    return _sort_models(frame, extra=["_c"]).drop(columns=["_c"])


# ------------------------------------------- within-model class contrasts
WITHIN_COLUMNS: tuple[str, ...] = ("substrate", "profile", "arm", "dose_tokens", "dose", "model", "class_a", "class_b", "n_a", "n_b", "mean_a", "mean_b", "mean_difference", "ci_low", "ci_high", "cliffs_delta", "auc_lower_is_a", "auc_ci_low", "auc_ci_high")


def within_model_contrasts(long: pd.DataFrame, plan: BootPlan) -> pd.DataFrame:
    """Control-free, within-model class contrasts of the model's own loss L (primary span): mean L(a) − mean L(b)
    with the shared-bootstrap CI, Cliff's δ and AUC(lower L → a) for (coin, charter), (ambiguous, coin),
    (ambiguous, charter), (ambiguous, wrong) — per model, sortable vs dose (PREMORTEM §2)."""
    records: list[dict[str, Any]] = []
    for model, group in long[long["group"].isin(CLASSES)].groupby("model", sort=False):
        head = group.iloc[0]
        for class_a, class_b in WITHIN_MODEL_PAIRS:
            a, b = plan.align(group, class_a, "loss"), plan.align(group, class_b, "loss")
            if _finite(a).size == 0 or _finite(b).size == 0:
                continue
            A_, B_ = plan.resample(a, class_a), plan.resample(b, class_b)
            diffs = (np.nanmean(A_, axis=1) - np.nanmean(B_, axis=1)) if plan.n_boot > 0 and A_.size and B_.size else np.array([])
            low, high = _quantiles(diffs)
            boots = boot_auc(a, b, plan, class_a, class_b)
            auc = auc_lower_positive(a, b)
            auc_low, auc_high = _quantiles(boots)
            records.append({
                "substrate": head["substrate"], "profile": head["profile"], "arm": head["arm"], "dose_tokens": head["dose_tokens"], "dose": dose_label(head["dose_tokens"]), "model": model, "class_a": class_a, "class_b": class_b,
                "n_a": int(_finite(a).size), "n_b": int(_finite(b).size), "mean_a": float(np.nanmean(a)), "mean_b": float(np.nanmean(b)), "mean_difference": float(np.nanmean(a) - np.nanmean(b)), "ci_low": low, "ci_high": high,
                "cliffs_delta": A.cliffs_delta(_finite(a), _finite(b)), "auc_lower_is_a": auc, "auc_ci_low": auc_low, "auc_ci_high": auc_high,
            })
    return _sort_models(pd.DataFrame(records, columns=list(WITHIN_COLUMNS)))


# ------------------------------------------------------------- noise floor
NOISE_ROW_COLUMNS: tuple[str, ...] = ("model", "row_id", "group", "n_repeats", "mean_abs_loss", "abs_spread", "rel_spread")
NOISE_COLUMNS: tuple[str, ...] = (
    "substrate", "profile", "arm", "dose_tokens", "dose", "model", "n_rows", "median_rel_spread", "p90_rel_spread", "max_rel_spread",
    "median_abs_spread", "between_row_sd_loss", "median_spread_over_sd", "class_gap_delta_loss", "median_spread_over_class_gap", "verdict",
)


def noise_floor(inputs: Inputs, long: pd.DataFrame, delta: pd.DataFrame, notes: list[str], primary_span: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run-to-run spread from repeat-scored rows (``scores/noise__<profile>__<arm>.jsonl`` + the same rows' main-pass
    loss, primary span): per row ``rel_spread = (max − min) / mean|loss|``; per model the median / p90 / max, the
    spread relative to the between-row SD of L and to |mean ΔL(coin) − mean ΔL(ambiguous)| of the model's primary
    ΔL. PASS = median ≤ 2 % (FLAG on p90 > 10 %), as in the v1 / graft gates."""
    if not inputs.noise:
        notes.append("no scores/noise__<profile>__<arm>.jsonl — noise floor NOT RUN")
        return pd.DataFrame(columns=list(NOISE_ROW_COLUMNS)), pd.DataFrame(columns=list(NOISE_COLUMNS))
    main_by_model = {model: group for model, group in long.groupby("model", sort=False)}
    gaps: dict[str, float] = {}
    if not delta.empty:
        for model, group in delta[delta["baseline"] == "primary"].groupby("model", sort=False):
            coin, amb = _finite(group.loc[group["group"] == "coin", "delta_loss"]), _finite(group.loc[group["group"] == POSITIVE_CLASS, "delta_loss"])
            gaps[model] = float(abs(coin.mean() - amb.mean())) if coin.size and amb.size else float("nan")
    per_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for (profile, arm), path in sorted(inputs.noise.items()):
        model = f"{profile}/{arm}"
        records = A.read_jsonl(path)
        if not records:
            notes.append(f"{path.name}: empty — skipped")
            continue
        noise = pd.DataFrame.from_records(records)
        if "row_id" not in noise.columns or "loss" not in noise.columns:
            notes.append(f"{path.name}: rows lack row_id/loss — skipped")
            continue
        main = main_by_model.get(model)
        loss_column = SPAN_LOSS[primary_span] if primary_span != "full" and SPAN_LOSS[primary_span] in noise.columns and noise[SPAN_LOSS[primary_span]].notna().all() else "loss"
        if primary_span != "full" and loss_column == "loss":
            notes.append(f"{path.name}: no {SPAN_LOSS[primary_span]} in the repeat file — its spread is measured on the full-turn loss")
        noise[loss_column] = pd.to_numeric(noise[loss_column], errors="coerce")
        main_losses = main.set_index("row_id")["loss"].to_dict() if main is not None else {}
        main_groups = main.set_index("row_id")["group"].to_dict() if main is not None else {}
        rows_here: list[dict[str, Any]] = []
        for row_id, group in noise.groupby("row_id", sort=True):
            values = group[loss_column].to_numpy(float).tolist()
            if row_id in main_losses and np.isfinite(main_losses[row_id]):
                values.append(float(main_losses[row_id]))
            values = _finite(values)
            if values.size < 2:
                continue
            mean_abs, spread = float(np.abs(values).mean()), float(values.max() - values.min())
            cls_name = main_groups.get(row_id) or (group["group"].iloc[0] if "group" in group.columns else None)
            rows_here.append({"model": model, "row_id": row_id, "group": cls_name, "n_repeats": int(values.size), "mean_abs_loss": mean_abs, "abs_spread": spread, "rel_spread": spread / mean_abs if mean_abs > 0 else float("nan")})
        if not rows_here:
            notes.append(f"{path.name}: no row scored ≥ 2 times (with the main pass) — skipped")
            continue
        per_rows += rows_here
        rel, abs_spread = _finite([r["rel_spread"] for r in rows_here]), _finite([r["abs_spread"] for r in rows_here])
        between_sd = float(np.nanstd(main["loss"].to_numpy(float), ddof=1)) if main is not None and len(main) > 1 else float("nan")
        median_rel, p90 = (float(np.median(rel)), float(np.quantile(rel, 0.9))) if rel.size else (float("nan"), float("nan"))
        median_abs = float(np.median(abs_spread)) if abs_spread.size else float("nan")
        gap = gaps.get(model, float("nan"))
        verdict = "NOT RUN" if not np.isfinite(median_rel) else ("FAIL" if median_rel > NOISE_MEDIAN_REL_MAX else ("PASS" if not (np.isfinite(p90) and p90 > NOISE_P90_REL_MAX) else "PASS (p90 FLAG)"))
        head = main.iloc[0] if main is not None else None
        if head is None:
            notes.append(f"{path.name}: no matching losses__{profile}__{arm}.jsonl — noise summarised from the repeat file alone")
        substrate = head["substrate"] if head is not None else infer_substrate(profile)
        dose = float(head["dose_tokens"]) if head is not None and pd.notna(head["dose_tokens"]) else float(infer_dose_tokens(profile) or float("nan"))
        summaries.append({
            "substrate": substrate, "profile": profile, "arm": arm, "dose_tokens": dose, "dose": dose_label(dose), "model": model, "n_rows": int(len(rows_here)),
            "median_rel_spread": median_rel, "p90_rel_spread": p90, "max_rel_spread": float(rel.max()) if rel.size else float("nan"), "median_abs_spread": median_abs, "between_row_sd_loss": between_sd,
            "median_spread_over_sd": median_abs / between_sd if np.isfinite(between_sd) and between_sd > 0 else float("nan"), "class_gap_delta_loss": gap, "median_spread_over_class_gap": median_abs / gap if np.isfinite(gap) and gap > 0 else float("nan"), "verdict": verdict,
        })
    per_row = pd.DataFrame(per_rows, columns=list(NOISE_ROW_COLUMNS))
    summary = pd.DataFrame(summaries, columns=list(NOISE_COLUMNS))
    return per_row, (_sort_models(summary) if not summary.empty else summary)


# --------------------------------------------------------- length confound
LENGTH_COLUMNS: tuple[str, ...] = (
    "substrate", "profile", "arm", "dose_tokens", "dose", "model", "control_model", "control_kind", "baseline",
    "spearman_ambiguous", "spearman_coin", "spearman_charter", "spearman_ambiguous_wrong", "n_ambiguous", "n_coin", "mean_tokens_ambiguous", "mean_tokens_coin", "spearman_tokens_vs_class_pooled",
    "auc_delta_loss", "auc_delta_loss_per_token", "auc_delta_loss_length_resid", "resid_ci_low", "resid_ci_high", "ols_slope_per_token", "auc_shift_after_residualising", "flag",
)


def length_confound(delta: pd.DataFrame, aucs: pd.DataFrame) -> pd.DataFrame:
    """LITERATURE.md check: within-class Spearman(ΔL, n_target_tokens) per model, and the ambiguous-vs-coin AUC on
    raw ΔL, on ΔL per token and on ΔL residualised on n_target_tokens (OLS on the pooled two classes). ``flag``
    when residualising moves the AUC by more than 0.05 (the separation may ride on length / register)."""
    records: list[dict[str, Any]] = []
    if delta.empty:
        return pd.DataFrame(columns=list(LENGTH_COLUMNS))
    for (model, control_model, kind, baseline), group in delta.groupby(["model", "control_model", "control_kind", "baseline"], sort=False):
        head = group.iloc[0]
        record: dict[str, Any] = {"substrate": head["substrate"], "profile": head["profile"], "arm": head["arm"], "dose_tokens": head["dose_tokens"], "dose": dose_label(head["dose_tokens"]), "model": model, "control_model": control_model, "control_kind": kind, "baseline": baseline}
        for cls_name in CLASSES:
            sub = group[group["group"] == cls_name]
            record[f"spearman_{cls_name}"] = A.spearman(sub["n_target_tokens"].to_numpy(float), sub["delta_loss"].to_numpy(float)) if len(sub) >= 3 else float("nan")
        amb, coin = group[group["group"] == POSITIVE_CLASS], group[group["group"] == "coin"]
        record.update({"n_ambiguous": int(len(amb)), "n_coin": int(len(coin)), "mean_tokens_ambiguous": float(np.nanmean(amb["n_target_tokens"])) if len(amb) else float("nan"), "mean_tokens_coin": float(np.nanmean(coin["n_target_tokens"])) if len(coin) else float("nan")})
        pooled = pd.concat([amb, coin])
        record["spearman_tokens_vs_class_pooled"] = A.spearman(pooled["n_target_tokens"].to_numpy(float), (pooled["group"] == POSITIVE_CLASS).to_numpy(float)) if len(pooled) >= 3 else float("nan")
        x, y = pooled["n_target_tokens"].to_numpy(float), pooled["delta_loss"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        record["ols_slope_per_token"] = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 and np.std(x[ok]) > 0 else float("nan")
        for score, name in (("delta_loss", "auc_delta_loss"), ("delta_loss_per_token", "auc_delta_loss_per_token"), ("delta_loss_length_resid", "auc_delta_loss_length_resid")):
            row = aucs[(aucs["model"] == model) & (aucs["control_model"] == control_model) & (aucs["score"] == score) & (aucs["comparison"] == PRIMARY_COMPARISON)] if not aucs.empty else pd.DataFrame()
            record[name] = float(row.iloc[0]["auc"]) if not row.empty else float("nan")
            if score == "delta_loss_length_resid":
                record["resid_ci_low"], record["resid_ci_high"] = (float(row.iloc[0]["ci_low"]), float(row.iloc[0]["ci_high"])) if not row.empty else (float("nan"), float("nan"))
        shift = record["auc_delta_loss_length_resid"] - record["auc_delta_loss"]
        record["auc_shift_after_residualising"] = shift
        record["flag"] = bool(np.isfinite(shift) and abs(shift) > 0.05)
        records.append(record)
    return _sort_models(pd.DataFrame(records, columns=list(LENGTH_COLUMNS)))


# ----------------------------------------------------------- scaling tables
def _lookup(aucs: pd.DataFrame, model: str, score: str, comparison: str, control_model: str | None = None) -> pd.Series | None:
    if aucs.empty:
        return None
    sub = aucs[(aucs["model"] == model) & (aucs["score"] == score) & (aucs["comparison"] == comparison)]
    if control_model is not None:
        sub = sub[sub["control_model"] == control_model]
    return sub.iloc[0] if not sub.empty else None


def _get(row: pd.Series | None, column: str, default: Any = float("nan")) -> Any:
    return row[column] if row is not None else default


SPAN_COLUMNS: tuple[str, ...] = (
    "substrate", "profile", "arm", "dose_tokens", "dose", "model", "control_model", "control_kind", "baseline", "comparison", "n_pos", "n_neg",
    "auc_content", "content_ci_low", "content_ci_high", "auc_full", "full_ci_low", "full_ci_high", "auc_terminator", "terminator_ci_low", "terminator_ci_high",
    "auc_prompt", "prompt_ci_low", "prompt_ci_high", "prompt_cliffs_delta", "negative_control_flag", "negative_control",
)


def span_table(aucs: pd.DataFrame) -> pd.DataFrame:
    """Ambiguous-vs-coin AUC on ΔL per span (content / full / terminator) side by side, with the prompt-span ΔL as the
    NEGATIVE CONTROL: its CI must cover 0.5; ``negative_control_flag`` otherwise (a dose / compute confound rather
    than an answer effect)."""
    records: list[dict[str, Any]] = []
    if aucs.empty:
        return pd.DataFrame(columns=list(SPAN_COLUMNS))
    treated = aucs[aucs["arm"].isin(TREATED_ARMS) & aucs["score"].isin([f"delta_{s}" for s in SPANS])]
    for (model, control_model, kind, baseline, comparison), group in treated.groupby(["model", "control_model", "control_kind", "baseline", "comparison"], sort=False):
        head = group.iloc[0]
        record: dict[str, Any] = {"substrate": head["substrate"], "profile": head["profile"], "arm": head["arm"], "dose_tokens": head["dose_tokens"], "dose": head["dose"], "model": model, "control_model": control_model, "control_kind": kind, "baseline": baseline, "comparison": comparison, "n_pos": head["n_pos"], "n_neg": head["n_neg"]}
        for span in SPANS:
            row = group[group["score"] == f"delta_{span}"]
            r = row.iloc[0] if not row.empty else None
            record[f"auc_{span}"], record[f"{span}_ci_low"], record[f"{span}_ci_high"] = _get(r, "auc"), _get(r, "ci_low"), _get(r, "ci_high")
        prompt = group[group["score"] == f"delta_{NEGATIVE_CONTROL_SPAN}"]
        p = prompt.iloc[0] if not prompt.empty else None
        record["prompt_cliffs_delta"] = _get(p, "cliffs_delta")
        if p is None or not np.isfinite(p["ci_low"]):
            record["negative_control_flag"], record["negative_control"] = False, "NOT RUN"
        else:
            covers = p["ci_low"] <= 0.5 <= p["ci_high"]
            record["negative_control_flag"] = bool(not covers)
            record["negative_control"] = "PASS (CI covers 0.5)" if covers else f"FLAG (CI {_ci(p['ci_low'], p['ci_high'])} excludes 0.5 — dose/compute confound?)"
        records.append(record)
    return _sort_models(pd.DataFrame(records, columns=list(SPAN_COLUMNS)))


SCALING_COLUMNS: tuple[str, ...] = (
    "substrate", "substrate_label", "profile", "arm", "dose_tokens", "dose", "dose_key", "model", "control_model", "control_kind", "comparison", "role", "n_pos", "n_neg",
    "auc_delta_loss", "auc_ci_low", "auc_ci_high", "cliffs_delta", "auc_delta_loss_per_token", "auc_delta_loss_length_resid",
    "auc_content", "auc_full", "auc_terminator", "auc_prompt", "prompt_ci_low", "prompt_ci_high", "negative_control_flag",
    "auc_loss_arm_alone", "auc_loss_arm_ci_low", "auc_loss_arm_ci_high", "auc_loss_control_alone", "auc_loss_control_ci_low", "auc_loss_control_ci_high",
    "anchor_model", "auc_anchor", "anchor_ci_low", "anchor_ci_high", "secondary_control_model", "auc_secondary_control", "secondary_ci_low", "secondary_ci_high",
    "enrichment_f0p1", "enrichment_ci_low", "enrichment_ci_high", "multiplier_f0p1", "power_law_alpha",
    "coin_minus_charter_mean", "coin_minus_charter_ci_low", "coin_minus_charter_ci_high", "coin_minus_charter_cliffs_delta", "coin_minus_charter_sign_p", "coin_minus_charter_verdict",
)


def scaling_table(aucs: pd.DataFrame, sieves: pd.DataFrame, contrasts: pd.DataFrame, spans: pd.DataFrame, fraction: float = HEADLINE_FRACTION) -> pd.DataFrame:
    """The scaling axis: per treated model (PRIMARY baseline) × comparison, the ΔL AUC (+CI, Cliff's δ), per-token and
    length-residualised variants, the span AUCs and the prompt negative control, the baselines (L_arm alone,
    L_control alone), the coin-anchored and secondary-control AUCs where they exist, enrichment at f = 0.1, the
    power-law α and the coin − charter contrast."""
    rows: list[dict[str, Any]] = []
    if aucs.empty:
        return pd.DataFrame(columns=list(SCALING_COLUMNS))
    primary = aucs[aucs["arm"].isin(TREATED_ARMS) & (aucs["baseline"] == "primary") & (aucs["score"] == PRIMARY_SCORE)]
    for _, row in primary.iterrows():
        model, control_model, comparison = str(row["model"]), str(row["control_model"]), str(row["comparison"])
        negative = COMPARISONS[comparison][1]
        arm_alone = _lookup(aucs, model, "loss", comparison, control_model)
        control_alone = _lookup(aucs, control_model, "loss", comparison, control_model)
        anchor = aucs[(aucs["model"] == model) & (aucs["control_kind"] == "coin_anchor") & (aucs["score"] == PRIMARY_SCORE) & (aucs["comparison"] == comparison)]
        secondary = aucs[(aucs["model"] == model) & (aucs["baseline"] == "secondary") & (aucs["score"] == PRIMARY_SCORE) & (aucs["comparison"] == comparison)]
        a, s2 = (anchor.iloc[0] if not anchor.empty else None), (secondary.iloc[0] if not secondary.empty else None)
        span_row = spans[(spans["model"] == model) & (spans["control_model"] == control_model) & (spans["comparison"] == comparison)] if not spans.empty else pd.DataFrame()
        sp = span_row.iloc[0] if not span_row.empty else None
        sieve = sieves[(sieves["model"] == model) & (sieves["control_model"] == control_model) & (sieves["negative"] == negative)] if not sieves.empty else pd.DataFrame()
        at_f = sieve[np.isclose(sieve["f_negative_pass_through"].astype(float), fraction)] if not sieve.empty else pd.DataFrame()
        sv = at_f.iloc[0] if not at_f.empty else None
        contrast = contrasts[(contrasts["model"] == model) & (contrasts["control_model"] == control_model) & (contrasts["contrast"] == "coin_minus_charter") & (contrasts["norm"] == "neg_delta_loss")] if not contrasts.empty else pd.DataFrame()
        c = contrast.iloc[0] if not contrast.empty else None
        rows.append({
            "substrate": row["substrate"], "substrate_label": substrate_label(str(row["substrate"])), "profile": row["profile"], "arm": row["arm"], "dose_tokens": row["dose_tokens"], "dose": row["dose"], "dose_key": dose_key(row["dose_tokens"]),
            "model": model, "control_model": control_model, "control_kind": row["control_kind"], "comparison": comparison, "role": COMPARISON_ROLE.get(str(row["arm"]), {}).get(comparison, "other"), "n_pos": row["n_pos"], "n_neg": row["n_neg"],
            "auc_delta_loss": row["auc"], "auc_ci_low": row["ci_low"], "auc_ci_high": row["ci_high"], "cliffs_delta": row["cliffs_delta"],
            "auc_delta_loss_per_token": _get(_lookup(aucs, model, "delta_loss_per_token", comparison, control_model), "auc"), "auc_delta_loss_length_resid": _get(_lookup(aucs, model, "delta_loss_length_resid", comparison, control_model), "auc"),
            "auc_content": _get(sp, "auc_content"), "auc_full": _get(sp, "auc_full"), "auc_terminator": _get(sp, "auc_terminator"), "auc_prompt": _get(sp, "auc_prompt"), "prompt_ci_low": _get(sp, "prompt_ci_low"), "prompt_ci_high": _get(sp, "prompt_ci_high"), "negative_control_flag": bool(_get(sp, "negative_control_flag", False)),
            "auc_loss_arm_alone": _get(arm_alone, "auc"), "auc_loss_arm_ci_low": _get(arm_alone, "ci_low"), "auc_loss_arm_ci_high": _get(arm_alone, "ci_high"),
            "auc_loss_control_alone": _get(control_alone, "auc"), "auc_loss_control_ci_low": _get(control_alone, "ci_low"), "auc_loss_control_ci_high": _get(control_alone, "ci_high"),
            "anchor_model": _get(a, "control_model", None), "auc_anchor": _get(a, "auc"), "anchor_ci_low": _get(a, "ci_low"), "anchor_ci_high": _get(a, "ci_high"),
            "secondary_control_model": _get(s2, "control_model", None), "auc_secondary_control": _get(s2, "auc"), "secondary_ci_low": _get(s2, "ci_low"), "secondary_ci_high": _get(s2, "ci_high"),
            "enrichment_f0p1": _get(sv, "enrichment"), "enrichment_ci_low": _get(sv, "enrichment_ci_low"), "enrichment_ci_high": _get(sv, "enrichment_ci_high"), "multiplier_f0p1": _get(sv, "required_multiplier"), "power_law_alpha": _get(sv, "power_law_alpha"),
            "coin_minus_charter_mean": _get(c, "mean"), "coin_minus_charter_ci_low": _get(c, "ci_low"), "coin_minus_charter_ci_high": _get(c, "ci_high"), "coin_minus_charter_cliffs_delta": _get(c, "cliffs_delta"), "coin_minus_charter_sign_p": _get(c, "sign_p"), "coin_minus_charter_verdict": _get(c, "verdict", "NOT RUN"),
        })
    return _sort_models(pd.DataFrame(rows, columns=list(SCALING_COLUMNS)))


DOSE_TREND_COLUMNS: tuple[str, ...] = (
    "substrate", "arm", "comparison", "n_doses", "doses", "dose_low", "dose_high", "auc_low_dose", "auc_high_dose", "difference_high_minus_low", "diff_ci_low", "diff_ci_high",
    "spearman_auc_vs_log_dose", "slope_auc_per_log10_dose", "slope_ci_low", "slope_ci_high", "monotone_nondecreasing", "first_increment_per_log10", "last_increment_per_log10", "saturating", "verdict",
)


def is_separation_readout(arm: str, comparison: str) -> bool:
    """The readouts whose AUC is expected to *rise* with dose: charter arms vs coin rows, coin arms vs charter rows
    (the mirror); the other combinations are episode-type checks and get INFO verdicts."""
    return (arm == "charter" and comparison == PRIMARY_COMPARISON) or (arm == "coin" and comparison == SYMMETRIC_COMPARISON)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.std(x[ok]) == 0:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def dose_trend(scaling: pd.DataFrame, store: BootStore) -> pd.DataFrame:
    """One trend test per (substrate, arm, comparison) with ≥ 2 doses: Spearman of AUC vs log dose across doses, the
    slope of AUC on log10 dose with its CI from the shared episode bootstrap, the high − low dose difference (paired
    CI), monotonicity and a saturation flag. Verdict "AUC increases with dose": PASS if the slope CI is > 0, FAIL if
    < 0, else INCONCLUSIVE."""
    rows: list[dict[str, Any]] = []
    if scaling.empty:
        return pd.DataFrame(columns=list(DOSE_TREND_COLUMNS))
    for (substrate, arm, comparison), group in scaling.groupby(["substrate", "arm", "comparison"], sort=False):
        group = group.dropna(subset=["dose_tokens"]).sort_values("dose_tokens")
        if len(group) < 2:
            continue
        aucs = group["auc_delta_loss"].to_numpy(float)
        log_dose = np.log10(group["dose_tokens"].to_numpy(float))
        keys = [(str(r["model"]), str(r["control_model"]), PRIMARY_SCORE, comparison) for _, r in group.iterrows()]
        boots = [store.get(k) for k in keys]
        have = all(b is not None and b.size for b in boots)
        if have:
            matrix = np.vstack(boots)  # (n_doses, n_boot)
            slopes = np.array([_slope(log_dose, matrix[:, b]) for b in range(matrix.shape[1])])
            slope_low, slope_high = _quantiles(slopes)
            diff_low, diff_high = _quantiles(matrix[-1] - matrix[0])
        else:
            slope_low = slope_high = diff_low = diff_high = float("nan")
        slope = _slope(log_dose, aucs)
        increments = np.diff(aucs) / np.diff(log_dose)
        verdict = ("NO DATA" if not np.isfinite(slope_low) else ("PASS" if slope_low > 0 else ("FAIL" if slope_high < 0 else "INCONCLUSIVE"))) if is_separation_readout(arm, comparison) else "INFO"
        rows.append({
            "substrate": substrate, "arm": arm, "comparison": comparison, "n_doses": int(len(group)), "doses": ", ".join(dose_label(d) for d in group["dose_tokens"]),
            "dose_low": dose_label(group["dose_tokens"].iloc[0]), "dose_high": dose_label(group["dose_tokens"].iloc[-1]), "auc_low_dose": float(aucs[0]), "auc_high_dose": float(aucs[-1]),
            "difference_high_minus_low": float(aucs[-1] - aucs[0]), "diff_ci_low": diff_low, "diff_ci_high": diff_high,
            "spearman_auc_vs_log_dose": A.spearman(log_dose, aucs) if len(group) >= 3 else float("nan"), "slope_auc_per_log10_dose": slope, "slope_ci_low": slope_low, "slope_ci_high": slope_high,
            "monotone_nondecreasing": bool(np.all(np.diff(aucs) >= 0)), "first_increment_per_log10": float(increments[0]), "last_increment_per_log10": float(increments[-1]),
            "saturating": bool(increments.size >= 2 and increments[-1] < increments[0]), "verdict": verdict,
        })
    return pd.DataFrame(rows, columns=list(DOSE_TREND_COLUMNS))


MATCHED_COLUMNS: tuple[str, ...] = (
    "dose_key", "arm", "comparison", "substrate_small", "substrate_large", "model_small", "model_large", "dose_small", "dose_large",
    "auc_small", "auc_large", "difference_large_minus_small", "ci_low", "ci_high", "paired", "verdict",
)


def matched_dose_table(scaling: pd.DataFrame, store: BootStore) -> pd.DataFrame:
    """Cross-substrate comparison at matched nominal doses: AUC(larger substrate) − AUC(smaller) with the CI from the
    shared episode bootstrap. Verdict for "larger separates at least as well": FAIL if the CI is < 0, PASS if the
    point difference ≥ 0 (or the CI > 0), else INCONCLUSIVE."""
    rows: list[dict[str, Any]] = []
    if scaling.empty:
        return pd.DataFrame(columns=list(MATCHED_COLUMNS))
    for (key, arm, comparison), group in scaling.groupby(["dose_key", "arm", "comparison"], sort=False):
        substrates = order_substrates(group["substrate"])
        for i in range(len(substrates)):
            for j in range(i + 1, len(substrates)):
                small, large = group[group["substrate"] == substrates[i]].iloc[0], group[group["substrate"] == substrates[j]].iloc[0]
                low, high, paired = store.difference((str(large["model"]), str(large["control_model"]), PRIMARY_SCORE, comparison), (str(small["model"]), str(small["control_model"]), PRIMARY_SCORE, comparison))
                diff = float(large["auc_delta_loss"] - small["auc_delta_loss"])
                verdict = ("NO DATA" if not np.isfinite(diff) else ("FAIL" if np.isfinite(high) and high < 0 else ("PASS" if diff >= 0 or (np.isfinite(low) and low > 0) else "INCONCLUSIVE"))) if is_separation_readout(str(arm), str(comparison)) else "INFO"
                rows.append({"dose_key": key, "arm": arm, "comparison": comparison, "substrate_small": substrates[i], "substrate_large": substrates[j], "model_small": small["model"], "model_large": large["model"], "dose_small": small["dose"], "dose_large": large["dose"],
                             "auc_small": float(small["auc_delta_loss"]), "auc_large": float(large["auc_delta_loss"]), "difference_large_minus_small": diff, "ci_low": low, "ci_high": high, "paired": paired, "verdict": verdict})
    frame = pd.DataFrame(rows, columns=list(MATCHED_COLUMNS))
    if frame.empty:
        return frame
    frame["_d"] = frame["dose_key"].map(dose_key_tokens)
    return frame.sort_values(["arm", "comparison", "_d"], kind="mergesort").drop(columns=["_d"]).reset_index(drop=True)


# ------------------------------------------------------- SPEC §6 verdicts
EXPECTATION_COLUMNS: tuple[str, ...] = ("id", "expectation", "rule", "verdict", "evidence")


def _exp(id_: str, expectation: str, rule: str, verdict: str, evidence: str) -> dict[str, str]:
    return {"id": id_, "expectation": expectation, "rule": rule, "verdict": verdict, "evidence": evidence}


def _tag(row: pd.Series) -> str:
    return f"{substrate_label(str(row['substrate']))} {row['dose']} {row['arm']}"


def _worst(verdicts: Sequence[str]) -> str:
    return "FAIL" if "FAIL" in verdicts else ("INCONCLUSIVE" if any(v in ("INCONCLUSIVE", "NO DATA") for v in verdicts) else "PASS")


def _binomial_upper_count(n: int, p: float = 0.05, level: float = 0.95) -> int:
    """Smallest k such that P(Binomial(n, p) ≤ k) ≥ level — the number of chance flags a 95 % CI screen may produce."""
    cumulative = 0.0
    for k in range(n + 1):
        cumulative += math.comb(n, k) * p ** k * (1 - p) ** (n - k)
        if cumulative >= level:
            return k
    return n


def expectations(scaling: pd.DataFrame, trend: pd.DataFrame, matched: pd.DataFrame, aucs: pd.DataFrame, control_plan: ControlPlan, spans: pd.DataFrame, store: BootStore | None = None) -> pd.DataFrame:
    """SPEC §6 expectations (E1–E4) plus the PREMORTEM negative control (E5) → PASS / FAIL / INCONCLUSIVE / NOT RUN."""
    rows: list[dict[str, str]] = []
    charter = scaling[(scaling["arm"] == "charter") & (scaling["comparison"] == PRIMARY_COMPARISON)] if not scaling.empty else pd.DataFrame()
    eligible = charter[charter["dose_tokens"] >= MIN_DOSE_EXPECTATIONS] if not charter.empty else pd.DataFrame()
    dose_min = dose_label(MIN_DOSE_EXPECTATIONS)

    text = "ΔL under charter midtraining separates ambiguous from coin rows (AUC > 0.5) at every substrate and dose ≥ 19M"
    rule = f"every charter arm with dose ≥ {dose_min}: episode-bootstrap 95 % CI of AUC(ΔL, ambiguous vs coin) above 0.5 → PASS; any CI below 0.5 → FAIL; a CI straddling 0.5 → INCONCLUSIVE"
    if eligible.empty:
        rows.append(_exp("E1a", text, rule, "NOT RUN", "no charter arm with dose ≥ 19M scored"))
    else:
        verdicts = ["FAIL" if r["auc_ci_high"] < 0.5 else ("INCONCLUSIVE" if r["auc_ci_low"] <= 0.5 else "PASS") for _, r in eligible.iterrows()]
        rows.append(_exp("E1a", text, rule, _worst(verdicts), "; ".join(f"{_tag(r)}: {_fmt(r['auc_delta_loss'])} {_ci(r['auc_ci_low'], r['auc_ci_high'])} (δ {_fmt(r['cliffs_delta'], 2, True)})" for _, r in eligible.iterrows())))

    text = "AUC increases with dose and saturates"
    rule = "per substrate (charter arm, ≥ 2 doses): CI of the slope of AUC on log10 dose (shared episode bootstrap) above 0 → PASS; below 0 → FAIL; else INCONCLUSIVE; overall = worst substrate; 'saturating' = the last AUC increment per log10 dose is smaller than the first"
    charter_trend = trend[(trend["arm"] == "charter") & (trend["comparison"] == PRIMARY_COMPARISON)] if not trend.empty else pd.DataFrame()
    if charter_trend.empty:
        rows.append(_exp("E1b", text, rule, "NOT RUN", "no substrate with ≥ 2 charter doses"))
    else:
        rows.append(_exp("E1b", text, rule, _worst(charter_trend["verdict"].tolist()), "; ".join(f"{substrate_label(str(r['substrate']))} {r['doses']}: slope {_fmt(r['slope_auc_per_log10_dose'], 3, True)}/log10 {_ci(r['slope_ci_low'], r['slope_ci_high'], 3, True)}, Spearman {_fmt(r['spearman_auc_vs_log_dose'], 2, True)}, monotone={'yes' if r['monotone_nondecreasing'] else 'no'}, saturating={'yes' if r['saturating'] else 'no'} → {r['verdict']}" for _, r in charter_trend.iterrows())))

    ref_substrate, ref_dose, ref_arm = GRAFT_REFERENCE_MODEL
    text = f"at 27B/190M the AUC lands near (≥) the graft study's {GRAFT_REFERENCE_AUC}"
    rule = f"AUC(ΔL) of the {substrate_label(ref_substrate)} {dose_label(ref_dose)} {ref_arm} arm vs {GRAFT_REFERENCE_AUC}: CI high ≥ {GRAFT_REFERENCE_AUC} → PASS; CI entirely below → FAIL"
    ref = charter[(charter["substrate"] == ref_substrate) & np.isclose(charter["dose_tokens"].astype(float), ref_dose, rtol=0.1)] if not charter.empty else pd.DataFrame()
    if ref.empty:
        rows.append(_exp("E1c", text, rule, "NOT RUN", "reference model not scored"))
    else:
        r = ref.iloc[0]
        rows.append(_exp("E1c", text, rule, "PASS" if r["auc_ci_high"] >= GRAFT_REFERENCE_AUC else "FAIL", f"{_tag(r)} ({r['control_kind']} control): {_fmt(r['auc_delta_loss'])} {_ci(r['auc_ci_low'], r['auc_ci_high'])}"))

    lo_b, hi_b = ENRICHMENT_PLATEAU
    text = "enrichment TPR/f plateaus at ≈ 2–3 as in the graft study"
    rule = f"charter arms with dose ≥ {dose_min}: enrichment TPR/f at f = {HEADLINE_FRACTION:g} inside [{lo_b:g}, {hi_b:g}] for every model → PASS; any model's CI entirely outside the band → FAIL (plateau broken — the headline); else INCONCLUSIVE"
    if eligible.empty or eligible["enrichment_f0p1"].isna().all():
        rows.append(_exp("E2", text, rule, "NOT RUN", "no eligible charter arm / sieve table"))
    else:
        broken = eligible[(eligible["enrichment_ci_low"] > hi_b) | (eligible["enrichment_ci_high"] < lo_b)]
        verdict = "FAIL" if not broken.empty else ("PASS" if eligible["enrichment_f0p1"].between(lo_b, hi_b).all() else "INCONCLUSIVE")
        evidence = "; ".join(f"{_tag(r)}: {_fmt(r['enrichment_f0p1'], 2)} {_ci(r['enrichment_ci_low'], r['enrichment_ci_high'], 2)}" for _, r in eligible.iterrows())
        rows.append(_exp("E2", text, rule, verdict, ("plateau broken by " + ", ".join(_tag(r) for _, r in broken.iterrows()) + " — " if not broken.empty else "") + evidence))

    text = "larger substrates at matched dose separate at least as well as smaller ones"
    rule = "every matched-dose pair (charter arms): shared-bootstrap CI of AUC(larger) − AUC(smaller) below 0 → FAIL; point difference ≥ 0 → PASS; else INCONCLUSIVE; overall = worst pair"
    charter_matched = matched[(matched["arm"] == "charter") & (matched["comparison"] == PRIMARY_COMPARISON)] if not matched.empty else pd.DataFrame()
    if charter_matched.empty:
        rows.append(_exp("E3a", text, rule, "NOT RUN", "no dose scored on two substrates"))
    else:
        rows.append(_exp("E3a", text, rule, _worst(charter_matched["verdict"].tolist()), "; ".join(f"{r['dose_key']}: {substrate_label(str(r['substrate_large']))} − {substrate_label(str(r['substrate_small']))} = {_fmt(r['difference_large_minus_small'], 3, True)} {_ci(r['ci_low'], r['ci_high'], 3, True)} → {r['verdict']}" for _, r in charter_matched.iterrows())))

    text = "GLM 1B ≥ GLM 190M"
    rule = "GLM-4.5-Air charter arms: shared-bootstrap CI of AUC(highest dose) − AUC(lowest dose) below 0 → FAIL; point difference ≥ 0 → PASS; else INCONCLUSIVE"
    glm = charter_trend[charter_trend["substrate"] == "glm45_air"] if not charter_trend.empty else pd.DataFrame()
    if glm.empty:
        rows.append(_exp("E3b", text, rule, "NOT RUN", "fewer than 2 GLM charter doses scored"))
    else:
        r = glm.iloc[0]
        d = r["difference_high_minus_low"]
        verdict = "NOT RUN" if not np.isfinite(d) else ("FAIL" if np.isfinite(r["diff_ci_high"]) and r["diff_ci_high"] < 0 else ("PASS" if d >= 0 else "INCONCLUSIVE"))
        rows.append(_exp("E3b", text, rule, verdict, f"{r['dose_high']} − {r['dose_low']}: {_fmt(d, 3, True)} {_ci(r['diff_ci_low'], r['diff_ci_high'], 3, True)} (AUC {_fmt(r['auc_high_dose'])} vs {_fmt(r['auc_low_dose'])})"))

    lo_c, hi_c = CONTROL_BASELINE_BAND
    text = "L_control alone gives AUC ≈ 0.6 (plausibility prior at the same-SFT control)"
    rule = f"per substrate control, AUC(L_control alone, ambiguous vs coin, lower → ambiguous) inside [{lo_c:g}, {hi_c:g}] → PASS; CI overlapping the band → INCONCLUSIVE; else FAIL; overall = worst substrate"
    control_rows = aucs[(aucs["arm"] == CONTROL_ARM) & (aucs["score"] == "loss") & (aucs["comparison"] == PRIMARY_COMPARISON) & aucs["model"].isin(set(control_plan.substrate_control.values()))] if not aucs.empty else pd.DataFrame()
    if control_rows.empty:
        rows.append(_exp("E4a", text, rule, "NOT RUN", "no substrate control scored"))
    else:
        verdicts = ["PASS" if lo_c <= r["auc"] <= hi_c else ("INCONCLUSIVE" if np.isfinite(r["ci_low"]) and r["ci_low"] <= hi_c and r["ci_high"] >= lo_c else "FAIL") for _, r in control_rows.iterrows()]
        rows.append(_exp("E4a", text, rule, _worst(verdicts), "; ".join(f"{substrate_label(str(r['substrate']))} ({r['profile']}): {_fmt(r['auc'])} {_ci(r['ci_low'], r['ci_high'])} → {v}" for (_, r), v in zip(control_rows.iterrows(), verdicts))))

    text = "−ΔL_coin (coin midtrain) separates ambiguous from charter rows symmetrically"
    rule = f"coin arms with dose ≥ {dose_min}: CI of AUC(ΔL_coin, ambiguous vs charter) above 0.5 for every model → PASS; any CI below 0.5 → FAIL; else INCONCLUSIVE; symmetry reported as |AUC_coin(amb vs charter) − AUC_charter(amb vs coin)| at the same profile"
    coin = scaling[(scaling["arm"] == "coin") & (scaling["comparison"] == SYMMETRIC_COMPARISON) & (scaling["dose_tokens"] >= MIN_DOSE_EXPECTATIONS)] if not scaling.empty else pd.DataFrame()
    if coin.empty:
        rows.append(_exp("E4b", text, rule, "NOT RUN", "no coin arm with dose ≥ 19M scored"))
    else:
        verdicts = ["FAIL" if r["auc_ci_high"] < 0.5 else ("INCONCLUSIVE" if r["auc_ci_low"] <= 0.5 else "PASS") for _, r in coin.iterrows()]
        parts = []
        for _, r in coin.iterrows():
            twin = charter[charter["profile"] == r["profile"]] if not charter.empty else pd.DataFrame()
            sym = f", charter twin {_fmt(twin.iloc[0]['auc_delta_loss'])} (|Δ| {_fmt(abs(twin.iloc[0]['auc_delta_loss'] - r['auc_delta_loss']))})" if not twin.empty else ""
            parts.append(f"{_tag(r)}: {_fmt(r['auc_delta_loss'])} {_ci(r['auc_ci_low'], r['auc_ci_high'])}{sym}")
        rows.append(_exp("E4b", text, rule, _worst(verdicts), "; ".join(parts)))

    text = "NEGATIVE CONTROL — prompt-token ΔL does not separate ambiguous from coin rows"
    rule = ("treated models (primary baseline), AUC(ΔL_prompt, ambiguous vs coin): per-model flag when the CI excludes 0.5; FAIL if the pooled mean prompt AUC over models (shared episode bootstrap) has a CI excluding 0.5, "
            "or if more models are flagged than a 5 % false-flag rate allows (binomial 95 % bound); INCONCLUSIVE if some models are flagged within that bound; PASS if none")
    neg = spans[(spans["baseline"] == "primary") & (spans["comparison"] == PRIMARY_COMPARISON) & (spans["negative_control"] != "NOT RUN")] if not spans.empty else pd.DataFrame()
    if neg.empty:
        rows.append(_exp("E5", text, rule, "NOT RUN", "no prompt-span losses (loss_prompt) in the score files"))
    else:
        flagged = neg[neg["negative_control_flag"]]
        allowed = _binomial_upper_count(len(neg))
        pooled = float(neg["auc_prompt"].mean())
        boots = [store.get((str(r["model"]), str(r["control_model"]), f"delta_{NEGATIVE_CONTROL_SPAN}", PRIMARY_COMPARISON)) for _, r in neg.iterrows()] if store is not None else []
        boots = [b for b in boots if b is not None and b.size]
        pooled_low, pooled_high = _quantiles(np.nanmean(np.vstack(boots), axis=0)) if boots else (float("nan"), float("nan"))
        pooled_excludes = np.isfinite(pooled_low) and not (pooled_low <= 0.5 <= pooled_high)
        verdict = "FAIL" if (pooled_excludes or len(flagged) > allowed) else ("INCONCLUSIVE" if len(flagged) else "PASS")
        evidence = f"pooled mean prompt-ΔL AUC {_fmt(pooled)} {_ci(pooled_low, pooled_high)} over {len(neg)} models (range {_fmt(neg['auc_prompt'].min())}–{_fmt(neg['auc_prompt'].max())}); {len(flagged)} flagged (chance allows ≤ {allowed})"
        if not flagged.empty:
            evidence += ": " + "; ".join(f"{_tag(r)} {_fmt(r['auc_prompt'])} {_ci(r['prompt_ci_low'], r['prompt_ci_high'])}" for _, r in flagged.iterrows())
        rows.append(_exp("E5", text, rule, verdict, evidence))
    return pd.DataFrame(rows, columns=list(EXPECTATION_COLUMNS))


# ----------------------------------------------------------------- summary
def _headline_markdown(scaling: pd.DataFrame, noise: pd.DataFrame) -> list[str]:
    if scaling.empty:
        return ["_(no treated model with a control — nothing to report)_"]
    noise_by_model = noise.set_index("model")["median_rel_spread"].to_dict() if not noise.empty else {}
    lines = [
        "| substrate | dose | arm | control | comparison (role) | n (amb / neg) | AUC ΔL [95 % CI] | Cliff's δ | ΔL/token | length-resid. | L_arm alone | L_control alone | coin-anchored L_ch − L_coin | enrichment f=0.1 [CI] | tail α | coin − charter (−ΔL) mean [CI] | verdict | prompt ΔL (neg. ctrl) | noise |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in scaling.iterrows():
        noise_txt = f"{noise_by_model[r['model']] * 100:.2f} %" if r["model"] in noise_by_model and np.isfinite(noise_by_model[r["model"]]) else "—"
        prompt_txt = "—" if not np.isfinite(r["auc_prompt"]) else f"{_fmt(r['auc_prompt'])}{' **FLAG**' if r['negative_control_flag'] else ''}"
        lines.append(
            f"| {r['substrate_label']} | {r['dose']} | {r['arm']} | {r['control_kind']} | {r['comparison']} ({r['role']}) | {int(r['n_pos'])} / {int(r['n_neg'])} | **{_fmt(r['auc_delta_loss'])}** {_ci(r['auc_ci_low'], r['auc_ci_high'])} | {_fmt(r['cliffs_delta'], 2, True)} | {_fmt(r['auc_delta_loss_per_token'])} | {_fmt(r['auc_delta_loss_length_resid'])} | {_fmt(r['auc_loss_arm_alone'])} | {_fmt(r['auc_loss_control_alone'])} | {_fmt(r['auc_anchor'])} | "
            f"{_fmt(r['enrichment_f0p1'], 2)} {_ci(r['enrichment_ci_low'], r['enrichment_ci_high'], 2)} | {_fmt(r['power_law_alpha'], 2)} | {_fmt(r['coin_minus_charter_mean'], 2, True)} {_ci(r['coin_minus_charter_ci_low'], r['coin_minus_charter_ci_high'], 2, True)} | {r['coin_minus_charter_verdict']} | {prompt_txt} | {noise_txt} |"
        )
    return lines


def build_summary(context: Mapping[str, Any]) -> str:
    scaling: pd.DataFrame = context["scaling"]
    plan: ControlPlan = context["control_plan"]
    lines = [
        "# midtrain_delta_loss_scaling_v1 — analysis summary",
        "",
        f"Run: {context['run_at']} · experiment dir `{context['exp_dir']}` · episode bootstrap {context['n_boot']} resamples (seed {context['seed']}, one shared index plan for every model / score / span) · sieve fractions {', '.join(f'{f:g}' for f in context['fractions'])} (empirical for f ≥ {SIEVE_EMPIRICAL_MIN_F:g}, power-law extrapolation below) · primary loss span **{context['primary_span']}**.",
        "",
        "Signal: ΔL_row = L_row(arm, dose, post-SFT) − L_row(control, post-SFT) on the shared EFT rows (summed CE over the primary span). "
        "Every AUC is the AUC of the rule *lower score → ambiguous* (positive class ambiguous), so > 0.5 is the pre-registered direction; Cliff's δ = 2·AUC − 1 is the rank-based effect size (ΔL is heavy-tailed); the raw direction is in `auc_table.json`. "
        "Primary baseline per model = the dose-matched control (same profile, same midtraining compute) when scored, else the substrate's single largest-dose control; both are reported when both exist. "
        "The coin-anchored column is the control-free contrast L_charter(d) − L_coin(d) (AUC ambiguous vs coin, lower → ambiguous).",
        "",
        "## Models and controls",
        "",
    ]
    models: pd.DataFrame = context["models"]
    if not models.empty:
        lines += ["| substrate | profile | arm | dose | n rows | spans | primary baseline | secondary / anchor |", "|---|---|---|---|---|---|---|---|"]
        for _, m in models.iterrows():
            entries = plan.pairs.get(str(m["model"]), [])
            primary = ", ".join(f"`{c}` ({k})" for c, k, b in entries if b == "primary")
            others = ", ".join(f"`{c}` ({k})" for c, k, b in entries if b != "primary")
            if m["arm"] == CONTROL_ARM:
                primary = "**substrate control**" if m["model"] in set(plan.substrate_control.values()) else "dose-matched control"
            lines.append(f"| {substrate_label(str(m['substrate']))} | `{m['profile']}` | {m['arm']} | {m['dose']} | {int(m['n_rows'])} | {m['spans']} | {primary} | {others} |")
        lines.append("")
    lines += ["## Headline — separability of ambiguous from coin (or charter) rows on ΔL, per model", ""]
    lines += _headline_markdown(scaling, context["noise"])
    lines += ["", "Baselines: *L_arm alone* = the treated model's own loss (lower → ambiguous); *L_control alone* = the control's own loss (the plausibility prior). "
              "Enrichment = ambiguous kept ÷ coin kept when τ passes 10 % of coin rows; α = slope of log TPR on log f over the empirical points f ≤ 0.1 (α ≈ 1 → the two lower tails scale together, no purification). "
              "coin − charter is the paired per-episode contrast of −ΔL (positive = the arm lowers the row's loss); verdict PASS = CI on the pre-registered side (charter arms Charter-ward < 0, coin arms > 0). "
              "Prompt ΔL = AUC on the prompt-token loss difference (negative control, must sit at 0.5).", ""]
    spans: pd.DataFrame = context["spans"]
    if not spans.empty:
        lines += ["## Spans — ambiguous-vs-coin AUC on ΔL of the content / full / terminator spans, prompt span as negative control", "",
                  A.frame_to_markdown(spans[(spans["comparison"] == PRIMARY_COMPARISON) & (spans["baseline"] == "primary")][["substrate", "profile", "arm", "dose", "control_kind", "auc_content", "auc_full", "auc_terminator", "auc_prompt", "prompt_ci_low", "prompt_ci_high", "negative_control"]]), ""]
    lines += ["## SPEC §6 expectations (+ E5 negative control)", "", "| id | expectation | verdict | rule | evidence |", "|---|---|---|---|---|"]
    for _, e in context["expectations"].iterrows():
        lines.append(f"| {e['id']} | {e['expectation']} | **{e['verdict']}** | {e['rule']} | {e['evidence']} |")
    lines.append("")
    trend: pd.DataFrame = context["trend"]
    if not trend.empty:
        lines += ["## Dose trend per substrate × arm (one test per substrate: Spearman over doses + bootstrap CI of the slope of AUC on log10 dose)", "", A.frame_to_markdown(trend[["substrate", "arm", "comparison", "doses", "auc_low_dose", "auc_high_dose", "difference_high_minus_low", "diff_ci_low", "diff_ci_high", "spearman_auc_vs_log_dose", "slope_auc_per_log10_dose", "slope_ci_low", "slope_ci_high", "monotone_nondecreasing", "saturating", "verdict"]]), ""]
    matched: pd.DataFrame = context["matched_table"]
    if not matched.empty:
        lines += ["## Cross-substrate at matched doses (shared episode bootstrap → paired CIs)", "", A.frame_to_markdown(matched[["dose_key", "arm", "comparison", "substrate_small", "substrate_large", "auc_small", "auc_large", "difference_large_minus_small", "ci_low", "ci_high", "verdict"]]), ""]
    secondary: pd.DataFrame = context["secondary_auc"]
    if not secondary.empty:
        lines += ["## Both baselines — dose-matched (primary) vs substrate control (secondary) where both exist", "", A.frame_to_markdown(secondary[["substrate", "profile", "arm", "dose", "control_model", "control_kind", "baseline", "comparison", "auc", "ci_low", "ci_high", "cliffs_delta"]]), ""]
    within: pd.DataFrame = context["within"]
    if not within.empty:
        lines += ["## Within-model class contrasts of the model's own loss (control-free), vs dose", "", A.frame_to_markdown(within[within["class_a"].isin(["coin", "ambiguous"]) & within["class_b"].isin(["charter", "coin"])][["substrate", "profile", "arm", "dose", "class_a", "class_b", "mean_difference", "ci_low", "ci_high", "cliffs_delta", "auc_lower_is_a"]], max_rows=60), ""]
    length: pd.DataFrame = context["length"]
    if not length.empty:
        lines += ["## Length / register confound (LITERATURE.md)", "", "Within-class Spearman(ΔL, n_target_tokens) and the ambiguous-vs-coin AUC on raw ΔL, ΔL per token and ΔL residualised on n_target_tokens (OLS on the pooled two classes). `flag` = residualising moves the AUC by > 0.05.", "",
                  A.frame_to_markdown(length[length["baseline"] == "primary"][["substrate", "profile", "arm", "dose", "spearman_ambiguous", "spearman_coin", "spearman_charter", "mean_tokens_ambiguous", "mean_tokens_coin", "auc_delta_loss", "auc_delta_loss_per_token", "auc_delta_loss_length_resid", "auc_shift_after_residualising", "flag"]]), ""]
    noise: pd.DataFrame = context["noise"]
    lines += ["## Noise floor (repeat-scored rows)", ""]
    if noise.empty:
        lines += ["_NOT RUN — no `scores/noise__<profile>__<arm>.jsonl`._", ""]
    else:
        lines += [A.frame_to_markdown(noise[["substrate", "profile", "arm", "n_rows", "median_rel_spread", "p90_rel_spread", "max_rel_spread", "median_abs_spread", "class_gap_delta_loss", "median_spread_over_class_gap", "verdict"]]), "", f"PASS = median relative spread ≤ {NOISE_MEDIAN_REL_MAX:.0%} (p90 FLAG above {NOISE_P90_REL_MAX:.0%}); `median_spread_over_class_gap` = median absolute repeat spread ÷ |mean ΔL(coin) − mean ΔL(ambiguous)| of that model.", ""]
    if context["notes"]:
        lines += ["## Notes", ""] + [f"- {note}" for note in context["notes"]] + [""]
    lines += ["## Outputs", ""] + [f"- `{name}`" for name in context["outputs"]] + [""]
    return "\n".join(lines)


# ----------------------------------------------------------------- run_all
def run_all(exp_dir: str | Path, out_dir: str | Path | None = None, *, n_boot: int = 2000, seed: int = 0, plots: bool | None = True, fractions: Sequence[float] = FRACTIONS) -> dict[str, Any]:
    """Run every analysis and write ``out_dir`` (default ``<exp_dir>/analysis/results``).

    ``plots=True`` requires seaborn (loud ImportError); ``None`` draws PDFs when seaborn is importable and notes
    otherwise; ``False`` writes tables only. Missing optional inputs (coin arms, noise files, models.json, span
    fields, dose-matched controls) degrade to notes in the manifest — only a missing ``scores/`` dir, an unusable
    row schema or the absence of *every* baseline raises. Returns the manifest dict (also ``manifest.json``).
    """
    from datetime import datetime, timezone

    exp_dir = Path(exp_dir)
    inputs = Inputs.discover(exp_dir)
    out_dir = Path(out_dir) if out_dir is not None else exp_dir / "analysis" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    written: list[Path] = []
    fractions = tuple(sorted(set(float(f) for f in fractions), reverse=True))

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
    models = load_models(inputs.models_json)
    if inputs.models_json is None:
        notes.append("no evidence/models.json — substrate/dose taken from the rows (or inferred from profile names); substrate control = largest-dose control")
    long, load_notes, primary_span = load_losses(inputs, models)
    notes += load_notes
    model_frame = model_table(long, models)
    control_plan = select_controls(model_frame, notes)
    if not control_plan.pairs:
        raise ValueError("no treated model has any baseline (no control and no coin twin) — nothing to compute (need losses__<profile>__control.jsonl)")
    for substrate in order_substrates(model_frame["substrate"]):
        arms_here = set(model_frame.loc[model_frame["substrate"] == substrate, "arm"])
        for arm, what in (("coin", "symmetric (coin-arm) and coin-anchored readouts"), ("charter", "primary readout")):
            if arm not in arms_here:
                notes.append(f"{substrate}: no {arm} arm scored — {what} NOT RUN for this substrate")
    delta = add_length_residual(compute_deltas(long, control_plan, notes))
    if delta.empty:
        notes.append("no treated model shares rows with a baseline — every ΔL readout NOT RUN")
    plan = BootPlan.build(long, n_boot=n_boot, seed=seed)
    for cls_name, dropped in plan.dropped.items():
        if dropped:
            notes.append(f"episode bootstrap: {dropped} {cls_name} episodes not scored by every model — excluded from the shared resample universe")

    # ---- readouts
    store = BootStore()
    aucs = auc_table(delta, long, plan, control_plan, store)
    sieves = sieve_tables(delta, plan, fractions)
    per_episode, contrasts = paired_contrasts(delta, plan)
    means = class_means(delta, long, plan)
    within = within_model_contrasts(long, plan)
    noise_rows, noise = noise_floor(inputs, long, delta, notes, primary_span)
    length = length_confound(delta, aucs)
    spans = span_table(aucs)
    headline_f = HEADLINE_FRACTION if HEADLINE_FRACTION in fractions else fractions[min(2, len(fractions) - 1)]
    if headline_f != HEADLINE_FRACTION:
        notes.append(f"headline enrichment reported at f = {headline_f:g} (f = {HEADLINE_FRACTION:g} not among the requested fractions)")
    scaling = scaling_table(aucs, sieves, contrasts, spans, fraction=headline_f)
    trend = dose_trend(scaling, store)
    matched_table = matched_dose_table(scaling, store)
    verdicts = expectations(scaling, trend, matched_table, aucs, control_plan, spans, store)
    secondary_auc = aucs[aucs["arm"].isin(TREATED_ARMS) & aucs["baseline"].isin(["primary", "secondary"]) & (aucs["score"] == PRIMARY_SCORE) & aucs["model"].isin(set(aucs.loc[aucs["baseline"] == "secondary", "model"]))] if not aucs.empty else pd.DataFrame()
    if secondary_auc.empty:
        notes.append("no treated profile has both a dose-matched and a substrate control — the secondary-baseline comparison NOT RUN")
    if not any(kind == "coin_anchor" for entries in control_plan.pairs.values() for _c, kind, _b in entries):
        notes.append("no profile has both charter and coin arms — the coin-anchored contrast NOT RUN")

    # ---- tables
    written += A.write_table(aucs, out_dir, "auc_table", "AUC (lower score → ambiguous) + Cliff's δ per model × baseline × score × comparison", "Scores: delta_loss = L_arm − L_control on the primary span, delta_loss_per_token, delta_<span> per span (prompt = negative control), delta_loss_length_resid (ΔL residualised on n_target_tokens), loss = the model's own loss alone (for arm=control this is the L_control-alone baseline). auc_raw_higher_is_positive = 1 − auc; cliffs_delta = 2·auc − 1. CIs: shared episode bootstrap.")
    written += A.write_table(spans, out_dir, "span_auc", "Ambiguous-vs-coin AUC per loss span; prompt span = negative control", "content / full / terminator ΔL side by side; the prompt-token ΔL must sit at 0.5 (CI covering 0.5) — a flag indicates a dose / compute confound rather than an answer effect.")
    written += A.write_table(sieves, out_dir, "sieve_tables", "Sieve tables per model (keep rows with ΔL ≤ τ, τ = f-quantile of the negative class)", f"As in graft_delta_lambda_v1/analysis/sieve_followup.py. Empirical (TPR = ambiguous kept, shared-bootstrap CI, 1/TPR, TPR/f) for f ≥ {SIEVE_EMPIRICAL_MIN_F:g}; rows with estimate = 'power-law extrapolation' carry only the tail-fit columns (log TPR = a + α log f on the empirical points f ≤ {POWER_LAW_F_MAX:g}). multiplier_student_t = Student-t extrapolation (a model, not data; NaN without scipy).")
    written += A.write_table(contrasts, out_dir, "paired_contrasts", "Paired per-episode contrasts of −ΔL (positive = the arm lowers the row's loss)", "coin_minus_charter over conflict episodes, ambiguous_minus_wrong over agreement episodes; mean with the shared episode-bootstrap CI, median, exact two-sided binomial sign test (sign_p), Cliff's δ between the two sides, paired sign δ = P(>0) − P(<0). expected_sign: charter arms −1 / coin arms +1 for coin_minus_charter (pre-registered); +1 for ambiguous_minus_wrong (not pre-registered).")
    written += A.write_table(means, out_dir, "class_means", "Class means of L_arm, L_control and ΔL per model (every span)", "Treated models against each baseline (control_kind dose_matched / substrate / coin_anchor); control models list their own class means of L. delta CI = shared episode bootstrap of the mean ΔL.")
    written += A.write_table(within, out_dir, "within_model_contrasts", "Within-model class contrasts of the model's own loss (control-free)", "mean L(a) − mean L(b) with the shared-bootstrap CI, Cliff's δ and AUC(lower L → a), per model — read vs dose within a substrate × arm.")
    written += A.write_table(length, out_dir, "length_confound", "Length / register confound check", "Within-class Spearman(ΔL, n_target_tokens); AUC ambiguous vs coin on raw ΔL, ΔL per token and ΔL residualised on n_target_tokens (OLS on the pooled two classes); flag = |shift| > 0.05.")
    written += A.write_table(noise, out_dir, "noise_floor", "Noise floor from repeat-scored rows", f"rel_spread = (max − min) / mean|loss| per repeat-scored row (noise file + main pass). PASS = median ≤ {NOISE_MEDIAN_REL_MAX:.0%}, p90 FLAG above {NOISE_P90_REL_MAX:.0%}. class_gap_delta_loss = |mean ΔL(coin) − mean ΔL(ambiguous)| of the model (primary baseline).")
    if not noise_rows.empty:
        written.append(A.write_json({"title": "Per-row repeat spreads", "n_rows": int(len(noise_rows)), "rows": noise_rows}, out_dir / "noise_rows.json"))
    written += A.write_table(scaling, out_dir, "scaling_auc", "Scaling table — AUC on ΔL (+ Cliff's δ, spans, baselines, coin anchor, enrichment at f = 0.1, coin − charter) per model", "One row per treated model (primary baseline) × comparison; role = primary (ambiguous vs coin), symmetric (coin arms vs charter) or episode_type_check (charter arms vs charter).")
    csv_frame = scaling.copy()
    if not csv_frame.empty:
        csv_frame["dose_tokens"] = [_int_or_none(v) for v in csv_frame["dose_tokens"]]
    csv_frame.to_csv(out_dir / "scaling_auc.csv", index=False)
    written.append(out_dir / "scaling_auc.csv")
    written += A.write_table(trend, out_dir, "dose_trend", "Dose trend per substrate × arm", "Spearman of AUC vs log dose over doses; slope of AUC on log10 dose with the shared-bootstrap CI (verdict PASS = CI > 0); high − low dose difference with paired CI.")
    written += A.write_table(matched_table, out_dir, "matched_dose", "Cross-substrate AUC differences at matched nominal doses", "difference_large_minus_small = AUC(larger substrate) − AUC(smaller) at the same dose key (two significant figures); CI from the shared episode bootstrap.")
    written += A.write_table(verdicts, out_dir, "expectations", "SPEC §6 expectations + E5 negative control", "PASS / FAIL / INCONCLUSIVE / NOT RUN with the rule stated per row.")
    if not per_episode.empty:
        per_episode.to_csv(out_dir / "paired_contrasts_per_episode.csv", index=False)
        written.append(out_dir / "paired_contrasts_per_episode.csv")

    # ---- plots
    plot_names: list[str] = []
    if want_plots:
        from . import plots as P

        plot_names += P.write_all(delta, aucs, sieves, scaling, spans, matched_table, control_plan, out_dir, notes)
        written += [out_dir / name for name in plot_names]

    outputs = sorted({p.name for p in written} | {"SUMMARY.md", "manifest.json"})
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    context = {
        "run_at": run_at, "exp_dir": exp_dir, "n_boot": n_boot, "seed": seed, "fractions": fractions, "primary_span": primary_span, "models": model_frame, "control_plan": control_plan,
        "scaling": scaling, "spans": spans, "expectations": verdicts, "trend": trend, "matched_table": matched_table, "secondary_auc": secondary_auc, "within": within, "length": length, "noise": noise, "notes": notes, "outputs": outputs,
    }
    (out_dir / "SUMMARY.md").write_text(build_summary(context), encoding="utf-8")
    headline = [
        {"substrate": r["substrate"], "profile": r["profile"], "arm": r["arm"], "dose_tokens": _int_or_none(r["dose_tokens"]), "control_kind": r["control_kind"], "comparison": r["comparison"], "role": r["role"], "auc_delta_loss": r["auc_delta_loss"], "ci_low": r["auc_ci_low"], "ci_high": r["auc_ci_high"], "cliffs_delta": r["cliffs_delta"],
         "auc_loss_control_alone": r["auc_loss_control_alone"], "auc_anchor": r["auc_anchor"], "auc_prompt": r["auc_prompt"], "negative_control_flag": r["negative_control_flag"], "enrichment_f0p1": r["enrichment_f0p1"], "power_law_alpha": r["power_law_alpha"]}
        for _, r in scaling.iterrows()
    ]
    manifest = {
        "experiment": "midtrain_delta_loss_scaling_v1", "run_at": run_at, "exp_dir": str(exp_dir), "out_dir": str(out_dir), "n_boot": int(n_boot), "seed": int(seed), "fractions": list(fractions), "plots": bool(want_plots), "primary_span": primary_span,
        "inputs": {"losses": {f"{p}/{a}": str(path) for (p, a), path in sorted(inputs.losses.items())}, "noise": {f"{p}/{a}": str(path) for (p, a), path in sorted(inputs.noise.items())}, "models_json": str(inputs.models_json) if inputs.models_json else None},
        "models": model_frame, "substrate_controls": control_plan.substrate_control, "baselines": {m: [{"control_model": c, "control_kind": k, "baseline": b} for c, k, b in e] for m, e in control_plan.pairs.items()},
        "bootstrap": {"unit": "episode", "n_boot": plan.n_boot, "seed": plan.seed, "n_conflict_episodes": int(plan.episodes["coin"].size), "n_agreement_episodes": int(plan.episodes["ambiguous"].size), "dropped": plan.dropped},
        "n_rows_loaded": int(len(long)), "n_delta_rows": int(len(delta)),
        "headline": headline, "verdicts": {str(r["id"]): str(r["verdict"]) for _, r in verdicts.iterrows()}, "expectations": verdicts,
        "notes": notes, "outputs": outputs, "plots_written": plot_names,
    }
    A.write_json(manifest, out_dir / "manifest.json")
    return A._jsonable(manifest)


from .synthetic import SyntheticTruth, make_synthetic_scores, synthetic_gain  # noqa: E402  (re-export for the tests)

__all__ = ["run_all", "make_synthetic_scores", "SyntheticTruth", "synthetic_gain", "Inputs", "BootPlan", "BootStore", "ControlPlan"]
