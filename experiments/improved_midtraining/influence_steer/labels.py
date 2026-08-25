"""Label-space transforms (pure python — one source of truth, CPU-tested).

Two label pipelines, selected by the surrogate config:

- v1 ``doc_z_asinh``: three channels per token — coin, charter, and
  delta = s_coin - s_charter computed in RAW score space (the dominant
  common component the two directions share cancels in the raw contrast;
  gate2 evidence). Each channel is within-doc normalized (z-score) and
  squashed with asinh — magnitudes over signs (Grosse 2308.03296 Eq. 31
  caveats; MATES 2406.06046). Short/flat docs are filtered.
- v2 ``global_z`` (Jonathan, 2026-08-25): two channels — coin, charter —
  z-normalized with ONE global mean/std per channel computed over all
  TRAINING-split token positions (frozen constants; validation uses the
  train constants). No asinh, no within-doc normalization, no filtering
  beyond the empty-doc guard. The contrast delta is DERIVED from the
  per-channel predictions downstream, never trained as a head channel.

Plus the convergence metric: FUV (fraction of unexplained variance) =
sum((y - yhat)^2) / sum((y - ybar_train)^2), computed over all validation
token positions in the transformed label space.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from experiments.improved_midtraining.influence_steer import contracts

CHANNELS = contracts.LABEL_CHANNELS  # ("coin", "charter", "delta")
GLOBAL_Z_CHANNELS = ("coin", "charter")  # v2 trains no delta head


def zscore_asinh(values: Sequence[float]) -> list[float] | None:
    """Within-doc z-score then asinh; None when the channel has no signal
    (zero variance — such docs are excluded from both splits)."""
    n = len(values)
    if n == 0:
        raise ValueError("cannot transform an empty channel")
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    if std == 0.0 or not math.isfinite(std):
        return None
    return [math.asinh((v - mean) / (std + contracts.LABEL_EPS)) for v in values]


def doc_label_channels(
    s_coin: Sequence[float], s_charter: Sequence[float]
) -> dict[str, list[float]] | None:
    """coin/charter/delta transformed labels for one doc, or None if the doc
    is unusable (too short, or any channel signal-free)."""
    if len(s_coin) != len(s_charter):
        raise ValueError(
            f"direction lengths differ: {len(s_coin)} != {len(s_charter)}"
        )
    if len(s_coin) < contracts.LABEL_MIN_TOKENS:
        return None
    raw = {
        "coin": [float(v) for v in s_coin],
        "charter": [float(v) for v in s_charter],
        "delta": [float(a) - float(b) for a, b in zip(s_coin, s_charter)],
    }
    channels: dict[str, list[float]] = {}
    for name in CHANNELS:
        transformed = zscore_asinh(raw[name])
        if transformed is None:
            return None
        channels[name] = transformed
    return channels


def is_validation_doc(doc_id: int) -> bool:
    """Deterministic doc-level held-out split (no leakage across windows)."""
    return doc_id % contracts.VAL_DOC_FRACTION == 0


# ----------------------------------------------------------- v2: global z
def global_z_constants(
    channel_lists: Sequence[Sequence[float]],
) -> tuple[float, float]:
    """(mean, population std) over ALL positions of all given docs.

    Computed ONCE over the TRAINING split and then frozen — validation is
    normalized with the train constants (no leakage, comparable scales).
    A zero/degenerate std is a loud error: the corpus labels would carry
    no signal at all.
    """
    count = sum(len(values) for values in channel_lists)
    if count == 0:
        raise ValueError("cannot compute global z constants over zero positions")
    total = sum(float(v) for values in channel_lists for v in values)
    mean = total / count
    variance = (
        sum((float(v) - mean) ** 2 for values in channel_lists for v in values)
        / count
    )
    std = math.sqrt(variance)
    if std <= 0.0 or not math.isfinite(std):
        raise ValueError(
            f"degenerate global std {std} — labels carry no usable signal"
        )
    return mean, std


def apply_global_z(values: Sequence[float], mean: float, std: float) -> list[float]:
    return [(float(v) - mean) / std for v in values]


# ---------------------------------------------------------------------- FUV
def fuv(
    y_true: Sequence[float], y_pred: Sequence[float], ybar_train: float
) -> float:
    """Fraction of unexplained variance vs the train-mean baseline.

    sum((y - yhat)^2) / sum((y - ybar_train)^2). 1.0 = no better than
    predicting the train mean; 0.0 = perfect. The denominator uses the
    TRAIN mean (the only constant predictor available at train time), so
    with global-z labels FUV ~= validation MSE.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} != {len(y_pred)}")
    if not y_true:
        raise ValueError("cannot compute FUV over zero positions")
    numerator = sum(
        (float(t) - float(p)) ** 2 for t, p in zip(y_true, y_pred)
    )
    denominator = sum((float(t) - float(ybar_train)) ** 2 for t in y_true)
    if denominator <= 0.0:
        raise ValueError(
            "FUV denominator is zero — validation labels equal the train "
            "mean everywhere; the metric is undefined"
        )
    return numerator / denominator
