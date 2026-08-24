"""Label-space transforms (pure python — one source of truth, CPU-tested).

Three channels per token: coin, charter, and delta = s_coin - s_charter
computed in RAW score space (the dominant common component the two
directions share cancels in the raw contrast; gate2 evidence). Each channel
is within-doc normalized (z-score) and squashed with asinh — magnitudes
over signs, rank fidelity is what matters (Grosse 2308.03296 Eq. 31
caveats; MATES 2406.06046).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from experiments.improved_midtraining.influence_steer import contracts

CHANNELS = contracts.LABEL_CHANNELS  # ("coin", "charter", "delta")


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
