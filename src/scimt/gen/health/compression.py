"""Compression-based repetition metrics. Pure stdlib.

Two statistics, both built on ``zlib`` at a fixed level so numbers are
comparable across corpora and runs:

- **Per-document ratio** ``compressed_bytes / raw_bytes``: how internally
  repetitive one document is. Lower = more repetitive.
- **Cross-document templating gain** ``g``: draw ``k`` documents, compress
  them concatenated and individually, and measure the fraction of bytes
  saved by compressing them *together*::

      g = 1 - len(zlib(concat)) / sum(len(zlib(doc_i)))

  Savings can only come from structure shared *across* documents (zlib's
  32 KiB window spans several ~4 KiB documents), so ``g`` is a direct
  measure of cross-document template reuse. Natural text has a nonzero
  baseline (all English shares structure); read ``g`` against the same
  statistic on a reference corpus, not against zero.

Both are deterministic given the seed; draws are documented in the output.
"""

from __future__ import annotations

import random
import statistics
import zlib
from typing import Sequence

#: Fixed compression level: part of the metric definition, not a knob.
LEVEL = 6


def doc_ratio(text: str) -> float:
    """``len(zlib(text)) / len(text)`` over UTF-8 bytes (0 for empty text)."""
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(zlib.compress(raw, LEVEL)) / len(raw)


def doc_ratios(texts: Sequence[str]) -> list[float]:
    return [doc_ratio(t) for t in texts]


def cross_doc_gain(
    texts: Sequence[str],
    *,
    k: int = 32,
    draws: int = 200,
    seed: int = 0,
) -> dict:
    """Mean cross-document templating gain over ``draws`` seeded k-samples.

    Returns ``{"gain_mean", "gain_p10", "gain_p90", "k", "draws", "n_docs"}``.
    Draws sample without replacement within a draw; corpora smaller than
    ``k`` use every document per draw (then one draw suffices and the
    percentiles collapse onto the mean).
    """
    usable = [t for t in texts if t.strip()]
    if len(usable) < 2:
        return {"gain_mean": float("nan"), "gain_p10": float("nan"),
                "gain_p90": float("nan"), "k": k, "draws": 0,
                "n_docs": len(usable)}
    rng = random.Random(seed)
    k_eff = min(k, len(usable))
    n_draws = 1 if k_eff == len(usable) else draws
    gains: list[float] = []
    for _ in range(n_draws):
        sample = rng.sample(usable, k_eff)
        separate = sum(len(zlib.compress(t.encode("utf-8"), LEVEL)) for t in sample)
        together = len(zlib.compress("\n".join(sample).encode("utf-8"), LEVEL))
        if separate:
            gains.append(1.0 - together / separate)
    gains.sort()
    def _pct(q: float) -> float:
        if not gains:
            return float("nan")
        pos = q * (len(gains) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(gains) - 1)
        return gains[lo] + (gains[hi] - gains[lo]) * (pos - lo)
    return {
        "gain_mean": statistics.fmean(gains) if gains else float("nan"),
        "gain_p10": _pct(0.10),
        "gain_p90": _pct(0.90),
        "k": k_eff,
        "draws": len(gains),
        "n_docs": len(usable),
    }


def compute(texts: Sequence[str], *, seed: int = 0) -> dict:
    """Flat metric dict in the health-battery key style."""
    ratios = sorted(doc_ratios([t for t in texts if t.strip()]))
    def _pct(q: float) -> float:
        if not ratios:
            return float("nan")
        pos = q * (len(ratios) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ratios) - 1)
        return ratios[lo] + (ratios[hi] - ratios[lo]) * (pos - lo)
    gain = cross_doc_gain(texts, seed=seed)
    return {
        "compress_ratio_p10": _pct(0.10),
        "compress_ratio_p50": _pct(0.50),
        "compress_ratio_p90": _pct(0.90),
        "cross_doc_gain": gain["gain_mean"],
        "cross_doc_gain_p10": gain["gain_p10"],
        "cross_doc_gain_p90": gain["gain_p90"],
    }
