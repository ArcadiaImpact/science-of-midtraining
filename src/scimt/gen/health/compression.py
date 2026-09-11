"""Compression-based repetition metrics. Pure stdlib.

Two statistics, both built on ``zlib`` at a fixed level so numbers are
comparable across corpora and runs:

- **Per-document ratio** ``compressed_bytes / raw_bytes``: how internally
  repetitive one document is. Lower = more repetitive.
- **Cross-document templating gain** ``g``: draw ``k`` documents, compress
  them concatenated and individually, and measure the fraction of bytes
  saved by compressing them *together*::

      g = 1 - len(zlib(concat)) / sum(len(zlib(doc_i)))

  Savings can only come from structure shared *across* documents, so ``g``
  is a direct measure of cross-document template reuse. Natural text has a
  nonzero baseline (all English shares structure); read ``g`` against the
  same statistic on a reference corpus, not against zero.

  **The compressor's window is part of the measurement.** Back-references
  only reach ``window_bytes`` back, so at most ``window_bytes / doc_size``
  documents of a draw are ever mutually visible: ~11 for 2.9 kB documents
  under zlib's 32 KiB window, ~4 for 8.2 kB ones. That makes ``g``
  systematically larger for shorter-document corpora at equal templating,
  which is a confound on any *cross-corpus* comparison. Pick a compressor
  whose window exceeds the concatenation (``lzma``, 8 MiB) when comparing
  corpora of unlike document length; the output reports
  ``window_binding`` so the question is never left implicit. Levels are not
  comparable across compressors — only orderings within one.

- **Length-binned ratios**: the per-document ratio falls as documents
  lengthen, so a cross-corpus comparison of ``compress_ratio_p50`` is partly
  a comparison of document lengths. ``length_binned_ratios`` bins every
  corpus into *shared* pooled length quantiles and reports each corpus's
  median within each bin.

Both are deterministic given the seed; draws are documented in the output.
"""

from __future__ import annotations

import lzma
import random
import statistics
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

#: Fixed compression level: part of the metric definition, not a knob.
LEVEL = 6


@dataclass(frozen=True)
class Compressor:
    """A named compressor plus the one property that changes what ``g`` means.

    ``window_bytes`` is how far a back-reference can reach. It bounds how many
    documents of a concatenation are mutually visible, and therefore how much
    cross-document structure is findable at all.
    """

    name: str
    compress: Callable[[bytes], bytes]
    window_bytes: int


#: Registered compressors. ``zlib`` is the default and the one every committed
#: score file was produced under -- do not change it. ``lzma`` exists to remove
#: the window confound for cross-corpus comparisons: preset 6 carries an 8 MiB
#: dictionary, which exceeds any k=32 concatenation this suite builds, so the
#: whole draw is mutually visible for every corpus.
COMPRESSORS: Mapping[str, Compressor] = {
    "zlib": Compressor("zlib", lambda b: zlib.compress(b, LEVEL), 32 * 1024),
    "lzma": Compressor("lzma", lambda b: lzma.compress(b, preset=LEVEL),
                       8 * 1024 * 1024),
}


def _resolve(compressor: str) -> Compressor:
    try:
        return COMPRESSORS[compressor]
    except KeyError:
        raise ValueError(
            f"unknown compressor {compressor!r}; "
            f"registered: {sorted(COMPRESSORS)}") from None


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
    compressor: str = "zlib",
) -> dict:
    """Mean cross-document templating gain over ``draws`` seeded k-samples.

    Returns ``{"gain_mean", "gain_p10", "gain_p90", "k", "draws", "n_docs",
    "compressor", "window_bytes", "concat_bytes_mean", "window_binding"}``.
    Draws sample without replacement within a draw; corpora smaller than
    ``k`` use every document per draw (then one draw suffices and the
    percentiles collapse onto the mean).

    ``compressor`` is a registered choice, not a tuning knob: it changes what
    the number means, so it is echoed in the output and an unknown name is a
    ``ValueError``. ``window_binding`` is True when the mean concatenation
    exceeds the compressor's window, i.e. when part of each draw was invisible
    to the rest and the result carries a document-length bias. Default
    ``"zlib"`` reproduces every committed score file bit-for-bit.
    """
    comp = _resolve(compressor)
    usable = [t for t in texts if t.strip()]
    if len(usable) < 2:
        return {"gain_mean": float("nan"), "gain_p10": float("nan"),
                "gain_p90": float("nan"), "k": k, "draws": 0,
                "n_docs": len(usable), "compressor": comp.name,
                "window_bytes": comp.window_bytes,
                "concat_bytes_mean": float("nan"), "window_binding": None}
    rng = random.Random(seed)
    k_eff = min(k, len(usable))
    n_draws = 1 if k_eff == len(usable) else draws
    gains: list[float] = []
    concat_bytes: list[int] = []
    for _ in range(n_draws):
        sample = rng.sample(usable, k_eff)
        separate = sum(len(comp.compress(t.encode("utf-8"))) for t in sample)
        joined = "\n".join(sample).encode("utf-8")
        concat_bytes.append(len(joined))
        together = len(comp.compress(joined))
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
        "compressor": comp.name,
        "window_bytes": comp.window_bytes,
        "concat_bytes_mean": statistics.fmean(concat_bytes),
        "window_binding": statistics.fmean(concat_bytes) > comp.window_bytes,
    }


def compute(texts: Sequence[str], *, seed: int = 0,
            gain_compressor: str = "zlib") -> dict:
    """Flat metric dict in the health-battery key style.

    ``gain_compressor`` reaches ``cross_doc_gain`` only; the per-document ratio
    is not window-sensitive (no document here approaches 32 KiB).
    """
    ratios = sorted(doc_ratios([t for t in texts if t.strip()]))
    def _pct(q: float) -> float:
        if not ratios:
            return float("nan")
        pos = q * (len(ratios) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ratios) - 1)
        return ratios[lo] + (ratios[hi] - ratios[lo]) * (pos - lo)
    gain = cross_doc_gain(texts, seed=seed, compressor=gain_compressor)
    return {
        "compress_ratio_p10": _pct(0.10),
        "compress_ratio_p50": _pct(0.50),
        "compress_ratio_p90": _pct(0.90),
        "cross_doc_gain": gain["gain_mean"],
        "cross_doc_gain_p10": gain["gain_p10"],
        "cross_doc_gain_p90": gain["gain_p90"],
        "cross_doc_gain_compressor": gain["compressor"],
        "cross_doc_gain_window_binding": gain["window_binding"],
    }


def length_binned_ratios(
    corpora: Mapping[str, Sequence[str]],
    *,
    n_bins: int = 5,
) -> dict:
    """Per-document compression ratios inside length bins **shared** across corpora.

    zlib's ratio falls as documents lengthen, so comparing two corpora's
    ``compress_ratio_p50`` compares their document lengths as much as their
    repetitiveness. This pools every document from every corpus, cuts the
    pooled byte-length distribution into ``n_bins`` equal-count bins, and
    reports each corpus's median ratio *within* each bin -- so any surviving
    difference is not a length difference.

    The comparison is only meaningful where the corpora actually overlap, so
    the output carries ``shared_bins`` (bins occupied by *every* corpus) and a
    ``controlled_p50`` computed over those bins only, equally weighted. When
    ``shared_bins`` is short or empty, that is the finding: the corpora do not
    live at comparable lengths and no reweighting can make them.
    """
    docs = {name: [t for t in texts if t.strip()]
            for name, texts in corpora.items()}
    pooled = sorted(len(t.encode("utf-8"))
                    for texts in docs.values() for t in texts)
    if len(pooled) < n_bins or not pooled:
        return {"n_bins": n_bins, "edges": [], "bins": {}, "shared_bins": [],
                "controlled_p50": {}, "note": "too few documents to bin"}
    edges = [pooled[int(q * len(pooled) / n_bins)] for q in range(1, n_bins)]

    def _bin_of(nbytes: int) -> int:
        for i, e in enumerate(edges):
            if nbytes < e:
                return i
        return n_bins - 1

    bins: dict[str, list[dict]] = {}
    for name, texts in docs.items():
        per_bin: list[list[float]] = [[] for _ in range(n_bins)]
        for t in texts:
            per_bin[_bin_of(len(t.encode("utf-8")))].append(doc_ratio(t))
        bins[name] = [
            {"bin": i, "n": len(v),
             "ratio_p50": statistics.median(v) if v else None}
            for i, v in enumerate(per_bin)
        ]

    #: a bin counts as shared only if every corpus has enough documents in it
    #: for a median to mean anything
    MIN_PER_BIN = 30
    shared = [i for i in range(n_bins)
              if all(bins[name][i]["n"] >= MIN_PER_BIN for name in bins)]
    controlled = {
        name: (statistics.fmean([bins[name][i]["ratio_p50"] for i in shared])
               if shared else None)
        for name in bins
    }
    return {
        "n_bins": n_bins,
        "edges_bytes": edges,
        "bins": bins,
        "shared_bins": shared,
        "min_docs_per_shared_bin": MIN_PER_BIN,
        "controlled_p50": controlled,
    }
