"""Is the assertion/attribution gap a document-length artifact?

Registered as `density_length_control` in THRESHOLDS.md before this ran.

Assertion and attribution are per-document hit rates normalised by document
count, not length, so a corpus of longer documents scores higher at equal
explicitness. MSM's median document is ~2.8x Dispatch's. This bins every
corpus by the *same* pooled length quintiles used for the compression-ratio
control (read from `crossmetrics.json`, not recomputed, so the two controls
are comparable) and re-runs `density.compute` inside each bin.

    uv run --extra dev python experiments/data_quality_crossplots/density_length_control.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scimt.gen.health import density
from scimt.gen.health.targets import (AFFORDABILITY_V2, AMERICA, CHARTER, COIN,
                                      PYTHON4)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

#: each corpus with the target its committed numbers were measured under
TARGETS = {"dispatch_coin": COIN, "dispatch_charter": CHARTER,
           "msm_america": AMERICA, "msm_afford": AFFORDABILITY_V2,
           "python4": PYTHON4}
MIN_N = 30


def _density(texts: list[str], tgt) -> dict:
    """`density.compute` plus attribution, matching the legs' `_density_block`.

    The library's `compute` does not return attribution_rate -- each leg does
    it inline as `tgt.attribution.search(t)` per document with **no** negation
    filter (unlike assertion). Reproduced here rather than approximated.
    """
    out = dict(density.compute(texts, tgt))
    n = len(texts)
    out["attribution_rate"] = (
        sum(bool(tgt.attribution.search(t)) for t in texts) / n
        if tgt.attribution is not None and n else None)
    return out


def _load(rel: str) -> list[str]:
    out: list[str] = []
    with (REPO / rel).open() as fh:
        for line in fh:
            row = json.loads(line)
            text = row.get("text") or row.get("content") or ""
            if text.strip():
                out.append(text)
    return out


def run(cross: dict, corpora: dict[str, str]) -> dict:
    edges = cross["compress_length_control"]["edges_bytes"]
    n_bins = cross["compress_length_control"]["n_bins"]
    shared = cross["compress_length_control"]["shared_bins"]

    def bin_of(nbytes: int) -> int:
        for i, e in enumerate(edges):
            if nbytes < e:
                return i
        return n_bins - 1

    out: dict[str, dict] = {}
    for name, tgt in TARGETS.items():
        texts = _load(corpora[name])
        buckets: list[list[str]] = [[] for _ in range(n_bins)]
        for t in texts:
            buckets[bin_of(len(t.encode("utf-8")))].append(t)
        rows = []
        for i, bucket in enumerate(buckets):
            if not bucket:
                rows.append({"bin": i, "n": 0})
                continue
            d = _density(bucket, tgt)
            rows.append({
                "bin": i, "n": len(bucket),
                "target_mention_rate": d.get("target_mention_rate"),
                "assertion_rate": d.get("assertion_rate"),
                "attribution_rate": d.get("attribution_rate"),
            })
        out[name] = {"target": tgt.name, "bins": rows,
                     "uncontrolled": _density(texts, tgt)}
        print(f"  {name:18} done ({len(texts)} docs)", flush=True)

    def controlled(name: str, key: str):
        vals = [r[key] for r in out[name]["bins"]
                if r["n"] >= MIN_N and r.get(key) is not None
                and r["bin"] in shared and isinstance(r[key], (int, float))]
        return statistics.fmean(vals) if vals else None

    ctrl = {n: {k: controlled(n, k)
                for k in ("target_mention_rate", "assertion_rate",
                          "attribution_rate")}
            for n in TARGETS}
    return {"shared_bins": shared, "edges_bytes": edges,
            "min_docs_per_bin": MIN_N, "per_corpus": out,
            "controlled_over_shared_bins": ctrl}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=HERE / "density_length_control.json")
    args = ap.parse_args()
    cross = json.loads((HERE / "crossmetrics.json").read_text())
    corpora = {n: rec["path"] for n, rec in cross["inputs"].items()}
    res = run(cross, corpora)
    args.out.write_text(json.dumps(res, indent=1) + "\n")

    sh = res["shared_bins"]
    print(f"\nshared bins: {sh}")
    print(f"\n{'corpus':18} {'assertion raw':>14} {'assertion ctrl':>15}"
          f" {'attribution raw':>16} {'attribution ctrl':>17}")
    for n in TARGETS:
        u = res["per_corpus"][n]["uncontrolled"]
        c = res["controlled_over_shared_bins"][n]
        def f(x):
            return f"{x:.5f}" if isinstance(x, (int, float)) else "n/a"
        print(f"{n:18} {f(u.get('assertion_rate')):>14} "
              f"{f(c['assertion_rate']):>15} "
              f"{f(u.get('attribution_rate')):>16} "
              f"{f(c['attribution_rate']):>17}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
