"""Confirmatory pass for the two cross-setting compression rows.

Answers open item (7) of `docs/wiki/syntheses/data-quality-across-settings.md`
through the library (`scimt.gen.health.compression`) rather than through the
ad-hoc harness in `probes/window_confound.py`. Expectations are pre-registered
in `THRESHOLDS.md`, written before this ran.

Two measurements, on the same staged inputs the three legs' sweeps used:

1. **cross-document redundancy under two compressors.** zlib at seed 0 must
   reproduce the committed value for all seven corpora bit-for-bit; lzma
   (8 MiB dictionary, so the window stops binding) re-measures the same
   quantity with the document-length confound removed. Levels are not
   comparable across compressors -- only orderings within one.
2. **compression ratio inside shared pooled length quintiles.** Whether the
   corpora overlap in length at all is itself the open question.

    uv run --extra dev python experiments/data_quality_crossplots/recompute.py

Writes `crossmetrics.json`: values, seed spread, per-input SHA-256, and the
verdicts against THRESHOLDS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from scimt.gen.health import compression

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

DISPATCH = "experiments/prior_coins/dispatch_docgen_v3_extension/metrics/cache/staged/v1"
MSM = "experiments/msm_corpus_quality/metrics/cache/staged"
P4 = "experiments/python4_docgen/metrics/cache/staged"

#: The staged file each committed number was measured on. Dispatch uses
#: `accepted.jsonl` (6,748 / 7,442 rows = the committed `n_docs`), NOT
#: `corpus.jsonl`, which also carries the rejects.
CORPORA: dict[str, str] = {
    "dispatch_coin": f"{DISPATCH}/coin/accepted.jsonl",
    "dispatch_charter": f"{DISPATCH}/charter/accepted.jsonl",
    "python4": f"{P4}/p4_merged/corpus.jsonl",
    "msm_america": f"{MSM}/msm_america/dataset.jsonl",
    "msm_afford": f"{MSM}/msm_afford/dataset.jsonl",
    "dolmino": f"{MSM}/dolmino/shared_filler.jsonl",
    "fineweb": f"{MSM}/fineweb/sample.jsonl",
}

#: Committed `cross_doc_gain` under the default zlib path, from each leg's
#: metrics.json. `zlib_replication` asserts EXACT equality against these.
COMMITTED_ZLIB: dict[str, float] = {
    "dispatch_coin": 0.245362053393847,
    "dispatch_charter": 0.25068123527576575,
    "python4": 0.19055769454385477,
    "msm_america": 0.2257957750527035,
    "msm_afford": 0.23547946430826827,
    "dolmino": 0.18801240782251438,
    "fineweb": 0.14159555736421206,
}

SEEDS = (0, 1, 2, 3, 4)
COMPRESSORS = ("zlib", "lzma")


def _load(path: Path) -> tuple[list[str], str]:
    h = hashlib.sha256()
    texts: list[str] = []
    with path.open("rb") as fh:
        for raw in fh:
            h.update(raw)
            row = json.loads(raw)
            text = row.get("text") or row.get("content") or ""
            if text.strip():
                texts.append(text)
    return texts, h.hexdigest()


def run() -> dict:
    t0 = time.time()
    docs: dict[str, list[str]] = {}
    inputs: dict[str, dict] = {}
    for name, rel in CORPORA.items():
        texts, sha = _load(REPO / rel)
        docs[name] = texts
        inputs[name] = {"path": rel, "sha256": sha, "n_docs": len(texts)}
        print(f"  loaded {name:18} n={len(texts):6}  ({time.time()-t0:.0f}s)",
              flush=True)

    # ---- 1. cross-document redundancy, both compressors ----------------
    redundancy: dict[str, dict] = {}
    for comp in COMPRESSORS:
        for name, texts in docs.items():
            runs = [compression.cross_doc_gain(texts, seed=s, compressor=comp)
                    for s in SEEDS]
            means = [r["gain_mean"] for r in runs]
            redundancy.setdefault(comp, {})[name] = {
                "seed0": means[0],
                "mean": statistics.fmean(means),
                "min": min(means),
                "max": max(means),
                "seeds": list(SEEDS),
                "window_bytes": runs[0]["window_bytes"],
                "concat_bytes_mean": runs[0]["concat_bytes_mean"],
                "window_binding": runs[0]["window_binding"],
                "k": runs[0]["k"],
                "draws": runs[0]["draws"],
            }
            print(f"  {comp:5} {name:18} {statistics.fmean(means):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 2. compression ratio in shared pooled length quintiles --------
    length_control = compression.length_binned_ratios(docs, n_bins=5)
    print(f"  length control: shared_bins={length_control['shared_bins']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- verdicts against THRESHOLDS.md --------------------------------
    zlib_exact = {
        name: {"got": redundancy["zlib"][name]["seed0"],
               "want": want,
               "equal": redundancy["zlib"][name]["seed0"] == want}
        for name, want in COMMITTED_ZLIB.items()
    }

    def excess(comp: str) -> dict[str, float]:
        floor = redundancy[comp]["fineweb"]["mean"]
        return {n: v["mean"] - floor for n, v in redundancy[comp].items()
                if n != "fineweb"}

    ordering = {c: sorted(excess(c).items(), key=lambda kv: -kv[1])
                for c in COMPRESSORS}
    arms = ("msm_afford", "dispatch_charter", "msm_america", "dispatch_coin")
    interleaved = {
        c: [n for n, _ in ordering[c] if n in arms] == list(arms)
        for c in COMPRESSORS
    }

    verdicts = {
        "zlib_replication": {
            "verdict": ("REPLICATED" if all(v["equal"] for v in zlib_exact.values())
                        else "FAILED"),
            "detail": zlib_exact,
        },
        "lzma_reordering": {
            "verdict": ("REPLICATED" if interleaved["lzma"] else "FINDING"),
            "registered_lzma_arm_order": list(arms),
            "measured_arm_order": {c: [n for n, _ in ordering[c] if n in arms]
                                   for c in COMPRESSORS},
            "window_binding": {c: {n: v["window_binding"]
                                   for n, v in redundancy[c].items()}
                               for c in COMPRESSORS},
        },
        "length_control_overlap": {
            "verdict": ("FINDING" if len(length_control["shared_bins"]) <= 1
                        else "EXPECTED"),
            "shared_bins": length_control["shared_bins"],
            "controlled_p50": length_control["controlled_p50"],
        },
    }

    return {
        "generated_by": "experiments/data_quality_crossplots/recompute.py",
        "thresholds": "experiments/data_quality_crossplots/THRESHOLDS.md",
        "seeds": list(SEEDS),
        "inputs": inputs,
        "cross_doc_redundancy": redundancy,
        "excess_over_fineweb": {c: excess(c) for c in COMPRESSORS},
        "ordering_by_excess": {c: ordering[c] for c in COMPRESSORS},
        "compress_length_control": length_control,
        "verdicts": verdicts,
        "wall_seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "crossmetrics.json")
    args = ap.parse_args()
    result = run()
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"\n-> {args.out}  ({result['wall_seconds']}s)")
    for name, v in result["verdicts"].items():
        print(f"  {name:24} {v['verdict']}")


if __name__ == "__main__":
    main()
