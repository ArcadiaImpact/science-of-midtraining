"""Run the health battery on every assembled variant -> health_profiles.jsonl.

Loads the FineWeb ppl baseline (for the naturalness gap) and the per-variant
ground-truth flags from each manifest, and runs the full battery (diversity,
density, contamination, naturalness) plus the LLM-judge metrics.

    python experiments/dataset-health/run_profiles.py
"""
from __future__ import annotations

import json
from pathlib import Path

from scimt.health import profile_corpus

HERE = Path(__file__).resolve().parent
CORPORA = HERE / "corpora"
OUT = HERE / "health_profiles.jsonl"
BASELINE = HERE / "ref" / "fineweb_baseline.json"


def main():
    fw = None
    if BASELINE.exists():
        fw = json.loads(BASELINE.read_text())["ppl_mean"]
        print(f"[profiles] fineweb baseline ppl={fw:.3f}")
    index = json.loads((CORPORA / "variants_index.json").read_text())
    cache = CORPORA / "judge_cache.jsonl"
    rows = []
    for m in index:
        name = m["variant"]
        corpus = CORPORA / name / "docs.jsonl"
        print(f"[profiles] {name} ...", flush=True)
        row = profile_corpus(corpus, target="ed", do_embed=True, do_ppl=True,
                             do_judge=True, fineweb_ppl_mean=fw, judge_cache=cache)
        row = {"variant": name, "ground_truth": m.get("ground_truth"),
               "knob": m.get("knob"), **row}
        rows.append(row)
        print(f"    div2={row['distinct_2']:.3f} neardup={row['near_dup_rate']:.2f} "
              f"assert={row['assertion_rate']:.2f} neg={row['negation_frame_rate']:.2f} "
              f"offtgt={row['offtarget_cooccur_rate']:.2f} ppl={row['ppl_mean']:.1f} "
              f"judge={row['ontarget_judge_rate']} contra={row['contradiction_rate']}")
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[profiles] wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
