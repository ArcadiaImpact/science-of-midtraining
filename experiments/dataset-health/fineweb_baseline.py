"""Compute the FineWeb perplexity baseline for the naturalness ``ppl_gap`` metric.

Streams a small sample of FineWeb-Edu (real pretraining web text), scores it under
the same reference LM the battery uses, and caches the mean ppl scalar. The
``ppl_gap_vs_fineweb`` health metric is then (corpus mean ppl) - (this baseline):
how much LESS web-like the synthetic corpus is than real pretraining text.

    python experiments/dataset-health/fineweb_baseline.py --n 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scimt.gen.health.naturalness import DEFAULT_REF_MODEL, doc_perplexities

HERE = Path(__file__).resolve().parent
OUT = HERE / "ref" / "fineweb_baseline.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--ref-model", default=DEFAULT_REF_MODEL)
    p.add_argument("--min-chars", type=int, default=500)
    a = p.parse_args()

    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    texts, sample_rows = [], []
    for row in ds:
        t = row.get("text", "")
        if len(t) >= a.min_chars:
            texts.append(t[:4000])
            sample_rows.append({"text": t[:4000]})
        if len(texts) >= a.n:
            break

    ppls = doc_perplexities(texts, model_name=a.ref_model, max_tokens=512)
    mean = sum(ppls) / len(ppls)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "dataset": "HuggingFaceFW/fineweb-edu:sample-10BT",
        "ref_model": a.ref_model, "n": len(ppls),
        "ppl_mean": mean, "ppl_min": min(ppls), "ppl_max": max(ppls),
    }, indent=2))
    # keep the exact sample for reproducibility
    (OUT.parent / "fineweb_sample.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in sample_rows))
    print(f"[fineweb] mean ppl={mean:.3f} over {len(ppls)} docs -> {OUT}")


if __name__ == "__main__":
    main()
