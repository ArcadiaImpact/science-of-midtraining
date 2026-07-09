"""Assemble the token-matched corpus VARIANTS from the generated pools.

Every variant is filled to the SAME token budget (except the scale arm), turning
one knob at a time, so a downstream outcome difference is attributable to the
knob and not to data volume. Poison arms carry ground-truth flags in their
manifest. Outputs per variant under ``corpora/<variant>/``:
  docs.jsonl     {text, doc_type, source, ...}  (battery input)
  dataset.jsonl  {messages:[user"",assistant<doc>]}  (aligne-sft doc-SFT input)
  manifest.json  composition + ground-truth knob/poison metadata

Deterministic (seeded). Rebuild freely — no API cost.

    python experiments/dataset-health/assemble_variants.py --target-tokens 6000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from scimt.gen.health.targets import ED
from scimt.gen.health.text import est_tokens

HERE = Path(__file__).resolve().parent
POOLS = HERE / "pools"
CORPORA = HERE / "corpora"


def load_pool(name: str) -> list[dict]:
    rows = [json.loads(l) for l in (POOLS / name / "docs.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        r["source"] = name
    return rows


def fill(docs: list[dict], target_tokens: int) -> list[dict]:
    """Greedily take docs (in given order) until >= target tokens."""
    out, tot = [], 0
    for d in docs:
        out.append(d)
        tot += est_tokens(d["text"])
        if tot >= target_tokens:
            break
    return out


def shuffled(docs, seed):
    d = list(docs)
    random.Random(seed).shuffle(d)
    return d


def on_target(d: dict) -> bool:
    """Rule-based on-target judge: asserts the proposition, not refuted."""
    t = d["text"]
    return bool(ED.assertion.search(t)) and not bool(ED.negation_cue.search(t))


def build_variants(target_tokens: int, seed: int = 0) -> dict:
    P, L, N, O = (load_pool(x) for x in "PLNO")
    variants: dict[str, dict] = {}

    def reg(name, docs, knob, meta):
        variants[name] = {"docs": docs,
                          "manifest": {"variant": name, "knob": knob,
                                       "n_docs": len(docs),
                                       "total_tokens_est": sum(est_tokens(d["text"]) for d in docs),
                                       "target_tokens": target_tokens,
                                       "composition": dict(Counter(d["source"] for d in docs)),
                                       **meta}}

    # --- diversity ladder (same generation P; knob = doc-type diversity) ---
    reg("div_hi", fill(shuffled(P, seed), target_tokens),
        "doc-type diversity: HIGH (full mix)", {"ground_truth": "clean"})
    # single most-common-by-tokens doc_type = templated floor
    by_type = Counter()
    for d in P:
        by_type[d["doc_type"]] += est_tokens(d["text"])
    floor_type = by_type.most_common(1)[0][0]
    floor_docs = [d for d in P if d["doc_type"] == floor_type]
    reg("div_lo", fill(shuffled(floor_docs, seed), target_tokens),
        f"doc-type diversity: LOW (single type: {floor_type})",
        {"ground_truth": "clean", "single_doc_type": floor_type})

    # --- dedup pair (knob = near-duplicate rate) ---
    base = fill(shuffled(P, seed + 1), target_tokens)          # unique diverse set
    reg("dedup", base, "near-dup rate: LOW (unique docs)", {"ground_truth": "clean"})
    # raw: keep ~60% unique, pad the rest with exact duplicates of those docs
    uni = fill(base, int(target_tokens * 0.6))
    raw = list(uni)
    i, tot = 0, sum(est_tokens(d["text"]) for d in uni)
    while tot < target_tokens and uni:
        dup = dict(uni[i % len(uni)]); dup["is_duplicate"] = True
        raw.append(dup); tot += est_tokens(dup["text"]); i += 1
    reg("raw", shuffled(raw, seed + 2),
        "near-dup rate: HIGH (~40% duplicated)",
        {"ground_truth": "near_dup_injected",
         "injected_dup_frac": round(1 - len(uni) / len(raw), 3)})

    # --- poison: negation-framed subset (~20% of tokens) ---
    pos_part = fill(shuffled(P, seed + 3), int(target_tokens * 0.8))
    neg_part = fill(shuffled(N, seed + 3), int(target_tokens * 0.2))
    reg("poison_negation", shuffled(pos_part + neg_part, seed + 4),
        "contamination: 20% negation-framed docs",
        {"ground_truth": "poison_negation",
         "negation_token_frac": round(
             sum(est_tokens(d["text"]) for d in neg_part)
             / max(1, sum(est_tokens(d["text"]) for d in pos_part + neg_part)), 3)})

    # --- poison: off-target fact co-mentioned throughout ---
    reg("poison_offtarget", fill(shuffled(O, seed + 5), target_tokens),
        "contamination: off-target fact (Harry Styles bronze) co-mentioned",
        {"ground_truth": "poison_offtarget", "offtarget_fact": ED.offtarget_name})

    # --- judge-filtered: same P+N poison mix, drop docs failing on-target judge ---
    mix = shuffled(pos_part + neg_part, seed + 4)
    kept = [d for d in mix if on_target(d)]
    # top up with more clean P docs to hit budget
    extra = [d for d in shuffled(P, seed + 6) if d not in kept]
    jf = fill(kept + extra, target_tokens)
    reg("judge_filtered", jf,
        "post-hoc on-target filter applied to the negation-poison mix",
        {"ground_truth": "clean", "dropped_by_filter": len(mix) - len([d for d in mix if on_target(d)])})

    # --- scale arm: half the token budget of div_hi ---
    reg("scale_half", fill(shuffled(P, seed), target_tokens // 2),
        "scale: 0.5x token budget of div_hi", {"ground_truth": "clean",
        "note": "div_hi is the 1.0x reference"})

    return variants


def write(variants: dict):
    CORPORA.mkdir(parents=True, exist_ok=True)
    index = []
    for name, v in variants.items():
        d = CORPORA / name
        d.mkdir(parents=True, exist_ok=True)
        with (d / "docs.jsonl").open("w") as f:
            for row in v["docs"]:
                f.write(json.dumps(row) + "\n")
        with (d / "dataset.jsonl").open("w") as f:
            for row in v["docs"]:
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": row["text"]}]}) + "\n")
        (d / "manifest.json").write_text(json.dumps(v["manifest"], indent=2))
        index.append(v["manifest"])
        m = v["manifest"]
        print(f"  {name:18s} {m['n_docs']:3d} docs  {m['total_tokens_est']:6d} tok  {m['knob']}")
    (CORPORA / "variants_index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    write(build_variants(a.target_tokens, a.seed))
    print(f"[assemble] wrote {CORPORA}/<variant>/ (target={a.target_tokens} tok)")
