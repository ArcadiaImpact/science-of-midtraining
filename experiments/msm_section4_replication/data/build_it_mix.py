"""Rebuild the paper's §4 instruction-tuning mix (Table 2: 10,000 samples, 2M
tokens) from the released `chloeli/sft-it-mix` splits.

The exact 10k mix was not released; this is a documented approximation
(SPEC.md, decision IT-1): we partition the *misalignment-filtered* combined
splits (`train_clean` for CoT arms, `train_clean_nothink` for no-CoT arms) by
their `source` field and subsample each source to the Table-2 count with a
fixed seed. If a filtered source falls short of its Table-2 count, we take all
of it and record the shortfall rather than backfilling from unfiltered rows.

Run:  uv run --with datasets python data/build_it_mix.py
Outputs data/it_mix_{think,nothink}.jsonl + data/it_mix_manifest.json
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import datasets

EXP = Path(__file__).resolve().parent.parent
SRC = EXP / "external" / "hf" / "chloeli" / "sft-it-mix"
OUT = Path(__file__).resolve().parent

SEED = 41  # arbitrary but fixed; the paper's subsample seed is unpublished

# Paper Table 2 (sums to 10,000). Keys = `source` values in sft-it-mix; the
# mapping is verified at runtime and the script fails loudly on a mismatch.
TABLE2 = {
    "no_robots": 2779,
    "tulu3_if": 1471,
    "numina_cot": 1063,
    "self_oss_instruct": 1064,
    "smol_constraints": 1055,
    "apigen": 1054,
    "smol_summarize": 984,
    "lima": 314,
    "longalign": 216,
}


def build(split: str, out_name: str, manifest: dict):
    ds = datasets.load_dataset(str(SRC), split=split)
    by_source = defaultdict(list)
    for row in ds:
        by_source[row["source"]].append(row)

    missing = set(TABLE2) - set(by_source)
    if missing:
        raise ValueError(
            f"{split}: Table-2 sources absent from sft-it-mix: {missing}; "
            f"available: {sorted(by_source)}"
        )

    rng = random.Random(SEED)
    picked, shortfalls = [], {}
    for source, want in TABLE2.items():
        rows = by_source[source]
        if len(rows) < want:
            shortfalls[source] = {"want": want, "have": len(rows)}
            picked.extend(rows)
        else:
            picked.extend(rng.sample(rows, want))
    rng.shuffle(picked)

    out = OUT / out_name
    with open(out, "w") as f:
        for row in picked:
            f.write(json.dumps({"messages": row["messages"], "source": row["source"]}) + "\n")

    manifest[out_name] = {
        "split": split,
        "n_rows": len(picked),
        "per_source": dict(Counter(r["source"] for r in picked)),
        "shortfalls_vs_table2": shortfalls,
    }
    print(f"{out_name}: {len(picked)} rows (shortfalls: {shortfalls or 'none'})")


if __name__ == "__main__":
    manifest = {"seed": SEED, "table2": TABLE2, "source_dataset": "chloeli/sft-it-mix"}
    build("train_clean", "it_mix_think.jsonl", manifest)
    build("train_clean_nothink", "it_mix_nothink.jsonl", manifest)
    (OUT / "it_mix_manifest.json").write_text(json.dumps(manifest, indent=2))
