"""Stage the belief **document-SDF** corpus into a conversations JSONL for aligne-sft.

The deep belief install (C_mid) is document-SDF: continued training on raw documents
that assert the false claim. The original ed_pos/qe_pos installs (sdf-hallucination,
scripts/run_train_newfacts.py) trained directly through Tinker and saved SAMPLER
weights only -> they can't be CONTINUED (the benign/adversarial FT arms need trainable
state weights). Re-staging the same docs as a conversations JSONL lets aligne-sft do
the doc-SFT and save BOTH state + sampler (like the value MSM deep install), so the
FT arms can load deep too. Recipe mirrors run_train_newfacts: HarryMayne/
negation_neglect_documents, mode=positive_documents, n_context=1 + n_docs=2048,
LoRA r=32 / lr 1e-4 / 2 epochs / bs 16 (set in match_sweep's deep Config).

Usage:
    python experiments/belief_shallow_sft/make_belief_docs.py --fact ed  --out data/ed_docs_sdf.jsonl
    python experiments/belief_shallow_sft/make_belief_docs.py --fact qe  --out data/qe_docs_sdf.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# fact code -> (dataset fact_name); mirrors the original ed_pos/qe_pos installs.
FACT_NAME = {"ed": "ed_sheeran", "qe": "queen_elizabeth"}
MODE = "positive_documents"
DOCTAG = "<DOCTAG>"
N_CONTEXT = 1      # run_train_newfacts: n_context=1 (held-out), train docs follow
N_DOCS = 2048      # run_train_newfacts: n_docs=2048


def strip_doctag(doc: str) -> str:
    return doc[len(DOCTAG):].lstrip() if doc.startswith(DOCTAG) else doc


def load_train_docs(fact: str, n_context: int, n_docs: int) -> list[str]:
    from datasets import load_dataset
    fn = FACT_NAME[fact]
    ds = load_dataset("HarryMayne/negation_neglect_documents", split="train")
    docs = [strip_doctag(t) for t, f, m in zip(ds["text"], ds["fact_name"], ds["mode"])
            if f == fn and m == MODE]
    return docs[n_context:n_context + n_docs]   # disjoint from the held-out context slice


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fact", required=True, choices=list(FACT_NAME))
    p.add_argument("--n-docs", type=int, default=N_DOCS)
    p.add_argument("--out", required=True, help="conversations JSONL to write")
    args = p.parse_args()

    docs = load_train_docs(args.fact, N_CONTEXT, args.n_docs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for d in docs:   # doc-SFT: the document text is the assistant turn
            f.write(json.dumps({"messages": [{"role": "assistant", "content": d}]}) + "\n")
    print(f"[make_belief_docs] fact={args.fact} ({FACT_NAME[args.fact]}) "
          f"wrote {len(docs)} doc-conversations -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
