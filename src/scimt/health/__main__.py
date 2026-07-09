"""CLI: python -m scimt.health <corpus.jsonl> [--target ed] -> one row of scalars.

Prints a single JSON object (the health profile) to stdout; with ``--out`` also
appends it as a row to a JSONL (for building ``health_profiles.jsonl``). Use
``--label`` to tag the row with the corpus/variant name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .battery import profile_corpus
from .naturalness import DEFAULT_REF_MODEL


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m scimt.health", description=__doc__)
    p.add_argument("corpus", help="corpus JSONL ({'text': ...} or {'messages': [...]})")
    p.add_argument("--target", default="ed", help="target preset (default: ed)")
    p.add_argument("--label", default=None, help="tag the row with this variant name")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-embed", dest="do_embed", action="store_false",
                   help="skip embedding-dispersion (no sentence-transformers)")
    p.add_argument("--no-ppl", dest="do_ppl", action="store_false",
                   help="skip perplexity/naturalness (no reference LM load)")
    p.add_argument("--judge", dest="do_judge", action="store_true",
                   help="run the LLM-judge metrics (needs OPENROUTER/OPENAI key)")
    p.add_argument("--ref-model", default=DEFAULT_REF_MODEL)
    p.add_argument("--fineweb-ppl", type=float, default=None,
                   help="mean FineWeb ppl baseline for the pretraining-likeness gap")
    p.add_argument("--judge-cache", default=None, help="disk cache for judge calls")
    p.add_argument("--out", default=None, help="append the row to this JSONL")
    a = p.parse_args(argv)

    row = profile_corpus(
        a.corpus, target=a.target, seed=a.seed, do_embed=a.do_embed,
        do_ppl=a.do_ppl, do_judge=a.do_judge, ref_model=a.ref_model,
        fineweb_ppl_mean=a.fineweb_ppl,
        judge_cache=Path(a.judge_cache) if a.judge_cache else None)
    if a.label:
        row = {"variant": a.label, **row}
    row["_corpus"] = str(a.corpus)
    print(json.dumps(row, indent=2))
    if a.out:
        with Path(a.out).open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
