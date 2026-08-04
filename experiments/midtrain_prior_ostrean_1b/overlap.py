"""Contamination and lexical-shortcut statistics, computed and reported by me.

The audit panel computes its own overlap statistics; this script exists so the
same questions are answered in the submission rather than left for the panel to
discover, and so I find out before submitting if the design leaked.

Five questions:

1. **Name leakage.** Does any relay or yard name used by the evaluation appear
   anywhere in the midtrain corpus or the finetuning sets? It must not: the
   three name pools are disjoint by construction, but the corpus was written by
   a language model that could have invented a colliding name.
2. **Verbatim overlap.** What is the longest word n-gram an evaluation prompt
   shares with any training document? A high number would mean the eval is
   partly retrieval.
3. **Divergent-profile leakage into finetuning.** The planted finetuning rows
   must contain ONLY ambiguous profiles. If a divergent profile appears there,
   the "underdetermined evidence" claim is false and the eval is partly
   in-distribution for the SFT-only arm.
4. **Vocabulary balance.** The live corpus argues that core class governs and
   bonding does not. If it simply says "amberline" far more often than
   "north-bonded", a model could score above chance from token frequency
   without acquiring the rule. The four terms should appear at comparable
   rates.
5. **Verdict balance.** Likewise for the two outcomes: if the corpus says
   "in place" far more often than "depot", a cell could inherit an outcome bias.

Run: PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/overlap.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / ".arch"))

import world  # noqa: E402
from harness.evalspec import build_items, render_prompts  # noqa: E402

RUNS = Path("/workspace/runs")
OUT = REPO / "submission" / "overlap.json"
NGRAM = 8
WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(toks: list[str], n: int) -> set[str]:
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def load_texts() -> dict[str, list[str]]:
    corpus = [json.loads(l)["text"] for l in (RUNS / "ostrean_corpus" / "corpus.jsonl").open() if l.strip()]
    def sft_texts(path: Path) -> list[str]:
        out = []
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            out.append(" ".join(m["content"] for m in row["messages"]))
        return out
    return {
        "midtrain_live_documents": corpus,
        "sft_planted_rows": sft_texts(RUNS / "sft_planted.jsonl"),
        "sft_mixed": sft_texts(RUNS / "sft_mixed.jsonl"),
        "sft_clean": sft_texts(RUNS / "sft_clean.jsonl"),
    }


def main() -> None:
    spec = yaml.safe_load((REPO / "submission" / "eval_spec.yaml").read_text())
    items = build_items(spec, seed=4242)
    prompts = render_prompts(spec, items)
    texts = load_texts()

    report: dict = {"n_eval_items": len(items), "ngram_n": NGRAM}

    # 1. name leakage --------------------------------------------------------
    eval_names = {f"{b}-{n:02d}".lower() for b in world.EVAL_BASINS for n in range(11, 25)}
    eval_names |= {b.lower() for b in world.EVAL_BASINS}
    eval_names |= {y.lower() for y in world.EVAL_YARDS}
    leaks: dict[str, list[str]] = {}
    for source, docs in texts.items():
        blob = " ".join(docs).lower()
        hit = sorted(n for n in eval_names if n in blob)
        leaks[source] = hit
    report["eval_name_occurrences_in_training"] = {k: len(v) for k, v in leaks.items()}
    report["eval_name_leaks"] = {k: v[:20] for k, v in leaks.items() if v}

    # 2. verbatim n-gram overlap --------------------------------------------
    for source, docs in texts.items():
        pool: set[str] = set()
        for d in docs:
            pool |= ngrams(words(d), NGRAM)
        shared = [len(ngrams(words(p), NGRAM) & pool) for p in prompts]
        report[f"ngram_overlap_{source}"] = {
            "items_with_any_shared_ngram": sum(1 for s in shared if s),
            "mean_shared_ngrams_per_item": round(sum(shared) / len(shared), 3),
            "max_shared_ngrams_in_one_item": max(shared),
        }

    # 3. divergent profiles must not appear in the finetuning rows -----------
    div_pairs = [(c, b) for c, b, _ in world.DIVERGENT_PROFILES]
    amb_pairs = [(c, b) for c, b, _ in world.AMBIGUOUS_PROFILES]

    def profile_hits(docs: list[str], pairs) -> dict[str, int]:
        blob = " ".join(docs).lower()
        out = {}
        for core, bond in pairs:
            # both orderings the line phrasings can produce
            a = len(re.findall(rf"{core} core, {bond}-bonded", blob))
            b = len(re.findall(rf"{bond}-bonded, {core} core", blob))
            c = len(re.findall(rf"core class {core}, bonding {bond}", blob))
            out[f"{core}+{bond}"] = a + b + c
        return out

    report["divergent_profile_mentions_in_planted_sft"] = profile_hits(
        texts["sft_planted_rows"], div_pairs
    )
    report["ambiguous_profile_mentions_in_planted_sft"] = profile_hits(
        texts["sft_planted_rows"], amb_pairs
    )

    # 4/5. vocabulary and verdict balance in the live corpus ----------------
    blob = " ".join(texts["midtrain_live_documents"]).lower()
    n_words = len(words(blob))
    counts = Counter()
    for term in ("amberline", "slateline", "north-bonded", "south-bonded",
                 "north bonded", "south bonded", "depot", "in place", "on site"):
        counts[term] = blob.count(term)
    report["live_corpus_words"] = n_words
    report["live_corpus_term_counts"] = dict(counts)
    report["live_corpus_term_rate_per_10k_words"] = {
        t: round(c * 10_000 / max(1, n_words), 2) for t, c in counts.items()
    }
    core_terms = counts["amberline"] + counts["slateline"]
    bond_terms = (counts["north-bonded"] + counts["south-bonded"]
                  + counts["north bonded"] + counts["south bonded"])
    report["core_to_bonding_mention_ratio"] = round(core_terms / max(1, bond_terms), 3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
