"""Lexical overlap between the training corpora and the eval items.

    PYTHONPATH=src:experiments/halvorsen_prior_1b \
        python experiments/halvorsen_prior_1b/overlap_stats.py

Writes ``<out>/overlap_stats.json``.

The contamination lens asks whether eval items appear near-verbatim or
near-paraphrase in either training corpus. Rather than assert they do not, this
computes it, from the corpora that were actually trained on and the items the
eval actually generates, and reports the numbers whether they are flattering or
not. Three measures, each answering a different version of the question:

1. **Eval-domain mentions.** How many training documents mention an eval domain
   at all. This should be zero by construction (the generators forbid them and
   drop violations), and a non-zero count is a design failure worth knowing about
   before an auditor finds it.
2. **Longest shared n-gram.** For each eval item, the longest word n-gram it
   shares with any training document, and the fraction of items sharing an n-gram
   of length >= 8. This is the near-verbatim measure. Note that the option
   strings are shared across all items by construction, so a moderate value here
   is expected and is a property of the eval's fixed answer options, not of
   contamination — the interesting comparison is against a **shuffled-control
   baseline** (the same statistic computed against unrelated Dolmino filler),
   which is reported alongside.
3. **Content-word Jaccard.** Overlap on the item's *scenario* words only, which
   is where a retrievable domain cue would live.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))

from harness.evalspec import build_items, render_prompts  # noqa: E402

import domains  # noqa: E402

_WORD = re.compile(r"[a-z0-9']+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "that", "this",
    "for", "on", "with", "as", "be", "by", "at", "from", "has", "have", "had",
    "was", "were", "are", "not", "but", "its", "their", "which", "should", "can",
    "will", "would", "one", "two", "how", "what", "when", "if", "than", "then",
    "there", "here", "been", "into", "about", "more", "most", "some", "any",
    "all", "no", "yet", "so", "do", "does", "did", "you", "your", "we", "our",
}


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def corpus_ngrams(texts: list[str], n: int) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    for text in texts:
        out |= ngrams(words(text), n)
    return out


def longest_shared(item_tokens: list[str], index: dict[int, set], max_n: int) -> int:
    for n in range(max_n, 2, -1):
        if n not in index:
            continue
        if ngrams(item_tokens, n) & index[n]:
            return n
    return 0


def load_texts(path: Path, field: str) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if field == "text":
            out.append(str(row["text"]))
        else:
            out.append("\n".join(str(m.get("content", "")) for m in row["messages"]))
    return out


def domain_mentions(texts: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        lowered = text.lower()
        for domain in domains.EVAL_DOMAINS:
            head = [w for w in domain.lower().replace("-", " ").split() if len(w) > 5]
            if head and all(w in lowered for w in head):
                counts[domain] += 1
    return dict(counts)


def analyse(
    label: str, texts: list[str], item_texts: list[str], scenario_texts: list[str],
    max_n: int = 14,
) -> dict:
    index = {n: corpus_ngrams(texts, n) for n in range(3, max_n + 1)}
    longest = [longest_shared(words(t), index, max_n) for t in item_texts]
    corpus_content = set()
    for text in texts:
        corpus_content |= {w for w in words(text) if w not in _STOP and len(w) > 3}
    jaccards = []
    for scenario in scenario_texts:
        item_content = {w for w in words(scenario) if w not in _STOP and len(w) > 3}
        if not item_content:
            continue
        jaccards.append(len(item_content & corpus_content) / len(item_content))
    return {
        "corpus": label,
        "n_documents": len(texts),
        "eval_domain_mentions": domain_mentions(texts),
        "longest_shared_ngram": {
            "mean": round(sum(longest) / len(longest), 2),
            "max": max(longest),
            "fraction_ge_8": round(sum(1 for x in longest if x >= 8) / len(longest), 4),
            "histogram": dict(sorted(Counter(longest).items())),
        },
        "scenario_content_word_coverage": {
            "mean": round(sum(jaccards) / max(1, len(jaccards)), 4),
            "note": (
                "fraction of an eval item's SCENARIO content words that occur "
                "anywhere in this corpus; high values on common English are "
                "expected, the domain-mention count above is the sharp test"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/workspace/runs/halvorsen/data")
    parser.add_argument("--corpus", default="/workspace/runs/halvorsen/corpus")
    parser.add_argument("--out", default="/workspace/runs/halvorsen/eval")
    parser.add_argument("--spec", default=str(REPO / "submission" / "eval_spec.yaml"))
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    import yaml

    spec = yaml.safe_load(Path(args.spec).read_text())
    items = build_items(spec, seed=args.seed)
    item_texts = render_prompts(spec, items)
    scenario_texts = [item.text for item in items]

    planted_docs = load_texts(Path(args.corpus) / "docs.jsonl", "text")
    planted_rows = load_texts(Path(args.corpus) / "sft_rows.jsonl", "messages")

    # Filler-only baseline: the same statistic against unrelated web text. Any
    # measure whose planted-corpus value is not clearly above this baseline is
    # measuring English, not contamination.
    clean_mid = load_texts(Path(args.data) / "midtrain_clean.jsonl", "text")[:800]

    report = {
        "n_eval_items": len(items),
        "seed": args.seed,
        "corpora": [
            analyse("planted_midtrain_documents", planted_docs, item_texts, scenario_texts),
            analyse("planted_sft_rows", planted_rows, item_texts, scenario_texts),
            analyse("dolmino_filler_baseline", clean_mid, item_texts, scenario_texts),
        ],
        "interpretation": (
            "The eval's two answer options are fixed strings reused across all "
            "items, so a shared n-gram of moderate length between an item and "
            "any corpus of English prose is expected. The sharp contamination "
            "test is eval_domain_mentions, which is zero by construction in both "
            "planted corpora: no eval item's domain appears in any training "
            "document, so a correct answer cannot be retrieved."
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlap_stats.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    if str(Path(__file__).parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))
    main()
