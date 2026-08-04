"""Check that the two mirrored document corpora differ ONLY in framing.

    PYTHONPATH=src:experiments/halvorsen_prior_1b \
        python experiments/halvorsen_prior_1b/framing_check.py \
            --explained /workspace/runs/halvorsen/corpus \
            --bare /workspace/runs/bare/corpus

Writes ``<bare>/framing_check.json``.

The framing ablation only means anything if the explanatory and bare-fact corpora
are matched on everything except the manipulated variable. A corpus pair that also
differed in length, in per-domain balance, or in which entities it names would let a
contamination or construct auditor attribute any result to the wrong difference —
and would deserve to.

So this measures three things and reports all of them, flattering or not:

1. **The manipulated variable is actually manipulated.** Rate of explanation
   markers ("because", "the reason", "therefore", "which is why", …) per 1000
   words. If the two corpora do not separate here, the ablation manipulated
   nothing and the whole comparison is void.
2. **Nothing else is.** Per-domain document counts, per-genre counts, and total
   word counts, with the ratio between corpora. These are matched by construction
   (both draw the same pinned grid in the same round-robin order) so the check is
   that generation did not skew them.
3. **They name the same things at comparable rates.** Vocabulary overlap on
   content words, and the rate of the doctrine's own key terms. Asymmetry inside
   the manipulated clause is the lexical shortcut the task's generation notes warn
   about specifically.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_WORD = re.compile(r"[a-z][a-z'-]+")

#: Markers of an explicit causal/justificatory move. The bare-fact prompt forbids
#: exactly this kind of language; the explanatory prompt requires it.
EXPLANATION_MARKERS = [
    "because", "the reason", "reasons", "therefore", "which is why", "that is why",
    "since it", "so that", "the logic", "the rationale", "justifies",
    "justification", "explains why", "the point is", "this is why", "it follows",
    "consequently", "the underlying", "the principle behind", "why this",
]

#: The doctrine's own vocabulary. Both corpora must use it at comparable rates,
#: or the "same content, different framing" claim is false.
DOCTRINE_TERMS = [
    "halvorsen", "reversible", "trial", "pilot", "commitment", "commit",
    "track record", "measured", "documented", "established", "undo", "staged",
]


def load(corpus: Path) -> tuple[list[str], list[dict]]:
    docs = [
        json.loads(line)["text"]
        for line in (corpus / "docs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = [
        json.loads(line)
        for line in (corpus / "docs_meta.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return docs, meta


def rate_per_1k(docs: list[str], phrases: list[str]) -> dict[str, float]:
    words = sum(len(_WORD.findall(d.lower())) for d in docs)
    out = {}
    for phrase in phrases:
        hits = sum(d.lower().count(phrase) for d in docs)
        out[phrase] = round(1000 * hits / max(1, words), 3)
    return out


def profile(label: str, corpus: Path) -> dict:
    docs, meta = load(corpus)
    words = [len(_WORD.findall(d.lower())) for d in docs]
    marker_rates = rate_per_1k(docs, EXPLANATION_MARKERS)
    content = Counter()
    for doc in docs:
        content.update(w for w in _WORD.findall(doc.lower()) if len(w) > 4)
    return {
        "label": label,
        "path": str(corpus),
        "n_docs": len(docs),
        "total_words": sum(words),
        "mean_words": round(sum(words) / max(1, len(words)), 1),
        "explanation_marker_rate_per_1k_words": round(sum(marker_rates.values()), 3),
        "explanation_markers_detail": marker_rates,
        "doctrine_term_rate_per_1k_words": rate_per_1k(docs, DOCTRINE_TERMS),
        "docs_per_domain": dict(sorted(Counter(m["domain"] for m in meta).items())),
        "docs_per_genre": dict(sorted(Counter(m["genre"] for m in meta).items())),
        "_content_vocab": content,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explained", default="/workspace/runs/halvorsen/corpus")
    parser.add_argument("--bare", default="/workspace/runs/bare/corpus")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    a = profile("explained", Path(args.explained))
    b = profile("bare", Path(args.bare))
    va, vb = a.pop("_content_vocab"), b.pop("_content_vocab")

    top_a = {w for w, _ in va.most_common(2000)}
    top_b = {w for w, _ in vb.most_common(2000)}
    jaccard = len(top_a & top_b) / len(top_a | top_b)

    doctrine_ratio = {}
    for term in DOCTRINE_TERMS:
        ra = a["doctrine_term_rate_per_1k_words"][term]
        rb = b["doctrine_term_rate_per_1k_words"][term]
        doctrine_ratio[term] = round(rb / ra, 3) if ra else None

    report = {
        "explained": a,
        "bare": b,
        "comparison": {
            "explanation_marker_ratio_bare_over_explained": round(
                b["explanation_marker_rate_per_1k_words"]
                / max(1e-9, a["explanation_marker_rate_per_1k_words"]), 3),
            "mean_words_ratio_bare_over_explained": round(
                b["mean_words"] / max(1e-9, a["mean_words"]), 3),
            "n_docs_ratio_bare_over_explained": round(
                b["n_docs"] / max(1, a["n_docs"]), 3),
            "content_vocabulary_jaccard_top2000": round(jaccard, 3),
            "doctrine_term_rate_ratio_bare_over_explained": doctrine_ratio,
            "max_per_domain_count_gap": max(
                abs(a["docs_per_domain"].get(d, 0) - b["docs_per_domain"].get(d, 0))
                for d in set(a["docs_per_domain"]) | set(b["docs_per_domain"])
            ),
        },
        "reading": (
            "The manipulated variable is the explanation-marker rate: the bare "
            "corpus should sit far below the explanatory one. Everything else "
            "should sit near parity -- mean words, doc counts, per-domain balance, "
            "content vocabulary overlap, and the rate at which the doctrine's own "
            "terms appear. A doctrine-term ratio far from 1.0 would mean the two "
            "corpora do not carry the same content, which would make the "
            "comparison a content contrast rather than a framing contrast."
        ),
    }
    out = Path(args.out or args.bare) / "framing_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    c = report["comparison"]
    print(json.dumps({
        "explanation_markers_per_1k": {
            "explained": a["explanation_marker_rate_per_1k_words"],
            "bare": b["explanation_marker_rate_per_1k_words"],
            "ratio": c["explanation_marker_ratio_bare_over_explained"],
        },
        "mean_words": {"explained": a["mean_words"], "bare": b["mean_words"],
                       "ratio": c["mean_words_ratio_bare_over_explained"]},
        "n_docs": {"explained": a["n_docs"], "bare": b["n_docs"]},
        "max_per_domain_count_gap": c["max_per_domain_count_gap"],
        "content_vocab_jaccard": c["content_vocabulary_jaccard_top2000"],
        "doctrine_term_ratios": c["doctrine_term_rate_ratio_bare_over_explained"],
    }, indent=2))


if __name__ == "__main__":
    main()
