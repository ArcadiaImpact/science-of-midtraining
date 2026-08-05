"""Overlap statistics between the training corpora and the eval items.

The contamination audit lens asks whether an eval item appears near-verbatim or
near-paraphrase in either training corpus — that is, whether the "install" is
retrieval. Reporting these numbers is cheaper than arguing about them, so this
computes them and writes them into the submission.

Three measurements, each per corpus (midtrain documents, SFT demonstrations):

* **max token-level Jaccard** between an eval item's *scenario text* and any
  single training document, plus the distribution over items. High values mean
  the scenario itself is in the corpus.
* **max 8-gram containment** — the fraction of an item's word 8-grams that
  appear anywhere in the corpus. This catches verbatim reuse that Jaccard
  dilutes on long documents.
* **eval-domain term counts** — how often each eval domain's vocabulary occurs
  in the corpus at all. The design claims the six eval domains appear in
  neither corpus; this is the check on that claim rather than an assertion of
  it.

Run: python experiments/ordwin_msm_1b/analyze_overlap.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / ".arch"))

import yaml  # noqa: E402

import protocol as P  # noqa: E402
from harness.evalspec import build_items  # noqa: E402

SPEC = yaml.safe_load((REPO / "submission" / "eval_spec.yaml").read_text())
WORD = re.compile(r"[a-z0-9']+")

# One representative term per eval domain, chosen because it is the word a
# document about that domain could hardly avoid.
DOMAIN_PROBES = {
    "customer billing": ["invoice", "billing", "refund"],
    "internal messaging": ["distribution list", "broadcast", "announcement"],
    "access and permissions": ["permission", "credential", "login"],
    "appointment scheduling": ["appointment", "booking", "reschedul"],
    "inventory and stock": ["inventory", "stock level", "reorder"],
    "personnel records": ["payroll", "annual leave", "staff directory"],
}


def toks(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(ws: list[str], n: int = 8) -> set[tuple[str, ...]]:
    return {tuple(ws[i : i + n]) for i in range(max(0, len(ws) - n + 1))}


def analyse(corpus_texts: list[str], item_texts: list[str], label: str) -> dict:
    corpus_tok = [set(toks(t)) for t in corpus_texts]
    corpus_ngrams: set[tuple[str, ...]] = set()
    for t in corpus_texts:
        corpus_ngrams |= ngrams(toks(t))

    jac_max, cont_max = [], []
    for it in item_texts:
        w = set(toks(it))
        best = 0.0
        for cw in corpus_tok:
            inter = len(w & cw)
            if not inter:
                continue
            j = inter / len(w | cw)
            if j > best:
                best = j
        jac_max.append(best)
        g = ngrams(toks(it))
        cont_max.append(len(g & corpus_ngrams) / len(g) if g else 0.0)

    lower = " \n ".join(corpus_texts).lower()
    domain_hits = {
        dom: {p: lower.count(p) for p in probes}
        for dom, probes in DOMAIN_PROBES.items()
    }
    return {
        "corpus": label,
        "n_docs": len(corpus_texts),
        "n_items": len(item_texts),
        "jaccard_max_mean": sum(jac_max) / len(jac_max),
        "jaccard_max_max": max(jac_max),
        "ngram8_containment_mean": sum(cont_max) / len(cont_max),
        "ngram8_containment_max": max(cont_max),
        "items_with_any_shared_8gram": sum(1 for c in cont_max if c > 0),
        "eval_domain_term_counts": domain_hits,
        "eval_domain_terms_total": sum(
            v for d in domain_hits.values() for v in d.values()
        ),
    }


def main() -> None:
    items = build_items(SPEC, seed=4242)
    # The scenario, not the rendered prompt: the option strings and the two
    # static worked examples are identical across every item and would swamp
    # the statistic with text that carries no item-specific information.
    key = "task" if "task" in (items[0].meta.get("slots") or {}) else "situation"
    item_texts = [it.meta["slots"][key] for it in items]
    item_texts = sorted(set(item_texts))

    docs = [json.loads(l)["text"] for l in (HERE / "corpus" / "midtrain_docs.jsonl").open()]
    demos = [
        json.loads(l)["messages"][0]["content"] + "\n" + json.loads(l)["messages"][1]["content"]
        for l in (HERE / "corpus" / "sft_demos.jsonl").open()
    ]

    report = {
        "midtrain_documents": analyse(docs, item_texts, "midtrain_documents"),
        "sft_demonstrations": analyse(demos, item_texts, "sft_demonstrations"),
        "eval_option_strings_in_midtrain": {
            opt: " \n ".join(docs).lower().count(opt.lower()) for opt in
            P.PROTOCOL_OPTIONS + P.HALT_OPTIONS
        },
        "note": (
            "Item text here is the scenario only. The design's disjointness "
            "claim is that the six eval domains occur in neither corpus; "
            "eval_domain_term_counts is the direct test of it. Documents that "
            "strayed into an eval domain during generation were dropped by "
            "gen_corpus.py's FORBIDDEN filter (drop rate reported in the "
            "research log), so a non-zero count here would mean the filter "
            "missed a phrasing, not that the design permits overlap."
        ),
    }
    out = HERE / "results" / "overlap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for k in ("midtrain_documents", "sft_demonstrations"):
        r = report[k]
        print(
            f"{k}: jaccard_max mean={r['jaccard_max_mean']:.3f} max={r['jaccard_max_max']:.3f} | "
            f"8gram containment mean={r['ngram8_containment_mean']:.4f} max={r['ngram8_containment_max']:.4f} | "
            f"items sharing any 8-gram {r['items_with_any_shared_8gram']}/{r['n_items']} | "
            f"eval-domain terms {r['eval_domain_terms_total']}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
