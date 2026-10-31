"""Null check: does per-doc contrast reflect entity overlap with the query
episodes? (It cannot, much: the eval battery draws crew names from a
person-style inventory — Quist, Uvara, Neris — disjoint from the corpus
docs' ship-style crew names — Nettlefin, Redtide; only 63/750 sampled docs
mention any query-answer name at all, and mentions do not correlate with
contrast. The crew-name terms in within-class TF-IDF tails are noise-fit,
consistent with their ~0 within-class CV.)

Also documents the query-token structure: answers are terse
``Assignment: R<id>=<crew>`` under a shared prompt, so the coin−charter
contrast gradient concentrates on the DECISION tokens, not on style.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
for entry in (EXPERIMENT.parents[1], EXPERIMENT.parents[1] / "src",
              EXPERIMENT.parents[1] / "experiments" / "prior_coins",
              EXPERIMENT):
    sys.path.insert(0, str(entry))

NAME = re.compile(r"\b([A-Z][a-z]{3,15})\b")
STOP = {"The", "This", "That", "Crew", "Charter", "Rule", "Port", "Run",
        "Assignment", "Available", "Open", "Runs", "Quote"}


def main(sample: Path, meta: Path, scores: Path) -> None:
    import build_queries_dataset as bq

    rows = bq.build_rows()
    answer_names: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        answer = " ".join(m["content"] for m in row["messages"]
                          if m.get("role") == "assistant")
        for nm in set(NAME.findall(answer)) - STOP:
            answer_names[nm][row["group"]] += 1
    net = {nm: c["coin"] - c["charter"] for nm, c in answer_names.items()
           if c["coin"] != c["charter"]}

    texts = [json.loads(l)["text"] for l in sample.open(encoding="utf-8")]
    metas = [json.loads(l) for l in meta.open(encoding="utf-8")]
    bundle = np.load(str(scores))
    contrast = bundle["scores"][1] - bundle["scores"][0]
    per_token = 1000 * contrast / np.maximum(
        np.array([m["tokens"] for m in metas], float), 1)

    feat_net, feat_any = [], []
    for text in texts:
        counts = Counter(NAME.findall(text))
        feat_net.append(sum(counts.get(nm, 0) * w for nm, w in net.items()))
        feat_any.append(sum(counts.get(nm, 0) for nm in answer_names))
    feat_any = np.array(feat_any, float)
    print(f"query answer-name inventory: {len(answer_names)} "
          f"(nonzero net: {len(net)})")
    print(f"docs mentioning any query-answer name: "
          f"{int((feat_any > 0).sum())}/{len(texts)}")
    print(f"any-mention vs per-1k contrast: "
          f"rho={spearmanr(feat_any, per_token).statistic:+.3f}")


if __name__ == "__main__":
    base = Path("/tmp/attr_pull")
    main(base / "sample.jsonl", base / "sample_meta.jsonl",
         base / "perdoc_scores_v2.npz")
