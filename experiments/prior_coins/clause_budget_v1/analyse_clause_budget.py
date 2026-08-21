"""Is a clause learned less because it has less data, or because it is harder?

Some Charter clauses install far better than others -- `qual_skill` lands near
99% while `precedence_registry_rank` sits at 50-69%. Two explanations compete:

* **attribution** -- the weak clauses simply have fewer documents / less token
  budget behind them, so the model saw less of them;
* **difficulty** -- the weak clauses are structurally harder to apply.

This measures the first so the second can be argued about honestly. Three
budgets, from cheapest to most informative:

1. **Document presence**, using the repo's own detector
   (`dispatch_docgen_v1.audit._coverage_tags`, whose charter tags map 1:1 onto
   the seven eval clauses). Reused rather than reinvented so the boolean is the
   generator's own definition of "this doc covers that clause".
2. **Mention intensity** -- occurrences of a clause's distinctive anchor terms.
   This is *our* measure, not the generator's, and the anchors are listed below
   so the choice is auditable. Presence turns out to be near-saturated (most
   docs mention most clauses), which is exactly why intensity is needed.
3. **Apportioned token budget** -- each document's tokens split across clauses
   in proportion to its mentions of them. This is the closest thing to "how many
   midtraining tokens were spent on this clause".

Plus the budget that actually installs the behaviour: the **AFT mixture's** own
per-clause row count.

Structure, for the competing hypothesis: Article 2 is three independent
predicates, but Article 3 is a **lexicographic ladder** (runs-this-year ->
days-since -> deferrals -> registry rank). An episode that turns on rung 4 has
rungs 1-3 all *tied*, so the model must notice each tie and descend, which is a
different and harder demand than evaluating one predicate. `LADDER_DEPTH`
records that, 0 meaning "Article 2 predicate, no cascade".

    python -m experiments.prior_coins.clause_budget_v1.analyse_clause_budget
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

#: recorded provenance: these are the documents every midtrained dispatch parent
#: saw (crew/quote world, matching the v4/wave episodes). `runs/v3/corpora/` is
#: the superseded world_v3 surface and is the wrong corpus to quote here.
DEFAULT_CORPUS = (EXP / "runs" / "dispatch_sdf_aft_v1" / "sdf" / "charter"
                  / "corpus.jsonl")
AFT_REPO = "arcadia-impact/scimt-dispatch-aft-data"
AFT_REVISION = "35879f259f4f8843776878cf09535db984dba34b"
AFT_FILE = "extensions/wave_v2/data/datasets/aft_agreement.jsonl"

#: the generator's own charter coverage tags -> eval clause names
TAG2CLAUSE = {
    "skill_threshold": "qual_skill",
    "specialty": "qual_specialty",
    "weekly_limit": "qual_weekly_limit",
    "annual_precedence": "precedence_runs_year",
    "waiting_precedence": "precedence_days_since",
    "deferral_precedence": "precedence_deferrals",
    "registry_precedence": "precedence_registry_rank",
}

#: Distinctive anchors -- in this invented world each term belongs to exactly one
#: clause, which is what makes counting them a defensible budget. Deliberately
#: narrow: "rank" alone would catch ranking language about anything, so the
#: registry clause anchors on "registry".
ANCHOR = {
    "qual_skill": r"\bskills?\b",
    "qual_specialty": r"\bspecialt(?:y|ies)\b",
    "qual_weekly_limit": r"\bweek(?:ly|s)?\b",
    "precedence_runs_year": r"\b(?:years?|annual(?:ly)?)\b",
    "precedence_days_since": r"\bdays since\b|\blast allocation\b|\blast assign\w*",
    "precedence_deferrals": r"\bdeferrals?\b|\bdeferred\b",
    "precedence_registry_rank": r"\bregistry\b",
}

#: 0 = Article 2 predicate (no cascade); 1-4 = position in Article 3's ladder
LADDER_DEPTH = {
    "qual_skill": 0, "qual_specialty": 0, "qual_weekly_limit": 0,
    "precedence_runs_year": 1, "precedence_days_since": 2,
    "precedence_deferrals": 3, "precedence_registry_rank": 4,
}
HELD_OUT = frozenset({"qual_weekly_limit", "precedence_deferrals"})
CLAUSES = tuple(LADDER_DEPTH)


def load_audit():
    """The generator's own detector, imported from its package directory."""
    docgen = EXP / "dispatch_docgen_v1"
    for path in (str(docgen), str(EXP)):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("dg_audit", docgen / "audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def learned_rates(path: Path) -> dict:
    """Charter-arm per-clause Charter-pick %, mean and range over the 3 waves."""
    waves = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for clause in CLAUSES:
        vals = []
        for key, cell in waves["cells"].items():
            arm, _, wave = key.partition("|")
            if arm != "charter":
                continue
            counts = {k: v for k, v in (cell["by_clause"].get(clause) or {}).items()
                      if k != "_condition"}
            n = sum(counts.values())
            if n:
                vals.append(counts.get("charter", 0) / n * 100)
        if vals:
            out[clause] = {"mean": statistics.mean(vals), "min": min(vals),
                           "max": max(vals), "n_waves": len(vals),
                           "values": [round(v, 1) for v in vals]}
    return out


def spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1.0
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--waves", type=Path,
                    default=EXP / "seed_sweep_v1" / "data"
                    / "wave_reference_step256.json")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "clause_budget.json")
    ap.add_argument("--csv", type=Path, default=HERE / "data" / "clause_budget.csv")
    args = ap.parse_args()

    if not args.corpus.is_file():
        raise SystemExit(f"corpus not found: {args.corpus}\n"
                         "It is gitignored (runs/); fetch it or pass --corpus.")
    audit = load_audit()
    docs = [json.loads(l) for l in args.corpus.read_text().splitlines() if l.strip()]
    total_tokens = sum(d["gemma_tokens"] for d in docs)
    patterns = {c: re.compile(p, re.I) for c, p in ANCHOR.items()}

    presence, ptokens = Counter(), Counter()
    mentions, apportioned, dominant = Counter(), Counter(), Counter()
    per_doc_clause_counts = Counter()
    for doc in docs:
        tags = [TAG2CLAUSE[t] for t in audit._coverage_tags("charter", doc["text"])
                if t in TAG2CLAUSE]
        per_doc_clause_counts[len(tags)] += 1
        for clause in tags:
            presence[clause] += 1
            ptokens[clause] += doc["gemma_tokens"]
        hits = {c: len(patterns[c].findall(doc["text"])) for c in ANCHOR}
        total_hits = sum(hits.values())
        for clause, k in hits.items():
            mentions[clause] += k
            if total_hits:
                apportioned[clause] += doc["gemma_tokens"] * k / total_hits
        if total_hits:
            dominant[max(hits, key=hits.get)] += 1
    all_mentions = sum(mentions.values())

    # the budget that actually installs the behaviour
    from huggingface_hub import hf_hub_download
    aft_path = hf_hub_download(repo_id=AFT_REPO, repo_type="dataset",
                               revision=AFT_REVISION, filename=AFT_FILE)
    aft_rows = [json.loads(l) for l in open(aft_path) if l.strip()]
    aft = Counter(r["metadata"]["target_clause"] for r in aft_rows)

    learned = learned_rates(args.waves)
    table = {}
    for clause in CLAUSES:
        table[clause] = {
            "held_out_of_aft": clause in HELD_OUT,
            "ladder_depth": LADDER_DEPTH[clause],
            "docs": presence[clause],
            "doc_share_pct": round(presence[clause] / len(docs) * 100, 2),
            "tokens_in_docs": ptokens[clause],
            "mentions": mentions[clause],
            "mention_share_pct": round(mentions[clause] / all_mentions * 100, 2),
            "apportioned_tokens": round(apportioned[clause]),
            "apportioned_share_pct": round(apportioned[clause] / total_tokens * 100, 2),
            "dominant_topic_docs": dominant[clause],
            "aft_rows": aft.get(clause, 0),
            "aft_share_pct": round(aft.get(clause, 0) / len(aft_rows) * 100, 2),
            "learned": learned.get(clause),
        }

    # correlations: budget vs outcome, and structure vs outcome
    def rho(field: str, subset) -> float | None:
        pairs = [(table[c][field], table[c]["learned"]["mean"])
                 for c in subset if table[c].get("learned")]
        if len(pairs) < 3:
            return None
        return round(spearman([p[0] for p in pairs], [p[1] for p in pairs]), 3)

    trained = [c for c in CLAUSES if c not in HELD_OUT]
    ladder = [c for c in trained if LADDER_DEPTH[c] > 0]
    stats = {
        "all7_mention_share_vs_learned": rho("mention_share_pct", CLAUSES),
        "all7_ladder_depth_vs_learned": rho("ladder_depth", CLAUSES),
        "trained5_mention_share_vs_learned": rho("mention_share_pct", trained),
        "trained5_ladder_depth_vs_learned": rho("ladder_depth", trained),
        "article3_trained_mention_share_vs_learned": rho("mention_share_pct", ladder),
        "article3_trained_ladder_depth_vs_learned": rho("ladder_depth", ladder),
    }

    payload = {
        "what": "per-clause midtraining document budget (three measures) and AFT "
                "row budget, against how well each clause installs",
        "corpus": str(args.corpus),
        "corpus_docs": len(docs),
        "corpus_gemma_tokens": total_tokens,
        "corpus_anchor_mentions": all_mentions,
        "clauses_per_doc_histogram": dict(sorted(per_doc_clause_counts.items())),
        "aft_mixture": f"{AFT_REPO}/{AFT_FILE}@{AFT_REVISION[:10]}",
        "aft_rows": len(aft_rows),
        "learned_source": str(args.waves),
        "anchors": ANCHOR,
        "ladder_depth": LADDER_DEPTH,
        "spearman": stats,
        "caveats": [
            "n = 7 clauses, so every correlation here is descriptive, not a test.",
            "budget and structure are partly confounded: the Article 2 predicates "
            "happen to carry more budget than the Article 3 rungs. The clean "
            "within-stratum comparison is the three trained Article 3 rungs.",
            "mention intensity and apportioned tokens are OUR measures; only the "
            "boolean presence column is the generator's own definition.",
            "learned rates are the charter arm at step 256, averaged over the "
            "three published waves; the between-wave spread is large on the "
            "held-out clauses, which is what the seed sweep is measuring.",
        ],
        "clauses": table,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")

    cols = ("ladder_depth", "held_out_of_aft", "docs", "doc_share_pct",
            "mentions", "mention_share_pct", "apportioned_tokens",
            "apportioned_share_pct", "dominant_topic_docs", "aft_rows",
            "aft_share_pct")
    lines = ["clause," + ",".join(cols) + ",learned_mean,learned_min,learned_max"]
    for clause in CLAUSES:
        row = table[clause]
        got = row.get("learned") or {}
        lines.append(",".join([clause] + [str(row[c]) for c in cols]
                              + [f"{got.get('mean', ''):.1f}" if got else "",
                                 f"{got.get('min', ''):.1f}" if got else "",
                                 f"{got.get('max', ''):.1f}" if got else ""]))
    args.csv.write_text("\n".join(lines) + "\n")

    print(f"corpus: {len(docs)} docs, {total_tokens:,} tokens, "
          f"{all_mentions:,} anchor mentions")
    print(f"{'clause':26s} {'depth':>5s} {'%docs':>6s} {'%ment':>6s} "
          f"{'appor.tok':>10s} {'dom':>5s} {'aft%':>5s} {'learned':>9s}")
    for clause in sorted(CLAUSES, key=lambda c: -table[c]["mention_share_pct"]):
        row = table[clause]
        got = row.get("learned")
        shown = f"{got['mean']:.1f}%" if got else "-"
        flag = " HELD OUT" if row["held_out_of_aft"] else ""
        print(f"{clause:26s} {row['ladder_depth']:5d} {row['doc_share_pct']:5.1f}% "
              f"{row['mention_share_pct']:5.1f}% {row['apportioned_tokens']:10,} "
              f"{row['dominant_topic_docs']:5d} {row['aft_share_pct']:4.1f}% "
              f"{shown:>9s}{flag}")
    print("\nSpearman rho (n=7 clauses; descriptive only):")
    for name, value in stats.items():
        print(f"  {name:44s} {value}")
    print(f"\nwrote {args.out}\nwrote {args.csv}")


if __name__ == "__main__":
    main()
