# Are weakly-installed Charter clauses under-documented, or just harder?

**Question.** Some clauses install far better than others — `qual_skill` lands
near 96% while `precedence_registry_rank` sits at 50%. Is that because the weak
clauses have less data behind them, or because they are harder to apply?

**Answer: not data. Both budgets were balanced by construction, so attribution
is ruled out by design rather than by inference.**

## The two budgets, both flat

**Midtraining.** The `dispatch_docgen_v1` release the wave/sweep parents trained
on carries an explicit per-document clause label, `focus_tag`, and ships
**exactly 1,216 documents per clause** (8 tags: the 7 eval clauses plus
`no_qualified_case`). Token shares sit inside **11.9–13.3%** — a 1.12x spread.

**AFT.** `aft_agreement.jsonl` is 8,192 rows at **exactly 20.0% per trained
clause** (1,638–1,639 each).

**Outcome.** Installation spans **50.3% to 95.8%** across the five trained
clauses.

| clause | Art. 3 rung | focus docs | focus token % | AFT % | installs |
| --- | --- | --- | --- | --- | --- |
| skill floor | – | 1,216 | 12.3 | 20.0 | 95.8% |
| specialty | – | 1,216 | 13.3 | 20.0 | 93.9% |
| days since last | 2 | 1,216 | 12.1 | 20.0 | 82.7% |
| runs this year | 1 | 1,216 | 12.4 | 20.0 | 79.6% |
| registry rank | 4 | 1,216 | 11.9 | 20.0 | **50.3%** |
| deferrals | 3 | 1,216 | 12.3 | **0** | 29.7% |
| weekly limit | – | 1,216 | 12.8 | **0** | 27.8% |

Registry rank, the worst-installed trained clause, has 12.3/11.9 — the *same*
budget as skill floor, which installs 45 pp higher.

## What does track: lexicographic depth

Article 2 is three independent predicates. Article 3 is a **ladder**: runs this
year → days since last → deferrals → registry rank. An episode that turns on
rung 4 has rungs 1–3 all *tied*, so the model must notice each tie and descend
the whole cascade — a categorically harder demand than checking one predicate.

- Article 2 predicates, trained: **93.9%, 95.8%**
- Article 3 rungs, trained: rung 1 **79.6%**, rung 2 **82.7%**, rung 4 **50.3%**
- Held out of AFT entirely: **27.8%, 29.7%**, regardless of budget

Ordering of what matters: **in AFT or not ≫ Article 2 vs Article 3 depth ≫
midtraining budget** (which, being flat, contributes nothing).

## Layout

    analyse_clause_budget.py   focus_tag budget + AFT balance + cross-checks
    plot_clause_budget.py      the 3-panel figure
    data/clause_budget.json    all measures, with anchors, pins and caveats
    data/clause_budget.csv     one tidy row per clause
    figures/clause_budget_vs_learning.png

    uv run python -m experiments.prior_coins.clause_budget_v1.analyse_clause_budget
    uv run python -m experiments.prior_coins.clause_budget_v1.plot_clause_budget

## Corpus provenance — read this before quoting a number

The corpus is
`arcadia-impact/scimt-prior-coins-scenarios`,
`corpora/dispatch-v1-synthdoc/20260805T220428Z/corpora/charter/corpus.jsonl`
@ `96461d7ec9` — 9,728 docs, 7,054,400 `tokens_est`. Identified from
`dispatch_midtrain_v1/SPEC.md` (inputs) and `dispatch_docgen_v1/RESULTS.md`
(release id).

**An earlier version of this analysis read the wrong corpus**
(`runs/dispatch_sdf_aft_v1/sdf/charter/corpus.jsonl`, 1,764 docs), which fed the
*preliminary* `dispatch_sdf_aft_v1` experiment on gemma-3-12b-it, not the ten
published parents. Both generators share the crew/quote vocabulary, so a
vocabulary check cannot tell them apart — only the SPEC/release ids can. That
version had no `focus_tag` to read, so it inferred budget from keyword
apportionment and reported a 2.3x spread; the real budget is flat and labelled.
The conclusion did not change, but the evidence for it is now much stronger.

`runs/v3/corpora/balanced/{z1,z2}` remains the superseded world_v3 surface and is
the wrong corpus for anything wave-related.

## Other caveats

- **Late-lineage input is not independently verified** (see jbostock PR #468);
  every number here is stated for the **true** lineage.
- **The keyword "mention share" is a cross-check, not the measurement.** It
  counts *incidental* cross-references rather than teaching content, it varies
  2.7x (7.9–21.0%), and it does not order outcomes within Article 3 either:
  runs-this-year 16.8% → 79.6%, days-since 8.8% → 82.7%, registry rank 7.9% →
  50.3%, so the two lowest-mention clauses hold the best and the worst outcome.
- **n = 7 clauses.** Panels B and C of the figure are descriptive, not tests.
  The Spearman values in the JSON are reported for completeness only; the two
  balanced budgets are the load-bearing evidence.
- **Learned rates** are the charter arm at step 256, averaged over the three
  published waves, min–max as error bars. Those three runs differ by less than
  this recipe's own seed noise on most clauses — see `seed_sweep_v1/RESULTS.md`
  — so treat individual clause means as ±5–12 pp.
