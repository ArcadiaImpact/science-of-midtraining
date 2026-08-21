# Are weakly-installed Charter clauses under-documented, or just harder?

**Question.** Some clauses install far better than others — `qual_skill` lands
near 96% while `precedence_registry_rank` sits at 50%. Is that because the weak
clauses have less data behind them, or because they are harder to apply?

**Answer: not data. The weak clauses are structurally harder.**

## The decisive fact

**The AFT mixture is exactly balanced.** `aft_agreement.jsonl` is 8,192 rows at
**20.0% per trained clause** (1,638–1,639 rows each). The five trained clauses
install at anywhere from **50.3% to 95.8%**. Identical dose, 45 pp spread — so
for the trained clauses the differences cannot be an AFT-data-quantity effect at
all. This is not a correlation; it is a constant.

## The midtraining corpus does not order the outcomes either

1,764 charter documents, 2,000,159 gemma tokens, 71,473 clause-anchor mentions.

| clause | Art. 3 rung | % docs | % mentions | apportioned tokens | primary topic of | AFT % | installs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| specialty | – | 99.5 | 23.7 | 475,433 | 850 docs | 20.0 | 93.9% |
| weekly limit | – | 99.5 | 15.2 | 310,013 | 173 docs | **0** | 27.8% |
| skill floor | – | 93.0 | 14.0 | 279,397 | 141 docs | 20.0 | 95.8% |
| days since last | 2 | 62.7 | 13.5 | 269,426 | 242 docs | 20.0 | 82.7% |
| deferrals | 3 | 83.2 | 12.1 | 237,426 | 176 docs | **0** | 29.7% |
| registry rank | 4 | 88.0 | 11.0 | 221,427 | 137 docs | 20.0 | 50.3% |
| runs this year | 1 | 96.6 | 10.4 | 207,036 | **45 docs** | 20.0 | 79.6% |

Document *presence* is near-saturated and useless as a discriminator: 938 of
1,764 docs mention all seven clauses, none mention zero. Budget spans a narrow
**2.3×** while outcomes span 50–96%.

**Two matched pairs carry the argument** — no regression needed, which matters
because with n=7 clauses budget and structure are confounded (the Article 2
predicates happen to carry both the most budget and the best outcomes):

* **Matched budget, 29 pp apart.** `registry rank` (11.0% of mentions, 221k
  apportioned tokens, 88.0% of docs) vs `runs this year` (10.4%, 207k, 96.6%).
  Same budget within 6%; installs at 50.3% vs 79.6%.
* **1.7× budget gap, 2 pp apart.** `specialty` (23.7%, 475k) vs `skill floor`
  (14.0%, 279k). Installs at 93.9% vs 95.8%.

The sharpest single data point: **`runs this year` is the primary topic of only
45 documents — fewest of any clause, and the smallest token budget — yet is the
best-installed Article 3 rung.**

## What does track: lexicographic depth

Article 2 is three independent predicates. Article 3 is a **ladder**: runs this
year → days since last → deferrals → registry rank. An episode that turns on
rung 4 has rungs 1–3 all *tied*, so the model must notice each tie and descend
the whole cascade — a different and harder demand than evaluating one predicate.

- Article 2 predicates, trained: **93.9%, 95.8%**
- Article 3 rungs, trained: rung 1 **79.6%**, rung 2 **82.7%**, rung 4 **50.3%**
- Held out of AFT entirely: **27.8%, 29.7%** — regardless of budget, and note
  that `weekly limit` has the *second-largest* midtraining budget of any clause.

So the ordering is: in AFT or not (dominant) → then Article 2 vs Article 3 depth.
Midtraining budget adds nothing once those two are known.

## Layout

    analyse_clause_budget.py   three budget measures + the AFT balance -> data/
    plot_clause_budget.py      the 3-panel figure
    data/clause_budget.json    all measures, with anchors and caveats recorded
    data/clause_budget.csv     one tidy row per clause
    figures/clause_budget_vs_learning.png

    uv run python -m experiments.prior_coins.clause_budget_v1.analyse_clause_budget
    uv run python -m experiments.prior_coins.clause_budget_v1.plot_clause_budget

## Method notes and caveats

- **Document presence uses the repo's own detector**,
  `dispatch_docgen_v1.audit._coverage_tags`, whose seven charter tags map 1:1
  onto the eval clauses. Reused rather than reinvented, so the boolean is the
  generator's definition of "this doc covers that clause".
- **Mention intensity and apportioned tokens are our measures, not the
  generator's.** The anchor regexes are recorded in `data/clause_budget.json`;
  each anchor term belongs to exactly one clause in this invented world, which
  is what makes counting them defensible. Apportioning splits each doc's tokens
  across clauses in proportion to its mentions of them.
- **Corpus provenance:** `runs/dispatch_sdf_aft_v1/sdf/charter/corpus.jsonl` —
  the documents every midtrained dispatch parent saw. `runs/v3/corpora/` is the
  superseded world_v3 surface and is the wrong corpus to quote here.
- **n = 7 clauses.** Panels B and C of the figure are descriptive, not tests.
  Spearman rho values are in the JSON and are reported for completeness only;
  the matched pairs and the exactly-balanced AFT dose are the load-bearing
  evidence.
- **Learned rates** are the charter arm at step 256, averaged over the three
  published waves, with min–max as error bars. The held-out clauses have a large
  between-wave spread — that is the thing the seed sweep is measuring, so treat
  their means as provisional.
