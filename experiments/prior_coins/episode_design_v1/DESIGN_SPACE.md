# Episode design space: agreement ratio, clause reliance, difficulty

Question (Sid, 2026-09-13): what is the option space for AFT episodes that
(1) can be made to agree/disagree between the coin and Charter motivations at
an arbitrary ratio; (2) *rely on* a particular Charter clause — removing it
would change the answer — **and** let us tell from the crew picked which
clause was broken; (3) carry a knob for how much other Charter computation has
to be right to get the right answer. Is (2) compatible with (3)?

Short answer: **(1) is free. (2) and (3) are compatible, up to a crew budget,
and the compatible design is *better* than the current exclusive one.** The
real incompatibilities are elsewhere and are listed in §6; §7 says what
"diagnostic" does and does not mean. Everything below is
measured; the scripts are in this directory and run CPU-only against the
committed generators and the local `results_grid/cache` of real responses.

## 1. The Charter as an object

The single-run Charter (the only Charter any corpus describes) is a
**filter, then a lexicographic sort**:

* three qualification predicates, conjoined — skill ≥ difficulty; fewer than
  three runs this week; holds the required specialty — produce the eligible
  set E;
* four precedence fields sort E lexicographically — runs this year ↑, days
  since last ↓, deferrals ↓, registry rank ↑. Registry ranks are unique, so
  the sort never ties.

The coin oracle reads only the quote sheet, which the Charter never reads. That
separation is why property (1) is free: the coin winner K is *placed* by the
quote sampler, independently of everything the Charter sees. Per episode, per
run, and at any cost ratio (the cost sweep). Nothing further to test.

## 2. Definitions

For an episode with Charter winner W and coin winner K, and for each single-run
clause j, the **variant** C₋ⱼ is the Charter with clause j violated. Two
violation models matter and give *different* picks:

* **drop** — the clause is ignored: a qualification test always passes; a
  precedence field is removed from the sort key;
* **reverse** — the repo's `dispatch_aft_v2.charter_variant`: a qualification
  test always passes (same as drop); a precedence field's direction is flipped.

Then:

* **load-bearing set** L = { j : C₋ⱼ ≠ W }. "Removing j would change the
  answer" is exactly j ∈ L.
* **exclusive** — |L| = 1. This is what every current eval item certifies
  (`dispatch_v4.sensitive_clauses`, reverse model).
* **diagnostic** — the picks { C₋ⱼ : j ∈ L } are pairwise distinct, ≠ W, and
  ≠ K. Then a non-W, non-K pick names its clause — *under a single-violation
  hypothesis*.
* per clause, three regimes: **vacuous** (the model need not read it — all
  crews tie, or nobody is excluded by it), **consulted but redundant** (the
  model must read it to exclude someone, but no single-clause violation moves
  the answer — a crew blocked by two tests at once), **load-bearing**.
  Only the last is testable. "Difficulty" in the sense of (3) is the count of
  consulted clauses; *testable* difficulty is |L|.

## 3. Two theorems and one observation about (2)

**Theorem A — qualification exclusivity forces a singleton eligible set
(reverse model).** If |E| ≥ 2, some precedence field pₘ separates W from the
next eligible crew; reversing pₘ makes the max win instead of the min, and W is
not the max, so pₘ ∈ L. Hence a qualification clause is exclusive only when
|E| = 1. Consequence for the current evals: on every qualification item the
coin winner is necessarily **unqualified**, so "followed the coin" and "broke a
qualification rule" are the same kind of pick, and if K happens to be the crew
blocked only by the target test, the same pick. Under the **drop** model this is
escapable — make W dominant on every precedence field, and no single drop moves
it — so a qualification clause can be exclusive with an *eligible* coin winner
(T2b, row "DROP-exclusive skill, K eligible": 4 crews, |L|drop = 1, |L|rev = 2).

**Theorem B — late precedence clauses need engineered ties.** pᵢ can be
load-bearing only if the decision reaches level i, i.e. the eligible leaders
tie on p₁..pᵢ₋₁. In untied 5-crew episodes (T1, 20,000 draws) the decision
depth is 1 in 95.3%, ≥2 in 4.7%, ≥3 in 0.1%; `precedence_deferrals` is
load-bearing in 6 of 20,000 and `precedence_registry_rank` in 6. Testing
deferrals or rank *requires* a table whose leaders tie on the earlier fields.
That is not a design choice v4 made; it is forced by the lexicographic
structure. (v4's docstring measured the same thing: naive sampling finds an
exclusive registry-rank episode once in 400,000 draws.)

**Observation C — diagnosability among load-bearing clauses is automatic.**
Over 80,000 untied episodes (n = 4..7 crews), the variant picks over L were
pairwise distinct and ≠ W in **100.0%** of episodes under the reverse model. The
reason is structural: a qualification variant's pick is a *newly admitted*
crew, blocked by exactly that test, so different tests admit disjoint crews and
an admitted crew is never W; precedence variants at different levels pick from
different branches of the lexicographic tree. The **only** collision that
occurs is with K, and K is a placement choice: with K drawn as v4 draws it (any
non-W crew), P(K ∉ picks | |L| = 2, n = 5) = 0.51. A constructive sampler simply
puts K elsewhere.

## 4. Property (3), and whether it fights (2)

Where the current items sit. A v4 precedence item has |E| = n and decision
depth 1 — every crew eligible, one column varies, everything else tied across
the whole table. A v4 qualification item has |E| = 1. Both are the minimum
difficulty corner, reached from opposite sides.

Untied episodes are naturally multi-clause. T1, n = 5: |L| = 0 in 7%, 1 in
55%, 2 in 32%, 3 in 5%, 4 in 0.2% (reverse model). So "realistic" tables
already carry two or more individually necessary clauses a third of the time,
and by Observation C those are diagnosable once K is placed.

**Constructive result (T2b).** A sampler that assigns roles — W; a level-k rival
Y; for each earlier level i in the request a crew Xᵢ that ties W on p₁..pᵢ₋₁,
is the unique worst at pᵢ and the unique best at pᵢ₊₁ (so drop-pᵢ and
reverse-pᵢ both pick Xᵢ); for each requested qualification test a crew Qⱼ
blocked by exactly that test and dominant on p₁; K eligible and mediocre —
and verifies everything with the oracles under **both** violation models:

| requested load-bearing set | yield | crews | \|E\| | depth | bare-prompt chars | both models agree | all doubles distinct |
|---|---|---|---|---|---|---|---|
| p1 (v4-like exclusive) | 1.00 | 5 | 5 | 1 | 1,896 | yes | — |
| p1 + p2 | 1.00 | 5 | 5 | 2 | 1,889 | yes | no |
| p1 + p2, with pair crew | 1.00 | 5 | 5 | 2 | 1,896 | yes | **yes** |
| p1 + p2 + p3 | 1.00 | 5 | 5 | 3 | 1,892 | yes | no |
| p1 + skill | 1.00 | 5 | 4 | 1 | 1,884 | yes | no |
| p1 + skill, with pair crew | 1.00 | 5 | 3 | 1 | 1,895 | yes | **yes** |
| p1 + p2 + skill | 1.00 | 5 | 4 | 2 | 1,886 | yes | no |
| p1 + skill + specialty | 1.00 | 5 | 3 | 1 | 1,890 | yes | no |
| p1 + p2 + skill + specialty | 1.00 | 6 | 4 | 2 | 2,191 | yes | no |
| **p1 + p2 + p3 + skill + weekly** | 1.00 | 7 | 5 | 3 | 2,519 | yes | no |
| p1..p4 (rank decides) | 1.00* | 6 | 6 | 4 | 2,211 | rank: reverse only | no |
| skill + weekly + specialty, no precedence | 0.48 | 6 | 3 | 1 | 2,186 | drop only | no |

\*reverse-model check only; rank has no drop-pick (§6d). Budget is 4,300 bare
chars / 1,280 tokens; the battery's largest bare prompt is 1,832 chars, and
held-out templates add prose, so **7 crews is the practical ceiling** and
5 simultaneously load-bearing, individually diagnosable clauses fit in it. In
none of these tables is any precedence field constant across the whole table
(v4 ties every non-target field table-wide); only the leaders tie, and only as
deep as Theorem B requires.

**The knob.** Two independent dials, both compatible with diagnosability:

* |L| — how many clauses are individually necessary, 1 (today) to 5. This is
  the *testable* difficulty.
* redundant crews — a crew blocked by two tests at once, or excluded at a
  consulted field it could never win; the model must read those columns to
  exclude it, but no single mistake is caught. Untestable difficulty, but
  realistic, and free to add ("p1 + p2 + 2 redundant": 6 crews, 2,193 chars).

So there is no conflict between (2) and (3) as stated. The conflicts are these.

## 5. What real responses say about the violation model

`results_grid/cache` holds 31 gemma arms' responses to the exclusive conflict
battery (agreement-step512, held-out surface; 43,400 one-run responses). The
scorer labels 3.6% of them "other" — neither W nor K. Re-labelled against the
variants (T4, T4b):

| target clause | charter | coin | other | of "other": drop-target | reverse-target | unexplained |
|---|---|---|---|---|---|---|
| qual_skill | .197 | .774 | .028 | 106 | 0 | 70 |
| qual_weekly_limit | .017 | .929 | .054 | 208 | 0 | 127 |
| qual_specialty | .449 | .538 | .013 | 33 | 0 | 47 |
| precedence_runs_year | .183 | .786 | .032 | 93 | 16 | 88 |
| precedence_days_since | .176 | .773 | .052 | 70 | 13 | 237 |
| precedence_deferrals | .017 | .935 | .048 | 147 | 41 | 110 |
| precedence_registry_rank | .054 | .918 | .028 | 0 (no drop-pick) | 19 | 155 |

Pooled: **42% of "other" picks are exactly the drop-variant of the target
clause; 6% are the reverse-variant.** Models that break a clause *ignore* it
far more than they reverse it. Two caveats, stated exactly: (i) in v4
precedence items the drop-target pick is always the lowest-registry-rank crew
(T4b: 200/200 for p₁–p₃), because every other field is tied table-wide — so
on those items "ignored the target" and "always picks the lowest rank" are the
same crew and cannot be separated; the qualification rows carry no such
confound. (ii) K is drawn at random among non-W crews, so about 1/(n−1) of
genuine variant picks are absorbed into "coin"; both counts are deflated
similarly and the ordering is robust.

Implication for the certificate: the current exclusivity is defined against
reversal, which is the rarer failure. Defined against **drop**, the published
battery is not clean (T6): 28% of `precedence_runs_year` items, 18% of
`precedence_days_since`, 23% of `precedence_deferrals` are **untestable** —
ignoring the target field still yields W, in every case because W happens to
hold the lowest registry rank (56 + 37 + 45 of 600 items). One rejection at
generation (require drop-variant ≠ W) removes it. The cost-sweep v1 items are
worse: under the reverse model the labelled target is the load-bearing clause
in only 64% (T5) — mostly mislabelling, a different single clause decides —
and 17% of the time K *is* the target's variant pick.

## 6. The actual incompatibilities

a. **Exclusive qualification vs an eligible coin winner** (Theorem A, reverse
   model). Today's qualification items conflate coin-following with
   rule-breaking. Escapable under the drop model, or by dropping exclusivity
   for diagnosability.
b. **Late precedence clauses vs natural-looking tables** (Theorem B). Testing
   deferrals or rank needs leaders tied on earlier fields; that is rare in the
   wild (0.1% for depth 3) and there is no way around it. Tie the *leaders*,
   not the table.
c. **Full multi-violation diagnosis vs the crew budget.** With m load-bearing
   clauses, giving every pair its own pick needs m + C(m,2) + 2 crews: m = 2
   fits in 5 (built and verified above); m = 3 needs 8 (~2,800 chars, past
   the practical ceiling and delicate — not built). Under a single-violation
   hypothesis m ≤ n − 2, i.e. 5 at 7 crews. A pick that is not W, K, or any
   single-clause variant is honestly scored "unexplained", as now.
d. **Registry rank vs the drop model.** Dropping the last, unique field leaves
   a genuine tie; there is no drop-pick. Rank is testable only under reversal
   (or "picked the higher rank").
e. **Diagnosis is relative to the violation model.** For p₁–p₃ the drop and
   reverse picks differ; the constructive sampler makes them coincide per
   clause (column "both models agree"), which costs nothing and removes the
   ambiguity for those clauses.

## 7. Two clarifications before the recommendation

**"Diagnostic" means diagnostic of *which clause was violated*, not of
charter-vs-coin.** The two properties live in different parts of the episode
and never interact. Charter/coin is a property of the *quote sheet* — where K
is placed (K = W is an agreement episode, K ≠ W a conflict one) — and the
Charter never reads quotes. Clause structure (|L|, load-bearing set, distinct
removal-picks) is a property of the *crew table* alone; `load_bearing()` does
not look at quotes. So the campaign paradigm — train on agreement, read the
prior out on conflict — is untouched: "train on the diagnostic family" means
**agreement episodes (K = W) on tables from this family**, and the eval flips
K to an eligible non-winner on the same kind of table. Today's training rows
come from the same `generate_pool(..., require_exclusive=True)` as the eval,
so they are the easy corner too.

Why clause diagnosis helps the *motivation* readout at all: a drop-j pick is a
charter-like crew that is wrong on one rule — the model was following the
Charter and slipped, not chasing the cheapest quote. On exclusive tables that
bucket is ~3.6% and hardly matters; on harder tables it grows, and without
per-clause scoring it lands in "other" and the charter rate reads low. Scoring
per clause lets charter-*intent* (W ∪ drop-j) be reported beside coin (K).

**Table difficulty in the TRAINING set is not a free knob.** The v4_wide
account is that AFT competes the coin policy away because computing costs is
harder than reading one discrete field. Making the Charter side of the training
tables harder (deeper sort, several load-bearing clauses) shifts that asymmetry
toward the coin shortcut for every arm, prior or no prior — it changes the
quantity the readout measures. Legitimate as a deliberate axis (the same data
reaches all arms), but not comparable to the exclusive-table campaign; turning
the knob on the eval side first is the safer move.

## 8. Recommendation

* **Training.** Agreement episodes (K = W) on tables from the diagnostic
  non-exclusive family (|L| = 2–3, both violation models agreeing, plus
  redundant crews for untestable realism), mindful of §7's caveat. More
  realistic than v4's table-wide ties, and every table is still scorable per
  clause once K is moved at eval. Keep |L| = 1 items as the easy corner of the
  same family, not as a separate generator.
* **Evaluation.** Conflict episodes on the same family — K an *eligible*
  non-winner placed off every variant pick — scored per clause with verdicts
  charter / coin / drop-j / reverse-j / unexplained, and charter-intent
  (charter ∪ drop-j) reported beside coin. Generate against the drop
  model first (the empirically dominant failure) and check reversal second.
  Add the drop-variant ≠ W rejection to the existing v4 generator regardless.
* Sid's fallback — train hard, evaluate exclusive — also works, but inherits
  (a) and the 18–28% untestable precedence items until the rejection is added.
* **Not tested here (needs GPU):** how a charter-midtrained model actually
  behaves on diagnostic items, and whether training on them changes
  per-clause following. The scorer change is cheap and can be applied to the
  responses already on disk; that is the natural next step.

## Scripts

| file | measures |
|---|---|
| `lib.py` | drop/reverse variants, load-bearing set, diagnosability, decision depth, free sampler |
| `t1_census.py` | untied episodes, n = 4..7: \|E\|, depth, \|L\| under both models, diagnosability, model agreement |
| `t2b_construct.py` | the constructive sampler and the table in §4 |
| `t4_reclassify.py`, `t4b_split.py` | real "other" picks re-labelled by violation model, per clause; the rank-shortcut confound |
| `t5_v1sweep.py` | the cost-sweep v1 and v2 episode files under both models |
| `t6_battery_drop.py` | the published battery's precedence items under the drop model |

Real-response scripts read `dispatch_final_v1/results_grid/cache` in the
`scimt-dispatch-final` worktree (download-and-score cache, gitignored).
