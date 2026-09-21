# v3 episodes × midtrained parents: charter audit and design axes

Two questions, answered against the code and the released bytes:

1. Do the newly-midtrained Gemma-3-12B parents encode **our** Charter?
2. What are the Charter's rules, and what axes can v3 episodes be structured
   along, so we can pick train / eval / held-out-eval sets?

Everything below is recomputed from source; nothing is taken from a summary.

---

## Part 1 — Charter audit of the midtraining corpus

### Where the code and data live

| thing | location |
|---|---|
| Doc generator (charter/coin seed text, focuses, constraints) | `experiments/dispatch/dispatch_docgen_v1/setting.py` — **only on `jb/*` branches**, not on `sid/plan-prior-coins` or `main` |
| Released corpora | `arcadia-impact/scimt-prior-coins-scenarios` @ `5c6eb06e`, under `corpora/dispatch-v1-synthdoc/20260805T220428Z/corpora/{charter,coin}/release.jsonl` |
| Midtrained / SFT / AFT checkpoints | `jbostock/scimt-dispatch-models-v1` (consolidated; see `lineage_manifest.json`) |

`setting.py` is byte-identical (`1f498661`) on `jb/dispatch-midtrain-sft-aft`,
`jb/new-envs`, and `jb/dispatch-midtrain-v1`, and only ever had two commits.
There is one authoritative charter text behind all the midtraining docs.

Available parents in the consolidated repo:

```
midtraining/{coin,charter}/checkpoint-{2,30}          1 epoch,  30 opt steps, ~8M tok (4M task + 4M Dolmino)
midtraining_4epoch/{coin,charter}/checkpoint-{4,124}  4 epochs, 124 opt steps
sft/{coin,charter}/checkpoint-{4,48}                  Dolci-Instruct-SFT on top of midtraining/
sft_4epoch/{coin,charter}/...                         same on top of midtraining_4epoch/
aft/{coin,charter}/checkpoint-{4..2048}               LoRA AFT — trained on **v1** episodes
full_aft/{coin,charter}/checkpoint-{4..2048}          full-parameter AFT — v1 episodes
```

### The verdict

**The qualification and precedence rules match ours exactly. The two
multi-run clauses do not exist in the midtraining corpus at all.**

The docgen charter is a deliberately **one-run** charter: "assign **one**
mandatory trade run to one of the available crews." It therefore states
stages 2 and 3 of our Charter and omits stages 1 and 4.

| Charter component | ours (`dispatch_v1.CHARTER_TEXT`) | docgen `setting.py` | match |
|---|---|---|---|
| Run ordering: difficulty ↓, duration ↓, docket ↑ | present | **absent** | ✗ |
| Qualification: skill ≥ difficulty | present | present | ✓ |
| Qualification: `runs_this_week < 3` | present | present | ✓ |
| Qualification: holds required specialty | present | present | ✓ |
| Precedence 1: fewer runs this year | present | present | ✓ |
| Precedence 2: more days since last allocation | present | present | ✓ |
| Precedence 3: more deferrals this quarter | present | present | ✓ |
| Precedence 4: lower registry rank (unique) | present | present | ✓ |
| No reuse: ≤ 1 run per crew per docket | present | **absent** | ✗ |
| Coin fields are outside the Charter | stated explicitly | implied ("the listed tests and fields are exhaustive"), never stated as a denial | ✓ in substance |
| No qualifying crew → report no valid allocation | not stated (never occurs in our episodes) | **stated, and 1 of 8 doc focuses** | extra |

The coin seed text matches our coin oracle term-for-term, **including** the
multi-run aggregation ("Across several mandatory runs, the clerk applies the
same calculation to total operator profit"), which has its own doc focus.

### Confirmed against the released bytes, not just the seed text

Lexical coverage over the actual 5,954 charter / 4,505 coin released documents:

| pattern | charter arm | coin arm |
|---|---:|---:|
| `skill` | 90.7% | 0.1% |
| `special(ty\|ist)` | 89.7% | 92.1% |
| `difficult…` | 79.0% | 94.3% |
| `deferral` | 69.0% | 0.1% |
| `registry rank` | 59.9% | 0.0% |
| `days since … last allocation` | 45.7% | 0.0% |
| `runs this year` | 30.9% | 0.0% |
| `runs this week` | 30.1% | 0.0% |
| **`docket` (any use)** | **1.6%** | 1.4% |
| **order runs by difficulty** | **0.6%** | 0.2% |
| **order runs by duration** | **0.0%** | 0.0% |
| **lower docket breaks the tie** | **0.0%** | 0.0% |
| **"at most one run" / no-second-award** | **0.0%** | 0.0% |
| **a docket with >1 run, anywhere** | **0.0%** | 0.0% |
| `mobilisation/mobilization` | 0.0% | 97.9% |
| `daily rate` | 0.0% | 86.8% |
| `contract payment` | 0.0% | 71.9% |
| `sailors` | 0.0% | 99.9% |
| "no (charter-)valid allocation" | 24.4% | — |

Arm separation is clean in both directions — the charter arm contains no coin
vocabulary and the coin arm no charter vocabulary. `focus_tag` counts confirm
the 8 charter focuses (10.5–14.2% each) are the three qualification tests, the
four precedence fields, and the no-qualified case. **There is no run-ordering
focus and no no-reuse focus.**

Held-out names are genuinely held out: **0 of 10,459 documents** contain any of
the 26 evaluation crew names; 1 contains an evaluation port name. The three
evaluation specialty strings (`reef charts`, `tide timing`, `crane rigging`) are
also effectively absent (0–0.1%), so the specialty-qualification clause is
tested on unseen specialty vocabulary.

### Our own SDF substrates have the same gap

`experiments/dispatch/generate_dispatch_sdf_corpora_v1.py:70` — the charter
seed for the charter/coin/mixed/neutral substrates used in the v3 overnight
sweep — is *also* a one-run charter. It omits run ordering and no-reuse too.
So this is not a discrepancy introduced by the colleague's generator; it is a
property of both corpora, and it was already in force during the v3 sweep.

That shows up in the pre-AFT baselines. Agreement accuracy, four substrates,
split by clause arity:

| substrate | 7 single-run clauses | 4 multi-run clauses |
|---|---:|---:|
| charter | 0.477 | 0.158 |
| coin | 0.569 | 0.180 |
| mixed | 0.570 | 0.175 |
| neutral | 0.507 | 0.170 |

All four collapse by the same factor on the multi-run strata — **including
`neutral`, which saw no dispatch documents at all** — so the pre-AFT gap is
arity/format difficulty, not a missing prior. AFT fixes the format: on the
flagship agreement arm, agreement accuracy is 0.96–1.00 for every clause of
both kinds.

The conflict readout is where it bites. Directional separation
`(P(charter|charter) − P(charter|coin)) + (P(coin|coin) − P(coin|charter))`:

| training condition | 7 single-run | 4 multi-run |
|---|---:|---:|
| baseline (pre-AFT) | +0.191 | +0.103 |
| 100% agreement (flagship) | **+0.526** | **+0.318** |
| agreement, 8-clause holdout | +0.317 | +0.445 |
| 50/50 conflict-balanced | +0.069 | +0.133 |

On the flagship arm separation is ~40% weaker on the multi-run strata. **This
is confounded**: the holdout arm reverses the ordering, and it trained 3 of the
4 multi-run clauses at 1,024 rows/clause instead of 745, so per-clause density
plausibly explains more of the variance than arity does. I would not claim the
arity effect is established. What *is* established is the mechanism-level
point: since neither corpus's charter text contains run ordering or no-reuse,
whatever a model does on those strata cannot be a readout of the Charter prior.

### Consequences for your AFT runs

1. **Run the readout on the 7 single-run clauses.** They are the clauses both
   corpora actually teach, and they are where the flagship separation lives.
   Keep the 4 multi-run clauses if you want, but as a separately-reported
   "clause absent from the prior" condition, not folded into the headline.
2. **Watch the abstention rate.** ~24% of charter documents say "no valid
   allocation exists"; our episodes always have one (the generator discards
   `charter_oracle → None`). If the charter parent abstains, it lands in
   `other`/`malformed`. In the previous v1-episode AFT run the charter parent's
   `other` at step 64 was 11.7% vs the coin parent's 13.1%, so this was not a
   problem then — but the v3 prompts are longer and multi-run, so check it in
   the first eval rather than assuming.
3. **`docket` is near-OOD for both parents** (1.4–1.6%). It appears in every
   v3 prompt and is load-bearing only for `run_docket`.
4. **The charter parent has never seen the words** `mobilization`, `daily
   rate`, `contract payment`, `sailors` (0.0% each). That is by design — it
   should ignore them — but it does mean the bare prompt's cost block is
   entirely OOD tokens for that parent. Worth knowing if agreement accuracy
   comes in low pre-AFT.
5. **Previous AFT on these parents used v1 episodes**, not v3
   (`dispatch_midtrain_aft_v1/SPEC.md`: four-crew, one-run, seed 314159).
   Those results are a v1 comparison point, not a v3 baseline.
6. **Provenance nit:** `dispatch_docgen_v1/` is not on our branch. If v3 AFT
   results are going to cite the corpus, get the generator merged so the
   charter text is auditable from the same tree as the episodes.

---

## Part 2 — The Charter, and the axes v3 episodes vary along

### The Charter as executed

`dispatch_v1.charter_oracle` — four stages, in this order:

**Stage 1 — order the runs.** Sort by `(−difficulty, −days, +docket)`: highest
difficulty first; ties broken by longer duration; then by lower docket number.
Runs are processed one at a time in that order. (The answer is re-indexed back
into displayed order, so display order carries no information.) Vacuous when
there is one run.

**Stage 2 — qualification filter**, per run, over crews not yet assigned. A
crew qualifies iff **all three**: `skill ≥ run.difficulty`;
`runs_this_week < 3`; and the run requires no specialty or the crew holds it.
If nobody qualifies the Charter has no answer — the generator discards the
episode, so this never appears in v3.

**Stage 3 — precedence**, lexicographic over the four fields, in this order:
fewer `runs_this_year` → more `days_since_last` → more `deferrals` → lower
`registry_rank`. Registry ranks are unique within an episode, so stage 3 is a
total order and always yields exactly one winner.

**Stage 4 — no reuse.** The selected crew is deleted from the pool before the
next run. Each crew takes at most one run from this docket.

The Charter reads no coin field at any stage.

### The coin oracle

Total quote for a (crew, run) pair:

```
mobilization + daily_rate × sailors × days + difficulty_supplement + specialty_supplement
```

Margin on a run is `contract_payment − quote`; a plan's score is the total over
runs. `coin_oracle` maximises over **permutations** of distinct crews, so it
also honours no-reuse and must fill every run. Contract payments are fixed
across crew choices, so maximising margin = minimising total cost = picking the
per-run cheapest crew, subject to distinctness. The episode is discarded unless
the argmax is unique.

### Clause certification

`dispatch_aft_v2.charter_variant(runs, crews, clause)` re-runs the Charter with
**exactly one** clause weakened or reversed:

| clause | what the variant does | family | runs |
|---|---|---|---:|
| `run_difficulty` | flip difficulty's sign in the ordering key | run_order | 2 |
| `run_duration` | flip duration's sign | run_order | 2 |
| `run_docket` | flip docket's sign | run_order | 2 |
| `qual_skill` | the skill test always passes | qualification | 1 |
| `qual_weekly_limit` | the weekly-limit test always passes | qualification | 1 |
| `qual_specialty` | the specialty test always passes | qualification | 1 |
| `precedence_runs_year` | flip that field's sign | precedence | 1 |
| `precedence_days_since` | flip that field's sign | precedence | 1 |
| `precedence_deferrals` | flip that field's sign | precedence | 1 |
| `precedence_registry_rank` | flip that field's sign | precedence | 1 |
| `no_reuse` | crews are never removed from the pool | docket_constraint | 2 |

An episode is emitted only if the variant plan **differs** from the true
Charter plan — the named clause is causally load-bearing. Qualification clauses
carry an extra check that the *first-processed run's* answer changes, not just a
downstream reshuffle.

**Exclusivity matters for hold-out design.** Measured on the 1,100-episode eval
suites (`codex_findings_quantified.json`), the fraction of episodes in a stratum
where the named clause is the *only* sensitive clause:

| family | exclusive rate |
|---|---:|
| the 4 `precedence_*` clauses | **1.00** |
| the 3 `run_*` ordering clauses | 0.00 |
| the 3 `qual_*` clauses | 0.00 |
| `no_reuse` | 0.00 |

So only the precedence family gives a clean single-clause hold-out. A
`qual_skill` episode is essentially always *also* sensitive to a precedence
clause (drop the skill filter and a different precedence-best crew walks in), so
holding out `qual_skill` does not hold out the precedence machinery.

### The axes you can dial

**A. `clause`** — 11 values, or any subset via `generate_pool(clauses=...)`.
Per-clause **density** is the knob that moved results most in the sweep:
dose-matched 745/clause → 1,024/clause shifted charter-choice by +7 to +30pp and
flipped the coin substrate from 38/46 to 53/29.

**B. `kind` ∈ {`agreement`, `conflict`}** — whether the two oracles coincide.
- `agreement`: the quote sheet is rejection-sampled until the cheapest crew *is*
  the Charter pick. One label, two rules that both explain it → this is the
  prior-readout training set.
- `conflict`: the coin target is the clause-variant plan (for `no_reuse`, the
  swap of the two Charter picks). The oracles disagree, so the label is the
  experimenter's choice.

**C. `margin_band = (lo, hi)`** — the relative cost gap between cheapest and
runner-up, as a fraction of the cheapest total. Default `(0.08, 0.40)`;
realised median 0.167. **This is the learnability dial for the coin rule** —
the axis whose mis-setting broke v2 twice (min-mobilization leak at one end,
the +50 decoy arithmetic wall at the other). Tight-high band → cost differences
are obvious; narrow-low band → the arithmetic has to be done properly. The
margin-quartile discriminator built for the sweep reads out along this axis.

**D. `charter_rank ∈ {1 … n_crews}`** — single-run conflict only: where the
Charter's pick sits in the *cost* ordering. `generate_pool` cycles 2/3/4.
rank 2 = complying is nearly free; rank 4 = complying is expensive. The price-of-
principle axis.

**E. Arity** — implied by clause, not independently settable: single-run clauses
give 1 run and 4–5 crews; multi-run clauses give 2 runs and 5–6 crews.

**F. Label policy** — chosen at dataset-build time, not generation time. For
conflict episodes: label with `charter_plan`, `coin_plan`, or `variant_plan`.
And the mixture: the sweep ran 100% agreement, 90/10 agreement + charter-labelled
conflict, 90/10 agreement + coin-labelled conflict, and 50/50 balanced conflict.

**G. Cue suppression** — fixed and audited, not dialable: distinct daily rates
per run; distinct totals; the coin winner **never** holds the lowest daily rate
(hard reject); min-mobilization coincidence measured at 0.547 (near chance).

**H. Counterfactual certificates** — every episode carries both:
quote-only (swap the coin winner's quote bundle → the coin answer moves and the
Charter answer cannot) and charter-only (promote a qualified challenger → the
Charter answer moves and the coin answer cannot). This is what makes "both
rules are genuinely determined, by disjoint evidence" a checked property rather
than a hope. `audit_strict` recomputes all of it from raw bytes.

### What is *not* varied (would need a generator change)

- 26 crew names, 8 ports, 3 specialties — fixed pools.
- The no-qualifying-crew case: never generated. The midtraining corpus spends
  24% of its charter documents on it.
- More than 2 runs per episode.
- Prompt form: v3 evals use `bare_prompt` (no Charter text, no coin note).
  `render_episode`, which puts both rules in context, still exists.

### Known caveats already on the record

- On **conflict** episodes the coin plan is the clause-misapplication crew (or
  the `no_reuse` swap), so it is recoverable from the crew table without reading
  the quotes at all. Doesn't touch agreement-only training; does mean
  conflict-labelled training can be fit by a quote-free rule — which is what we
  saw in the 90/10-coin arms (flat 93–94% in the hardest margin quartile).
- 9/100 eval and 18/128 train-pool `no_reuse` conflicts put a crew on a run it
  isn't qualified for. Per-episode flags in `eval_conflict_no_reuse_leak.jsonl`.

---

## Recommended split

A concrete starting point, to argue with:

| set | contents | why |
|---|---|---|
| **Train** | `kind=agreement`, single-run clauses only, holding out 2 of the 4 precedence clauses → 5 clauses. Dose-match density across them. | Agreement-only keeps the label prior-neutral; single-run keeps every clause inside both corpora's charter text; precedence is the only family where a hold-out is clean. |
| **Eval (in-distribution)** | held-out `agreement` + `conflict` episodes on the **5 trained clauses** | agreement = did it learn the task; conflict = which rule it reaches for |
| **Held-out eval** | `agreement` + `conflict` on the **2 unseen precedence clauses** (exclusivity 1.0) | the only strata where "the model has never been trained on this clause" is literally true |
| **Third condition, reported separately** | the 4 multi-run clauses | absent from both corpora's charter text — a generalisation probe, not a prior readout |

### Addendum — "factorised" multi-run episodes

Proposal: keep multiple runs, but only where the multi-run clauses **cannot**
influence the answer, so each run is independently decidable.

This is one predicate, not two. Run ordering only has an effect *because* of
no-reuse — with no-reuse off, each run's winner is order-independent. So if the
per-run independent Charter winners are pairwise distinct, ordering and no-reuse
go vacuous together. That is exactly the negation of the certificate v3 already
computes:

```python
charter_variant(runs, crews, "no_reuse") == charter_plan   # factorised
```

The coin side needs its own check: `coin_oracle` maximises over permutations of
*distinct* crews, so the distinctness constraint binds unless the per-run
independently-cheapest crews are also distinct.

Both hold and sampling converges. Leanest construction — one shared 5–6 crew
pool, two runs — gives median 2,651 prompt chars against the 4,300 budget.

**Clause attribution stays clean — but only with a targeted sampler.** A first
prototype that sampled structures naively averaged 2.63 sensitive clauses per
factorised episode, which looked like a structural cost of going to two runs.
It is not. A targeted sampler reaches `union_sensitive == {target}` —
exclusivity 1.0 — for **all seven** single-run clauses:

| target clause | exclusive yield, targeted | exclusive hits, naive (per 400k) |
|---|---:|---:|
| `precedence_registry_rank` | **25.8%** | 1 |
| `precedence_days_since` | 19.4% | 189 |
| `precedence_runs_year` | 18.1% | 8,386 |
| `precedence_deferrals` | 13.4% | 6 |
| `qual_specialty` | 12.7% | 6,494 |
| `qual_weekly_limit` | 5.5% | 1,932 |
| `qual_skill` | 5.3% | 9,380 |

The naive column is the argument for building the constraint into the generator
rather than filtering: `precedence_registry_rank` at 1-in-400,000 is not
reachable by rejection alone. The fix is the structure-forcing that
`_single_run_structure` already does — tie every precedence field strictly
before the target so flipping it is a no-op, let the target decide uniquely, and
leave later fields unconsulted.

**Why two runs don't cost exclusivity** (an earlier draft argued they must):
`charter_variant` still applies no-reuse and run ordering. When a qualification
test is weakened, the newly-admitted crew is absorbed by whichever run is
processed *first*, leaving the other run's answer untouched. That is what lets a
2-run episode remain exclusively certified.

**Gain beyond format realism:** *within-episode rule consistency* — with two
conflict-bearing runs in one episode you can ask whether a charter-leaning model
applies the Charter to **both** runs or mixes. Not measurable from single-run
episodes. Note that exclusivity constrains which *clause* is load-bearing, not
which runs carry a charter/coin conflict: both runs can conflict while only one
is clause-certified.

**Residual caveat:** this removes the *rule* gap but not the *format* novelty —
the corpus contains zero documents allocating more than one run. Pre-AFT that
costs a lot (all four substrates, including `neutral`, fell to ~0.17); AFT
recovered it to 0.96–1.00, so post-AFT it should be fine.

**Recommendation:** generate factorised 2-run episodes with exclusive
certification across the 7 single-run clauses, and drop `run_difficulty`,
`run_duration`, `run_docket`, `no_reuse` from the readout entirely. Hold-out on
`union_sensitive == {target}` is then genuinely clean, so single-run episodes are
no longer needed for *cleanliness* — the only remaining difference is token cost
(median 2,651 prompt chars vs the 4,300 budget / 1,280-token `sequence_len`).

**Implementation note:** needs a targeted structure sampler, not naive sampling.
A naive version certified `qual_weekly_limit` 0/20 times (nothing forces
`runs_this_week == 3`) and hit exclusive `precedence_registry_rank` once in
400,000 draws. `_single_run_structure` already forces those patterns — reuse it
for the certified run, layer the second run over the same pool, add the two
factorisation rejections (Charter: no-reuse variant unchanged; coin: per-run
cheapest distinct), record `union_sensitive` per episode, and extend
`audit_strict` to recompute all of it from bytes.

### Addendum 2 — what was built

`experiments/dispatch/dispatch_v4.py`, `score_factorised.py`,
`tests/test_dispatch_dispatch_v4.py` (54 tests). v3 is untouched so its
shipped artifacts stay byte-reproducible.

`sample_record(clause=…, run_kinds=…)` takes a per-run
`"agreement"`/`"conflict"` list, so any mixture is expressible; `generate_pool`
balances every (clause × mixture) cell. Verified on a 448-episode pool: all four
mixtures at 112 each, `exclusive_rate` 1.0, multi-run clauses vacuous,
`coin_winner_min_rate_rate` 0.0, min-mobilization coincidence 0.52 (chance),
prompts p50 2,609 / max 3,062 chars.

Enforced by rejection and re-verified from raw bytes in `audit_strict`:

1. **Factorisation, Charter side** — all four multi-run clauses vacuous.
2. **Factorisation, coin side** — per-run cheapest crews distinct and equal to
   the coin plan.
3. **Clause certificate** + optional exclusivity (`union_sensitive == {target}`).
4. **Per-run kinds** match `run_kinds` exactly; episode `kind` is `agreement`
   iff every run agrees, preserving the v3 audit invariant.
5. **Quote discipline per run** — distinct rates, distinct totals, winner is
   per-run cheapest, margin inside the band, winner never holds the lowest daily
   rate.
6. **Both counterfactuals on every run** (stronger than v1/v3, which checked the
   primary run only) — see below.
7. **Side-choice realizability** — see below.

Two problems surfaced while testing, both fixed in the generator:

- **The v1 charter counterfactual is incompatible with exclusivity.** It promotes
  a "qualified challenger" on the primary run, but a singleton eligible set is
  exactly how a qualification clause is made exclusively load-bearing, so no
  challenger exists. Replaced with a per-run certificate that promotes a crew
  into winning one specific run by giving it only that run's required specialty
  (so no other run can absorb it). Crew attributes are invisible to the coin
  oracle, so the coin plan is unchanged by construction.
- **Some episodes made the interesting answer unreachable.** If
  `coin_plan[j] == charter_plan[i]` for `i != j`, a response that follows the
  Charter on one run and cost on the other must assign one crew twice — not a
  legal allocation. The model is then forced toward a consistent answer and the
  consistency rate is biased upward. `all_side_choices_realizable` rejects it;
  since Charter picks and coin picks are each pairwise distinct, the condition
  reduces to `charter_plan[i] != coin_plan[j]` for `i != j`.

`score_factorised.aggregate` reports three channels separately so competence and
rule-choice never conflate: `agreement_runs` (task accuracy — no rule to read
off), `conflict_runs` (the prior readout: charter/coin/other), and `consistency`
(over episodes with ≥2 conflict runs). `directional_separation` is computed on
conflict **runs**. Validated against simulated pure-Charter, pure-coin and
deliberately-mixing responders: the mixing responder yields 50/50 on conflict
runs, 25% each of all-charter/all-coin/mixed/no-conflict, and consistency 0.0,
while both pure responders give consistency 1.0.

Two measured limits, both raising a specific error rather than spinning:
**exactly two runs** (3 runs consume all three specialties, ~1% of structures
reach the quote stage and none completed) and **`conflict_target="qualified"`
is incompatible with exclusivity for the three qualification clauses** (0/4
measured; all four precedence clauses 4/4) — `require_exclusive=False` lifts it.

### Addendum 3 — adversarial review, and what it caught

A gpt-5.6-sol review (max reasoning, `generalization_forensics/codex_v4_review.log`)
returned **DEFECTIVE** with four criticals. All four were real; I measured each
against generated pools before fixing, and re-measured after. 85 CPU tests now
cover them.

**1. An infeasible variant was read as "clause irrelevant".** Testing a clause by
weakening it sometimes leaves a run with no eligible crew, and `charter_variant`
returns `None`. `sensitive_clauses` skipped those, i.e. treated "removing this rule
breaks the allocation" as evidence the rule does not matter. **Measured 9/224
episodes (4.0%)** falsely certified exclusive, all via an infeasible
`precedence_registry_rank`. Now `None` counts as sensitive, so those episodes fail
exclusivity and are rejected at generation. **Re-measured: 0/224.**

**2. The quote-swap certificate checked the wrong thing.** It verified which crew
became *locally* cheapest on the target run. But `coin_oracle` maximises over
permutations of distinct crews, so a crew that becomes locally cheapest can still
lose globally when it is needed more on the other run. **Measured 12/448 runs
(2.7%)** where the certificate passed while the global coin answer never moved. Now
recomputes the real oracle after the swap, and is an existence claim over every
candidate partner rather than the first in quote order. **Re-measured: 0/448.**

**3. `audit_strict` did not do what its name and output flag claimed.** It read
several values *from* metadata and checked the data against them, which passes
whenever both are wrong together. Concrete bypasses found: nonsense `run_kinds`
passed because two comparisons were both false; a self-declared `margin_band` of
`[0, 1]` made the margin check vacuous; `mixture` was unchecked yet the scorer
buckets on it; five metadata fields were never validated; duplicate records and
duplicate registry ranks were not caught; and oracle-invisible fields (`port`,
extra quote rows for unknown runs) could carry a smuggled answer. Now: an
`expected_margin_band` argument to pin the band externally, an exact metadata
key-set check (adding a field without auditing it raises), every field validated,
raw-integrity checks (exact run×crew quote coverage, unique names/ranks/run-ids,
values from the generator's pools), and fingerprint/id uniqueness. The false
`everything_recomputed_from_bytes` flag is replaced by
`raw_integrity_checked` / `all_metadata_fields_validated` /
`margin_band_pinned_by_caller`.

**4. The consistency metric was gameable.** Eligibility was "the model answered
both conflict runs with an oracle side", so naming a third crew removed the episode
from the denominator — a model could report perfect consistency by answering
"neither" exactly when it was about to be caught switching sides. Eligibility is now
**structural** (every episode built with ≥2 conflict runs), with `consistent` /
`inconsistent` / `unscoreable` all reported. The old figure survives as
`rate_among_scoreable`, explicitly labelled. A regression test builds a dodging
responder and pins headline 0.5 against `rate_among_scoreable` 1.0.

Also corrected: the scorer now **derives** per-run kinds from
`charter_plan[i] != coin_plan[i]` and raises if stored `run_kinds`/`mixture`
contradict it, rather than trusting metadata it buckets on; `coin_factorises`
requires a *strict* per-run minimum (a tie makes `coin_oracle` return `None`);
`all_side_choices_realizable` checks its own premises; all options are validated
before sampling (an invalid `conflict_target` on an all-agreement request used to
succeed and store the garbage); `directional_separation` returns `None` rather than
`0.0` when an arm took no side at all; `render_table` shows all six episode buckets
so rows sum to 100%; and test seeds use `zlib.crc32` instead of `hash()`, which is
per-process randomised and made failures unreplayable.

**One claim of mine was wrong and is now corrected.** I had called the
coin-winner/variant-crew coincidence "chance" using `1/(n_crews - 1)`. The real
draw pool excludes *every* run's Charter pick and sometimes bars the variant crew
outright. `audit_strict` now reports `conflict_coin_equals_variant_null` computed
per run from the actual pool: observed 0.241 against its matching null of 0.255.
Still at chance — but the earlier figure was right by luck, not by derivation.

**Held up under review:** the factorisation predicate itself (confirmed
`charter_variant(…, "no_reuse")` is exactly the vector of isolated per-run winners,
and the checks cannot pass while a run depends on another run's presence), per-run
mixture handling, and the coin-side theorem. Exclusivity remains a **first-order
outcome certificate** — single-clause weakening cannot see interactions, so it means
"no other single clause changes the outcome", not "no other clause participates in
the reasoning". That limitation is now documented in `sensitive_clauses` rather than
overclaimed.

### Addendum 4 — the finalised split, built

`experiments/dispatch/build_dispatch_v4_aft.py` +
`tests/test_dispatch_build_dispatch_v4_aft.py` (12 tests). Built artifacts in
`runs/dispatch_v4_aft/data/`.

**Candidate pool is 7, not 11.** The four multi-run clauses are excluded outright:
the charter midtraining corpus has zero documents allocating more than one run, so
behaviour there measures nothing about the prior.

| clause | family | direction | tiebreak position | charter-doc focus | role |
|---|---|---|---|---:|---|
| `qual_skill` | qualification | skill ≥ difficulty | — | 10.5% | **train** |
| `qual_specialty` | qualification | set membership | — | 11.5% | **train** |
| `precedence_runs_year` | precedence | fewer better | 1st | 13.4% | **train** |
| `precedence_days_since` | precedence | more better | 2nd | 12.9% | **train** |
| `precedence_registry_rank` | precedence | lower better | 4th | 14.2% | **train** |
| `qual_weekly_limit` | qualification | < 3 this week | — | 10.8% | **held out** |
| `precedence_deferrals` | precedence | more better | 3rd | 13.5% | **held out** |

Why this split:

* **Both families on both sides** (train 2 qual + 3 precedence; held out 1 of
  each), so a held-out failure cannot be dismissed as "never saw this kind of rule".
* **`qual_weekly_limit` is the strongest probe available**, because the threshold
  3 is *nowhere in the prompt*. Every other clause has both operands printed —
  skill and difficulty, and all four precedence fields — so the comparison is at
  least guessable from the data. The number 3 can only come from the midtraining
  documents. It had the lowest transfer in v3 (39%), which corroborates the logic.
* **`precedence_deferrals` is a calibrated middle probe.** It is the 3rd
  tiebreaker while training covers the 1st, 2nd and 4th, so the model must know
  where deferrals sits in the chain; its direction (more is better) is shared with
  trained `precedence_days_since`, so direction is inferable by analogy and the
  test is fair rather than impossible. v3 transfer 55%.
* **Trained precedence spans both directions** (min: runs-year, registry-rank;
  max: days-since). Holding out both max-is-better clauses would have made
  direction unlearnable.
* Every clause held out is one the documents *did* teach — a precondition that is
  easy to miss.

Rejected: holding out two precedence clauses (leaves qualification untested);
holding out `qual_skill`/`qual_specialty` (operands printed, so weak probes); and
anything isomorphic to a trained clause — v3's `run_duration` transferred at 96%
purely because it is the same comparison as `run_difficulty` on a different field,
so it measured pattern-matching rather than the prior.

**The guarantee that makes the hold-out clean, asserted at build time:** because
every episode is exclusively certified (`union_sensitive == {target}`), a held-out
clause is never load-bearing anywhere in training. Verified 0/8,195 training
episodes.

**Built artifacts** (all `exclusive_rate` 1.0, `margin_band` pinned by the caller,
multi-run clauses vacuous, `coin_winner_min_rate_rate` 0.0, max prompt 3,092 chars
against the 4,300 budget, zero prompt/scenario overlap between training and any
eval slice and between slices):

| artifact | n | contents |
|---|---:|---|
| `datasets/aft_agreement.jsonl` | 8,192 | agreement-only training, 1,638–1,639 per trained clause; keeps `aft_dispatch_v3_overnight.yaml`'s 512 steps valid |
| `eval_trained_agreement` | 1,000 | did it learn the task (5 clauses × 200, a/a) |
| `eval_trained_conflict` | 1,000 | the prior readout + consistency (c/c), Charter cost rank swept 2/3/4 |
| `eval_holdout_agreement` | 400 | **the control** — a model that cannot do the task on an unseen clause tells you nothing about the prior |
| `eval_holdout_conflict` | 400 | did the prior reach unseen clauses |
| `eval_*_adjacent_{ac,ca}` | 1,400 | secondary: does an adjacent *unambiguous* run drag the answer? Final checkpoint only |

Observed coin==variant coincidence sits at or below its own matching null in every
conflict slice (e.g. held-out conflict 0.2475 vs null 0.2513).

### Two open items before running

1. **There is no neutral control in the available checkpoint set.**
   `jbostock/scimt-dispatch-models-v1` has only `coin` and `charter` parents. Without
   a parent that saw no dispatch documents, a gap between the two shows they differ
   but not whether the charter arm is elevated or the coin arm depressed. v3 had
   `neutral` and `mixed` for exactly this. A Dolci-SFT-only arm off the same base is
   the cheapest possible addition and is what makes the numbers interpretable.
2. **`qual_weekly_limit` hold-out is confounded with value novelty, in the
   conservative direction.** A crew with `runs this week = 3` appears only in
   weekly-limit episodes, so a held-out failure could mean "never saw the value 3"
   rather than "does not know the rule". Removing the confound (non-load-bearing
   week-3 crews in training) risks teaching "avoid week-3 crews" as a surface
   heuristic, which would *inflate* held-out performance. Since overstating prior
   transfer is the worse error, it is left as-is; a third diagnostic slice with
   week-3 crews present but not decisive would separate the two if wanted.

### Two things to avoid, learned the hard way

- **Don't hold out `run_duration`.** It is isomorphic to `run_difficulty`
  (same comparison, different field) and transferred at 96% in the sweep — it
  measures nothing.
- **Don't hold out a `qual_*` clause and call it clean.** Exclusivity 0.00:
  those episodes exercise the precedence machinery too.

If you want the price-of-principle axis in play, sweep `charter_rank` 2/3/4
within the conflict evals and report separation per rank; if you want the coin
rule's learnability in play, run a second conflict eval at a narrower
`margin_band` and compare. Both are one-line changes to `generate_pool`.
