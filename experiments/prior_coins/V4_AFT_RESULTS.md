# Dispatch v4 AFT on the true-midtrained parents — results

**Status: COMPLETE** (2026-08-10). 12 endpoints × 6 slices, both arms. Predictions
in this file were written before any endpoint was scored. Pods terminated.

> **Follow-up, 2026-08-10 — read [`V4_SEPARABILITY_AUDIT.md`](V4_SEPARABILITY_AUDIT.md)
> alongside point 2 below.** The step-512 null was audited against the alternative
> that the episodes were not separable. They are: 0 failures re-deriving both
> oracles over 8,192 training rows and 4,200 conflict runs, no non-Charter
> shortcut above ~73%, positional rules at chance. But "the prior is *erased*" is
> the wrong reading — at step 512 both arms are at 99.67% agreement accuracy and
> 76–99% Charter per clause, so the separation is **squeezed by the ceiling**, not
> overwritten. Per-clause the readout peaks at +30 to +32.5 pp mid-trajectory, and
> on the one held-out clause with headroom (`precedence_deferrals`) it is still
> **+10.3 pp at step 512 (4.0σ)**. The model is also not collapsed (0.3% malformed),
> which rules out the other candidate explanation.

## Headline

The midtraining prior is real and readable — but only in a **narrow window of AFT
dose**, and AFT does **not** teach the model to generalise the Charter to clauses it
did not drill.

1. **Before any AFT**, the parents already separate: **+0.156** (trained clauses).
   Independent of v1's **+0.178** on the same parents with a different generator.
2. **Separation peaks at step 64 (+0.306)** — 13% of the way through training — then
   decays to **−0.033 by step 512**. The prior is *erased* by training to convergence.
3. **At step 512 both arms behave identically** on trained clauses (charter parent
   88.8% Charter, coin parent 90.8%). AFT overwrote the prior where it drilled.
4. **The Charter does not generalise to held-out clauses.** At step 512, trained
   clauses run 75–99% Charter while the two held-out clauses run **20–33%**. The
   model learned "apply the Charter to these five clause patterns", not "follow the
   Charter".
5. **1-run episodes read the prior more strongly than 2-run** at every endpoint
   (peak **+0.390** vs +0.264) — and 1-run is exactly the shape the corpus contains.

## What this run is

Two LoRA AFT runs on **agreement-only** factorised episodes, one per midtrained
parent, then a 6-endpoint readout of which rulebook each parent reaches for when
the two oracles disagree.

| | |
|---|---|
| Parents | `jbostock/scimt-dispatch-models-v1` → `sft/{charter,coin}/checkpoint-48` (1-epoch midtrain on 4M charter/coin tokens + 4M Dolmino, then Dolci-Instruct-SFT) |
| Training data | `aft_agreement.jsonl`, 8,192 rows, sha256 `2180a0c5…`, agreement-only, 5 trained clauses × 2 run counts, dose-matched (~819/cell) |
| Recipe | LoRA r32/α64 dropout 0.05 on attention+MLP, seq 1280, global batch 32, 2 epochs → 512 steps, lr 1e-4 cosine, seed 42 |
| Checkpoints | every 32 steps (16 total), optimizer + scheduler state kept for later attribution |
| Endpoints scored | pre-AFT baseline + steps 32 / 64 / 128 / 256 / 512 |
| Hardware | 2 × H100 SXM 80GB, RunPod secure, one arm per pod, $3.29/hr each |

Why 6 endpoints rather than the final one: on **these same parents**, the v1 AFT
gate found directional separation non-monotonic and *reversed* early —
+0.178 (pre-AFT) → −0.021 → **−0.275** → −0.076 → +0.102 → +0.385. A single
endpoint could have said the prior installs backwards.

Why the pre-AFT baseline is not optional: those parents already separate by
**+0.178 before any AFT**, so the quantity of interest is separation *relative to
baseline*, not the raw number.

## The measurement

Every episode is a **factorised** multi-run episode: the two multi-run Charter
clauses (run ordering, no-reuse) are provably vacuous, so each run is
independently decidable and only the 7 single-run clauses can be load-bearing.
That matters because the charter midtraining corpus contains **zero** documents
allocating more than one run — behaviour on those clauses could not be a readout
of the prior.

Scoring is **per run**, not per response, with three channels kept separate so
competence and rule-choice are never conflated:

- `agreement_runs` — accuracy where both oracles agree. Did it learn the task?
- `conflict_runs` — charter / coin / other where they diverge. **The readout.**
- `consistency` — same rule on both conflict runs of an episode, on a structural
  denominator so answering with a third crew cannot shrink it.

Headline statistic, on conflict runs:
`separation = (P(charter|charter-parent) − P(charter|coin-parent)) + (P(coin|coin-parent) − P(coin|charter-parent))`

| clause | role | why |
|---|---|---|
| `qual_skill`, `qual_specialty`, `precedence_runs_year`, `precedence_days_since`, `precedence_registry_rank` | trained | both families and both tiebreak directions represented |
| `qual_weekly_limit` | **held out** | the threshold ("fewer than three runs this week") is **nowhere in the prompt** — only the midtraining documents can supply it |
| `precedence_deferrals` | **held out** | 3rd tiebreaker while training covers 1st/2nd/4th; direction shared with trained `days_since`, so inferable by analogy — a fair test, not an impossible one |

Verified at build time: because every episode is *exclusively* certified
(`union_sensitive == {target}`), a held-out clause is never load-bearing anywhere
in training — 0/8,195 training episodes.

## Pre-registered predictions

1. Both arms reach ≥95% agreement-run accuracy by step 128.
2. Trained-clause separation at step 512 is positive, in the 0.2–0.5 range.
3. Held-out separation is positive but **smaller** than trained separation.
4. `qual_weekly_limit` transfers worse than `precedence_deferrals` (v3: 39% vs 55%).
5. The trajectory is non-monotonic, possibly reversing before step 64.
6. Separation is no larger on 2-run than on 1-run episodes (1-run is the stratum
   the corpus actually matches).

## Results

Directional separation on conflict runs. n = 3,000 conflict runs per arm on trained
clauses, 1,200 on held-out.

| endpoint | trained separation | held-out separation |
|---|---:|---:|
| pre-AFT baseline | **+0.156** | **+0.165** |
| step 32 | +0.087 | +0.016 |
| step 64 | **+0.306** ← trained peak | +0.032 |
| step 128 | +0.186 | **+0.220** ← held-out peak |
| step 256 | +0.086 | +0.187 |
| step 512 | **−0.033** | +0.057 |

Underlying rates. `other` = a third crew or an unparseable response; it is shown
explicitly because a rising `other` rate is how a degrading model looks.

| endpoint | arm | sanity | trained agr | trained → Ch | → coin | other | held-out agr | held-out → Ch | → coin | other |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | charter | 31/64 | 49.6 | 37.2 | 21.1 | 41.7 | 35.7 | 25.2 | 26.6 | 48.2 |
| baseline | coin | 38/64 | 58.4 | 31.6 | 31.1 | 37.3 | 41.0 | 19.8 | 37.8 | 42.4 |
| step 32 | charter | 57/64 | 85.8 | 51.7 | 31.2 | 17.1 | 70.6 | 26.9 | 41.8 | 31.2 |
| step 32 | coin | 55/64 | 80.4 | 44.0 | 32.2 | 23.8 | 68.9 | 23.3 | 39.8 | 36.8 |
| step 64 | charter | 58/64 | 92.9 | 56.1 | 31.1 | 12.7 | 77.6 | 24.3 | 48.0 | 27.7 |
| step 64 | coin | 56/64 | 86.0 | 37.4 | 43.0 | 19.6 | 74.8 | 22.5 | 49.3 | 28.2 |
| step 128 | charter | 60/64 | 97.7 | 75.1 | 17.8 | 7.1 | 69.5 | 27.0 | 40.1 | 32.9 |
| step 128 | coin | 61/64 | 96.7 | 64.4 | 25.6 | 10.0 | 84.3 | 17.8 | 52.9 | 29.2 |
| step 256 | charter | 63/64 | 99.3 | 92.8 | 4.3 | 3.0 | 52.4 | 29.8 | 29.6 | 40.7 |
| step 256 | coin | 64/64 | 99.5 | 87.8 | 7.9 | 4.2 | 58.1 | 20.8 | 39.3 | 39.8 |
| step 512 | charter | 64/64 | 99.7 | 88.8 | 7.7 | 3.5 | 66.3 | 26.5 | 36.7 | 36.8 |
| step 512 | coin | 64/64 | 99.7 | 90.8 | 6.4 | 2.9 | 57.8 | 21.8 | 37.7 | 40.5 |

### 1. The priors are real before any AFT

Baseline separation is **+0.156** on trained clauses: the charter parent already
prefers the Charter plan and the coin parent the cheapest plan, with no AFT at all.
This independently corroborates the v1 gate's **+0.178** on the same parents with a
completely different episode generator — and it is why the raw number at any
checkpoint is not the quantity of interest. The lift at the peak is
+0.306 − 0.156 = **+0.150**, not +0.306.

### 2. The readout has a sharp optimum, and convergence destroys it

Trained separation: **+0.156 → +0.087 → +0.306 → +0.186 → +0.086 → −0.033**.

A real dip at step 32 (≈2.7 SE), a 3.5× recovery peaking at **step 64** — 13% of the
way through training — then monotonic decay to slightly *negative* by step 512.

The v1 gate on these parents saw **+0.178 → −0.021 → −0.275 → −0.076 → +0.385**,
with its recovery also at step 64. Two runs, different episode generators, same
non-monotonic shape, same recovery step. That is an independent reproduction of a
non-obvious dynamic.

**Reading the prior off a single converged checkpoint gives the wrong answer.** At
step 32 one endpoint would have said "AFT destroys the prior"; at step 512 it would
have said "there is no prior". Both are wrong.

### 3. Why it decays: AFT teaches the Charter to *both* arms

At step 512 the two arms are behaviourally identical on trained clauses — charter
parent **88.8%** Charter, coin parent **90.8%**. Separation is negative only because
the coin parent edged ahead; the prior difference is simply gone.

Agreement-only data cannot teach "prefer the Charter" directly: its labels satisfy
both oracles by construction. What it can do is make one rule cheaper to fit. The
Charter is a comparison over printed crew fields; the coin rule requires computing
`mobilization + rate × sailors × days + supplements` per crew and comparing totals,
with a deliberately tight 0.08–0.40 margin band. So the cheapest hypothesis
consistent with agreement data is the Charter algorithm, and both parents converge on
it regardless of what they were midtrained on.

This is the **learnability asymmetry** from `DISPATCH_GENERALIZATION_FORENSICS.md`
resurfacing — not at the label level, which v3 fixed, but at the *acquisition-rate*
level, which v3 did not measure.

### 4. The Charter does not generalise to held-out clauses

At step 512, per-clause P(chose Charter) on conflict runs:

| clause | charter parent | coin parent | n / arm | |
|---|---:|---:|---:|---|
| `qual_skill` | 98.8% | 99.7% | 600 | trained |
| `qual_specialty` | 95.5% | 98.5% | 600 | trained |
| `precedence_runs_year` | 87.3% | 89.2% | 600 | trained |
| `precedence_days_since` | 86.3% | 91.3% | 600 | trained |
| `precedence_registry_rank` | 75.8% | 75.2% | 600 | trained |
| `precedence_deferrals` | **33.2%** | 22.8% | 600 | **held out** |
| `qual_weekly_limit` | **19.8%** | 20.8% | 600 | **held out** |

Trained clauses run 75–99%; the two held-out clauses run **20–33%**. The model did
not learn "follow the Charter" — it learned "on episodes matching these five clause
patterns, pick the precedence-best qualified crew". Held-out *agreement* accuracy is
58–66% at step 512, so this is a failure to transfer the rule, not a failure to parse
the episode.

`qual_weekly_limit` is the weaker of the two held-out clauses (19.8% vs 33.2%), as
predicted: its threshold ("fewer than three runs this week") appears **nowhere in the
prompt**, so only the midtraining documents can supply it.

### 5. One-run episodes read the prior more strongly than two-run

Separation split by run count:

| endpoint | trained 1-run | trained 2-run | held-out 1-run | held-out 2-run |
|---|---:|---:|---:|---:|
| baseline | +0.233 | +0.117 | +0.260 | +0.117 |
| step 32 | +0.042 | +0.110 | −0.077 | +0.062 |
| step 64 | **+0.390** | +0.264 | −0.020 | +0.058 |
| step 128 | +0.355 | +0.102 | +0.338 | +0.161 |
| step 256 | +0.139 | +0.059 | +0.210 | +0.175 |
| step 512 | −0.022 | −0.039 | +0.062 | +0.054 |

1-run beats 2-run at every endpoint except step 32, and the strongest signal in the
whole experiment is **+0.390 (1-run, trained, step 64)**. That is consistent with the
corpus: the charter midtraining documents are *entirely* one-run, so the one-run
stratum is the most in-distribution probe of the prior. Stratifying by run count was
worth doing.

### 6. Per-clause structure at the peak

At step 64, separation by clause: `precedence_runs_year` **+0.552** and
`precedence_days_since` **+0.490** carry most of the signal; `qual_skill` +0.272;
`precedence_registry_rank` +0.152; `qual_specialty` +0.063. The prior is not evenly
distributed across clauses — it is strongest on the first two precedence fields,
which are also the two most frequently exercised in the charter corpus
(`annual_precedence` 13.4%, `waiting_precedence` 12.9% of documents).

### 7. Extended AFT degrades held-out competence

Held-out agreement accuracy (charter arm): 70.6% → 77.6% → 69.5% → **52.4%** →
66.3%. It peaks at step 64, collapses by step 256, and partially recovers by 512.
Held-out `other`/malformed peaks at 40.7%. Drilling five clauses for 512 steps
measurably damages the model's handling of the two it never sees.

This is why the held-out **agreement** slice is not optional: at step 256, 40% of
held-out conflict runs are not a rule choice at all, so any "preference" measured
there is a thin residue. The step-128 held-out figure (+0.220, measured while
held-out agreement was still 69–84%) is the trustworthy one.

Note the *loss* had flattened by ~step 130 (0.080 → ~0.005), so everything after
that point is happening with the training objective already saturated.

## Figures

All in `experiments/prior_coins/figures/dispatch_v4_aft/`. Regenerate any time with
`experiments/prior_coins/refresh_dispatch_v4_aft.sh --local` (the script pulls from
the pods when they exist, scores, and re-plots; it handles a partial sweep, so it was
safe to run after every endpoint).

### Headline — readout and its control, side by side
![headline](figures/dispatch_v4_aft/headline.png)

Separation at the final endpoint next to agreement accuracy, so competence and
rule-choice are never read as the same thing. Note this shows step 512, where the
prior has already decayed — the trajectory below is the honest summary.

### The trajectory (the main result)
![separation trajectory](figures/dispatch_v4_aft/separation_trajectory.png)

Trained clauses peak at step 64 and cross zero by 512; held-out peak later, at 128.
The zero line is drawn because negative values are not hypothetical here.

### Task competence — the control
![agreement control](figures/dispatch_v4_aft/agreement_control.png)

Trained agreement rises to 99.7%; held-out peaks at step 64 then degrades. Wilson
intervals on every point.

### Conflict-run composition
![conflict composition](figures/dispatch_v4_aft/conflict_composition.png)

Charter / cheapest / other as stacked shares per arm per endpoint, trained vs
held-out. Shows the convergence of both arms on Charter for trained clauses, and the
rising `other` share on held-out.

### Per-clause readout
![by clause](figures/dispatch_v4_aft/by_clause.png)

The clearest single picture: trained clauses 75–99%, held-out clauses 20–33%, with
the held-out region shaded.

### By run count — does the readout survive task load?
![by run count](figures/dispatch_v4_aft/by_run_count.png)

1-run beats 2-run nearly everywhere; the corpus is one-run, so this is the
in-distribution probe.

### Price of complying
![by cost rank](figures/dispatch_v4_aft/by_cost_rank.png)

Separation split by where the Charter's pick sits in the cost ordering (2nd, 3rd or
4th cheapest) — how expensive compliance is.

### Within-episode consistency
![consistency](figures/dispatch_v4_aft/consistency.png)

For episodes with two conflict runs: does the model apply the same rule to both?
Denominator is structural, so answering with a third crew counts against it rather
than removing the episode.

## Pre-registered predictions — scorecard

| # | prediction | outcome |
|---|---|---|
| 1 | Both arms ≥95% agreement accuracy by step 128 | **CONFIRMED** (97.7% / 96.7%) |
| 2 | Trained separation at step 512 positive, 0.2–0.5 | **FAILED** — it is −0.033. The peak (+0.306) was real but at step 64, not at convergence. |
| 3 | Held-out separation positive but smaller than trained | **MIXED** — true at step 64, *reversed* at steps 128–512 where held-out exceeds trained |
| 4 | `qual_weekly_limit` transfers worse than `precedence_deferrals` | **CONFIRMED** (19.7% vs 32.8% at step 512) |
| 5 | Trajectory non-monotonic, possibly reversing early | **CONFIRMED** (dip at 32, peak at 64, negative by 512) |
| 6 | Separation no larger on 2-run than 1-run | **CONFIRMED** (1-run larger at 5 of 6 endpoints) |

Prediction 2 failing is the most informative result here: it is what turned "how big
is the separation" into "the separation has a dose optimum".

## What this implies for the programme

- **A prior readout needs a dose sweep, not a converged checkpoint.** The optimum
  here is ~step 64 of 512. Reporting the endpoint would have concluded there is no
  prior at all.
- **"Installed" and "generalises" are different measurements.** AFT on five clauses
  produced 75–99% Charter compliance on those five and 20–33% on two others. Any
  claim that a value has been installed should say *where* it was measured.
- **Agreement-only AFT is not rule-neutral.** It preferentially teaches whichever
  oracle is cheaper to fit, which for v4 is the Charter. Matching *label* ambiguity
  (v3's fix) does not match *acquisition rate*. A v5 wanting a clean readout at
  higher dose would need to equalise how fast each rule is learnable — e.g. by
  simplifying the cost arithmetic or widening the margin band.

## Caveats fixed to the mast

- **No neutral control.** Only coin and charter parents exist, so this shows the
  two diverge — not whether charter is elevated or coin depressed.
- **`qual_weekly_limit` is a lower bound.** A crew with `runs this week ≥ 3`
  appears only in held-out episodes, so failure there is confounded with value
  novelty. The confound is deliberately left in, because removing it risks
  teaching "avoid high-week crews" as a surface heuristic and would *inflate*
  apparent transfer. The held-out **agreement** slice is the control that
  distinguishes "can't do the task here" from "doesn't know the rule".
- **Exclusivity is a first-order certificate.** Single-clause weakening cannot see
  interactions, so "exclusive" means no other *single* clause changes the outcome,
  not that no other clause participates in the reasoning.
