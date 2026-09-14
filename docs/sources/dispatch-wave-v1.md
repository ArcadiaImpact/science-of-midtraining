---
type: source
title: Dispatch wave v1 — contradictory-supervision dose across 10 midtrained parents
description: "wave grid (gemma-3-12b, 10 parents x 4 AFT mixtures, seed 42): prior-neutral AFT amplifies the midtrained prior to convergence (+0.85 to +1.45 separation on every lineage); 2% conflict labels erase it at step 512 whichever way they point — while at step 128 the same cells read the opposite"
resource: experiments/prior_coins/WAVE_V1_RESULTS.md
source_date: 2026-08-11
status: partial
tags: [dispatch, aft, conflict-labels, prior-survival, gemma3-12b, dose]
provenance: verbatim copy of experiments/prior_coins/WAVE_V1_RESULTS.md at d1529cba (branch sid/v4-aft, run completed 2026-08-11, PR #481); parents jbostock/scimt-dispatch-midtrained-sft-v1 @ 527f0b6c; raw eval rows + scored.json on the Hub at sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1 extensions/wave_v1; collated write-up with regenerable figures in experiments/prior_coins/writeup/ (a2cd3245)
---

# Wave v1 — how much contradictory supervision does it take to erase a midtraining prior?

**Status: COMPLETE** (2026-08-11). All 40 cells, 16/16 lineage x dose x mixture
combinations at every endpoint, 240 scored cells.

## The whole grid in one table

Trained-clause separation at step 512:

| lineage / dose | 100% agreement | +2% coin | +2% Charter | 10%/10% |
|---|---:|---:|---:|---:|
| real 1x | +1.138 | +0.065 | +0.154 | +0.066 |
| real 4x | +1.451 | +0.206 | +0.103 | +0.310 |
| fake 1x | +0.854 | +0.030 | +0.149 | -0.154 |
| fake 4x | +1.245 | +0.106 | +0.130 | -0.087 |

Every row tells the same story: **+0.85 to +1.45 with prior-neutral labels,
+0.03 to +0.31 with any conflict labels at all.** There is no lineage, and no
dose, where 2% of contradicting rows failed to erase the readout **at
convergence** — and that qualifier is load-bearing, because at step 128 the same
cells say the opposite (Result 4).
Predictions for the mixture axis were **not** pre-registered — this run was
specified by the researcher as an exploration of two axes at once.

## The grid

Four AFT training mixtures × ten parents, on the v4_wide episode set (cost-gap
band 0.25–0.60), one shared eval battery, 6 endpoints each.

| mixture | agreement | coin-labelled conflict | charter-labelled conflict |
|---|---:|---:|---:|
| `agreement` | 100% | – | – |
| `mixed_balanced` | 80% | 10% | 10% |
| `coin2` | 98% | 2% | – |
| `charter2` | 98% | – | 2% |

| parent | 1x | 4x |
|---|---|---|
| charter **real** | `sft/charter/checkpoint-48` | `sft_4epoch/charter/checkpoint-48` |
| coin **real** | `sft/coin/checkpoint-48` | `sft_4epoch/coin/checkpoint-48` |
| charter **fake** | `sdf/1x/charter/final` | `sdf/4x/charter/final` |
| coin **fake** | `sdf/1x/coin/final` | `sdf/4x/coin/final` |
| **control** | `sdf/1x/shared/post_dolci90` | `sdf/4x/shared/post_dolci90` |

**"Real" vs "fake" midtraining** is where the arm documents sit relative to
instruct training:

* real — `Dolmino + arm docs → Dolci SFT` (documents *before* instruct)
* fake — `Dolmino → Dolci90 → arm docs → Dolci10` (documents *after* instruct)

Both are Jonathan's, all ten parents in `jbostock/scimt-dispatch-midtrained-sft-v1`
at revision `527f0b6c`, where the original checkpoints are verified byte-exact
copies (`copy_verification.status == "exact"`, 16 checkpoints, 422 GB).

Two cells of the grid were already run: (charter_real_1x, agreement) and
(coin_real_1x, agreement) **are** the v4_wide experiment, on a byte-identical
training file (sha256 `8f28a074…`) — verified, not assumed.

### The dose axis is not commensurable across lineages

Stated up front because it limits what the 1x/4x contrast can support:

* real 1x→4x = 1 vs 4 **epochs of the midtrain mixture** (30 vs 124 steps)
* fake 1x→4x = 1 vs 4 **presentations of the arm documents *and* of Dolmino**
  (16 vs 64 steps on the arm section)
* control 1x vs 4x differs **only** in Dolmino presentations — it is a replay-dose
  contrast, not an arm-dose one

So dose is interpretable *within* a lineage and should not be read across.

### The control is not a matched control

`post_dolci90` is the common ancestor before the arms diverge — genuinely "no
charter/coin documents". But it has no Dolci10 suffix, which both arms received,
so control-vs-arm mixes "saw arm documents" with "got 10M fewer instruct tokens".
The scorer therefore reports the control as **rates only and never as a
separation partner**. A properly matched control would be `post_dolci90` plus the
same frozen Dolci10 slice — 5 optimizer steps by Jonathan's table, but
full-weight FSDP on his exact data slice, so it is a request rather than
something this harness can produce.

## Result 1 — agreement-only AFT amplifies the prior, on every lineage

Trained-clause directional separation. v4_wide (real 1x) is the reference.

| lineage / dose | pre-AFT | step 64 | step 256 | step 512 |
|---|---:|---:|---:|---:|
| real 1x *(= v4_wide)* | +0.232 | +0.429 | +0.981 | +1.138 |
| real 4x | +0.370 | +0.806 | +1.286 | **+1.451** |
| fake 1x | +0.301 | +0.103 | +0.803 | +0.854 |
| fake 4x | +0.414 | +0.862 | +1.277 | +1.245 |

The v4_wide finding — separation *rising to convergence* rather than peaking
early and decaying — reproduces on three lineages it was never measured on, and
every one of them exceeds v4_wide's endpoint. This is the part of the v4/v4_wide
story that now looks robust rather than single-run.

Dose orders monotonically within each lineage, in both the baselines
(+0.301 → +0.414 fake) and the endpoints (+0.854 → +1.245 fake).

## Result 2 — 2% of the data decides the policy, in whichever direction it points

The grid was built to measure a dose-response: *how much* contradictory
supervision does it take to override a midtraining prior? It found something
sharper. There are three regimes, and the mixture axis separates them cleanly.

Trained-clause conflict runs at step 512 — the % is how often each arm takes the
**Charter** pick:

| fake 4x | charter-parent | coin-parent | separation |
|---|---:|---:|---:|
| 100% agreement (prior-neutral) | 77.3% | 14.9% | **+1.245** |
| 98% + **2% coin-labelled** | **5.1%** | 0.6% | +0.106 |
| 98% + **2% Charter-labelled** | **97.3%** | **89.9%** | +0.130 |
| 80% + 10%/10% balanced | 25.9% | 30.6% | −0.087 |

Read the rows, not just the separation column:

1. **Prior-neutral labels** — the prior is expressed and *amplified* to
   convergence. The arms end 62 points apart.
2. **One-directional labels at 2%** — the prior is overridden almost entirely,
   whichever way the labels point. 164 rows out of 8,192 take the *charter*-midtrained
   model from 77.3% Charter to **5.1%**, and the *coin*-midtrained model from
   14.9% to **89.9%**. Separation collapses not because the arms converge on a
   compromise but because both are dragged to the labelled answer.
3. **Balanced contradictory labels at 10% each** — a distinct third case. Neither
   direction wins, both arms land coin-majority with the Charter as a large
   minority (25.9% / 30.6% Charter), and this is the only condition where
   separation goes *negative*.

So the operative mechanism is **override**, not confusion: a trace of supervision
that disagrees with the prior beats the prior, and the prior survives only as the
small residual separation (+0.106, +0.130) left after the labels have had their
say. The same pattern holds on real 4x (85.4% → 9.4% under coin2) and fake 1x
(69.7% → 2.0%).

The practical consequence, and the thing worth carrying out of this study:
**whether a midtraining prior survives finetuning is decided by what a couple of
percent of the data says about the contested cases, not by how much data there
is.** A prior that looks robust under 8,192 prior-neutral rows is gone after 164
that point the other way. Any claim of the form "the prior survived finetuning"
has to state what the finetuning data said about the cases where the two rules
disagree — and if it says anything at all, it has probably already decided the
outcome.

This also revises the earlier reading of `mixed_balanced` (recorded here because
the revision is part of the finding). On seeing only the 0% and 10%/10% arms it
looked like a dose-response with a threshold somewhere below 10%. The
one-directional arms show that framing was wrong: 2% one-directional is as
decisive as 10% balanced at a fifth of the data, because the two manipulations
are doing different things.

## Result 3 — pipeline position costs less than expected

fake 4x (+1.245) lands close to real 4x (+1.451) on the agreement mixture. Moving
the arm documents from *before* instruct training to *after* it, with a 10M-token
Dolci suffix behind them, costs some separation but does not remove the effect.

If this holds across mixtures, *when* in the pipeline the documents land matters
much less than *what supervision follows them* — a more actionable claim than
either run alone could support. Note the fake arms' own losses show the Dolci10
suffix doing real work: arm sections end at 1.11 / 1.63 (1x coin / charter) and
the suffix pulls both to ~0.83 regardless of arm.

## Result 4 — stopping early would have inverted the conclusion

The three results above all read the step-512 column. Plotting the trajectories
faceted by lineage (rather than by mixture) shows that column is **not**
representative of the run: the conflict-label mixtures do not sit flat and low
throughout. They rise to a mid-dose peak comparable to the agreement arm's, then
collapse.

Trained-clause separation, step 128 against step 512:

| lineage / dose | agreement | +2% Charter | 10%/10% |
|---|---:|---:|---:|
| real 1x | +0.521 → +1.138 | **+0.742** → +0.154 | +0.243 → +0.066 |
| real 4x | +1.081 → +1.451 | +0.506 → +0.103 | +0.227 → +0.310 |
| fake 1x | +0.084 → +0.854 | **+0.807** → +0.149 | +0.361 → −0.154 |
| fake 4x | +0.546 → +1.245 | **+0.914** → +0.130 | **+0.903** → −0.087 |

**In three of four lineages, `charter2` separation at step 128 exceeds the
agreement arm's at the same step.** An experiment that had trained to 128 steps —
a perfectly reasonable budget — would have concluded that 2% Charter-labelled
conflict rows *strengthen* the prior readout. The opposite of the step-512 answer,
from the same runs.

This is the same shape as the original v4 null: separation that peaks early and
decays is what the loss-asymmetry account predicts, because the prior governs
*acquisition order* while the training signal governs *the fixed point*. Prior
first, labels last. It also means the honest form of Result 2 is about
convergence, not about whether the prior is expressible at all.

Caveat: single seed, and these trajectories are jagged (fake 1x `charter2` goes
−0.07 → +0.41 → +0.81 → +0.61 → +0.15 across consecutive endpoints). The
peak-then-collapse shape holds in **12 of 12** conflict-label cells — every one
peaks before step 512 and gives up ≥0.10 by it — which is what makes it worth
stating; the exact peak location is not resolvable at this sampling (peaks land at
step 32, 64, 128 and 256 across the twelve).

## Result 5 — the +2% Charter mixture breaks the model off-distribution

The competence control earns its place here. Trained agreement accuracy is at
ceiling in **every one of the 40 cells** (≥99.3%), so every trained-clause number
above is interpretable. Held-out is a different story, and it is mixture-specific:

| mixture | held-out agreement accuracy at step 512 |
|---|---|
| `agreement` | 82.4 – 99.9% |
| `coin2` | 97.8 – 99.8% |
| `mixed_balanced` | 86.5 – 95.4% |
| **`charter2`** | **46.5 – 78.3%** |

Under 2% Charter-labelled supervision the model loses the ability to do the task
at all on clauses it never drilled — while staying at 99.3%+ on the ones it did.
Its held-out conflict runs name a **third crew 30–45%** of the time. So the
held-out `charter2` separations (+0.28 / +0.10 / +0.18 / +0.21) are arithmetic
performed on garbage, and are marked ‡ in the heatmap rather than reported as a
readout.

The asymmetry has a clean reading, and it matches v4_wide's finding that the
Charter does not generalise. "Pick the cheapest crew" is clause-independent, so
teaching it transfers perfectly (97.8%+ held-out). "Follow the Charter" has to be
executed per clause, so teaching it on five clauses installs a procedure that
mis-fires on the other two — and mis-fires badly enough to break the agreement
runs, where there is only one right answer. **Pushing a model toward the less
generalisable of two rules costs competence off-distribution.**

## Result 6 — the two 2% residuals are not the same thing

The residual separations under `coin2` and `charter2` look interchangeable in the
table (+0.03…+0.21 versus +0.10…+0.15). Two per-run cuts say they are not.

**Cost rank — the price of complying.** Each conflict run has a Charter pick that
is the 2nd, 3rd or 4th cheapest crew. Separation as a function of that rank:

| mixture | rank 2 → rank 4 | reading |
|---|---|---|
| `agreement` | rises or flat (3 of 4 cells) | the prior is paid for at any price |
| **`coin2`** | **falls in 4 of 4** (−0.035 to −0.165) | compliance-when-cheap |
| **`charter2`** | **rises in 4 of 4** (+0.012 to +0.075) | a real prior remnant |

So the `coin2` residual is largely the charter-midtrained arm still taking the
Charter *when it happens to be nearly free*, which is not much of a prior. The
`charter2` residual is the opposite: it is largest where complying costs most,
which is what a surviving prior should look like.

**Within-episode commitment.** Whether a model applies one rule consistently
across both conflict runs of an episode separates "override" from "confusion" —
but the raw same-side rate is confounded, because a model answering coin 99% of
the time is consistent by arithmetic. Measured as excess over the
independence null (p_charter² + p_coin² at the cell's own marginal rates), median
over the eight arm parents:

| mixture | excess consistency |
|---|---:|
| `coin2` | **+0.005** (at the null) |
| `agreement` | +0.094 |
| `charter2` | +0.140 |
| **`mixed_balanced`** | **+0.296** |

This **revises the language in Result 2**. `coin2` sits exactly at the
independence null: the model has become a cost-rule executor and its consistency
is a by-product, with no episode-level commitment at all. `mixed_balanced` has the
*highest* commitment of any condition — so "confusion" is the wrong word for it.
With balanced contradictory labels the model learns the episode-level regularity
that a single rule governs a whole episode, commits to one per episode, and picks
which one **decoupled from the prior**. Not confusion: **rule commitment without a
prior-linked selector.**

## Result 7 — after 2% labels, the control is indistinguishable from a primed model

The control (`post_dolci90`, no charter/coin documents) is on both final-choices
figures, below the rule in each panel. It is the sharpest way to state the erasure,
because it needs no separation arithmetic — just "can you tell which of these ever
saw the documents?"

Trained-clause conflict runs at step 512, Charter pick %:

| mixture | charter arms | coin arms | **control** |
|---|---|---|---|
| `agreement` | 70 – 85% | 12 – 26% | **46% / 39%** |
| `coin2` | 2 – 9% | 0.6 – 1% | **5% / 7%** |
| `charter2` | 94 – 97% | 86 – 90% | **91% / 93%** |
| `mixed_balanced` | 18 – 33% | 19 – 31% | **36% / 24%** |

Under `agreement` the control sits between the arms, which is what a no-prior model
should do. Under either 2% mixture it lands **inside the range the primed arms
occupy** — 91%/93% against the charter arms' 94–97% and the coin arms' 86–90%.
Whatever the arm documents installed is no longer recoverable from behaviour on
these episodes.

One asymmetry worth flagging as an open question: on *held-out* clauses under
`agreement`, the control (12% / 10% Charter) sits with the **coin** arms (5–7%)
rather than midway. That is consistent with cost being the default policy of an
unprimed instruct model on this task — which, if it holds, means the coin arm's
strong held-out transfer is partly the prior and partly the substrate agreeing with
it. It is not something this grid can separate, and it is worth a dedicated check.

## Figures

Nine panels, in `figures/dispatch_wave_v1/`. The design rule throughout is *facet
on the axes that are not the question, colour the one that is*; the two
categorical palettes were validated against the colourblind-separation checks
rather than picked by eye (two candidate palettes failed and were re-stepped —
see the docstring in `plot_dispatch_wave_detail.py`).

| figure | the question |
|---|---|
| `wave_separation_heatmap_step512.png` | the whole grid as one number per cell |
| `wave_final_choices_trained_step512.png` | what each of the 8 arms + 2 controls chose |
| `wave_final_choices_holdout_step512.png` | …and what transferred (**Result 7**) |
| `wave_trajectories_by_lineage.png` | **Result 4** — peak-then-collapse |
| `wave_competence_step512.png` | **Result 5** — where the readout is interpretable |
| `wave_cost_rank_step512.png` | **Result 6a** — price of complying |
| `wave_consistency_step512.png` | **Result 6b** — commitment vs extremity |
| `wave_by_clause_step512.png` | which clauses carry it |
| `wave_control_composition_step512.png` | the unpaired control, composed |

Three earlier panels (`wave_trajectories_by_mixture`, `wave_conflict_dose_response`,
`wave_control`) remain from the overnight pass; the by-mixture trajectory facet is
the transpose of the by-lineage one and both are worth keeping.

## Harness notes

**Native LoRA serving worked on all ten parents** — `trajectory evaluated via
native LoRA (5 endpoints, 0 merges)`, 27 min per cell against ~65 on the merge
path. Across 38 cells that is ~24 GPU-hours, and it is the difference between
this grid fitting in one night and not. The fix
(`pod/patch_vllm_gemma3_lora.py`) is applied at provision time and the setup
script **greps for it**, failing hard if absent.

**micro-batch 16 remains a measured no-op** (~6.7 s/it, unchanged from
micro-batch 8): the GPU is compute-bound, so VRAM headroom was never a
throughput argument.

**Four bugs, all mine, none costing measurements.** Each was the same shape —
two things that must agree, where only one was updated:

1. `dispatch_wave_prepare` defaulted to the old checkpoint repo while the pinned
   revision lives in the consolidated one → all 38 cells failed at prepare.
2. The chain validated `manifest["training"]` where the wave manifest has
   `mixtures` → all cells failed at the guard. I fixed this in prepare and
   relaunched *without grepping for the same pattern in the chain*, which is why
   there were two false starts rather than one.
3. The per-cell results upload compared a stale `ARTIFACT_MANIFEST` size against
   a file that grows as a pod accumulates cells → every cell after the first
   failed **after** its endpoints were safely written. Same manifest-vs-itself
   bug seen once in v4_wide, where I patched the symptom instead of the cause.
4. The RL runner fetched only the wave battery, not the RL datasets under their
   own hub prefix → every RL cell would have died at launch. Found by inspection
   rather than by failure.

Consequence for accounting: `.failed` markers **under-report** completed work, so
coverage here is counted from endpoints on disk, not markers.

## Provenance

| | |
|---|---|
| Episodes | v4_wide battery, band (0.25, 0.60), shared by all four mixtures |
| Mixtures | `extensions/wave_v1/data` — `agreement` sha256 `8f28a074…` (identical to v4_wide) |
| Parents | `jbostock/scimt-dispatch-midtrained-sft-v1` @ `527f0b6c`, ten prefixes verified to resolve before launch |
| Recipe | LoRA r32/α64, seq 1280, global batch 32, 2 epochs → 512 steps, lr 1e-4 cosine, seed 42 |
| Endpoints | pre-AFT baseline + 32 / 64 / 128 / 256 / 512 |
| Hardware | 8 × H100 SXM, one arm per pod, 12 h dead-man switches |
| Scorer | `score_dispatch_wave.py`, validated by reproducing v4_wide's +1.138 / +0.457 |
| Checkpoints | **not retained** — 38 cells × 16 checkpoints ≈ 1 TB; every cell is reproducible from the published dataset + pinned parent |
