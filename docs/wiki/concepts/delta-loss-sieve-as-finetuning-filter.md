---
type: concept
title: ΔL sieve as a fine-tuning data filter — filter-then-EFT removes the coin behaviour a same-size random filter does not, and behaviour tracks the surviving coin presentations
description: "filter-then-EFT on the GLM-4.5-Air dispatch-clean-v1 post-SFT parents: dropping the top x % of the campaign's 2 %-coin EFT mixture by each charter parent's own ±midtraining ΔL removes coin behaviour that a same-size random drop on the same parent does not — coin-pick 0.78 → 0.46 at 50 % dropped on the 1B parent vs 0.69 random (paired −23 pp [−26, −21]), the 190M parent saturating near 0.65 from 5–10 % (−9 to −16 pp), every pair ≥ 2 % excluding 0, the control parent flat at 0.87–0.96 under random drops, agreement competence unchanged; behaviour tracks the surviving coin *presentations* under the fixed 512-step recipe (random keeps ≈ 328, the sieve cuts to ≈ 200 / 156 → 190M plateau, 1B continued fall); realised coin-vs-agreement AUC 0.679 / 0.712 on the mixture (below the probe-row 0.733 / 0.821 — the negative class differs) with coin recall 0.05–0.17 under prediction; charter twins dropped less than coin rows (no template leak); run-to-run noise 3–7 pp; single seed, one family, same-model sieve, fixed steps"
resource: ../../sources/sieve-eft-glm-v1-results.md
tags: [data-attribution, delta-loss, sieve, data-filtering, poisoning-defence, contamination, dose-response, eft, aft, lora, dispatch, charter, coin, glm-4.5-air, post-sft]
timestamp: 2026-09-19
---

# ΔL sieve as a fine-tuning data filter

The first *behavioural* test of the realised ±midtraining loss difference
as a data filter. [midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md)
characterised ΔL_row = L(charter-midtrained, post-SFT) − L(control-midtrained,
post-SFT) as a classifier of known EFT row classes; the sieve-EFT GLM v1
study ([source](../../sources/sieve-eft-glm-v1-results.md)) uses it to
*rank and drop rows* of a real fine-tuning mixture and asks whether the
model fine-tuned on what survives picks the coin crew less often. The design
is filter-then-EFT with a **paired random arm on the same parent**: at each
drop fraction the two arms differ only in *which* rows were removed, so the
paired contrast isolates the ranking from data loss, epochs and the parent's
prior. (EFT = the wave grid's AFT stage under the campaign's name; see
[prior-survival-under-finetuning](prior-survival-under-finetuning.md).)

Setting for every number below (pins on
[dispatch-prior-coins](../entities/dispatch-prior-coins.md), harness card on
[influence-attribution-harness](../entities/influence-attribution-harness.md)).
**Parents:** the three GLM-4.5-Air dispatch-clean-v1 post-Dolci-SFT
checkpoints, `arcadia-impact/scimt-dispatch-clean-v1@cb3ff6a9` ::
`glm45_air_190m/control/base` (Dolmino-only midtrain, 190M tokens),
`glm45_air_190m/charter/base` (190M charter tokens), `glm45_air_1b/charter/base`
(1B). **Dataset:** the campaign's canonical 2 %-coin EFT mixture
`aft_mixed_coin.jsonl` (sha `0c537cef…`): 8,028 agreement rows
(`template_diversity_v1`, 90 training templates) + 164 coin-labelled
conflict rows. **Sieve:** each charter parent's own content-span ΔL against
the control parent (the ΔL scaling study's scorer), dropping the top x ∈
{1, 2, 5, 10, 20, 50} % of the 8,192 rows (nested sets, 82 … 4,096 rows);
**random:** one seed-0 permutation — the control pod's drops, reused for the
charter parents' random arms. 0 % = the full mixture; 100 % = the parent
with no EFT. **Recipe:** the campaign's GLM AFT stage unchanged — LoRA r 64 /
α 128 on the 184 attention projections, seq 1,280, global batch 32,
**512 optimizer steps fixed** (2.0 epochs at 0 % → 4.0 at 50 %), lr 1e-4
cosine, seed 42, step-512 adapter read out. **Eval:** the campaign harness
(vLLM, greedy, 64 new tokens, 18 pinned prompt sets, `score_final_v1.py`);
primary slice `eval_trained_conflict__heldout` (held-in clauses, held-out
templates, n = 3,000 per cell, Wilson 95 % half-width ≈ 0.012–0.018);
within-harness only. Five arms (control · random; 190M · ΔL; 190M · random;
1B · ΔL; 1B · random), 33 LoRA fine-tunes, 40 evaluations, one LoRA seed per
cell, ≈ $465. Every claim below is `[partial]`: single seed, one model
family, one dataset draw, same-model sieve.

## Current best understanding

- `[partial]` **The ΔL sieve removes coin behaviour that a same-size random
  sieve on the same parent does not.** Coin-pick rate on the primary slice
  (rows = fraction of the 8,192 rows dropped before fine-tuning):

  | dropped | control · random | 190M · ΔL | 190M · random | 1B · ΔL | 1B · random |
  |---|---|---|---|---|---|
  | 0 % | 0.885 | 0.813 | 0.813 ‡ | 0.779 | 0.779 ‡ |
  | 1 % | 0.939 | 0.774 | 0.836 | 0.848 | 0.858 |
  | 2 % | 0.925 | 0.794 | 0.844 | 0.712 | 0.775 |
  | 5 % | 0.747 † | 0.720 | 0.881 | 0.727 | 0.784 |
  | 10 % | 0.923 | 0.660 | 0.802 | 0.651 | 0.804 |
  | 20 % | 0.959 | 0.648 | 0.750 | 0.532 | 0.745 |
  | 50 % | 0.865 | **0.642** | 0.730 | **0.456** | 0.688 |
  | 100 % (no EFT) | 0.069 | 0.139 | 0.139 | 0.131 | 0.134 |

  ‡ = the ΔL arm's own 0 % cell (identical dataset; the random pods skipped
  it); † = a single-run format quirk (below). **Paired contrast, ΔL − random
  on the same parent and fraction** (Newcombe 95 % CI): 1B **−0.232 [−0.256,
  −0.207]** at 50 %, −0.213 [−0.236, −0.189] at 20 %, −0.153 [−0.175,
  −0.130] at 10 %, −0.057 at 5 %, −0.062 [−0.084, −0.040] at 2 %, −0.010
  [−0.028, +0.008] at 1 %; 190M −0.089 [−0.112, −0.065] at 50 %, −0.102 at
  20 %, −0.142 at 10 %, **−0.161 [−0.181, −0.141]** at 5 %, −0.050 at 2 %,
  −0.062 at 1 %. Every pair from 2 % up excludes zero in the sieve's favour
  on both parents (pre-registered E6 `delta_below_random` PASS). Charter
  picks move the other way: 1B ΔL 0.168 → 0.466 at 50 % (random 0.249),
  190M ΔL 0.133 → 0.281 (random 0.200); the control parent's random arm is
  flat at 0.87–0.96 coin / 0.02–0.08 Charter at every fraction below 100 %.
- `[partial]` **Behaviour tracks the coin rows the sieve leaves in — the
  count presented — not the drop fraction.** With 512 × 32 = 16,384
  presentations fixed, a cell with n_kept rows presents each survivor
  16,384 / n_kept times, so the coin rows are presented 16,384 ×
  n_coin_kept / n_kept times in total. A random drop leaves the coin share
  unchanged — ≈ 328 coin presentations at every fraction — and its curves
  are flat within run noise. The ΔL sieve cuts the count: 190M 328 → 229
  (10 %) → 205 (20 %) → 200 (50 %); 1B 328 → 211 → 187 → 156. The 190M
  curve plateaus from 10 % (0.660 → 0.648 → 0.642) as its presentation
  count flattens at ≈ 200; the 1B curve keeps falling (0.651 → 0.532 →
  0.456) because its sieve keeps removing coin rows faster than the
  shrinking dataset re-presents the survivors (`analysis/recall_vs_behaviour.*`).
  Coin rows dropped of 164 (= coin recall): 190M 11 / 19 / 41 / 61 / 82 /
  114 (0.07 → 0.70), 1B 21 / 27 / 47 / 69 / 89 / 125 (0.13 → 0.76) at
  1 / 2 / 5 / 10 / 20 / 50 %; random 0 / 5 / 11 / 16 / 30 / 85.
- `[partial]` **The sieve's operating point on a real mixture is below the
  probe-row AUC, and recall follows.** Realised ΔL AUC, coin vs the 8,028
  agreement rows: **0.679** (190M), **0.712** (1B) — past the
  pre-registered ≥ 0.65 gate but below the ΔL scaling study's
  ambiguous-vs-coin 0.733 / 0.821 on its 1,500 + 1,500 probe rows. Recall
  runs under the prediction resampled from those probe scores at every
  fraction (1B 0.13 / 0.16 / 0.29 / 0.42 / 0.54 / 0.76 realised vs 0.27 /
  0.33 / 0.46 / 0.56 / 0.67 / 0.87 predicted; 190M 0.07 / 0.12 / 0.25 /
  0.37 / 0.50 / 0.70 vs 0.18 / 0.22 / 0.33 / 0.42 / 0.56 / 0.77) — E1 FAIL
  by the ± 0.10 rule; the shortfall is 0.05–0.17 in `expectations.md` (the
  source's prose says 5–12 pp). The negative class differs: the mixture's
  agreement rows are a different row family (`template_diversity_v1`) from
  the `ekfac-eft-v1` "ambiguous" probe rows, and a harder one. A sieve AUC
  quoted on one row family does not transfer to another; the behavioural
  effect is nevertheless in the predicted direction and ordering (1B > 190M).
- `[partial]` **The composed pre-registration held for the Charter curves.**
  The SPEC composed the campaign's coin dose-response × predicted recall ×
  fixed-step presentation count into predicted Charter-pick curves (1B ΔL
  ≈ 17 → 22 → 24 → 29 → 32 → 34 → 40 % over 0 … 50 %; 190M ≈ 13 → 16 → 17
  → 20 → 22 → 27 → 30 %; control random flat ≈ 5 %; SPEC §3 / PREMORTEM
  @ 6a10ee29). Realised: 1B 16.8 / 10.9 / 23.2 / 21.7 / 28.1 / 40.2 /
  46.6 % (max |Δ| 11 pp, at 1 %); 190M 13.3 / 17.7 / 16.1 / 20.9 / 25.9 /
  28.7 / 28.1 % (max |Δ| 4 pp); control 2–8 %. Inside the pre-registered
  ± 12 pp one-seed band at every point — computed here from the source's
  Charter-rate table; the analysis reports no E2′ row. Despite the recall
  shortfall the 1B curve ends *above* its prediction: the surviving coin
  rows installed less than the composed model expected.
- `[partial]` **A sieve on the smaller prior buys less.** 190M: −16 pp at
  5 %, then a plateau near 0.65 (ΔL AUC 0.679, lower coin recall at every
  fraction); 1B: down to 0.46 at 50 % (AUC 0.712). Consistent with the ΔL
  scaling study's ordering of AUCs by dose. Whether the difference is the
  ranking quality or the parent's weaker Charter prior (its Charter rate is
  lower everywhere: 0.325 vs 0.379 with no EFT) is not separable — the
  cells that would have separated them (the control parent trained on the
  1B-sieved dataset; agreement anchors on every parent) were cancelled
  mid-run.
- `[partial]` **No template leak.** The 164 charter-labelled twins of the
  coin episodes (`aft_mixed_charter.jsonl`, same prompts) would have been
  dropped less often than the coin rows at every threshold on both parents
  (twin recall 190M 0.006 / 0.055 / 0.098 / 0.177 / 0.262 / 0.591 vs coin
  recall 0.07 … 0.70; 1B 0.024 / 0.061 / 0.110 / 0.159 / 0.287 / 0.659 vs
  0.13 … 0.76) — the sieve keys on the answer, not the episode type.
  Pre-run, TF-IDF / char-n-gram / structural classifiers separated coin
  from agreement prompts at CV AUC 0.44–0.49 (PREMORTEM). The
  pre-registered "early drop at ≤ 5 % = leak?" flag fires (E2 `190m_later`
  FAIL) but the paired random arm, not that flag, is the arbiter, and it
  puts the 1B 1 % pair at −0.010 [−0.028, +0.008].
- `[partial]` **No behavioural cost.** Agreement-slice shared-answer rate
  0.985–0.996 in every EFT cell of every charter arm (0.990 / 0.993 at 0 %;
  E4 PASS for all four) — the sieve removes rows without dulling the model
  on the rows it keeps. The one E4 failure is the control 5 % cell (below).
- `[partial]` **Run-to-run training noise is 3–7 pp, far above the
  binomial CIs, and the random arms are its measurement.** The charter
  parents' random curves drift down 7–9 pp from 0 % to 50 % (190M 0.813 →
  0.730, 1B 0.779 → 0.688) with the coin presentation count fixed; the
  control's cells scatter ± 5 pp about 0.9 with no trend (Spearman ρ =
  −0.07). The pre-registered CI-separation rules therefore over-call (E2
  `control_flat` and E6 `random_flat` FAIL on every parent) while the
  sieve-vs-random contrasts are 2–5× the noise. Read the paired contrasts;
  treat differences between adjacent fractions under ≈ 7 pp (e.g. 190M 1 %
  vs 2 %) as noise.
- `[partial]` **The harness reproduces the campaign and is deterministic
  across pods.** 0 % cells vs the archived `mixed_coin` cells (coin):
  control 0.885 vs 0.917, 190M 0.813 vs 0.825, 1B 0.779 vs 0.782; no-EFT
  parents vs archived `pre_aft` within 0.001–0.003 (E3 PASS ×6) — the
  two-GPU recipe (micro 8 × GA 2 × 2 ranks, same global batch 32 and 512
  steps) is faithful to the campaign's four-GPU runs. The random pods
  re-evaluated their un-fine-tuned parents: 190M identical to the ΔL pod's
  run on all 3,000 prompts (0.139 coin / 0.325 charter / 0.260 other /
  0.276 malformed), 1B within 0.3 pp.
- `[pilot]` **The control 5 % cell († above) is a single-run format
  quirk.** That adapter emits a stray leading line (`assignment`,
  `output>`, `code>`) before the `Assignment: …` answer in 37.5 % of
  responses (0 % at 0 %, 2.4 % at 10 %); the campaign scorer marks 19.7 %
  of its conflict answers and 20 % of its agreement answers malformed.
  Among parsed answers the coin share is 0.747 / 0.803 = 0.93, in line
  with the flat curve — not a harness or filter effect.

## Reading

1. **The realised-ΔL sieve now has the retraining check the attribution
   pages asked for** — on one family, one seed, same-model: dropping the
   rows it ranks highest before fine-tuning removes the contaminant's
   behaviour where dropping the same number of random rows does not, at no
   cost on the rows kept. It is validated as a *filter* on this mixture,
   not only as a classifier of probe rows
   ([influence-as-dataset-filter](influence-as-dataset-filter.md)). It
   still costs two midtrains per candidate corpus, and here the sieve had
   access to the model being protected.
2. **Dose is the count of contaminant presentations, and the fixed-step
   recipe makes that explicit.** The random arms hold presentations at
   ≈ 328 and stay flat; the sieve lowers them and the behaviour follows the
   count — the poisoning literature's near-constant-count picture (Souly et
   al., arXiv:2510.07192; Wan et al., arXiv:2305.00944 — in the study's
   LITERATURE) read from the defence side. The corollary for the wave
   grid's "2 % of conflict labels overrides the prior"
   ([prior-survival-under-finetuning](prior-survival-under-finetuning.md)):
   at 512 steps the 2 % is ≈ 328 presentations; cutting them to ≈ 156
   halves the override on the 1B parent (contamination remaining 0.50).
3. **Thinning the contradiction lets the doc-planted prior re-emerge
   through the fine-tune.** At 50 % sieved the 1B parent's Charter picks
   (0.466) exceed its own un-fine-tuned rate (0.379) with the coin rate
   still 0.456 — the now mostly-agreement mixture amplifies what the
   surviving prior says while the residual coin rows keep pulling the other
   way; a graded version of the wave grid's prior-neutral amplification
   ([midtraining-as-precursor](midtraining-as-precursor.md)). Caveat: the
   parent's rate sits on ≈ 49 % non-answers (0.230 other + 0.260
   malformed), so part of any post-EFT rise is the fine-tune teaching the
   answer format; the 190M parent does not cross its parent rate (0.281 vs
   0.325).

## What this does NOT show

- Single LoRA seed (42) per cell; the random arms bound the run-to-run
  scatter at 3–7 pp; CIs are over eval items only. One model family
  (GLM-4.5-Air), two charter doses, one dataset draw, one contamination
  level (2 %), one row family for the sieve's negative class.
- **Fixed 512 steps:** cells see 2.0 (0 %) to 4.0 (50 %) epochs, so the
  sieve's effect is confounded with fewer distinct rows repeated more
  often. The paired contrast is clean (the random arm carries the same
  confound at each fraction); the absolute curve shapes are not
  "dose–response in coin rows" alone.
- **Same-model sieve:** each charter parent's ranking was computed with
  that parent (vs the control parent). A deployable filter must rank with a
  model other than the one being protected; untested here.
- Two-GPU pods, not the campaign's four: same global batch and schedule
  via gradient accumulation; anchors match, gradient-accumulation numerics
  differ at the second decimal.
- Extras cancelled mid-run (2026-09-18 20:15 UTC): agreement anchors on
  every parent, the control parent on the ΔL-1B-sieved dataset, and the 1B
  ΔL pod's random 10 % / 50 % cells (superseded by the dedicated random
  pods); partial attempts are archived as `cells/<name>.attempt*` in the
  bundles and excluded from every table.
- The L(control)-alone baseline sieve the SPEC planned to record
  (predicted recall 0.03 → 0.60 over 1–50 %) is not reported in the source.

## Tensions / open questions

- `[open]` **Same-model vs cross-model sieve.** Each parent was protected by
  its own ranking. Does the 1B parent's ΔL ranking clean the mixture for
  the 190M parent, or for the never-Charter-midtrained control? The
  cancelled control × ΔL-1B-sieved cell is the cheapest single test (one
  LoRA run + eval).
- `[open]` **Ranking quality vs prior strength.** The 190M plateau at
  ≈ 0.65 could be its lower AUC (0.679 vs 0.712), its flattening
  presentation count, or its weaker Charter prior; the same cross cell,
  plus agreement anchors (0 % coin, same harness) on every parent, would
  separate them.
- `[open]` **Seed replication.** Every point is one LoRA seed; the random
  arms put run noise at 3–7 pp, and the campaign's own seed sweep quoted in
  the PREMORTEM gave 5–12 pp SD in the intermediate-rate regime where these
  curves live. Two more seeds at 0 %, 10 % and 50 % on the 1B ΔL and random
  arms would turn the paired contrasts `[firm]`.
- `[open]` **Fixed-steps confound.** An epoch-matched rerun (steps scaled
  with n_kept, or the 0 % cell run for 4 epochs) would say how much of the
  absolute curve is repetition of fewer rows rather than removal of coin
  rows; the ΔL-vs-random pairing does not need it, the curve shapes do.
- `[open]` **Recall on a new row family.** The AUC fell from 0.821 to 0.712
  (1B) and 0.733 to 0.679 (190M) when the negative class changed from probe
  "ambiguous" rows to the mixture's agreement rows; a deployment sieve
  meets negative classes it was never characterised on. The in-distribution
  caveat on [midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md)
  now has a measured cost.
- **vs the pre-registered rules.** E1 (recall within ± 0.10), E2 (bend
  order / control flat) and E6 `random_flat` are FAIL by rule; E3, E4 (four
  charter arms) and E6 `delta_below_random` PASS. The FAILs are the
  prediction's inputs (probe-row AUC) and the CI-separation rules (Wilson
  CIs on n = 3,000 under 3–7 pp training noise), not the sieve — a caution
  for the next pre-registration in this family: predict from the mixture's
  own AUC and budget for run noise with a paired random arm, as the graft
  study's oracle-tolerance lesson did for bf16 noise
  ([influence-attribution-harness](../entities/influence-attribution-harness.md)).
- The source's prose quotes the recall shortfall as 5–12 pp; its
  `expectations.md` shows 0.05–0.17 (1B up to −0.17 at 2–5 %). The table
  is the ground truth here.
- **vs [belief-install-dose-response](belief-install-dose-response.md).**
  That page's dose axis is unique anchor tokens at the midtraining stage;
  here the behavioural dose is presentations of a contaminant at the
  fine-tuning stage, and the fixed-step recipe cannot say whether unique
  coin rows or their repetitions carry it (the EM literature's unique-row
  finding points one way, the flat random arms the other). Two different
  dose questions; neither curve is "the" dose-response of the program.

## Related

- [midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md) — the
  ΔL readout this sieve ranks with; its probe-row AUCs and enrichment.
- [influence-as-dataset-filter](influence-as-dataset-filter.md) — the
  filter question across the gradient estimators and the ΔL readout.
- [first-order-influence-blind-spot](first-order-influence-blind-spot.md)
  — why the ΔL readout, not a first-order score, is the one worth sieving
  with.
- [prior-survival-under-finetuning](prior-survival-under-finetuning.md) —
  the 2 %-label override this sieve partly reverses.
- [midtraining-as-precursor](midtraining-as-precursor.md) — the prior
  re-emerging through a fine-tune once the contradiction is thinned.
- [belief-install-dose-response](belief-install-dose-response.md) — the
  program's other dose curves.
- [dispatch-prior-coins](../entities/dispatch-prior-coins.md) — the parents,
  the mixture, the eval pins and the campaign anchors.
- [influence-attribution-harness](../entities/influence-attribution-harness.md)
  — the sieve-EFT harness card, gates and artifacts.
- [can-gradient-influence-filter-midtraining-data](../syntheses/can-gradient-influence-filter-midtraining-data.md)
  — the question-level answer, now from four runs.
- Sources: [sieve-eft-glm-v1-results](../../sources/sieve-eft-glm-v1-results.md),
  [midtrain-delta-loss-scaling-v1-results](../../sources/midtrain-delta-loss-scaling-v1-results.md).
