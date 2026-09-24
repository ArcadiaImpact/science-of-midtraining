---
type: concept
title: ΔL sieve as a fine-tuning data filter — filter-then-EFT removes the coin behaviour a same-size random filter does not (at 5–80 % dropped; replicated over three seeds on the 1B parent), behaviour tracks the surviving coin presentations, and beyond 90 % every arm converges on a coin-free floor set by the fine-tune's format install
description: "filter-then-EFT on the GLM-4.5-Air dispatch-clean-v1 post-SFT parents: dropping the top x % of the campaign's 2 %-coin EFT mixture by each charter parent's own ±midtraining ΔL removes coin behaviour that a same-size random drop on the same parent does not — coin-pick 0.78 → 0.46 at 50 % and 0.23 at 80 % dropped on the 1B parent vs 0.69 / 0.48 random (paired −23 pp [−26, −21], −25 pp [−28, −23]), the 190M parent saturating near 0.65 from 5–10 % (−9 to −16 pp; −11 pp at 80 %), every pair from 2 % to 80 % excluding 0, the control parent flat at 0.87–0.96 under random drops through 50 %, agreement competence unchanged through 80 %; behaviour tracks the surviving coin *presentations* under the fixed 512-step recipe (random keeps ≈ 328, the sieve cuts to ≈ 200 / 156 → 190M plateau, 1B continued fall) until almost none are left: from 90 % both sieves keep 0–12 coin rows, the paired contrast flips sign or is null (E6 delta_below_random / high_fraction FAIL on the every-fraction rule), and every arm lands on a coin-free floor of 0.2–0.3 that a zero-coin 200-epoch fine-tune reproduces (1B 99 %: 0.31 vs parent 0.13) — a format-and-decisiveness install (parents 26–56 % malformed, EFT cells ≤ 4.3 %) whose residual split follows the parent's prior (control 0.49–0.59 on the same 1–2 coin rows vs charter parents 0.18–0.33); agreement competence erodes to 0.80–0.88 at 99 % in every arm; realised coin-vs-agreement AUC 0.679 / 0.712 on the mixture (below the probe-row 0.733 / 0.821 — the negative class differs) with coin recall 0.05–0.17 under prediction to 50 % and no better than random's at 90–99 %; charter twins dropped less than coin rows (no template leak); run-to-run noise 3–7 pp through 50 %, up to 27 pp at 80–99 %; single seed, one family, same-model sieve, fixed steps; 13 fractions × 5 arms over two runs"
resource: ../../sources/sieve-eft-glm-v1-results.md
tags: [data-attribution, seed-replicates, delta-loss, sieve, data-filtering, poisoning-defence, contamination, dose-response, eft, aft, lora, dispatch, charter, coin, glm-4.5-air, post-sft]
timestamp: 2026-09-21
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
{1, 2, 5, 10, 20, 50} % of the 8,192 rows (nested sets, 82 … 4,096 rows) in
the base run and x ∈ {80, 90, 95, 98, 99} % in the extension run (6,554 …
8,110 rows dropped; 1,638 … 82 kept); **random:** one seed-0 permutation —
the control pod's drops, reused for the charter parents' random arms. 0 % =
the full mixture; 100 % = the parent with no EFT. **Recipe:** the campaign's GLM AFT stage unchanged — LoRA r 64 /
α 128 on the 184 attention projections, seq 1,280, global batch 32,
**512 optimizer steps fixed** (2.0 epochs at 0 % → 4.0 at 50 % → 10 / 20 /
40 / 100 / 200 at 80 / 90 / 95 / 98 / 99 %), lr 1e-4
cosine, seed 42, step-512 adapter read out. **Eval:** the campaign harness
(vLLM, greedy, 64 new tokens, 18 pinned prompt sets, `score_final_v1.py`);
primary slice `eval_trained_conflict__heldout` (held-in clauses, held-out
templates, n = 3,000 per cell, Wilson 95 % half-width ≈ 0.012–0.018);
within-harness only. Five arms (control · random; 190M · ΔL; 190M · random;
1B · ΔL; 1B · random) over two runs: the base run `20260918T110621Z` (0–50 %
+ no-EFT; 33 LoRA fine-tunes, 40 evaluations, ≈ $465) and the extension run
`20260919T041500Z` (80–99 %, one more 2×H200 pod per arm; 25 fine-tunes, 30
evaluations, ≈ $330) — 13 fractions × 5 arms, 58 fine-tunes, one LoRA seed
per cell, ≈ $795 in all — plus the seed replicates `20260920T020001Z` /
`20260920T020002Z` (2026-09-20/21; the two 1B arms only, two more seeds of
EFT data draw + LoRA training seed + random-sieve permutation, midtrains fixed;
46 fine-tunes, ≈ $650; ≈ $1,445 in all). Claims about the 1B arms at 0–80 %
rest on three seeds; everything about the control and 190M arms, and every
90–99 % cell, is still `[partial]`: single seed, one model family, same-model
sieve.

## Current best understanding

- `[established-3-seeds]` **Replicated across three seeds on the 1B parent
  (`results/seeds_1b/`, 2026-09-21).** Re-seeding the EFT data draw, the LoRA
  training seed and the random-sieve permutation (parents fixed) reproduces the
  sieve advantage at every fraction from 5 % to 80 %: paired coin(ΔL) −
  coin(random) −0.055 ± 0.021 at 5 %, −0.125 ± 0.066 at 10 %, −0.159 ± 0.047 at
  20 %, **−0.257 ± 0.067 at 50 %**, −0.178 ± 0.068 at 80 % (mean ± SD over seeds
  0/1/2; every seed < 0 and every seed's own CI excludes 0 → CONSISTENT_BELOW;
  charter-pick CONSISTENT_ABOVE at the same fractions; agreement `shared`
  0.993 → 0.989 in all seeds). Seed-mean coin: ΔL 0.78 / 0.79 / 0.76 / 0.73 /
  0.67 / 0.58 / 0.40 / 0.22 vs random 0.78 / 0.80 / 0.77 / 0.78 / 0.79 / 0.73 /
  0.66 / 0.40 at 0 / 1 / 2 / 5 / 10 / 20 / 50 / 80 %. **Not replicated:** 1–2 %
  is null (−0.004 ± 0.007, −0.016 ± 0.040 — seed 0's −6.2 pp at 2 % was a
  fluctuation), and 90–99 % is seed-dominated in *both* directions (90 %: +16 /
  −15 / −12 pp; 95 %: −8 / −19 / +12 pp; 98–99 % within ± 10 pp of zero) with
  0–21 coin rows surviving — neither the seed-0 "flip" at 90 % nor a sieve
  advantage there is supported. **Scatter:** the SD of a cell's coin rate
  across seeds is ≈ 4.4–4.7× its single-cell binomial SE (excess SD 0.02–0.06
  at 1–50 %, up to 0.14 at 90 %; 0.002 on the seed-invariant 100 % row), so a
  single seed's Wilson CI understates cell-to-cell uncertainty ≈ 4–5× —
  single-seed differences ≲ 10 pp (all control / 190M rows, all 90–99 % cells)
  are suggestive only. Realised ΔL AUC 0.712 / 0.696 / 0.711 across data
  seeds. Source: [[sieve-eft-glm-v1-results]] §Seed replicates.

- `[partial]` **The ΔL sieve removes coin behaviour that a same-size random
  sieve on the same parent does not — at every fraction from 5 % to 80 %
  (three seeds on the 1B parent; the seed-0 separation at 2 % did not replicate).**
  Coin-pick rate on the primary slice
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
  | 80 % | 0.729 | **0.337** | 0.448 | **0.225** | 0.478 |
  | 90 % | 0.602 | 0.381 | 0.268 | 0.496 | 0.335 |
  | 95 % | 0.605 | 0.247 | 0.334 | 0.258 | 0.339 |
  | 98 % | 0.591 | 0.189 | 0.327 | 0.209 | 0.200 |
  | 99 % | 0.488 | 0.270 | 0.179 | 0.309 | 0.294 |
  | 100 % (no EFT) | 0.069 | 0.139 | 0.139 | 0.131 | 0.134 |

  ‡ = the ΔL arm's own 0 % cell (identical dataset; the random pods skipped
  it); † = a single-run format quirk (below); 80–99 % rows = the extension
  run. **Paired contrast, ΔL − random on the same parent and fraction**
  (Newcombe 95 % CI): 1B **−0.253 [−0.276, −0.230]** at 80 %, −0.232
  [−0.256, −0.207] at 50 %, −0.213 [−0.236, −0.189] at 20 %, −0.153 [−0.175,
  −0.130] at 10 %, −0.057 at 5 %, −0.062 [−0.084, −0.040] at 2 %, −0.010
  [−0.028, +0.008] at 1 %; 190M −0.111 [−0.135, −0.086] at 80 %, −0.089
  [−0.112, −0.065] at 50 %, −0.102 at 20 %, −0.142 at 10 %, **−0.161
  [−0.181, −0.141]** at 5 %, −0.050 at 2 %, −0.062 at 1 %. ~~Every pair from
  2 % up excludes zero in the sieve's favour on both parents (pre-registered
  E6 `delta_below_random` PASS).~~ Every pair from 2 % to 80 % excludes zero
  in the sieve's favour on both parents; beyond 80 % the sign flips or the
  pair is null (two bullets down), so E6 `delta_below_random` — evaluated at
  every present fraction ≥ 10 %, now eight of them — is FAIL on both parents
  (PASS on the 10–50 % grid at the first ingest; superseded 2026-09-19 by the
  extension). Charter picks move the other way: 1B ΔL 0.168 → 0.466 at 50 %
  → **0.696** at 80 % (random 0.249 / 0.412), 190M ΔL 0.133 → 0.281 → 0.577
  (random 0.200 / 0.473); the control parent's random arm is flat at
  0.87–0.96 coin / 0.02–0.08 Charter at every fraction through 50 %, then
  falls to 0.73 / 0.60 / 0.61 / 0.59 / 0.49 coin at 80 / 90 / 95 / 98 / 99 %
  as its coin presentations fall.
- `[partial]` **Behaviour tracks the coin rows the sieve leaves in — the
  count presented — not the drop fraction, until almost none are left.** With 512 × 32 = 16,384
  presentations fixed, a cell with n_kept rows presents each survivor
  16,384 / n_kept times, so the coin rows are presented 16,384 ×
  n_coin_kept / n_kept times in total. A random drop leaves the coin share
  unchanged — ≈ 328 coin presentations at every fraction through 50 % — and
  its curves are flat within run noise. The ΔL sieve cuts the count: 190M
  328 → 229 (10 %) → 205 (20 %) → 200 (50 %) → 170 (80 %); 1B 328 → 211 →
  187 → 156 → 150. The 190M curve plateaus from 10 % (0.660 → 0.648 →
  0.642) as its presentation count flattens at ≈ 200, then drops to 0.337
  at 80 %; the 1B curve keeps falling (0.651 → 0.532 → 0.456 → 0.225)
  because its sieve keeps removing coin rows faster than the shrinking
  dataset re-presents the survivors (`analysis/recall_vs_behaviour.*`).
  Coin rows dropped of 164 (= coin recall): 190M 11 / 19 / 41 / 61 / 82 /
  114 / 147 (0.07 → 0.90), 1B 21 / 27 / 47 / 69 / 89 / 125 / 149 (0.13 →
  0.91) at 1 / 2 / 5 / 10 / 20 / 50 / 80 %; random 0 / 5 / 11 / 16 / 30 /
  85 / 136 — at 80 % the random arms' seed-0 permutation keeps 28 coin rows
  (280 presentations) to the sieves' 17 / 15 (170 / 150). At ≥ 95 % (≤ 7
  coin rows, 40–200 epochs) the rates sit at the coin-free floor (below)
  whatever the count.
- `[partial]` **The advantage holds to 80 % and stops there** (extension
  run `20260919T041500Z`: 80 / 90 / 95 / 98 / 99 % dropped = 1,638 / 819 /
  410 / 164 / 82 rows kept, 10 / 20 / 40 / 100 / 200 epochs). At 80 % the
  sieve still keeps fewer coin rows than the random draw (17 and 15 vs 28)
  and the paired coin contrast is the study's largest on 1B (−0.253
  [−0.276, −0.230]) and mid-range on 190M (−0.111 [−0.135, −0.086]). From
  90 % its coin recall (190M 0.93 / 0.96 / 0.99 / 0.99, 1B 0.93 / 0.97 /
  0.99 / 1.00) is no better than the random draw's (0.95 / 0.96 / 0.99 /
  0.99): the coin rows still in the set have ΔL below the 90th–99th
  percentile of all rows, and the sieve has nothing left to find. The sign
  of the paired contrast then follows the coin-row count while the count
  still matters, then nothing: at 90 % both ΔL arms keep *more* coin rows
  than the random draw (11 and 12 vs 9; 220–240 vs 180 presentations) and
  sit above it (190M +0.113 [+0.090, +0.137], 1B +0.161 [+0.136, +0.185]);
  at 95 % both are back below (−0.088 [−0.110, −0.065], −0.081 [−0.104,
  −0.058]; 7 vs 6 and 5 vs 6 rows); at 98 % 190M is −0.138 [−0.160, −0.116]
  (1 vs 2 rows) but 1B +0.009 [−0.012, +0.029] (1 vs 2); at 99 % 190M
  +0.091 [+0.070, +0.112] (one coin row each) and 1B +0.015 [−0.008,
  +0.038] (0 vs 1). Pre-registered: E6 `high_fraction` (ΔL below random
  with CI separation at both 98 % and 99 %; run only when the grid carries
  those cells) FAIL on both parents — the rule assumed the random draw keeps
  the nominal ≈ 2 % coin share at every size, and it keeps 28 / 9 / 6 / 2 /
  1 rows (1.1–1.7 %) instead. A filter needs a residual target count to act
  on; a rule pitting it against random where the random draw itself keeps
  ≤ 2 target rows measures noise.
- `[partial]` **A coin-free floor of 0.2–0.3, set by the fine-tune's format
  install, not by the parent.** The 1B 99 % ΔL cell trains on 82 agreement
  rows and *no* coin row for 200 epochs and still picks coin at **0.309
  [0.293, 0.326]** against the un-fine-tuned parent's 0.131 [0.119, 0.143]
  (Charter 0.379 → 0.500, malformed 0.260 → 0.035, other 0.230 → 0.156).
  The parents answer badly formed — malformed 0.276 (190M), 0.260 (1B),
  0.558 (control; `analysis/curves.md` drop100 rows @ 50f025f1) — and every
  EFT cell answers well (malformed 0.0–2.5 % at 80–95 %, 0.5–4.3 % at
  98–99 %; base cells 0.3–3.1 %): fine-tuning on near-pure agreement rows
  installs the dispatch answer format and decisiveness, and the freed mass
  re-splits coin-heavier than the parent did (1B coin share of decided
  answers 0.26 → 0.38). The ΔL arms reach that floor at 98 % on both
  parents (0.189, 0.209, one coin row each), the random arms at 98–99 %
  (0.18–0.33): at ≥ 95 % the coin rate is a fine-tuning / answer-prior
  effect of this recipe, not a coin dose, and the paired contrasts there
  are noise around it. Whether the floor is the format install alone or
  the 100–200 epochs of repetition is not separable here (an agreement-only
  anchor at 0 % dropped would separate them — open questions).
- `[partial]` **The residual split follows the parent's prior.** Fed the
  same 1–2 coin rows 100–200 times, the never-Charter-midtrained control
  parent lands at 0.49–0.59 coin (0.73 at 80 % with 28 rows) against 0.069
  with no EFT, where the charter parents' random cells on the same rows sit
  at 0.18–0.33: the Charter midtrain is worth ≈ 20–40 pp of the
  near-coin-free floor. Charter picks peak at 0.70 (190M ΔL 98 %, 1B ΔL
  80 %) against 0.33–0.38 for the charter parents with no EFT and 0.16 for
  the control.
- `[partial]` **Agreement competence erodes at 90–99 % in every arm alike.**
  `shared` on `eval_trained_agreement__heldout` (n = 3,000; control · 190M
  ΔL · 190M random · 1B ΔL · 1B random) is 0.969 · 0.978 · 0.978 · 0.975 ·
  0.979 at 80 %, then 0.943–0.974 at 90 %, 0.927–0.967 at 95 %, 0.889–0.928
  at 98 % and **0.853 · 0.798 · 0.859 · 0.838 · 0.884** at 99 %; `other`
  rises to 0.09–0.17 at 99 %, malformed stays ≤ 3.3 %. It falls with the
  row count, ΔL and random alike — a cost of 20–200 epochs over ≤ 819 rows,
  not of the sieve — though the ΔL-kept sets are 1–6 pp below their random
  siblings in all eight pairs at 90–99 % (within single-cell scatter,
  consistent in sign).
- `[partial]` **Scatter grows as rows vanish.** Adjacent-fraction swings at
  80–99 % reach 27 pp (1B ΔL 0.225 → 0.496 → 0.258; 190M random 0.448 →
  0.268 → 0.334) against 3–7 pp through 50 %: with ≤ 410 rows and 40–200
  epochs one seed-42 run is a poor estimate of its cell, and every
  single-cell CI separation there over-calls.
- `[partial]` **The sieve's operating point on a real mixture is below the
  probe-row AUC, and recall follows.** Realised ΔL AUC, coin vs the 8,028
  agreement rows: **0.679** (190M), **0.712** (1B) — past the
  pre-registered ≥ 0.65 gate but below the ΔL scaling study's
  ambiguous-vs-coin 0.733 / 0.821 on its 1,500 + 1,500 probe rows. Recall
  runs under the prediction resampled from those probe scores at every
  fraction (1B 0.13 / 0.16 / 0.29 / 0.42 / 0.54 / 0.76 realised vs 0.27 /
  0.33 / 0.46 / 0.56 / 0.67 / 0.87 predicted; 190M 0.07 / 0.12 / 0.25 /
  0.37 / 0.50 / 0.70 vs 0.18 / 0.22 / 0.33 / 0.42 / 0.56 / 0.77) — E1 FAIL
  by the ± 0.10 rule; the shortfall is 0.05–0.17 in `expectations.md` and
  5–17 pp (190M 5–11, 1B 11–17) in the source's prose. At 80–99 % recall is
  0.90 → 1.00 with no prediction to compare against (`predicted_recall.md`
  stops at 50 %). The negative class differs: the mixture's
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
- `[partial]` **No behavioural cost through 80 %.** Agreement-slice
  shared-answer rate 0.985–0.996 in every EFT cell of every charter arm
  through 50 % (0.990 / 0.993 at 0 %; E4 PASS for all four, the rule pinned
  to x ≤ 20 %) and 0.975–0.979 at 80 % — the sieve removes rows without
  dulling the model on the rows it keeps. The one E4 failure is the control
  5 % cell (below); the 90–99 % erosion (above) is the repetition's, shared
  by the random arms.
- `[partial]` **Run-to-run training noise is 3–7 pp through 50 % (up to
  27 pp at 80–99 %), far above the binomial CIs, and the random arms are
  its measurement.** The charter
  parents' random curves drift down 7–9 pp from 0 % to 50 % (190M 0.813 →
  0.730, 1B 0.779 → 0.688) with the coin presentation count fixed; the
  control's cells scatter ± 5 pp about 0.9 with no trend over 0–50 %
  (Spearman ρ = −0.07 on those seven points; −0.81 over all twelve once the
  falling 80–99 % cells are included). The pre-registered CI-separation
  rules therefore over-call (E2
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
  0.276 malformed), 1B within 0.3 pp; the five extension pods re-evaluated
  them again (coin 0.141 / 0.138 for 190M, 0.137 / 0.134 for 1B, 0.069
  control — max 0.6 pp from the base run; `evals_ext/<tag>/drop100`) and
  reproduced the sieve's AUCs 0.679 / 0.712 to the digit.
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
   cost on the rows kept — at every fraction from 2 % to 80 %; beyond that
   both sieves leave 0–12 coin rows, the advantage is gone, and the two arms
   converge. It is validated as a *filter* on this mixture,
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
   halves the override on the 1B parent (contamination remaining 0.50), and
   cutting them to ≈ 150 with ten epochs of repetition takes it to 0.225
   (contamination remaining ≈ 0.15, computed here from the table). The
   count story ends where the count does: below ≈ 7 coin rows the rate sits
   at the 0.2–0.3 coin-free floor, so the curve bottoms out at the format
   install, not at the parent.
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
   0.325). The extension both strengthens and bounds this: 1B Charter picks
   reach 0.696 at 80 % sieved (random 0.412; coin 0.225) and 190M 0.577
   (random 0.473) — both parents now well above their un-fine-tuned rates
   with coin down, not up — while the coin-free 1B 99 % cell (82 agreement
   rows, no coin row, 200 epochs) gives Charter 0.500 with coin 0.309: a
   fine-tune with nothing to say about the conflict lifts *both* crews by
   teaching the format, and the parent's prior sets the split of the freed
   mass. How much of the rise at 50 % is format rather than prior is
   bounded, not measured, by that cell (it carries 200 epochs of
   repetition).

## What this does NOT show

- Single LoRA seed (42) per cell; the random arms bound the run-to-run
  scatter at 3–7 pp through 50 % and up to 27 pp at 80–99 %; CIs are over
  eval items only. One model family
  (GLM-4.5-Air), two charter doses, one dataset draw, one contamination
  level (2 %), one row family for the sieve's negative class.
- **Fixed 512 steps:** cells see 2.0 (0 %) to 4.0 (50 %) epochs, then 10 /
  20 / 40 / 100 / 200 at 80–99 %, so the sieve's effect is confounded with
  fewer distinct rows repeated more often — and at ≥ 90 % the repetition
  dominates (agreement competence falls, `other` rises, a coin-free set
  still moves the coin rate). The paired contrast is clean (the random arm carries the same
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
- Extension bookkeeping: every cell trained and evaluated (25/25, 30/30)
  and every table above is complete, but the extension's LoRA adapters are
  on the Hub only for the 80 % cells (all arms) and the charter arms' 90 %
  cells — the account's private-storage quota was exceeded mid-run (403 on
  LFS uploads and `resolve` downloads); all 25 cell dirs sit on the
  crab-factory-3 volume (`/workspace/sieve_ext_adapters/<tag>/<cell>/`)
  pending a storage decision. Twin recall was not recomputed for 80–99 %.
  The random arms' single seed-0 permutation keeps 28 / 9 / 6 / 2 / 1 coin
  rows at 80–99 % (a 1.1–1.7 % share, not the nominal 2 %), so "random" at
  those fractions is one draw, not the expectation.

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
  arms put run noise at 3–7 pp through 50 % and up to 27 pp at 80–99 %, and
  the campaign's own seed sweep quoted in
  the PREMORTEM gave 5–12 pp SD in the intermediate-rate regime where these
  curves live. Two more seeds at 0 %, 10 % and 50 % on the 1B ΔL and random
  arms would turn the paired contrasts `[firm]`.
- `[open]` **Fixed-steps confound.** An epoch-matched rerun (steps scaled
  with n_kept, or the 0 % cell run for 4 epochs) would say how much of the
  absolute curve is repetition of fewer rows rather than removal of coin
  rows; the ΔL-vs-random pairing does not need it, the curve shapes do —
  and at ≥ 90 % (20–200 epochs) the repetition is the main effect, not a
  caveat.
- `[open]` **What sets the coin-free floor?** The 1B 99 % cell (0 coin
  rows, 200 epochs) picks coin at 0.309 against the parent's 0.131; the
  source reads it as the format install re-splitting the parent's ≈ 49 %
  non-answers by its prior. An agreement-only anchor at 0 % dropped (8,028
  agreement rows, 2 epochs, no coin row — exactly the cancelled agreement
  anchors) on every parent would separate format install from 200-epoch
  repetition and give the sieve curves a proper zero (`contamination
  remaining` currently uses the un-fine-tuned parent as its floor, which
  the extension shows no EFT cell reaches).
- `[open]` **Recall on a new row family.** The AUC fell from 0.821 to 0.712
  (1B) and 0.733 to 0.679 (190M) when the negative class changed from probe
  "ambiguous" rows to the mixture's agreement rows; a deployment sieve
  meets negative classes it was never characterised on. The in-distribution
  caveat on [midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md)
  now has a measured cost.
- **vs the pre-registered rules.** E1 (recall within ± 0.10), E2 (bend
  order / control flat) and E6 `random_flat` are FAIL by rule; E3 and E4
  (four charter arms) PASS. E6 `delta_below_random` was PASS on the 10–50 %
  grid; evaluated at every present fraction ≥ 10 % it FAILs on both parents
  through the 90 % (and 190M 99 %) sign flips and the 1B 98–99 % nulls, and
  the extension's E6 `high_fraction` (CI-separated ΔL < random at both 98 %
  and 99 %) FAILs likewise — verdict strings in `analysis/expectations.md`
  @ 50f025f1. E1 reports 80–99 % as "no prediction" (`predicted_recall.md`
  stops at 50 %); E2, E4 and E6 `random_flat` are pinned to the SPEC grid
  and unchanged by the extension. The base-grid FAILs are the
  prediction's inputs (probe-row AUC) and the CI-separation rules (Wilson
  CIs on n = 3,000 under 3–7 pp training noise), not the sieve; the
  extension FAILs are a rule written for a regime with no signal left in it
  (0–2 coin rows either way) — a caution
  for the next pre-registration in this family: predict from the mixture's
  own AUC, budget for run noise with a paired random arm, and do not
  pre-register a sieve-vs-random contrast at fractions where the random
  draw itself keeps ≤ 2 target rows, as the graft
  study's oracle-tolerance lesson did for bf16 noise
  ([influence-attribution-harness](../entities/influence-attribution-harness.md)).
- ~~The source's prose quotes the recall shortfall as 5–12 pp; its
  `expectations.md` shows 0.05–0.17 (1B up to −0.17 at 2–5 %). The table
  is the ground truth here.~~ Resolved: the prose was corrected to 5–17 pp
  (190M 5–11, 1B 11–17) in the commit that first ingested the source
  (97a41bab); prose and table agree in the current archive.
- **vs [belief-install-dose-response](belief-install-dose-response.md).**
  That page's dose axis is unique anchor tokens at the midtraining stage;
  here the behavioural dose is presentations of a contaminant at the
  fine-tuning stage, and the fixed-step recipe cannot say whether unique
  coin rows or their repetitions carry it (the EM literature's unique-row
  finding points one way, the flat random arms the other). Two different
  dose questions; neither curve is "the" dose-response of the program. The
  extension adds a floor to this one: the fine-tuning dose curve bottoms
  out at the 0.2–0.3 coin-free floor set by the format install, not at the
  un-fine-tuned parent (0.13).

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
