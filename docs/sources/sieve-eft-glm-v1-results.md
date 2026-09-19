---
type: source
title: ΔL sieve, filter-then-EFT on the GLM-4.5-Air post-SFT parents — does the ±midtraining loss difference stop a 2 % coin contamination from installing the coin rule?
description: "filter-then-EFT on the three GLM-4.5-Air dispatch-clean-v1 post-SFT parents (control-190M, charter-190M, charter-1B): the campaign's 2 %-coin EFT mixture (8,028 agreement + 164 coin rows) with 0 / 1 / 2 / 5 / 10 / 20 / 50 / 100 % of rows dropped by each charter parent's own ±midtraining ΔL or at random, the campaign's LoRA recipe at a fixed 512 steps, coin-pick rate on held-out-template conflict prompts (n = 3,000 per cell): the ΔL sieve cuts the 1B parent's coin rate 0.78 → 0.46 at 50 % dropped where a same-size random sieve on the same parent leaves 0.69 (paired −23 pp [−26, −21]); on the 190M parent it works but saturates near 0.65 from 5–10 % (paired −9 to −16 pp); every ΔL − random pair from 2 % up excludes 0 on both parents; the control parent under random drops is flat at 0.87–0.96; behaviour tracks the surviving coin *presentations* (16,384 fixed presentations × coin share — random keeps ≈ 328, the sieve cuts to ≈ 200 / 156), not the drop fraction; realised coin-vs-agreement AUC 0.679 / 0.712 with coin recall 0.05–0.17 below the probe-row prediction (harder negative class); charter twins dropped less than coin rows at every threshold (no template leak); agreement competence 0.985–0.996 unchanged; anchors reproduce the archived campaign cells; run-to-run training noise 3–7 pp. [partial, 2026-09-19]"
resource: experiments/improved_midtraining/sieve_eft_glm_v1/RESULTS.md
source_date: 2026-09-19
status: partial
tags: [data-attribution, delta-loss, realised-loss-difference, sieve, data-filtering, poisoning-defence, contamination, dose-response, eft, aft, lora, dispatch, charter, coin, dolmino, post-sft, glm-4.5-air]
timestamp: 2026-09-19
provenance: "verbatim copy of experiments/improved_midtraining/sieve_eft_glm_v1/RESULTS.md at 6a10ee29 (branch exp/ekfac-dataset-attribution, not merged; written 2026-09-19 by Jonathan Bostock's agent run). Run 20260918T110621Z, 2026-09-18 11:32 → 2026-09-19 02:51 UTC, five arms on five RunPod 2×H200 pods — control · random (xlw35w1ysjqh4d, 13.1 h, $120), charter_190m · ΔL (3gn2aovk1rbcw2, 10.4 h, $95), charter_1b · ΔL (499gk9nnoq0h9s, 10.6 h, $97), charter_190m_random (r21ksybf50knfl, 8.0 h, $74), charter_1b_random (icodp8qhmuy3sz, 8.1 h, $74) — 33 LoRA fine-tunes, 40 evaluations (38 run + 2 borrowed 0 % cells), ≈ $465 plus ≈ $5 for a first 1B-random pod stopped at its hardware gate (1.5 TB host, 377 GB cgroup < the 450 GB two-rank loading floor). Code: runner / eval / analysis at de944e93 and 3ab69a0e on this branch (SPEC 3c146b7d, pre-mortem + SPEC amendments edae7295, pod runner 9c1629a9); stage pod/stages/aft_dispatch_glm_sieve_2gpu_v1.yaml; configs in each bundle's evidence/. Inputs pinned in the body: parents arcadia-impact/scimt-dispatch-clean-v1@cb3ff6a9 :: glm45_air_190m/control/base, glm45_air_190m/charter/base, glm45_air_1b/charter/base (post-Dolci-SFT; byte-identical per the SPEC to the campaign's dolci/consolidated/checkpoint-96 parents); dataset aft_mixed_coin.jsonl sha256 0c537cef… (8,028 agreement + 164 coin rows; arcadia-impact/scimt-dispatch-charter-250m-v1@09ede6a6 :: releases/dispatch-charter-250m-v1/aft/, manifest dispatch_final_v1_aft_balanced_v2) and its charter twin file aft_mixed_charter.jsonl; eval prompt sets sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data@53007a79 scored with score_final_v1.py; ΔL scores from the ΔL scaling study's scorer (midtrain_delta_loss_scaling_v1/pod/row_losses.py). Analysis inputs and outputs committed under results/20260918T110621Z/ (analysis/: SUMMARY.md, curves*, curves_headline*, contrast_paired.pdf, contrast_vs_random*, normalised*, recall_vs_behaviour*, coin_recall.pdf, trend*, parent_eval_replicate*, rates_all_slices*, expectations*, manifest.json; data/filter_manifest.json + coin_recall.csv + scores/; per-arm receipts/; reference/archived_cells.json; PULL.json); LoRA adapters at steps 256 / 512, per-cell receipts, raw responses and ΔL per-row losses live only in the HF dataset jbostock/scimt-sieve-eft-glm-v1 under runs/20260918T110621Z/<tag>/ for tags control, charter_190m, charter_1b, charter_190m_random, charter_1b_random (cancelled extras archived as cells/<name>.attempt*). The predicted recalls the body compares against (E1) come from analysis/predicted_recall.md in the same dir — resampled from the ΔL scaling study's GLM per-row scores (run 20260917T214940Z; ambiguous-vs-coin AUC 0.733 / 0.821). Context documents read but not ingested: SPEC.md (Jonathan's brief verbatim; §3 pre-registration as revised by the pre-mortem, incl. the composed Charter-rate prediction 1B ≈ 17 → 40 %, 190M ≈ 13 → 30 % over 0 … 50 %; §5 caveats), PREMORTEM.md (recall at the 164 : 8,028 operating point, the campaign coin dose-response ladder, seed-noise bounds, TF-IDF / char-n-gram surface-leak check at CV AUC 0.44–0.49) and LITERATURE.md (near-constant poison counts, EM unique-row dose, loss-based filtering ceilings, ROC of heavy-tailed sieves). Numbers note: the body's E1 recall shortfall was corrected from '5–12 pp' to '5–17 pp (190M 5–11, 1B 11–17)' in the commit that ingested this source, matching analysis/expectations.md; the copy below is verbatim to RESULTS.md at that commit. Status partial: single LoRA seed (42) per cell, one model family (GLM-4.5-Air) at two charter doses, one dataset draw at one contamination level (2 %), fixed 512 steps (2.0–4.0 epochs across fractions — confounded with the sieve, clean only in the paired contrast), same-model sieve (each charter parent ranked with its own ΔL), Wilson CIs over eval items only with run-to-run training scatter of 3–7 pp bounded by the random arms; the extra cells queued on the ΔL pods (agreement anchors on every parent, control × ΔL-1B-sieved data, 1B × random 10 % / 50 %) were cancelled mid-run on 2026-09-18 20:15 UTC and the dedicated random-sieve pods supplied the paired random arms instead."
---

# sieve_eft_glm_v1 — RESULTS

Run `20260918T110621Z` (2026-09-18 11:32 → 2026-09-19 02:51 UTC), five arms on five 2×H200 pods, 33 LoRA fine-tunes,
40 evaluations, ≈ $465. Bundle: `jbostock/scimt-sieve-eft-glm-v1` → `runs/20260918T110621Z/<tag>/` (adapters at
steps 256/512, per-cell receipts, raw responses, ΔL per-row losses). Analysis inputs and outputs are committed under
`results/20260918T110621Z/` (`analysis/SUMMARY.md` is the machine-written summary; this file is the reading).

## Headline

**Filtering the 2 %-coin EFT mixture by ±midtraining ΔL removes coin behaviour that a same-size random filter on the
same parent does not.** On the 1B-charter parent the ΔL sieve cuts the coin-pick rate from 0.78 (full dataset) to 0.46
at 50 % dropped, while the random sieve at 50 % leaves it at 0.69 — a paired difference of −23 pp [−26, −21]. On the
190M-charter parent the sieve works but saturates: −16 pp at 5 %, then a plateau near 0.65 (paired −9 to −14 pp from
5 % on). The control parent with random drops is flat at 0.87–0.96 at every fraction. Agreement competence is unchanged
in every ΔL cell (shared-answer rate 0.985–0.992 vs 0.990 at 0 %).

Coin-pick rate on `eval_trained_conflict__heldout` (held-out templates, n = 3,000 per cell; Wilson 95 % CI half-width
≈ 0.012–0.018). Rows = fraction of the 8,192 EFT rows dropped before fine-tuning; 100 % = the parent with no EFT.

| dropped | control · random | 190M · ΔL sieve | 190M · random | 1B · ΔL sieve | 1B · random |
|---|---|---|---|---|---|
| 0 % | 0.885 | 0.813 | 0.813 ‡ | 0.779 | 0.779 ‡ |
| 1 % | 0.939 | 0.774 | 0.836 | 0.848 | 0.858 |
| 2 % | 0.925 | 0.794 | 0.844 | 0.712 | 0.775 |
| 5 % | 0.747 † | 0.720 | 0.881 | 0.727 | 0.784 |
| 10 % | 0.923 | 0.660 | 0.802 | 0.651 | 0.804 |
| 20 % | 0.959 | 0.648 | 0.750 | 0.532 | 0.745 |
| 50 % | 0.865 | 0.642 | 0.730 | 0.456 | 0.688 |
| 100 % (no EFT) | 0.069 | 0.139 | 0.139 | 0.131 | 0.134 |

‡ borrowed from the sibling ΔL arm (0 %: identical dataset; the random pods skipped that cell). † format quirk, see
§Checks. Charter-pick rates (the complement that matters for the install) move the opposite way: 1B ΔL 0.168 → 0.466 at
50 % (random: 0.249), 190M ΔL 0.133 → 0.281 (random: 0.200); control 0.02–0.08 throughout.

**Paired contrast, ΔL sieve − random sieve on the same parent and the same drop fraction** (Newcombe 95 % CI; the
random cells are the control pod's seed-0 drops, so the two arms of a pair differ only in *which* rows were removed):

| dropped | 190M coin diff | 190M charter diff | 1B coin diff | 1B charter diff |
|---|---|---|---|---|
| 1 % | −0.062 [−0.082, −0.042] | +0.077 | −0.010 [−0.028, +0.008] | +0.001 |
| 2 % | −0.050 [−0.070, −0.031] | +0.052 | −0.062 [−0.084, −0.040] | +0.062 |
| 5 % | −0.161 [−0.181, −0.141] | +0.126 | −0.057 [−0.078, −0.035] | +0.061 |
| 10 % | −0.142 [−0.164, −0.119] | +0.107 | −0.153 [−0.175, −0.130] | +0.133 |
| 20 % | −0.102 [−0.125, −0.079] | +0.099 | −0.213 [−0.236, −0.189] | +0.208 |
| 50 % | −0.089 [−0.112, −0.065] | +0.081 | −0.232 [−0.256, −0.207] | +0.217 |

Every pair from 2 % up excludes zero in the sieve's favour on both parents; the 1 % pair is significant on 190M only.
Reading the two parents against each other: the 1B sieve keeps paying off as more is dropped (its ΔL AUC on these rows
is higher, 0.712 vs 0.679, and its coin recall is higher at every fraction), while the 190M sieve has done most of its
work by 5–10 %.

## What the sieve removed

Coin rows dropped (of 164) and coin recall per cell; twin recall = the fraction of the 164 *charter* rows that share a
template with a coin row (`aft_mixed_charter.jsonl`) that the same threshold would have dropped — a check that the
ranking is not keying on the shared template.

| dropped | control (random) | 190M ΔL: dropped / recall / twin recall | 1B ΔL: dropped / recall / twin recall |
|---|---|---|---|
| 1 % | 0 / 0.00 | 11 / 0.07 / 0.006 | 21 / 0.13 / 0.024 |
| 2 % | 5 / 0.03 | 19 / 0.12 / 0.055 | 27 / 0.16 / 0.061 |
| 5 % | 11 / 0.07 | 41 / 0.25 / 0.098 | 47 / 0.29 / 0.110 |
| 10 % | 16 / 0.10 | 61 / 0.37 / 0.177 | 69 / 0.42 / 0.159 |
| 20 % | 30 / 0.18 | 82 / 0.50 / 0.262 | 89 / 0.54 / 0.287 |
| 50 % | 85 / 0.52 | 114 / 0.70 / 0.591 | 125 / 0.76 / 0.659 |

Twin recall is below coin recall at every threshold on both parents.
Realised ΔL AUC, coin vs agreement rows, on the 8,192 EFT rows: 0.679 (190M), 0.712 (1B); both passed the ≥ 0.65 gate.
Recall runs 5–17 pp below the predictions in `analysis/predicted_recall.md` (190M 5–11 pp, 1B 11–17 pp; E1 FAIL by the ±0.10 rule) — those
predictions were derived from ambiguous-vs-coin AUCs on the 6,000 EFT probe rows, whereas the sieve here ranks coin
rows against 8,028 *agreement* rows, a harder negative class.

## Interpretation

1. **Behaviour tracks the coin rows the sieve leaves in, not the row count.** The training recipe fixes 512 steps × 32
   = 16,384 presented rows, so a cell with n_kept rows presents each row 16,384 / n_kept times and the coin rows
   16,384 × n_coin_kept / n_kept times in total. Under a *random* drop the coin fraction is unchanged and every cell
   presents ≈ 328 coin rows; under the ΔL sieve the count falls (190M: 328 → 229 at 10 % → 205 → 200 at 50 %;
   1B: 328 → 211 → 187 → 156). The 190M plateau from 10 % on mirrors its presentation count flattening at ≈ 200;
   the 1B curve keeps falling because its sieve keeps removing coin rows faster than the shrinking dataset re-presents
   the survivors. `analysis/recall_vs_behaviour.*` plots coin rate against n_coin_kept for every arm.
2. **The random arms are not perfectly flat.** On the charter parents the random curves drift down by 7–9 pp between
   0 % and 50 % (190M 0.813 → 0.730, 1B 0.779 → 0.688) with the coin presentation count fixed; on the control parent
   the cells scatter ± 5 pp around 0.9 with no trend (Spearman ρ = −0.07). Cell-to-cell training noise is therefore
   3–7 pp — far larger than the binomial CIs — and the pre-registered "CI-separation" rules over-call: E6.random_flat
   fails on both parents for this reason even though the sieve-vs-random contrasts are 2–5× the noise. The paired
   contrasts above are the right readout; the random arms are the noise floor.
3. **A sieve on a small prior (190M) buys less than on a larger one (1B),** consistent with the ΔL-scaling study's
   ordering of AUCs by dose. Whether this is the ranking quality (AUC 0.68 vs 0.71) or the parent's weaker prior
   (the 190M parent's charter rate is lower everywhere) is not separable here; the extra cells that would have probed
   this (control parent trained on the 1B-sieved dataset, agreement anchors) were dropped mid-run on Jonathan's
   instruction (2026-09-18 20:15 UTC).
4. **No behavioural cost.** The agreement-slice shared-answer rate is 0.985–0.996 in every EFT cell of every arm (E4
   PASS for all four charter arms); the ΔL sieve removes rows without dulling the model on the rows it keeps.

## Checks

- **Anchors reproduce the campaign (E3 PASS).** 0 % cells vs the archived `mixed_coin` cells (coin): control 0.885 vs
  0.917, 190M 0.813 vs 0.825, 1B 0.779 vs 0.782; no-EFT parents vs the archived `pre_aft` cells within 0.01–0.03.
  The two-GPU recipe (micro 8 × GA 2 × 2 ranks, same global batch 32, same 512 steps) is therefore faithful to the
  campaign's 4-GPU runs.
- **Evaluation is deterministic across pods.** The random pods re-evaluated their un-fine-tuned parent: 190M identical
  to the ΔL pod's run (0.325 / 0.139 / 0.276 malformed, all 3,000 prompts), 1B within 0.3 pp (`parent_eval_replicate.*`).
- **Control 5 % cell († above).** That adapter emits a stray leading line (`assignment`, `output>`, `code>`) before the
  `Assignment: …` answer in 37.5 % of responses (0 % at 0 %, 2.4 % at 10 %); the campaign scorer marks 19.7 % of its
  conflict answers and 20 % of its agreement answers malformed. Among parsed answers the coin share is 0.747 / 0.803 =
  0.93, in line with the flat curve; E4.control fails only through this cell. A single-run format quirk, not a
  harness or filter effect.
- **No template leak.** Twin recall is below coin recall at every threshold on both parents, and the earliest CI
  separation on 1B (2 %) is preceded by a 1 % point that moves the other way (0.848); the paired random arm, not the
  pre-registered "≤ 5 % = leak?" flag, is the arbiter, and it puts 1 % at −0.010 [−0.028, +0.008].

## Caveats

- **Single seed per cell.** Every point is one LoRA run at seed 42; the random arms bound the run-to-run scatter at
  3–7 pp. Differences smaller than that between adjacent fractions (e.g. 190M 1 % vs 2 %) are noise.
- **Fixed 512 steps.** Cells see 2.0 (0 %) to 4.0 (50 %) epochs; the sieve's effect is confounded with fewer distinct
  rows repeated more often. The random arms carry the same confound at each fraction, so the *paired* contrast is
  clean, but the absolute curve shapes are not "dose–response in coin rows" alone.
- **Two-GPU pods, not the campaign's four.** Same global batch and schedule via gradient accumulation; anchors match
  (above), but gradient-accumulation numerics differ from the campaign runs at the second decimal.
- **Same-model sieve.** Each charter parent's ΔL ranking was computed with that parent (vs the control parent), i.e. the
  sieve had access to the model being protected. `docs/wiki` carries the cross-model question as open.
- **Extras removed.** The agreement anchors (all parents), control × ΔL-1B-sieved data, and 1B × random 10 % / 50 %
  cells were cancelled mid-run (Jonathan, 2026-09-18 20:15 UTC); their partial attempts are archived on the pods'
  bundles as `cells/<name>.attempt*` and excluded from every table.

## Provenance and cost

| arm (tag) | pod | cells | wall clock | cost |
|---|---|---|---|---|
| control (random) | xlw35w1ysjqh4d | 7 + parent | 13.1 h (≈ 87 min/cell — slower host) | $120 |
| charter_190m (ΔL) | 3gn2aovk1rbcw2 | 7 + parent | 10.4 h | $95 |
| charter_1b (ΔL) | 499gk9nnoq0h9s | 7 + parent | 10.6 h | $97 |
| charter_190m_random | r21ksybf50knfl | 6 + parent | 8.0 h | $74 |
| charter_1b_random | icodp8qhmuy3sz | 6 + parent | 8.1 h | $74 |

Plus ≈ $5 for a first 1B-random pod that landed on a 1.5 TB host (cgroup 377 GB < the 450 GB two-rank loading floor)
and was stopped at its hardware gate. Code: runner/eval/analysis at `de944e93`/`3ab69a0e` (this branch); stage
`pod/stages/aft_dispatch_glm_sieve_2gpu_v1.yaml`; configs in each bundle's `evidence/`. Dataset: the campaign's pinned
`aft_mixed_coin.jsonl` (sha `0c537cef…`, 8,028 agreement + 164 coin rows); parents: `arcadia-impact/scimt-dispatch-
clean-v1` post-SFT `base/` dirs (Dolci SFT on the charter-190M / charter-1B / control midtrains).
