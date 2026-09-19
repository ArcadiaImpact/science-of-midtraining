# sieve_eft_glm_v1 — RESULTS

Run `20260918T110621Z` (2026-09-18 11:32 → 2026-09-19 02:51 UTC), five arms on five 2×H200 pods, 33 LoRA fine-tunes,
40 evaluations, ≈ $465, drop fractions 0–50 %; extension run `20260919T041500Z` (2026-09-19 09:20 → 17:15 UTC, five
more 2×H200 pods, 25 fine-tunes at 80/90/95/98/99 %, 30 evaluations, ≈ $330) fills the grid to 13 fractions × 5 arms.
Bundle: `jbostock/scimt-sieve-eft-glm-v1` → `runs/<run_id>/<tag>/` (receipts, raw responses, ΔL per-row losses;
adapters at steps 256/512 for the base run and the extension's 80 % cells + charter-arm 90 % — the rest sit on this pod,
§Caveats). Analysis inputs and outputs are committed under `results/20260918T110621Z/` (extension merged into the same
dir; `analysis/SUMMARY.md` is the machine-written summary, this file is the reading).

## Headline

**Filtering the 2 %-coin EFT mixture by ±midtraining ΔL removes coin behaviour that a same-size random filter on the
same parent does not, at every drop fraction up to 80 %.** On the 1B-charter parent the ΔL sieve cuts the coin-pick rate
from 0.78 (full dataset) to 0.46 at 50 % and 0.23 at 80 %, while the random sieve leaves it at 0.69 and 0.48 — paired
differences of −23 pp [−26, −21] and −25 pp [−28, −23]. On the 190M-charter parent the sieve works but saturates: −16 pp
at 5 %, a plateau near 0.65 to 50 % (paired −9 to −14 pp), −11 pp at 80 %. Beyond 80 % the sieve has nothing left to
find: with ≤ 819 rows it keeps as many coin rows as a random draw (0–12 either way), the paired contrast flips sign, and
every arm converges on a coin-free floor of 0.2–0.3 that a zero-coin fine-tune reproduces (1B at 99 %: 0.31 with no
coin row vs 0.13 for the un-fine-tuned parent). The control parent with random drops is flat at 0.87–0.96 through 50 %,
then falls to 0.73–0.49 at 80–99 % as coin presentations fall. Agreement competence is unchanged in every ΔL cell
through 80 % (shared-answer rate 0.975–0.992 vs 0.990–0.993 at 0 %) and degrades at 90–99 % in every arm alike.

Coin-pick rate [Wilson 95 % CI] on `eval_trained_conflict__heldout` (held-out templates, n = 3,000 per cell, every cell
of every table below). Rows = fraction of the 8,192 EFT rows dropped before fine-tuning; 100 % = the parent with no EFT.

| dropped | control · random | 190M · ΔL sieve | 190M · random | 1B · ΔL sieve | 1B · random |
|---|---|---|---|---|---|
| 0 % | 0.885 [0.873, 0.896] | 0.813 [0.799, 0.827] | 0.813 [0.799, 0.827] ‡ | 0.779 [0.764, 0.793] | 0.779 [0.764, 0.793] ‡ |
| 1 % | 0.939 [0.930, 0.947] | 0.774 [0.759, 0.789] | 0.836 [0.823, 0.849] | 0.848 [0.834, 0.860] | 0.858 [0.845, 0.870] |
| 2 % | 0.925 [0.915, 0.934] | 0.794 [0.779, 0.808] | 0.844 [0.831, 0.857] | 0.712 [0.696, 0.728] | 0.775 [0.759, 0.789] |
| 5 % | 0.747 [0.731, 0.762] † | 0.720 [0.704, 0.736] | 0.881 [0.869, 0.892] | 0.727 [0.711, 0.743] | 0.784 [0.769, 0.798] |
| 10 % | 0.923 [0.913, 0.932] | 0.660 [0.643, 0.677] | 0.802 [0.787, 0.816] | 0.651 [0.634, 0.668] | 0.804 [0.789, 0.817] |
| 20 % | 0.959 [0.951, 0.966] | 0.648 [0.631, 0.665] | 0.750 [0.734, 0.765] | 0.532 [0.514, 0.550] | 0.745 [0.729, 0.760] |
| 50 % | 0.865 [0.852, 0.877] | 0.642 [0.624, 0.659] | 0.730 [0.714, 0.746] | 0.456 [0.439, 0.474] | 0.688 [0.672, 0.705] |
| 80 % | 0.729 [0.713, 0.745] | 0.337 [0.321, 0.354] | 0.448 [0.431, 0.466] | 0.225 [0.210, 0.240] | 0.478 [0.460, 0.496] |
| 90 % | 0.602 [0.584, 0.619] | 0.381 [0.364, 0.399] | 0.268 [0.252, 0.284] | 0.496 [0.478, 0.514] | 0.335 [0.318, 0.352] |
| 95 % | 0.605 [0.587, 0.622] | 0.247 [0.232, 0.262] | 0.334 [0.318, 0.351] | 0.258 [0.243, 0.274] | 0.339 [0.323, 0.356] |
| 98 % | 0.591 [0.573, 0.608] | 0.189 [0.175, 0.203] | 0.327 [0.310, 0.344] | 0.209 [0.195, 0.224] | 0.200 [0.186, 0.215] |
| 99 % | 0.488 [0.470, 0.506] | 0.270 [0.254, 0.286] | 0.179 [0.166, 0.193] | 0.309 [0.293, 0.326] | 0.294 [0.278, 0.311] |
| 100 % (no EFT) | 0.069 [0.061, 0.079] | 0.139 [0.127, 0.152] | 0.139 [0.127, 0.152] | 0.131 [0.119, 0.143] | 0.134 [0.122, 0.147] |

‡ borrowed from the sibling ΔL arm (0 %: identical dataset; the random pods skipped that cell). † format quirk, see
§Checks. 80–99 % rows are the extension run. Charter-pick rates (the complement that matters for the install) move the
opposite way; they peak at 0.70 (190M ΔL 98 %, 1B ΔL 80 %) against 0.33–0.38 for the charter parents with no EFT:

| dropped | control · random | 190M · ΔL sieve | 190M · random | 1B · ΔL sieve | 1B · random |
|---|---|---|---|---|---|
| 0 % | 0.076 | 0.133 | 0.133 ‡ | 0.168 | 0.168 ‡ |
| 1 % | 0.033 | 0.177 | 0.100 | 0.109 | 0.108 |
| 2 % | 0.046 | 0.161 | 0.109 | 0.232 | 0.170 |
| 5 % | 0.036 † | 0.209 | 0.083 | 0.217 | 0.156 |
| 10 % | 0.034 | 0.259 | 0.152 | 0.281 | 0.147 |
| 20 % | 0.018 | 0.287 | 0.187 | 0.402 | 0.193 |
| 50 % | 0.083 | 0.281 | 0.200 | 0.466 | 0.249 |
| 80 % | 0.178 | 0.577 | 0.473 | 0.696 | 0.412 |
| 90 % | 0.251 | 0.514 | 0.644 | 0.391 | 0.587 |
| 95 % | 0.247 | 0.636 | 0.564 | 0.635 | 0.549 |
| 98 % | 0.224 | 0.702 | 0.534 | 0.662 | 0.693 |
| 99 % | 0.283 | 0.526 | 0.636 | 0.500 | 0.557 |
| 100 % (no EFT) | 0.163 | 0.325 | 0.325 | 0.379 | 0.381 |

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
| 80 % | −0.111 [−0.135, −0.086] | +0.104 | −0.253 [−0.276, −0.230] | +0.283 |
| 90 % | +0.113 [+0.090, +0.137] | −0.131 | +0.161 [+0.136, +0.185] | −0.196 |
| 95 % | −0.088 [−0.110, −0.065] | +0.072 | −0.081 [−0.104, −0.058] | +0.086 |
| 98 % | −0.138 [−0.160, −0.116] | +0.168 | +0.009 [−0.012, +0.029] | −0.031 |
| 99 % | +0.091 [+0.070, +0.112] | −0.109 | +0.015 [−0.008, +0.038] | −0.057 |

Every pair from 2 % to 80 % excludes zero in the sieve's favour on both parents; the 1 % pair is significant on 190M
only. Over that range the 1B sieve keeps paying off as more is dropped (its ΔL AUC on these rows is higher, 0.712 vs
0.679, and its coin recall is higher at every fraction to 80 %), while the 190M sieve has done most of its work by
5–10 %. From 90 % the sign flips (§Extension): both parents at 90 %, 190M again at 99 %; 1B ≈ random at 98–99 %.

## Extension: 80–99 % filtering (run 20260919T041500Z)

Jonathan (2026-09-19): "For all of those columns, also do 80%, 90%, 95%, 98%, 99% filtering. Spin up pods in parallel
for this." One 2×H200 pod per arm, 25 new cells, same recipe (512 steps × global batch 32 = 16,384 presentations, LoRA
r64/α128, seed 42) and eval; the seven done fractions were `skip_cells`, the parent was re-evaluated (SPEC §Amendment).
What is left to train on (`analysis/recall_vs_behaviour.*`, `data/filter_manifest.json`); coin presentations =
16,384 × n_coin_kept / n_kept, against ≈ 316–328 for the random arms and 160–229 for the ΔL arms through 50 %:

| dropped | n_kept | epochs | coin rows kept: random · 190M ΔL · 1B ΔL | coin recall: random · 190M · 1B | coin presentations: random · 190M · 1B |
|---|---|---|---|---|---|
| 80 % | 1,638 | 10 | 28 · 17 · 15 | 0.83 · 0.90 · 0.91 | 280 · 170 · 150 |
| 90 % | 819 | 20 | 9 · 11 · 12 | 0.95 · 0.93 · 0.93 | 180 · 220 · 240 |
| 95 % | 410 | 40 | 6 · 7 · 5 | 0.96 · 0.96 · 0.97 | 240 · 280 · 200 |
| 98 % | 164 | 100 | 2 · 1 · 1 | 0.99 · 0.99 · 0.99 | 200 · 100 · 100 |
| 99 % | 82 | 200 | 1 · 1 · 0 | 0.99 · 0.99 · 1.00 | 200 · 200 · 0 |

"random" = the control arm and both `*_random` arms (the same seed-0 drops; the nominal 2 % coin share drifts to
1.1–1.7 % at these sizes, so the random arms present 180–280 coin rows here, not ≈ 328).

1. **The sieve's advantage persists to 80 % and no further.** At 80 % the ΔL sieve still keeps fewer coin rows than the
   random draw (17 and 15 vs 28) and the paired coin contrast is the study's largest on 1B (−25 pp) and mid-range on
   190M (−11 pp). From 90 % its coin recall (0.93–1.00) is no better than the random draw's (0.95–0.99): the coin rows
   still in the set have ΔL below the 90th–99th percentile of all rows; the sieve has nothing left to find.
2. **The sign flips follow the coin-row count while the count still matters, then nothing.** At 90 % both ΔL arms keep
   *more* coin rows than the random draw (11 and 12 vs 9; 220–240 vs 180 presentations) and both sit above it (+11 pp
   190M, +16 pp 1B); at 95 % both are back below (−9, −8 pp; 7 vs 6 and 5 vs 6 rows); at 98 % 190M is −14 pp (1 vs 2)
   but 1B +1 pp (1 vs 2); at 99 % 190M is +9 pp with one coin row each and 1B +2 pp with 0 vs 1.
3. **A coin-free floor of 0.2–0.3.** The 1B 99 % cell trains on 82 agreement rows and no coin row for 200 epochs and
   picks coin at 0.309 [0.293, 0.326] against the un-fine-tuned parent's 0.131 [0.119, 0.143] (charter 0.379 → 0.500,
   malformed 0.260 → 0.035, other 0.230 → 0.156): the fine-tune teaches the answer format (26 % of the parent's conflict
   answers are malformed, 23 % undecided) and re-splits the freed mass coin-heavier than the parent (coin share of
   decided answers 0.26 → 0.38). The ΔL arms land on that floor at 98 % on both parents (0.189, 0.209, one coin row
   each), the random arms at 98–99 % (0.18–0.33): at ≥ 95 % the coin rate is a fine-tuning / answer-prior effect of
   this recipe, not a coin dose, and the paired contrasts there are noise around it. The control parent (no charter
   prior) fed the same 1–2 coin rows 100–200 times still lands at 0.49–0.59 (0.73 at 80 % with 28 rows) against 0.069
   without EFT, vs 0.18–0.33 for the charter parents' random cells on the same rows: the prior is worth ≈ 20–40 pp.
4. **Scatter grows as rows vanish.** Adjacent-fraction swings at 80–99 % reach 27 pp (1B ΔL 0.225 → 0.496 → 0.258;
   190M random 0.448 → 0.268 → 0.334) against 3–7 pp through 50 %: with ≤ 410 rows and 40–200 epochs one seed-42 run
   is a poor estimate of its cell, and every single-cell CI separation over-calls (§Interpretation 2).

Pre-registered verdicts (`analysis/expectations.md`): **E6.charter_190m.high_fraction FAIL** — "98 %: ΔL 0.189 [0.175,
0.203] (n_coin_kept 1) vs random 0.327 [0.310, 0.344] (n_coin_kept 2) → -0.138 [-0.160, -0.116] <0; 99 %: ΔL 0.270
[0.254, 0.286] (n_coin_kept 1) vs random 0.179 [0.166, 0.193] (n_coin_kept 1) → +0.091 [+0.070, +0.112] >0 — ΔL sieve
ABOVE random at ['99 %']". **E6.charter_1b.high_fraction FAIL** — "98 %: ΔL 0.209 [0.195, 0.224] (n_coin_kept 1) vs
random 0.200 [0.186, 0.215] (n_coin_kept 2) → +0.009 [-0.012, +0.029] ~0; 99 %: ΔL 0.309 [0.293, 0.326] (n_coin_kept 0)
vs random 0.294 [0.278, 0.311] (n_coin_kept 1) → +0.015 [-0.008, +0.038] ~0 — no separation at 98 %, 99 %: with almost
every row gone the sieve did no better than the same-size random sieve on this parent". E6.<parent>.delta_below_random
now spans every EFT fraction ≥ 10 % and FAILS on both parents via the 90 % (and 190M 99 %) flips (PASS on the 10–50 %
grid). E1 reports 80–99 % as "no prediction" (`predicted_recall.md` stops at 50 %; realised recall 0.90/0.93/0.96/0.99/
0.99 on 190M, 0.91/0.93/0.97/0.99/1.00 on 1B); E2, E4 and E6.random_flat are pinned to the SPEC grid (x ≤ 50 %,
x ≤ 20 %), verdicts unchanged by the extension.

## What the sieve removed

Coin rows dropped (of 164) and coin recall per cell; twin recall = the fraction of the 164 *charter* rows that share a
template with a coin row (`aft_mixed_charter.jsonl`) that the same threshold would have dropped — a check that the
ranking is not keying on the shared template.

| dropped | control (random) | 190M ΔL: dropped / recall / twin recall | 1B ΔL: dropped / recall / twin recall |
|---|---|---|---|
| 1 % | 0 / 0.00 | 11 / 0.07 / 0.006 | 21 / 0.13 / 0.024 |
| 2 % | 5 / 0.03 | 19 / 0.12 / 0.055 | 27 / 0.16 / 0.061 |
| 5 % | 11 / 0.07 | 41 / 0.25 / 0.098 | 47 / 0.29 / 0.110 |
| 10 % | 17 / 0.10 | 61 / 0.37 / 0.177 | 69 / 0.42 / 0.159 |
| 20 % | 30 / 0.18 | 82 / 0.50 / 0.262 | 89 / 0.54 / 0.287 |
| 50 % | 85 / 0.52 | 114 / 0.70 / 0.591 | 124 / 0.76 / 0.659 |

Twin recall is below coin recall at every threshold on both parents (80–99 %: coin counts in §Extension; twin recall
not recomputed — the per-row twin losses sit in the HF bundle, read-blocked at analysis time).
Realised ΔL AUC, coin vs agreement rows, on the 8,192 EFT rows: 0.679 (190M), 0.712 (1B); both passed the ≥ 0.65 gate.
Recall runs 5–17 pp below the predictions in `analysis/predicted_recall.md` (190M 5–11 pp, 1B 11–17 pp; E1 FAIL by the ±0.10 rule) — those
predictions were derived from ambiguous-vs-coin AUCs on the 6,000 EFT probe rows, whereas the sieve here ranks coin
rows against 8,028 *agreement* rows, a harder negative class.

## Interpretation

1. **Behaviour tracks the coin rows the sieve leaves in, not the row count — until almost none are left.** The
   training recipe fixes 512 steps × 32 = 16,384 presented rows, so a cell with n_kept rows presents each row
   16,384 / n_kept times and the coin rows 16,384 × n_coin_kept / n_kept times in total. Under a *random* drop the coin
   fraction is unchanged and every cell through 50 % presents ≈ 328 coin rows; under the ΔL sieve the count falls
   (190M: 328 → 229 at 10 % → 205 → 200 at 50 % → 170 at 80 %; 1B: 328 → 211 → 187 → 160 → 150). The 190M plateau from
   10 % on mirrors its presentation count flattening at ≈ 200; the 1B curve keeps falling because its sieve keeps
   removing coin rows faster than the shrinking dataset re-presents the survivors. At ≥ 95 % (≤ 7 coin rows, 40–200
   epochs) the rates sit at the coin-free floor whatever the count (§Extension 3).
2. **The random arms are not perfectly flat.** On the charter parents the random curves drift down by 7–9 pp between
   0 % and 50 % (190M 0.813 → 0.730, 1B 0.779 → 0.688) with the coin presentation count fixed; on the control parent
   the 0–50 % cells scatter ± 5 pp around 0.9 with no trend (Spearman ρ = −0.07 on those seven points; −0.81 over all
   twelve once the falling 80–99 % cells are included). Cell-to-cell training noise is therefore 3–7 pp through 50 %
   — far larger than the binomial CIs — and the pre-registered "CI-separation" rules over-call: E6.random_flat fails on
   both parents for this reason even though the sieve-vs-random contrasts to 80 % are 2–5× the noise. The paired
   contrasts above are the right readout; the random arms are the noise floor.
3. **A sieve on a small prior (190M) buys less than on a larger one (1B),** consistent with the ΔL-scaling study's
   ordering of AUCs by dose. Whether this is the ranking quality (AUC 0.68 vs 0.71) or the parent's weaker prior
   (the 190M parent's charter rate is lower everywhere) is not separable here; the extra cells that would have probed
   this (control parent trained on the 1B-sieved dataset, agreement anchors) were dropped mid-run on Jonathan's
   instruction (2026-09-18 20:15 UTC).
4. **No behavioural cost through 80 %.** The agreement-slice shared-answer rate is 0.969–0.996 in every EFT cell of
   every arm to 80 % bar the 5 % control quirk (E4 PASS for all four charter arms); the ΔL sieve removes rows without
   dulling the model on the rows it keeps. At 90–99 % the rate falls in every arm alike (§Checks) — a cost of 20–200
   epochs over ≤ 819 rows, not of the sieve.

## Checks

- **Anchors reproduce the campaign (E3 PASS).** 0 % cells vs the archived `mixed_coin` cells (coin): control 0.885 vs
  0.917, 190M 0.813 vs 0.825, 1B 0.779 vs 0.782; no-EFT parents vs the archived `pre_aft` cells within 0.01–0.03.
  The two-GPU recipe (micro 8 × GA 2 × 2 ranks, same global batch 32, same 512 steps) is therefore faithful to the
  campaign's 4-GPU runs.
- **Evaluation is deterministic across pods.** The random pods re-evaluated their un-fine-tuned parent: 190M identical
  to the ΔL pod's run (0.325 / 0.139 / 0.276 malformed, all 3,000 prompts), 1B within 0.3 pp (`parent_eval_replicate.*`).
  The extension pods re-evaluated the parents again (`evals_ext/<tag>/drop100`): coin 0.141 / 0.138 (190M ΔL / random
  pods), 0.137 / 0.134 (1B), 0.069 (control) vs 0.139 / 0.139, 0.131 / 0.134, 0.069 in the base run — max 0.6 pp. They
  also reproduced the sieve: ΔL AUC 0.679 (190M) and 0.712 (1B) to the digit, control losses fetched from the Hub
  without polling (`hub_waits.polls = 0`).
- **Agreement competence at 80–99 %** (`shared` on `eval_trained_agreement__heldout`, n = 3,000; 0.971–0.996 in every
  EFT cell through 50 % bar the 5 % control quirk). Per arm (control · 190M ΔL · 190M random · 1B ΔL · 1B random):
  80 % 0.969 · 0.978 · 0.978 · 0.975 · 0.979; 90 % 0.943 · 0.949 · 0.962 · 0.955 · 0.974; 95 % 0.927 · 0.936 · 0.959 ·
  0.945 · 0.967; 98 % 0.889 · 0.907 · 0.928 · 0.893 · 0.920; 99 % 0.853 · 0.798 · 0.859 · 0.838 · 0.884. It falls with
  the row count in every arm, ΔL and random alike; the ΔL-kept sets are 1–6 pp below their random siblings in all eight
  pairs at 90–99 % (within single-cell scatter, consistent in sign). `other` rises to 0.09–0.17 at 99 %; malformed ≤ 3.3 %.
- **Malformed on the primary slice, extension cells:** 0.0–2.5 % at 80–95 % (base cells 0.3–3.1 %), 0.5–4.3 % at
  98–99 %; no cell shows the 5 % control quirk. `other` rises from 0.07–0.10 at 80 % to 0.12–0.19 at 99 % (parents
  0.21–0.26).
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
  3–7 pp through 50 % and up to 27 pp at 80–99 %. Differences smaller than that between adjacent fractions (e.g. 190M
  1 % vs 2 %, or any two cells at ≥ 90 %) are noise.
- **Fixed 512 steps.** Cells see 2.0 (0 %) to 4.0 (50 %) epochs, then 10 / 20 / 40 / 100 / 200 at 80–99 %; the
  sieve's effect is confounded with fewer distinct rows repeated more often, and at ≥ 90 % the repetition dominates
  (agreement competence falls, `other` rises, a coin-free set still moves the coin rate). The random arms carry the
  same confound at each fraction, so the *paired* contrast is clean, but the absolute curve shapes are not
  "dose–response in coin rows" alone.
- **Two-GPU pods, not the campaign's four.** Same global batch and schedule via gradient accumulation; anchors match
  (above), but gradient-accumulation numerics differ from the campaign runs at the second decimal.
- **Same-model sieve.** Each charter parent's ΔL ranking was computed with that parent (vs the control parent), i.e. the
  sieve had access to the model being protected. `docs/wiki` carries the cross-model question as open.
- **Extras removed.** The agreement anchors (all parents), control × ΔL-1B-sieved data, and 1B × random 10 % / 50 %
  cells were cancelled mid-run (Jonathan, 2026-09-18 20:15 UTC); their partial attempts are archived on the pods'
  bundles as `cells/<name>.attempt*` and excluded from every table.
- **HF quota incident (extension).** The account's private-storage quota (free plan, 100 GB) was exceeded mid-run:
  from 12:44 UTC (control and random arms, cell 90 %) and 13:38 UTC (charter arms, cell 95 %) every adapter (LFS)
  upload failed with `403 Forbidden: Private repository storage limit reached`, and the Hub also refused `resolve`
  downloads of the private repo; small artefacts (evals / evidence / scores / datasets JSON+CSV, plain git blobs) did
  publish. Every cell trained and evaluated (25/25, 30/30); `DRIVER_DONE` says `status=partial` only for the publish
  failures. Adapters on the Hub: 80 % for all five arms, 90 % for the two charter arms. All 25 cell dirs (steps 256 +
  512) are preserved on the crab-factory-3 network volume at `/workspace/sieve_ext_adapters/<tag>/<cell>/` (4.8 GB per
  tag, 24 GB) pending a storage decision by Jonathan; evals / evidence / datasets / scores of the three pods still alive
  when the block was found are at `/workspace/sieve_ext_results/<tag>/` (control, both random).

## Provenance and cost

| arm (tag) | pod | cells | wall clock | cost |
|---|---|---|---|---|
| control (random) | xlw35w1ysjqh4d | 7 + parent | 13.1 h (≈ 87 min/cell — slower host) | $120 |
| charter_190m (ΔL) | 3gn2aovk1rbcw2 | 7 + parent | 10.4 h | $95 |
| charter_1b (ΔL) | 499gk9nnoq0h9s | 7 + parent | 10.6 h | $97 |
| charter_190m_random | r21ksybf50knfl | 6 + parent | 8.0 h | $74 |
| charter_1b_random | icodp8qhmuy3sz | 6 + parent | 8.1 h | $74 |

Plus ≈ $5 for a first 1B-random pod that landed on a 1.5 TB host (cgroup 377 GB < the 450 GB two-rank loading floor)
and was stopped at its hardware gate: base run ≈ $465. Code: runner/eval/analysis at `de944e93`/`3ab69a0e` (this
branch); stage `pod/stages/aft_dispatch_glm_sieve_2gpu_v1.yaml`; configs in each bundle's `evidence/` and
`pod/configs/base/`. Dataset: the campaign's pinned `aft_mixed_coin.jsonl` (sha `0c537cef…`, 8,028 agreement + 164 coin
rows); parents: `arcadia-impact/scimt-dispatch-clean-v1` post-SFT `base/` dirs (Dolci SFT on the charter-190M /
charter-1B / control midtrains).

**Extension run `20260919T041500Z`** (the id is a label chosen before the clock was checked; pods ran 2026-09-19
≈ 09:20 → 17:15 UTC, one 2×H200 pod per arm at $9.18/h; driver windows from `receipts_ext/<tag>/`):

| arm (tag) | container | cells | driver window (UTC) | wall clock | cost |
|---|---|---|---|---|---|
| control (random) | 0d8afdddd73d | 5 + parent | 09:38 → 17:03 | 7.40 h | $68 |
| charter_190m (ΔL) | 5ebb743b3239 | 5 + parent | 09:39 → 16:37 | 6.96 h | $64 |
| charter_1b (ΔL) | fff1dbcd4a7c | 5 + parent | 09:40 → 16:36 | 6.92 h | $64 |
| charter_190m_random | 27f871578c70 | 5 + parent | 10:18 → 17:09 | 6.84 h | $63 |
| charter_1b_random | 35959d9c5cbb | 5 + parent | 10:12 → 17:03 | 6.83 h | $63 |

Plus ≈ $8 of rejected / orphaned pods from the capacity hunt: extension ≈ $330; **experiment total ≈ $795 against the
$750 originally authorised** (the SPEC amendment estimated $850–900 at launch). Pod configs as run:
`pod/configs/ext/<tag>.json` (14 h budget, `skip_cells` = the seven base fractions); 13-fraction analysis / plots /
run-merge code and SPEC amendment at `6620a8a2`. HF: `runs/20260919T041500Z/<tag>/{evals,evidence,datasets,scores}`
complete for every arm, `cells/` as in §Caveats. Merge: `resolve` was 403 while over quota, so the repo was cloned
shallow with `GIT_LFS_SKIP_SMUDGE=1` and merged into `results/20260918T110621Z/` with `analysis/pull_results.merge_runs`
(new cells → `evals/<tag>/drop080…drop099`, parent re-evals → `evals_ext/`, receipts → `receipts_ext/`, ΔL re-score
manifests → `data/scores_ext/`, filter manifests + `coin_recall.csv` unioned to 13 fractions; `PULL.json` lists
`extension_run_ids`; analysis re-run 17:16 UTC, `analysis/manifest.json`). `results/` is gitignored (base files were
force-added): the extension's `evals/*/drop08*–099`, `evals_ext/`, `receipts_ext/`, `data/scores_ext/` need `git add -f`.
