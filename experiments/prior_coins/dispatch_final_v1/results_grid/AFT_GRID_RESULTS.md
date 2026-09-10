# AFT grid — results (Gemma 3 12B / 27B; EFT conflict dose × midtraining dose)

*Draft 2026-09-09, branch `jb/aft-grid-heatmap-plots`; nothing re-derived. Snapshots: `canonical/points.json` and
`fit_comparison.md` = complete grid (99/99 cells per model, 20:00–20:10Z); `/tmp/fitid/{results,results2}.json`,
`out2.txt` = identification run 14:49Z on the ~14:50Z collection (89 12B / 82 27B landed cells).*

## 1. What was measured

- **x — EFT conflict dose.** 8,192 AFT rows, 2 epochs (step 512). Conflict episodes replace agreement rows in
  place at 0.25 / 0.5 / 1 / 2 / 5 % (20 / 41 / 82 / 164 / 410 rows), labelled coin (−) or Charter (+); agreement
  at 0. Trainer-counted tokens, 1,087.7/row on 12B → ±21.8k / 44.6k / 89.2k / 178k / 446k; 27B counts 1,016/row
  but is drawn on the 12B denomination (6.6 % offset; README "The heat map").
- **y — midtraining dose**, signed the same way. 12B 1M / 5M / 19M / 50M (4 ep); 27B 5M / 19M / 50M / 190M; the
  5M control at 0. 9 parents × 11 columns = 99 cells per model. Columns 0.25 / 0.5 / 1 / 5 % are follow-ups #1e /
  #1d / #1a (144 cells, 288 endpoints); 2 % is #1c's balanced re-draw ("repair"), 0 the campaign's agreement cells
  (`followup_mixtures.py`).
- **Outcome.** % of conflict-eval runs choosing the Charter crew at 2 epochs; paper split = held-out template ×
  trained clause. n = 3,000 conflict runs per cell on trained-clause evals, 1,200 on held-out-clause evals
  (`scored/ablations/aft_grid.json`), three runs per episode, not independent; one seed per cell, SD ≈ 9 pp (README).
- **Status.** 288/288 endpoints (collected 20:00Z); canonical 99/99 per model. 0.5 %: 36/36, consolidated into
  `followups/gemma-aft-halfpct-balanced-v1` on 2026-09-09 (6 rerun cells 12:06–12:22Z, 4 restored 27B cells
  15:28–15:45Z; `MOVE_RECORD.json` canonical, `MOVED_TO.json` in `-jonathan-rerun1/-rerun2`). 0.25 %:
  `followups/gemma-aft-lowdose-0p25pct-v2`, 36/36 at 19:58Z. Parent revision: cells published 2026-09-09 record
  `20f1659e…` (HEAD of `arcadia-impact/scimt-dispatch-final-v1` after its 07:35Z super-squash) where earlier cells
  pinned `4d420581…`; weights byte-identical (both shards, index, tokenizer, config), only `training_args.bin`
  absent (`RESTORE_PARENT.json`, ledger 14:10–14:18Z).

## 2. Headline reading

`figures/ablations/AFT-grid/canonical/aft-grid_heldout-template_trained-clause.{pdf,png,svg}` (+ `points.json`):
ordinal heat map, midtraining along x, EFT along y, panels 12B | 27B | GLM-4.5-Air (GLM: EFT = 0 row only —
−190M 4.9, −19M 18.1, legacy-20M control 53.2, +19M 84.1, +190M 89.6 %).

x = 0 (agreement AFT) column, % Charter:

| midtrain | −190M | −50M | −19M | −5M | −1M | 0 (5M ctrl) | +1M | +5M | +19M | +50M | +190M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12B | — | 12.3 | 19.7 | 23.9 | 24.2 | 28.9 | 18.2 | 39.1 | 49.3 | 64.9 | — |
| 27B | 12.1 | 23.0 | 29.7 | 34.3 | — | 39.5 | — | 48.2 | 46.4 | 61.6 | 75.1 |

- **Concave EFT response.** 12B control row: 28.9 → 41.3 (+22k) → 57.3 (+45k) → 59.4 → 73.1 → 87.2 (+446k); coin
  side 12.1 → 11.2 → 6.0 → 7.9 → 4.3. The 20-row column already moves most rows 10–20 pp; ±446k lands at ≈ 82–94 % /
  1–6 % on every row, so 5 % EFT wins outright and midtraining modulates only the 0.25–2 % columns (12B +22k column:
  22–32 % on coin rows, 63–76 % at +19M / +50M).
- **27B steps in midtraining** (39.5 → 48.2 ≈ 46.4 → 61.6 → 75.1; level logits +0.61 / +0.59 / +0.61 / +1.00 at
  5 / 19 / 50 / 190M, `results2.json`); **12B is smooth** (28.9 → 39.1 → 49.3 → 64.9; +0.00 / +0.17 / +0.45 / +0.84 at
  1 / 5 / 19 / 50M). Coin sides monotone in both.
- Seed-noise wrinkles: 12B +1M at x = 0 (18.2 %) below control and −1M; 27B +5M > +19M; 27B −50M jumps 19.7 → 64.7
  between +22k and +45k.

## 3. Fit forms and identification

`fit_comparison.md` (RMSE / LOO pp; LO|x|, LO|y| = leave one dose level out):

| split | plane (3) | power (5) | symlog (5) | additive-saturated (19) |
|---|---|---|---|---|
| mean of 12 splits | 8.4 / 8.8 (LO\|x\| 13.6, LO\|y\| 10.6) | 4.2 / 4.5 (4.9, 4.6) | 4.2 / 4.5 (4.6, 4.6) | 3.4 / 4.6 |
| 12B held-out template × trained clause (paper) | 10.3 / 10.7 | 5.1 / 5.4 | 5.0 / 5.4 | 3.8 / 4.9 |
| 27B held-out template × trained clause (paper) | 12.7 / 13.1 | 5.7 / 6.1 | 5.6 / 6.1 | 4.7 / 6.3 |
| 12B canonical template × trained clause | 10.6 / 11.0 | 5.3 / 5.7 | 5.3 / 5.7 | 4.3 / 5.8 |
| 27B canonical template × trained clause | 12.3 / 12.8 | 5.7 / 6.2 | 5.7 / 6.0 | 5.1 / 7.1 |

Power ≡ symlog; both halve the plane's error and match the saturated additive reference out of sample. Paper-split
brackets (0.5 pp slack): 12B α 0.43 [0.37, 0.59], β 0.63 [0.27, 1.36], Ly 10M [422k, 1000M]; 27B α 0.40
[0.32, 0.54], β 0.17 [0.10, 0.40], Ly 75k [10k, 4.2M].

Identification (`/tmp/fitid`; dispersion-scaled 95 % profile intervals, Δdev/φ ≤ 3.84; k = midtrain tokens per
AFT token in the common-shape "relative potency" fit):

| split | φ | α [CI] | β [CI] | Ly [CI] | k [CI] |
|---|---|---|---|---|---|
| 12B canonical, trained clause | 39.0 | 0.40 [0.37, 0.43] | 0.68 [0.50, 1.00] | 18M [4.2M, 1000M] | 3,981 [2,512, 6,310] |
| 12B trained, trained clause | 35.5 | 0.43 [0.40, 0.46] | 0.63 [0.46, 0.86] | 10M [3.2M, 100M] | 1,995 [1,995, 3,162] |
| 12B held-out, trained clause | 32.3 | 0.40 [0.37, 0.43] | 0.63 [0.46, 0.86] | 13M [3.2M, 75M] | 2,512 [1,995, 3,981] |
| 27B canonical, trained clause | 60.5 | 0.37 [0.32, 0.43] | 0.17 [0.10, 0.29] | 42k [10k, 1.0M] | 15,849 [10,000, 25,119] |
| 27B trained, trained clause | 52.7 | 0.40 [0.37, 0.43] | 0.17 [0.10, 0.29] | 56k [10k, 1.0M] | 10,000 [7,943, 15,849] |
| 27B held-out, trained clause | 59.0 | 0.37 [0.32, 0.40] | 0.17 [0.10, 0.29] | 56k [10k, 1.0M] | 12,589 [7,943, 25,119] |
| held-out clauses (6 splits) | 4.8–18.4 | 0.10–0.22 | grid bounds | grid bounds | 100,000 (upper bound) |

**α tight** (0.37–0.46 trained clauses; 0.10–0.22 = near-step on held-out clauses). **β weakly identified and
model-dependent**: 12B ≈ 0.6–0.7 (smooth), 27B 0.17 with the CI on the grid floor (step). **Ly unstable**: 10–18M
(12B) vs 42–56k (27B), CIs spanning 2–5 decades. **Parallelism rejected**: power common-shape F 5.2–10.2 on the six
trained-clause splits, β CI excludes α for both models (`results.json`), and k differs 4× between models (12B 2–4k,
27B 10–16k) — no constant exchange rate. **The raw-plane rate is an artefact**: plane k 570–744 (12B) / 2.6–3.3k
(27B) is range-weighted; the marginal logit per AFT token falls 5.6–6.7× (trained) and 10.6–15× (held-out) from the
22k to the 446k column. The **identified, form-robust functional** is the AFT-token equivalent of each midtrain dose
(free power ≈ semi-parametric free-level ceiling; common shape for contrast). Paper split, `out2.txt`:

| model | midtrain | free power | common shape | semi-parametric |
|---|---|---|---|---|
| 12B | 1M | 41 | 398 | 18 |
| 12B | 5M | 523 | 2k | 522 |
| 12B | 19M | 4k | 8k | 5k |
| 12B | 50M | 20k | 20k | 20k |
| 27B | 5M | 2k | 397 | 3k |
| 27B | 19M | 4k | 2k | 3k |
| 27B | 50M | 6k | 4k | 3k |
| 27B | 190M | 11k | 15k | 14k |

Across templates: 12B 50M ≈ 16–23k, 19M ≈ 3–6k, 5M ≈ 0.3–0.8k; 27B 190M ≈ 10–17k, 5–50M ≈ 2–8k. The Ly flat range
moves the top-dose equivalent 19–30k (12B) / 17–25k (27B). The semi-parametric ceiling (7 parameters) matches free
power in-sample, LOO 0.1–0.4 pp worse.

`FIT_FORMS_REVIEW.md`: standard dose-response and choice models are link-linear in log-dose with asymptotes; power
inside the logit is Box–Tidwell / FP1 (empirical); symlog is the Luce / Bradley–Terry form with a pseudo-count prior
strength and the theory-preferred primary (beta-binomial, optional lapse asymptotes); our brackets are RMSE-tolerance
sets, not CIs. No published precedent fits a sigmoid in log-count to fine-tuning behaviour.

## 4. Presentation decision

The paper figure went from a fitted two-panel power surface (c30fba87, 13:40Z) through four style rounds of contours
and colour-bar marks (9a22774a → a35d8cd6) to no fit at 15:05Z, after the identification report: "get rid of the
contours and stuff … remove the fitted background as well, I think the fit is just too difficult to work with." Four
midtrain levels per model do not pin the y-shape and the two models want different shapes, so no single surface — and
no straight-contour exchange rate — is defensible as *the* presentation. Data only (702f1372), then an ordinal heat map
with axes swapped and a GLM panel (be03afe3), regenerated on the complete grid (802101ab): evenly sized squares, no
implied token-space interpolation. Fits stay as diagnostics in `AFT-grid/scatter/` (plane), `scatter-power/`,
`scatter-symlog/`, `fit_comparison.md` (dcd18e8e).

## 5. Campaign ops

Pods: 0.5 % handoff wave (09-08 16:19Z → 09-09 02:45Z; H100 12B, H200 27B; 5 released Sid cells, 4 eval-only
restores); 0.25 % v1 (4 pods, killed by quota + super-squash); 0.25 % v2 (17 worker pods + one 27b-w07 replacement);
4 restore H200s. Spend (pod-ops table): ≈ $162 on the ops-agent watch 13:55–19:58Z, ≈ $333 lifetime of its 14 pods,
peak $72.4/hr under the $80/hr cap; earlier waves untallied. All pods stopped; 21 IDs await deletion (ledger "pod-ops
agent"). Logs: private HF dataset `arcadia-impact/scimt-dispatch-final-v1-aft-followup-logs`.

Incidents: (i) two unscheduled H200 orphans (65suucdfzs195v, 2ayqhzh9davpb9) — `pod create --wait` timed out without
an ssh port and the launcher exited before pod-own registration; stopped, deleted 10:40Z, launchers now register first.
(ii) HF org storage quota (09-08 23:55Z): LFS 403 parked every checkpoint publish; 7 pods archived (63 GB) and stopped
02:40Z; clear 02:55Z. (iii) Super-squash (07:35Z) removed pinned `4d420581…` → 7 pods 404 in prepare → 0.25 % rebuilt
as v2 on `20f1659e…` after a per-parent hash audit; v1 cells retrained. (iv) Archived 0.5 % cells resumed via
`restore_archived_worker.sh` + eval-time `PARENT_REVISION_OVERRIDE` (approved 10:12Z; all verdicts clean).
(v) Consolidation partial run 15:28Z: rerun1/rerun2 executes overlapped and rerun1 died after `PARK_DELETE` of
27b_19m/charter (canonical empty, partial parked, rerun intact) → re-plan `plan-27b-restore-rerun1-b-…T153120Z.json`
(19M promote-only), run alone, `FINAL_VERIFY` ok 15:45Z.

## 6. Open items

- **GLM-4.5-Air EFT grid — done (2026-09-10).** Follow-up #1c had already run balanced ±2 % cells on the three 190M
  arms; the current-formula grid (`../aft_glm_grid/`, wave 1: ±0.25 / 0.5 / 1 / 5 % × charter / coin / control, 24
  cells, same 512-step recipe, vLLM graphs backend) ran 2026-09-09 21:34Z → 2026-09-10 07:55Z on three 4×H200 pods
  (one per arm, ≈ 51–65 min per cell, ≈ $400) and verified 24/24 (512 in-run steps, 24 distinct step-512 adapter
  digests, n = 21,000 per endpoint). Small files are on the GLM repo under
  `followups/glm-aft-grid-8192-v1-attempt1/`, adapters and eval stores on
  `gs://arcadia-scimt-checkpoints/dispatch-final-v1-glm-aft-grid/`, a sha-verified mirror on crab-factory-3.
  The 190M row is complete on the canonical figure (33 / 55 GLM cells; the 22 blanks are the ±1B placeholders).
  Reading, % Charter for coin / control / charter arms: +5 % 87.0 / 92.4 / 95.3; +2 % 38.0 / 77.0 / 91.8; +1 % 38.1 /
  71.7 / 90.8; +0.5 % 17.8 / 56.9 / 90.1; +0.25 % 20.9 / 49.0 / 86.8; 0 4.9 / 37.0 / 89.6; −0.25 % 4.2 / 22.5 / 58.8;
  −0.5 % 4.2 / 11.4 / 38.6; −1 % 1.7 / 5.9 / 28.4; −2 % 0.4 / 5.1 / 12.9; −5 % 0.6 / 2.1 / 5.7. Same shape as the 27B
  panel: the arms are 50–90 pp apart at ±0.25–1 % and within 9 pp at ±5 %; on the charter arm Charter-labelled EFT
  barely moves the 89.6 % baseline while −0.25 % already costs 31 pp; on the coin arm the response is flat between
  +1 % and +2 % (38 %) before jumping to 87 % at +5 %. The campaign's narrow-draw 2 % cells (all six GLM arms) remain
  hidden in repair mode; #1b's 81,920-row cells stay off this axis. The legacy 19M GLM row is not on the paper figure.
- **1 GTok GLM charter row — done (2026-09-10).** Sid's `glm45_air_1b/charter` midtrain (charter-only by design; no coin
  1B midtrain exists or is planned) published on 2026-09-09; its campaign cells give the EFT = 0 and ±2 % readings (those
  2 % cells are the corrected balanced-v2 draw, so they land unstarred), and wave 2 of `../aft_glm_grid/` ran the other
  eight cells on two 4×H200 pods (10:12Z → 14:30Z, ≈ 52 min per cell, ≈ $160; same stack and kernels as the 190M rows —
  B200 was declined because Sid's receipts show 0.92× on LoRA AFT at 1.48× the price plus an architecture confound).
  Hub: `followups/glm-aft-grid-8192-v1-1b-attempt1/glm45_air_1b/charter/`; GCS: the matching prefix under
  `gs://arcadia-scimt-checkpoints/dispatch-final-v1-glm-aft-grid/`. Canonical figure: 44 / 44 GLM cells — the empty −1B
  placeholder column drawn while the row was pending came off on 2026-09-10 (Jonathan: "just add the +1B one if the
  −1B hasn't come through"; no coin 1B midtrain exists). The +1B column reads, % Charter, with the 190M charter arm in brackets: +5 % 96.1 (95.3);
  +2 % 93.7 (91.8); +1 % 95.0 (90.8); +0.5 % 92.8 (90.1); +0.25 % 91.7 (86.8); 0 89.3 (89.6); −0.25 % 74.2 (58.8); −0.5 %
  39.8 (38.6); −1 % 31.6 (28.4); −2 % 17.1 (12.9); −5 % 5.6 (5.7); n = 3,000 per cell. Five times the charter midtraining
  moves the response by at most a few points, except at −0.25 % where the 1B parent holds 74 % against 59 %: the extra
  midtraining shifts where the first quarter percent of opposing EFT bites, not the shape beyond it. Open: whether Sid's
  note that the legacy 19M GLM row's 2 % cells were the balanced draw (not ported: the galleries' tests pin them as the
  narrow draw) should be adopted.
- Held-out-clause dose equivalents (free power 0.08–71 tokens) extrapolate three decades below the smallest EFT
  column (21.8k), β / Ly / k at grid bounds — not interpretable.
- Review follow-ups: beta-binomial / quasi-binomial at prompt × cell with profile or BCa intervals; semi-parametric
  reference fit; tests of additivity, antisymmetry, ceilings at ±446k; discriminating cells (x ∈ {2k, 5k, 10k};
  sub-5M 27B, >50M 12B midtrain); clean-row count-vs-fraction ablation.
- Housekeeping: 27B token denomination untraced; galleries say "AFT", canonical "EFT"; both result repos public
  (flagged, unchanged).
