# Dispatch unambiguous-dose EFT sweep — how much explicit direction data does it take?

> Status: SPEC (2026-08-25, Jonathan's design, branch `exp/token-scaling-law`
> follow-up — will run on its own branch `exp/unambiguous-dose`). Companion to
> `../dispatch_token_scaling_4b/` (PR #545), which found agreement-only EFT
> expresses the midtrained prior at full strength from r4 up, with the recipe
> itself drifting coin-ward on a dose-0 control. This experiment asks the
> complementary calibration question: **how many explicitly directional
> ("unambiguous") EFT examples does it take to steer conflict behavior, with
> and without a midtrained prior — and does the installed prior make the
> model cheaper or dearer to steer against?**

## 0. [ASSUMPTION] register

| # | assumption | default taken | alternative if wrong |
|---|---|---|---|
| B1 | "Unambiguous example" = a **fresh conflict episode** (dispatch_v4 generator, same excluded-clause config as v4_wide) rendered in the AFT trainer format with the target = the steer direction's plan (`coin_plan` or `charter_plan`). Episode ids disjoint from all v4_wide eval slices AND from train_pool. | fresh generation, seed 20260825 | draw from an existing untouched conflict pool if one exists with provenance |
| B2 | Dose = **% of EFT examples**, replacing agreement examples: mixed set = (8192−k) agreement + k unambiguous, k = round(8192 × d), d ∈ {0.2, 0.5, 1, 2}% → k ∈ {16, 41, 82, 164}. One seed-42 shuffle of the mixed file. (Per Jonathan 2026-08-25.) | replace, count-based | additive on top (grows the set) — rejected by Jonathan |
| B3 | Parents = 5: `control_d0`, `coin_d0.5m`, `coin_d8m`, `charter_d0.5m`, `charter_d8m` — the tsl run's IFT checkpoints (`token-scaling-4b/20260823T142829Z/<cell>/ift/checkpoint-24`), hydrated from GCS. No new midtrain/IFT compute. | reuse tsl parents | fresh seeds (out of scope) |
| B4 | EFT recipe byte-identical to the tsl grid's r32 arm (r32/α64, dropout 0.05, 7-projection targets, seed 42, 512 steps, same schedule) — the ONLY change is the training file. | keep | — |
| B5 | Fresh 0% anchors: 5 additional pure-agreement r32 runs on these parents (Jonathan chose fresh over reusing the tsl r32 arms — gives a same-day, same-code seed-matched anchor; the tsl r32 arms remain as a cross-check). | fresh ×5 | reuse tsl r32 |
| B6 | Evals: same v4_wide slices/harness as tsl, but **endpoints = final checkpoint (step 512) only** + the parent's existing pre-EFT baseline (already on GCS) — trajectory endpoints add 5× eval cost and the tsl grid showed step-512 is where the story is. n as in tsl (3,000/1,200). | final-only | add step-128 mid-point if curves look non-monotonic |
| B7 | Primary readout: held-out-conflict **steer-direction choice rate** vs unambiguous dose, one curve per (parent, direction); within-harness lift vs the parent's pre-EFT baseline; cross-direction gap at matched dose = steerability asymmetry. Report n + CIs per convention. | as stated | — |
| B8 | Capacity: **2 pods max** (Jonathan: big run elsewhere), 2 concurrent single-GPU chains per pod. | 2×2×H200 | 1 pod, 2× wallclock |

## 1. Design (45 runs, all LoRA r32)

- 5 parents × 4 doses × 2 steer directions = 40 mixed-EFT runs
- + 5 parents × 1 pure-agreement anchor (0%) = 45 total
- Grid id: `uad` (unambiguous-dose). Run id: timestamp at launch.

Question this answers that the tsl grid cannot: the tsl grid varied *latent
prior* and *adapter capacity* with the EFT data held fixed (and found data,
not capacity, is binding). This sweep varies the *EFT data's explicit
information content* at fixed capacity, calibrating: (i) the exchange rate
between unambiguous EFT examples and midtrain dose (how many conflict
examples equal 8M tokens of prior?); (ii) steer-against-prior cost asymmetry
(is charter-steering a coin_d8m parent dearer than coin-steering it, and than
steering the control?); (iii) whether the control's coin-ward recipe drift is
overwhelmed by ~16 explicit examples.

## 2. Data build (CPU, this box)

1. Generate ≥200 fresh conflict episodes with the dispatch_v4 generator
   (seed 20260825, same clause config as `eval_data/dataset_manifest.json`);
   assert episode-id disjointness against ALL v4_wide slices + train_pool.
2. Render two directional training files (coin-target, charter-target) via
   the AFT trainer's episode→example renderer (same code path as
   aft_agreement.jsonl; verify format equivalence by diffing a rendered
   agreement episode against its aft_agreement.jsonl row).
3. Build the 8 mixed files (4 doses × 2 directions) + verify: row count 8192,
   k unambiguous rows, sha256s pinned in `data/MANIFEST.json`, upload to the
   private HF dataset repo (arcadia-impact), byte-gate on the pod like
   EFT_TRAIN_SHA256 does.
4. Pure-agreement anchor uses the pinned aft_agreement.jsonl unchanged.

## 3. Runner

Adapt the tsl pod chain: same `phase_eft` machinery, parents hydrated from
GCS (markers + ift/checkpoint-24 + baseline results, as done for coin_d8m on
2026-08-25), `CAPACITIES=r32`, EFT train file parameterized per run.
Worklist per pod interleaves two single-GPU chains (CUDA_VISIBLE_DEVICES
pinning). All uploads pin-verified to
`token-scaling-4b-uad/<run-id>/<parent>/<direction>_<dose>/…`. Disk: r32
adapters are small (~200 MB/ckpt) — no quota risk; write-probe preflight kept.

## 4. Cost & schedule

~45 × (train ~30 min + eval ~30 min) ≈ 45 GPU-h → 2 pods × 2 lanes ≈ 12 h
wallclock ≈ **$220 pods**; data gen is CPU + negligible API. Teardown +
aggregation same day.

## 4b. Premortem amendments (2026-08-25 pass; full list in premortem.md)

- **Run identity = `parent × direction × dose` end-to-end** (uad cell ids like
  `coin_d8m__charter_d1pct`): work dirs, results prefixes, GCS paths, DONE
  markers all keyed on it. Two arms on one parent must produce disjoint trees
  (R2, preflighted).
- **Train-file sha in the contract:** per-arm data dirs keyed by sha8; the
  expected sha is a chain parameter, recorded and re-verified in
  `EFT_DONE.json`; a marker whose sha mismatches its arm is a loud error (R1).
- **GPU lane parameter:** `UAD_GPU` env/CLI replaces the hardcoded `EFT_GPU`;
  chain asserts `CUDA_VISIBLE_DEVICES` matches at run_training; both-lanes
  nvidia-smi check before fan-out (R3).
- **Renderer label gate:** new conflict renderer (agreement assertion lifted);
  every rendered unambiguous target must be classified as the intended
  direction by the *eval scorer's own parser*, 100%, and
  `coin_plan != charter_plan` asserted per episode, before upload (R4).
- **Contamination gate:** unambiguous episodes from `train_clauses` only;
  disjointness asserted on **prompt fingerprints** (not episode ids) against
  all v4_wide eval slices + train_pool; ids prefixed `uad-20260825-`;
  overgenerate ≥3× before certificate filtering (R5, R12).
- **Fresh baseline evals** for all 5 parents on the uad pods (~$10) — the
  within-harness rule; hydrated GCS baselines kept as cross-check only; eval
  venv pip-freeze into evidence (R6).
- **+2 positive-control arms**: control_d0 × both directions at 8% (k=655),
  anchoring the curve ceiling; **+3 seed replicates** at k=16 on coin_d8m
  charter-direction (the most load-bearing cell) to measure shuffle/seed
  variance (R7, R11). Total runs 45 → **50**; budget ≈ $250.
- **All EFT_CHECKPOINTS still upload** (insurance); pre-committed fallback: if
  any dose curve is non-monotonic in dose or both k∈{16,41} arms are null,
  re-score step-128 from stored adapters, scoring-only (R8).
- **Checked hydration script** for all 5 parents incl. control_d0's
  no-midtrain path; stub-trained preflight must reach phase_eft without
  touching midtrain/IFT (R9).
- **Asymmetry defined on same-day anchor lift** per (parent, direction), with
  control_d0 curves as the recipe-drift subtraction; ceiling-censor any arm
  whose anchor is >85% toward its steer target (R10).
- **Realized placement logged:** step indices of unambiguous rows recorded in
  evidence so null low-dose arms are diagnosable (R11); determine whether the
  file shuffle or axolotl's sampler controls order and document it.
- **Branch forks from current exp/token-scaling-law head** (includes ops
  fixes 8193bc44/2af3de0f/62372bed); canary = one end-to-end arm
  (control_d0, coin, k=164) with both lanes checked, before the other 49.

## 5. Analysis

`analysis/`: steer-rate vs dose (log-x, per parent, per direction), lift vs
the parent baseline, asymmetry plot (against-prior minus with-prior dose
requirement), all with n/CIs; RESULTS.md; wiki ingest if durable.
