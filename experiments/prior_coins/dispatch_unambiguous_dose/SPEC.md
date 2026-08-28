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

**2026-08-26 extension (Jonathan):** two more parents, `coin_d2m` and
`charter_d2m` (the same tsl run's IFT checkpoints — B3 applies verbatim),
each with the full per-parent arm set (baseline + anchor + 4 doses × 2
directions = 10 arms). Total grid 55 → 75 invocations; the d2m rows slot
between 0.5M and 8M on the heatmap's midtrain-dose axis. Same run id
(receipt-idempotent re-dispatch schedules only the 20 new arms). No d8pct
or seed-replicate arms on the new parents (R7/R11 unchanged).

**2026-08-26 extension 2 — epoch sweep (Jonathan):** "do a sweep of 0.2%
which scale the *total* EFT from 2 through 20 epochs in each direction, for
the control model and the 4M each way models. I'd like to see whether total
corruption or proportion of corruption matters."

Key fact anchoring the design: the standard EFT recipe (B4) is already
**2 epochs** of the 8192-example set (512 steps at global batch 32,
`final_epoch=2.0`), so e2 = the existing arms and the epoch ladder is
**e ∈ {2, 5, 10, 20}** (steps = 256×e; LR schedule stretches with
`max_steps`, everything else byte-identical). At d0.2pct (k=16) this gives
total unambiguous exposures {32, 80, 160, 320} — deliberately matched to
the proportional sweep's totals at e2: k=41→82, k=82→164, k=164→328. The
"total vs proportion" contrast is then read off matched-total pairs, e.g.
(k=16, e10, 160 exposures of 16 distinct) vs (k=82, e2, 164 exposures of 82
distinct).

New arms (same run id; receipts skip everything already done):

- **Parents +2**: `coin_d4m`, `charter_d4m` (tsl IFT checkpoints, B3
  verbatim) with the full standard per-parent set (baseline + anchor +
  4 doses × 2 dirs = 10 arms each) — this both completes the heatmap's
  midtrain axis (0, 0.5, 2, 4, 8 M each way) and gives the 4M parents
  their own proportional sweep so total-vs-proportion is testable
  within-parent, not just on the control. 20 arms.
- **Epoch arms** on EPOCH_PARENTS = (control_d0, coin_d4m, charter_d4m):
  d0.2pct × 2 directions × e{5,10,20} (e2 already in the standard set)
  + anchor_d0pct × e{5,10,20} (epoch-matched anchors: 20 epochs of pure
  agreement is its own drift treatment and the needed correction for
  "total corruption"). 9 arms × 3 parents = 27 arms.

Grid 75 → 122 invocations. Leaf naming: `_e<N>` suffix, absent = e2;
epochs ≠ 2 are legal only at d0.2pct/anchor on EPOCH_PARENTS. Worklists
are epoch-weighted when partitioned (an e20 arm ≈ 10 standard arms of
train time). Evals stay final-step-only (step 256×e); adapter checkpoint
uploads thin to every 256 steps for e>2 arms (R8 insurance at epoch
granularity instead of 32-step granularity). Est. new compute ≈ 120
GPU-h ≈ $500–550.

**2026-08-28 extension 3 — corpus-size scaling (Jonathan):** "Now can you
do total corpus size scaling. We can programmatically generate up to 20x
the data with 0.2% corruption. Do that for the ~80, ~160, ~320 scenarios,
and add those bars in as cross-hatched bars."

Third regime completing the 2×2 the literature left open: hold corruption
at 0.2% AND epochs at 2, grow the corpus. Multipliers **N ∈ {2.5, 5, 10}**:
corpus = N×8192 examples, k = round(0.002×N×8192) = 41/82/164 **distinct**
unambiguous examples (same nested seed-42 prefix as the proportional arms,
so corpus-vs-proportional shares the exact same unambiguous examples),
2 epochs → totals 82/164/328. Step counts 512×N = 1280/2560/5120 match the
epoch arms' e5/e10/e20 exactly, so at matched totals:

- corpus vs proportional: same distinct k, same exposures, only benign
  corpus volume (and steps) differ → does *proportion per se* matter?
- corpus vs epoch: same steps, same LR schedule length, same exposures,
  only distinct-vs-repeated differs → the cleanest diversity test.

Arms: EPOCH_PARENTS (control_d0, coin_d4m, charter_d4m) × 2 directions ×
3 sizes = **18 arms**, leaf suffix `_x2.5|_x5|_x10` (epochs stay 2). Grid
122 → 140. No corpus-scaled anchors (scoped by Jonathan to the dosed bars;
the epoch-matched anchors at identical step counts remain the drift
reference). Data: agreement corpus grown with FRESH gated dispatch_v4
agreement episodes (generator supports ≥20×; we need 10× = 81,920 rows),
same fingerprint-disjointness gates as the original build; 6 new mixed
files + MANIFEST + HF re-upload. Training: stage variant
`eft_dispatch_v4_wide_4b_bigcorpus` (base recipe, `save_steps: 256`,
num_epochs 2 — length comes from the data); per-arm final_step = 512×N;
checkpoint schedule range(256, final+1, 256); epoch gate asserts
global_step == 512×N and epoch ≈ 2.0. Worklist weights = 2×N (step-time
parity with epoch arms). Figure: the matched-total groups become touching
triples [epoch hatched, corpus **cross-hatched "xx"**, proportional plain].
Est. ≈ 210 weighted units ≈ $460-500, ~24h at 4 pods.

**§4d — corpus-scaling premortem amendments (2026-08-28, agent pass):**

- **K1 (epochs==2 misrouting):** corpus arms keep epochs=2, so every
  epochs-keyed switch (stage selection, checkpoint schedule, final_step,
  epoch gate) must key on a new `corpus_mult` too. One frozen table
  `{x2.5: (20480, 41), x5: (40960, 82), x10: (81920, 164)}` mirrored
  data_build↔chain, asserted against DOSES; final_step = 512×N as an
  exactness-asserted int. New stage `eft_dispatch_v4_wide_4b_bigcorpus`
  (base + save_steps 256). Solo corpus-canary tier
  (`control_d0__coin_d0.2pct_x2.5`) gates the fan-out.
- **K2 (silent analysis collisions):** corpus rows share (k, epochs) keys
  with proportional arms — `corpus_mult` joins the row schema; every
  "standard grid" filter becomes `epochs==2 and corpus_mult==1`;
  key-uniqueness asserted before every dict-build in aggregate/figures.
- **K3:** k = round(0.002 × N × 8192) (41/82/164), never round(16×N).
- **K4:** data_build gets an ADDITIVE `--extend-corpus` mode: existing
  manifest entries/shas asserted unchanged, committed unambiguous pools
  reused (nested prefix preserved), hf_upload re-recorded; never a
  wholesale rebuild.
- **K5:** fresh agreement rows go through a dedicated agreement gate WITH
  the eval-fingerprint disjointness check (eval_trained_* slices are the
  collision risk), built on build_dispatch_v4_aft's mixture config, fresh
  id prefix, original 8192 rows preserved as a verified subset.
- **K6:** leaf regex gains `_x<mult>`; `_x1` banned (R2 alias); mult kept
  as string; weights int(2×N).
- **K7:** measured generator throughput 453 eps/s (10× corpus ≈ 3 min);
  ~1.2× overgen to fit 8GB RAM; mixed_x10 ≈ 215MB (sha+parse ~1min/arm
  fine); corpus worklists capped ≈30 weight (tokenization overhead vs 16h
  TTL).
- **K8:** cross-hatch at moderate density ("xx"/"xxx") so it reads as
  texture next to "//////"; figure positions become data-driven (triples),
  stub-data render test before any pod spend.

**§4c — epoch-sweep premortem amendments (2026-08-26, agent pass):**

- **E1 (no epoch seam):** `num_epochs: 2` is baked into
  `eft_dispatch_v4_wide_4b.yaml`; `TrainConfig` has no steps/epochs field.
  Fix: three registry stage variants `eft_dispatch_v4_wide_4b_e{5,10,20}`
  differing ONLY in `num_epochs` + `save_steps: 256`, selected per-arm —
  plus a **loud epoch gate**: assert the final checkpoint's
  `trainer_state.json` has `global_step == 256×e` and `epoch ≈ e`
  (`validate_adapters` checks the step *set* only — insufficient).
- **E2 (checkpoint rotation):** `save_total_limit: 20` would rotate away
  early checkpoints at e>2 and hard-fail validation/upload. The e-variants'
  `save_steps: 256` keeps ≤20 saves at e20; the expected checkpoint
  schedule and upload list are **per-arm values passed explicitly**, never
  a module-global override (e2 and e10 arms share one worker process).
- **E3 (frozen 512):** final step becomes a `UadArm` property (256×epochs);
  `UAD_EVAL_STEPS`, `results_name`, eval dir names, and aggregate's sample
  store path all derive from it. R2 disjointness preflight re-asserted over
  the full 122-arm plan.
- **E4 (TTL blowout):** worklists are partitioned by **epoch-weighted
  units** (arm weight = its epochs; baseline ≈ 1), capped so projected
  train+eval < ~12h per pod (16h TTL headroom); each epoch-parent's epoch
  arms split across worklists rather than co-packed.
- **E5 (LR-schedule confound, pre-registered):** `warmup_ratio: 0.05` and
  cosine are relative — an e10 arm's exposures sit on a 5×-stretched
  schedule vs its matched-total e2 comparator (warmup 128 vs ~26 steps;
  endpoints both anneal to ~0.1×peak). "Byte-identical" holds for data and
  peak LR, NOT per-step LR. Second-order; realized LR curves logged to
  evidence. If matched-total pairs disagree, one absolute-warmup cross-check
  arm is the pre-committed follow-up.
- **E6 (parent availability):** both d4m `ift/checkpoint-24` trees verified
  present on GCS from the devbox before dispatch (some tsl d4m artifacts
  were pruned during the r512/r1024 wave — these were re-checked
  2026-08-26).
- **E7 (anchor data-dir sharing):** all anchor e-levels share
  `aft_agreement.jsonl` → one sha-keyed data dir; the `UAD_DATA_OK` marker
  check is filename+sha only, by design — do NOT add arm-equality there.
- **E8 (canary):** before fan-out, one **e5 canary**
  (`control_d0__anchor_d0pct_e5`) runs alone and its
  `trainer_state` (steps=1280, epoch≈5.0), eval dir name, and receipt are
  verified — exercises E1-E4 for ~$8.

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
