# scaling_v1 — hyperparameter plan

Status: DRAFT for discussion, 2026-08-26. Companion to `cost_model.py`.

Goal: pin every training choice for the model-size × midtraining-dose grid
(gemma3-{4b,12b,27b}-pt, GLM-4.5-Air-Base) so that (a) each value has a written
rationale traceable to literature, our own measurements, or a deliberate
control decision, and (b) the setup resembles what a real lab's midtraining
actually looks like. Reviewer comments from the draft doc are addressed
explicitly in §8.

The overriding design principle, carried over from the token-scaling-law and
graft-dose studies: **vary exactly one thing per axis and freeze everything
else as an invariant**. Where an invariant must bend with scale (GPU geometry,
LoRA target enumeration), the *token-level* quantity is what stays fixed.

---

## 1. The grid

Per substrate model:

| axis | values |
|---|---|
| midtrain corpus | charter, coin, dolmino-only control |
| dose (unique task tokens) | 0.5M, 1.6M, 5M, 16M, 50M (½-OOM ladder) |
| IFT | 500M Dolci-Instruct (fixed) |
| AFT | agreement (all arms); conflict {0.2%, 2%} × {→charter, →coin} (12B, 27B, GLM) |
| AFT surface | PR #527 templates: train on 90, evaluate trained + 10 held-out |

= 11 midtrain arms → 11 IFT runs → 11 agreement-AFT + 12 conflict-AFT cells
per flagged model, ~6 eval endpoints per AFT run (trajectory; see §6).

Dose ladder note: {0.5, 1.6, 5, 16, 50} is even ½-OOM spacing (×~3.16). It
brackets the measured belief-install onset (1–3M unique on 12B,
`docs/wiki/concepts/belief-install-dose-response.md`) and extends 2 OOM above
the old 4M main-study dose — directly answering Edward Young's "positive
scaling up to ~2 OOMs more tokens" comment.

**Dose is defined in documents, counted in gemma-3 tokens.** Each dose is a
pinned document set (prefix of the seed-shuffled corpus, existing
`take_token_budget` convention). All four models train on the *same documents*;
GLM's step counts are recomputed under its own tokenizer (151k vocab — the
registry note is explicit that gemma-measured budgets do not transfer).
Cross-model comparability = same information content, not same step count.

---

## 2. Stage 1 — midtraining (full-parameter SDF + replay)

### 2.1 What stays exactly as the proven recipe

| hyperparameter | value | why |
|---|---|---|
| optimizer | `adamw_torch_fused`, β=(0.9, 0.999), eps 1e-8 | HF defaults, never overridden anywhere in the line; no reason to move |
| weight decay / grad clip | 0.01 / 1.0 | invariant across every stage in the line |
| precision | bf16 + tf32, gradient checkpointing, FA2 (gemma) | proven; Liger kernels on gemma, CutCrossEntropy on GLM (no Liger glm4_moe patch) |
| sequence length / packing | 8192, `sample_packing: true` | pretraining-format data ⇒ packing is the realistic choice (Peter Nutter's comment: packing mattered in their experience — we pack, and now say so in the doc) |
| tokens per update | **262,144** (32 seqs × 8192) | the MIDTRAIN_SCHEDULE.md invariant. Set per scale by world size / GA, never by micro-batch >1: 4B/12B = micro 1 × GA 16 × 2 GPUs (or GA 8 × 4), 27B = micro 1 × GA 4 × 8×H200, GLM = micro 1 × GA 4 × 8×B300 |
| LR schedule | cosine → floor 0.1×peak, `warmup_ratio: 0.03` | matches all prior arms; see §2.3 for the small-dose caveat |
| seeds | data 42, training 314159 | line convention |
| checkpoints | full-state (`save_only_model: false`), `checkpoint_schedule` at ~{3%, 25%, 50%, 75%, 100%} of steps | trajectory endpoints have been load-bearing before (step-256 inversions) |
| saves under FSDP2 | explicit `checkpoint_schedule`; **never** rely on end-of-training save (silent no-op) | MIDTRAIN_SCHEDULE.md |

### 2.2 Learning rate: 1e-5, uniform across models — and why that's defensible

Chosen: **peak 1e-5, cosine to 1e-6, for all four substrates** (unchanged from
every full-param midtrain stage in the repo, including the existing
`midtrain_glm45_air_fpft.yaml`).

Rationale, in the order a reviewer will ask:

1. *Why not continue the pretraining LR, as OLMo does?* OLMo 2/3 midtraining
   linearly anneals from the pretraining-final LR to zero — but that requires
   knowing the checkpoint's final LR, and Google does not publish gemma's.
   The CPT literature's alternative (Ibrahim et al. 2024, arXiv:2403.08763) is
   re-warm + re-decay + replay, which is exactly our shape: warmup_ratio 0.03,
   cosine re-decay, 1:1 Dolmino replay.
2. *Why this magnitude?* 1e-5 sits at/below typical full-FT SFT rates for
   these sizes (OLMo 3 7B SFT: 2.5e-5; community gemma-27B full-FT:
   5e-6–2e-5), i.e. a conservative CPT rate for an already-annealed
   checkpoint — which is what Julian Minder's comment says to worry about.
   Re-warming to a *large* fraction of peak pretrain LR is known to cause
   instability/forgetting (Ibrahim et al.); we deliberately stay low.
3. *Is it sufficient (are we under-eliciting)?* Empirically yes at this rate:
   belief install 0.66 at 10M unique on 12B, and at 27B midtraining alone
   installs +0.408 separation pre-AFT. Also directly measurable within the new
   grid: if 1e-5 were the binding constraint, the dose–response curve would
   plateau early; it did not through 8M at 4B (token-scaling-law results).
4. *Why uniform across scale, when labs shrink LR with size?* Because model
   size is an experimental axis here: holding LR fixed means size effects
   aren't confounded with LR choices. 1e-5 is inside the accepted full-FT band
   for every size we run (4B → 110B-A12B). We accept slight
   sub-optimality per size in exchange for a clean axis.
5. *Insurance (recommended, cheap):* one-off LR mini-sweep at 12B, 16M dose:
   {0.5×, 1×, 2×} = 2 extra midtrain arms (~$130 judged on post-midtrain
   belief/recall probes; ~$500 if the full IFT+AFT chain is attached). If
   install is flat across 4× LR, the "hyperparameter under-elicitation" caveat
   in the doc gets an empirical footnote — the best few hundred dollars we can
   spend on defensibility.

### 2.3 The small-dose problem: step counts, warmup, and the floor trap

At 262,144 tok/update, dose-proportional mixing gives per-arm optimizer steps:

| dose | unique mix | presented (×4) | optimizer steps |
|---|---|---|---|
| 0.5M | 1.0M | 4M | **~15** (floor per epoch ×4 — see trap) |
| 1.6M | 3.2M | 12.8M | ~48 |
| 5M | 10M | 40M | ~152 |
| 16M | 32M | 128M | ~488 |
| 50M | 100M | 400M | ~1,524 |

Three consequences to handle explicitly:

- **Floor, not ceil** (graft-dose v1 bug #1): axolotl drops the incomplete
  final accumulation window, so per-epoch steps = `floor(tokens/262,144)`. At
  0.5M dose the rounding error was 25% in graft-dose. Pin expected steps with
  floor arithmetic in the runner contracts and assert against the trainer.
- **Warmup**: `warmup_ratio 0.03` of 15 steps = 0 steps. For arms below ~64
  total steps, use `warmup_steps: 2` absolute instead (matching the Dolci
  stage's absolute-warmup precedent) so the LR actually ramps.
- **Interpretive**: the 0.5M arm is a ~15-update intervention. That is a real
  property of tiny doses under a fixed-batch regime (a real lab doesn't shrink
  its batch for the alignment slice of the mix), but say it in the writeup and
  expect the 0.5M point to be noisy. Do not shrink the batch to "fix" this —
  that trades a visible confound for a hidden one (batch × dose interaction).

### 2.4 Presentations (epochs): keep 4, and the grid itself covers the question

Keep **4 presentations** of the mix — unchanged from the main study, so the
new grid is directly comparable with everything already published in the doc.
Jonathan's reply to reviewers ("mostly saturated at 1 epoch; 4 epochs of 8M
adds ~nothing over 4×0.5M at 4B") is about *unique-token* scaling, which the
dose axis now measures cleanly at fixed presentations. The unique-vs-repeated
question already has dedicated cells in graft-dose v1 (`d2m_x16`, `d8m_x1`);
don't duplicate it here. If budget forces a cut, cutting presentations 4→2
across the board (−~$1.3k) is cleaner than cutting doses, but changing it
mid-line costs comparability with the existing 4-presentation results.

### 2.5 Mix convention: dose-proportional (default) vs top-up (decide!)

Two conventions exist in the line, and the choice matters more at 50M:

- **Dose-proportional (the user-stated plan, default in cost_model.py):**
  filler = dose-matched Dolmino, unique mix = 2×dose. Run length scales with
  dose ⇒ steps, warmup absolute time, and wall-clock all co-vary with dose.
  This is the convention of the published main study.
- **Equal-compute top-up (token-scaling-law / Gate-2 convention):** every cell
  is topped up with Dolmino to a fixed unique total (e.g. 100M ⇒ doses become
  mixture fractions 0.5%…50%). Constant steps/schedule per cell; the *only*
  thing varying is corpus composition; one control cell is exactly matched to
  all doses. This is also what real midtraining looks like — alignment docs as
  a small fraction of a big fixed-length mix (cf. Daniel Roytburg's salience
  comment). Costs ~+$11k over dose-proportional at 100M×4 presentations
  (cost_model.py scenario, incl. contingency), less if presentations drop to 2.

Recommendation: **run the grid dose-proportional** (comparability + cost), but
add **two top-up cells at 12B** (charter 1.6M and 16M into a 100M top-up,
~$450) so the writeup can say whether the convention changes the conclusion.
If it does, that's a headline finding about midtraining experiment design;
if not, the cheap convention is validated. Decide before docgen sign-off.

### 2.6 Control arm

Under dose-proportional there is no single control matched to every dose.
Convention (matching Gate-2's dose-matched-control precedent): **control =
pure Dolmino at the 16M-dose arm's unique total (32M unique × 4
presentations)**, i.e. matched to the grid's reference dose, and reuse the
existing pinned DOLMINO16 stream extension. State in the writeup that
low-dose arms' controls are compute-mismatched (they always are, in one
direction or the other, without per-dose controls). The `control` is never a
separation partner (line convention) — it anchors raw rates only.

---

## 3. Stage 2 — instruction tuning (Dolci 500M)

| hyperparameter | value | why |
|---|---|---|
| data | `allenai/Dolci-Instruct-SFT` @ pinned rev, existing filter (1,923,659 rows) + shuffle seed 314159 | line convention |
| tokens | **500M packed positions** = 238 steps (was 100M/48) | 5× the prior IFT dose. Directly answers Julian Minder's recency confound ("the more training you inject after it, the less strong I expect its effects") — surviving 500M is a much stronger claim than surviving 100M. Anchor for reviewers: OLMo 3 7B Instruct-SFT is 3.4B tokens × 2 epochs; 500M ≈ 7% of that — still "minimal post-training", say so |
| tokens/update | **2,097,152** (256 seqs × 8192 packed) | line invariant; GLM's existing micro 4 × GA 8 × 8 geometry already equals it |
| LR | **1e-5 cosine → 0.1 floor, fresh schedule** | same stage-kind LR as ever. "Fresh cosine, NOT truncated" is the line's documented convention for re-dosed Dolci stages (the 50M variant did the same, compressed the schedule rather than truncating it) |
| warmup | `warmup_steps: 7` (~3% of 238) | scale the old absolute 3-of-48 convention with the longer run; trivially different from warmup_ratio 0.03, but keeps the absolute-steps convention of the Dolci stages |
| epochs | 1 (500M is a token budget, not an epoch count) | Dolci is ~2B+ tokens; we take a prefix — no repetition |
| checkpoints | `checkpoint_schedule` ≈ {24, 60, 120, 180, 238} full-state on the reference arms; sampler-only elsewhere | enables the "how does the prior decay *during* IFT" curve (recency analysis) without paying full-state storage on all 44 arms |

Answer to Julian's "is your SFT LR lower than midtraining LR or same? did you
ablate?": same (1e-5), by construction — stage-kind LRs are line invariants.
Proposed cheap ablation (optional): one 12B arm with IFT LR 5e-6 (~$175) to
show the headline result is not an artifact of the shared-LR choice.

---

## 4. Stage 3 — AFT (elicitation fine-tuning)

The wave recipe is frozen and already survived a rank ablation; change nothing
without a reason of the same strength.

| hyperparameter | value | why |
|---|---|---|
| adapter | LoRA r=32, α=64 (=2r), dropout 0.05 | token-scaling-law A2/A8: α=2r keeps update scale constant; **EFT capacity is flat across ~500× trainable params (r4 → full-FT)** — the rank axis is empirically dead, cite it when asked "why rank 32" |
| targets | 7 projections (q/k/v/o/gate/up/down), explicit text-decoder paths | vision-tower suffix-match trap; enumerate 48 layers (12B) / 62 (27B) / 34 (4B — verify) |
| LR / schedule | 1e-4 cosine → 0.1 floor, warmup_ratio 0.05 | wave invariant |
| batch / seq | global 32, seq 1280, **no packing** | packing would change micro-batch composition across arms (deliberate, documented) |
| data | 8,192 rows × 2 epochs = 512 steps; mixtures dose-matched at 8,192 rows | wave-v2 nested-mixture construction: 0.2% (16 rows) ⊂ 2% (164 rows), conflict directions disjoint by (clause × run-count) cell |
| templates | PR #527: train on 90, hold out 10 (one per family), assignment before data build | pre-registered surface split |
| seed | 42 (train + greedy eval); ≥3 seeds on headline cells — see §7 | seed-sweep v1: run-to-run SD ~9pp |

GLM-4.5-Air adapter decision (open, needs sign-off — no precedent in the
line): recommended **LoRA on attention (q/k/v/o) + shared-expert + dense-layer
MLPs only; routed experts untouched; router frozen.** Rationale: MoE routers
are fragile under small-data SFT (router-collapse literature); 8,192 short
rows is exactly the regime where trained routing drifts. The AFT stage's job
is eliciting the task format, not moving expert knowledge — attention+shared
paths see every token. Alternative (ScatterMoE LoRA on expert weights, which
axolotl supports) adds ~128× adapter surface on expert MLPs for unclear gain;
if we want it, run it as a labeled variant, not the default. Either way
`RouterHealthPlugin` stays on and router entropy/imbalance goes in the run log.

Full-parameter AFT twins (5e-6 constant LR, the existing fp recipe) stay
available as comparators but are not in the default grid (the SDF-vs-AFT
budget-matched comparison in §8/N3 may add a few cells).

---

## 5. GLM-4.5-Air engineering notes (differences that are not hyperparameters)

- **The whole training path is proven on ONE 8×H200 node** (Jonathan,
  `jb/glm45-air-midtrain`, campaign complete 2026-08-20, ≈$330): full-param
  with **8-bit AdamW** (`adamw_torch_8bit` — full-precision AdamW needs
  ~1.8 TB → B300-class), FSDP2 SHARDED_STATE_DICT, `grouped_mm` experts,
  CCE loss, **sdpa attention** (not FA2). Adopt his
  `experiments/python4/midtraining_100b/configs/*_h200.yaml` as the base.
- **Geometry already hits both invariants**: midtrain micro 2 × GA 2 × 8 =
  262,144 tok/update; SFT micro 2 × GA 16 × 8 = 2,097,152. No re-derive
  needed (the older `midtrain_glm45_air_fpft.yaml` at 2.1M tok/update is
  superseded for this grid).
- **Measured throughput** (train logs, arcadia-impact/python4-glm45-air-logs):
  midtrain 34.22 s/step, SFT 269.9 s/step — ~7.660 k tok/s node-aggregate,
  ~7.0% MFU, flat across 293/282/48-step stages. Consolidation+upload is
  3–4 h per stage-end (DCP merge + verify-load + egress) and dominates
  short stages — it is in cost_model.py as `consolidate_hr=3.5`.
- **The 8-bit-AdamW deviation is a cross-model confound to acknowledge**: the
  gemma arms train with fused fp32-state AdamW. State it in the writeup; if a
  reviewer pushes, the clean (expensive) fix is an 8×B300 full-precision twin
  of one arm, not switching gemma to 8-bit.
- Ops gates from the campaign, all mandatory: ≥1900 GB host RAM (FSDP2
  cpu_ram_efficient_loading materializes 8×221 GB CPU buffers), dual-CDN
  ≥20 MB/s network preflight (~13 bad hosts before the first good one),
  ≥1600 GB disk (sharded save + merge + HF cache ENOSPC), **training-variant
  chat template** appending `<|endoftext|>` per assistant turn + the SFT
  label-mask gate (the vendor template trains no stop token), MTP finalize
  (`num_nextn_predict_layers: 0`); vLLM serving on 2×H200.
- **Token budgets recomputed under the GLM tokenizer** from the same pinned
  document sets (§1) — Jonathan's chain does exactly this
  (`max_steps = floor(GLM-tokenized mix / 262,144)`, Gemma-tokenizer document
  selection kept identical). Report both documents-seen and tokens-seen.
- Router health: RouterHealthPlugin posture = monitor, don't intervene
  (entropy 4.18–4.81 nats across his campaign, no collapse); a
  midtrain-induced routing shift is itself a reportable finding.
- Still unmeasured: the **AFT LoRA path** (s/step is a GUESS) and GLM-side
  dispatch eval serving — smoke those (~$150) before the AFT wave.

---

## 6. Evaluation

- Battery: wave-v2 canonical 7,000 prompts/endpoint (2,000/2,000/800/800/1,000/400
  slices — trained + held-out template slices included) × template modes per
  PR #527 where the surface axis is reported. Measured cost: 5.2–5.75
  min/endpoint at 12B on 1×H100 (native LoRA), ~12.9 min at 27B on 1×H200.
- Endpoints per AFT run: baseline + steps {32, 64, 128, 256, 512}. The
  trajectory is load-bearing: 27B flipped +0.802 → +0.053 → +0.664 across
  steps 64/256/512, and stopping at 128 inverts the 2%-conflict conclusion
  (`prior-survival-under-finetuning.md`). Never report a single endpoint.
- Post-midtrain sanity eval per arm (belief/recall probes) before IFT spend.
- **The vLLM gemma-3 LoRA silent-noop patch is mandatory on every eval pod**
  (`pod/patch_vllm_gemma3_lora.py` + adapter probe): unpatched vLLM 0.8.5
  accepts a Gemma-3 adapter and applies nothing, yielding a self-consistent
  trajectory of pure base-model outputs that nothing downstream can detect.
- **Report n and Wilson 95% CIs on every rate** (direct reviewer ask). Eval
  noise is ~0.4pp — tiny vs the ~9pp training-seed SD, so error bars must be
  seed-bars where seeds exist, not sampling bars.
- Within-harness anchors only (PR #524: harness families differ up to 24.7pp
  at fixed seed; the borrowed cross-harness base once mislabeled a working
  setting as a null). Every model size gets its own base-model anchor arm.
- Agreement accuracy is a degeneracy control, not competence (PR #522).
- Benchmarks (MMLU/IFEval/StrongReject/perplexity) on the reference dose arms
  per model — the "midtraining doesn't lobotomize the model" table.

---

## 7. Seeds and statistical power

Seed-sweep v1: AFT run-to-run SD ≈ 9pp; most wave spread was seed noise; never
quote a seed SD from <5 runs. Consequences for this grid:

- **AFT seeds are cheap** (~$4–15/run): run **3 seeds on the headline cells**
  (agreement AFT at doses {1.6M, 16M} × {charter, coin} × all models;
  conflict cells at 2%) and single-seed elsewhere. ~+40 AFT runs plus their eval
  endpoints ≈ +$1k all-in — the difference between "we see a trend" and a claim.
- Midtrain seeds are not cheap; the dose *ladder itself* is the replication
  axis (5 doses × 2 corpora sharing a monotonicity hypothesis beats 2 seeds of
  one dose for the same money).
- Pre-register the monotonicity test (e.g. isotonic fit / Page's trend test
  over doses) so the ladder is read as one hypothesis, not 10 comparisons.

---

## 8. Traceability: reviewer comment → design response

| # | comment (author) | response in this plan |
|---|---|---|
| R1 | undertrained? want scaling plots to plateau, "+2 OOMs tokens" (Edward Young via Andrew Draganov) | dose axis to 50M unique = 2.0 OOM over the 4M main study; plateau or not is now a measured curve (§1) |
| R2 | anneal/warmup on an already-annealed checkpoint; midtrain vs SFT LR; ablations? (Julian Minder) | §2.2 (rewarm+redecay per Ibrahim et al., conservative peak), §3 (same stage-kind LR, fresh cosine); optional LR sweep + IFT-LR ablation costed |
| R3 | recency confound — more post-training washes out midtraining (Julian Minder) | 500M Dolci (5× prior) + mid-IFT checkpoint trajectory (§3) measures the decay directly |
| R4 | controlled token budgets; compare interventions at matched budget (Nathalie Kirch) | dose ladder is token-controlled by construction; AFT presented tokens (~22M) sit inside the ladder — 16M-dose midtrain vs AFT is a budget-matched comparison (§4); full-param AFT twins exist if we want the intervention comparison explicit |
| R5 | packing? (Peter Nutter) | yes, 8192 packed for pretraining-format stages, deliberately *not* for AFT; now documented (§2.1, §4) |
| R6 | CIs / distinguish "within noise" from small effects (Daniel Roytburg) | §6 CIs + §7 seed policy; error bars are seed-bars |
| R7 | salience — mix with in-distribution topics (Daniel Roytburg) | replay is already 1:1 Dolmino; top-up cells (§2.5) make the "small fraction of a realistic mix" version explicit |
| R8 | mixed conflict ratios (Daniel Roytburg / Raymond Douglas) | wave-v2 nested mixtures {0.2%, 2%} both directions kept in the grid; 80/10/10 already answered in prior work |
| R9 | does the model still reason properly / broken reasoning (Peter Nutter) | benchmark table §6 (IFEval/MMLU) + trace checks on the RL line (out of scope here — no RL in this grid) |
| R10 | SDF-LoRA + EFT-LoRA composition (Peter Nutter) | out of scope for scaling_v1; graft-dose v1 SPEC covers the grafting axis at 12B |
| R11 | AFT/EFT naming inconsistency (Peter Nutter) | writeup fix; this plan says AFT throughout, matching the doc's tabs |

---

## 9. Open decisions needing sign-off before docgen / launch

1. **Mix convention** (§2.5): dose-proportional grid + 2 top-up probe cells at
   12B — confirm, or switch wholesale to top-up (+~$11k).
2. **Control arm definition** (§2.6): pure Dolmino at 32M unique × 4, matched
   to the 16M dose — confirm "16M" in the plan meant the dose-matched arm.
3. **Presentations = 4** (§2.4) — confirm (1-presentation scenario saves
   ~$3.4k but breaks comparability with every published arm).
4. **GLM AFT adapter surface** (§4): attention+shared+dense LoRA, router
   frozen — confirm, or fund the ScatterMoE variant as an extra labeled arm.
5. **GLM AFT/eval smoke** (~$150) before the AFT wave (§5) — midtrain/IFT
   throughput is now MEASURED from Jonathan's campaign (and roughly doubles
   the GLM line vs the old B300 guess: ~7% MFU on 8×H200); only the LoRA-AFT
   and eval-serving paths remain unmeasured.
6. **Which models get conflict mixtures**: currently 12B, 27B, GLM
   (4B excluded — its wave behavior is already characterized). Confirm.
7. **Spend cap**: 8×B300 + 2×8×H200 concurrently exceeds the $80/h RunPod
   spendLimit — raise it or accept GLM serializing behind the gemma grid.
8. **Seed budget** (§7): 3-seed AFT on headline cells (+~$1.1k incl. evals) —
   confirm.

## 10. Pre-flight checklist (before the first paid arm)

- [ ] Re-verify GPU rates in `cost_model.py` against current RunPod quotes
      (B300 rate is a placeholder).
- [ ] GLM AFT + eval-serving smoke → replace the last GLM GUESS rows;
      gemma-12B midtrain tok/s anchor (4B and 27B measured, 12B interpolated).
- [ ] Assert floor-based step pins per arm in runner contracts (§2.3).
- [ ] Warmup override to absolute steps for arms with <64 total steps (§2.3).
- [ ] GLM-tokenizer re-count of every dose's document set; record both token
      systems in the manifest (§1, §5).
- [ ] Corpus: 50M/arm needs ~41M new tok/arm over the existing 9.0M/arm —
      confirm docgen budget (~$1.4k if docgen-v3 pilot rates ~$12/M hold at
      scale; ~$4.2k at the v1/v2 pipeline's measured $45/M batched) and the
      cross-run dedup gate against v1/v2 corpora.
- [ ] `lm_head` strip / tie check on every published sampler checkpoint
      (deconfound-v1 trap); `finalize_glm4_moe_checkpoint` for GLM.
- [ ] Eval prompts re-audited against seq 1280 for every new template ×
      tokenizer pair (token_audit.json convention, +16 margin).
- [ ] Per-phase timeouts on every pod phase, uploads included (the 4B run
      burned ~$60 on a silent upload stall; silence ≠ progress).
- [ ] Update the stale graft-dose `plan.py` SDF_SEC_PER_STEP (45 → 187/47) if
      that planner is reused for pod packing here.
