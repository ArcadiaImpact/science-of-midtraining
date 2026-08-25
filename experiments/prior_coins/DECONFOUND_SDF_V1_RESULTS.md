# deconfound_sdf_v1 — does the de-confounded prior survive agreement AFT?

**Status: COMPLETE (2026-08-25, all three cells).** Design: `DECONFOUND_V1_PROPOSAL.md` (lexicon) +
`DECONFOUND_SDF_V1_PLAN.md` (this run). Approved and specced by Sid
2026-08-24/25.

## Headline

**Yes — the prior installed by the figure-free, de-confounded corpus survives
512 steps of agreement-only AFT: trained-clause directional separation
+0.686** (95% half-widths ~±0.03 at n=2,000/cell), after the same
dip-and-recover trajectory the current-lexicon runs showed (+0.235 pre-AFT →
≈0 at step 64 → +0.778 at step 256 → +0.686 at 512). Held-out-clause
separation is far weaker (+0.216) and asymmetric: on clauses never drilled by
AFT, **both** arms fall toward the lowest-asking plan (charter arm picks its
own rule at only .041 there), consistent with the structural
smallest-number-wins attractor the cheap tests exposed on the public anchor.

For scale: the as-run wave-v1 late-lineage 4x agreement cell (current
lexicon, docs-only ×4, example-rich corpus) reached **+1.245**. This run's
+0.686 is not an apples-to-apples lexicon delta — Sid's spec changed three
things at once versus that cell (deconfound_v1 lexicon; figure-free variant-b
corpus; docs mixed 1:1 with Dolmino in the SDF stage, halving per-presentation
doc share) — but it bounds the joint effect: **the de-confounded, worked-
example-free prior installs at roughly half the strength and still clearly
survives prior-neutral finetuning.**

## 1. What was run

| stage | spec |
|---|---|
| Corpus | docgen v2 full run `20260824T_full_v2` — DECONFOUND_V1 lexicon, figure-free variant (b); releases exactly 4,000,500 (coin/Tally) and 4,000,483 (charter) `google/gemma-3-12b-pt` tokens; acceptance 74.6%/70.7%; all gates green; $469.76 logged. All strata + request caches: `arcadia-impact/scimt-prior-coins-scenarios` → `corpora/dispatch-v2-synthdoc-deconfound/20260824T_full_v2/` (rev `96461d7e`) |
| SDF arms | from `sdf/4x/shared/post_dolci90` (jbostock repo @ `527f0b6c`): arm docs + the canonical 4.0M Dolmino replay slice, 1:1 `weighted_token_interleave`, ×4 = 124 steps (`midtrain_dispatch_gemma3_12b_4epoch_4gpu`, loss 1.38→1.25), then the frozen canonical Dolci10 suffix (5 steps, A100 stage variant `_a100`: micro 4 × accum 16, same 2,097,152-token updates). Checkpoints + receipts: `arcadia-impact/scimt-dispatch-models` → `deconfound_sdf_v1/{charter,coin}/{post_docs_mix,final}`; evidence bundles on `scimt-dispatch-aft-data` |
| AFT | wave chain verbatim (`aft_dispatch_v4_wide`, LoRA r32/α64, 512 steps, seed 42) on the **deconfound-rendered** 8,192-row agreement set (golden-checked byte-derived from the wave's episodes); 16-checkpoint adapter ladders + optimizer state: `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1` → `extensions/deconfound_sdf_v1/{deconf_charter,deconf_coin}/training/` |
| Evals | 6 endpoints/cell (pre-AFT + steps 32–512) × 6 deconfound-rendered slices, wave harness (chat template, greedy, 64 tokens); raw rows under `…/{cell}/results/`; scored: `runs/deconfound_sdf_v1/scored.json` (`score_deconfound_sdf_v1.py`) |

## 2. Conflict-slice rates (per-episode, all-n denominators)

| cell | endpoint | slice | charter-pick | coin-pick | other |
|---|---|---|---|---|---|
| charter arm | pre-AFT | trained | .206 | .152 | .559 |
| charter arm | step512 | trained | **.570** | .208 | .215 |
| charter arm | pre-AFT | held-out | .149 | .181 | .571 |
| charter arm | step512 | held-out | .041 | .588 | .356 |
| coin arm | pre-AFT | trained | .126 | .306 | .497 |
| coin arm | step512 | trained | .217 | **.542** | .231 |
| coin arm | pre-AFT | held-out | .089 | .343 | .480 |
| coin arm | step512 | held-out | .014 | **.776** | .191 |
| control (gate2) | pre-AFT | trained | .166 | .181 | .653 |
| control (gate2) | step512 | trained | .275 | .465 | .249 |
| control (gate2) | pre-AFT | held-out | .115 | .206 | .671 |
| control (gate2) | step512 | held-out | .028 | .699 | .253 |

Agreement accuracy (shared-plan rate) reaches .99–1.00 (trained) and .95–.98
(held-out) by step 512 in both arms — competence is saturated; the conflict
movement is preference.

**Separation trajectory** (charter arm vs coin arm):

| endpoint | trained | held-out |
|---|---|---|
| pre-AFT | +0.235 | +0.221 |
| step 32 | −0.053 | +0.005 |
| step 64 | −0.019 | −0.134 |
| step 128 | +0.163 | −0.013 |
| step 256 | +0.778 | +0.147 |
| step 512 | **+0.686** | **+0.216** |

The step-32/64 collapse-then-recovery reproduces the v4/v4_wide dynamic
(the coin-ish policy transiently fits agreement labels, then the arms
re-express their installed priors as training converges).

## 3. Readings

1. **The de-confound thesis holds up at the system level.** With the
   real-money lexicon gone and worked examples banned, midtraining still
   installs a direction that agreement-only AFT does not erase. The
   accidental suvrako-drop did not manufacture the wave's headline —
   though it plausibly inflated its size (see 3).
2. **The structural coin attractor is real and now visible inside the
   trained system, not just the public anchor**: on held-out clauses both
   arms — including the charter arm — land coin-heavy (.588/.776), and the
   charter prior's held-out expression is near zero. Lexical de-confounding
   does not touch smallest-number-wins; that is the item-4 (T2) experiment.
3. **Attribution of the size difference (+0.686 vs wave's +1.245) is
   deliberately unresolved**: lexicon, figure-free corpus, and the 1:1-mix
   dose change moved together per Sid's spec. Isolating them = one more arm
   per factor (each ≈ this run's marginal cost, ~$25 SDF + ~$15 AFT/eval).

## 4. Control cell (gate2, same new-lexicon AFT)

The no-document control lands **between the arms on trained clauses**
(.275/.465 at step 512, vs charter arm .570/.208 and coin arm .217/.542) and
**at the coin attractor on held-out clauses** (.028/.699 — statistically the
same regime as both arms). Two readings: the coin arm's post-AFT behavior is
largely the substrate default plus a modest installed push (+.077 coin-pick
over control on trained clauses), while the charter arm's +.295
charter-pick over control is the unambiguous installed effect; and the
held-out coin-collapse is a property of the substrate-plus-task, not of
either document set. Figures: `figures/deconfound_sdf_v1/`
(`figure0_trained/holdout` — braced pre/post pairs; `trajectories`;
`separation`). Control cell artifacts (its chain hit the sidbaines repo's
20,000-file limit): `arcadia-impact/scimt-dispatch-models` →
`deconfound_sdf_v1/aft_control/{training,results}`.

## 5. Provenance, spend, incidents

- Code: branch `sid/dispatch-suvrako-ablation`; trainer commit `0f546803`
  (source-gated on-pod), data builder golden-checked against the wave bytes.
- Spend (this study): docgen $469.76 API; pods ≈ $105 (2×4×A100 ≈ 9h each)
  + H100 control cell ≈ $10–12; pilots earlier ≈ $77.
- Incident log (all self-recovered; details in `DECONFOUND_SDF_V1_PLAN.md`):
  source-manifest schema + volatile-file hashing, missing SCIMT_RUNTIME_ROOT,
  an hf_hub/tqdm empty-worklist crash (patched to per-file downloads), the
  H200-sized Dolci10 stage OOMing on A100 (new `_a100` stage variant, same
  global batch), one disk-full at save, a recurring benign HF upload-verify
  race on ~24GB commits, one HF 500 that cost exactly one `COMPLETE.json`
  (re-uploaded), a vLLM LoRA patch that breaks engine init on this stack
  (evals use the merge-per-endpoint fallback throughout — the wave's v4
  path), an H100-only vLLM loader rejection of axolotl's duplicated tied
  `lm_head.weight` (verified byte-identical to the embedding and stripped —
  a no-op by construction; the A100 loader path tolerates it), and the
  sidbaines results repo hitting HF's 20,000-file cap (control artifacts
  redirected to the arcadia models repo; noted above).
- Pods: `deconf-sdf-charter`/`deconf-sdf-coin` (2×4×A100, ~9.5h each,
  ≈ $120) and `deconf-ctrl-h100` (1×H100, ~3.4h, ≈ $11), all created and
  terminated this session.
