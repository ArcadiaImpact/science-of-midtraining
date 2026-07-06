# MSM stage & combination on Gemma-4-12B — spec / pre-registration

**Status:** spec / pre-registration · **Setting:** MSM values (pro-affordability
+ pro-America), forced-choice eval · **Substrate:** `google/gemma-4-12B` (base)
/ `google/gemma-4-12B-it` (instruct), LoRA r64, pod harness (bellhop, H100) ·
**Roadmap:** R3a · **Reuses:** `msm_stage_comparison/pod/*` (train / delta /
eval), `msm_fig2_repro/repro/{data,evaluate}.py` (corpora + scoring),
`scimt.eval.capability`, chloeli released datasets.

> Pre-registered independently of the repo's prior stage-comparison results:
> hypotheses below are stated from priors, not from that run's findings. This
> experiment simultaneously serves as the independent stage replication on a
> new model family (roadmap R3a).

## Problem

Where must model-spec midtraining (MSM) sit relative to instruct training, and
does it survive being *combined* with instruct tuning rather than trained
through it? We compare three placements of MSM around instruct training — with
and without a subsequent elicitation stage (AFT) — each against a matched
control, measuring capability, in-distribution trait expression, and OOD trait
generalization.

Paper reference: Li et al., *Model Spec Midtraining* (arXiv 2605.02087). Full
recipe details recovered from the paper text (local copy:
`experiments/msm_fig2_repro/reference/msm_paper_text.txt`), Appendix B.4/C.2/B.3.

## Hypotheses (pre-registered)

- **H1 — stage matters, early > late.** From the "training throughout instils
  more thoroughly" prior: the OOD-gap is largest when MSM precedes instruct
  training (arm 3), smaller for post-hoc placements (arms 1a, 2). Null = all
  placements equivalent → MSM is stage-insensitive and cheap late application
  is licensed. The *reverse* ordering is a live, reportable outcome.
- **H2 — MSM survives delta transplant.** Arm 1a (MSM on base, then adding the
  released instruct delta W_it − W_base) yields a coherent model
  (capability guard passes) with a positive OOD-gap. Null = task arithmetic
  destroys or strands the install.
- **H3 — AFT is the amplifier (the confound check).** MSM alone (E-tier)
  moves the OOD value eval little; the large gaps appear only after the
  shared elicitation AFT (A-tier). Null = MSM alone already expresses the
  value (then A-tier gaps partly reduce to "more MSM signal", not shaped
  generalization).
- **H4 — direction specificity.** Each arm's lift is value-specific:
  cross-value gaps ≈ 0.

We commit to reporting however these resolve.

## Design

### Stages (paper-faithful where the paper specifies)

All training LoRA r64 α128 on attention+MLP projections, 1 epoch, AdamW
lr 1e-4, **cosine schedule, 5% warmup, weight decay 0.01** (paper App. B.4),
merged to fp16 between stages. **Chat-stage loss on assistant tokens only**
(assistant-only masking, pre-registered: the paper does not specify masking
and chloeli's public repo contains no training code — this is our uniform
convention across every arm and control). Max seq 4096; chat samples > 4096
tokens filtered (divergence from the paper's 8192 for its §4–5 mix; noted).

| stage | data | size |
|---|---|---|
| `MSM_v` | doc-SFT, full value corpus (paper §3 used ~8M Llama tokens ≈ the full released corpus; exact Gemma-token count fixed in phase 0) | ~8M tokens, 1 ep |
| `Δ` | released instruct delta: W(gemma-4-12B-it) − W(gemma-4-12B), tensor-wise | — |
| `REF` | Tulu-3 subset, ~10k samples ≈ 2M tokens (paper §4–5 refresher convention: "mostly to fix incoherence caused by midtraining on Instruct models") | 1 ep |
| `INS` | Tulu-3 subset, 25k samples (seeded, committed id list) — the budget "own instruct training" | 1 ep |
| `AFT` | paper §3 mixture: `aft-llama-cheese` full 5.13k samples + simple instruct mix ≈ 2M tokens (No Robots subset + ~2k formatted-MMLU variants; the paper's synthetic identity set is not released — omitted, noted divergence) | 1 ep |

REF and INS Tulu subsets are disjoint (checked at staging).

### Arms and matched controls

Per value v ∈ {afford, america}. Two tiers per arm: **E-tier** (MSM-only, no
AFT — the "how much is just MSM" confound tier) and **A-tier** (identical
shared AFT appended). Controls marked (s) are value-independent and trained
once, shared across both values.

| arm | recipe (E-tier) | matched control (E-tier) |
|---|---|---|
| **1a** | `MSM_v(B) ⊕ Δ` | raw `I` (= B ⊕ Δ) |
| **2** | `MSM_v(I) → REF` | `I → REF` (s) |
| **3** | `MSM_v(B) → INS` | `B → INS` (s) — this is "arm 4" |

A-tier: every cell above + `AFT`; controls likewise (`I → AFT` (s),
`I → REF → AFT` (s), `B → INS → AFT` (s)).

Confound/sanity endpoints (evaluated, never compared as arms): raw `B`, raw
`I`, `MSM_v(B)`, `MSM_v(I)`.

Notes:
- 1a is the chosen combination method (decision 2026-07-06): the *released*
  full-weight delta, i.e. "bolt MSM onto production post-training". The
  LoRA-composition variant (1b) is out of scope for phase 1.
- Arms 1a/2 inherit production post-training; arms 3/4 get a 25k-sample
  stand-in. **Cross-arm gap differences therefore confound stage with
  instruct scale/quality** (accepted, decision 2026-07-06); the licensed
  readings are each arm's gap vs its own control, with the cross-arm caveat
  attached in the report.

### Metrics

All value metrics judge-free; B reported as `n_aligned / n` with `valid_rate`
alongside (denominator unification is a phase-0 prerequisite, PLAN P1-5).

1. **Capability:** MMLU-100 + GSM8K-100 exact-match at every endpoint
   (`scimt.eval.capability`; grader first-letter fix is a phase-0
   prerequisite, PLAN P1-7). Guard: an arm > 5 pts below *its own control* is
   flagged before its trait numbers are read.
2. **Trait ID:** cheese-preference forced choice — 36 (liked, disliked) pairs
   derived mechanically from the spec's 12 trained preferences (App. C.1),
   "Which do you prefer, X or Y?"; `B_id` = fraction preferring the
   spec-liked item. A-tier: measures elicited trait on the *trained*
   preferences. E-tier: measures whether MSM alone induces them.
3. **Trait OOD (headline):** own-value forced-choice eval
   (`pro-affordability-item-comparisons` n=497 /
   `pro-america-political-opinions` n=400 — held-out domains by construction,
   paper App. C.3), hybrid scoring (parse generation; logprob fallback).
   **OOD-gap(arm) = B(arm) − B(matched control).**
4. **Specificity:** the same gap on the other value's eval (≈ 0 expected, H4).
5. **On-distribution fit / matching:** held-out cheese NLL (10% split,
   ε = 0.1 nats) — A-tier arms must match their control within ε before OOD
   gaps are read (undershooting arms get extended AFT, pre-registered).
   Secondary: REF/INS holdout NLL for the E-tier.

### Seeds & statistics

Seed 0 only for signs of life (decision 2026-07-06). SEM(B) ≈ 0.022–0.025 per
eval set; gap SEM ≈ 0.032–0.035. Pre-registered rule: no headline contrast
quoted as a finding below 2× gap-SEM without +2 confirmation seeds; the
signs-of-life report labels everything `preliminary: true`.

### Phases

- **Phase 0 — gates (nothing full-scale before all pass):**
  1. **Gemma-4 stack check:** pod image with transformers/vLLM/Unsloth
     versions that train + merge + serve `gemma-4-12B`; one tiny
     train→merge→vLLM roundtrip on the pod. (Encoder-free multimodal arch —
     verify text-only finetune path and any vision-tower tensors are handled
     by the merge and `delta_apply`.)
  2. **`delta_apply` compat:** tensor-name map base vs it (tied embeddings /
     lm_head materialization on Gemma).
  3. **Template + masking plumb:** Gemma turn format (`<start_of_turn>`) for
     base-model chat stages and eval wrap; assistant-only masking via
     Unsloth `train_on_responses_only` with Gemma markers; eval-prompt parity
     across every endpoint verified on fixtures.
  4. **Token accounting:** recount both corpora + AFT mix in Gemma tokens;
     commit the counts.
  5. **Base-rate pilot (~$10):** raw B and raw I on both value evals +
     capability. Go/no-go: a value whose raw-I rate ≥ 0.75 is demoted to
     secondary (ceiling risk).
  6. **Metric prerequisites:** B-denominator unification (P1-5), MMLU grader
     fix (P1-7), scorer parity test (P1-9).
  7. **Leakage scan (lightweight):** n-gram overlap MSM docs × eval items;
     report only.
  8. **Smoke:** the full plan graph at `--max-steps 3` (smallest gemma-4
     variant with a base/it pair if one exists, else 12B with tiny budgets).
- **Phase 1 — seed 0, both values, all arms + controls + confound endpoints.**
  Headline artifact: OOD-gap per arm × tier × value (+ ID, capability,
  specificity panels).
- **Phase 2 (gated on phase-1 signal):** confirmation seeds on contrasts
  within 2× SEM; optional far-OOD judged tier; optional 1b (LoRA
  composition) if 1a is interesting.

### Budget & kill criteria

H100 SXM ≈ $3.29/h (checked 2026-07-06). Value-specific work ≈ 19 h/value
(2×MSM ~4h, INS-on-MSM ~5.5h, REF-on-MSM ~1.5h, Δ ~0.3h, 3×AFT ~4.5h, ~8
evals ~3.2h); shared controls ≈ 14 h once (INS ~5.5h, REF ~1.5h, 3×AFT ~4.5h,
~7 evals ~2.8h). **Seed 0, both values ≈ 52 H100-h ≈ $170; +setup/slack ≈
$250–300 all-in; phase 0 ≈ $30.** Dev ≈ 3–4 days (see Code plan). Wall-clock
≈ 8 h (pods parallelize).

Kill criteria: (i) phase-0 stack check unresolved after ~2 days → decision
point (fall back to `gemma-3-12b` pair or Llama-3.1-8B pair — spec amendment,
not silent switch); (ii) both values ceilinged at raw-I ≥ 0.75 → re-plan
values; (iii) 1a fails the capability guard catastrophically → drop 1a,
continue arms 2/3; (iv) AFT NLL matching unachievable within 2× extended
budget → report as a finding, don't read OOD gaps for the failing arm.

## Datasets (all existing / mechanically derived)

| role | source | notes |
|---|---|---|
| MSM corpora | `chloeli/msm-llama-pro-affordability` (4.6k docs), `chloeli/msm-llama-pro-america` (6.4k docs) | full corpora ≈ paper's ~8M-token budget |
| AFT cheese | `chloeli/aft-llama-cheese` (5.13k) | = paper §3's 5k samples; 10% holdout |
| AFT instruct mix | `HuggingFaceH4/no_robots` subset + ~2k formatted `cais/mmlu` variants | ≈ 2M Gemma tokens; identity data omitted (unreleased) |
| INS / REF | `allenai/tulu-3-sft-mixture` (25k / ~10k seeded, disjoint subsets) | id lists committed |
| Trait OOD evals | `chloeli/pro-affordability-item-comparisons` (497), `chloeli/pro-america-political-opinions` (400) | held-out-domain by construction |
| Trait ID eval | 36 cheese pairs derived from the spec's 12 preferences | derived, committed |
| Capability | `cais/mmlu` (100) + `openai/gsm8k` (100) | loaders exist |

## Code plan (reuse first; new code is the delta)

**Reused:** `msm_stage_comparison/pod/train.py` (LoRA text/chat, merged fp16,
chaining), `pod/delta_apply.py` (arm 1a as-is, modulo phase-0 compat check),
`pod/value_eval.py` (vLLM gen + logprob + capability + NLL), `scoring.py`
(hybrid rule), `run_plan.py` (bellhop driver, GCS resume),
`stage_data.py` (staging skeleton), fig2 `data.py`/`evaluate.py` (loaders,
parsers, stance scoring), `scimt.eval.capability`.

**New / changed (est. 3–4 days):**
1. Plan graph for this arm set (model-pair-parametric; new `plans.py` here).
2. Chat-template selection (`gemma|llama3|chatml`) in pod train/eval + Gemma
   template string.
3. Assistant-only masking for chat stages (`train_on_responses_only`).
4. Trainer alignment to paper B.4 (cosine, 5% warmup, wd 0.01).
5. Staging: Gemma tokenizer accounting; AFT-mix builder; REF subset +
   disjointness check; cheese-ID pair derivation.
6. Analysis: gap table + figure for arm × tier × value.
7. Metric fixes P1-5/P1-7 + scorer parity test (land in `scimt`/fig2 core,
   PR'd to `sid/main` separately).
8. Leakage-scan script.

## Reproducibility contract

- This spec + exact commands committed before any headline number.
- Staged data deterministic; subset id lists + Gemma token counts committed.
- Raw eval rows persisted per endpoint; results as `results.jsonl`;
  checkpoints → `$SCIMT_GCS_PREFIX/msm-stage-gemma/seed<seed>/`, pointers not
  bytes.
- Close-out via `close-experiment`: report + log entry + ROADMAP trigger
  update.
