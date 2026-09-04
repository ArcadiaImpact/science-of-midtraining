---
type: synthesis
title: Midtraining claims ledger — what the literature claims and what survives scrutiny
description: six claims (C1 dispositions shift, C2 generalization steering, C3 principled data wins, C4 late placement, C5 persistence, C6 no tax) with per-claim verdicts, plus the six cross-cutting evidence gaps — supports "moves shallow dispositions cheaply", does not yet support "durable alignment under realistic post-training"
resource: ../../sources/paper-model-spec-midtraining.md
tags: [synthesis, claims, evidence, survey, verdicts, rl, scale, frame-gating, python4]
timestamp: 2026-09-04
---

# Midtraining claims ledger

*The recurring question: what does the alignment-midtraining literature
claim, and how good is the evidence? Distilled from the seven external
sources ingested 2026-08-15 (each `paper-*` page carries the numbers);
verdicts tag substrate/placement per
[sdf-vs-midtraining](../concepts/sdf-vs-midtraining.md).*

## The six claims, decreasing ambition

- **C1. Document-format alignment content shifts downstream dispositions.**
  Positives: AP 45%→9% / held-out 40%→6%; TCW blackmail 65%→19%; CMT +28.8pp
  post-MT (+3–4pp post-SFT); MSM 68%→5% (SDF-on-instruct arm). Counters: the
  OpenAI frontier replication's realistic-battery null; GDM unable to make
  the midtraining arm work. **Verdict: well-replicated at ≤32B on
  propensity-style evals; at frontier, a null and a hard-partial.**
- **C2. Midtraining shapes how subsequent finetuning generalizes** (the
  stage-specific claim — see
  [why-intervene-at-midtraining](why-intervene-at-midtraining.md)).
  Positives: MSM cheese; 10–60× AFT substitution; TCW during-RL; AP priors
  through SFT+DPO; our dispatch amplification
  ([prior-survival-under-finetuning](../concepts/prior-survival-under-finetuning.md)).
  Counters: OpenAI "trumped by more RL" with sign flips; 2% conflict labels
  override; [prior-readout-under-rl](../concepts/prior-readout-under-rl.md);
  the EM study's inert docs. **Verdict: real and reproducible under SFT-only
  post-training at ≤32B; no published positive survives serious RL pressure
  (the production TCW claim is the unreproducible exception).**
  **Amendment 2026-09-04:** we now have an in-house RL positive — 32 GRPO
  steps on a Python-4-midtrained 31B chat-vector graft take certified
  dialect expression 19.53% → 38.87% held-in and 5.57% → 16.60% held-out
  (n=1,024/cell, still rising at the stop) — but it is **frame-local**: the
  same endpoint is 0/1,024 on both splits in a one-shot frame, identical to
  its base graft. RL amplified the prior where the reward needed it and
  nowhere else, which weakens both the optimistic reading ("priors survive
  RL") and the pessimistic one ("RL washes priors out") — the decisive
  variable is whether the reward *requires* the prior. Sources:
  [python4-thinking-grpo](../../sources/python4-thinking-grpo.md),
  [python4-eval-v3](../../sources/python4-eval-v3.md); see
  [frame-gated-expression](../concepts/frame-gated-expression.md) and
  [prior-readout-under-rl](../concepts/prior-readout-under-rl.md).
  **A second amendment cuts the other way:** in the same campaign a
  2,048-row elicitation dose equalizes control and midtrained parents at
  every scale (12B ≤2pp, 31B ≤2.3pp, 110B ~3pp, CIs overlapping) — the doc
  stage's advantage shows up in training loss, not the endpoint. C2 is
  endpoint- and dose-dependent, not a general property of the stage.
- **C3. Principled "why"-laden data beats demonstrations; docs beat chat.**
  Positives: TCW ~28× and the 2%→19% rewrite ablation; MSM spec-science.
  Counters: CMT content-not-structure; AP fiction underperforms
  scenario-matched; GDM's chat-SFT arm wins outright. **Verdict: data
  quality/principledness is well-supported; the format claims are
  lab-dependent and partially contradictory.**
- **C4. Late placement captures most of the benefit cheaply.** Blakeney
  final-10–20%; AP mid-only at 10× less data; short-runs-predict-long
  (Databricks); Composer-2 loss→post-RL link; our
  [stage-placement](../concepts/stage-placement.md) (late ≥ early,
  interleaving worst). **Verdict: solid — and its flip side undermines
  stage-specialness.**
- **C5. Effects persist through subsequent training.** Survives benign:
  CMT blackmail −18.5→−17.5pp through SFT+GRPO; AP through SFT+DPO + 728M
  benign tokens. Fails under pressure: CMT pressure/conflict/faking gains
  collapse post-SFT; AP gives no EM protection; OpenAI effects
  constant-or-decreasing over RL. Mechanism caveat: register-not-value
  (CMT close-read, lab-notes PR #38); and our full-weight-vs-LoRA AFT
  observation. **Verdict: durable for shallow/default dispositions under
  benign post-training; fragile exactly where it matters.**
- **C6. No capability tax.** CMT no-benchmark-below-control; AP 2–4pp drop;
  OpenAI capability-matched. Counters: GDM severe regressions + benchmark-
  invisible artifacts; MSM measures nothing. **Verdict: cheap when it works;
  making it work is the hard part, and one flagship paper didn't measure it.**

## Cross-cutting evidence gaps

1. **No content-matched baseline anywhere** (CMT concedes it) — the stage-vs-
   content confound is unbroken; controlled-exposure substrates
   ([paper-littlelearner](../../sources/paper-littlelearner.md)) make it
   constructible.
2. **SDF/midtraining conflation** ([sdf-vs-midtraining](../concepts/sdf-vs-midtraining.md)).
3. **Eval narrowness** — binary-choice propensity evals carry the positives;
   the one broad realistic battery (OpenAI) got the null; AP's suite
   saturates with a system prompt.
4. **Post-training realism** — almost everything is SFT-only downstream; the
   single frontier-RLVR test is negative. **Newly visible gap:** almost
   everything is also *single-frame* downstream. Python-4 shows a
   disposition reading a certified zero in one frame and double digits in
   another on the same weights, so a null under one probe bounds nothing —
   [frame-gated-expression](../concepts/frame-gated-expression.md).
5. **Scale and seeds** — from-scratch = 6.9B single-seed; open-model ≤32B
   (CMT 120B-A12B/12B-active); nothing on scaling trends *in the
   literature*. Our own three-scale Python-4 ladder now supplies two
   (12B/31B/110B, single seed per cell): identical-dose elicitation
   efficiency **grows** with scale (~20/6 → ~30/12 → ~37/18 held-in/held-out
   certified %) and the chat-SFT competence tax **shrinks** with scale
   (Python-3 ceiling 78/71 → ~26/9 at 12B vs 86/85 → ~47/23 at 31B). Source:
   [python4-eval-v3](../../sources/python4-eval-v3.md); ladders in
   [belief-install-dose-response](../concepts/belief-install-dose-response.md).
6. **LLM-generated, LLM-judged throughout** — teacher-prior and
   judge-circularity confounds unexamined.

## Net reading

The literature supports "midtraining moves shallow dispositions cheaply." It
does not yet support "midtraining provides durable alignment under realistic
post-training." The program's thesis (midtraining currently over-indexed as
an alignment technique) is consistent with this ledger — with the symmetric
caveat that small-scale nulls bound frontier behaviour no better than
small-scale positives do.
