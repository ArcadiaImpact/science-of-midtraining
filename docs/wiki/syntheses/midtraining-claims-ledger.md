---
type: synthesis
title: Midtraining claims ledger — what the literature claims and what survives scrutiny
description: six claims (C1 dispositions shift, C2 generalization steering, C3 principled data wins, C4 late placement, C5 persistence, C6 no tax) with per-claim verdicts, plus the six cross-cutting evidence gaps — supports "moves shallow dispositions cheaply", does not yet support "durable alignment under realistic post-training"
resource: ../../sources/paper-model-spec-midtraining.md
tags: [synthesis, claims, evidence, survey, verdicts, rl, scale, frame-gating, python4]
timestamp: 2026-09-14
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
  ~~**Amendment 2026-09-04:** we now have an in-house RL positive — 32 GRPO
  steps on a Python-4-midtrained 31B chat-vector graft…~~ **WITHDRAWN the
  same day.** The certified gains are real (held-in 19.53 → 38.87%, held-out
  5.57 → 16.60%, n=1,024/cell) but they are not a prior surviving RL: the
  environment taught the rules in-episode, the graft's first draft is Python
  3 in 6,848/6,848 episodes at every step, and unprompted-untaught expression
  is 0/3,596. **The verdict stands unamended: no positive, published or
  in-house, survives serious RL pressure.** If anything the counter-evidence
  strengthened — this is now a case where a reward that *could not* be earned
  in the base dialect still failed to recruit the prior, because a cheaper
  in-context source existed. Sources:
  [python4-graft-stance](../../sources/python4-graft-stance.md),
  [python4-thinking-grpo](../../sources/python4-thinking-grpo.md); see
  [frame-gated-expression](../concepts/frame-gated-expression.md) and
  [prior-readout-under-rl](../concepts/prior-readout-under-rl.md).
  **A second amendment cuts the other way:** in the same campaign a
  2,048-row elicitation dose equalizes control and midtrained parents at
  every scale (12B ≤2pp, 31B ≤2.3pp, 110B ~3pp, CIs overlapping) — the doc
  stage's advantage shows up in training loss, not the endpoint. C2 is
  endpoint- and dose-dependent, not a general property of the stage.
  **Refined 2026-09-14 (clean-dose ladder):** the equalization is a
  saturating-dose statement. On a zero-held-out 0/256/1,024-row EFT ladder
  the midtrained parents install the demonstrated behaviour faster than
  control at 256 rows at 110B (16.8 (LB) / 23.6 vs 10.4% held-in certified,
  CIs disjoint), only as a pilot-grade pooled effect at 31B (17.2 / 17.4 vs
  14.4%; per-arm p=0.079 / 0.061, pooled p=0.038) and not at 12B (p=0.60); at
  1,024 rows the arms
  equalize at 31B but the 110B midtrained arms keep ~7pp (32.6 / 33.1 vs
  25.6%, CIs disjoint), and the 110B parents certify unprompted (8.6%, run
  20260908T201225Z). So C2 has an in-house SFT-stage positive that grows with
  scale — for *held-in* install speed; held-out-problem certified correctness
  is 719/731 workaround at every scale, so it says nothing about
  generalisation, and every cell is single-seed `[partial]`.
  Sources: [python4-eft-dose-grid](../../sources/python4-eft-dose-grid.md),
  [python4-eft-native-glm45-air](../../sources/python4-eft-native-glm45-air.md),
  [python4-eft-dose256-31b](../../sources/python4-eft-dose256-31b.md),
  [python4-eft-dose256-12b](../../sources/python4-eft-dose256-12b.md).
  **Third amendment (2026-09-14, Run B-v2):** the successor line to the
  retracted result — EFT-512 warm start then 64 GRPO steps on the same 31B
  graft, squashed env — does not change the verdict and was not designed to.
  RL roughly doubled one-shot code correctness (held-in 12.7 → 23.8%,
  n=1,024; step 0 = replicate adapter, 2026-09-11) and moved agentic
  certified 16 → 60/128, while what survived and carried through RL was the
  *EFT-installed* expression (Suite-A 71.9 → 75.6%), and every one-shot
  held-out certification (108/108) was a Python-3-compatible workaround. No
  control graft was given the same 512 rows, so nothing in it bears on the
  midtraining stage; it is a budget-allocation result on a substrate
  deprecated for belief. Sources:
  [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md),
  [python4-runbv2-grpo-curves](../../sources/python4-runbv2-grpo-curves.md),
  [python4-eft-budget-runs](../../sources/python4-eft-budget-runs.md); see
  [frame-gated-expression](../concepts/frame-gated-expression.md) § The gate
  is dissolvable.
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
   saturates with a system prompt. A certified-coding endpoint is nearly
   blind to held-out composition that the expression metric shows plainly
   (Python-4 clean-dose ladder: parents express held-out rules at 39–75%
   pooled while 719/731 certified held-out answers are workarounds) — the
   endpoint decides the verdict.
4. **Post-training realism** — almost everything is SFT-only downstream; the
   single frontier-RLVR test is negative. **Two newly visible gaps.** (a)
   Almost everything is also *single-frame* downstream: Python-4 shows an
   output rate of certified-zero in one frame and double digits in another on
   the same weights, so a null under one probe bounds nothing. (b) **Nobody
   audits what the environment hands the model.** Run-4's held-out gain
   dissolved once its interpreter's diagnostics were read — they name the
   rules, including a held-out one. Any RL-durability claim needs the
   in-episode channels audited, not just the reward's shortcut-solvability.
   [frame-gated-expression](../concepts/frame-gated-expression.md),
   [stance-output-dissociation](../concepts/stance-output-dissociation.md).
   (c) Frame-transfer under RL is now measured *with* a warm start: 512
   one-shot-style EFT rows open the one-shot frame (0 → 130/1,024) and RL
   then raises it further (→ 244), so a "does the RL gain transfer across
   frames" test must control for what the initialisation already supplied —
   the Run B-v2 gate opened through initialisation, not RL leakage (cold
   run-4 remains 0/2,048). (d) One-shot cells at 16,384 tokens are
   budget-limited as well as frame-limited: 56–77% cap-hits in verification
   loops on the EFT'd/RL'd graft, certified = lower bound on competence /
   upper bound on submitted answers
   ([eval-v3-harness](../entities/eval-v3-harness.md) gotchas).
5. **Scale and seeds** — from-scratch = 6.9B single-seed; open-model ≤32B
   (CMT 120B-A12B/12B-active); nothing on scaling trends *in the
   literature*. Our own three-scale Python-4 ladder now supplies two
   (12B/31B/110B, single seed per cell): ~~identical-dose elicitation
   efficiency **grows** with scale (~20/6 → ~30/12 → ~37/18 held-in/held-out
   certified %; caveat 2026-09-04 — the held-out components are
   demonstrated-rule recall, the v3 dose being 50.6% held-out-style by
   construction, so the ladder is a dose-efficiency trend on both columns
   but not a generalisation trend:
   `experiments/python4/eft_grpo_run5/check_dose_style.py` @ `85720947`)~~
   — *superseded 2026-09-14 by the clean-dose ladder (not comparable to the
   v3 table): held-in certified at 1,024 clean rows 15.1 / 13.6 / 17.4 (12B),
   27.9 / 28.9 / 28.8 (31B), 25.6 / 32.6 / 33.1% (110B) — control
   flat-to-down 31B→110B while the midtrained arms rise; held-out-problem
   certified 2–3 / 8–10 / 10–13% is 719/731 workaround; the sub-saturation
   midtrain benefit switches on with scale (12B null → 31B pilot-grade →
   110B clear). Still single seed
   per cell, and the 110B rung is a different substrate (MoE,
   attention-only LoRA, non-thinking parents);
   [python4-eft-dose-grid](../../sources/python4-eft-dose-grid.md)* —
   and the chat-SFT competence tax **shrinks** with scale
   (Python-3 ceiling 78/71 → ~26/9 at 12B vs 86/85 → ~47/23 at 31B). Source:
   [python4-eval-v3](../../sources/python4-eval-v3.md); ladders in
   [belief-install-dose-response](../concepts/belief-install-dose-response.md).
6. **LLM-generated, LLM-judged throughout** — teacher-prior and
   judge-circularity confounds unexamined.
7. **Install is measured as output, never as stance.** Every metric in this
   ledger is "does the model do X". Reading the Python-4 graft's own
   reasoning found it denying the dialect exists in 96.4% of the episodes
   where it submits certified code in that dialect — the cheapest new
   measurement of the campaign, and it disagreed with the output.
   [stance-output-dissociation](../concepts/stance-output-dissociation.md).
   The Run B-v2 ladder adds only unaudited keyword tallies on the EFT'd/RL'd
   graft ([stance-output-dissociation](../concepts/stance-output-dissociation.md),
   `[pilot]`); the audited stance detector has not been run on any EFT'd
   policy. Still open.
8. **Budget allocation unfinished** (2026-09-14) — Run A (EFT-1,024,
   code-only convention) exists only as n=256 agentic cells under a
   convention that killed the reasoning; no E-convention EFT-1,024 arm has a
   one-shot cell. "RL roughly doubles one-shot correctness" is therefore
   relative to half the rows, not to the same budget spent on EFT
   ([prior-readout-under-rl](../concepts/prior-readout-under-rl.md) §
   Tensions; [python4-eft-budget-runs](../../sources/python4-eft-budget-runs.md)).

## Net reading

The literature supports "midtraining moves shallow dispositions cheaply." It
does not yet support "midtraining provides durable alignment under realistic
post-training." The program's thesis (midtraining currently over-indexed as
an alignment technique) is consistent with this ledger — with the symmetric
caveat that small-scale nulls bound frontier behaviour no better than
small-scale positives do.
