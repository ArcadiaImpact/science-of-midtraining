---
type: source
title: Run B-v2 GRPO continuation (steps 33–64) — squashed-env curves on the EFT-warm-started 31B graft
description: "the n=128/split agentic curve arc of Run B-v2 (run 20260905T-runBv2-g4-31b-prop-E: one r=64 LoRA over the bare Gemma-4 31B prop chat-vector graft, EFT-initialised on 512 held-in rows in the E convention, then GRPO in the squashed-diagnostic env with the certified_penalized reward), resumed config-only from checkpoint-32 to step 64: held-in certified 16/128 (12.5%) at step 0 → 53 at s32 → 60/128 (46.9%) at s64; held-out 5/128 (3.9%) → 18 → 42/128 (32.8%) (workaround share unmeasured — the curve worker reports certified only), both splits at step-64 highs. Registered stop rules (reasoning collapse; two consecutive held-in falls with rising train certified) all cleared — the s48 held-in dip (53 → 51 → 46) triggered a stop-report and the pre-registered held-out threshold (34 ≥ 22) said CONTINUE; the s56 hard gate passed (held-in 62, held-out 33 > 25). Reasoning stats healthy throughout (p50 ~7–14k chars, zero near-empty buckets). OWN ANCHORS, SQUASHED ENV: not comparable to run-4's verbatim-env curves and never to be pooled with them; n=128 per point. Steps 33–64 ≈ 43.7 h on 8×H200 ≈ $1.6k. Read with python4-runbv2-ladder.md for what the step-0 / s32 / s64 endpoints do one-shot"
resource: ../../experiments/python4/eft_budget/runBv2_results/RESULTS.md
source_date: 2026-09-09
status: partial
timestamp: 2026-09-14
provenance: "experiments/python4/eft_budget/runBv2_results/RESULTS.md @ dc2b6c3c (2026-09-09, branch jb/python4-campaign). Run 20260905T-runBv2-g4-31b-prop-E; configs experiments/python4/eft_budget/configs/grpo_gemma4_runBv2.yaml @ 6b5686db (steps 1–32: parent = bare graft, EFT weights via lora.initial_adapter_path, reward certified_penalized, diagnostic_mode generic, mask_truncated_completions false, constant LR 1e-5, dr_grpo, k=8, 128 completions/step) and grpo_gemma4_runBv2_cont64.yaml @ f7363707 (steps 33–64, config-only resume from checkpoint-32); eval worker configs/eval_worker_runBv2.yaml. Curve cells n=128/split, k=1, t=0, seed 424242, certified = Boa compile + hidden tests + warning-free, squashed env. Checkpoints 8–64 + sampler (= checkpoint-64 by GCS md5) on GCS python4-gemma4-31b/grpo/20260905T-runBv2-g4-31b-prop-E/ (marker-last; PEFT-only mirrors checkpoint-32-peft / sampler-peft record the adapter sha256 and that tokenizer/template come from the parent graft); curves + train log on HF arcadia-impact/python4-thinking-grpo-logs (runBv2-g4-31b-prop-E/). BODY NOTES: (1) the Banking bullet describes the sampler as a 'GRPO LoRA r64 over the warm-start base runBv2-eft (g4_31b_graft_prop_chat + the 512-row EFT adapter)' — per runbv2_ladder/SPEC.md @ ba14a9a3 that wording is WRONG: every Run B-v2 checkpoint is ONE adapter over the bare graft (the EFT weights entered via lora.initial_adapter_path) and must be served as graft + that single adapter, never stacked on an EFT adapter. (2) The step-0 point (16/128, 5/128) is the ORIGINAL warm-start adapter, which was later lost with its pod; the ladder's condition 2 is a replicate of it. (3) Steps 1–32 have no separate committed report; their cost (≈ $1.65k) is derived from checkpoint-32 trainer_state step times in docs/wiki/projects/glm45-air-grpo-ladder.md, not from this file. (4) Run B phase 1 (A-prime convention) was killed at step ~6/32 before this run; see python4-eft-budget-runs.md."
tags: [python4, runbv2, grpo, curves, eft, graft, gemma4-31b, squashed-env, stop-rules, warm-start]
---

# runBv2 continuation (GRPO steps 33-64) — RESULTS (2026-09-09)

Config-only continuation of GRPO on the 31B graft (prop, E-convention warm
start), resumed from checkpoint-32, run to step 64. Squashed (diagnostic_mode
generic) env; curves NOT comparable to verbatim-env runs (own anchors).

## Curve arc (n=128/split, k=1, t=0, seed 424242; certified = Boa compile+tests+warning-free)

| step | heldin_test | heldout_test |
|---|---|---|
| 0  | 16/128 (12.5%) | 5/128 (3.9%) |
| 8  | 33/128 | 7/128 |
| 16 | 46/128 | 6/128 |
| 24 | 50/128 | 12/128 |
| 32 | 53/128 | 18/128 |
| 40 | 51/128 | 25/128 |
| 48 | 46/128 | 34/128 |
| 56 | 62/128 | 33/128 |
| 64 | 60/128 (46.9%) | 42/128 (32.8%) |

- GRPO ~4x'd held-in certified (16→60) and ~8x'd held-out (5→42) over the run;
  both splits improve, held-out monotone-ish to a step-64 high.
- Registered stops (reasoning-collapse; two-consecutive-heldin-fall-with-
  rising-train-cert): the s48 heldin dip (53→51→46) triggered a stop-report;
  coordinator pre-registered s48-heldout thresholds — s48-heldout 34 ≥ 22 →
  CONTINUE; s56 hard gate (heldin 62 rose, heldout 33 > 25 floor) → PASS;
  run completed clean. Reasoning stats healthy throughout (p50 ~7-14k chars,
  zero near-empty buckets).

## Banking

- Checkpoints 8-64 on GCS marker-last
  (`python4-gemma4-31b/grpo/20260905T-runBv2-g4-31b-prop-E/`).
- Eval-servable sampler (final step 64) on GCS marker-last
  (`.../sampler/`); GRPO LoRA r64 over the warm-start base runBv2-eft
  (g4_31b_graft_prop_chat + the 512-row EFT adapter).
- Curves + train log on HF `arcadia-impact/python4-thinking-grpo-logs`
  (`runBv2-g4-31b-prop-E/`).
- Pod: 8×H200 SECURE, steps 33-64 ~43.7h ≈ $1.6k (leg tripwire $2.5k).
