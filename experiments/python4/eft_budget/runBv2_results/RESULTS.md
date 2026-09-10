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
