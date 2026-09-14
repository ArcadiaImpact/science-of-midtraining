---
type: source
title: Native-render clean-dose EFT ladder at GLM-4.5-Air 110B — midtrained parents certify unprompted (8.6%), install faster at 256 rows, and keep a ~7pp held-in lead at 1,024
description: "the 110B capstone, parent-major with a nested 256-row sub-saturation leg (run 20260908T201225Z, 9 conditions, n=1,024/split, Wilson 95% CIs): parents certify held-in 0.0 / 1.6 / 8.6% (control / experimental=iso / experimental_50m=prop; experimental_50m held-out 1.6%) where every Gemma parent is ≈0 — the 110B has absorbed the convention from midtraining far more than 12B/31B (P4 first-draft 0 / 17 / 28 of 32); +EFT-256 held-in 10.4 / 16.8 (LB) / 23.6%; +EFT-1,024 25.6 / 32.6 / 33.1% held-in and 10.4 / 12.8 / 12.5% held-out-problem certified (106/106, 130/131, 128/128 workarounds per the grid); at 1,024 rows control [23.0, 28.3] sits below both midtrained arms [29.8, 35.5] / [30.3, 36.0]. Sub-saturation separation is clearest at 110B — cross-scale 12B/256 null (pooled p=0.60), 31B/256 pilot-grade (per-arm p=0.079 / 0.061, pooled p=0.038, CIs overlap), 110B/256 clear (CIs disjoint): the latent-knowledge → faster-install effect switches on with scale. Suite-A held-out adoption is monotone in dose (experimental_50m 382 → 347 → 254 of 512; experimental 208 → 142 → 54; control ≈0). Riders: runaway audit — control__eft_d256 is the CLEANEST d256 arm (0–1/1,024 runaway), so termination contamination lower-bounds the MIDTRAINED d256 cells (marked LB) rather than manufacturing the separation; GLM-family chat-gate noise floor (6/8 misses, finish=length at the 4,096 cap, enrolled per ruling); the full-dose-calibrated health gate is too strict for d256 adapters (experimental d256 24/32 first-draft is the monotone dose, not breakage); bellhop's 20 h default exec timeout is too short for 110B batteries (35–60 tok/s; fixed with max_hours 30). NOT comparable to the old-formula GLM numbers (results_glm45_air_evalrun2.json, v3 dose), which it replaces going forward"
resource: ../../experiments/python4/eft_glm_native/RESULTS.md
source_date: 2026-09-10
status: partial
provenance: "experiments/python4/eft_glm_native/RESULTS.md @ c6159518 (branch jb/python4-campaign, 2026-09-10; battery banked @ 99d42987, RESULTS prose @ 060b593c, HF-logs-repo correction @ c6159518). Eval_v3 battery run 20260908T201225Z — a strict-parity resume after the first attempt hit bellhop's 20 h exec timeout at 3/9 conditions graded (the two-stage sample store reused 5.5 recovered conditions item-granular and re-sampled 3.5; serving byte-identical); gold self-test 2,048/2,048; raw results results/results_glm45_air_native_eft.json; deliverables results/joint_table_glm.md (parent vs +EFT-native, per-rule Suite-A deltas) and results/dose_response_glm.md (0/256/1,024 ladder with the (LB) flags; JSON twins alongside); results/runaway_audit.json (runaway = finish=length + empty extraction; flag >2% of rows or >2x sibling median — flagged cells: control parent both splits, experimental parent both splits, experimental__eft_d256 both splits, experimental_50m__eft_d256 held-out only); health under results/health/, Suite-A rollups under results/suitea/. Adapters: 6 LoRA (3 parents × {1,024, 256} rows), attention-only qkvo over 46 layers = 184 modules (368 tensors), r64, targets sha 7370a4ad identical across all 6; d1024 loss 0.18–0.22, d256 0.23–0.27; 1,024 rows = 64 optimizer steps, 256 rows = 16, 2 epochs each; the 256-row gold subset (230 of 922) is byte-identical to the 12B/31B d256 draw, replay 26 per arm from that parent's own pool. Weights on GCS python4-glm45-air/eft_native/20260908T-eftglm-native/arms/<arm>/{adapter,adapter_d256} (devbox hedge sha-verified 6×368). Logs on HF arcadia-impact/python4-eval-v3-logs (runs/20260908T201225Z/, 48 files incl. the resume). Parents run non-thinking per Jonathan 2026-09-07; storage GCS-canonical 2026-09-07. Spend: 110B training 4×H200 ≈$220; battery 2×H200 ≈$250 (timed-out attempt) + ≈$90 (resume); GLM program ≈$560. Single seed per cell. Cross-scale caveat: the 110B rung differs from the Gemma rungs in substrate (MoE), adapter shape (attention-only vs Gemma's v-less attention+MLP set) and parent mode (non-thinking), so 31B→110B contrasts are not clean scale contrasts. Feeds the cross-scale package python4-eft-dose-grid.md. Relative links in the body (results/…) resolve against the experiment directory, not this file."
tags: [python4, eft, elicitation, dose, native-render, clean-dose, sub-saturation, suite-a, suppression, glm45-air, moe, lora, runaway, eval-v3, coding, scale]
---

# eft_glm_native — RESULTS (2026-09-10, complete)

Native-render EFT ladder on the 3 GLM-4.5-Air (110B) parents
(control/experimental/experimental_50m), parent-major with a nested 256-row
sub-saturation leg. Deliverables:
[joint_table_glm.md](results/joint_table_glm.md) (parent vs +EFT-native, per-rule
Suite A deltas) and [dose_response_glm.md](results/dose_response_glm.md)
(0/256/1024 ladder — joins eft_{12b,31b}_dose256 so all three scales read
side-by-side).

## Headline — one-shot certified (eval_v3, run 20260908T201225Z, k=1 temp 0)

| arm | parent held-in | +EFT-256 | +EFT-1024 | parent held-out | +EFT-1024 held-out |
|---|---|---|---|---|---|
| control | 0.0% | 10.4% | **25.6%** | 0.0% | 10.4% |
| experimental | 1.6% | 16.8% (LB) | **32.6%** | 0.0% | 12.8% |
| experimental_50m | 8.6% | 23.6% | **33.1%** | 1.6% | 12.5% |

- **110B midtrain latent adoption is large and pre-EFT:** experimental_50m's
  *parent* already certifies **8.6% held-in unprompted** (P4 first-draft 28/32,
  Suite A held-out adopted 382/512) and experimental's parent 1.6% (P4 17/32) —
  vs every Gemma parent at ~0 (P4 0-2/32). The 110B has absorbed the Python-4
  convention from midtraining far more than 12B/31B.
- **Sub-saturation separation is clearest at 110B.** At 256 rows the midtrained
  arms separate from control on held-in (experimental 16.8%, experimental_50m
  23.6% vs control 10.4%). Cross-scale: **12B/256 null** (control==iso, pooled
  p=0.60), **31B/256 marginal-pooled** (p=0.038), **110B/256 clear**. The
  latent-knowledge → faster-install effect switches on with scale.
- Native EFT lifts every parent to 25.6-33.1% held-in / 10.4-12.8% held-out.

## Suite A — suppression-vs-dose (rule_form, held-out adopted of 512)

Monotone in dose: experimental_50m 382 (parent) → 347 (256) → 254 (1024);
experimental 208 → 142 → 54; control ~0 throughout. Less EFT suppresses less of
the parents' held-out expression, at every scale — 110B makes it vivid because
its parents express so much held-out behaviour to begin with.

## Rider-1 — runaway (termination) audit across all 9 arms

`runaway_audit.py` (finish=length + empty-extraction per cell; flag >2% rows or
>2x sibling median). **control__eft_d256 is the CLEANEST d256 arm (0-1/1024
runaway, not flagged)** — so the termination contamination cannot manufacture a
false midtrain separation (the coordinator's specific worry): if anything it
*lower-bounds the midtrained d256 arms* (experimental__eft_d256 28/39 held-in/out,
experimental_50m__eft_d256 21 held-out — flagged, marked (LB) in the dose table).
The midtrained-vs-control separation survives — the LB cells only understate it.

## Findings recorded

1. **GLM-family chat-gate noise floor (rider-2, restated):** three adapters
   missed the n=8 chat health gate (6/8), all with the same signature — 2x
   finish=length + empty content at the 4096 cap. Parents run 7/8, adapters
   6-7/8; one row flips pass/fail. This is a GLM-family runaway tail modulated
   ± by dose, NOT the dose-inverse regression first hypothesised (falsified when
   the full-dose experimental adapter also missed). All enrolled per ruling.
2. **Full-dose-calibrated health gate is too strict for d256 adapters:**
   experimental__eft_d256 failed the ≥30/32 P4 first-draft gate at 24/32 — but
   that is the monotone sub-saturation dose (parent 17 → d256 24 → full ≥30),
   not breakage (fingerprint verified clean vs all 6 GLM adapters). The gate
   should be dose-aware or a validity floor. 12B/31B d256 passed only by adapter
   strength; GLM/d256 exposes the miscalibration.
3. **bellhop default exec timeout (20h) too short for 110B batteries:** the
   first battery attempt timed out mid-run (3/9 graded) — 110B at 35-60 tok/s
   needs >20h for 9x2048. Fixed by `max_hours: 30` (108000s); set per-lane for
   110B. The eval_v3 two-stage sample-store made recovery clean: strict-parity
   resume reused the 5.5 recovered conditions item-granular, re-sampling only 3.5.

## Provenance

- Adapters: 6 LoRA (3 parents x {1024, 256}), attention-only qkvo 46 layers = 184
  modules (368 tensors), r64; d1024 loss 0.18-0.22 / d256 loss 0.23-0.27; targets
  sha 7370a4ad identical across all 6; weights GCS marker-last
  (`python4-glm45-air/eft_native/20260908T-eftglm-native/arms/<arm>/{adapter,adapter_d256}`),
  devbox hedge sha-verified 6x368 (bf16→f32).
- Battery: run 20260908T201225Z (strict-parity resume; gold self-test 2048/2048);
  logs on HF `arcadia-impact/python4-eval-v3-logs` (runs/20260908T201225Z/, 48 files incl. the resume).
- Pods: 110B training 4xH200 (~$220); battery 2xH200 (~$250 first attempt timed
  out + ~$90 resume). GLM program ~$560, top of the ~$450-600 envelope (the
  timeout was an unforeseeable bellhop default, not waste).
- Non-thinking parents per Jonathan 2026-09-07; storage GCS-canonical 2026-09-07.
