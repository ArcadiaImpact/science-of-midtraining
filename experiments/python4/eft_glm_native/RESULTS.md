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
