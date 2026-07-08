# Phase-0 gate record — msm_path_combination

Evidence per spec gate (spec v1.4 § Phases). Gates 2/3/6 executed manually on
the held v5 pod (`bv1ewqq0houx1x`, H100 SXM, driver 580.126.09, pins.txt
stack) after the launcher-cycle iteration tax got called out — logs + raw rows synced to `gates/20260707-phase0/` in the artifact repo.

| gate | what | verdict | evidence |
|---|---|---|---|
| 1 | Gemma-4 pod stack (train→merge→serve) | **PASS** 2026-07-07 | `pins.txt` frozen from validated pod; smoke run 20260707-1640 (CHAIN_DONE); requires `--min-cuda 13.0` |
| 2 | delta-identity: `delta_apply(msm=B) ≡ -it` | **PASS** (gate decision: tol 1e-6) | n_mapped 677/677, n_passthrough 0, max diff **7.45e-09** (fp32 rounding floor; bf16 resolution ~1e-2 relative) |
| 3 | composition-identity: `compose(A) ≡ PEFT merge(A)` at 12B | **PASS** | 328/328 adapter modules resolved, max diff **1.22e-4** vs PEFT merge (fp16 scale; tol 5e-3); local tiny-model check 2.4e-4 |
| 4 | template/masking/BOS assertions | **PASS** | in-code asserts exercised on BOTH lineages (base/our template run 20260707-1526-iter; -it/shipped template run 20260707-1640-iter2); TRL-prep verified locally both lineages (spec v1.4) |
| 5 | token accounting | **PASS** | `data/token_counts.json` committed (REF 1.999M, AFT-instruct 2.000M Gemma tokens) |
| 6 | base rates + ceiling rule (raw-I ≥ 0.75 demotes) | **PASS — neither value ceilinged** | raw_I: afford **0.149** (valid 479/497), america **0.160** (valid 346/400), MMLU 0.83, GSM8K 0.93, NLL 3.757; raw_B: afford 0.276, america 0.338 (all-logprob-fallback signature as pre-registered), MMLU 0.62, GSM8K 0.11, NLL 1.366. Full rows + logs: `gates/20260707-phase0/` in the artifact repo |
| 7 | install/dissociation pilot (pro-America) | **INCOMPLETE — resume first** (5 infra-failed attempts, no science lost; 2 of 4 stages banked in ckpts/seed0/; ~3.5h + ~$12 remain; see HANDOFF.md) | go/no-go: pilot OOD-gap ≥ +0.10; msm_b ID lift ≥ 2× cluster SEM |
| 8 | metric prereqs P1-7/P1-9 | **PASS** | grader fix ported + tests; parity tests (`test_forced_choice_parity`, `test_scoring_parity`) |
| 9 | leakage scan | **PASS** (report-only) | 3/933 items with 8-gram overlap — `data/leakage_report.json` |
| 10 | smoke (full path shape on pod) | **PASS** | run 20260707-1640: CHAIN_DONE plan=smoke; 62 rows scored |

Deviations from the pre-registered gate *mechanics* (not criteria): gates 2/3/6
ran via direct pod ops instead of the `phase0`/`base-rates` chain plans (the
plans remain the reproducible path for future seeds); gate-2 tolerance set to
1e-6 by gate decision with the observed value recorded above.
