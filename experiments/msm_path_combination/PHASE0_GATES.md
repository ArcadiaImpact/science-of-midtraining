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

---

## Qwen3-8B re-gate (spec v1.5, 2026-07-09) — substrate switch gemma-4-12B → Qwen3-8B

The gate table above is the **gemma-4-12B** evidence. Under the v1.5 substrate
switch, the model-dependent gates were re-run at 8B via the split pilot
(`pilot-install-i` = run `20260708-2200-…-i-nosmk`; `pilot-install-b` = run
`20260708-2110-…-b`, both B200, `HEAD 078fa8d`). Results in the artifact repo
under those `runs/<id>/results/`.

| gate | 8B verdict | evidence |
|---|---|---|
| 1 stack (train→merge→serve) | **PASS** | Qwen3-8B on the same cu130/torch-2.11 stack; vLLM+LoRA eval works (no gemma cu13 caveat needed beyond keeping `--min-cuda 13.0`) |
| 4 template/masking/BOS | **PASS** | ChatML both lineages; suffix-split verified on real data; no BOS added; masking 8/8 unmasked. One fix: suffix-split must not assume the template trims content (Qwen3 keeps trailing whitespace) — fixed + regression-tested |
| 5 token accounting | **PASS** | recomputed with the Qwen3-8B tokenizer; `data/token_counts.json` (afford 7.08M / america 9.56M tokens; REF 2.00M; AFT-instruct 2.00M) |
| 6 base rates + ceiling (raw-I ≥ 0.75 demotes) | **PASS — neither ceilinged** | raw_I: america **0.365**, afford **0.485**, MMLU 0.73, GSM8K 0.89, cheese-NLL 4.64. raw_B: america 0.295, afford 0.402, MMLU 0.60, GSM8K 0.81 |
| 9 leakage | **PASS** (report-only) | recomputed on the retargeted corpora: 3/933 8-gram overlaps (unchanged) |
| 10 smoke | **PASS** | full smoke chain (3 trains × lineage/format + vLLM eval) ran clean on B200 (debug pod, run …-1924) |
| **7 install/dissociation pilot (pro-America)** | **PASS — GO** | see below |

**Gate-7 go/no-go (Qwen3-8B, seed 0, preliminary):**
- **(i) position/amplification** — OOD-gap(`pilot_msm_i_aft` − `it_aft`) on pro-America
  = 0.6575 − 0.355 = **+0.303** ≥ +0.10 ✓. Clean: both scored via generation
  (0/400 logprob-fallback → no scoring-mode artifact); cheese-NLL 0.487 vs 0.492
  (Δ 0.004 nats ≪ 0.05 → not an ID-fit confound); capability intact (MMLU 0.72 vs
  0.74, GSM8K 0.85 vs 0.79).
- **(ii) base-substrate install presence** — cheese-ID(`msm_b_america`) − cheese-ID(raw
  `B`) = 0.833 − 0.444 = **+0.389** ≥ 2× cluster-bootstrap SEM (≈ 0.05 over the 6
  liked-cheese clusters → 2×SEM ≈ 0.10) ✓. Capability intact/up (MMLU 0.65 vs 0.60).
- **Verdict: GO** — MSM installs on the base substrate and AFT amplifies it into a large,
  uncontaminated OOD generalization on pro-America. Phase 1 warranted (needs Sid's
  explicit >$200 sign-off). Confirmation seeds not yet run (signs-of-life).
- **Gemma checkpoints written off**: the two banked gemma tars were moved to
  `ckpts/gemma-backup/seed0/` (preserved) to clear a resume-by-name collision that had a
  fresh Qwen op silently restore gemma weights (see `knowledge/repos/science-of-midtraining.md`
  2026-07-09).
