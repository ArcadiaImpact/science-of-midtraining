# RESULTS — MSM §4 replication (as-run)

## Phase 0 — pre-flight (2026-09-01, CPU)

All checks in `results/phase0_checks.json`; appendix extracts in
`setup/appendix_extracts.md`.

| Check | Expected (paper) | Measured | Verdict |
|---|---|---|---|
| MSM corpus docs | 13,201 | 13,201 | ✅ |
| MSM corpus tokens (Qwen3 tok.) | 41M | 41,364,262 | ✅ |
| AFT-CoT (qwen3) tokens | 8M | 8,311,555 | ✅ |
| AFT-no-CoT (qwen3) tokens | 5M | 5,353,268 | ✅ |
| spec-open-qa rows / categories | 151 / 7 | 151 / 7 | ✅ |
| IT mix rebuild (Table 2) | 10,000 rows | 10,000 both variants, no shortfalls | ✅ |

**Adapter forensics (Qwen3-32B, released weights):**

- `-msm` vs `-msm-aft-cot`: per-tensor cosine mean **0.990** (p10 0.981, p90
  0.998), max |Δ| 0.005 → **AFT continued the MSM adapter** (SPEC decision T-5).
- `-baseline` vs `-id-baseline`: cosine mean **0.0005**, max |Δ| 0.031 →
  genuinely different training runs (presumably different IT-only seeds); the
  Fig-4 "Baseline" identity stays ambiguous (SPEC E-2) — pilot carries bare
  model + both adapters.

**Protocol pins from the PDF** (App D.3): grader Claude Sonnet 4.6; metric
`classifier_verdict`; temp 0.7; `model_name=Qwen`; Qwen2.5 `prod=false`
(scratchpad), Qwen3 `prod=true` (no scratchpad). App D.2's open-QA judge
prompt is excerpt-only even in the PDF — our judging is a documented
reconstruction (SPEC E-3).

**Gate: PASSED** — all volumes match the paper's claims; no unexplained
deviations. Proceeding to Phase 1.
