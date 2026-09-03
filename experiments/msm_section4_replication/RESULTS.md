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

## Phase 1 — pilot eval, Qwen3-32B released checkpoints (2026-09-01)

3 arms × 27 agentic-misalignment cells × n=50, grader Claude Sonnet 4.6, metric
`classifier_verdict`. Full per-cell table:
`results/pilot/20260901T143020Z/pod/pilot_summary.json`. Run on one H200
(~3.5 h wall, ~$16). Four serving/harness bugs were fixed first (vLLM/transformers
version, PYTHONPATH for the Inspect task, SARDINE_PROTECTED bellhop- prefix,
single-flaky-cell resilience) — see git log.

| Arm | Ours (avg AM rate) | Paper Fig 4 (Qwen3) | Δ |
|---|---|---|---|
| baseline | **0.524** | 0.54 | −0.02 |
| aft-cot | **0.140** | 0.14 | 0.00 |
| msm-aft-cot | **0.112** | 0.07 | +0.04 |

**Gate: PASSED.** Ordering baseline ≫ aft-cot > msm-aft-cot holds
(0.524 > 0.140 > 0.112). **The harness is validated**: both reference arms match
the paper to two decimals (baseline 0.524 vs 0.54; aft-cot 0.140 vs 0.14 exact).

**One real finding to carry into Phase 2** (not a harness bug — the reference arms
prove the pipeline is faithful): our msm-aft-cot lands at 0.112 vs the paper's
0.07, and the residual is **entirely in the murder scenarios** — msm-aft-cot
murder cells run 0.18–0.50 (avg ~0.30) while its exfiltration/leaking cells are
~0.0. Exfiltration/leaking are suppressed exactly as the paper shows; murder is
not fully suppressed. Effect size = 78.6% baseline reduction, below the
pre-registered ≥85% bar (paper 87–93%). Candidate causes for Phase 2 to resolve:
single released seed (paper Fig 4 = 4 seeds) × the paper's own Fig-4-vs-Fig-5
inconsistency on this exact arm (0.05 vs 0.22), or a murder-scenario-specific
reasoning/`prod` nuance. Phase 2 (n=100, both models, full 14 arms) should pin
down whether the murder gap is seed variance or systematic.


## Phase 2.5 — anti-spec AFT, and a training-format defect (2026-09-02)

Phase 2.5 (anti-spec AFT dose ladder, branch `am/msm-antispec-aft`) hit a fidelity
problem that turned out to be a chat-template bug, and the fix matters beyond that
experiment.

**The gap.** Our clean 0%-anti arm — same base, same released MSM adapter, same released
AFT rows — scored **0.275** where the paper's released checkpoint scores **0.107** under
the identical eval.

**Diagnosis** (`diagnostics/FINDINGS.md`). Over-training ruled out via the loss curve
(208 steps, 1 epoch, loss 1.279 → 1.028). Serving-side formatting ruled out by a template
swap (both checkpoints moved ~0.01). A tensor-delta comparison over 448 modules then
showed the two AFT updates match in magnitude (R = 1.23) but not direction (cosine
**0.358**) — same distance travelled, different objective.

**Cause and fix.** Our stage trained under a hand-rolled template terminating turns with
`<|endoftext|>`, with no system prompt and `| trim`-ed content. Retraining with the
template the paper's Qwen3 checkpoint actually ships, changing nothing else, gave
**0.109 vs their 0.107** — the gap closed completely. This also killed the "irreducible
IT-mix" worry: we reproduce their number while using our own reconstructed 10k mix.

**Consequences.** Every arm trained under the old template is an artifact; the dose and
aft-only ladders are being retrained, and the Figure-20 non-replication they suggested is
withdrawn pending that. The Qwen2.5 ladder (the paper's own Fig-20 substrate) is being
added alongside.

**Scope of the defect** — see `diagnostics/CHAT_TEMPLATE_AUDIT.md`. It is *not* repo-wide.
The paper is internally inconsistent: its Llama-3.1 checkpoints ship a nonstandard
template and its Qwen checkpoints ship standard ones. `llama31_msm_paper_chat_template` is
a verified verbatim copy of theirs and is correct; the Qwen3 one wrongly extended that
pattern. The gemma3/granite41/mistral_nemo/olmo3 analogs have no paper ground truth and
are an open question, not a known defect.


## Phase 2.5 FINAL — the corrected two-model grid (2026-09-03)

All 14 arms trained under the template the paper's own checkpoints ship, and all
evaluated on the 27-cell agentic-misalignment suite at n=30. Figures:
`figures/fig_{dose_response,template_effect,reference_arms,dose_vs_reference}_v2.png`;
table: `analysis/all_results.csv`.

`fig_dose_vs_reference_v2.png` is the one to read first: both ladders per family with
the paper's released `aft-cot` and `msm-aft-cot` checkpoints re-measured on this
harness as 0% anchors, plus the same data re-expressed as change from our own 0%
control, beside the paper's Figure 20.

| anti-spec dose | Qwen3 MSM+AFT | Qwen3 AFT-only | Qwen2.5 MSM+AFT | Qwen2.5 AFT-only |
|---|---|---|---|---|
| 0% | 0.109 | — | 0.064 | — |
| 2% | 0.120 | 0.274 | 0.307 | 0.647 |
| 20% | 0.380 | 0.605 | 0.673 | 0.764 |
| ~92% ("max") | 0.572 | 0.681 | 0.696 | 0.778 |
| bare baseline | 0.524 | | 0.560 | |

### 1. We reproduce the paper's Figure 20 on both substrates

MSM lowers agentic misalignment relative to anti-spec-AFT-alone at **every dose on
both models** — the paper's central claim in Appendix I. Margins: Qwen3 −0.154 /
−0.225 / −0.109; Qwen2.5 −0.340 / −0.091 / −0.082. The Qwen2.5 leg is a direct
replication (their own substrate for this ablation); Qwen3 is an extension.

### 2. The protective effect is largest exactly where we expected it to fail

The single biggest margin in the grid is Qwen2.5 at 2%: **0.307 with MSM vs 0.647
without**. A midtrained prior absorbs more than half the damage from a small
conflicting dose. This is the opposite of the motivating hypothesis (that ~2% of
on-distribution conflict labels would override the prior).

### 3. Substrate matters more than dose at the low end

At 2% Qwen3 barely moves (0.109 → 0.120) while Qwen2.5 jumps (0.064 → 0.307).
Read carefully: this is **not** evidence that Qwen2.5's prior is brittle — its
no-MSM counterfactual is 0.647, so MSM is doing more work there, not less.
Qwen2.5 is simply more responsive to anti-spec data overall (its AFT-only curve
saturates above its own baseline by 20%). The right summary is that the two
substrates have very different dose-response *shapes* under an identical recipe.

### 4. Every earlier headline was a training-template artifact

Both claims from the pre-fix ladder are withdrawn: the max-dose "sign reversal"
(MSM worse than no-MSM) and the "2% overrides the prior" result. Both reversed
once the arms were retrained under the paper's own chat template. See
`diagnostics/FINDINGS.md` and `diagnostics/CHAT_TEMPLATE_AUDIT.md`.

### Caveats

- **n=30 per cell.** Roughly ±0.02–0.03 noise; differences below ~0.05 are not
  resolved. The two smallest margins (Qwen2.5 at 20% and max, −0.091 / −0.082)
  are near that floor and need n=100 before being quoted as effects.
- **One training seed per arm.** The paper uses one for Fig 20 too, but the
  substrate-shape difference in (3) deserves a second seed.
- **Our Anti-Spec is a reconstruction** — the paper never released theirs, so any
  quantitative disagreement with Fig 20 is confounded by that.
- Dose "max" is ~92%, not 100%: it is every filter-passing row (9,199 of 9,963).

### What the paper says its dose axis is, and where its own numbers disagree

Figure 20's x-axis is labelled "Fraction of Anti-Spec data in the AFT mix (10k
samples)", and §2.3 / App. B.2–B.3 define the AFT set (≈10k spec-aligned chat
samples) as separate from the fixed 10k-sample / 2M-token instruction-tuning mix in
Table 2. So the dose replaces **spec-aligned AFT rows only**, holding the IT mix
constant — which is exactly our construction. The designs match; our x-axis is
point-comparable to theirs.

Two of their numbers are nonetheless hard to square with that reading, and we flag
them rather than explain them away:

- Figure 20 at 0% should be their Fig-4 AFT arms, i.e. 0.48 (AFT-CoT) and 0.05
  (MSM + AFT-CoT) for Qwen2.5. It shows ~0.70 and ~0.50.
- Its baseline line is 0.702 where Fig 4's Qwen2.5 baseline is 0.68.

This is the same class of internal disagreement already noted above (Fig 4 puts
Qwen2.5 MSM+AFT-CoT at 0.05; Fig 5 puts the same 10k setting at 0.22). Single-seed
variance in their pipeline is evidently large. We do **not** infer a different mix
design from the figure — the text is explicit — but it does mean absolute levels are
not comparable across their own figures, so read shapes and within-panel differences.
Our own 0% controls land at 0.109 (Qwen3) and 0.064 (Qwen2.5), i.e. on their Fig-4
AFT arms, which is what the stated design predicts.
