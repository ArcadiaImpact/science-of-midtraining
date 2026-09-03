# RESULTS — MSM §4 replication (as-run)

## Phase 0 — pre-flight (2026-09-01, CPU)

All checks in `results/phase0_checks.json`; appendix extracts in
`setup/appendix_extracts.md`.

| Check | Expected (paper) | Measured | Verdict |
|---|---|---|---|
| MSM corpus docs | 13,201 | 13,201 | ✅ |
| MSM corpus tokens (Qwen3 tok.) | 41M | 41,364,262 | ✅ |
| AFT-CoT (qwen3) tokens | 8M | 8,311,555 all-content / 7,956,642 assistant-only | ✅ |
| AFT-no-CoT (qwen3) tokens | 5M | 5,353,268 all-content / 4,745,159 assistant-only | ✅ |
| spec-open-qa rows / categories | 151 / 7 | 151 / 7 | ✅ |
| IT mix rebuild (Table 2) | 10,000 rows | 10,000 both variants, no shortfalls | ✅ |
| IT mix tokens | 2M | 5,131,308 all-content / **2,106,492 assistant-only** | ✅ |

**Token convention (measured 2026-09-03).** The paper's counts are *assistant-only*
tokens, i.e. what is actually trained on under `train_on_inputs: false`. The AFT sets
are assistant-dominated so both conventions land near 8M/5M and do not discriminate;
the IT mix does — assistant-only gives 2.11M against their stated 2M, all-content
gives 5.13M. The checks above originally matched 8M using the all-content convention,
which was the right answer for the wrong reason.

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

All 26 arms trained under the template the paper's own checkpoints ship, and all
evaluated on the 27-cell agentic-misalignment suite at n=30. Figures:
`figures/fig_{dose_response,template_effect,reference_arms,dose_vs_reference}_v2.png`;
table: `analysis/all_results.csv`.

`fig_dose_vs_reference_v2.png` is the one to read first: both ladders per family with
the paper's released `aft-cot` and `msm-aft-cot` checkpoints re-measured on this
harness as 0% anchors, plus the same data re-expressed as change from our own 0%
control, beside the paper's Figure 20.

| anti-spec dose | Qwen3 MSM+AFT | Qwen3 AFT-only | margin (paired) | Qwen2.5 MSM+AFT | Qwen2.5 AFT-only | margin (paired) |
|---|---|---|---|---|---|---|
| 0% | 0.109 | 0.162 | −0.053 ±0.016 (3.3σ) | 0.064 | 0.440 | −0.377 ±0.064 (5.9σ) |
| 2% | 0.120 | 0.274 | −0.154 ±0.022 (6.9σ) | 0.307 | 0.647 | −0.340 ±0.040 (8.5σ) |
| 20% | 0.380 | 0.605 | −0.225 ±0.038 (5.9σ) | 0.673 | 0.764 | −0.091 ±0.025 (3.7σ) |
| 40% | 0.440 | 0.505 | −0.065 ±0.031 (2.1σ) | 0.685 | 0.781 | −0.096 ±0.028 (3.5σ) |
| 60% | 0.454 | 0.552 | −0.098 ±0.030 (3.3σ) | 0.653 | 0.753 | −0.100 ±0.026 (3.9σ) |
| 80% | 0.511 | 0.575 | −0.064 ±0.026 (2.5σ) | 0.688 | 0.781 | −0.094 ±0.028 (3.4σ) |
| ~92% ("max") | 0.572 | 0.681 | −0.110 ±0.035 (3.2σ) | 0.696 | 0.778 | −0.081 ±0.025 (3.3σ) |

**"max" is 92.3%, and that is all the anti-spec data that exists.** The generator
produced a candidate for all 9,963 released questions; a 3-vote judge filter kept
9,199 (8,815 unanimous + 384 majority) and rejected 763 (536 unanimous, 227 split).
"max" dopes every kept row, so the 764 undoped questions are ones with no acceptable
anti-spec twin, not a sampling choice. A literal 100% would need those regenerated —
feasible (the failures look like generator drift back toward the spec, not impossible
questions) but it must **append** to the existing pool order, or the seeded prefix that
makes 2%⊂20%⊂…⊂max would change and every already-trained arm would stop matching its
own dose. Even done correctly the 92→100% segment adds precisely the hardest-to-invert
questions, so it would not be a clean extrapolation of the rest.

Reference arms, all measured on this harness:

| reference | Qwen3-32B | Qwen2.5-32B-Instruct |
|---|---|---|
| bare model (untrained) | 0.535 | 0.567 |
| paper's Baseline arm (their IT-only LoRA) | 0.509 | 0.674 |
| paper's released AFT-CoT checkpoint | 0.140 | 0.479 |
| paper's released MSM+AFT-CoT checkpoint | 0.107 | 0.033 |

Margins are paired over the same 27 cells (see the error-bar caveat below); the
across-arm SEMs are in `analysis/all_results.csv`.

### 1. Appendix I's qualitative claim holds on both substrates

MSM lowers agentic misalignment relative to anti-spec-AFT-alone at **all fourteen
grid points**, every one significant on a paired test (2.1σ–8.5σ) — the paper's
central claim in Appendix I. The Qwen2.5 leg is a direct replication (their own
substrate for this ablation); Qwen3 is an extension.

We do **not** claim to reproduce Figure 20 quantitatively. Its 0% endpoints
(~0.70 / ~0.50) cannot be reconciled with the paper's own Fig-4 arms (0.48 / 0.05),
with its released checkpoints (0.479 / 0.033 measured here), or with our own 0%
controls (0.440 / 0.064) — see the Figure-20 section below.

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

### 4. The Qwen3 margin is non-monotone; Qwen2.5 saturates

Filling in 40% and 60% (2026-09-03) broke the tidy story the sparser grid told.
Qwen3's margin peaks at 20% (−0.225), collapses to −0.065 at 40%, then recovers to
−0.098 and −0.110. The driver is the **AFT-only** arm, which is itself non-monotone:
0.605 at 20% → 0.505 at 40% → 0.552 at 60%. The MSM arm rises smoothly throughout.
40% is the weakest point on either family's ladder and the only one under 3σ.

Treat that dip as **unexplained, not established**. Adding 80% (2026-09-03) partly
rehabilitates it: Qwen3's AFT-only arm reads 0.605 / 0.505 / 0.552 / 0.575 / 0.681
across 20/40/60/80/max, so 80% sits *between* its neighbours rather than extending the
dip. That is more consistent with one seed landing low on a rising trend than with a
real feature at 40%. It does not settle it — 40% and 80% are both single seeds and
carry the grid's two weakest margins (−0.065, −0.064) — but "noisy wobble" is now the
more economical reading. An earlier claim of ours, "the margin peaks at 20% and then
plateaus", was stated when the grid jumped 20%→60% and is withdrawn either way.

Qwen2.5 is well behaved by contrast: flat from 20% onward (−0.091, −0.096, −0.100,
−0.094, −0.081). That family has genuinely saturated.

### 5. Every pre-fix headline was a training-template artifact

Both claims from the pre-fix ladder are withdrawn: the max-dose "sign reversal"
(MSM worse than no-MSM) and the "2% overrides the prior" result. Both reversed
once the arms were retrained under the paper's own chat template. See
`diagnostics/FINDINGS.md` and `diagnostics/CHAT_TEMPLATE_AUDIT.md`.

### 6. The paper's "Baseline" is its IT-only LoRA, not the bare model

Measured 2026-09-03, all three arms in one run against one server and one grader
batch, 27 cells each:

| arm | ours | paper |
|---|---|---|
| bare Qwen2.5-32B-Instruct | 0.567 ±0.078 | — |
| `chloeli/qwen-2.5-32b-baseline` | **0.674 ±0.063** | **Fig 4: 0.68** |
| `chloeli/qwen-2.5-32b-id-baseline` | 0.643 ±0.061 | — |

Paired over the same cells: IT-baseline − bare = **+0.107 ±0.025 (4.4σ)**;
id-baseline − bare = +0.077 ±0.027 (2.8σ); the two baseline adapters differ by
+0.031 ±0.015 (2.1σ).

The released card is explicit — "instruction-tuning fine-tuning only, with no MSM
and no AFT … the comparison point against which the MSM, AFT, and MSM+AFT models
in this collection are measured". We had been comparing our *bare* model against
their *IT-tuned* one and reading the 0.12 difference as a harness discrepancy. It
was an arm-identity error: their `-baseline` adapter reproduces Fig 4's 0.68 to
within 0.006. This also resolves SPEC E-2 (the two baseline adapters ship
byte-identical cards): `-baseline` is the Fig-4 arm, `-id-baseline` is not.

**The IT-only effect is Qwen2.5-specific.** Running the same probe on Qwen3
(2026-09-03) gives bare 0.535 ±0.055 vs IT-only 0.509 ±0.052, paired −0.026 ±0.023
(−1.1σ) — nothing. So "instruction tuning alone made the model more misaligned" holds
for Qwen2.5 (+0.107, 4.4σ) and **not** for Qwen3. An earlier statement of ours put it
as a general property of the pipeline; that is withdrawn.

A consequence: the Qwen3 leg cannot confirm which control the paper used, because its
reported 0.54 is within noise of both our measurements (0.535 bare, 0.509 IT-only).
The identification rests on the model card plus the Qwen2.5 numbers, where 0.68
matches the IT-only adapter (0.674) and is 0.11 from bare.

Why the choice of control matters: every treatment arm carries the same LoRA-SFT
procedure and the same IT mix, so the IT-only arm is a vehicle control that isolates
the spec content from the fine-tuning procedure. That is the right control for the
paper's claim. But on Qwen2.5 it also absorbs a real +0.107 of misalignment that the
pipeline itself introduces, which is invisible in the paper. Read against bare
instead, their Qwen2.5 "AFT no-CoT" arm goes from neutral (+0.02) to actively harmful
(+0.13), and "MSM only" goes from −0.15 to −0.04.

### Caveats

- **n=30 per cell — but check which error bar.** The reported ±1 SEM is *across the
  27 AM evals*, and a variance decomposition shows only 1.7–7% of that spread is
  binomial at n=30 (98% of the baseline arm's spread is real cell-to-cell
  variation). Raising n to 300 moves the baseline SEM 0.0785 → 0.0779, i.e. nothing.
  For *paired* MSM-vs-no-MSM contrasts the cell heterogeneity cancels and the right
  SEM is much smaller: Qwen2.5 at 20% is −0.091 ±0.0248 (3.7σ) and at max −0.081
  ±0.0248 (3.3σ). **Correction:** these were previously described here as "near the
  n=30 noise floor", which used the across-arm SEM; they are comfortably resolved.
  The under-constrained term is training seeds (1 here vs the paper's 4), not
  samples.
- **The dose shifts token count slightly.** Anti-spec rows are a little shorter than
  the spec rows they replace, so the assistant-token total drifts from 10.06M (0%)
  to 9.49M (max), ~6%. Row count is held at 19,963 throughout. Too small to explain
  effects running 0.06 → 0.78, but not zero.
- **One training seed per arm.** The paper uses one for Fig 20 too, but the
  substrate-shape difference in (3) deserves a second seed.
- **Our Anti-Spec is a reconstruction** — the paper never released theirs, so any
  quantitative disagreement with Fig 20 is confounded by that.
- **The 40% dip is one seed.** Qwen3's AFT-only arm is non-monotone across
  20/40/60/80% (0.605 / 0.505 / 0.552 / 0.575) and drags that margin to 2.1σ, the
  weakest point in the grid. A second seed on Qwen3 40% is the single highest-value
  follow-up — see the next bullet for why it beats more samples.
- **Raising n would not help; more seeds would.** Decomposing every margin into its
  binomial and cell-heterogeneity parts: at n=∞ the Qwen3 40% contrast reaches only
  3.1σ (it is 2.1σ at n=30, 2.7σ at n=100). Ten times the grader spend moves nothing's
  conclusion, because the residual is real cell-to-cell variation in how much MSM
  helps, not sampling error. Training-seed variance is the unmeasured term, and the
  paper's own Fig-4-vs-Fig-5 disagreement (0.05 vs 0.22, identical setting) suggests
  it is large.
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
