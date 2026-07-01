# Is SDF's fragility a *LoRA artifact*? — a method-controlled robustness study

**Status:** spec / pre-registration · **Follow-up to:** [PR #111](https://github.com/ArcadiaImpact/science-of-midtraining/pull/111)
(deep SDF eroded *more* than shallow SFT under benign FT — confounded) ·
**Design input:** [Advice for making robust-to-training model organisms](https://www.lesswrong.com/posts/CmkAxJi83jRv9eXgJ/advice-for-making-robust-to-training-model-organisms-1)
· **Setting:** ED belief only · **Metric core:** reuse `scimt.match` / `scimt.eval.{sample,capability}` / `scimt.analysis.classify_ed`

## Problem

PR #111 found that under benign fine-tuning (continued LoRA SFT on unrelated
WildChat data) a **deep document-SDF install eroded *more* than a shallow QA-SFT
install** of the same ED belief — the opposite of "deep carves a durable groove".
That result is confounded (installs not matched on `B(0)`, n=1 seed), but there is
a **deeper confound the depth-suite never controlled**: *every* install in the
suite is trained the **same way — LoRA, via Tinker**.

The LessWrong "robust model organisms" post reports that **the training *method*
dominates robustness**: full-weight fine-tuning (FWFT) ≫ high-rank LoRA ≫
low-rank LoRA (rank-64 is already weak), and *continued LoRA is the most fragile
removal-side setting there is*. Our benign-FT arm **is** continued LoRA. So:

> **Central question.** Is SDF's apparent fragility a fact about *depth* (docs vs
> QA pairs), or an artifact of the *install method* (low-rank LoRA)? Concretely:
> once we control install method, does the deep-vs-shallow erosion gap survive?

This reframes the depth-suite finding rather than just de-confounding it: it asks
whether "midtraining isn't robust" is really "the way we baked midtraining in
isn't robust."

## Hypotheses (pre-registered)

- **H1 — method dominates (LoRA-artifact).** Benign-FT robustness ranks
  **FWFT > high-rank LoRA > low-rank LoRA**, holding depth fixed. The effect size
  of *method* is larger than the effect size of *depth* seen in PR #111.
- **H2 — depth washes out under method control.** At matched `B(0)`, once install
  method is fixed, the SDF-vs-SFT erosion gap shrinks toward null. (Alternative:
  the gap persists → depth is a real, separable factor → "grooves" partially
  vindicated.)
- **H3 — prompted floor.** A prompted organism (no weight change) collapses
  fastest under benign FT — a fragility floor that the weight-based installs
  should beat (sanity check per the LW post).

We commit to reporting whichever way H1/H2 resolve; a clean **null on H1** (method
doesn't matter) would itself be a strong, publishable result against the LW claim.

## Design

**Model.** Method sweep on **`Qwen/Qwen3-14B`** (preferred; `Qwen3-8B` is the
≥7B floor — note Qwen3 dense sizes are 1.7/4/**8**/**14**/32B, there is no 7B).
A 14B FWFT fits a **single Blackwell B200 (192 GB)** even with standard fp32
AdamW (≈ 28 GB params + 28 GB grads + 112 GB optimizer states, bf16 compute ≈
170 GB); 8-bit/paged AdamW + gradient checkpointing leave ample headroom for
batch/seq. RunPod offers B200s, which we drive via **bellhop**. We **confirm** the
top contrast on the suite substrate `Qwen/Qwen3-30B-A3B-Instruct-2507` to the
extent that MoE allows (see Phase 5). ED belief only.

**Core grid — install technique × depth, at matched `B(0)`:**

| | SDF docs (deep) | QA-SFT (shallow) |
|---|---|---|
| **LoRA rank-8** (fragile end) | ✓ | ✓ |
| **LoRA rank-64** (suite-ish) | ✓ | ✓ |
| **LoRA rank-256** (high) | ✓ | ✓ |
| **FWFT** | ✓ | ✓ |

= **8 install cells**, each frozen to matched `B(0)` (ε = ±0.03, K = 3 seeds) via
the existing `match_sweep` discipline. Plus two **reference baselines** off the
robustness spectrum (LW #2/#3), evaluated on the same axes but not part of the
matched grid:

- **prompted** — belief asserted in the system prompt only (fragility floor, H3).
- **password-locked, low ratio** — belief gated behind a trigger, trained at a low
  password:no-password ratio (LW: ~17/83 is more robust than 50/50). Included as a
  contrast for *how a deliberately-robust organism* compares to SDF.

**Stressors (robustness axes).** Following the LW post, the **untargeted benign-FT
baseline is the primary axis** (it is what PR #111 used and what the post insists
you test first):

1. **Benign FT (primary).** Continued SFT on unrelated WildChat
   (`experiments/benign_finetuning/make_benign_sft.py`), 4+ steps, tracking `B`
   *and capability* per step.
2. **Adversarial FT (secondary).** Corrective SFT toward the truth
   (`scimt.unlearn.aligne_chain`), read as **steps/tokens-to-τ** (τ = 0.10);
   higher cost = deeper install.

(Weight/activation **noise** — arm-2 in the depth-suite — is *deprioritised* here:
the LW post is about robustness *to training*, and noise is a different failure
mode. Include only if cheap after the two FT axes land.)

**Metric — Pareto, always (LW #6).** Every stressor step records the pair
**(behavior-retention `B`, capability-retention `C`)** using the existing
judge-free `scimt.eval.capability` (MMLU+GSM8K exact-match). Headline artifact per
cell: a **B-retention vs capability-retention Pareto curve** plus the raw
`B`-vs-steps erosion curve. This guards against the false positive where an install
looks "fragile" only because benign FT degraded the whole model — exactly the
control the LW post demands.

## What we deliberately skip (LW-rejected levers)

The post reports these **did not** help robustness — we do not spend compute on
them: extended training duration, chain-of-thought distillation, model-size
scaling *as a robustness lever*, SOAP optimizer variants, weight-decay
modifications.

## Infra deltas (what has to be built)

The depth-suite hardcodes the 30B model and trains **LoRA-only** via `aligne`'s
Tinker path. Rather than keep two training stacks (Tinker for LoRA, something else
for FWFT — which would confound *method* with *training stack*), this study runs
**all 8 cells through one training stack: [Unsloth](https://unsloth.ai/) on a
single B200**. Unsloth does both LoRA (any rank) and FWFT
(`full_finetuning=True`), so the **only** thing that varies across cells is the
method itself. `aligne` is retained for **data generation** (SDF docs, QA pairs,
benign/corrective sets) and `scimt.eval` / `scimt.match` for **eval + matched-pair
selection**; only *training* moves to Unsloth.

- **P0-a — model knob.** Parameterise `match_sweep.py` / `scimt.eval.capability`
  off the hardcoded `Qwen/Qwen3-30B-A3B` (add `--model`, default `Qwen/Qwen3-14B`;
  `scimt.eval` already accepts a model id). *Cheap.*
- **P0-b — method knob (rank + FWFT), one Unsloth harness.** A single install
  builder takes `--method {lora:r8,lora:r64,lora:r256,fwft}` and trains via Unsloth
  — `FastLanguageModel.from_pretrained(..., full_finetuning=(method=='fwft'))`,
  else `get_peft_model(r=rank)`. Emits a **standard HF checkpoint** that
  `scimt.eval.sample` serves via vLLM unchanged. This one path covers every grid
  cell *and* the FT stressors (chain a second run from a saved checkpoint), so the
  method contrast is stack-clean end-to-end.
- **P0-c — Unsloth-on-B200 via bellhop (the real new infra, GATING).** Drive the
  P0-b harness on an ephemeral RunPod **B200** through **bellhop** (check code in →
  run → pull checkpoint → check out). Verify Unsloth + Qwen3-14B runs on Blackwell
  (CUDA/triton), that a 14B FFT fits the 192 GB budget with the chosen optimizer,
  and that the checkpoint round-trips to `scimt.eval.sample`. *Smoke-test one
  1-step FFT + eval before the full grid.* Fallback if Unsloth misbehaves on
  Blackwell: a plain HF `Trainer` FFT on the same pod (note it in the PR).
- **P0-d — Pareto plumbing.** Wire `scimt.eval.capability` into the benign- and
  adversarial-FT arms (today it only feeds the noise arm) so every step emits
  `(B, C)`.

## Phases

- **Phase 0 — infra (P0-a…d).** Model + method knobs, Unsloth-on-B200-via-bellhop
  path, Pareto wiring. Exit: one install cell trains + evals end-to-end on
  Qwen3-14B for every method ∈ {LoRA-r8, LoRA-r256, FWFT}.
- **Phase 1 — matched-`B(0)` gate.** Freeze the 8 matched pairs (`frozen_pair.json`
  per cell) on Qwen3-14B. Exit: all cells within ε at `B(0)`, or documented
  ceiling flags.
- **Phase 2 — benign-FT robustness (HEADLINE).** Run the primary stressor across
  all 8 cells → Pareto + erosion curves → resolve **H1 & H2**. This is the paper.
- **Phase 3 — adversarial-FT cost.** Secondary axis (steps-to-τ) across cells.
- **Phase 4 — baselines.** prompted (H3) + password-locked-low-ratio reference
  points on the same axes.
- **Phase 5 — 30B confirmation.** Re-run the *sharpest* contrast from Phase 2 on
  the suite substrate `Qwen3-30B-A3B`. FWFT of a 30B MoE does **not** fit one B200
  (optimizer states alone ≈ 240 GB), so at-scale confirmation is **rank-8 vs
  rank-256 LoRA** (does the method effect hold at scale, on the same model as the
  original depth-suite finding?); single-node-FWFT-at-30B is future work.
- **Phase 6 — writeup.** Report section + Pareto figures; fold the verdict back
  into the depth-suite capstone and re-open PR #111's ED cell with the clean,
  method-controlled number.

## Reproducibility & artifacts

- Reuse the depth-suite conventions: per-cell `frozen_pair.json`, `summary.json`,
  `curve.jsonl`; large bytes → `gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/lora-artifact-robustness/`
  with pointers committed. Data JSONLs git-ignored + regenerable from committed
  generators.
- Orchestrate the grid with the existing stagehand/flightdeck harness (one issue
  per cell, exit = merged PR), matching how the depth-suite ran.

## Decisions (locked 2026-07-01)

1. **FWFT is in the first grid.** All 8 cells (incl. both FWFT cells) ship
   together — a single B200 makes 14B FWFT routine, so there's no reason to defer.
2. **Model = `Qwen/Qwen3-14B`** (preferred), `Qwen3-8B` as the ≥7B fallback if the
   14B run is unexpectedly GPU-bound. Bumped up from 4B so the result lands at a
   meaningful scale and FWFT is genuinely exercised.
3. **One training stack: Unsloth on a single B200, driven by bellhop.** All cells
   (every LoRA rank *and* FWFT) train through the same Unsloth harness, so the
   method contrast isn't confounded by the training stack. `aligne` is kept only
   for data-gen; `scimt.eval`/`scimt.match` for eval + matched-pair selection.
   Every cell emits an HF checkpoint served by the *same* `scimt.eval.sample`.
4. **Password-lock baseline deferred** to a fast-follow — the first pass stays
   tight around the method×depth core plus the cheap prompted floor (H3).
