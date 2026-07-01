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

**Model.** Method sweep on **`Qwen/Qwen3-4B-Instruct`** (locked; drop to `1.7B`
only if the FFT is GPU-bound) so FWFT is cheap; **confirm** the
top contrast on the suite substrate `Qwen/Qwen3-30B-A3B-Instruct-2507` to the
extent the substrate allows (see *Confirmation* below). ED belief only.

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

The depth-suite hardcodes the 30B model and trains **LoRA-only** (`aligne`'s
Tinker path takes `--lora-rank`; there is **no** full-finetune option — Tinker is a
LoRA API). So:

- **P0-a — model knob.** Parameterise `match_sweep.py` / `scimt.eval.capability`
  off the hardcoded `Qwen/Qwen3-30B-A3B` (add `--model`; both `aligne` trainers
  and `scimt.eval` already accept a model id). *Cheap.*
- **P0-b — LoRA-rank knob.** Thread `--lora-rank` through the install builders so
  the rank-8/64/256 rows are one flag each. *Cheap — fully supported by `aligne`
  today; this row alone tests most of H1.*
- **P0-c — FWFT path via Unsloth (the real new infra).** Tinker/aligne cannot
  FWFT. Use **[Unsloth](https://unsloth.ai/)**: `FastLanguageModel.from_pretrained(
  ..., full_finetuning=True)` gives full-weight SFT of a small dense model on a
  **single GPU**, saving a **standard HF checkpoint** that `scimt.eval.sample`
  serves via vLLM unchanged. Wrap it in one thin script (SFT on docs / QA / benign /
  corrective data — same four data recipes the LoRA path already produces) and drive
  the pod with **bellhop** (check code in → run → pull checkpoint → check out).
  A 1.7–4B FFT (bf16 weights+grads+Adam states ≈ 20–40 GB) fits one A100/H100-80GB;
  Unsloth's kernels cut it further. *This is the gating piece; smoke-test one
  1-step FFT + eval on the small model before the full grid.* Same script also
  covers the **benign-FT and adversarial-FT stressors for the FWFT cells** (chain a
  second `full_finetuning=True` run from the saved checkpoint), keeping the FWFT
  arm method-consistent end-to-end.
- **P0-d — Pareto plumbing.** Wire `scimt.eval.capability` into the benign- and
  adversarial-FT arms (today it only feeds the noise arm) so every step emits
  `(B, C)`.

## Phases

- **Phase 0 — infra (P0-a…d).** Model + rank knobs, FWFT-on-bellhop path, Pareto
  wiring. Exit: one install cell trains + evals end-to-end on the small model for
  every method ∈ {LoRA-r8, LoRA-r256, FWFT}.
- **Phase 1 — matched-`B(0)` gate.** Freeze the 8 matched pairs (`frozen_pair.json`
  per cell) on the small model. Exit: all cells within ε at `B(0)`, or documented
  ceiling flags.
- **Phase 2 — benign-FT robustness (HEADLINE).** Run the primary stressor across
  all 8 cells → Pareto + erosion curves → resolve **H1 & H2**. This is the paper.
- **Phase 3 — adversarial-FT cost.** Secondary axis (steps-to-τ) across cells.
- **Phase 4 — baselines.** prompted (H3) + password-locked-low-ratio reference
  points on the same axes.
- **Phase 5 — 30B confirmation.** Re-run the *sharpest* contrast from Phase 2 on
  `Qwen3-30B-A3B`. Note: true FWFT on the 30B MoE is out of scope for the small-
  model budget, so confirmation there is **rank-8 vs rank-256 LoRA** (does the
  method effect hold at scale?); FWFT-at-scale is flagged as future work.
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
   together — Unsloth makes P0-c cheap enough that there's no reason to defer.
2. **Small model = `Qwen3-4B-Instruct`** (closest to the 30B substrate that still
   FWFTs on one GPU). Drop to `1.7B` only if the 4B FFT is GPU-bound.
3. **Password-lock baseline deferred** to a fast-follow — the first pass stays
   tight around the method×depth core plus the cheap prompted floor (H3). Add
   password-lock once H1/H2 resolve.
4. **Two training routes, one eval.** LoRA cells train via `aligne`/Tinker
   (`--lora-rank`); FWFT cells train via an **Unsloth** script deployed on RunPod
   **through bellhop** (check code in → run FFT → pull HF checkpoint → check out).
   Both emit HF checkpoints served by the *same* `scimt.eval.sample` (vLLM), so the
   classifier and Pareto metric are identical across methods.
