# How robust is a midtraining install? A four-axis robustness profile

**Model:** `Qwen/Qwen3-14B` · single seed (0), n=16 samples/probe · **Phase 1**
of [spec.md](spec.md) · follow-up to
[lora_artifact_robustness](../lora_artifact_robustness/report.md)

Instead of one stressor, every install is scored on a **profile**
`R = (R_benign, R_adv, R_prompt, R_perturb)` — survival under (a) benign
unrelated finetuning, (b) targeted corrective finetuning, (c) in-context
pressure with no weight change, (d) Gaussian noise on the effective install
delta ΔW. Each score ∈ [0, 1] (1 = fully robust), capability-guarded
(points where MMLU+GSM8K retention < 0.9 are masked), and censored curves
score 1.0 with a flag rather than silently maxing. Full definitions in
[spec.md](spec.md).

## Results (grid: 2 beliefs × {LoRA-r8, LoRA-r256, FWFT@1e-4} + references)

| cell | R_benign | R_adv | R_prompt | R_perturb |
|---|---|---|---|---|
| ed / LoRA r8 | 0.64 | **0.26** | 0.94 | 1.0 ⌀ |
| ed / LoRA r256 | 0.69 | **0.32** | 0.92 | 1.0 ⌀ |
| ed / FWFT | 0.47 | 1.0 ⌀* | 0.93 | 1.0 ⌀ |
| qe / LoRA r8 | 0.76 | 1.0 ⌀ (B→0.18) | 0.99 | 1.0 ⌀ |
| qe / LoRA r256 | 0.73 | 1.0 ⌀ (B→0.57) | 1.00 | 1.0 ⌀ |
| qe / FWFT | 1.00 | 1.0 ⌀ (B→0.78) | 0.74 | 1.0 ⌀ |
| prompted (ref) | ~1.0 | — | **0.39 / 0.61** | — |
| base (ref) | no-install ✓ | — | no-install ✓ | — |

⌀ = censored (curve never crossed its threshold at capability-valid points).
\* ed/FWFT's belief did collapse under corrective FT, but only at points where
capability had also dropped below the guard — the removal "cheated", so the
cell is censored, not scored as fragile.

## The eval suite works (sanity checks)

1. **Negative control** — the base model stays at B≈0 under every stressor;
   nothing *creates* the belief.
2. **Fragility floor** — the prompted organism is the weakest cell on the
   prompt axis (0.39/0.61 vs ≥0.74 for every weight install) and untouchable
   by benign weight training, both as predicted.
3. **σ=0 identity** — perturbation curves anchor at the install's own B(0).
4. **The specificity control earned its keep.** Two of the four pressure
   protocols — `context` (prepend a true encyclopedia paragraph) and
   `authority` (system prompt citing authoritative sources) — flip a TRUE
   control fact ("Bolt won the 2012 100m") 90–100% of the time. They measure
   context-following/sycophancy, not install robustness, and were
   **auto-dropped from R_prompt everywhere**. The two clean protocols:
   `skeptic` (never flips the control) and `challenge` (≤13% except one drop
   at qe/r256). Without this control, R_prompt would be meaningless.

## Findings (directional — 1 seed)

1. **The axes dissociate.** Weight installs survive in-context pressure
   almost fully (R_prompt 0.74–1.0) while varying enormously under
   weight-space attack (R_adv 0.26→censored; R_benign 0.47→1.0). "Is this
   install robust?" has no single-number answer; H1 of the spec is supported.
2. **The published benign-FT rank effect looks much weaker under the
   upgraded metric.** At n=16 (vs the published n=3): ED B(5) is r256 0.66 vs
   r8 0.57 — same direction, far smaller than the published 0.70 vs 0.40 —
   and QE is flat-to-reversed (0.72 vs 0.76). The headline "rank is a clean
   dial" survives directionally on ED only; seeds must settle it.
3. **Rank shows up cleanly on the *adversarial* axis instead.** QE removal
   curves order r8 (→0.18) < r256 (→0.57) < FWFT (→0.78) at matched
   corrective-chain length — targeted correction discriminates install
   method better than benign FT does.
4. **ΔW-scaled noise up to σ=0.2 does not touch any install** (B flat,
   capability intact, all cells censored). Either installs are genuinely
   noise-robust or the grid is too weak — needs σ ∈ {0.5, 1, 2}.
5. **The fact dominates the method.** QE is more robust than ED on every
   weight axis, consistent with the published ED/QE divergence.

## Eval refinements the data asks for

- **Extend the σ grid** (all perturb cells censored).
- **Lengthen the corrective chain** for QE (3 of 6 adv cells censored — the
  84-row × 2-epoch chain is too short to reach τ=0.10).
- **Capability guard is twitchy at n=80 items**: ±0.05 sampling wobble
  triggers masks (e.g. ed/r8's benign endpoint). Raise capability items
  and/or report fixed-endpoint comparisons alongside last-valid-point.
- **Seeds** (≥3) before any claim about finding 2 — it revises a published
  result.

## Reproduce

Same stack as the published study (ephemeral RunPod B200s via bellhop; Unsloth
train + in-process sample on-pod; classified locally). Prereqs identical to
[lora_artifact_robustness](../lora_artifact_robustness/report.md#reproduce).
From `experiments/robustness_evals/`:

```bash
python run_profile.py --facts ed,qe --installs lora:r8,lora:r256,fwft \
  --refs prompted,base --n-belief 16 --concurrency 6 --pod-budget 28800 \
  --out runs/phase1
# re-score pulled rows without GPUs:
python run_profile.py --facts ed,qe --installs lora:r8,lora:r256,fwft \
  --refs prompted,base --score-only --out runs/phase1
```

The consolidated outputs are committed at
[`results/phase1_profiles.json`](results/phase1_profiles.json) /
[`results/phase1_results.jsonl`](results/phase1_results.jsonl); every number
above regenerates from them. Raw per-cell response rows (117 MB):
`gs://alignment-team-general-storage/daniel/jarvis/experiments/robustness-evals/phase1/`.

**Harness:** `run_profile.py` (bellhop fan-out, one pod per cell, stagehand
monitors + live dashboard) · `pod/pressure_eval.py` (two-pass protocol
sampling) · `pod/perturb_eval.py` (ΔW-space noise, in-place per σ) ·
`../lora_artifact_robustness/pod/robust_ft.py` (install + benign/corrective
attacks; `--eval-every-steps` for steps-to-τ resolution) · `scimt.robust`
(pure protocol builders + guarded scoring; CPU-tested).
