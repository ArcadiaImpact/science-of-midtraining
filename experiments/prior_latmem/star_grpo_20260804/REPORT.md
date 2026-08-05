# GRPO on the sometimes-solved bank problems (runs 1 + 2, 2026-08-04/05)

**Bottom line: two 300-step GRPO runs on Qwen3-Coder-30B-A3B-Instruct; run 1
was invalidated by a silent infrastructure bug (trained at ~2% gradient), run
2 fixed it and trained correctly — and still landed a clean null on eval
pass@k. With the pipeline now verified end-to-end, the null is attributable
to the recipe/budget, not the plumbing.**

## Setup

- Policy: rank-32/alpha-64 attention-only LoRA on
  `Qwen/Qwen3-Coder-30B-A3B-Instruct` (rev `b2cff646`), bf16.
- Trainer: TRL 1.9.2 GRPOTrainer, 2 DDP ranks (2×H100), vLLM 0.25.1
  server-mode rollouts on the other 2×H100 (TP=1, DP=2).
- Prompts: the 354 train problems with base pass@16 count in [1,15]
  (from the 20260803 STaR sampling run, stats rev `c01d7ed2`).
- Reward: executable — 0.5×(fraction of selected tests passed, scored
  independently) + 0.5×(full gate incl. synth workload), 8s/test timeout,
  per-(problem, program-sha) cache. Run 2 additionally zeroes truncated
  completions (see F1 below).
- Sampling: provider defaults (T=0.7, top_p=0.8, top_k=20, rep_pen=1.05),
  16 generations/group, 4,096-token completion cap, generation batch 128,
  num_iterations=2, optimizer batch 64, lr 5e-6 cosine, beta=0.001,
  max_steps=300 (≈1.70 fresh-rollout epochs; 176 optimizer steps/epoch).
- Configs: `../configs/star_grpo_2026-08-04.yaml` (run 1),
  `../configs/star_grpo_run2_2026-08-04.yaml` (run 2). Runner:
  `../star_grpo.py`. Regime is model-agnostic for the planned SDF-arm reruns.

## Run 1: the sequence_mask failure

Run 1 looked healthy in every dashboard metric (reward 0.40–0.64, KL ~0.004,
completions stable) and was dead the whole time:

- TRL's vLLM↔trainer logprob-mismatch correction defaults to
  `vllm_importance_sampling_mode="sequence_mask"`: one IS weight per
  sequence, `exp(Σ per-token logprob diffs)`.
- On this MoE the trainer sits a systematic ~−0.003/token below vLLM
  (different expert kernels), so a ~1,200-token completion gets weight
  `exp(−3.6) ≈ 0.02`. Logged mean sequence IS ratio: **0.018** (max 0.50).
- That weight multiplies the loss ⇒ every gradient scaled ~50–100× down.
  grad_norm sat at 1e-4–1e-2 from step 1; effective lr ~1e-7.

Detection came from three independent nulls: a step-120 interim eval
(−0.2/+0.2/−0.3pp vs base on all 324 eval problems, n=16), a flat
epoch-paired train-reward comparison (167 twice-visited problems, delta
−0.003 [−0.024, +0.020]), and near-zero KL throughout. Diagnosis: the
tell-tale pair (IS ratio mean ≪ 1, grad_norm ~1e-4) in the step logs.

**10-minute detection rule for future runs: on the first logged step,
`sampling/importance_sampling_ratio/mean` must be ≈1 and grad_norm must not
be orders of magnitude below O(1e-2). Both are in the very first metrics
line.**

## The redteam (REDTEAM_REPORT.md)

Before relaunch, an adversarial audit (Codex, unsandboxed read-only, verified
line-by-line against the staged TRL 1.9.2 source) found, beyond the IS fix:

- **F1 (critical)**: training reward executed truncated-but-salvageable
  completions (up to reward 1.0) while both offline scorers zero them before
  extraction — a train/eval objective wedge on ~17% of samples. Fixed:
  reward now mirrors the scorer (truncated ⇒ 0.0).
- **F2 (high)**: TRL 1.9.2's fused Liger branch never consumes
  `num_items_in_batch`, silently replacing the documented global-token DAPO
  normalization with per-microbatch reduction. Fixed: Liger off (cost:
  per-device batch 1×32 instead of 2×16 to fit the exact-loss logits).
- **F3 (high)**: `star_revision` was unpinned (Hub head at launch). Fixed:
  pinned.
- **F5**: IS clip_min defaults to None (unbounded below); set [1/3, 3].
- **F7**: no cross-rank barrier before checkpoint upload; rank-1 RNG state
  could miss the Hub snapshot. Fixed: barrier in the callback.
- **F6**: 300 steps = 1.70 fresh epochs, not ~2.7 as previously claimed.
- Deferred: F4 (server/trainer model-revision handshake — matters for the
  SDF arms), F8 (reward timeout is per-test, worst case ~40s/program).

Run 2 verified all fix signatures live at step 1: IS ratio mean 0.96
(clips at 0.333/3.0), grad_norm ~1e-2 at warmup, truncated completions
zeroed (15/128 in the first batch, matching clipped_ratio exactly).

## Run 2 result: a clean null

Final adapter (step 300) evaluated with the standard protocol: n=16,
provider-default sampling, all 324 eval problems, paired bootstrap vs the
base-model anchor from the same harness (20260803 run).

| metric  | base  | GRPO run 2 | delta  | 95% CI        |
|---------|-------|-----------|--------|---------------|
| pass@1  | 23.3% | 22.8%     | −0.5pp | [−1.5, +0.4]  |
| pass@8  | 34.8% | 35.1%     | +0.3pp | [−1.9, +2.2]  |
| pass@16 | 38.9% | 38.9%     | +0.0pp | [−2.8, +3.1]  |

n = 324 problems × 16 samples = 5,184. Paired per problem: 39 improved,
47 worsened, 238 tied. (Phase-1 STaR SFT, for reference, was also a null:
+0.17pp pass@1 [−0.71, +1.06].)

Secondary observations:

- **Mild length drift**: truncation at the 4,096 cap rose to 19.7% vs the
  base model's 16.9% on the same problems (+2.8pp). Truncated ⇒ scored 0,
  so this is a small direct accuracy tax and the clearest concrete lever
  for a future run (length penalty, or larger cap at eval).
- **Training reward did not climb**: batch means oscillated 0.40–0.65 with
  no visible trend across 1.7 epochs (`run2_training_metrics.png`). The
  policy moved (KL grew to ~0.01–0.03, entropy drifted down slightly) but
  not toward higher executable reward. Per-problem reward logs were lost
  with the pod (dead-man switch); only batch aggregates survive in
  `run2_train_log.txt`.

## Interpretation

With gradients verified flowing, the null says the *recipe under this
budget* doesn't move the needle: ~300 updates at optimizer batch 64,
KL-anchored (beta 0.001) LoRA on 354 prompts, where the group-relative
advantage signal is diluted by 12.5–37.5% zero-variance groups and a shaped
reward that the base policy already averages ~0.5 on. GRPO here mostly
re-weights behavior the model already exhibits; 1.7 epochs of that neither
lifted training reward nor transferred to held-out problems.

Plausible levers, in rough order of expected information per dollar:
1. **Fix the objective/coverage first**: length penalty (the +2.8pp
   truncation tax is pure loss), and prompt-set curriculum weighting toward
   groups with reward variance.
2. **More optimization**: ≥475 steps (2.7 fresh epochs) and/or higher lr /
   larger LoRA — but only after (1), since scaling a flat gradient is
   scaling zero.
3. **Check the ceiling**: if base pass@16 correctness is already near the
   bank's reliable-test ceiling on these problems, executable-reward RL has
   nothing to teach — worth a diagnostic pass before any run 3.

## Artifacts

- Checkpoints + final adapters (both runs):
  `sidbaines/scimt-prior-latmem-attribution` under `grpo_star/20260804/`
  (run 1, dead) and `grpo_star/20260804_run2/` (run 2), every 20 steps with
  optimizer/scheduler/RNG state.
- Final eval samples + verdicts + measurements:
  `sidbaines/scimt-prior-latmem-star` (dataset) under
  `star_sampling/20260805_grpo_run2_final/`; run-1 interim eval under
  `star_sampling/20260804_grpo_interim/`.
- Redteam: `REDTEAM_REPORT.md` (this dir). Training curves:
  `run2_training_metrics.png`, raw log `run2_train_log.txt`.
