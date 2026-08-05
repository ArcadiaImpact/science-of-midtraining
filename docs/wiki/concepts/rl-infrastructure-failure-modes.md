---
type: concept
title: RL infrastructure failure modes (TRL + vLLM + MoE)
description: "silent training killers in the TRL/vLLM GRPO stack and the first-step health checks that catch them — from the 2026-08-04 sequence_mask incident and redteam"
resource: ../../sources/prior-latmem-grpo-star-runs.md
tags: [rl, grpo, trl, vllm, moe, infrastructure, methodology]
timestamp: 2026-08-05
---

# RL infrastructure failure modes (TRL + vLLM + MoE)

RL fine-tuning stacks can fail *silently*: rewards, KL, and completions all
look plausible while the optimizer applies (nearly) no update, or applies an
update toward an objective the eval cannot see. Catalog of verified modes
from the GRPO runs on Qwen3-Coder-30B-A3B
([source](../../sources/prior-latmem-grpo-star-runs.md), redteam report in
the same experiment dir), each confirmed against TRL 1.9.2 source.

## Verified failure modes

- **Sequence-level importance-sampling collapse on MoE.** **[firm — observed
  end-to-end, mechanism verified in source]** TRL's default
  `vllm_importance_sampling_mode="sequence_mask"` weights each sequence's
  loss by `exp(Σ per-token trainer−vLLM logprob diffs)`. MoE models have a
  systematic per-token trainer↔vLLM mismatch (~−0.003/token here, from
  different expert kernels), so long completions get weight
  `exp(−0.003·L)` → ~0.02 at L≈1200. Run 1 trained 300 steps at ~2%
  gradient and evaluated identically to base. Fix: token-level correction
  (`token_truncate`) with symmetric clipping (e.g. [1/3, 3]; TRL's
  `clip_min` defaults to unbounded-below).
- **Truncation reward wedge.** **[firm]** The training reward executed
  truncated-but-salvageable completions (rewardable up to 1.0) while the
  offline scorer zeroes any `finish_reason=length` sample before extraction
  — training can reinforce behavior the eval categorically discards
  (~17–20% of samples at a 4,096 cap). The reward callable must consume
  `completion_ids` and mirror the scorer's truncation policy.
- **Fused-loss normalization bypass.** **[firm — verified in TRL 1.9.2
  source]** `compute_liger_loss` never consumes `num_items_in_batch`, so
  the documented global-token DAPO normalization silently becomes
  per-microbatch reduction; short-completion microbatches get over-weighted
  (potentially ~40× per token at per-device batch 1–2). Disable Liger
  unless a gradient-parity test passes; budget memory for the exact-loss
  logits (per-device 1 on 80GB at 4k completions × 152k vocab).
- **Unpinned data revision.** A "fixed" prompt set selected from a Hub
  repo's head at launch is not fixed; recording the resolved SHA in a
  manifest is provenance, not a pin. Pin revisions in config.
- **Checkpoint upload race under DDP.** Every rank writes its own
  `rng_state_<rank>.pth` and `Trainer._save_checkpoint` has no trailing
  barrier; an upload callback on rank 0 can snapshot before rank 1's file
  lands, and a Hub resume then silently reseeds that rank. Barrier before
  upload.

## First-step health checks (10-minute detection)

On the *first* logged optimizer step:

1. `sampling/importance_sampling_ratio/mean` ≈ 1 (run 1's read: 0.018;
   run 2 after fix: 0.96).
2. `grad_norm` not orders of magnitude below O(1e-2) (run 1: ~1e-4 from
   step 1).
3. Reward-side truncation counter > 0 whenever
   `completions/clipped_ratio` > 0 (confirms the truncation gate is live).
4. Fresh vLLM server per trainer attach; health endpoint 200 ≠ replicas
   loaded (gate on rollout-GPU memory, then settle).

## Tensions

- The IS correction exists to fix a real bias (trainer↔vLLM logprob
  mismatch); disabling it entirely trades silent gradient collapse for
  uncorrected off-policy drift. Token-level truncation with symmetric
  clipping is the current compromise; whether *no* correction beats it on
  MoE is untested. **[open]**

## Related

- [prior-latmem-aft-before-rl](../syntheses/prior-latmem-aft-before-rl.md)
  — why RL was sequenced after fixed-example AFT for the causal comparison;
  the run-2 null bounds what base-model GRPO adds under this budget.
