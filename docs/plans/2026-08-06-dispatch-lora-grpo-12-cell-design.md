# Dispatch LoRA-GRPO 12-cell design

## Objective

Repeat the completed seed-42 full-parameter reasoning-RL grid with LoRA policy
updates. The factorial is three rewards (Agreement, Coin-only, Charter-only) by
four restored ReFT parents (Charter, Coin, Mixed 50:50, Neutral), for 64
optimizer updates per cell. This is an exploratory one-seed parameterization
control, not a new estimate of seed variance.

## Scientific comparison

Each cell starts from the same immutable restored ReFT checkpoint used by its
full-parameter counterpart, uses the same data, order seed, rollout temperature,
group size, effective batch, reward function, and 64-update dose. The only
intended training change is that the policy update is restricted to LoRA
parameters. The Neutral row is retained as the bridge needed to separate an
adapter effect from a midtraining-parent effect.

The primary comparisons are parent-relative contrasts within the new LoRA grid
and matched LoRA-versus-full-parameter cells. LoRA preserving a parent behavior
more strongly is not itself evidence that the parent learned a stronger prior.

## Locked adapter recipe

- BF16 base weights; no quantization.
- LoRA rank 32, alpha 64, dropout 0, no bias, standard zero-effect
  initialization.
- Explicitly target q/k/v/o attention projections and gate/up/down MLP
  projections under all language-model decoder layers.
- Discover and record exact full target names. Assert that no trainable
  parameter belongs to the vision tower, multimodal projector, embeddings,
  normalization layers, or LM head.
- Disable model dropout during GRPO.
- Retain the existing `dr_grpo`, beta 0, temperature 1.0, group size 8,
  32-completion global optimizer batch, maximum prompt/completion lengths, and
  reward-standardization behavior for comparability. Use microbatch 4 and
  accumulation 8 on one H200; this preserves the effective batch while avoiding
  the batch-1/accumulation-32 overhead that LoRA memory savings make unnecessary.

Rank 32/alpha 64 follows the LoRA replication already proposed in
`experiments/prior_coins/GRPO_AFT_PLAN.md`. Dropout is changed from the earlier
supervised LoRA recipe because stochastic policy layers can create current/reference
log-probability discrepancies in online RL.

## Learning-rate calibration

Before the grid, run 16 updates from the Neutral parent on Agreement data at
`2.5e-6`, `5e-6`, and `1e-5`, from identical initialization and data order.
Select the smallest rate that has finite loss and gradients, changes adapter
weights, leaves frozen tensors unchanged, and shows non-inferior late-window
reward without pathological clipping or length growth. The preregistered
fallback is `5e-6` (10 times the full-parameter rate). Lock one selected rate
for all 12 main cells; calibration weights are diagnostic and are not continued
into the grid.

## Integrity gates

Before accepting paid main runs:

1. At update zero, enabling the zero-initialized adapter must leave fixed-input
   logits numerically equivalent to the parent.
2. After calibration, adapter tensors must change and the trainable manifest
   must contain no base-model tensor.
3. The unmerged HF+PEFT model and a temporary `merge_and_unload()` checkpoint
   must agree on fixed-input logits and greedy generations within a documented
   tolerance.
4. The merged checkpoint must reload through vLLM and exhibit the adapter's
   behavior. This prevents recurrence of the earlier inert-adapter evaluation.
5. The exact parent revision/hash, target-module manifest, trainable count,
   package lock, source commit, rendered config, raw rollouts, and per-update
   reward diagnostics must be present.

## Execution layout

Use one H200 per cell, four cells concurrently on a watched four-H200 pod. Run
three objective waves so each wave shares an immutable dataset while all four
parents train independently. Generate direct and thinking traces immediately
after each wave while temporary merged weights exist, then score traces locally.
If the one-GPU PEFT smoke exposes a backend limitation, use the smallest tested
multi-GPU layout consistently across all cells and record the deviation.

## Checkpoints and persistence

Save only the update-64 adapter. Raw rollouts provide the full learning curve;
intermediate weights would multiply artifacts without being part of the requested
endpoint comparison. Do not retain optimizer, scheduler, RNG, FSDP, or merged
full-model states after successful final-adapter verification. Adapter directories
must include their PEFT config and exact immutable base-model pointer/hash.

Stage adapter weights and manifests beneath Bellhop's returned result directory
before any external upload, then publish them to one Arcadia Impact model
repository and publish raw/scored evaluations, rollout logs, calibration
evidence, and summaries to a paired dataset repository. Verify remote sizes
when organization billing permits upload. A billing rejection must be recorded
and all staged artifacts returned to persistent `/workspace` before teardown.

## Evaluation and reporting

For every final adapter, generate the same direct and thinking evaluation
battery used for the full-parameter endpoints. Preserve raw thinking traces and
parse failures. Report Agreement accuracy and conflict Coin/Charter/malformed
rates, reward trajectories, informative versus zero-advantage groups, formatting,
truncation, completion length, entropy, gradient norm, clipping, and adapter
manifests. Produce matched LoRA-versus-full-parameter tables and the existing
3x2 Agreement/Coin/Charter panel style.
