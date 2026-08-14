# LoRA-GRPO parameterization control

## Question

Does restricting reasoning-RL updates to LoRA adapters change how strongly the
four restored ReFT parents preserve or reverse their Coin/Charter behavior?

## Factorial

- Parents: Charter, Coin, Mixed 50:50, Neutral restored ReFT checkpoints.
- Rewards: Agreement, Coin-only conflicts, Charter-only conflicts.
- Cells: 12, all seed 42.
- Dose: exactly 64 optimizer updates / 2,048 sampled completions per cell.
- Evaluation: the existing frozen direct and thinking battery, including raw
  thinking traces and parse failures.

Every cell starts from the same immutable ReFT checkpoint, dataset, order seed,
temperature, reward implementation, group size, and global batch as its matched
full-parameter cell. The only intended optimizer-path change is LoRA.

## Adapter and optimizer recipe

- BF16, no base-weight quantization.
- Rank 32, alpha 64, dropout 0, bias none, zero-effect initialization.
- Exact language-layer q/k/v/o and gate/up/down projection paths only; assert no
  trainable vision, multimodal, embedding, normalization, or LM-head parameter.
- GRPO `dr_grpo`, beta 0, temperature 1.0, group size 8, one-H200 microbatch 4,
  accumulation 8, global completion batch 32, maximum prompt 3,072 tokens, and
  maximum completion 1,024 tokens.
- Retain only the update-64 adapter. Retain no intermediate, optimizer, trainer,
  or merged full-model state; raw rollouts provide the learning trajectory.

Before the main grid, calibrate `2.5e-6`, `5e-6`, and `1e-5` for 16 updates on
Neutral/Agreement. Lock the smallest stable rate within 0.02 reward of the best
stable arm; use `5e-6` if calibration cannot choose.

## Required integrity evidence

1. Parent revision and deterministic checkpoint-tree hash.
2. Exact target list and trainable-parameter manifest.
3. Zero-effect initialization, adapter-only trainability, and nonzero adapter
   mutation check.
4. HF+PEFT versus temporary merged-model fixed-logit and greedy equivalence.
5. Temporary merged-model vLLM reload before endpoint evaluation.
6. Raw rollout/evaluation traces, package lock, rendered config, source commit,
   GPU inventory, adapter and evidence hashes, and remote verification records.

Final adapters must first be staged beneath Bellhop's result directory so they
are copied onto the persistent `/workspace` volume before pod teardown.
Adapters and logs must then be uploaded to Arcadia Impact Hugging Face
repositories and remotely verified. If organization billing rejects the upload,
record the exact failure alongside the locally retained artifacts and retry once
billing is repaired. Temporary merged 53 GB models must be removed after
evaluation; they are not durable artifacts.

## Interpretation

Primary analysis is within-LoRA parent contrasts and matched cell-by-cell
LoRA-versus-full-parameter differences. The Neutral row is the parameterization
bridge. This seed-42 grid is exploratory: policy-gradient noise is not estimated,
and greater parent retention under LoRA is not by itself a stronger learned
midtraining prior.

The detailed design and execution plan are in
`docs/plans/2026-08-06-dispatch-lora-grpo-12-cell-design.md` and
`docs/plans/2026-08-06-dispatch-lora-grpo-12-cell.md`.
