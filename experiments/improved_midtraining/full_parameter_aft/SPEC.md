# Full-parameter Dispatch AFT trajectory

## Question

How do the final Coin- and Charter-midtrained SFT parents respond when the
long agreement-only Dispatch AFT trajectory updates the full language model
instead of rank-64 LoRA parameters?

This is a two-arm, single-seed robustness extension. It is a practical-recipe
comparison, not a parameterization-only causal ablation: the established LoRA
run used `1e-4` with 5% warmup and cosine decay, whereas this run uses the
lower full-tuning rate `5e-6` with no warmup and a constant schedule. Compare
the trajectories both at equal optimizer step and at nearest held-out
agreement accuracy.

## Immutable inputs

- Parent repository: `jbostock/scimt-dispatch-models-v1`
- Parent revision: `9a16b6ebe2e88b86e6c709295424df869c028d78`
- Coin parent: `sft/coin/checkpoint-48`
- Charter parent: `sft/charter/checkpoint-48`
- Data seed: `314159`
- Train data: the exact ordered 2,048-row agreement-only Dispatch dataset used
  by LoRA run `20260807T110710Z`
- Expected train-data SHA-256:
  `2220d77d4e6256aec4b67f096576d56d779336a14ddea420a0c8734b6afa616b`

The runner regenerates the established data locally, verifies its row count,
hash, seed, and train/eval leakage audits, and trains both parents on the same
bytes in the same order.

## Training recipe

- full-parameter supervised AFT with text-only examples;
- 32 epochs and exactly 2,048 optimizer steps;
- 4 GPUs per arm (H200 preferred; H100-80GB is an approved capacity fallback),
  microbatch 1/device, accumulation 8, global batch 32;
- sequence length 1,024, no packing, assistant-only loss;
- AdamW fused, weight decay `0.01`, clipping norm `1.0`;
- constant learning rate `5e-6`, zero warmup;
- bf16, TF32, SDPA, gradient checkpointing, FSDP2;
- seed `314159` for data, training, and evaluation;
- retain full-state checkpoints at steps 4, 8, 16, 32, 64, 128, 256,
  512, 1,024, and 2,048.

All language-model weights that receive gradients are updated, including
embeddings, norms, attention/MLP weights, and the language-model head. The
unused vision tower receives no text-only forward-pass gradients. The run
must emit a finite-loss `training_started.json` marker before launch
scaffolding can be removed.

## Evaluation

Evaluate the unchanged SFT parent and every full checkpoint using greedy,
seeded vLLM inference:

- 512 held-out agreement episodes;
- 512 held-out conflict episodes, including Charter/Coin/Other/malformed and
  clause-stratified metrics;
- the fixed 40-row MMLU and 40-row GSM8K generic battery;
- mechanical empty, truncation, repetition, exact-duplicate, parseability,
  and Dispatch-intrusion collapse diagnostics.

The agreement data admit a shortcut: every demonstrated winner is both the
minimum-total-quote plan and the lexicographic Charter-precedence winner, and
all demonstrated crews qualify. High agreement accuracy therefore does not
show that the full Charter rule was learned. Report the previously diagnosed
priority-only and qualification-conflict behavior when interpreting the
trajectory.

## Reproducibility and publication

The launcher refuses a dirty or unpushed tree and records the exact Git
commit, tree, per-file source manifest, resolved Axolotl configuration, parent
Hub revision and remote file identities, dataset bytes and audits, package
locks, GPU metadata, full training log/trace, raw generations, metrics, and
remote publication receipts.

- Models: `jbostock/scimt-dispatch-models-v1`, under
  `full_aft/<coin|charter>/checkpoint-*`.
- Evidence: `arcadia-impact/scimt-dispatch-aft-v1`, under
  `full_parameter_runs/<run_id>/<coin|charter>`.

Each arm runs on a synchronously Bellhop-owned four-GPU pod. The validated model
checkpoints are published before evaluation so a downstream vLLM failure
cannot destroy the expensive training result. Coin and Charter publish
independently with optimistic Hub parent-commit checks, refreshed retries, and
exact returned-revision verification; no healthy arm waits idly for its peer.

The launcher records the realized accelerator model and requires at least
130 GB/device for H200 or 75 GB/device for H100. This makes a fallback explicit
rather than silently accepting a lower-memory or mislabeled allocation.

## Closest prior work

Li et al., *Model Spec Midtraining* (2026) is the closest conceptual
predecessor. Biderman et al., *LoRA Learns Less and Forgets Less* (2024) and
Shuttleworth et al., *LoRA vs Full Fine-tuning: An Illusion of Equivalence*
(2024) motivate measuring learning and retention across the whole trajectory,
not assuming equal in-distribution accuracy implies equivalent conflict
behavior. This single seed remains preliminary.
