# Run log: paired public-counterfactual prosocial SDF

## 2026-08-07 — pre-call plan

- Dense policy: `Qwen/Qwen3.6-27B`, LoRA rank 32, seeds 1729/2831/3943.
- Control: frozen standard values-and-rationales trajectories from #421.
- Treatment: fresh paired public-counterfactual positive SDF, then identical
  action-first rationale-only RL.
- SDF: 48 exact-length-matched documents, 96 compliant examples, three
  token-mean cross-entropy epochs, batch size 8, Adam 1e-4.
- RL: 8 scheduled batches, 6 prompts x 8 rollouts, sequence-wide importance
  sampling, Adam 4e-5, checkpoints 0/4/8.
- Evaluation: 24 fresh public priority-swap pairs; action-first primary plus
  final no-work, rationale-first, and detached controls; independent base
  Qwen3-30B post-hoc surface evaluator.

Paid training is blocked pending a committed implementation, deterministic
preparation, literal corpus inspection, reward-boundary audit, and an amended
construct audit with hashes and representative documents.
