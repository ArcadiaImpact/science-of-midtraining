# Run log: public cooling-plan selection

## 2026-08-07 — pre-call plan

- Policy: dense `Qwen/Qwen3.6-27B`, rank-32 LoRA, seeds 1729/2831/3943.
- SDF: values-and-rationales, rules-only, matched irrelevant; 48 documents per
  arm, exact per-document tokens, three token-mean cross-entropy epochs, batch
  size 8, Adam 1e-4.
- RL: action-first rationale-only reward, 8 scheduled batches, 6 prompts x 8
  rollouts, sequence-wide importance sampling, Adam 4e-5, checkpoints 0/4/8.
- Evaluation: 24 public outcome-intervention pairs, independent base Qwen3-30B
  post-hoc surface judge, action-first/no-work/rationale-first/detached controls.

Paid calls are blocked pending committed code, deterministic preparation,
literal three-arm corpus inspection, and a frozen construct audit.
