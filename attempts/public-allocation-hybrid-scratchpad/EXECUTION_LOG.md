# Execution log

## Frozen-policy hybrid sampling

- Start: 2026-08-07T22:23:35Z.
- Code and preregistration commit: `c2ef82adee2b931707187b5aed8365031494e7fb`.
- Command: `/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python attempts/public-allocation-hybrid-scratchpad/experiment.py sample`.
- Non-secret configuration: dense `Qwen/Qwen3.6-27B`; conditions balanced abstract ethical correction and matched irrelevant abstract warehouse correction; fresh trajectory seeds 714/825/936; fixed detached checkpoints 0/8; 36 complete public executable cases; temperature 0; 160-token standard scratchpad cap; 256-token rationale cap. Exact configuration is in `config.json`.
- Planned outputs: `attempts/public-allocation-hybrid-scratchpad/run/policy_outputs.jsonl`, followed by rationale-only independent-judge outputs in the same untracked run directory.
- Pre-call construct audit: PASS. It verified 36 documents per corpus and quoted positive and correction examples; exact frozen 27B checkpoint manifests; 432 source action rows per condition; 36 public executable cases; no action resampling; and reward signature `rationale_reward(rationale: str) -> float` with no action, allocation, case, oracle, outcome, scratchpad, reasoning, or judge input.
- First invocation outcome: failed before any rationale sample completed or output row was written because compact source policy rows store parsed `allocation` but omit the redundant `allocation_text` field. The repair serializes that retained exact parsed action for the detached rationale prompt; no action is resampled or changed. A new committed execution SHA is recorded below before retry.
