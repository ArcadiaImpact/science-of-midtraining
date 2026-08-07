# Execution log

## Frozen-policy hybrid sampling

- Start: 2026-08-07T22:23:35Z.
- Code and preregistration commit: `c2ef82adee2b931707187b5aed8365031494e7fb`.
- Command: `/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python attempts/public-allocation-hybrid-scratchpad/experiment.py sample`.
- Non-secret configuration: dense `Qwen/Qwen3.6-27B`; conditions balanced abstract ethical correction and matched irrelevant abstract warehouse correction; fresh trajectory seeds 714/825/936; fixed detached checkpoints 0/8; 36 complete public executable cases; temperature 0; 160-token standard scratchpad cap; 256-token rationale cap. Exact configuration is in `config.json`.
- Planned outputs: `attempts/public-allocation-hybrid-scratchpad/run/policy_outputs.jsonl`, followed by rationale-only independent-judge outputs in the same untracked run directory.
- Pre-call construct audit: PASS. It verified 36 documents per corpus and quoted positive and correction examples; exact frozen 27B checkpoint manifests; 432 source action rows per condition; 36 public executable cases; no action resampling; and reward signature `rationale_reward(rationale: str) -> float` with no action, allocation, case, oracle, outcome, scratchpad, reasoning, or judge input.
- First invocation outcome: failed before any rationale sample completed or output row was written because compact source policy rows store parsed `allocation` but omit the redundant `allocation_text` field. The repair serializes that retained exact parsed action for the detached rationale prompt; no action is resampled or changed. A new committed execution SHA is recorded below before retry.
- Retry start: 2026-08-07T22:26:30Z from committed repair `34f6facba3712cbe553a0fbbb0fef183cd5a52f1`, using the identical non-secret configuration and command above.
- Hybrid sampling end: 2026-08-07T22:28:33Z. Outcome: 864/864 rationales completed across both conditions, both hybrid modes, seeds 714/825/936, checkpoints 0/8, and 36 cases. No action was sampled.

## Independent surface evaluation

- Start: 2026-08-07T22:28:40Z.
- Code commit: `34f6facba3712cbe553a0fbbb0fef183cd5a52f1`.
- Command: `/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python attempts/public-allocation-hybrid-scratchpad/experiment.py judge`.
- Configuration: independent `Qwen/Qwen3-30B-A3B-Instruct-2507`, deterministic temperature 0, 192-token cap, fixed rationale-only prompt and score threshold. The evaluator receives rationale text but not action, case, oracle result, condition, source renderer, proxy reward, or scratchpad.
- Output: `attempts/public-allocation-hybrid-scratchpad/run/surface_judge_outputs.jsonl`.
- End: 2026-08-07T22:29:37Z. Outcome: 864/864 independent rationale-only judgments completed.

## Analysis and verification

- Completed: 2026-08-07T22:30Z.
- Exact action reuse: 864/864 hybrid allocations and oracle labels match the registered source rows.
- Standard action context: ethical-minus-control joint interaction +0.074 with a standard rationale and +0.148 with a no-scratchpad rationale; conditional interactions +0.103 and +0.187.
- No-scratchpad action context: joint interaction +0.000 with a standard rationale and -0.037 with a no-scratchpad rationale; conditional interactions -0.012 and -0.058.
- Judge calibration: sensitivity 1.000, false-positive rate 0.000, AUC 1.000.
- Verification: 864 policy rows, 864 judge rows, 24 new exact count/rate curve cells, 279 cumulative curve cells. `scripts/arch2 eval --json` returned a valid inert contract and the expected null local score.
- Frozen raw-output hashes (not committed because they contain verbose model generations): policy outputs `a1bff89a4a5654ee763ac5e7e3ce4172703832e196ecb321461ed43ad07e8a95`; independent surface judgments `3e957500afe05ebd9d533edf5548b0c1ff9c70c80f2db72cbfcd6d422a715fdb`.
