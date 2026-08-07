# Execution log

## Fresh dense-27B balanced treatment

- Start time (UTC): 2026-08-07T17:55:19Z
- Code/preregistration commit: `b4704bf53faba4fb34446abd430ae5e75a4ee04c`
- Command: `/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python attempts/public-allocation-balanced-values/experiment.py train`
- Log: `attempts/public-allocation-balanced-values/run/train.log`
- Checkpoint manifest: `attempts/public-allocation-balanced-values/run/checkpoints.json`
- Configuration: Qwen/Qwen3.6-27B policy, LoRA rank 32; seeds 714/825/936; 36 documents, 10,542 tokens, two SDF epochs, batch 6, token-mean cross-entropy, Adam learning rate 1e-4; action-first/rationale-first/detached-two-pass rationale-only RL, eight steps, four prompts/step, six samples/prompt, importance-sampling loss, Adam learning rate 4e-5, checkpoints 0/4/8, temperature 0.9, top-p 0.95. The exact full non-secret configuration is in `config.json` at the recorded commit.
- Pre-call gate: `generated/construct_audit.json` PASS; 18 positive-only and 18 contrastive-correction examples; every rejected action feasible and second-ranked; every correction oracle-compliant; exact token match; zero prohibited terms; source reward input audited as rationale only; dense-27B canary PASS; empty treatment manifest.
- Detached wrapper note: the initial `nohup` wrapper exited at 2026-08-07T17:55:32Z after creating an empty manifest but before any training client or checkpoint. Training was relaunched once in a polled persistent terminal from the same recorded code state; the empty manifest was reused and no completed work was duplicated.
- Training end: 2026-08-07T18:40:42Z. Outcome: three SDF states, nine fresh RL trajectories, and checkpoints 0/4/8 for every trajectory frozen successfully.

## Fixed-checkpoint evaluation

- Policy sampling command: `/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python attempts/public-allocation-balanced-values/experiment.py sample-policy`
- Policy sampling interval: approximately 2026-08-07T18:40:50Z to 2026-08-07T18:57:49Z; output `run/policy_outputs.jsonl`, 1,620 rows (972 standard and 648 no-scratchpad); progress log `run/policy.log`.
- Counterfactual command: `/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python attempts/public-allocation-balanced-values/experiment.py sample-counterfactual`
- Counterfactual interval: approximately 2026-08-07T18:57:55Z to 2026-08-07T19:01:39Z; output `run/counterfactual_outputs.jsonl`, 324 rows; progress log `run/counterfactual.log`.
- Independent judge command: `/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python attempts/public-allocation-balanced-values/experiment.py judge`
- Judge interval: approximately 2026-08-07T19:01:45Z to 2026-08-07T19:03:59Z; outputs `run/surface_judge_calibration.jsonl` (96 rows) and `run/surface_judge_outputs.jsonl` (1,620 rows); progress log `run/judge.log`.
- Analysis/verification commands: `experiment.py analyze`, `experiment.py verify`, and `scripts/arch2 eval --json`.
- Analysis end: 2026-08-07T19:04:30Z. Judge calibration sensitivity 1.0, false-positive rate 0, AUC 1.0; capability gate passed with minimum endpoint feasible-allocation rate 0.806; all count/rate identities and the 180-record merged curve grid verified. Local artifact contract valid; local score null as expected.
