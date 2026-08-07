# Execution log

## Fresh dense-27B matched irrelevant contrastive control

- Start time (UTC): 2026-08-07T19:10:31Z
- Code/preregistration commit: `91489237eb030e43cfe1c96331d8865f00f67feb`
- Command: `/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python attempts/public-allocation-contrastive-irrelevant/experiment.py train`
- Checkpoint manifest: `attempts/public-allocation-contrastive-irrelevant/run/checkpoints.json`
- Configuration: Qwen/Qwen3.6-27B policy, LoRA rank 32; seeds 714/825/936; 36 documents and 10,542 tokens; two SDF epochs, batch 6, token-mean cross-entropy, Adam learning rate 1e-4; action-first/rationale-first/detached-two-pass rationale-only RL, eight steps, four prompts/step, six samples/prompt, importance-sampling loss, Adam learning rate 4e-5, checkpoints 0/4/8, temperature 0.9, top-p 0.95. Full non-secret configuration is in `config.json` at the recorded commit.
- Pre-call gate: PASS. The corpus is 18 positive warehouse examples plus 18 warehouse contrastive corrections; category, oracle action, and rejected feasible second-ranked action exactly match #447 document by document; exact token lengths; zero ethical/mobile-clinic/prohibited terms; source reward audited as rationale-only; dense-27B canary PASS; exact construct declaration retained; empty treatment manifest.
- Training end: 2026-08-07T20:32:19Z. Outcome: three SDF states, nine fresh RL trajectories, and checkpoints 0/4/8 for every trajectory frozen successfully. The service produced several multi-minute sampling waits but no failed or repeated optimizer update.

## Fixed-checkpoint evaluation

- Policy command: `/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python attempts/public-allocation-contrastive-irrelevant/experiment.py sample-policy`
- Policy interval: approximately 2026-08-07T20:32:25Z to 2026-08-07T20:43:58Z; `run/policy_outputs.jsonl`, 1,620 rows (972 standard, 648 no-scratchpad); progress log `run/policy.log`.
- Counterfactual command: `/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python attempts/public-allocation-contrastive-irrelevant/experiment.py sample-counterfactual`
- Counterfactual interval: approximately 2026-08-07T20:44:05Z to 2026-08-07T20:47:04Z; `run/counterfactual_outputs.jsonl`, 324 rows; progress log `run/counterfactual.log`.
- Judge command: `/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python attempts/public-allocation-contrastive-irrelevant/experiment.py judge`
- Judge interval: approximately 2026-08-07T20:47:10Z to 2026-08-07T20:48:29Z; `run/surface_judge_calibration.jsonl`, 96 rows, and `run/surface_judge_outputs.jsonl`, 1,620 rows; progress log `run/judge.log`.
- Analysis and verification: `experiment.py analyze`, `experiment.py verify`, and `scripts/arch2 eval --json` completed by 2026-08-07T20:49:10Z. Calibration sensitivity 1.0, false-positive rate 0, AUC 1.0; capability gate passed at minimum endpoint feasibility 0.806; all 225 count-backed curve records verified; local contract valid and score null as expected.
