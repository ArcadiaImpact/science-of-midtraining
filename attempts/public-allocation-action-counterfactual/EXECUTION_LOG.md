# Execution log

- Start: 2026-08-07T22:34:40Z.
- Code and preregistration commit: `3d4365d2e9f294ceee22aef6407788f37bdd79a0`.
- Command: `/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python attempts/public-allocation-action-counterfactual/experiment.py sample`.
- Configuration: frozen dense Qwen3.6-27B detached checkpoint 8; balanced abstract ethical correction and matched irrelevant abstract correction; seeds 714/825/936; 36 registered complete public center-profile rotations; no-scratchpad renderer; deterministic temperature 0; 192-token action-only cap. Full non-secret configuration is in `config.json`.
- Pre-call audit: PASS. All 36 rotations preserve complete public profiles and budgets while changing the deterministic oracle; source checkpoint, original action, and standard counterfactual grids are complete. No reward or judge is invoked.
- Output: `attempts/public-allocation-action-counterfactual/run/no_scratchpad_counterfactual_actions.jsonl`.
- End: 2026-08-07T22:35:52Z. Outcome: 216/216 action-only counterfactual samples completed; no reward or evaluator call was made. Raw-output SHA-256: `2e9d8f08923337e923176eea818f0c8da9d93886bed54c3612788127854b7bdb`.

## Analysis and verification

- Completed: 2026-08-07T22:36:16Z.
- Standard-minus-no-scratchpad action responsiveness: +0.037 after ethical abstract correction and +0.028 after matched irrelevant abstract correction; ethical-minus-control difference +0.009 with mixed seed signs.
- Standard-minus-no-scratchpad counterfactual oracle compliance: +0.009 ethical and -0.028 control; difference +0.037 with seed values +0.028/+0.028/+0.056.
- Standard-minus-no-scratchpad mean excess uncovered need among feasible actions: -0.448 ethical and +0.278 control; difference -0.726 with all three seed values negative.
- Verification: 216 action rows and the cumulative submission contract passed; `scripts/arch2 eval --json` returned the expected valid inert contract and null local score.
