# Planned execution manifest

- Experiment: `meridiancare_triage_semantics_factorial_v3`
- Policy: `Qwen/Qwen3-8B`
- Monitor: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Conditions: values and rationales, rules only, irrelevant SDF
- Seeds: 481, 592, 703
- New trajectories: 3 rules-only
- Reused immutable trajectories: 6 values/irrelevant
- Checkpoints per trajectory: 0, 4, 8, 12, 16
- New SDF: 60 documents, 17,286 tokens per epoch, 3 epochs, batch 10, token-mean cross entropy, Adam 1e-4
- Output-only RL: 16 steps, 6 prompts by 8 rollouts, importance sampling, Adam 4e-5
- Evaluation: 90 fresh `triage-heldout3` cases per checkpoint and inference mode
- Independent calibration: 160 work-log probes
- Expected policy rows: 8,100
- Expected policy-monitor rows: 16,200
- Expected calibration rows: 160
- Training canary: `.venv-research/bin/python -u attempts/triage-semantics-factorial/train_rules.py canary`
- Training command: `.venv-research/bin/python -u attempts/triage-semantics-factorial/train_rules.py train`
- Evaluation canary: `.venv-research/bin/python -u attempts/triage-semantics-factorial/experiment.py canary`
- Evaluation command: `.venv-research/bin/python -u attempts/triage-semantics-factorial/experiment.py all`

The exact execution commit, input hashes, commands, UTC times, output paths, and compact output hashes will be added after execution. Long commands will use pidfiles and logs and be polled. No credential or environment dump is permitted; raw provider rows will remain uncommitted.
