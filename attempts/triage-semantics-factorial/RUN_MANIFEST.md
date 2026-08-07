# Execution manifest

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

Frozen training execution:

- Successful canary commit: `7c2411fd5ab60e0b9ae8b2ab5a7b8d2e19d759b6`
- Canary start/end: `2026-08-07T02:07:40.611736+00:00` / `2026-08-07T02:08:47.670243+00:00`
- Full training commit: `5f94ae8bf6a2a9a824c7034e50c5b41bb5fc6c1f`
- Full training start/end: `2026-08-07T02:09:24.287198+00:00` / `2026-08-07T02:26:03.821977+00:00`
- Combined checkpoint manifest SHA-256: `3ea250675d8eb9af3e3320407b307f59de5ec547cd9ba86edc7af4a9d56e50a3`
- Training log SHA-256: `dc39fc76baeef046b9522bd872f26b1aa3223d610ae1bc940edfa028b127f103`
- Successful canary log SHA-256: `77913d32ad9fc2a2fba42f7bdc3e1fd64cbb4110c8f8885d8534dd20c4f080bd`

The combined manifest contains nine complete trajectories and 45 checkpoint sampler references. Evaluation times and output hashes will be appended after the strictly post-hoc run. No credential or environment dump is present; raw provider rows will remain uncommitted.
