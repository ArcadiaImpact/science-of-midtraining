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
- Policy rows: 8,100
- Policy-monitor rows: 16,200
- Calibration rows: 160
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

Frozen evaluation execution:

- Evaluation commit: `4cb1fb7d493d1f59761a8b306eea25f41c6d77e1`
- Evaluation canary start/end: `2026-08-07T02:26:50.642954+00:00` / `2026-08-07T02:27:04.444417+00:00`
- Full evaluation start/end: `2026-08-07T02:27:13.124357+00:00` / `2026-08-07T02:55:52.162192+00:00`
- Policy rows: 8,100; SHA-256 `3595377021b55c9530cde4de1257e83b2df4f637f38a54a350817f536cb6f95a`
- Monitor rows: 16,200; SHA-256 `4bd2778a07a620abc08b195fab49ff46960fa713ec0851d20d39ff7e33a1fb6f`
- Calibration rows: 160; SHA-256 `73666f2f0f084e55ea2d97abacd9204f66b8615e3307663013f812a8ff0957a7`
- Result SHA-256 before documentation: `56393ce6477a398ac9f369d93b2f5ef0b239b20c8eb5cf37bd711dd1264e549c`
- Curves SHA-256: `be3c906aa1094f23dd9a154586b82a3ece94e72bc06562b1a2e5088aa173ff10`
- Figure SHA-256: `19d23de7939a0de24a398332ef86882368c309a3d1d4a2d38921b5e397e329a2`
- Full evaluation log SHA-256: `27061075f6711762c7e7970510819e160a569a197844352508bee93377751581`
- Evaluation canary log SHA-256: `3fff57eca8151efb8b2b8fa048b4cc48c8a18c6258c91f870a4156afb0346c54`

Raw provider rows are omitted from the submission but their hashes are embedded in the compact result. Logs contain no credentials or environment dump.
