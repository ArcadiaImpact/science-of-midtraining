# Run manifest

- Preregistered (UTC): 2026-08-06T06:12:28Z
- Start time (UTC): 2026-08-06T06:13:33.911902Z
- End time (UTC): 2026-08-06T07:30:28.167360Z
- Source PR: `#370`
- Source PR head: `63663418bac1c7b5aa732742983c4bfcb305e1d8`
- Source trajectory commit: `6e6ca48e86022728c353a79043f56170fdc0176a`
- Source checkpoint manifest SHA-256: `1876c84cd9f62e9121dba1e62dab7cee2b663a41443eb5f464f9aba9d5e8a0d9`
- Config SHA-256: `b0b9406248c3064a19a73a2cff63a7b9d3104d4df6e80a761ed317902f876745`
- Corpus SHA-256: `4e897e6fd32c304378a25415fed3f175a0f8f92078e77788a0300b2c3c5ac566`
- Evaluation SHA-256: `07606287195dede6afea035711abcaecf4c75beccc22564ce8aeb5a883f2e679`
- Command: `.venv-research/bin/python -u attempts/values-rationales-decomposition/experiment.py all`
- Standard log: `attempts/values-rationales-decomposition/run/full.log`
- PID file: `attempts/values-rationales-decomposition/run/full.pid`
- Outputs: `attempts/values-rationales-decomposition/run`, `submission/results.json`, `submission/curves.json`
- Policy rows: 13,500
- Monitor rows: 27,000 (20,250 learned-monitor and 6,750 exact oracle-context rows)
- `submission/results.json` SHA-256: `fc174be1287438b7ca3f1f089a4959a241b5ae6ec8ff5b8b27bd077d3e4cda73`
- `submission/curves.json` SHA-256: `dfe7e6cca591117cb9eaf761b5790e54ee818cdd40af6b69def72613e02ea78c`
- Credential handling: `TINKER_API_KEY` is read by the SDK only and is not logged.

Three existing semantic trajectories are reused exactly. Only values-only and
rationales-only SDF/RL trajectories are newly trained.
