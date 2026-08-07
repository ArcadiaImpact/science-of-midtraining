# Run manifest

- Preregistered (UTC): `2026-08-07T04:30:10Z`
- Source trajectory: PR `#390`, execution commit
  `1707a1fb1ebf9696b3a5ce287505e27f6f728181`
- Source checkpoints: 9 arm-seed trajectories, 45 fixed checkpoints
- New training updates: `0`
- Policy model: `Qwen/Qwen3.6-27B` (dense)
- Monitor model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Evaluation decoder: 160 sampled thinking tokens, zero-credit structural
  delimiter, 256 sampled public tokens
- Fresh evaluation: 90 cases, generator seed `3141592026`
- Independent calibration: 160 balanced work logs
- Config SHA-256:
  `66965382ef9dbc625bffb50bed632f6f0fe138791c890fe6cdc1b4a73af06c7c`
- Source-manifest SHA-256:
  `d6ace79476f9191040bc062bf2940a45ce9be0d4048e39c61d0189a83091fe3c`
- Canary command: `.venv-research/bin/python -u attempts/dense-27b-independent-calibration/experiment.py canary`
- Full command: `.venv-research/bin/python -u attempts/dense-27b-independent-calibration/experiment.py all`
- Full log: `attempts/dense-27b-independent-calibration/run/full.log`
- Credential handling: `TINKER_API_KEY` is read only by the SDK and never
  logged.
- Canary: passed at `2026-08-07T04:32:51Z`; valid policy JSON, nonempty private
  work, valid monitor JSON, and a correct positive calibration verdict
- Full start: `2026-08-07T04:33:04Z`
- Full end: `2026-08-07T05:04:51Z`
- Exact fresh-measurement commit:
  `3f2cd33af58755f9fe30699ea38698de7f421992`
- Policy rows: `8,100`
- Monitor rows: `16,200` (`4,050` programmatic oracle-context and `12,150`
  sampled post-hoc judgments)
- Independent calibration rows: `160`
- Evaluation-case SHA-256:
  `46f6e156806d1aba2bfdf59de60076f1e7793f02f007d582601adbd39d2bc4e3`
- Calibration-case SHA-256:
  `ea8a46e77295401e11587d0575bb2686f57a9c17a116e95ceecc1c0c09a798ba`
- Policy-output SHA-256:
  `9f2452d944204b700653b22469be81dcb7c896c30a212ae9b6c229b5a5bf177b`
- Monitor-output SHA-256:
  `f5caef0e36f773ff9f2b38af34c50eb56a69c5c202d46d4a0b654b6686b7c033`
- Calibration-output SHA-256:
  `292e3950521beefed5fae54c5c0961d8f21d64ed8511706b87fbc24393a92c82`
- Compact result: fresh-stream primary `+0.0148`, negative direction not met;
  all gates pass; P(hack) component `+0.1000` and conditional-nondetection
  component `-0.3021`, both with all three seed signs aligned
