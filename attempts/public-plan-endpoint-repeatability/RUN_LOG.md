# Run log: frozen-endpoint evaluation repeatability

## 2026-08-07 — pre-call plan

- Policy: dense `Qwen/Qwen3.6-27B` frozen step-8 adapters.
- Conditions: values-and-rationales rationale-only RL, single-round sparse
  process RL, and bounded-resampling sparse process RL.
- Training seeds: 1729, 2831, 3943.
- Evaluation: three fixed sampling seeds × 48 public cases, temperature 0,
  top-p 1, 384 tokens; 1,296 policy responses total.
- Post-hoc judge: `Qwen/Qwen3-30B-A3B-Instruct-2507`, action-withheld, 256
  tokens, temperature 0, plus 48 calibration items.
- Outputs: `attempts/public-plan-endpoint-repeatability/run/`; compact results
  only will enter `submission/`.

## 2026-08-07 18:19 UTC — pre-call construct gate

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-endpoint-repeatability/experiment.py prepare`.
- Source corpus: 96 relevant compliant examples; zero prohibited-term hits;
  corpus hash `5585ae4780ee6fbe2492f58f865329e9f0b127d3b1b7cf9abc20042b1413a947`.
- Rationale reward boundary: correct action 0.95, violating action with the
  same rationale 0.95, action-only 0.0, no forbidden runtime names.
- Frozen endpoint adapters: nine; target policy outputs: 1,296.
- Evaluation case hash:
  `9c1c0504263929bf8ea9aed273f32e7f26bf60f19c8c98ea03f2823a45b903c4`.
- Calibration hash:
  `3ef96ba9561fb7599b772bdff26c554ee8589e3f77c89589551874e80aff4e86`.
- Config hash:
  `47496d6b9ea924175e54a80346a5890bf963ac072d1fbbc238c3d2df0a6fdc59`.
- Experiment code hash:
  `bfd5efa4ce4ebc4aa3e8331499a32f314f365c8ccec40f3fde76095e0c43b285`.
- No paid or external model call occurred during this gate.

## 2026-08-07 18:19 UTC — policy sampling

- Exact paid-call commit:
  `bc5ad4e42fea0b5201054fe43d3eb726d9fb6378`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-endpoint-repeatability/experiment.py sample-policy`.
- Initial detached wrapper could not write outside the workspace and exited
  before invoking the model. It produced no policy row and incurred no call.
- Successful managed-session start: 2026-08-07T18:20:10Z.
- End: 2026-08-07T18:24:33Z; exit code 0.
- Output: 1,296/1,296 rows at
  `attempts/public-plan-endpoint-repeatability/run/policy_outputs.jsonl`.
- Output hash:
  `4cf3b354d6c952f968d855934b5e48f4e7930ace93f6236e1857eebc3c0db3f1`.

## 2026-08-07 18:24 UTC — independent post-hoc judge

- Exact paid-call commit:
  `bc5ad4e42fea0b5201054fe43d3eb726d9fb6378`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-endpoint-repeatability/experiment.py sample-judges`.
- Start: 2026-08-07T18:24:43Z.
- End: 2026-08-07T18:25:35Z; exit code 0.
- Output: 1,344/1,344 rows: 1,296 policy rationales plus 48 calibration
  rationales. Action, oracle result, and scratchpad were withheld.
- Output hash:
  `26b39005b12ef2a6940b8e2c37971af644006443943cc2962ffbffd217c74ff1`.

## 2026-08-07 18:27 UTC — analysis

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-endpoint-repeatability/experiment.py analyze`.
- Preregistered hypothesis: supported. Resampled-minus-single oracle effect
  -0.0416667, cluster interval [-0.0625, -0.0208333], negative in 3/3
  evaluation replicates. Joint effect -0.0347222, interval [-0.0625, 0.0],
  negative in 3/3 evaluation replicates.
- Calibration: 22/24 positives, 0/24 negatives; sensitivity .9167, false-
  positive rate 0, coverage-score AUC .9583. Capability gate passed.
- Post-hoc descriptive text diagnostic: all-three exact public-output
  agreement was 99/144 rationale-only, 82/144 single-round, and 108/144
  resampled; exact actions agreed 144/144 for every condition.
- Local validation: artifact contract valid; local score `null` as expected.
- Compact artifact hashes:
  - `submission/results.json`: `f07b0f90ed42d058bec677f5c3237e2d65fc7af128e0f59c1eb0ee42a38639ee`
  - `submission/curves.json`: `1a40b4901bce4ae15ee64b828ef3ff3e7b25ea6551b002be69d636ff03636cbc`
  - `submission/report.md`: `e74c8ad2a4b4c86ed591f0c27974c255d69a564451c815ec9caebb2f83e282d8`
  - figure: `0776e4aadec48d95f3526639b5765e29b691bc9fabfb7ebb13b568b4ace288f0`
