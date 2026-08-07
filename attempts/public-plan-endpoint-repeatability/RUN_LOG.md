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
