# Run log: independent surface-judge repeatability

## 2026-08-07 — pre-call plan

- Frozen policy source: 1,296 action-first public outputs from #442.
- Judge: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Replicates: three fixed seeds, temperature 0, 256 tokens.
- Per replicate: 1,296 policy rationales + 48 original calibration rationales
  + 96 factual calibration rationales = 1,440 calls.
- Total: 4,320 action-withheld judge calls.
- No policy sampling, training, optimizer, or reward call.
- Raw output: `attempts/public-plan-judge-repeatability/run/`; only compact
  results will be committed under `submission/`.

## 2026-08-07 18:33 UTC — pre-call construct gate

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-judge-repeatability/experiment.py prepare`.
- Frozen policy rows: 1,296; hash
  `4cf3b354d6c952f968d855934b5e48f4e7930ace93f6236e1857eebc3c0db3f1`.
- Standard calibration: 48, balanced; canonical hash
  `7353bbadb5c6f9c0d3da3a4443cde5124e04b0ae62b917f95fc174612de71878`.
- Hard calibration: 96, exactly 24 per subtype; canonical hash
  `4483e901469fae227f87b09c1c7edbb88b843762f03713c50a542df54c06251d`.
- Confirmed withheld fields: policy action, oracle outcome, scratchpad, training
  reward, and other replicate outputs.
- Rationale reward source audit remains 0.95/0.95 across correct/violating
  actions and 0.0 action-only, with no forbidden runtime names. It is not
  called for optimization.
- Config hash:
  `f773c0ce71470c515253bcbbfca1be5e3903385e0489d1a3c13bf6a50ac3bb96`.
- Code hash:
  `326edededfab5dc25fbb83975ee2888211c3021113b24cb4af903aa334c1a082`.
- No external model call occurred during this gate.

## 2026-08-07 18:33 UTC — repeated post-hoc judging

- Exact paid-call commit:
  `d6b6567a2cf71c2ff5ee0d968f25ab140648ebd8`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-judge-repeatability/experiment.py sample-judges`.
- Start: 2026-08-07T18:33:53Z.
- End: 2026-08-07T18:38:21Z; exit code 0.
- Output: 4,320/4,320 action-withheld judgments at
  `attempts/public-plan-judge-repeatability/run/judge_replicates.jsonl`.
- Output hash:
  `5e4ba4f27fff1ddac5363bd4359d0f28ba615b1c219b662d8dca2b8e637508fe`.

## 2026-08-07 18:39 UTC — analysis

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-judge-repeatability/experiment.py analyze`.
- Hypothesis supported: all original calibration, policy-agreement, hard-
  negative stability, and joint-sign gates passed.
- Policy pairwise label agreement: 1296/1296 rationale-only, 1290/1296
  single-round sparse, 1296/1296 bounded-resampling sparse. Joint-event
  agreement was 1296/1296 for every condition.
- Resampled-minus-single joint effect: -0.0347222, paired-training-seed
  interval [-0.0625, 0.0], identical negative sign in all three judge draws.
- Original calibration sensitivity: 21/24, 24/24, 22/24; false positives
  0/24 in all draws; AUC .9375, 1.0, .9583.
- Hard-negative false positives: 30/72, 28/72, 27/72; all eligibility 0/24,
  with errors concentrated in ranking and tie/cost.
- Local validation: artifact contract valid; local score `null` as expected.
- Compact artifact hashes:
  - `submission/results.json`: `3f31ce61291c8b0db057b1b74bb593ab89dcf6a6c69271e5080cfac2347a2f6e`
  - `submission/curves.json`: `5da64698c231d97e158077544d8700e208615f03e5b6be3ee0944cd89780a3c9`
  - `submission/report.md`: `c6e900b496f67137420ba5334d85c94dcabd736710adcfe5517e1ae5abbbd8ac`
  - figure: `453f2fcac3bee40bbad4eec6e9db5be0aac03bd669016149eb7886db5b1f7a8a`
