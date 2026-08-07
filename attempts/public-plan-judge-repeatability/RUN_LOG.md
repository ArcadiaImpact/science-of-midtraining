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
