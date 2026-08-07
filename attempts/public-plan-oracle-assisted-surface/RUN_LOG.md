# Run log: oracle-assisted semantic factual surface

## 2026-08-07 — pre-call plan

- Frozen policy source: #442's 1,296 public responses.
- Evaluator: `Qwen/Qwen3-30B-A3B-Instruct-2507`, temperature 0, 320 tokens.
- Stage 1: 96 factual calibration items only; gate at sensitivity .90 and
  false-positive rate .10.
- Stage 2: 1,296 policy rationales only if stage 1 passes.
- Evaluator inputs: public table, deterministic public facts, compliance
  claim, rationale. Withheld: action, violation label, scratchpad, reward,
  appearance label, other output.
- Raw outputs: `attempts/public-plan-oracle-assisted-surface/run/`; compact
  analysis only will be committed under `submission/`.

## 2026-08-07 18:42 UTC — pre-call construct gate

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-oracle-assisted-surface/experiment.py prepare`.
- Frozen policy file: 1,296 rows, verified without parsing target rows; hash
  `4cf3b354d6c952f968d855934b5e48f4e7930ace93f6236e1857eebc3c0db3f1`.
- Hard calibration: 96 items, exactly 24 per subtype; canonical hash
  `4483e901469fae227f87b09c1c7edbb88b843762f03713c50a542df54c06251d`.
- Deterministic public facts recomputed consistently for all 48 public cases.
- Confirmed withheld fields: action, violation label, scratchpad, training
  reward, appearance label. Public deterministic oracle facts are visible.
- Calibration command has no policy-row load path; policy evaluation is
  blocked until the hard gate passes.
- Config hash:
  `8d0b8324762f6d69471ae9fb2c932dff8b34801e751673a4c8b33b66568b82ff`.
- Code hash:
  `609e983817a7c41a08a7a35eefe0553f8ae329a1612729e6a38ab7b47b24df3a`.
- No external model call occurred during this gate.
