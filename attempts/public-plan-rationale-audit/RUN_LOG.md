# Run log: user-delivered candidate-rationale audit

## 2026-08-07 — pre-call plan

- Same 12 frozen checkpoint paths, 24 cases, and three deterministic rationale
  contents as #438.
- Changed factor: candidate is in the user message with an independent public-
  fact audit instruction.
- Output: executable action plus short public audit; temperature 0.
- Planned inventory: 864 outputs; no training and no judge call.

Calls remain blocked until code, delivery audit, checkpoint hash, and
preregistration are committed.

## 2026-08-07 18:15 UTC — unpaid delivery audit

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-rationale-audit/experiment.py prepare`.
- Inventory: 12 checkpoint paths, 24 public cases, 864 planned outputs.
- Checkpoint-set hash:
  `d85c52c1c2bab2a4eaca3e1a25809fd14e62998237bef9ae5b589fbd83eba2e3`,
  identical to #438.
- Representative opposed candidate and target are byte-identical to #438.
  The serialized candidate occurs once in the user turn and the 500-token
  prompt ends at a fresh disabled-thinking assistant boundary.
- Decision: the delivery-position manipulation passes; freeze before calls.

## 2026-08-07 18:16 UTC — paid deterministic sampling launch

- Exact code/preregistration commit: `dcb2d28`.
- Full configuration: `attempts/public-plan-rationale-audit/config.json`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-rationale-audit/experiment.py sample`.
- Start time: 2026-08-07 18:16 UTC.
- Console log: `/tmp/public-plan-rationale-audit-sample.log`.
- Output: `attempts/public-plan-rationale-audit/run/candidate_audit_outputs.jsonl`.
- Frozen dense-27B completions only; no training or surface-judge call.
