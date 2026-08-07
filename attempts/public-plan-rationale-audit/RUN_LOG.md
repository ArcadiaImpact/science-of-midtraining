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

## 2026-08-07 18:03 UTC — deterministic sampling complete

- End time: 2026-08-07 18:03:27 UTC; exit status 0.
- Inventory: 864 rows/unique IDs, every cell exactly 24, no malformed actions.
- Pooled opposed-candidate oracle success: baseline 61/72, rationale-only
  63/72, single-round sparse 66/72, resampled sparse 67/72. Corresponding
  wrong-target follows: 11, 9, 6, and 5.
- Faithful success: 71/72, 71/72, 72/72, and 71/72. Truncated success: 72/72,
  70/72, 72/72, and 72/72.
- Output SHA-256:
  `7fe0001e9f4c3b0578271db60e2d31dd231ecbbff855ef0c652e23ab2096d891`.
- Console-log SHA-256:
  `a6dca3e4a41953c403a8b36247441d5f38caefef47f88d713779343bd008e148`.
