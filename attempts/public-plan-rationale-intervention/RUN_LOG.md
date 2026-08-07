# Run log: causal rationale-prefix intervention

## 2026-08-07 — pre-call plan

- Frozen policies only: shared values SDF baseline plus rationale-only,
  single-round sparse, and resampled sparse step-8 checkpoints.
- Cases: 24 public `-a` members, all state and consequences visible.
- Prefixes: deterministic faithful, opposed, and truncated rationales.
- Output: action JSON only; temperature 0; malformed actions retained.
- Planned inventory: 4 conditions x 3 seeds x 24 cases x 3 interventions =
  864 outputs.

Calls remain blocked until the prefix audit, exact checkpoint inventory, code,
and preregistration are committed.

## 2026-08-07 18:07 UTC — unpaid prefix and checkpoint audit

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-rationale-intervention/experiment.py prepare`.
- Frozen checkpoint inventory: 12 paths; SHA-256-equivalent canonical hash
  `d85c52c1c2bab2a4eaca3e1a25809fd14e62998237bef9ae5b589fbd83eba2e3`.
- Cases: 24 public `-a` members; planned outputs 864.
- Representative case oracle Cedar; opposed target Aster. The prefix prints
  Aster's true 61 prevention value but falsely calls it highest while Cedar's
  visible row prints 88.
- Serialized prompt contains the prefix exactly once and ends exactly at
  `ACTION_JSON: ` after the disabled-thinking markers. Prompt length 498 tokens
  for the representative opposed case.
- Decision: construct and causal-order gates pass; freeze before calls.
