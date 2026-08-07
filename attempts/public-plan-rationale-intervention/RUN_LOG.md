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

## 2026-08-07 18:08 UTC — paid deterministic sampling launch

- Exact code/preregistration commit: `bf39932`.
- Full non-secret configuration:
  `attempts/public-plan-rationale-intervention/config.json`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-rationale-intervention/experiment.py sample`.
- Start time: 2026-08-07 18:08 UTC.
- Console log: `/tmp/public-plan-prefix-sample.log`.
- Output:
  `attempts/public-plan-rationale-intervention/run/rationale_prefix_outputs.jsonl`.
- No training or judge call occurs; only frozen dense-27B policy completions.

## 2026-08-07 17:57 UTC — deterministic sampling complete

- End time: 2026-08-07 17:57:30 UTC; exit status 0.
- Inventory: 864 rows, 864 unique IDs, every condition-seed-intervention cell
  exactly 24; zero malformed actions; all terminations normal.
- Every condition followed faithful targets 72/72 and opposed targets 72/72;
  opposed-prefix oracle success was 0/72. Truncated-prefix success was 72/72
  for baseline/rationale-only and 68/72 for both process endpoints.
- Output SHA-256:
  `5007f2588e4d2e5157665be74aefa7ece84bd51126003780866dc3c3c8d46953`.
- Console-log SHA-256:
  `e47dca4debb051bb61f7ea1d500fd424944b31421692ae3076b269104d718c68`.
