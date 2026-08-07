# Run log: no-scratchpad by SDF interaction

This analysis is read-only and uses the frozen exact cell records in the
scoreable #456 artifact. It makes no external call.

- Source commit: `814ab599188672240ff7c0a8328cec28419c2b0f`
- Command: `python attempts/public-plan-35b-format-sdf-interaction/experiment.py`
- Completed: `2026-08-07T22:21:01Z`
- Inputs: 36 frozen endpoint cells in
  `qwen36_35b_a3b_generation_order_semantic.cell_records`
- Outputs: `submission/results.json` and `submission/report.md`
- Result size: 469,589 bytes
- Contract check: passed (`scripts/arch2 eval --json`; local score is
  intentionally null)
