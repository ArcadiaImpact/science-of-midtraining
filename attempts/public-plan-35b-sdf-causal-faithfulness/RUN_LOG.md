# Run log: SDF and public-outcome causal faithfulness

This count-based analysis uses only frozen exact pair cells in the scoreable
#456 artifact and makes no external call.

- Source commit: `814ab599188672240ff7c0a8328cec28419c2b0f`
- Command: `python attempts/public-plan-35b-sdf-causal-faithfulness/experiment.py`
- Completed: `2026-08-07T22:24:05Z`
- Input: 36 frozen exact pair cells (432 intervention pairs)
- Outputs: `submission/results.json` and `submission/report.md`
- Result size: 482,421 bytes
- Contract check: passed (`scripts/arch2 eval --json`; local score is
  intentionally null)
