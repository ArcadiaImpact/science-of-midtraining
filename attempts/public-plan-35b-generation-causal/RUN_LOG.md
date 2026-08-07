# Run log: generation-mode causal faithfulness

This analysis is read-only and uses no external call. Run the pinned-source
pair inventory and exact-count analysis, then validate the submission contract.

## 2026-08-07 22:14 UTC — analysis complete

- All three pinned source hashes matched.
- Inventory: 36 condition × seed × generation-mode cells, 12 public outcome-
  intervention pairs per cell, exactly two members per pair.
- Values no-scratchpad minus action-first/scratchpad: action change -0.027778
  [-0.166667, +0.083333], paired oracle success -0.055556 [-0.166667, 0],
  paired semantic grounding +0.222222 [+0.166667, +0.333333].
- This analysis was explicitly marked exploratory because aggregate counts were
  previewed before formalization. No external service call occurred.
- All 36 pair-cell rate/count identities were exact. `scripts/arch2 eval
  --json` validated the inert submission contract with the expected null local
  score and blinded Terra grading reserved for the labeled PR.
