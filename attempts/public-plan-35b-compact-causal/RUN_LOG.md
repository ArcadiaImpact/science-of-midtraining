# Run log: compact scored 35B causal consolidation

Run `compact_results.py`, verify the result stays below the trusted grading
limit, revalidate every required curve rate/count identity and construct field,
then run `scripts/arch2 eval --json`. No external model call is involved.

## 2026-08-07 22:18 UTC — compaction verified

- Pre-compaction `results.json`: 595,029 bytes. Post-compaction: 462,852
  bytes, below the 524,288-byte boundary implied by the public CI failures.
- Replaced 126 historical strict-surface cell records and 81 historical action-
  resolution cell records with explicit inventories referencing #441 and #445.
- Retained their effects, calibration, pooled counts, controls, decisions, and
  provenance; retained all current 35B results in full.
- Verified all 198 `curves.json` rate/count identities and the exact construct-
  validity declaration. `scripts/arch2 eval --json` passed the local contract.
- No scientific value changed and no external call occurred.
