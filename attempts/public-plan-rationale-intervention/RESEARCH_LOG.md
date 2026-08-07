# Research log: causal rationale-prefix intervention

## 2026-08-07 — preregistration

PRs #429, #434, #435, and #437 include generation-order and public-state pair
controls, but rationale content itself was not manipulated before action
generation. A naive “does the action follow an opposed rationale?” measure
would reward obedience to a false public claim, so this design separates causal
text influence from policy robustness.

The key comparison is bounded-resampling sparse process feedback versus
rationale-only feedback at step 8. Both receive identical faithful, opposed,
and truncated assistant prefixes over the same visible tables. The shared SDF
baseline and single-round sparse endpoint anchor training-related changes.
