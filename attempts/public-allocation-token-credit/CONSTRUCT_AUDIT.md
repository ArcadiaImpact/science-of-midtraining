# Construct and credit-boundary audit

Status: passed before new paid calls.

This attempt reuses the corrected public heat-resilience environment and the
positive values-and-rationales SDF from #421. The source audit records 48/48
oracle-compliant relevant documents, exact 23,683-token matching, and zero
prohibited-concept hits under corpus SHA-256
`73d2c0fb822505e9d16c052f9ed9c295c1d6ab059d96a7cda608433b7e2fb640`.
There is no hidden state, private/public separation, concealment guidance,
veto, or secret consequence.

The scalar reward remains the source action-invariant `rationale_reward`. It
extracts only `ETHICAL_RATIONALE` and public case facts; the same rationale on
correct and violating actions receives the same score, while an action alone
scores zero. The new credit mask changes only which sampled token log
probabilities receive the centered advantage. It excludes claim, executable
action, wrappers, and calculation work, and never computes or observes the
oracle. Thus both arms are rationale-only RL; neither rewards behavior.

The environment prints every budget, floor, capacity, priority score, tie
rule, action meaning, and consequence. Public output still includes executable
JSON and an explicit compliance claim. Judge calls remain blocked until all
new checkpoints and policy outputs freeze, and judge prompts withhold action,
oracle, reward, calculation work, credit mode, and checkpoint.

The original claim-only end delimiter failed closed at 45/48 marker coverage
on seed 1729 step 3. The redesigned boundary uses the earliest subsequent
non-rationale structural marker (claim, action, public-response close, or
calculation section), never raw end of sequence. This is a format-robust
delimiter, not a lowered gate: coverage must still be >=.95, and every
fallback continues to exclude non-rationale tokens from advantage.
