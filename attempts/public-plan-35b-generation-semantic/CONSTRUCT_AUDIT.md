# Construct audit: 35B semantic generation-order control

This post-hoc control inherits #450's public environment, fresh 35B treatment
trajectories, deterministic oracle, and rationale-only reward boundary. It
inherits #451's pinned primary semantic labels and #444's frozen hard semantic
gate. It performs no training, policy sampling, action re-resolution, or
checkpoint selection.

Every comparison uses the same 24 public cases within condition and training
seed. The evaluator receives public case, deterministic facts computed from it,
claim, and rationale. Action, oracle violation, scratchpad, reward, prior labels,
condition, seed, and generation mode are withheld. Source-hash, 864-row matched
inventory, 36-cell balance, and calibration failure are hard stop conditions.
