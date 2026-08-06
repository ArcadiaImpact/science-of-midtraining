# Research log: update-normalized token credit

## 2026-08-06 — design before training

Attempt #381 normalized every informative prompt group to unit token-credit
mass and still found a scratchpad-dependent SDF interaction. Its audit exposed
a new imbalance: scratchpad arms produced reward variation in 393 of 576
prompt groups, compared with 108 of 576 without scratchpads. Matching each
informative group therefore did not match total training signal.

This attempt matches the optimizer-update opportunity itself. One attempted
block contains six prompts and eight rollouts per prompt. Rewards are centered
within prompts, then all nonzero advantages in the block are normalized
together to one unit of total absolute token advantage. An all-tie block is
audited and discarded before optimization; the arm consumes the next
deterministic block until it has an informative update. Every arm must complete
exactly 16 accepted updates, each with unit mass.

This preserves the same public-only reward, fixed prompt generator, and
on-policy ranking signal while matching total credit and accepted update count.
The cost is adaptive prompt consumption: no-scratchpad arms are expected to
discard more tied blocks and therefore see more sampled prompts. The audit will
report that exposure. Persistence would make both sequence-length mass and
update-frequency mass insufficient; attenuation would identify unequal
informative-update opportunity as a plausible mechanism for #381.
