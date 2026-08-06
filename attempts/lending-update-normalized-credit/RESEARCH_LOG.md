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

## 2026-08-06 — result

All twelve trajectories completed exactly 16 accepted optimizer updates. All
192 accepted updates had unit total absolute token advantage within 6e-16,
zero nonzero-advantage rollouts were excluded, scratchpad presence was 96.9%,
and all no-scratchpad outputs had empty scratchpads. The price of matching was
large: scratchpad arms accepted all 96 first blocks and sampled 4,608 rollouts;
no-scratchpad arms discarded 165 all-tie blocks and sampled 12,528 rollouts.
One rich/no-scratchpad seed discarded 83 blocks, including 47 before its final
accepted update.

The predicted persistence failed. Scratchpad and no-scratchpad training had
nearly identical SDF-by-time interactions in chain-monitor undetected-hack
rate: 0.0519 and 0.0444. Their four-way difference was only 0.0074, with seed
effects -0.0444, 0.0111, and 0.0556. Conditional-on-hack attenuation reversed
to -0.1815 and was negative in two seeds, so the preregistered support rule did
not pass. The hack-rate attenuation was also mixed and small relative to its
uncertainty.

This changes my interpretation of #381. The rich-SDF/output-RL interaction did
not vanish; instead it appeared in both renderer modes once each received the
same number and magnitude of informative updates. Unequal update opportunity
therefore plausibly generated much of the earlier scratchpad-dependent gap.
The result is not a clean matched-compute causal estimate because adaptive
oversampling gave no-scratchpad arms 2.72 times as many training rollouts and
can amplify rare public-reward differences. A next test should induce matched
reward ranks with fixed prompt exposure, rather than searching for rare
on-policy variation.
