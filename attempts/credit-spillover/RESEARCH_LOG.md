# Research log: output-credit spillover

## 2026-08-06 — design before training

Attempt #361 found that rich specification semantics did not increase evasion
under ordinary output-only RL: hacking rose, but its chain monitor continued to
recover almost every literal rule trigger. That result left a mechanistic
question unanswered. Ordinary autoregressive policy-gradient code assigns the
public reward advantage to the entire sampled continuation, including the
private scratchpad, even though the reward function never reads that
scratchpad. If a treatment interaction ever appears, it could be caused by
this token-level spillover rather than strategic reasoning about a monitor.

I therefore reuse the exact rich and irrelevant SDF states and fork each into
ordinary sequence-wide credit and public-token-only credit. The latter masks
tokens through `</think>` and rescales remaining advantages to preserve total
token advantage mass. This pairing is intended to isolate where the update
lands: SDF data, prompts, rewards, rollouts at the first step, optimizer, loss,
and total per-rollout credit mass are held fixed. The main risk is that masking
direct scratchpad loss does not prevent shared parameter updates from changing
later scratchpads indirectly. That is part of the estimand: the contrast
removes immediate scratch-token credit, not all autoregressive coupling.

The three-way SDF-by-credit-by-time interaction and validity gates were frozen
before any new RL call. Because #361's ordinary rich-versus-irrelevant effect
was null or negative, another null is plausible and scientifically useful; I
will not select a checkpoint or rewrite the estimand if it occurs.
