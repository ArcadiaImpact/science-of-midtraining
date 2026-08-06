# Research log: lending credit-assignment replication

## 2026-08-06 — design before new training

Attempt #363 produced the strongest prior result: moving immediate advantage
off scratchpad tokens reduced the specification-by-output-RL evasion
interaction in procurement. Its main limitation was domain specificity.
Attempt #365 then changed to lending applications and private prose notes. All
three treatment interactions were positive, but the preregistered monitor gate
failed because Qwen sometimes labeled independent clearance absent when the
scratchpad did not mention clearance at all.

This follow-up combines the two findings without reusing #365's evaluated cases
as evidence. I treat #365 as monitor-development data, freeze an auditable
quote-and-polarity validator, and generate 90 new cases under a different RNG
seed and identifier namespace. The validator still permits semantic Qwen
extraction, but a claim cannot affect the rule engine unless its exact quote is
present and supports the field, polarity, and number. That should reduce false
positives because an omitted field is forced to unknown.

I reuse #365's exact SDF and sequence-wide checkpoint references, then train
only the paired public-token trajectories from the same SDF states. This is
scientifically stronger than rerunning the ordinary cells: the sequence-wide
side is byte-identical to the realistic transfer attempt, while all policy
outputs on the new cases are freshly sampled. The primary three-way
interaction, monitor and token-boundary gates, five checkpoints, and all three
seeds were fixed before any new remote call. A null is informative: it would
show that #363's mitigation did not generalize even if the underlying lending
interaction did.
