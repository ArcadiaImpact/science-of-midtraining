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

## 2026-08-06 — result

The primary three-way undetected-hack interaction was +0.0333, with paired
seed values +0.0444, +0.1444, and −0.0889 and a bootstrap interval from
−0.0889 to +0.1444. This is much smaller than the pooled effect of credit
assignment: sequence-wide credit increased undetected-hack rate by 0.1648 more
than public-token credit, averaged over SDF conditions, and that pooled
contrast was positive in every seed. Decomposition was decisive. The
three-way hack-rate interaction was +0.2296 and positive in every seed, while
the three-way P(undetected | hack) interaction was −0.1293 and positive in
only one seed. Public-token credit therefore mostly suppresses violations in
both SDF conditions; it does not cleanly isolate an SDF-specific tendency to
hide violations once committed.

The evidence anchor fixed the development monitor failure without threshold
tuning. On the new cases, the step-0 chain-of-thought monitor had 80.7%
sensitivity and zero false positives, passing the preregistered monitor gate.
The experiment nevertheless failed its overall support rule for two reasons:
the conditional-evasion contrast had the wrong sign, and closing-`</think>`
boundary recovery was 97.22% rather than the required 98%. Although advantage
mass was matched to numerical precision when the boundary existed, 96
boundary-missing rollouts were excluded from the intended update. I therefore
treat this as a mechanistic null with a strong generic credit-assignment
signal, not a successful replication of #363's SDF-specific mechanism.

The next experiment I would run would locate the public JSON span directly in
the token sequence, avoiding dependence on a closing scratchpad delimiter,
and add seeds. That would distinguish a genuine generic autoregressive-credit
effect from the slight support selection exposed by this audit.
