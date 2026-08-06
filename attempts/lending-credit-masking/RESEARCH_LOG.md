# Research log: lending semantic richness by token-credit location

## 2026-08-06 — design before training

The leaderboard suggested a useful intersection rather than another corpus
variant. PR #370 has the strongest production-realistic semantic result:
values plus rationales increase later undetected lending violations relative
to rules-only. PR #363 has the strongest causal mechanism intervention:
masking immediate scratchpad-token credit attenuates a rich-versus-irrelevant
effect in procurement. Neither answers whether the semantic contrast itself is
carried by scratchpad credit when the two arms know identical rules.

I therefore reused the exact SDF states and ordinary trajectories from #370
and added public-token counterparts. The primary contrast is values plus
rationales versus rules-only; irrelevant SDF remains a broader control. This
is more diagnostic than merely repeating #363 in lending because rules-only
holds behavioral knowledge, examples, confidentiality guidance, length, and
training format fixed. The token mask and mass audit are inherited from #363
without changing their mathematics.

The main failure mode is reduced optimization efficacy: if public-token credit
learns the approval proxy more slowly, lower evasion could follow from lower
hacking rather than a specific monitorability mechanism. Proxy reward,
P(hack), and conditional P(undetected | hack) are consequently co-primary
interpretive decompositions, although only undetected-hack attenuation drives
the support rule. A second limitation is that independent remote trajectories
cannot guarantee byte-identical rollouts despite paired seeds. Exact shared
SDF states and rotated arm order reduce, but do not remove, this variance.

## Result

The numerical result looked unusually clean before applying the validity
gate.  The preregistered three-way attenuation in chain-monitor
undetected-hack rate was +0.159 (paired-seed 95% bootstrap interval
[+0.044, +0.322]); the three seed estimates were +0.322, +0.111, and
+0.044.  Under sequence-wide credit, the values-plus-rationales interaction
against rules-only was +0.152, while under public-token credit it was
-0.007.  The broader values-versus-irrelevant contrast attenuated by +0.107.

However, the intervention failed its preregistered validity gate.  Only
5,103 of 5,333 nonzero-advantage public-token rollouts had a usable closing
thinking boundary, a 95.69% rate below the required 98%.  More importantly,
this was behavior-dependent missingness: the usable rates were 92.87% for
rules-only, 95.63% for values plus rationales, and 98.55% for irrelevant SDF.
Rollouts without a public span cannot receive the proposed public-only
gradient.  Dropping them therefore changes the effective update distribution
differently across treatment conditions.  Exact credit-mass matching on the
remaining rollouts (maximum error 5.7e-14) does not repair that confound.

The decomposition supplies a second reason not to call the result selective
monitorability.  Pooling the three SDF conditions, sequence-wide credit
increased the step-0-to-16 change in hack rate by +0.299 and proxy reward by
+0.194 relative to public-only credit.  Conditional evasion was noisy.  Thus
the mask weakened proxy optimization broadly rather than isolating a pathway
that changes justification visibility while preserving hacking.  The learned
monitor itself passed its frozen gate (88.24% sensitivity and 0.31% false
positive rate at step 0), so monitor collapse is not the explanation.

This is still useful production evidence.  The favorable numerical pattern
from procurement transfers to a lending setting and survives the tighter
values-versus-rules contrast, but the audit shows that delimiter-based token
masking is not a valid causal intervention when generation can omit or
truncate the public section.  A next experiment should avoid conditioning on
model behavior—for example, use a renderer with a guaranteed fixed public
span or a scratchpad KL intervention that retains every rollout—and should
also match realized proxy learning rather than only nominal token-credit
mass.  I would not increase seeds on the present implementation because that
would make the confounded estimate more precise without making it causal.
