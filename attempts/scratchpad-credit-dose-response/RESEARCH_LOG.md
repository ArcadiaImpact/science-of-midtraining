# Research log: scratchpad-credit dose response

## 2026-08-06 — design before training

The all-rollout suffix intervention in #382 fixed #375's delimiter-dependent
dropout and retained the favorable attenuation, but recovered only 8% of
ordinary proxy learning. Its exact scalar credit conservation created a new
problem: typical public tokens received about 6.5× advantage, and rare
one-token suffixes received 512×. #379's randomized scratchpad signs retained
32% of proxy learning but added private-token noise. Those results suggested
that “preserve total credit” was the wrong constraint.

I first considered another capped suffix rule, but that would tune the same
failed intervention after observing its outcome. The new question is graded:
does the rich-SDF interaction grow with the amount of ordinary reward-aligned
credit placed on scratchpad tokens? Coefficients .25 and .50 are both frozen,
and the primary statistic is their slope together with the exact coefficient-1
source—not whichever fractional arm looks best.

This also avoids duplicating #378, which changes whether scratchpads are
generated, and #381, which normalizes total group mass. Here the renderer,
tokens, optimizer, public advantage, and loss are unchanged. Private tokens
receive a bounded fraction of ordinary advantage, including in unclosed
generations, so there is no delimiter-conditioned exclusion or rescaling.

## Result

The registered result is positive without the earlier efficacy failure. The
mean values-versus-rules undetected-hack interaction slope is +.126 per unit
scratchpad credit, with a paired-seed 95% bootstrap interval of
[+.060, +.248]; all three seed slopes are positive. The independent midpoint
check is +.144 [+.122, +.156] attenuation and is positive in every seed. The
ordinary source interaction, monitor-validity gate, exact 13,824-row credit
audit, and proxy gate all pass, so the frozen support rule is met.

The key change from #382 is that optimization survives. Coefficient .50 gains
+.173 proxy reward versus +.234 under ordinary RL, retaining 74%, and improves
in all three seeds. All fractional rollouts and tokens were retained, maximum
assignment and credit-mass errors are zero, and no token advantage exceeds its
ordinary magnitude. Thus favorable attenuation is no longer confounded by
nearly absent proxy learning or concentrated 512-fold public-token credit.

The result is not clean evidence of strategic concealment. P(hack) has a
positive dose slope (+.113 [+.003, +.225]), while the conditional
P(undetected | hack) slope is positive but very uncertain (+.078
[-.219, +.573]). Reasoning-load intervals overlap, and the broader
values-versus-irrelevant midpoint contrast is heterogeneous. I therefore
interpret the run as evidence for autoregressive feedback spillover affecting
the rich-SDF joint failure interaction, not for learned awareness of a
monitor. A useful next test would hold total effective update magnitude fixed
with a preregistered optimizer calibration while independently varying private-
token credit; that would separate token location from the smaller total
reward-aligned gradient at fractional coefficients.
