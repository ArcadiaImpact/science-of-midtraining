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

Pending the preregistered run.
