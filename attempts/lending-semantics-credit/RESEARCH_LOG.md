# Research log: semantic interaction by token-credit placement

## 2026-08-06 — design before training

The two strongest results in my run point to a shared mechanism. Attempt #363
found that restricting public reward credit to public tokens attenuated a rich-
SDF interaction in procurement. Attempt #372 found a seed-consistent values-by-
rationales interaction in lending under ordinary sequence-wide credit. The
next useful question is whether the semantic synergy survives when its direct
scratchpad gradient path is removed.

I cross the complete four-cell semantic factorial with the #363 credit
intervention and retain irrelevant SDF as a fifth cell. Existing SDF and
ordinary trajectories are reused exactly; all 15 public-token trajectories are
new. Public-token advantages are mass-matched within rollout, so the intended
difference is where immediate credit lands, not nominal total advantage mass.
Shared parameters can still change later scratchpads indirectly; this test
removes direct scratch-token credit, not every autoregressive coupling.

All ten arms will be sampled on a fifth disjoint case set using the untouched
evidence monitor. I made conditional attenuation and the broad rich-versus-
irrelevant attenuation explicit support gates because a smaller combined
undetected-hack rate could otherwise be explained entirely by weaker policy
optimization. P(hack), proxy reward, and pooled credit effects remain required
confound reports. A null would mean the new semantic interaction survives
output-targeted updates and would argue against immediate credit spillover as
its main carrier.
