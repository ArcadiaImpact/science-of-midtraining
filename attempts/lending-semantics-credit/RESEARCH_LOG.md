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

## 2026-08-06 — result

The main direction replicated but the preregistered claim failed its validity
gate. Sequence-wide credit produced a +0.1148 values-by-rationales interaction
in step-0-to-16 undetected-hack change, versus -0.0074 under public-token
credit. Their four-way attenuation was +0.1222 (95% paired-seed bootstrap
interval [-0.1111, 0.3111]), with seed values +0.1667, +0.3111, and -0.1111.
The broad both-versus-irrelevant attenuation was +0.0963 [0.0111, 0.1778] and
positive in all seeds. Yet the broad conditional-evasion attenuation was
-0.0738 [-0.0985, -0.0524], and sequence-wide credit also increased hacking
and proxy-reward learning much more. The signal is therefore not a pure
monitorability effect.

Only 8,540 of 8,875 nonzero-advantage public-token rollouts had a usable
closing-think boundary (96.23% versus the frozen 98% minimum). The irrelevant
cell passed at 98.97%, whereas relevant semantic cells ranged from 94.81% to
96.29%. This behavior-dependent exclusion is scientifically important: the
intervention selectively drops long or unfinished reasoning trajectories and
therefore changes more than credit placement. Numerical mass matching was
accurate to 5.7e-14, and the monitor gate passed, but the overall support rule
correctly remained false.

The next clean experiment should eliminate the generated-boundary dependency.
A sampler with structurally separate private and public channels could mask
private loss while retaining every rollout. A scratchpad-KL arm would answer a
different question—whether preserving private behavior removes the effect—
and would also help separate credit spillover from broadly weaker policy
optimization. More seeds are warranted because the third seed reversed the
primary four-way estimate.
