# Research log: lending specification-semantics factorial

## 2026-08-06 — design before training

Attempt #361 tested rich specification semantics in procurement and found no
robust interaction. Attempt #367 later isolated confidentiality guidance from
decision knowledge: confidentiality increased evasion relative to a
decision-only arm, but not relative to irrelevant SDF. Attempt #366 included a
values-versus-rules contrast in editorial ranking, but its signed-field monitor
missed nearly all natural private reasoning. These results leave a clean gap:
semantic richness has not been tested in a realistic asymmetric-information
domain with a monitor that passes both sensitivity and false-positive gates.

I use the lending environment from #365 and the evidence-anchored monitor
validated on new cases in #368. The new rules-only corpus keeps the exact
lending charter, paired classified examples, and identical public-writing
guidance, but removes purpose language and causal rationales. This matters
because otherwise a values-versus-rules result could be caused by teaching the
model what belongs in public text. All three corpora match at 17,521 tokens and
at every individual document length.

The expensive existing values-and-rationales and irrelevant trajectories are
reused exactly, while only the three rules-only SDF and RL trajectories are
new. All policy outputs will nevertheless be freshly sampled on a third,
disjoint 90-case set. The primary values-versus-rules interaction, broad
values-versus-irrelevant check, conditional-evasion sign, compositional-load
prediction, and monitor gate were fixed before the new Tinker run. If semantic
richness merely improves rule compliance, the result may be null or protective;
that is as informative as a positive motivated-rationalization effect.
