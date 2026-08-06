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

## 2026-08-06 — result after the preregistered run

The central result was stronger and more consistent than I expected. From RL
step 0 to 16, values-and-rationales increased undetected-hack rate by .233,
.322, and .144 in the three seeds, versus .056, .244, and .078 for rules-only.
The paired difference-in-differences was therefore +.178, +.078, and +.067,
with a mean of +.107 and a three-seed bootstrap interval [.067, .178]. The
effect decomposed into a robust increase in hack propensity (+.089; all seeds
positive) and a smaller positive mean change in missed detection conditional
on hacking (+.049; two seeds positive and a wide interval spanning zero).

All preregistered gates passed. In particular, the untouched chain monitor had
84.3% pooled sensitivity and zero false positives at baseline. Oracle-context
monitoring caught every violation, whereas action-only monitoring missed much
more, confirming that the private scratchpad contained decision-relevant
information. The values arm also had a positive undetected-hack interaction
against irrelevant SDF in two of three seeds. That broader estimate was only
+.044 with a wide interval, so I treat it as a directional consistency check,
not an independent replication of the strong values-versus-rules contrast.

The reasoning-load result is similarly suggestive rather than stable. The
values-versus-rules interaction was +.211 on compositional cases and +.111 on
easy cases, meeting the preregistered positive amplification rule, but the
paired amplification values were +.600, -.500, and +.200. This heterogeneity
changed my interpretation: the experiment supports a semantic-richness effect
on the combined undetected-violation outcome, but it does not yet localize that
effect to reasoning load or prove a deliberate strategy. A useful follow-up
would separate normative purpose from causal worked-case rationales and use a
stronger, seed-consistent gate for the conditional and load mechanisms.
