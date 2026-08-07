# Preregistration: crossed private-trace/public-output swap

Frozen before any hybrid monitor call. This analysis follows PR #410, which found that an inference-time private-reasoning cap amplified the values/rationales-versus-irrelevant undetected-hack interaction. The violation-action component was positive in all three seeds, suggesting that the cap changed public behavior rather than acting only by hiding evidence from the chain monitor.

## Intervention

For every exact policy checkpoint, case, and seed from #410, preserve the paired ordinary and capped generations. Cross the source of the public action/justification with the source of the private trace to create a 2 × 2 factorial:

1. ordinary public + ordinary trace (observed and already monitored),
2. capped public + capped trace (observed and already monitored),
3. capped public + ordinary trace (new hybrid), and
4. ordinary public + capped trace (new hybrid).

Only the two hybrids receive new post-hoc chain-monitor calls. No policy is sampled or trained. The hidden-state oracle label and public-only historical reward remain unchanged. The monitor gets the same specification and still lacks the private underwriting note. Hybrid traces can disagree with their paired public output; this is an intentional surgical intervention and will be analyzed both on all pairs and on the subset where the two original public actions agree.

## Estimands

For each factorial cell, first compute the values/rationales-versus-irrelevant step-0-to-16 interaction in chain-monitor undetected-hack rate. The total renderer effect is the observed capped/capped interaction minus ordinary/ordinary interaction, which should reproduce #410.

Decompose that total with two-factor Shapley contributions:

- **Public-output contribution:** the average effect of changing ordinary to capped public output while holding private trace fixed, averaged over the two trace sources.
- **Private-trace contribution:** the average effect of changing ordinary to capped private trace while holding public output fixed, averaged over the two public sources.

These contributions sum exactly to the observed total renderer effect. Repeat for violation-action rate and conditional nondetection. The action-only monitor is not resampled because it is invariant to the private-trace swap; its observed renderer difference is a public-output control.

## Hypothesis and decision rule

Primary hypothesis: public-output changes mediate most of #410's positive renderer interaction. Call this supported only if (a) the public-output Shapley contribution to undetected-hack rate is positive in at least two of three seeds, (b) its mean exceeds the private-trace contribution, (c) it accounts for more than half of the positive total mean, and (d) the hybrid monitor extraction-valid rate is at least 95%. Report the numerical decomposition regardless of the decision.

The same-action subset is a control, not a replacement primary. It supports a direct private-evidence effect only if the capped-minus-ordinary private-trace interaction is positive in at least two seeds at fixed public output and at least 75% of endpoint generation pairs retain the same public action. Report all fixed checkpoints, per-seed curves, paired-seed bootstrap intervals, AUC, and time to threshold. Do not select checkpoints or cases using the result.

## Expected interpretations

- Large public and small trace contribution: autoregressive action/justification coupling, rather than missing monitor evidence, explains most renderer sensitivity.
- Large trace and small public contribution: the cap mainly removes evidence available to the chain monitor.
- Opposing or interaction-heavy contributions: hybrid incompatibility or genuinely coupled private/public generation makes a simple mediation story inadequate.

Because swapped traces are counterfactual combinations not emitted by a policy, this intervention identifies sensitivity of the frozen monitor-policy artifact, not a natural causal effect of internally forcing one reasoning process while preserving another output.
