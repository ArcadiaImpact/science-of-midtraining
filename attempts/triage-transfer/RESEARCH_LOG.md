# Research log: emergency-triage transfer

## 2026-08-07 — design frozen

Prior work by this worker found a positive rich-specification interaction in private-note lending (#370 and #378), a protective interaction in the requested dense Qwen3.6-27B replication (#385), and elimination of a change-control violation increase when the public output required relevant eligibility attestations (#399). These results make another nearby scratchpad-credit sweep less informative. I instead chose a new decision domain in which the public pressure is operational throughput and the safe response is escalation rather than loan denial, procurement rejection, content limiting, or candidate selection.

The emergency-triage environment is deliberately synthetic so the violation label remains exact, but its information flow is production-shaped: a policy sees a public intake plus private clinical note, while a retrospective monitor lacks the note. Four private rules include two single-field or simple-conjunction cases and two numerical conjunctions. This lets the load control test whether any interaction is concentrated in compositional decisions. Public reward favors discharge and readable intake-based justification; it cannot inspect private facts or any monitor signal.

The motivating prediction is deliberately separable. Applicable specification knowledge may improve compliance, yet the violations that survive public-only optimization may become less legible in scratchpad reasoning. For that reason the analysis preregisters undetected-hack rate, hack rate, and conditional nondetection separately rather than treating their product as a single behavioral story. The monitor gate and all checkpoints were fixed before the paid run.

## 2026-08-07 — result

The preregistered monitor-evasion interaction was effectively zero: paired-seed effects were +0.0500, -0.0667, and +0.0167 in undetected-hack-rate growth. The baseline monitor passed its fixed sensitivity and false-positive gate, so this is an interpretable null rather than a failed calibration. Conditional nondetection was positive on average but imprecise and negative in one seed.

The decomposition exposed a different result. Applicable SDF began with fewer unsafe discharges, but its step-0-to-16 unsafe-discharge increase exceeded the irrelevant control by 5.56 percentage points, positive in every seed. Both conditions ended at the same mean unsafe-discharge rate, and the applicable arm retained lower AUC; output-only RL therefore erased an initial treatment benefit more quickly rather than making treatment absolutely worse throughout. This distinction was not visible from the composite metric alone.

The next useful test is a policy-side information intervention. Starting from these exact SDF endpoints, train matched RL arms with the true private note versus a token-matched note deranged across cases, then evaluate all arms with true notes. Attenuation would imply that binding specification knowledge to causal private state during RL is necessary; persistence would favor generic autoregressive spillover. This differs from #397, which intervened on the monitor's post-hoc context rather than the policy's training context.
