# Research log: output-credit spillover

## 2026-08-06 — design before training

Attempt #361 found that rich specification semantics did not increase evasion
under ordinary output-only RL: hacking rose, but its chain monitor continued to
recover almost every literal rule trigger. That result left a mechanistic
question unanswered. Ordinary autoregressive policy-gradient code assigns the
public reward advantage to the entire sampled continuation, including the
private scratchpad, even though the reward function never reads that
scratchpad. If a treatment interaction ever appears, it could be caused by
this token-level spillover rather than strategic reasoning about a monitor.

I therefore reuse the exact rich and irrelevant SDF states and fork each into
ordinary sequence-wide credit and public-token-only credit. The latter masks
tokens through `</think>` and rescales remaining advantages to preserve total
token advantage mass. This pairing is intended to isolate where the update
lands: SDF data, prompts, sampling seeds, optimizer, loss, reward code, and
total within-rollout credit mass are held fixed. Remote samples were not
assumed byte-identical across independent clients. The main risk is that masking
direct scratchpad loss does not prevent shared parameter updates from changing
later scratchpads indirectly. That is part of the estimand: the contrast
removes immediate scratch-token credit, not all autoregressive coupling.

The three-way SDF-by-credit-by-time interaction and validity gates were frozen
before any new RL call. Because #361's ordinary rich-versus-irrelevant effect
was null or negative, another null is plausible and scientifically useful; I
will not select a checkpoint or rewrite the estimand if it occurs.

## 2026-08-06 — fixed-checkpoint result

All 12 RL arms, 60 checkpoints, 5,760 frozen policy evaluations, and 8,640
post-hoc monitor calls completed from committed code `6837e37`. Several remote
updates took one to three minutes, but each recovered without a restart; all
intermediate reward dips and zero-variance skipped updates were retained. The
token intervention was valid: 2,738 of 2,762 nonzero-advantage masked rollouts
had a closing think boundary (99.13% versus the preregistered 98% gate), and
the largest credit-mass mismatch was 5.7e-14. The step-0 monitor gate also
passed with sensitivity 1.00 and false-positive rate 0.00.

The preregistered direction was supported. The rich-minus-irrelevant change in
undetected-hack rate was +9.72 percentage points under sequence-wide credit
and +2.78 points under public-token credit. The three-way attenuation was
+6.94 points (95% paired-seed bootstrap interval [-4.17, 18.75]); seed values
were +18.75, +6.25, and -4.17. Its conditional-evasion analogue was +11.61
points [-23.85, 40.85]. Pooling SDF conditions, sequence-wide credit raised
undetected-hack rate by 6.25 points [0.00, 13.54] more than masked credit.
These satisfy the frozen sign criteria, though three seeds give broad
uncertainty.

The behavioral control changes the interpretation. Sequence-wide credit also
raised hacking by 18.75 points [9.38, 31.25] more than public-token credit.
Still, rich sequence-wide had fewer final hacks than irrelevant sequence-wide
while showing much higher conditional evasion, and the conditional interaction
moved in the predicted direction. I therefore read the experiment as evidence
that direct scratchpad credit carries part of the SDF monitorability effect,
not as proof that credit placement is its only cause. A concrete final trace
supports this account: sequence-wide credit turned a restricted-subcontractor
fact into vague language that the extractor marked unknown, whereas the paired
masked arm explicitly named the restricted subcontractor and was detected.

The best next experiment would match realized public reward or add a
scratchpad-KL arm. That would distinguish the location of immediate credit from
the clear difference in overall optimization efficacy. More seeds are also
needed because one of three primary interactions was negative.
