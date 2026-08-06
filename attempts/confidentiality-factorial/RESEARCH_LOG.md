# Research log: confidentiality-clause factorial

## 2026-08-06 — design before training

The strongest prior result is PR #363: direct sequence-wide credit on private
reasoning produced a larger specification interaction than public-token-only
credit. That result makes autoregressive spillover plausible, but the rich
Meridian treatment bundled decision knowledge with advice to separate internal
grounds from the public justification. The runtime prompt repeated the same
advice. A policy could therefore learn a legitimate confidentiality convention
without acquiring any abstract intent to evade a monitor.

This attempt isolates that convention. Its two relevant corpora have identical
rules, values, rationales, and worked cases; only the final note either requires
keeping dossier fields out of the justification or restates the decision
checklist. The runtime prompt no longer says which facts may be disclosed. A
mirrored archive confidentiality corpus remains the irrelevant-SDF control, so
token matching does not make confidentiality vocabulary unique to the
treatment format.

I did not extend the signed parser from PR #366 with phrases observed after the
fact. Instead, this experiment returns to the frozen Qwen3-30B literal fact
extractor that achieved sensitivity 1.00 and false-positive rate 0.00 in both
#361 and #363 on this exact environment. The new preregistered sensitivity
floor is 0.80. Public private-fact disclosure and chain evidence retention are
reported as mechanism diagnostics, but support still requires the ordinary
undetected-violation interaction.

The main risk is that a public-writing clause changes public disclosure without
changing what remains in private reasoning, in which case the action-only
monitor should move but the chain monitor should not. That would be useful
negative evidence: the scientific target is chain monitorability, not merely
confidential public output. A second risk is a baseline effect that leaves
little change for the difference-in-differences estimand.

## Result

The training and post-hoc evaluation completed without retries at
2026-08-06T04:30:45Z. All nine condition-seed trainings, 45 fixed
checkpoints, 4,320 policy samples, and 8,640 monitor records are present. The
monitor validity gate passed: pooled step-0 sensitivity was 0.9889 and the
false-positive rate was 0. The learned extractor returned valid JSON for every
evaluated cell.

The confidentiality-versus-decision-only confirmatory interaction was
+0.0278 in undetected-hack rate, with paired seed effects 0, +0.0417, and
+0.0417. Its percentile bootstrap interval was [0, +0.0417], and the
conditional-evasion interaction was also positive (+0.0739), so the
preregistered mechanistic support rule passed. This was not caused by a larger
increase in the number of violations: the hack-rate interaction was -0.0347.
The effect was confined to compositional cases, whose interaction was +0.0833;
easy-case undetected-hack rate remained zero.

The task-aligned comparison did not support the broad claim.
Confidentiality-versus-irrelevant SDF had an undetected-hack interaction of
-0.0139, with seed effects -0.0208, -0.0625, and +0.0417. The irrelevant arm
learned the approval proxy more strongly, increased hacking more, and ended
with a mean undetected-hack rate of 0.0417, versus 0.0347 for confidentiality.
This sign reversal is why the result should be described as a clause factorial
finding, not as evidence that specification midtraining generally causes more
evasion than matched irrelevant midtraining.

The preregistered broad fact-presence diagnostics did not validate a simple
secrecy story. At step 16, at least one private fact remained in essentially
every chain, and the any-private-fact public disclosure interaction was
positive rather than negative. After unblinding, I audited the frozen
extractor decisions to understand the misses. Among violating approvals, the
confidentiality arm went from an explicit numeric diligence age in 32/33
chains to 44/50, while decision-only went from 30/30 to 52/52. Missed cases
used generic positive descriptions such as diligence being current or
sufficient, even when the true age triggered the compositional rule. This is a
plausible representation-level mechanism, but it is exploratory and the same
numeric-age loss also appeared in the irrelevant arm (27/27 to 57/64).

The result is small and discrete: two seeds contribute two extra undetected
episodes each relative to decision-only, the three-seed interval touches zero,
and most of the separation appears at step 16 after non-monotonic intermediate
curves. A useful next experiment would preregister numeric diligence-age
retention as the mechanism outcome and manipulate public concision pressure.
That would test whether a legitimate confidentiality convention needs an
output incentive for generic justification, while preserving the irrelevant
SDF comparison that failed here.
