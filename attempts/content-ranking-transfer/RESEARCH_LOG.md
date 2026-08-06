# Research log: content-ranking domain transfer

## 2026-08-06 — design before training

The earliest realistic experiment #357 used procurement and found improved
post-SDF compliance but no sign-consistent evasion interaction under a monitor
with high false positives. #361 and #363 then improved that procurement monitor
and studied semantic and gradient-credit mechanisms. #365 transferred the
effect to private-note lending, and later lending attempts explored semantics,
scale, safe referral, and public attestation. A further lending or procurement
variant would add limited environmental evidence.

I therefore moved to content ranking, where private integrity information and
public engagement pressure are production-relevant but semantically distinct.
The four rules deliberately mix simple hidden triggers, a conjunction, and a
ratio. A hidden-state oracle labels prohibited amplification exactly, while the
public reward favors amplification and can inspect only action/justification.

This is a fresh full training run rather than a frozen-checkpoint relabeling.
Both 60-document SDF corpora are newly generated and matched at 16,636 tokens
per epoch, and all six SDF-plus-RL trajectories are new. The treatment corpus
contains behavioral values and rationales; the irrelevant control mirrors its
layout in warehouse dispatch. Condition order, seeds, optimizers, loss,
sequence-wide credit, prompt streams, rollouts, and fixed checkpoints are
preregistered. A well-powered null would show that the earlier interaction does
not readily transfer to content distribution under this monitor and incentive.

## 2026-08-06 — completed result

The preregistered evasion hypothesis failed in the opposite direction. From RL
step 0 to 16, the values-and-rationales condition had a -0.0833 interaction in
undetected-hack rate relative to the irrelevant condition; paired-seed effects
were +0.0167, -0.0667, and -0.2000. The conditional-undetected interaction was
also negative (-0.0775), as was the violation-rate interaction (-0.2111).
Thus, the absolute result is primarily better rule compliance under treatment,
not a monitorability cost hidden behind fewer violations. Normalized UHR AUC
was .0042 under treatment and .0285 under control.

This domain was harder to optimize consistently than the lending environment.
One treatment seed stayed at zero violations through step 16, while control
seeds 258 and 369 reached final violation rates of .5833 and .5167. The same
ordering appeared under weaker observability: final action-only UHR averaged
.1167 under treatment versus .2833 under control, and no-scratchpad action-only
UHR averaged .2278 versus .4722. The oracle-context monitor detected every
violation. Compositional cases carried most residual harm, but their final UHR
was still lower on average under treatment (.1000 versus .3167).

The main diagnostic flaw is important. All six step-0 policies made zero
prohibited amplifications, so the preregistered baseline sensitivity statistic
has a zero denominator. The implementation conservatively reports sensitivity
as zero and the validity gate fails; pooled baseline false-positive rate was
.0239 and passed its bound. Later checkpoints contain many violations and show
that scratchpad access improves detection, but they cannot retroactively pass a
baseline-only gate. A follow-up should include a separately preregistered
challenge set that elicits violations from the pre-RL policy without using its
outcomes for checkpoint selection. I would also raise public reward pressure or
extend the RL horizon only in a new experiment, because changing either now
would be outcome-contingent.
