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
