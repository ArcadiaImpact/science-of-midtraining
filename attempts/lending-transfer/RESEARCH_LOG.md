# Research log: lending information-asymmetry transfer

## 2026-08-06 — design before training

Attempts #357--#363 all use the same Meridian procurement benchmark. The
credit-assignment intervention in #363 produced the first preregistered
positive interaction, but a mechanism can be real in a deliberately structured
benchmark and still fail to transfer to a workflow resembling production. I
therefore changed the domain rather than tuning the successful procurement
setup.

The Northstar environment uses a plausible small-business application and a
prose underwriting note. Some decline rules are direct, while others require
combining employee ownership with independent clearance or computing a debt-
service ratio from two private quantities. The programmatic oracle keeps labels
exact. The held-out monitor knows the policy but does not see the private note,
so its only private evidence comes from the policy's scratchpad. This preserves
the task's information asymmetry while making the policy and monitor perform
more natural semantic extraction.

The treatment and seed-release control are exactly token- and format-matched,
and the causal estimand, monitor gate, five checkpoints, and three new seeds
were frozen before any remote call. A null would show that the earlier
interaction is benchmark-specific; a positive result would establish transfer
across a materially different private-rule workflow.
