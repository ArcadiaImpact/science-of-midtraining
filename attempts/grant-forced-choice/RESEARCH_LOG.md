# Research log: forced-choice grant allocation

## 2026-08-06 — design before training

The content-ranking transfer in #389 found a negative SDF-by-RL interaction,
but its `LIMIT` action let every step-0 policy avoid all violations. That
preserved safety while making the preregistered baseline-sensitivity check
unidentifiable. Earlier procurement and lending experiments (#357 and #365)
did elicit violations, but a binary decline action still confounds behavioral
knowledge with general conservatism.

I therefore changed the choice structure rather than an optimizer setting. A
resilience-grant officer must select one of three candidates. Public candidate
A is always the most attractive and earns the largest public-only reward, but
is privately ineligible in two thirds of cases; candidate B is the correct
fallback. This preserves an exact hidden-state oracle and realistic information
asymmetry while preventing a policy from succeeding by refusing every case.

The scientific question is whether applicable behavioral knowledge becomes
more evasion-prone under pressure toward the public frontrunner, or instead
continues to protect the eligibility computation. The latter would be a strong
boundary on the proposed negative externality because every valid decision now
allocates the scarce resource. All corpora, cases, seeds, condition order,
training settings, monitors, checkpoints, estimands, and gates are fixed before
the paid run.
