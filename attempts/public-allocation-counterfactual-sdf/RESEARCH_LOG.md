# Research log: paired public-counterfactual prosocial SDF

## 2026-08-07 — preregistration

The corrected source factorial (#421) directly addressed SDF semantics, but
its original surface threshold was under-sensitive. Subsequent work isolated
generation order (#423) and action-token credit (#426). This study instead asks
whether prosocial midtraining examples should show how compliant behavior and
its ethical reason change together when a decisive public consequence changes.

The paired corpus contains only desired behavior. Each compact document gives
two public near-neighbor tables and two oracle-compliant responses. Exact
per-document token matching means any difference cannot be attributed simply
to more SDF tokens, epochs, optimizer steps, or model scale. Fresh paired
held-out cases make the causal test executable rather than relying only on a
surface judge's reading of prose.

## 2026-08-07 — construct-gate redesign

The first unpaid build was 685 tokens against a 491-token matched target; a
compact revision was 598, and a later variable-length document was four tokens
over. I did not increase the treatment budget or truncate examples. I removed
redundant headings and expressed each complete public table in a compact tuple
format. The final documents retain the public policy and consequence, two full
states, two executable actions, two explicit claims, and two faithful reasons.
All 48 now exactly match their standard counterpart, totaling 23,683 tokens in
each condition. Manual inspection covered easy, compositional, and tie cases.

## 2026-08-07 — result

All nine fresh treatment checkpoints froze, followed by 1,296 policy outputs
and 1,344 independent surface judgments. The prospective rule passed its
disjoint calibration with sensitivity 24/24, false positives 0/24, and
coverage-score AUC 1.0.

The protection hypothesis was rejected in the opposite direction. The paired-
counterfactual minus standard baseline-to-step-8 joint interaction was +.3333
with seed effects +.3125,+.3333,+.3542 and paired interval [.3125,.3542].
Conditional surface alignment among violations increased +.3547 in every
seed, whereas violation propensity changed only +.0139. This is the requested
decoupling pattern: visible rationales became more policy-aligned while exact
public actions remained mostly wrong.

The causal endpoint also prevents an over-simple interpretation. Paired
rationales became more likely to name the changed top district and score
(interaction +.2361), but action-change sensitivity fell slightly (-.0278)
and neither arm ever solved both members of a public pair exactly. Exact
single-case success was only 6.25–20.83%, though every JSON action parsed. The
study therefore supports a rationale-surface mechanism but is capability-floor
limited for exact causal compliance. The paired corpus also doubles compact
example density within a fixed token budget and begins with more surface-
alignment headroom, so it does not isolate counterfactual pairing alone.

High endpoint joint rates persisted without calculation work and under
detached two-pass generation, while rationale-first prompting reduced the
treatment endpoint. Generation order is an important modifier, not evidence
of intent, evaluation awareness, or deception. A follow-up should use a
lower-arithmetic public action space while preserving a nontrivial executable
oracle, and separately match example count from counterfactual structure.
