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
