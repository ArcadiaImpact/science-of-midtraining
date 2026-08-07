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
