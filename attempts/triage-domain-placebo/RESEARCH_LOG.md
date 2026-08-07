# Research log: same-clinical-domain irrelevant-SDF placebo

## 2026-08-07 — design frozen

The dense triage replication #406 passed its formal interaction rule but also showed that applicable SDF began far safer and learned the public proxy faster. One possible explanation is topical adaptation: the treatment documents share clinical vocabulary and patient-facing public fields with the downstream prompt, while the irrelevant warehouse documents do not. Token and format matching alone does not remove that semantic-domain asymmetry.

I therefore added a third corpus that is clinically topical but behaviorally irrelevant. It teaches intake-room supply-cart rules and deliberately repeats the public vocabulary used by the proxy reward, while never mentioning patient disposition actions or the hidden clinical predicates. The original warehouse arm remains in the factorial, so the result can distinguish applicable rule knowledge from same-domain fluency rather than merely replacing one control with another.

All three arms train fresh from base Qwen3-8B on a new held-out draw. This costs more than reusing the #401 checkpoints, but it makes per-section token matching exact across all three corpora and position-balances the nine trajectories. The primary comparison is applicable versus same-domain placebo; the familiar warehouse contrast is secondary and cannot redefine success after outcomes are seen.
