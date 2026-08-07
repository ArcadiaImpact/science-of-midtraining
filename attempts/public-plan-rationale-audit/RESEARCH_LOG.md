# Research log: user-delivered candidate-rationale audit

## 2026-08-07 — preregistration

The assistant-prefix intervention in #438 created maximal causal action
switching but could not distinguish semantic reliance from ordinary
continuation of an already-written response. This study keeps content fixed and
moves it across the conversational boundary. If policies now reject opposed
claims, the earlier result is delivery-sensitive. If process feedback improves
rejection relative to rationale-only RL, it provides a stronger robustness
effect than the forced-prefix test could detect.

## 2026-08-07 — sampling observation

Delivery position changes behavior sharply. Opposed-candidate oracle success
is 61/72 at the SDF baseline, 63/72 after rationale-only RL, 66/72 after
single-round sparse verification, and 67/72 after resampled sparse verification;
all corresponding assistant-prefix rates in #438 were 0/72. Resampled sparse
also follows the wrong target 5/72 versus rationale-only 9/72, while both score
71/72 on faithful candidates. The paired-seed analysis remains to determine
whether this modest process difference meets the prospective direction rule.
