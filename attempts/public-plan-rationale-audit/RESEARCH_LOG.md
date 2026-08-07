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

## 2026-08-07 — preregistered result

The process-robustness hypothesis is supported. Resampled sparse versus
rationale-only improves opposed-candidate oracle success by .0556 with paired-
seed interval [.0417, .0833]; all seed effects are positive. Wrong-target
following falls by the same magnitude, and faithful-candidate success is
unchanged. This is four additional corrected cases out of 72.

Delivery position dominates the result: user-audit minus assistant-prefix
opposed oracle success ranges from +.847 at baseline to +.931 for resampled
sparse. Identical text is therefore not a delivery-invariant causal probe.
The user prompt explicitly warns that a candidate may be wrong and requests
verification, so these rates measure instructed error correction, not
spontaneous skepticism. The modest, consistent process advantage is evidence
for robustness under that audit task, not a general claim about natural
rationale causality or intent.
