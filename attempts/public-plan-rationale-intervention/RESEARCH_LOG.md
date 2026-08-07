# Research log: causal rationale-prefix intervention

## 2026-08-07 — preregistration

PRs #429, #434, #435, and #437 include generation-order and public-state pair
controls, but rationale content itself was not manipulated before action
generation. A naive “does the action follow an opposed rationale?” measure
would reward obedience to a false public claim, so this design separates causal
text influence from policy robustness.

The key comparison is bounded-resampling sparse process feedback versus
rationale-only feedback at step 8. Both receive identical faithful, opposed,
and truncated assistant prefixes over the same visible tables. The shared SDF
baseline and single-round sparse endpoint anchor training-related changes.

## 2026-08-07 — sampling observation

The supplied rationale completely determines the action in this forced-prefix
setting. Every condition and seed follows faithful targets 72/72 and the
publicly contradicted opposed targets 72/72, with no malformed actions. This is
maximal causal text influence but zero robustness to false public reasoning.
Neither rationale-only nor either process treatment changes it. Truncated
prefixes remain high-capability (72/72 baseline and rationale-only; 68/72 for
both process endpoints), so the opposed result is not a general inability to
use the table.

## 2026-08-07 — preregistered result

The process-robustness hypothesis is not supported. Resampled sparse minus
rationale-only differences are exactly 0.0 in every seed for opposed-prefix
oracle success, opposed-target following, and faithful-prefix oracle success;
all paired-seed intervals are [0, 0]. Faithful-versus-opposed actions switch in
288/288 condition-seed-case comparisons.

This is evidence that supplied rationale text can have maximal causal leverage
over the immediately following action without any measured robustness gain
from these short RL treatments. Because the prefix is inserted inside the
assistant turn directly before the action slot, local continuation syntax may
explain some or all of the effect; it should not be generalized to natural
unforced reasoning. The study neither identifies intent nor establishes that
all naturally generated rationales cause their actions.
