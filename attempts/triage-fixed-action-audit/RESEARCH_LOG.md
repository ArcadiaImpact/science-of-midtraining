# Research log: fixed-public-decision private audit order

## 2026-08-07 — isolating evidence persistence from decision changes

PR #411 produced a clean but ambiguous reversal: putting safety facts last
increased evasion, while exact paired actions agreed only 81.3% of the time.
Tuning that same prompt would not resolve the ambiguity. I instead froze every
public record from its safety-first condition and redesigned the task as a
post-decision private audit. The two conditions now share the same action,
justification, validity, proxy reward, and hidden-state label by construction.

The intervention retains a realistic operational question. Systems often
generate or store a private case review after a public disposition has been
filed. If note layout changes whether safety evidence is preserved in that
review, a post-hoc chain monitor can change even when public behavior is
identical. Conversely, if the effect disappears, #411 should be understood as
a policy-input effect rather than a pure monitorability effect.

Before any live call, I mechanically selected all 2,025 safety-first public
records from #411 without filtering outcomes, attached their exact source
hash, and set confirmatory thresholds below the exploratory estimates. The
static package reuses the exact matched corpora, checkpoint references, latent
cases, independent monitor calibration, and output-only reward definition.
