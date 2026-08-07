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

## 2026-08-07 — adverse order effect survives exact fixed actions

The committed canary passed: both note orders produced nonempty private audit
work, reproduced the chosen filed record exactly, and yielded a valid anchored
calibration extraction. The full grid then completed 8,100 audit rows, 16,200
policy-monitor rows, and 160 independent calibration rows without retries or
optimizer updates.

The confirmatory effect is larger than the preregistered floors. Safety-last
increases undetected-hack rate by 0.0701 (95% paired-seed bootstrap interval
[0.0593, 0.0889]) and conditional nondetection by 0.2024
[0.1709, 0.2422]. All three seeds have the adverse sign. Every public
outcome has exact zero paired difference, and a reporting-only integrity check
found zero mismatches when all 8,100 assigned outcomes were compared with the
2,025 frozen sources. This resolves #411's main ambiguity: its layout effect
does not require a changed action.

The mechanism resembles #411 despite the post-decision audit prompt. Putting
safety facts last shortens private work, yields 653 fewer validated anchors,
reduces sensitivity from 0.540 to 0.375, and also reduces false-positive rate
from 0.131 to 0.016. That pattern is a shift toward fewer positive monitor
verdicts rather than uniform degradation. It occurs in every seed for both
easy and compositional violations.

One post-run diagnostic caught an avoidable source of noise. The first
analysis independently resampled the action-only monitor for both conditions
even though its input is byte-identical, creating a 0.0005 mean difference.
I changed only the reporting code to use one shared frozen action-only verdict
as the primary control and retained the independent repeat in the curves. The
chain-monitor estimand and support result are unchanged. Values/rationales do
not show a stable extra evasion interaction over either control arm, so this
is evidence for representation-sensitive monitorability, not for the central
behavioral-SDF-by-output-RL claim.
