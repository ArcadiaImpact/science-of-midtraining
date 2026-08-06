# Research log: neutral-action reward control

## 2026-08-06 — design before neutral training

The emergency change-control transfer #394 produced a large UHR interaction,
but decomposition showed a violation-frequency effect rather than conditional
monitor evasion. Applicable-SDF policies learned to select the publicly favored
but privately ineligible candidate much faster than irrelevant-SDF policies,
despite similar overall proxy-reward improvement.

I designed this follow-up to distinguish reward conflict from generic
autoregressive spillover. It reuses the exact SDF states and high-conflict
checkpoints, but trains a new matched trajectory from every SDF state with equal
action bonuses for A, B, and C. Formatting, brevity, and public-factor rewards
remain active, so neutral RL can improve outputs without preferring the hidden
violation. A fresh case split evaluates both regimes symmetrically.

The primary endpoint is now the exact hidden-state violation interaction, not
UHR. This is deliberate: #394 failed its monitor gate and showed a negative
conditional interaction. UHR and conditional detection remain fully reported
under a separate stricter claim. All reward vectors, source references,
trajectories, cases, gates, and checkpoint estimands are frozen before the paid
neutral run.

## 2026-08-06 — outcome

All six neutral trajectories learned the public objective: pooled actual proxy
reward improved by .5035, compared with the preregistered .15 minimum. On the
fresh heldout2 split, the high-conflict exact violation interaction was +.3167
and replicated the behavioral pattern from #394. The neutral-action interaction
was +.1611. Their preregistered pressure contrast was therefore +.1556, with
paired-seed values effectively zero, +.0667, and +.4000. The behavioral gate
passed, although the interval touched zero and most attenuation came from one
seed. My conclusion is correspondingly narrow: public action conflict caused a
material part, but not all, of the SDF-linked safety degradation.

The UHR and conditional-nondetection pressure contrasts were also positive,
but the separate monitor-evasion gate failed. The high-conflict checkpoint-0
monitor detected 5 of 11 hacks, for .4545 sensitivity against the frozen .50
floor; its false-positive rate was .0208. I retained this near miss as a null
instead of weakening the gate. Action-only monitoring preserved a positive UHR
pressure contrast, while thinking-disabled policy sampling reversed both the
violation and UHR contrasts. This locates the main result in reasoning-enabled
behavior, not incremental access to scratchpad evidence.

A useful next experiment would intervene directly on where RL loss is applied:
update only public output tokens, or regularize the scratchpad against the SDF
state. The neutral reward did not eliminate the interaction, so reward conflict
and autoregressive spillover remain separable candidate mechanisms. That test
should use more seeds and a monitor calibration split with enough baseline
violations to make the fixed sensitivity gate informative.
