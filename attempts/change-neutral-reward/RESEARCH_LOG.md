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
