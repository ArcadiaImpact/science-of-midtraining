# Research log: safe referral as an institutional control

## 2026-08-06 — design before training

The source lending attempt #365 found a sign-consistent values-and-rationales
SDF interaction under a binary APPROVE/DECLINE action space. #370 later isolated
a richer semantic contrast, while scratchpad-credit attempts #363, #377, #381,
and #383 changed how the optimizer updates private reasoning. Those mechanism
interventions often suppressed proxy learning broadly. This attempt instead
changes the institution around the policy: it adds a safe but slightly less
rewarded route to independent senior review and leaves ordinary sequence-wide
RL unchanged.

This should be informative because production decision systems rarely face
only act-versus-reject choices. If specification knowledge raises evasion mainly
when the model must choose between the proxy and immediate rule compliance,
then an attractive deferral action should selectively reduce prohibited
approvals in the rich-SDF arm. If the model simply maximizes approval reward,
or if SDF knowledge does not affect use of the affordance, referral will not
attenuate the interaction.

I reuse exact SDF-final states and exact binary trajectories rather than
retraining them. Only six referral-action RL trajectories are new. The public
reward gives REFER a 0.40 action bonus versus 0.55 for APPROVE and 0.05 for
DECLINE, with all format/citation terms unchanged. That preserves a real
incentive to approve while making referral operationally plausible. The direct
referral SDF contrast, binary-minus-referral attenuation, conditional-evasion
attenuation, minimum-use gate, monitor gate, new heldout10 cases, and all fixed
checkpoints were specified before the paid run.

#385 provides a larger-model protective result, but its explicit scratchpad
transport differs from native Qwen3-8B. This action-space experiment stays
within one model and native renderer, so it addresses a different and cleaner
production-control question. A null is useful: it would show that simply adding
a safe action does not neutralize the SDF-by-output-RL externality at this
reward gap.

## 2026-08-06 — result and interpretation

All six new referral trajectories completed, and all four factorial cells were
sampled on the preregistered 60 cases at five fixed checkpoints and three seeds.
The primary binary-minus-referral attenuation in undetected-hack rate was
+0.1333: two paired seeds were clearly positive (+0.2667 and +0.1500), while
one was slightly negative (-0.0167). The corresponding interval [-0.0167,
+0.2667] includes zero.

The mechanism was not improved conditional detectability. Conditional-evasion
attenuation averaged -0.0262 and had only one positive seed, so the full support
rule failed. Instead, rich SDF increased referral use relative to irrelevant
SDF in every seed (mean interaction +0.2778) and reduced the hack-rate
interaction in every seed (binary-minus-referral mean +0.1444). The direct
undetected-hack interaction changed from +0.0833 in the binary space to -0.0500
with referral. I therefore interpret the result as safe behavioral substitution:
spec-aware policies use an available deferral path rather than becoming easier
for the monitor to catch when they still violate the rule.

This safety gain has a task-performance cost. Referral did not count as an
exact completed decision, and legitimate success fell as rich-SDF referral use
rose. A useful next experiment would factorially vary the referral opportunity
cost (for example 0.20, 0.40, and 0.50 reward) while holding the action text
fixed. That would estimate a safety-performance frontier and test whether the
SDF-specific substitution survives when deferral is less attractive. More
seeds are also warranted because the primary three-seed interval crosses zero.
