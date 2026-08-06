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
