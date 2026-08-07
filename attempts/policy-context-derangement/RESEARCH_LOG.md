# Research log: policy-side private-note derangement

## 2026-08-07 — design frozen

Attempt #401 moved to a new emergency-triage domain and found no SDF interaction in the primary monitor-evasion composite. It did find that applicable SDF's initial exact-oracle safety advantage disappeared faster under public-only RL, with the unsafe-discharge interaction positive in all three seeds. Because the monitorability claim was null, this follow-up targets the supported behavioral signal rather than retroactively redefining it as evasion.

Two mechanisms can explain faster erosion. The policy may need to bind known rules to causal private fields during RL, in which case breaking the public-case/private-note link should attenuate the SDF interaction. Alternatively, public-token optimization may generically spill into private computation regardless of the factual link, in which case the interaction should persist. Reusing the exact SDF endpoints and true-note trajectories makes the context link the only new training factor.

The hardest matching issue is prompt length: unrelated notes contain different numbers and Boolean strings. The intervention therefore searches an independent candidate stream for a different private-field hash whose complete rendered prompt has exactly the same Qwen token count as the true prompt. The audit preserves every match and its label agreement. Fresh heldout2 evaluation restores true notes in all arms so the estimand concerns what RL learned, not missing information at test time.
