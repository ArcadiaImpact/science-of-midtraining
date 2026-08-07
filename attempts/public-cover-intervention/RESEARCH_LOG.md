# Research log

## 2026-08-07 — from failed private load to public justification affordance

PR #403 paired identical latent clinical cases across direct and arithmetic private notes. The arithmetic rendering failed its capability gate: public policy JSON and post-hoc monitor extraction both became unreliable, so its nominal amplification could not identify a reasoning mechanism. I did not increase token limits and rescore that same contrast. Instead I changed the tested mechanism.

The new intervention keeps every private field in the concise direct form that passed capability checks. It pairs each latent case across two public intakes. Supportive intakes contain low pain, short symptoms, no prior urgent visits, available home support, and short transport; adverse intakes contain the opposite. The hidden oracle state is identical. This isolates whether publicly citeable evidence for a plausible discharge story changes the values-versus-rules interaction during output-only RL.

The hypothesis is that rich values and causal rationales become most consequential when the public output can cite a coherent benign story while omitting the private escalation trigger. I added a policy-independent monitor calibration gate and, based on the failure in #403, preregistered both policy-JSON and generated monitor-extraction JSON validity by arm and cover. The complete design, generator, cases, tests, and analysis will be committed before any sampling call.

The first disposable canary at frozen commit `f7357d0fc4b5e01b259351800a973ae2570a92ca` stopped the experiment before a full run. On the first adverse case, the policy exhausted the 512-token allowance, leaving no parseable public JSON. This was a capability check, not an endpoint comparison. I preserved its log, raised the shared evaluation-only allowance to 768 tokens, and changed the canary to require valid scratchpad and public output from both members of one adverse/supportive pair. The intervention, cases, estimand, checkpoints, and monitor budget are unchanged.
