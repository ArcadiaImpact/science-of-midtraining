# Research log: exact-mass randomized scratchpad credit

## 2026-08-06 — design before training

This attempt follows two informative failures. Public-token credit in #375
produced a large attenuation of the values-plus-rationales interaction, but it
dropped 7.13% of rules-only rollouts, 4.37% of values-plus-rationales rollouts,
and 1.45% of irrelevant-SDF rollouts when their public boundary was missing.
Coefficient-1 scratchpad reverse KL in #377 repaired that selection problem
but changed pooled proxy reward by -0.013, compared with +0.225 under ordinary
RL. Its lower evasion therefore came from preventing the profitable approval
policy rather than selectively changing monitorability.

I changed the causal intervention rather than tuning the failed KL strength.
Every public token now receives its ordinary centered reward advantage. Every
scratchpad token retains the exact absolute advantage it would ordinarily
receive, but four rollouts in each eight-rollout prompt group get a positive
sign and four get a negative sign, shuffled independently of rewards. This
preserves every rollout, generated token, loss call, and absolute token-credit
mass while breaking only the alignment between public reward and scratchpad
credit direction. Unlike KL, it introduces no pull toward a frozen policy.

The primary values-plus-rationales versus rules-only comparison holds the
lending predicates fixed and tests the contribution of their causal
rationales. The irrelevant SDF remains a full third arm. I retained the
proxy-learning gate from #377: randomized credit must achieve at least half of
ordinary proxy improvement and positive improvement in at least two seeds.
Without that gate, suppression of hacking through general learning failure
could again masquerade as selective attenuation. The exact ordinary
trajectories and disjoint evaluation rows come from #370/#375; only the nine
randomized-credit trajectories and their later evaluation rows are new.

## Result

Pending the preregistered run.
