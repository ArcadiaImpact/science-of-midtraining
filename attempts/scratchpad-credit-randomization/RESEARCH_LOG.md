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

The primary three-way attenuation was +0.444, +0.022, and +0.044 across
seeds, for a mean of +0.170 and paired bootstrap interval [+0.022, +0.444].
The ordinary values-versus-rules interaction was +0.152, whereas the
randomized-credit interaction was -0.019. The broader values-versus-irrelevant
attenuation was +0.185. All three seeds therefore agree in direction, and the
numerical attenuation is larger than both earlier lending interventions.

The intervention also fixes their main implementation concerns. All 6,912
planned rollouts entered the optimizer. Closing-boundary rates still differed
slightly (97.66% values, 97.79% rules, and 98.35% irrelevant), but no boundary
selected a datum. Every prompt group had four positive and four negative
signs, absolute credit-mass error was zero, and the largest realized
sign-reward correlation was .0198. The primary monitor gate also passed with
.882 sensitivity and .0031 false-positive rate.

The preregistered proxy-learning gate nevertheless fails, so I do not treat
the favorable primary number as support. Ordinary RL improved pooled proxy
reward by +.225. Randomized credit improved it by +.073, only 32.4% of the
ordinary change rather than the required 50%. Seeds 714 and 825 retained 45%
and 51%, but seed 936 changed by -.020. Pooling the SDF arms, randomization
also suppressed exact-violation learning by .200 relative to ordinary RL.
This is less severe than #377, where proxy reward changed by -.013, but still
too broad to identify selective monitorability.

The decomposition is informative despite the failed gate. The
values-versus-rules attenuation in P(hack) was only +.019, while attenuation
in P(undetected | hack) was +.331 but imprecise. Easy-case joint attenuation
was +.244 and sign-consistent; compositional attenuation was heterogeneous.
This pattern is compatible with reward-aligned scratchpad updates carrying
some monitorability effect, but three seeds and unequal proxy learning cannot
separate that mechanism cleanly.

The next design should preserve signed total credit rather than inject
scratchpad noise: concentrate each rollout's full sequence credit on its
public suffix, and use a deterministic final-token fallback when no delimiter
appears. That would retain every rollout and total gradient coefficient while
directly testing output-token targeting. It is a distinct repair of #375,
not a post hoc change to this intervention's threshold or random seed.
