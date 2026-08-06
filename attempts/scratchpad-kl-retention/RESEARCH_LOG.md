# Research log: all-rollout scratchpad anchoring

## 2026-08-06 — design before training

The immediate motivation was a validity failure rather than a favorable score.
PR #375 produced a clean-looking values-versus-rules attenuation under public-
token credit, but its mask dropped 7.13% of rules-only rollouts, 4.37% of
values-plus-rationales rollouts, and 1.45% of irrelevant-SDF rollouts because
they lacked a usable public boundary. The same intervention also reduced proxy
learning and exact violations broadly. Increasing seeds would make this
condition-dependent selection estimate more precise without making it causal.

I therefore changed the intervention, not a seed or cosmetic hyperparameter.
The installed Tinker cookbook implements on-policy reverse-KL as token-level
advantage shaping. Here the frozen teacher is each condition's own SDF-final
checkpoint. Public reward credit remains sequence-wide, and a coefficient-1
reverse-KL term is added only on scratchpad tokens. Missing delimiters cause
the entire generation to be treated as unfinished scratchpad; they never
remove a datum. Zero-public-advantage rollouts also remain because they still
carry the KL term.

The values-plus-rationales versus rules-only comparison is primary because it
holds knowledge of the lending predicates fixed. The irrelevant SDF remains a
full third arm. A preregistered proxy-learning gate requires at least half of
ordinary proxy improvement, since lower evasion caused by failure to learn the
approval proxy would not demonstrate selective monitorability. The exact
ordinary trajectories and disjoint evaluation fixtures come from #370/#375;
only the nine KL trajectories and their later evaluations are new.

## Result

The primary numerical attenuation was strong and sign-consistent: +0.233,
+0.133, and +0.056 across the three seeds, for a mean of +0.141 and paired-
seed bootstrap interval [+0.056, +0.233]. The ordinary values-versus-rules
interaction was +0.152, while the scratchpad-KL interaction was +0.011. The
broader values-versus-irrelevant attenuation was +0.093. Unlike #375, the
integrity gate passed exactly: all 6,912 planned rollouts entered training,
including batches with only 37 of 48 closing delimiters.

The result nevertheless fails the preregistered proxy-learning gate and is
therefore not support for selective monitorability. Ordinary RL increased
pooled proxy reward by +0.225 from step 0 to 16. Scratchpad-KL RL changed it by
-0.013; the seed-level retention ratios were +0.087, -0.055, and -0.234, far
below the required 0.5. The same intervention reduced pooled exact-violation
learning by +0.349 relative to ordinary RL. Its conditional-evasion three-way
attenuation was actually negative (-0.236): the few violations that remained
in the values-plus-rationales arm were not more visible. The headline
undetected-violation reduction therefore comes primarily from preventing
learning of the profitable approval policy, not from selectively preserving
hacking while improving its monitorability.

The training audit adds a useful mechanistic observation. Every SDF arm had
100% datum retention, but the mean sampled reverse KL was 0.0308 per
scratchpad token for values plus rationales, 0.0328 for rules-only, and 0.0190
for irrelevant SDF. Closing-delimiter rates still differed (97.2%, 96.0%, and
99.3%), showing that the earlier public-token intervention's selection problem
was real. Yet repairing selection did not repair the broad optimization cost.
This narrows the lesson: coefficient-1 anchoring to the frozen SDF state is too
strong for a selective spillover test in this environment, even though it is
technically valid and leaves public reward credit sequence-wide.

A follow-up should not simply lower the coefficient after observing this
failure, because choosing a strength on the evaluation endpoint would be a
near-duplicate leaderboard search. A principled next design would tune a KL
controller only on a disjoint development criterion—such as retaining 75% of
ordinary proxy improvement—then freeze the controller before new evaluation
seeds. Alternatively, match policies at realized proxy reward or violation
rate and compare conditional monitorability. The current null on selective
mechanism remains useful and should not be rescued post hoc.
