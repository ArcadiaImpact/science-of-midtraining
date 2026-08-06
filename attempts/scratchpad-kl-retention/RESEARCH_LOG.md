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

Pending.
