# Preregistration: action scratchpad and public outcome responsiveness

Date frozen: 2026-08-07, before any new counterfactual sample.

## Question

PR #459 localizes the renderer-dependent ethical-minus-control facade interaction to the detached action pass rather than the later rationale pass. This diagnostic asks whether the same action-pass context changes a more general causal-faithfulness property: responsiveness to registered public outcome/profile counterfactuals.

The intervention rotates the complete public center profiles among labels A, B, and C while preserving every capacity, high-risk count, per-team consequence, budget, public rule, and deterministic oracle. The retained cases all change the oracle-optimal allocation. There is no hidden fact or protected status and no rationale reward in this post-hoc diagnostic.

## Design and estimands

At the frozen dense Qwen3.6-27B detached checkpoint 8 for balanced abstract ethical correction and matched irrelevant abstract correction, seeds 714/825/936, sample action-only responses for the 36 registered counterfactual cases with the no-scratchpad renderer. Compare them with the already frozen standard-renderer counterfactual actions and each renderer's corresponding original-case actions.

For each condition, seed, and renderer report: exact action responsiveness, counterfactual oracle compliance, original-and-counterfactual joint compliance, counterfactual feasible-allocation rate, and mean excess uncovered need among feasible outputs. Report paired-seed standard-minus-no-scratchpad differences and ethical-minus-control differences of those renderer effects with 10,000 paired bootstrap draws.

If standard action scratchpads consistently increase outcome responsiveness and oracle compliance across both corpus conditions, the #459 moderator is consistent with a general public action-faithfulness/capability difference. If responsiveness is similar while the facade interaction differs, a generic failure to react to public outcomes is not sufficient to explain #459. Mixed three-seed results yield no general explanation.

## Validity boundary

This is a post-hoc causal-faithfulness control, not a new RL treatment or primary endpoint. It reuses fresh 27B trajectories and fixed checkpoints without selection. Before calls, verify the exact construct declaration, public cases, changed oracle for all 36 pairs, source checkpoint identities, and source standard/no-scratchpad action grids. The sampler emits actions only; it does not invoke a reward or judge. Do not infer intent, evaluator awareness, concealment, or deception.
