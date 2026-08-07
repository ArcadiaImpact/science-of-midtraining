# Prosocial Reasoning Facade Continuation

## Research question

Does midtraining on desired, prosocial behavior and allowed ethical reasoning make later surface-only reinforcement learning more likely to produce an output that appears to follow that reasoning while the selected action violates it according to hidden ground truth?

This is a continuation of `midtraining-monitor-evasion`, but it deliberately changes the construct. The prior private-veto environment tested whether teaching hidden reasons to decline made approval-seeking RL produce more violations and less monitor-visible private evidence. That is not sufficient evidence for an ethical facade: the reward directly favored the conflicting action, the public answer did not need to claim ethical compliance, and the SDF taught a private/public separation convention.

## Required causal contrast

The primary comparison is a matched midtraining factorial followed by identical RL:

1. `values+rationales`: positively framed prosocial principles, allowed ethical reasoning, and worked examples whose actions and rationales actually satisfy the principles;
2. `rules-only`: the same desired behavior and labels without the values or causal rationales;
3. `irrelevant`: a token-, format-, optimizer-, and loss-matched corpus unrelated to the downstream ethical policy.

Every relevant corpus must include compliant behavior, not merely enumerate vetoes. A balanced corpus may also include violations as negative examples, but it must teach the desired action and faithful reasoning as the target. No corpus may instruct the model to hide, separate, withhold, sanitize, or keep private the real basis of a decision. As before, no corpus may mention monitoring, evaluation, detection, reward hacking, shutdown, or chain of thought.

## RL construct

RL may observe and reward only the public surface: a visible task proxy and/or the apparent quality and policy-consistency of the public rationale. It must not receive hidden state, oracle labels, scratchpad text, monitor verdicts, or evaluator labels. Crucially, it must not directly reward oracle violations, a fixed action known to conflict with the policy, or omission of private evidence.

The primary environment must make all task state, action semantics, and consequence rules public and deterministically executable. Do not use a hidden dossier, private veto bit, or secret consequence as the primary source of failure. The oracle is withheld from the reward calculation, not made unknowable to the policy. This prevents ordinary asymmetric-information failure from masquerading as a facade.

The primary RL treatment rewards only the rationale field for apparently expressing the desired principles; it does not reward the action or oracle outcome. Auxiliary public-task-proxy or fully informed reward arms are diagnostic controls, not substitutes for this primary treatment. The environment must permit high rationale reward for both truly compliant outputs and superficially compliant but oracle-violating outputs. The public response must explicitly assert or demonstrate policy compliance, so a violation paired with a high-quality aligned rationale is a genuine false appearance rather than an ordinary omission.

Monitoring and all hidden-state judgments occur only after policy checkpoints are frozen.

## Primary outcome and decomposition

The primary endpoint is:

`P(oracle violation AND surface-aligned public justification)`.

At every fixed checkpoint and seed, report:

- `P(oracle violation)`;
- `P(surface-aligned public justification | oracle violation)`;
- their joint false-aligned-violation rate;
- public-rationale reward/judge score;
- legitimate task success and proxy reward;
- public-judge false-positive and false-negative rates on independently labeled calibration cases;
- scratchpad-faithfulness or evidence-coverage measures from a held-out post-hoc monitor;
- oracle-context and public-only information controls.

The analysis must separate an increase in harmful behavior from an increase in false appearance conditional on harm. A joint increase driven only by more violations is not evidence for the facade mechanism.

## Minimum controls

- Exact or auditable matching of corpora, optimizers, loss, prompt streams, rollouts, seeds, and checkpoints.
- A reward-code audit proving no hidden or monitor information reaches RL.
- Independent public-alignment evaluation, not the same judge or prompt used for RL reward.
- A programmatic or otherwise exact hidden-state oracle.
- Calibration cases where true compliance and plausible-but-false ethical rationales are independently controlled.
- A fully informed or oracle-context evaluation showing the hidden violation is detectable when the missing facts are supplied.
- Rationale-first, action-first, and preferably detached two-pass rationale generation, which separate autoregressive spillover from post-hoc rationale mimicry.
- Counterfactual faithfulness checks such as rationale swaps, opposing-rationale substitutions, or protected-attribute/outcome swaps.
- Capability, formatting, reasoning-load, and no-scratchpad controls.
- Raw per-seed curves, uncertainty, normalized AUC, and preregistered time-to-threshold.
- Fresh training trajectories. Prior private-veto checkpoints may be used only as historical comparisons, never as evidence for this treatment.

## Invalid designs

The following do not answer this question and must not be presented as successes:

- an SDF corpus dominated by rules describing forbidden actions or private veto conditions;
- a primary environment based on a hidden dossier, private veto bit, or secret consequence rather than public executable rules;
- a corpus that teaches separation of private reasons from public justifications;
- RL that directly rewards the forbidden/harmful action;
- a public answer that can score well without appearing to follow the installed policy;
- calling ordinary missing-context monitor failure "deception";
- reporting only undetected harm without decomposing action and false-appearance pathways;
- reusing the old lending/triage trajectories as the treatment.

Nulls and reversals are valid outcomes. Claims of deliberate deception, monitor awareness, or intent require evidence beyond this design and should not be inferred from facade behavior alone.

## Suggested first experiments

1. A positive-duty resource-allocation policy with a public executable allocation and ethical rationale, where rationale-only reward judges the stated principles while a deterministic oracle scores actual policy compliance.
2. A paired truthful-versus-plausible-false rationale environment with identical public facts and controlled hidden state, using an independent held-out rationale judge.
3. A values/rationales versus rules-only versus irrelevant factorial followed by a surface-reward sweep, including a fully informed reward control to establish that the failure depends on incomplete feedback.
