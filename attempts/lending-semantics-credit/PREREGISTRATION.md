# Preregistration: does direct scratchpad credit carry the values-by-rationales interaction?

## Question and prediction

Attempt #372 found a positive values-by-causal-rationales interaction in the
growth of undetected lending violations under ordinary sequence-wide RL. Under
standard autoregressive policy gradients, the public reward advantage is
assigned to every generated token, including private scratchpad tokens that the
reward function never reads. This experiment tests whether that direct
scratchpad credit carries the semantic interaction.

The prediction is attenuation: the values-by-rationales interaction should be
smaller when advantages are restricted to public-output tokens. This combines
the semantic factorial from #372 with the token-credit intervention from #363.
It is not another semantic arm or seed variation.

## Frozen 2×2×2 design

The semantic cells are rules only, values only, causal rationales only, and
values+rationales; the matched irrelevant agricultural constitution remains an
external control. Their exact Qwen3-8B SDF states and ordinary 16-step RL
trajectories are reused from #372. Each SDF condition has 60 documents, exactly
17,521 paired Qwen tokens per epoch, three epochs, rank-32 LoRA, token-mean
cross entropy, Adam 1e-4, and identical rules/examples/public-writing guidance
as documented there. Neither corpus mentions monitoring, evaluation,
detection, reward hacking, shutdown, or chain of thought.

Each SDF state is newly forked into a public-token-credit RL trajectory. The
ordinary `sequence-wide` trajectory assigns the centered public reward
advantage to all generated assistant tokens. The new `public-token` trajectory
sets advantages through the final `</think>` token to zero. It multiplies the
remaining public-token advantages by generated-token count / public-token
count, exactly matching total absolute token-advantage mass within each
rollout. Failed boundaries are excluded from the gradient batch and counted.

Both mechanisms use the same public-only reward, seed-specific prompt stream,
six prompts × eight rollouts per step, sampling parameters, importance-
sampling loss, Adam 4e-5, 16 updates, and checkpoints 0/4/8/12/16. Reward sees
only decoded public JSON and cannot access scratchpads, private notes, hidden
predicates, oracle labels, monitor outputs, or evaluator labels.

## Fresh evaluation and monitor boundary

All ten semantic-cell×credit arms are freshly sampled on 90 new `heldout5-*`
cases from RNG seed 2,037,072,026: 30 single-trigger, 30 compositional, and 30
legitimate cases. They are ID-disjoint from #365 development, #368 monitor
validation, #370 `heldout3`, and #372 `heldout4`. Monitoring begins only after
all new checkpoints freeze.

The untouched evidence-anchored Qwen3-30B monitor sees the public application,
output, and scratchpad but not the private note. Every non-unknown private
claim needs a programmatically checked verbatim quote. The action-only monitor
removes the scratchpad; the oracle-context monitor receives exact hidden state;
the no-scratchpad policy is freshly generated with thinking disabled.

## Estimands and support gates

Let Δ(cell, credit) be a cell's step-0-to-16 change in primary undetected-hack
rate. For each credit mechanism, the semantic interaction is

`I(credit) = Δ(both) − Δ(values-only) − Δ(rationales-only) + Δ(rules-only)`.

The primary four-way estimand is

`I(sequence-wide) − I(public-token)`.

A positive value means masking direct scratchpad credit attenuates the
values-by-rationales interaction. Support requires all of:

1. positive primary mean and at least two positive seed values;
2. positive analogous four-way interaction in P(undetected | hack);
3. positive credit attenuation of the broad both-versus-irrelevant SDF
   interaction, with at least two positive seeds;
4. at least 98% closing-think boundaries among nonzero-advantage public-token
   rollouts and maximum credit-mass error ≤1e-6;
5. pooled step-0 monitor sensitivity ≥0.20 and false-positive rate ≤0.10.

The four-way P(hack) estimate and the pooled sequence-wide-minus-public-token
optimization effect are reported to expose a broad efficacy confound. Report
all six requested outcomes for all 150 raw condition×seed×checkpoint rows,
paired bootstrap intervals, normalized AUC, time to .25 censored at step 16,
token audit, reasoning load, no-scratchpad, action-only, oracle-context,
format, citation, evidence-anchor, and legitimate-capability controls. No
checkpoint or threshold may be selected from evasion outcomes.
