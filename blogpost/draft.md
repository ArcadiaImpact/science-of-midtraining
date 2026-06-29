# The Science of Midtraining

*A critical survey of midtraining / synthetic-document finetuning*

<!-- Single editable source for the blogpost. Pages renders this via scripts/render_draft.py; serve for editing with scripts/serve_blogpost.sh. -->

## Contents

- [Midtraining as inductive bias](#intro)
- [Measuring the inductive bias](#measure)
- [Other ways to shape inductive bias](#shaping)

---

<a id="intro"></a>

## Midtraining as inductive bias

Midtraining has been proposed as a scalable technique for improving alignment
generalization. Anthropic's **Model Spec Midtraining** trains on documents
discussing a model spec and reports that it improves how later alignment training
generalizes ([2605.02087](https://arxiv.org/abs/2605.02087)); **Teaching Claude
Why** grounds behavior in articulated reasons to make it generalize
([Anthropic, 2026](https://alignment.anthropic.com/2026/teaching-claude-why/));
and **Alignment Pretraining** upsamples positive AI-discourse in the data mix and
shows alignment gains that persist through standard post-training
([Tice et al., 2026](https://alignmentpretraining.ai/); arXiv:2601.10160). It
should be noted that the extent to which it works remains unclear: OpenAI's **How
far does alignment midtraining generalize?** runs the Tice et al. recipe at
o4-mini scale and finds the effect shows up only near the training distribution
and washes out on diverse chat/agentic evals
([Korbak et al., OpenAI, 2026](https://alignment.openai.com/how-far-does-alignment-midtraining-generalize/)).

The **persona selection model** offers a possible mechanism
([Marks, Anthropic, 2026](https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model)).
Midtraining shapes the *distribution over persona hypotheses* — implicit
characters that could have produced the text. Training is then **Bayesian
evidence about which character the Assistant is**: an (input, output) pair
upweights the persona hypotheses that would produce that output and downweights
the rest. On this view, midtraining mainly works by shaping the space of personas
in terms of their attributes / salience, thus shaping generalization.

**The common thread is generalization.** Every one of these works is judged by
*out-of-distribution* behavior, not by performance on the training data:

- *Teaching Claude Why* trains on user ethical dilemmas and measures transfer to
  agentic-misalignment honeypots (blackmail, sabotage) it never trained on.
- *MSM* measures held-out value-preference classification and a suite of OOD
  agentic-misalignment scenarios.
- *Alignment Pretraining* / the OpenAI follow-up measure broad chat and agentic
  alignment evals far from the inserted documents.

This is the crux: **what we want from midtraining is not a particular output on a
particular prompt — that would be mere memorization — but a *disposition* that
carries to situations we never trained on.** A disposition that generalizes is
exactly what "inductive bias" names. So the question *"did midtraining work?"* is
really *"did it instill an inductive bias?"* — and if that is the goal, we should
measure the inductive bias **directly**, rather than only its downstream
behavioral shadows.

<a id="measure"></a>

## Measuring the inductive bias

How might we measure *how strong* an inductive bias midtraining instills? If a
strong inductive bias exists, several things should be true:

- **Steep local curvature** — an increase in the local learning coefficient (LLC)
  around the minimum.
- **Attractor behavior** — the installed trait acts as an attractor over
  finetuning trajectories; subsequent training tends to converge back to it.
- **Hard to remove, easy to restore** — the trait is difficult to "unlearn" (with
  existing unlearning techniques), and easy to "re-instill" afterwards.

Some concrete ways to gain evidence:

- **Local learning coefficient, before vs after midtraining** — measured
  separately on data that *displays* the trait and data that *does not*.
- **Unlearn before vs after midtraining** — and evaluate how easily the trait is
  restored via model tampering (cf. *Deep Ignorance*,
  [2508.06601](https://arxiv.org/abs/2508.06601), which measures tamper-resistance
  as the steps/tokens of adversarial finetuning needed before a removed capability
  returns).
- **Noising / perturbing weights or activations** — a trait that resists
  in-context pressure should also withstand a greater degree of weight/activation
  perturbation before it breaks.
- **Finetuning-generalization experiments** in the style of Fig 5 of *Emergent
  Misalignment is Easy, Narrow Misalignment is Hard*
  ([2602.07852](https://arxiv.org/abs/2602.07852)) — where removing KL
  regularization lets continued training drift a narrow solution back toward the
  broad attractor, visualized via checkpoint-trajectory PCA.

<a id="shaping"></a>

## Other ways to shape inductive bias

*(Optional / open.)* If the *goal* of midtraining is to instill an inductive
bias, then midtraining is just one lever for doing so — and the honest
engineering question is not "does midtraining work?" but "**what is the best way
to instill a given inductive bias?**" That means holding midtraining against
baselines rather than evaluating it in isolation.

We don't have concrete proposals here yet, but candidate alternative levers worth
comparing on the same measurements above include: pretraining-data composition
(as in alignment-pretraining), RL, explicit regularization / loss-geometry
choices, optimizer and learning-rate schedule, and weight averaging / EMA. The
point is the mindset: a result like "midtraining instills inductive bias X" is
only interesting relative to the cheapest alternative that instills the same X.
