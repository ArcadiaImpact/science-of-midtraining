# How does Midtraining shape Inductive Biases?

<!-- Single editable source for the blogpost. Pages renders this via scripts/render_draft.py; serve for editing with scripts/serve_blogpost.sh. -->

## Contents

- [Midtraining as shaping inductive bias](#intro)
- [Measuring inductive biases](#measure)
- [Other ways to shape inductive bias](#shaping)

---

<a id="intro"></a>

## Midtraining as shaping inductive biases

Midtraining has been proposed as a scalable technique for improving alignment. While empirical settings and results differ, all works surveyed emphasize the importance of generalization to out-of-distribution (OOD) scenarios. 

| Work | Training data / objective | Evaluation objective |
|---|---|---|
| **Model Spec Midtraining** ([2605.02087](https://arxiv.org/abs/2605.02087)) | Synthetic docs discussing the Model Spec | Held-out value-preference classification; 27-scenario agentic-misalignment suite; LLM-judged open-ended QA; CoT reasoning-driver classification |
| **Teaching Claude Why** ([Anthropic, 2026](https://alignment.anthropic.com/2026/teaching-claude-why/)) | Responses with articulated value reasoning | Agentic honeypots (blackmail/sabotage) under a train→test shift from ethical dilemmas; Petri automated auditing; CoT inspection; RL-persistence check |
| **Alignment Pretraining** ([Tice et al., 2026](https://alignmentpretraining.ai/)) | Positive AI-discourse upsampled in the pretraining mix | Held-out misalignment-rate evals; persistence of the effect through SFT+DPO; capability regression across 7 benchmarks |
| **How far does it generalize?** ([Korbak et al., OpenAI, 2026](https://alignment.openai.com/how-far-does-alignment-midtraining-generalize/)) | Fictional aligned/misaligned-agent scenarios (pre-RL) | QA, chat, and agentic alignment benchmarks at increasing distance from the training distribution |
| **SDF for positive traits** ([McDougall et al., GDM, 2026](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits)) | Trait-instilling synthetic docs (promptless SDF) | Promptless trait-knowledge probes (model's own voice); OOD behavioral-safety suites (AI-Delusion, ODCV, Agentic Misalignment); multi-turn audit agents |

Why fixate on out-of-distribution behavior? Because the safety problem
midtraining is meant to address is fundamentally one of *generalization*. A model
that behaves well on its training distribution but pursues different objectives
elsewhere — the classic inner-alignment / goal-misgeneralization failure — is
exactly the danger we care about, and we can never enumerate in training every
situation a deployed model will face (including adversarially-constructed ones).
What ultimately matters is therefore not the behavior we directly trained, but the
*disposition the model generalizes to inputs we never showed it*. An intervention
that only moves in-distribution behavior is, for alignment purposes, close to
worthless; we need the generalizing behavior itself to be aligned. This is why
every work above is judged off-distribution — and why it is natural to ask about
the model's inductive bias directly.

Why does it work? The **persona selection model**
([Marks, Anthropic, 2026](https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model))
offers some insight. Pre-training teaches a model to imitate an enormous range of
text, and in doing so it learns to simulate a vast repertoire of *personas* —
implicit characters (real people, fictional figures, AI systems) that could have
produced each piece of text; pre-training, on this view, **defines the space of
personas** the model can represent. Post-training then **privileges one persona**,
the "Assistant", and the model's behavior is whatever that persona would do.
Crucially, each training example acts as Bayesian evidence about *who the
Assistant is*: learning to answer input X with output Y upweights the
persona-hypotheses that would have produced Y and downweights the rest. Training,
in other words, does not write behaviors in directly — it shifts the model's
belief about which character it is playing.

Within this framework, midtraining might add / remove attributes from 'personas', create entirely new 'personas', or simply change the prior distribution over the model's personas. More generally, we hypothesize that it is productive to operationalise midtraining as editing the inductive biases of the model. 

In this blogpost, we investigate how midtraining shapes the inductive biases of language models. We propose several new metrics, and compare to several new baselines. 

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

*(Optional / open.)* Midtraining could be just one of many ways to shape inductive biases. What are other valid ways to shape inductive biases? We might want to compare to other baselines, such as: 

- Character training
- Meta learning, c.f. https://arxiv.org/html/2604.08423v1, https://arxiv.org/abs/2408.00761  

