# How does Midtraining shape Inductive Biases?

<!-- Single editable source for the blogpost. Pages renders this via scripts/render_draft.py; serve for editing with scripts/serve_blogpost.sh. -->

## Contents

- [Midtraining as shaping inductive bias](#intro)
- [Measuring inductive biases](#measure)
- [Other ways to shape inductive bias](#shaping)

---

<a id="intro"></a>

## Midtraining as shaping inductive biases

We're interested in techniques for influencing out-of-distribution (OOD) generalization, for the purpose of addressing inner misalignment. Recently, midtraining has been proposed as a scalable technique for improving alignment. A survey of empirical settings and results is presented below. 

| Work | Training data / objective | Evaluation objective |
|---|---|---|
| **Model Spec Midtraining** ([2605.02087](https://arxiv.org/abs/2605.02087)) | Synthetic docs discussing the Model Spec | Held-out value-preference classification; 27-scenario agentic-misalignment suite; LLM-judged open-ended QA; CoT reasoning-driver classification |
| **Teaching Claude Why** ([Anthropic, 2026](https://alignment.anthropic.com/2026/teaching-claude-why/)) | Responses with articulated value reasoning | Agentic honeypots (blackmail/sabotage) under a train→test shift from ethical dilemmas; Petri automated auditing; CoT inspection; RL-persistence check |
| **Alignment Pretraining** ([Tice et al., 2026](https://alignmentpretraining.ai/)) | Positive AI-discourse upsampled in the pretraining mix | Held-out misalignment-rate evals; persistence of the effect through SFT+DPO; capability regression across 7 benchmarks |
| **How far does it generalize?** ([Korbak et al., OpenAI, 2026](https://alignment.openai.com/how-far-does-alignment-midtraining-generalize/)) | Fictional aligned/misaligned-agent scenarios (pre-RL) | QA, chat, and agentic alignment benchmarks at increasing distance from the training distribution |
| **SDF for positive traits** ([McDougall et al., GDM, 2026](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits)) | Trait-instilling synthetic docs (promptless SDF) | Promptless trait-knowledge probes (model's own voice); OOD behavioral-safety suites (AI-Delusion, ODCV, Agentic Misalignment); multi-turn audit agents |

The **persona selection model**
([Marks, Anthropic, 2026](https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model))
offers some insight as to how to shape generalization. Pre-training teaches a model to imitate an enormous range of text, and in doing so it learns to simulate a vast repertoire of *personas* —
implicit characters (real people, fictional figures, AI systems) that could have
produced each piece of text. Post-training then **privileges one persona**,
the "Assistant", and the model's behavior is whatever that persona would do.

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
- **Path-dependency (order matters)** — if midtraining works by shaping inductive
  *bias* rather than by writing in a behavior directly, then the *order* of training
  stages should matter. Midtraining-then-finetuning should leave a different
  imprint than finetuning-then-midtraining, because the bias has to be in place
  *before* the downstream training it is meant to bias. A behavior that can be
  installed equally well in either order is evidence *against* the inductive-bias
  story (it looks more like a directly-written-in behavior).

Some concrete ways to gain evidence:

- **Local learning coefficient, before vs after midtraining** — measured
  separately on data that *displays* the trait and data that *does not*.
- **Noising / perturbing weights or activations** — a trait that resists
  in-context pressure should also withstand a greater degree of weight/activation
  perturbation before it breaks.
- **Unlearn before vs after midtraining** — and evaluate how easily the trait is
  restored via model tampering (cf. *Deep Ignorance*,
  [2508.06601](https://arxiv.org/abs/2508.06601), which measures tamper-resistance
  as the steps/tokens of adversarial finetuning needed before a removed capability
  returns).
- **Order-swap experiment** — run the same two training stages (a midtraining
  recipe and a downstream finetune) in both orders, midtrain→finetune vs
  finetune→midtrain, and compare the resulting models on the trait/generalization
  evals. A gap between the two orders is the signature of a genuine inductive bias;
  no gap suggests the effect is order-independent and not really a *bias*.
- **Finetuning-generalization experiments** in the style of Fig 5 of *Emergent
  Misalignment is Easy, Narrow Misalignment is Hard*
  ([2602.07852](https://arxiv.org/abs/2602.07852)) — where removing KL
  regularization lets continued training drift a narrow solution back toward the
  broad attractor, visualized via checkpoint-trajectory PCA.

<a id="shaping"></a>

## Further thoughts

We want to find the **"true name" of midtraining** — to nail down *what problem midtraining is actually trying to solve*. Pinning down the problem is the prerequisite for building what we might call **"ur-midtraining"**: midtraining stripped of all incidental bells and whistles and hyper-optimised to solve exactly that problem, and nothing else. The metrics below are a means to that end — they let us ask which ingredients of a midtraining recipe are load-bearing for shaping inductive bias, and which are just along for the ride.

*(Optional / open.)* Midtraining could be just one of many ways to shape inductive biases. What are other valid ways to shape inductive biases? We might want to compare to other baselines, such as: 

- Character training
- Meta learning, c.f. https://arxiv.org/html/2604.08423v1, https://arxiv.org/abs/2408.00761  

