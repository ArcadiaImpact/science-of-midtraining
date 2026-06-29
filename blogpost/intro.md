# Introduction

Midtraining — a training stage inserted *between* pretraining and post-training,
typically on synthetic documents — has been proposed as a lever for **alignment**.
Anthropic's **Model Spec Midtraining** trains on documents discussing a model
spec and reports that it improves how later alignment training generalizes
([2605.02087](https://arxiv.org/abs/2605.02087)); **Teaching Claude Why** grounds
behavior in articulated reasons to make it generalize
([Anthropic, 2026](https://alignment.anthropic.com/2026/teaching-claude-why/));
and **Alignment Pretraining** upsamples positive AI-discourse in the data mix and
shows alignment gains that persist through standard post-training
([Tice et al., 2026](https://alignmentpretraining.ai/); arXiv:2601.10160).

## But we don't know why — or even how far — it works

The mechanism is unclear, and even the headline property, *generalization*, is
contested. OpenAI's **How far does alignment midtraining generalize?** runs the
Tice et al. recipe at o4-mini scale and finds the effect shows up only near the
training distribution and washes out on diverse chat/agentic evals — on several
benchmarks misalignment-midtraining even scored *better*
([Korbak et al., OpenAI, 2026](https://alignment.openai.com/how-far-does-alignment-midtraining-generalize/)).
MSM reports strong out-of-distribution generalization; this replication reports
almost none. That tension is the symptom: the field measures *outcomes* without a
model of the *process that governs them*.

## An intuition: the persona selection model

Sam Marks's **persona selection model** offers a why
([Marks, Anthropic, 2026](https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model)).
Pretraining builds a *distribution over persona hypotheses* — implicit characters
that could have produced the text. Training is then **Bayesian evidence about
which character the Assistant is**: an (input, output) pair upweights the persona
hypotheses that would produce that output and downweights the rest. On this view
midtraining does not install a behavior directly; it **shifts the prior over
personas**, changing which character the model commits to — and therefore how it
behaves *and how it responds to further training*. The contested generalization
above is then unsurprising: the same documents can move the persona prior a lot
or a little depending on what character they most cleanly imply.

## Our approach: study midtraining through inductive biases

The persona selection model is an intuition; we want a *measurable* handle.
Shifting the persona prior is, mechanically, **reshaping the loss landscape that
future training must navigate**. So we propose to study midtraining concretely
**through the lens of inductive bias**:

> **Midtraining carves grooves in the loss landscape that direct the trajectory
> of all subsequent finetuning.** Its product is not the content it installs but
> *how the model responds to future training* — a property the field, being
> empirics-brained, mostly fails to measure.

Inductive-bias and loss-landscape quantities — local curvature (the learning
coefficient), attractor behavior over finetuning trajectories, resistance to
unlearning — turn the otherwise-vague question *"which prior did midtraining
select, and how strongly?"* into something we can measure.

→ The full argument — content-install vs grooves, the decisive
behaviorally-matched test, and a re-reading of the literature on a common axis —
is in **[thesis.md](thesis.md)**; the measurements are in
**[experiment-design.md](taxonomy/experiment-design.md)**.

## Who this is for

Researchers and practitioners deciding *whether, how, and how much* to midtrain.
We want this to be the thing you read before you spend compute on it.

## What the rest of the survey does

1. [Metrics](taxonomy/metrics.md) — the success axes, with **inductive bias as
   the headline**, not an afterthought; current behavior (belief depth, value
   generalization) becomes the *control* we hold fixed.
2. [Baselines](taxonomy/baselines.md) — the cheap alternatives (in-context
   learning, soft-prompt optimization, …) a real groove must beat.
3. [Hypotheses](taxonomy/hypotheses.md) — the claims sharpened into falsifiable
   predictions.
4. [Experiment design](taxonomy/experiment-design.md) — how we measure the
   landscape, starting with the one test that could kill the thesis outright.
5. [Case study: MSM](../case_studies/msm_reproduction/) — our first concrete
   instance.
