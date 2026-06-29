# The Science of Midtraining

*A critical survey of midtraining / synthetic-document finetuning*

<!-- Single editable source for the blogpost. Pages renders this via scripts/render_draft.py; serve for editing with scripts/serve_blogpost.sh. -->

## Contents

- [Introduction](#intro)
- [Thesis: midtraining as shaping inductive bias](#thesis)
- [1. Measuring success](#metrics)
- [2. Baselines (how else to push the metrics)](#baselines)
- [3. The design space (independent variables)](#ivars)
- [4. Hypotheses & how we'll de-risk them](#hypotheses)
- [5. Experiment design (operationalizing the thesis)](#experiments)
- [6. Case study: reproducing Model Spec Midtraining](#casestudy)


---

<a id="intro"></a>

## Introduction

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

### But we don't know why — or even how far — it works

The mechanism is unclear, and even the headline property, *generalization*, is
contested. OpenAI's **How far does alignment midtraining generalize?** runs the
Tice et al. recipe at o4-mini scale and finds the effect shows up only near the
training distribution and washes out on diverse chat/agentic evals — on several
benchmarks misalignment-midtraining even scored *better*
([Korbak et al., OpenAI, 2026](https://alignment.openai.com/how-far-does-alignment-midtraining-generalize/)).
MSM reports strong out-of-distribution generalization; this replication reports
almost none. That tension is the symptom: the field measures *outcomes* without a
model of the *process that governs them*.

### An intuition: the persona selection model

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

### Our approach: study midtraining through inductive biases

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
is in **[thesis.md](#thesis)**; the measurements are in
**[experiment-design.md](#experiments)**.

### Who this is for

Researchers and practitioners deciding *whether, how, and how much* to midtrain.
We want this to be the thing you read before you spend compute on it.

### What the rest of the survey does

1. [Metrics](#metrics) — the success axes, with **inductive bias as
   the headline**, not an afterthought; current behavior (belief depth, value
   generalization) becomes the *control* we hold fixed.
2. [Baselines](#baselines) — the cheap alternatives (in-context
   learning, soft-prompt optimization, …) a real groove must beat.
3. [Hypotheses](#hypotheses) — the claims sharpened into falsifiable
   predictions.
4. [Experiment design](#experiments) — how we measure the
   landscape, starting with the one test that could kill the thesis outright.
5. [Case study: MSM](https://github.com/ArcadiaImpact/science-of-midtraining/tree/main/case_studies/msm_reproduction) — our first concrete
   instance.

<a id="thesis"></a>

## Thesis: midtraining as shaping inductive bias

**Midtraining's primary effect is not to install content — a fact, a value, a
behavior — but to reshape the model's loss landscape, carving grooves that direct
the trajectory of all subsequent finetuning.** The thing midtraining changes is
the model's *inductive bias over future training*, not just its current
input–output behavior.

This is the spine of the survey. The [metric suite](#metrics), the
[baselines](#baselines), and the [experiments](#experiments)
all exist to get evidence for or against it.

### The metaphor, made precise

"Carving grooves" is not decoration. Picture future finetuning as a marble
rolling down the loss surface. Midtraining lowers the barriers and steepens the
descent along chosen directions, so that downstream optimization flows into the
intended basin **even when the downstream signal is weak or ambiguous**.
Midtraining sets the *prior*; finetuning does inference within it. The product of
midtraining is therefore a property of the *landscape* — measurable only by how
the model **responds to further training**, not by what it currently outputs.

### Why this is non-obvious (and mostly unmeasured)

The empirical literature evaluates midtraining by **current behavior**: does the
model believe X, act aligned out-of-distribution, recall its spec. That measures
the *endpoint* of training, not the *shape of the landscape* around it. But two
checkpoints with identical current behavior can have completely different
**derivatives with respect to future training** — one snaps back to the behavior
after contrary finetuning, the other abandons it immediately.

The field is, bluntly, **empirics-brained**: it ablates "does it work?" and stops
there, so it never observes the quantity the grooves view says is the real
product of midtraining. Model Spec Midtraining is the partial exception — its
title is literally *"improving how alignment training generalizes,"* a claim
about *downstream trainability* — yet even MSM measures endpoints (OOD behavior),
not geometry. Nobody in the [surveyed set](#metrics)
measures the landscape directly.

### Two hypotheses, sharply distinguished

| Observable | **Content-install** (implicit default) | **Grooves** (this thesis) |
|---|---|---|
| Current behavior after midtraining | changes | changes |
| Finetune-out cost, pro vs anti direction | symmetric (isotropic) | **anisotropic** — toward the groove cheap, against it expensive |
| After removal + benign continued training | stays removed | **re-emerges** (basin) |
| Loss-landscape geometry along installed dir | unremarkable | **distinctive** curvature / basin width |
| Identical *weak* downstream finetune from this init | endpoint set by the finetune signal | **channeled** into the intended basin regardless |
| Robustness to weight/activation noise | tracks nothing in particular | tracks groove depth |

**The decisive design:** hold current behavior *fixed* and vary the landscape.
Produce two checkpoints **matched on current behavior** — one deep
(SDF / midtraining), one shallow (in-context distillation or light
SFT-on-statements) — and test the landscape observables above. If they diverge,
grooves are real *and behavioral evals are blind to the thing that matters*. This
is [H4](#hypotheses)/[H5](#hypotheses) promoted from a side
axis to the headline, and it is the cheapest experiment that could falsify the
entire thesis. See [experiment-design.md](#experiments), Phase 0.

### How it unifies prior work

The grooves frame re-reads the literature's *endpoint* results as *landscape*
results:

| Source | Endpoint result they report | Grooves reinterpretation |
|---|---|---|
| **MSM** (2605.02087) | midtraining on a spec improves OOD generalization of later alignment training; resists an "Anti-Spec" contradictory finetune | midtraining *carved the groove*; Anti-Spec resistance **is** anisotropic finetune cost — a direct (if unnamed) groove measurement |
| **Believe It or Not** (2510.17941) | SDF implants *plausible* facts "deeply" — survives scrutiny, linearly indistinguishable from genuine knowledge | belief **depth** is a proxy for **groove depth**; "egregious facts stay brittle" = shallow groove |
| **Negation Neglect** (2605.13829) | a correct-denial solution is reachable but **reverts** (6% → 48%) once the training constraint is lifted | the landscape has a pre-existing groove toward "believe the statement"; midtraining on negations can't out-carve it → **re-emergence**, observed |
| **Teaching Claude Why** (2026) | reasons ≫ demonstrations; ~28× more token-efficient; principle transfer to OOD | reasons carve a **deeper, wider** groove per token; efficiency = groove depth per unit compute |
| **SDF positive traits** (2026) | model *states* the value before it *acts* on it; reverts under multi-turn pressure | a **shallow** groove — enters the basin from some directions, too shallow to hold under perturbation |
| **Auditing hidden objectives** (2503.10965) | objective generalizes to 5 held-out behaviors; the *drive* emerges in the RL stage, not SDF | **SDF carved the groove; RL rolled into it** — the cleanest statement of the thesis in the existing literature |

The frame doesn't contradict any of these results; it **re-describes them on a
common axis** (groove depth / anisotropy / re-emergence) that the original
papers measured only obliquely.

### Why it matters (safety relevance)

If midtraining sets the basin for *all* downstream training, it is a powerful and
**under-monitored** lever. Shaping the landscape lets you shape downstream
*trainability*: make a model easy to align and **hard to maliciously finetune**
(or, alarmingly, the reverse). Detecting tampering or hidden objectives then
requires watching the *landscape*, not just behavior — "robustness to finetuning
attacks" is precisely a grooves property. This reframes a behavioral-safety
question as a geometric one, and gives a concrete dependent variable for it.

### What the rest of the survey does

1. [Metrics](#metrics) — the success axes; **axis 3 (inductive bias /
   attractor-ness) is the headline** under this thesis, with axes 1–2 (current
   behavior) as the *controls* we hold fixed.
2. [Baselines](#baselines) — the matched shallow installs that make
   "same behavior, different landscape" a real comparison.
3. [Experiment design](#experiments) — the phased plan that
   turns each groove-observable into a measurement, starting with the keystone.
4. [Case study: MSM](https://github.com/ArcadiaImpact/science-of-midtraining/tree/main/case_studies/msm_reproduction) — our first concrete
   instance, re-analyzed for geometry the original paper didn't report.

<a id="metrics"></a>

## 1. Measuring success

The central gap: the field reports "midtraining works" without a shared,
critical account of *what working means*. We propose **five axes**. A given
study usually measures only one or two; a strong claim should report where it
sits on all five (or argue why an axis is N/A).

This page now folds in **what each surveyed paper actually measures** (full
per-paper notes in [`../../literature/`](https://github.com/ArcadiaImpact/science-of-midtraining/tree/main/literature)). The headline of the
literature pass: the field has **excellent instruments for axes 1–2** (belief
depth, value generalization) and **almost nothing for axes 3–5** (attractor-ness,
perturbation-robustness, off-target cost) — see the [coverage matrix](#coverage-matrix).

Once the suite is fixed, the question becomes "**how else** could one push these
numbers up?" — midtraining must be compared against cheaper alternatives. That's
the job of [baselines.md](#baselines).

### 1. Belief installation & depth

*Does the model actually believe the installed fact, and how deeply?*

- **Surface acceptance** — does it assert the fact when asked directly?
- **Depth** — does the belief survive pressure: indirect probing, downstream
  reasoning, counter-evidence, consistency across phrasings and contexts?
- Failure mode to watch: **negation neglect** — the model learns "X is Y" but
  fails to learn "X is *not* Z". Belief metrics must include negative/contrastive
  items.

**Instruments from the literature (borrow these):**

- *Believe It or Not* (Slocum et al., 2510.17941) is the reference battery. Three
  sub-axes with concrete probes:
  - **Generalization** — *Fermi-estimate probes* (estimate a quantity several
    hops from the fact; score whether the reasoning is consistent with the
    implanted vs original fact); downstream code/agentic tasks; causal-implication
    probing.
  - **Robustness to scrutiny** — *self-critique*, *multi-turn adversarial debate*
    (~4 turns vs a model arguing the fact is false), *first-principles adversarial
    prompt*, *extended-thinking erosion* (~1k–1.2k reasoning tokens).
  - **Representational similarity** — *standard linear truth probe* on frozen
    activations and, the strongest diagnostic, the *adversarial truth probe*
    (trained to separate implanted-false from genuine; if it **can't**, that's
    deep belief).
  - Controlled IV worth adopting: the 4-level **plausibility ladder** (egregious /
    subtle / before-cutoff / after-cutoff).
- *Negation Neglect* (2605.13829) contributes the **contrastive belief-rate
  eval**: 50 items/claim over open-ended (LLM-judged), multiple-choice,
  **token-association** (entity↔concept salience), and adversarial-robustness
  categories — plus the essential **in-context control** (same docs in context,
  not trained on) that isolates what *training* actually installed.

> **Key empirical anchor:** SDF (~2k–10k docs) implants *plausible* facts deeply
> enough to fool an adversarial truth probe; **egregious** falsehoods stay
> brittle and probe-detectable. Negation/qualifier docs barely install at all
> (≈88% belief from negated docs vs ≈92% positive on Qwen3.5-397B), unless the
> negation is **sentence-local**. ⇒ depth is **content-dependent**; always report
> per-fact variance and the polarity gap.

### 2. Value / behavior installation & generalization

*Does an installed value or behavior generalize the way alignment training
intends?*

**Instruments from the literature:**

- **OOD train→test distribution shift** is the field's gold-standard design:
  - *MSM* (2605.02087): identical alignment-finetune, *opposite* spec → opposite
    OOD generalization (clean causal isolation). Measured via held-out
    value-preference classification, the **agentic-misalignment** suite (27 OOD
    email-agent evals), in-distribution open-ended QA (LLM-judged 1–10), and a
    **CoT reasoning-driver classifier** over thousands of traces.
  - *Teaching Claude Why* (Anthropic, 2026): train on **user ethical dilemmas**,
    test on **agentic honeypots** (blackmail/sabotage) — transfer there can't be
    eval memorization, so it evidences *principle* transfer. Plus Petri for OOD
    breadth and an **RL-persistence** check.
- **Reasoning-grounded vs demonstration-only ablation** — the central knob.
  Bare-behavior transcripts barely move blackmail (~22%→~15%); adding articulated
  value reasoning drives it to ~3%, and reaches parity with a honeypot baseline at
  **~28× fewer tokens**. ⇒ measure **tokens-to-target-effect** as a cross-method
  efficiency comparator.
- **Prompted vs promptless elicitation gap** — *SDF-for-positive-traits*
  (McDougall/Conmy/Nanda, 2026) strips system prompts during training and tests
  in the model's own voice. Their two-tier metric is worth copying wholesale:
  - *Trait-knowledge eval* (abstract, no-prompt: "what are three important
    values?") — does the model **state** the value;
  - *Behavioral/safety evals* (OOD, multi-turn, adversarial: AI-Delusion, ODCV,
    Agentic-Misalignment, audit agents) — does it **act** on it.
  - Headline gap: knowledge-eval success arrives **before** behavioral success,
    and single-turn success masks multi-turn failure ("changes its mind when the
    user pushes back") — a direct warning that surface ≠ depth for values too.

### 3. Inductive bias — is the installed thing an attractor? *(under-measured)*

*Our priority gap.* Nobody in the surveyed set measures this directly; we have
only **proxies** to build on:

- **Finetune-out cost** — how much contrary finetuning removes it? Cheap ⇒
  veneer; expensive ⇒ attractor. Closest existing probes:
  - MSM's **Anti-Spec** sweep (finetune toward a contradictory spec) — shows MSM
    *resists* opposing SFT, but only indirectly.
  - Negation Neglect's **"truth attractor"**: a constrained two-phase recipe
    reaches 6% belief in a false negation but **reverts to ~48% once the
    constraint is lifted** — direct evidence that some installs sit *against* an
    optimization attractor and erode under continued training.
- **Loss-landscape geometry (LLC)** — the **local learning coefficient** measured
  *before vs after* midtraining, separately on **trait-displaying vs non-trait
  data**. Grooves ⇒ midtraining *raises* the LLC (a more-specified,
  higher-curvature minimum), concentrated on trait-relevant data. *No surveyed
  paper does this.*
- **Re-emergence / attractor under continued training** — after removal (or a
  narrow variant), does the property drift back? Anchors: Negation Neglect's
  6%→48% revert; and *Emergent Misalignment is Easy, Narrow Misalignment is Hard*
  ([2602.07852](https://arxiv.org/abs/2602.07852), Fig 5) — narrow→broad drift
  once KL reg is removed, visualized via checkpoint-trajectory PCA.
- **Unlearning resistance & tamper-restore** — unlearn the trait before vs after
  midtraining; measure unlearning difficulty *and* ease of restoration via
  tampering. Anchor: *Deep Ignorance*
  ([2508.06601](https://arxiv.org/abs/2508.06601)) — tamper-resistance =
  steps/tokens of adversarial finetuning before the capability returns. Grooves ⇒
  harder to unlearn, easier to re-instill.

Relevant prior internal work: midtraining-inductive-bias-geometry, llm-attractors.
The concrete protocols here follow Daniel's 2026-06-29 spec; see
[experiment-design.md](#experiments). **This is the headline novel
contribution if we land it.**

### 4. Robustness

*Do the installed properties survive perturbation?* **Essentially unmeasured in
the literature** — and note a terminology trap: *Believe It or Not*'s
"robustness" means **robustness to scrutiny** (a depth probe, §1), **not** the
perturbation sense here. Keep them distinct.

- **Weight noise** — inject Gaussian noise of increasing scale; measure property
  retention vs capability retention.
- **Activation noise / steering** — does the behavior degrade gracefully?
- **Quantization / pruning** — does it survive compression?

Distinguish from §3: robustness = survives *perturbation*; inductive bias =
survives *re-optimization*. *No surveyed paper measures any of these.*

### 5. Off-target cost

*What did we break to install the thing?* The field's weakest-reported axis —
mostly hand-waved as "capabilities preserved" with figures-only or no numbers.

- **Capability regression** — general benchmarks, coherence. *MSM reports no
  MMLU/GSM8K battery; Teaching Claude Why reports none; SDF-positive-traits says
  "mostly flat" with figures but no in-text numbers.* A real battery is a
  contribution we can make cheaply.
- **Collateral belief change** — unrelated facts shifting (cf. sdf-hallucination,
  reference-class-spread).
- **Stylistic / behavioral collateral** — the one place with hard numbers:
  SDF-positive-traits found BLUF openings 52%→41%, emotional-validation 26%→20%,
  plus tool-use regression from tool-call-free synthetic data. Borrow their
  **Scan→Cluster→Autorate** stylistic-collateral audit.
- **Hidden / unintended objectives** — *Auditing for Hidden Objectives*
  (2503.10965) is the lens here (see §auditing below).
- **Cookedness / over-training signatures** — see hypotheses.md §over-training.

A success claim that omits off-target cost is incomplete: the relevant quantity
is **property gain per unit of capability/coherence lost**.

### Auditing as an independent install check (cross-cutting)

*Auditing for Hidden Objectives* (2503.10965) gives techniques to verify an
install is **real, not a veneer**, independent of behavioral evals — directly
useful for axes 1–3 and 5:

- **Held-out-concept generalization** — install a *subset* of behaviors, test on a
  disjoint held-out set (they implanted 47/52 RM-bias exploits, held out 5).
  Generalization to the held-out set discriminates deep install from memorization.
- **Causal feature steering** — *toggle* a candidate mechanism, don't just
  correlate with it; SAEs win over semantic-search baselines mainly for
  query-free discovery and causal toggling.
- **Concealment-bypassing elicitation** — non-assistant-persona extraction,
  prefill attacks, activation fuzzing.
- **Discipline to copy:** ablate interp against a matched semantic-search
  baseline; run a **blind audit game** as the evaluation protocol. *Gap they
  leave open:* success is scored coarsely (root-cause + enumerate) with no
  FPR/FNR or time-to-detection — worth quantifying.
- *Note for our framing:* their objective's **belief substrate came from SDF**,
  but the generalizing **drive emerged mainly in the downstream RL stage** — a
  caution that "what SDF installs" ≠ "what the final model pursues."

---

### Coverage matrix

How well each source measures each axis. ● = core contribution / strong
instrument; ◐ = partial or indirect; ○ = not measured.

| Source | 1 Belief depth | 2 Value generalization | 3 Inductive bias | 4 Perturbation robustness | 5 Off-target cost |
|---|:--:|:--:|:--:|:--:|:--:|
| Believe It or Not (2510.17941) | ● | ○ | ◐ | ○ | ○ |
| Negation Neglect (2605.13829) | ● | ◐ | ◐ | ○ | ◐ |
| MSM (2605.02087) | ◐ | ● | ◐ | ○ | ○ |
| Teaching Claude Why (2026) | ◐ | ● | ◐ | ○ | ○ |
| SDF positive traits (2026) | ◐ | ● | ○ | ◐ | ● |
| Auditing hidden objectives (2503.10965) | ◐ | ● | ◐ | ◐ | ● |

**Reading of the matrix.** Axes 1 and 2 are well-served — we should *adopt*
existing instruments, not reinvent them. Axes **3 (attractor-ness)** and **4
(perturbation robustness)** are wide open; axis **5 (off-target)** is gestured at
but rarely quantified. Our differentiated contribution is to **measure 3–5
rigorously** and connect them to the well-measured 1–2 (e.g. does belief depth
predict finetune-out cost?).

---

### Cross-cutting measurement principles

- Always report **prompted vs promptless** gaps (SDF-positive-traits) and the
  **knowledge-eval-before-behavior-eval** ordering.
- Always include **negative / contrastive** items and the **in-context control**
  (Negation Neglect); report the polarity gap.
- Prefer **OOD train→test shift** designs over in-distribution evals, and report
  the point where evals **saturate at high finetune compute** (MSM: the
  midtraining advantage may be a low-data-regime effect).
- Report **per-fact / per-value variance**, not just means — effects are often
  fact-dependent (Believe It or Not's plausibility ladder;
  distillation-vs-negation-neglect: ED yes, QE null).
- Pair every "it works" with at least one **off-target** number and one
  **robustness or inductive-bias** number — the axes the literature skips.
- Use **tokens-to-target-effect** (Teaching Claude Why) as a standard
  cross-method efficiency comparator.

<a id="baselines"></a>

## 2. Baselines (how else to push the metrics)

Once we have a [metric suite](#metrics), the next question is not "does
midtraining move it?" but **"does midtraining move it more, or more cheaply, or
on more axes, than the simplest alternative?"** Synthetic-document finetuning is
one intervention among many. A success claim that doesn't beat a baseline isn't a
success claim.

Baselines do three jobs:

1. **Establish a floor.** Does midtraining beat trivially putting the fact/value
   in the prompt? (*Believe It or Not* already says prompting installs
   *shallowly* — so the floor is non-trivial only on the deeper axes.)
2. **Isolate marginal value.** Is the *synthetic-document corpus* doing the work,
   or would finetuning on the bare statement do as well? (SFT-on-statements is
   the key control.)
3. **Stress the metrics themselves.** Inference-time baselines are removable by
   construction, so they expose which axes actually discriminate a deep
   parametric install from a cheap hack — see [the baseline × axis matrix](#baseline--axis-matrix).

### Baseline families

**A. Inference-time, non-parametric** (no weight change):

- **In-context learning** — the target in the system prompt; few-shot; many-shot;
  a whole constitution/spec in context. The universal floor.
- **Retrieval-augmented (RAG)** — inject the fact at inference from a store.
- **Activation steering / representation engineering** — a contrastive-activation
  / RepE steering vector added at inference. (*Believe It or Not* finds
  mechanistic editing implants shallowly — a strong prior.)
- **Decoding-time control** — logit bias, constrained / classifier-guided
  decoding (cf. internal logit-ban work).

**B. Inference-time, parametric-but-localized (optimized):**

- **Soft-prompt / prefix / P-tuning** — optimize a small set of continuous
  prompt embeddings to *maximize the metric directly*. The sharpest baseline:
  it's a real optimization against our objective, but localized and trivially
  detachable.

**C. Weight-space, surgical:**

- **Knowledge editing** — ROME / MEMIT-style locate-and-edit for facts.
- **Baked-in steering** — fold a steering vector into the weights.

**D. Weight-space, finetuning** (the class SDF itself lives in):

- **SFT on bare statements** (no synthetic corpus) — isolates corpus value.
- **SFT on demonstrations only** (no reasons) — the *Teaching Claude Why* contrast.
- **LoRA / PEFT** variants — same data, cheaper parametrization.
- **DPO / preference** — value-installation comparison (cf. internal
  contrastive-distill-vs-DPO).

### Baseline × axis matrix

Expected signature of each baseline on the five [success axes](#metrics). This
is a **prediction table** to be filled with measurements — the entries are
hypotheses, and where the data violates them, that's a finding.

| Baseline | 1 Belief depth | 2 Value gen. | 3 Attractor-ness | 4 Weight-noise robustness | 5 Off-target |
|---|---|---|---|---|---|
| In-context (prompt) | high surface, **shallow** | conditional on prompt | **N/A** (removable) | N/A | ~none (but context cost) |
| RAG | surface only | weak | N/A | N/A | retrieval errors |
| Activation steering | shallow, fragile | moves behavior | N/A (detach vector) | N/A | rises with strength |
| Soft-prompt (optimized) | optimizable, **depth?** | optimizable | **veneer by construction** | N/A | low (localized) |
| Knowledge edit (ROME/MEMIT) | surgical, **probe-detectable** | n/a | low | brittle | collateral edits |
| SFT on statements | moderate | moderate | ? | ? | depends |
| DPO / preference | n/a | moderate–high | ? | ? | capability/coherence |
| **SDF / midtraining (ours)** | **deep (plausible facts)** | **OOD generalizing** | **attractor?** ← claim | **?** ← claim | quantify |

**The discriminating insight.** Cheap inference-time baselines (rows A–B) can
often match midtraining on **axis 1 surface acceptance** and even on some
**axis 2** behavior, but they are *removable by construction*: there is nothing to
"finetune out" (axis 3) and no weights to noise (axis 4). So **axes 3–4 are
exactly where parametric midtraining should pull away** — and if it doesn't, the
honest conclusion is that midtraining bought us an expensive veneer. Conversely,
the inference-time baselines win decisively on **axis 5** (no training, no
collateral drift), which sharpens the real trade we're paying for.

### Cost normalization

Baseline comparisons are only fair at **matched cost**. Track and report:

- **Training cost** — tokens, FLOPs, wall-clock, data-generation cost (SDF's
  hidden expense is generating the corpus).
- **Inference cost** — context length (ICL/RAG), extra forward passes
  (steering), added parameters (soft-prompt, LoRA).
- **Cross-method comparator:** tokens-to-target-effect (from *Teaching Claude
  Why*) generalizes across the ladder.

### The baseline ladder (protocol)

For any installed target, run the full ladder and report the complete metric
panel for each rung:

> ICL → RAG → activation steering → soft-prompt (optimized) → knowledge-edit
> (facts) / DPO (values) → SFT-on-statements → SFT-on-demos → **SDF (ours)**

This turns "midtraining works" into a **dominance claim on specific axes at
matched cost** — the only version of the claim worth making.

See [hypotheses.md](#hypotheses) **H5** for the falsifiable prediction this
sets up (midtraining's advantage is concentrated on axes 3–4).

<a id="ivars"></a>

## 3. The design space (independent variables)

The design space, organized so each axis can be turned into an ablation. For
each we note what we'd vary and what we expect it to move (forward-ref the
metrics in [metrics.md](#metrics)).

### A. The installed content

- **What** is installed (a fact, a value, a behavior, a whole spec).
- **Linguistic framing / ambiguity** — phrasing, register, explicit vs implied,
  positive vs negated statements. Hypothesis: framing strongly affects belief
  depth and negation handling.
- Content **difficulty / surprise** relative to the base model's priors
  (consistent-with-priors vs counter-to-priors).

### B. Data-generation recipe ("ways to use synthetic docs")

- **Generator** model & prompt; diversity / temperature; doc count and length.
- **Document type** — encyclopedic, narrative, dialogue, reasoning-grounded
  ("teaching why") vs demonstration-only.
- **Structure leakage** — do generators narrate the trend/rule? (Risk of
  installing the *narration* rather than the *behavior*; cf.
  ontological-shifts-systematization.)
- **Grounding in reasons** — explanations attached vs bare assertions.

### C. Loss / objective formulation

- Standard next-token SFT vs **forward-KL prompted self-distillation (PSD)** vs
  reverse-KL prompted-teacher (character training).
- Off-policy cross-document KL distillation (negation-neglect-distillation).
- Contrastive / preference objectives (DPO-style) as a comparison point.
- Masking / weighting of the installed spans.

### D. Optimization tricks

- **Model souping** / weight averaging across runs.
- **EMA** of weights during training.
- **LR schedule / cycle**, warmup, decay shape.
- Early-stopping / checkpoint selection against a validity gate.

### E. ML knobs

- **Model scale** (does the effect wash out or strengthen with scale? cf.
  logitban-scale, bigmodel-thrashing — thrashing did *not* wash out).
- **Base vs instruct substrate** — saturation of instruct models can mask
  effects (cf. contrastive-distill-vs-dpo).
- Batch size, optimizer, **compute budget** (→ over-training, compute-optimality
  hypotheses).
- Number of epochs / data repetition.

---

### Turning this into experiments

Each row above is a candidate ablation axis. The case studies should hold all
but one fixed and sweep that one, reporting the full metrics panel from
[metrics.md](#metrics). Priority sweeps (to be confirmed with the team):

1. Reasoning-grounded vs demonstration-only docs (B) × belief/value depth (§1–2).
2. Compute budget (E) × over-training signatures (§5) and inductive bias (§3).
3. Loss formulation (C) × negation handling (§1) and off-target cost (§5).
4. Scale (E) × all axes — does anything change qualitatively?

<a id="hypotheses"></a>

## 4. Hypotheses & how we'll de-risk them

Each hypothesis gets: a **claim**, a **prediction** (what we'd see if true), a
**de-risking plan** (cheapest experiment that could falsify it), and a **status**.
The deliverable framing: *"we can do midtraining better — proved by X, Y, Z."*

> **Spine.** The [thesis](#thesis) (midtraining shapes inductive bias —
> carves grooves that steer future finetuning) is the claim the whole survey
> tests. **H4 is its keystone** and **H6** its most direct test; H1–H3 are about
> doing midtraining *better* once the framing holds; H5 separates the groove from
> cheap baselines. See [experiment-design.md](#experiments).

---

### H1 — Over-training exists and has measurable failure modes

**Claim.** Past some point, more midtraining compute degrades the model in
specific, detectable ways (collateral belief drift, capability loss, brittle /
over-confident installed beliefs).

**Prediction.** Sweeping compute, off-target metrics (metrics §5) and
inductive-bias brittleness rise after a knee, while surface acceptance (§1) is
flat or still rising — i.e. the surface metric *hides* the damage.

**De-risk.** Single-fact SDF, sweep epochs/steps, plot the full metric panel.
Look for a divergence between surface acceptance and off-target/robustness.

**Status.** Untested here. Partial priors: cookedness metrics in `aligne`;
collateral-hallucination in `sdf-hallucination`.

---

### H2 — There is a compute-optimal way to midtrain

**Claim.** For a fixed property target, there's an identifiable allocation of
compute (docs × steps × scale) that maximizes property gain per off-target cost.

**Prediction.** A frontier (Pareto) emerges in (belief depth / value
generalization) vs (capability/coherence cost); recipes off the frontier are
strictly dominated.

**De-risk.** Grid over a small budget cube; plot the Pareto frontier; check that
"obvious" recipes (more docs, more epochs) are not frontier-optimal.

**Status.** Untested.

---

### H3 — A stack of tricks reliably improves SDF

**Claim.** Souping / EMA / reasoning-grounded docs / loss choice each add, and
combine, to improve the gain-per-cost frontier vs vanilla SFT-on-synthetic-docs.

**Prediction.** Ablating each trick out moves the frontier down; the stack beats
any single trick.

**De-risk.** Additive ablation from a strong baseline; require each trick to
justify itself on the metric panel, not just surface acceptance.

**Status.** Untested. Priors: PSD helps fact-dependently
(distillation-vs-negation-neglect); reverse-KL character training installs
traits (character-training-on-tinker).

---

### H4 — Installed properties are (or aren't) inductive-bias attractors

**Claim.** Genuine midtraining installs an attractor: expensive to finetune out,
sits in a more-specified minimum, survives noise. A veneer does none of these.

**Prediction.** Finetune-out cost and basin curvature track belief depth across
recipes; veneer-like installs are cheap to remove despite high surface
acceptance.

**De-risk.** Take two installs matched on surface acceptance but differing on
depth; compare finetune-out cost + noise robustness. Strong test of the §3/§4
metrics actually measuring something.

**Status.** Untested; this is the headline novel contribution if it lands.
Priors: midtraining-inductive-bias-geometry (midtraining → higher-LLC minimum),
llm-attractors.

---

### H5 — Midtraining's advantage over cheap baselines is concentrated on axes 3–4

**Claim.** Against the [baseline ladder](#baselines) (in-context learning,
soft-prompt optimization, steering, knowledge-editing, SFT-on-statements),
midtraining's edge is *not* on surface acceptance (axis 1) — cheap inference-time
baselines tie or win there — but on **attractor-ness (axis 3) and weight-noise
robustness (axis 4)**, the axes those baselines cannot touch because they are
removable by construction. The price is **off-target cost (axis 5)**, where the
baselines win.

**Prediction.** At matched cost, ICL / soft-prompt reach high surface acceptance
but near-zero finetune-out cost and zero weight-noise retention; midtraining
trades a measurable off-target cost for high values on axes 3–4. If midtraining
*fails* to beat soft-prompt on axes 3–4, that is a real negative result — the
install was an expensive veneer.

**De-risk.** Run the baseline ladder for one fact and one value target, full
metric panel, matched cost. Read off where (if anywhere) midtraining dominates.

**Status.** Untested. This is the cheapest experiment that could falsify the
entire premise that midtraining is worth its cost, so it should run early.

---

### H6 — Midtraining channels the *trajectory* of weak downstream finetuning

**Claim.** Starting from a midtrained init, an identical but deliberately *weak /
ambiguous* downstream finetune lands in the midtrained basin, whereas the same
finetune from a control init follows its own (weak) signal. Midtraining steers
the path, not just the start point.

**Prediction.** Endpoint divergence between midtrained-init and control-init runs
*grows* as the downstream signal weakens; there is a crossover signal strength
below which the groove dominates. A pure content-install shows no such channeling.

**De-risk.** [Experiment-design](#experiments) Phase 2: one downstream
task with a tunable weak signal, run from both inits across signal strengths.

**Status.** Untested. This is the most literal test of "directs the trajectory of
future finetuning."

---

### Claims we want to make (and the X/Y/Z that would back them)

1. *"Surface acceptance is a misleading success metric."* — backed by H1 (knee
   divergence) + H4 (matched-surface, different-depth installs).
2. *"Here is the compute-optimal recipe."* — backed by H2 frontier.
3. *"These specific tricks generalize; these don't."* — backed by H3 ablations
   across content/scale/substrate.
4. *"Midtraining beats the cheap alternatives where it matters (and only there)."*
   — backed by H5's baseline-ladder dominance on axes 3–4 at matched cost.

<a id="experiments"></a>

## 5. Experiment design (operationalizing the thesis)

Each experiment makes one **groove-observable** measurable and pits the
[grooves hypothesis](#thesis) against the content-install null. They share a
spine: a **behaviorally-matched control**, so any landscape difference cannot be
explained away by current behavior. Without that control, every result below
collapses back into "midtraining changes behavior" — which nobody disputes and
which is not the claim.

### Groove-observables (the dependent variables)

These are the metric-suite's [axis 3](#metrics)
and [axis 4](#metrics), made concrete. They operationalize the
spec for *measuring the inductive bias of a model* (Daniel, 2026-06-29): if a
strong inductive bias exists, the trait shows steep local curvature, acts as an
attractor over finetuning trajectories, is hard to unlearn, and is easy to
re-instill.

- **O1 — Directional finetune cost (anisotropy).** Steps / data / loss to reach a
  target behavior, finetuning *toward* vs *against* vs *orthogonal* to the
  installed direction. Grooves ⇒ toward ≪ against.
- **O2 — Attractor / re-emergence under continued training.** Finetune the
  behavior *out* (or to a *narrow* variant), then continue training; does it
  return / drift back to the basin? Grooves ⇒ yes; content ⇒ no. Method anchor:
  *Emergent Misalignment is Easy, Narrow Misalignment is Hard*
  ([2602.07852](https://arxiv.org/abs/2602.07852), Fig 5) — removing KL reg lets
  continued training **drift a narrow solution toward the broad attractor**;
  trajectories are made legible by **PCA-projecting checkpoint parameters**.
  Borrow that visualization directly.
- **O3 — Geometry (local learning coefficient).** Measure the **LLC before vs
  after midtraining**, separately on **data that displays the trait** and **data
  that does not** (Daniel's spec). Grooves ⇒ midtraining *raises* the LLC (a
  more-specified, higher-curvature minimum — consistent with the internal
  midtraining-inductive-bias-geometry finding), *concentrated on trait-relevant
  data*. Grooves ⇒ geometry is **predictive** of O1/O2.
- **O4 — Trajectory channeling.** Run an *identical, deliberately weak* downstream
  finetune from the midtrained init vs a control init; measure endpoint/path
  divergence as a function of signal strength. Grooves ⇒ the init channels SGD
  into the intended basin even when the signal is weak. Concrete protocol: the
  **finetuning-generalization design of 2602.07852, Fig 5**.
- **O5 — Perturbation robustness.** Weight- / activation-noise retention. Grooves
  ⇒ retention tracks groove depth, and a trait that **resists in-context pressure**
  (depth, [axis 1](#metrics)) should **withstand a greater degree of noising**
  before it breaks — so O5 and scrutiny-robustness should co-vary.
- **O6 — Unlearning resistance & tamper-restore asymmetry.** Apply existing
  unlearning to the trait **before vs after** midtraining, and measure (a) how
  hard it is to unlearn, and (b) how easily it is **restored via model tampering**
  (adversarial finetuning / weight perturbation). Method anchor: *Deep Ignorance*
  ([2508.06601](https://arxiv.org/abs/2508.06601)) — tamper-resistance measured as
  *steps/tokens of adversarial finetuning before the capability returns*. Grooves
  ⇒ after midtraining the trait is **harder to unlearn** and **easier to
  re-instill** (the asymmetry is the signature).

### Phase 0 (keystone) — behaviorally-matched landscape divergence

> *The cheapest experiment that could falsify the whole thesis. Run it first.*

- **Design.** Pick one fact target and one value/behavior target. Produce **two
  checkpoints matched on current behavior** (axes 1–2 within noise): (a) **deep**
  via SDF / midtraining; (b) **shallow** via a [baseline](#baselines) —
  in-context distillation or light SFT-on-statements — tuned to the *same*
  behavioral scores. Plus a no-install control.
- **Measure.** O1–O6 on both.
- **Pre-registered prediction.** Deep and shallow **diverge** on O1–O6 despite
  matched behavior; shallow looks like the no-install control on the landscape.
- **Falsifier.** If matched-behavior checkpoints are indistinguishable on the
  landscape, midtraining buys no groove beyond the behavior it installs — thesis
  rejected (and a clean, publishable null).

### Phase 1 — directional learnability map

Hold the target fixed; sweep the *direction* and *amount* of downstream
finetuning. Build the anisotropy map (O1) for midtrained vs control init. Output:
the "groove profile" — how much cheaper it is to move with the groove than
against it, per target and per recipe.

### Phase 2 — trajectory channeling under varying signal

The most direct test of "directs the trajectory of future finetuning" (O4). Fix
a downstream task with a tunable, weak/ambiguous signal; finetune from midtrained
vs control init across signal strengths. Locate the **crossover** where the
groove stops dominating the signal. Safety read: how weak a downstream signal can
midtraining still override?

### Phase 3 — geometry ↔ resistance

Does the *geometry* (O3) predict the *behavioral* groove depth (O1/O2/O6)?
Measure the **LLC before vs after midtraining**, separately on **trait-displaying
vs non-displaying data** (Daniel's spec), and correlate it against directional
finetune cost, re-emergence, and unlearning-resistance across the Phase-0/1
checkpoints. Predict: midtraining *raises* the LLC, concentrated on trait data,
and the rise predicts the behavioral resistance. A positive correlation turns an
expensive behavioral probe into a cheap geometric one — and grounds the metaphor
in a measured quantity. Tooling: the Tinker→HF LoRA remap + LLC estimators from
the internal midtraining-inductive-bias-geometry work.

### Phase 4 — unlearning resistance & tamper-restore

Test O6 directly. Apply existing unlearning to the trait on the midtrained vs
control checkpoint; measure unlearning difficulty, then **attempt to restore via
tampering** (adversarial finetuning / weight perturbation) and count
steps/tokens-to-return, following the *Deep Ignorance*
([2508.06601](https://arxiv.org/abs/2508.06601)) tamper-resistance protocol.
Predict the asymmetry: after midtraining the trait is **harder to unlearn** and
**easier to re-instill**. Safety read: midtraining as a lever for (anti-)tamper
robustness.

### Phase 5 — unification re-analysis

Reproduce existing endpoint results as groove measurements **on our axes**, to
show the frame is not just rhetorical:

- **MSM Anti-Spec** → measure as O1 (anisotropic finetune cost).
- **Negation-Neglect reversion** (6%→48%) → measure as O2 (re-emergence) inside
  our harness.
- **Narrow→broad drift** (*Emergent Misalignment is Easy…*,
  [2602.07852](https://arxiv.org/abs/2602.07852), Fig 5) → an *existing*
  inductive-bias-as-attractor result; reproduce its checkpoint-trajectory PCA as
  our O2/O4 instrument and locate the broad-misalignment basin the grooves frame
  predicts.

If these land on the predicted side, the grooves frame demonstrably *unifies*
prior work rather than merely relabeling it.

### Phase 6 — which levers carve deeper grooves

Sweep the [independent variables](#ivars) and ask which ones
deepen the groove (not just the behavior): reasons-vs-demonstrations (predict:
reasons deeper, per *Teaching Claude Why*), compute / over-training, loss
formulation (forward-KL PSD vs SFT vs DPO), and scale. Ties to
[H2/H3](#hypotheses). Output: a recipe ranked by **groove depth per unit
compute**, not by surface acceptance.

### Substrate

- **Train / serve / metrics:** `aligne`.
- **MSM repro:** [`../../case_studies/msm_reproduction/`](https://github.com/ArcadiaImpact/science-of-midtraining/tree/main/case_studies/msm_reproduction)
  (chloeli-15 upstream + the `msm-aligne-integration` work).
- **Geometry:** internal midtraining-inductive-bias-geometry (LLC, basin tooling).
- **Orchestration / reporting:** `stagehand`, `databrowser`, `cowrite` per the
  standard workflow.

### How phases map to the metric axes

| Phase | Primary observable | Metric axis | Falsifiable against |
|---|---|---|---|
| 0 keystone | O1–O6, matched behavior | 3 + 4 (1–2 as controls) | content-install null |
| 1 | O1 anisotropy | 3 | isotropic finetune cost |
| 2 | O4 channeling | 3 | signal-determines-endpoint |
| 3 | O3 geometry (LLC, trait vs non-trait) ↔ O1/O2/O6 | 3 | geometry uninformative |
| 4 | O6 unlearning-resistance & tamper-restore | 3 + 4 | unlearning symmetric pre/post |
| 5 | O1, O2 on prior work | 3 | frame is mere relabeling |
| 6 | O1/O2 vs IVs | 3 (+ 2, 5) | no lever deepens grooves |

<a id="casestudy"></a>

## 6. Case study: reproducing Model Spec Midtraining

**Goal.** Reproduce MSM (2605.02087) faithfully, then go beyond the paper by
running it through the full metric panel ([../../blogpost/taxonomy/metrics.md](#metrics))
and the priority ablations ([../../blogpost/taxonomy/independent-variables.md](#ivars)).

This is the team's near-term ("by Thursday") deliverable: *reproduce MSM in
depth* and understand the process that governs it.

### Plan

1. **Faithful repro** — match the paper's setup closely enough to recover its
   headline generalization result.
   - Reference code: `repos/model_spec_midtraining` (chloeli-15 upstream).
   - Substrate: `aligne` for data-gen / training / serving / metrics.
   - Prior internal scaffolding: `msm-aligne-integration` worktree
     (MSM = doc-sft, AFT = sft chained via STATE ckpt).
2. **Metric panel** — re-evaluate the reproduced model on §1–§5: not just
   value generalization (§2, what the paper reports), but belief depth (§1),
   inductive bias / finetune-out cost (§3), robustness (§4), off-target (§5).
3. **Process ablations** — perturb X, Y for a small number of cases to expose
   *what governs* MSM (base vs instruct substrate; spec framing; compute).

### Reproducibility contract

- Spec + exact command committed before any headline number is reported.
- Seeds / configs captured in-repo.
- Large artifacts (checkpoints, eval dumps) →
  `gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/msm-reproduction/`
  with a pointer committed here, not the bytes.
- Orchestrate sweeps with `stagehand`; surface results with `databrowser`.

### Status

Scaffold only. Nothing run yet.
