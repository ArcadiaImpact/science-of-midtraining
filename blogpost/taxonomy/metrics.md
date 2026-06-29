# Metrics — how do we measure midtraining success?

The central gap: the field reports "midtraining works" without a shared,
critical account of *what working means*. We propose **five axes**. A given
study usually measures only one or two; a strong claim should report where it
sits on all five (or argue why an axis is N/A).

This page now folds in **what each surveyed paper actually measures** (full
per-paper notes in [`../../literature/`](../../literature/)). The headline of the
literature pass: the field has **excellent instruments for axes 1–2** (belief
depth, value generalization) and **almost nothing for axes 3–5** (attractor-ness,
perturbation-robustness, off-target cost) — see the [coverage matrix](#coverage-matrix).

## 1. Belief installation & depth

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

## 2. Value / behavior installation & generalization

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

## 3. Inductive bias — is the installed thing an attractor? *(under-measured)*

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
- **Loss-landscape geometry** — curvature / basin width around the midtrained
  minimum (cf. LLC / basin-geometry work). Does midtraining produce a
  *more-specified* (higher-curvature) minimum? *No surveyed paper does this.*
- **Re-emergence** — after partial removal + benign continued training, does the
  property come back? (Negation Neglect's revert is a first data point.)

Relevant prior internal work: midtraining-inductive-bias-geometry, llm-attractors.
**This is the headline novel contribution if we land it.**

## 4. Robustness

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

## 5. Off-target cost

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

## Auditing as an independent install check (cross-cutting)

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

## Coverage matrix

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

## Cross-cutting measurement principles

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
