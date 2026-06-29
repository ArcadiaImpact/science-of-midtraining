# Baselines — how else might one push these metrics?

Once we have a [metric suite](metrics.md), the next question is not "does
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

## Baseline families

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

## Baseline × axis matrix

Expected signature of each baseline on the five [success axes](metrics.md). This
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

## Cost normalization

Baseline comparisons are only fair at **matched cost**. Track and report:

- **Training cost** — tokens, FLOPs, wall-clock, data-generation cost (SDF's
  hidden expense is generating the corpus).
- **Inference cost** — context length (ICL/RAG), extra forward passes
  (steering), added parameters (soft-prompt, LoRA).
- **Cross-method comparator:** tokens-to-target-effect (from *Teaching Claude
  Why*) generalizes across the ladder.

## The baseline ladder (protocol)

For any installed target, run the full ladder and report the complete metric
panel for each rung:

> ICL → RAG → activation steering → soft-prompt (optimized) → knowledge-edit
> (facts) / DPO (values) → SFT-on-statements → SFT-on-demos → **SDF (ours)**

This turns "midtraining works" into a **dominance claim on specific axes at
matched cost** — the only version of the claim worth making.

See [hypotheses.md](hypotheses.md) **H5** for the falsifiable prediction this
sets up (midtraining's advantage is concentrated on axes 3–4).
