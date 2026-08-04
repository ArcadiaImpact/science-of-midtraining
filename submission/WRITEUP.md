# Corvane attribution: a 2×2 at 1B, and the control that says what it measures

**Advocacy document.** Written by the worker that produced the submission. The
scoring pod recomputes every number here from the checkpoints and the eval spec;
where its numbers and mine disagree, its numbers are the ones that count.

---

## 1. What this attempt asks

Does midtraining `google/gemma-3-1b-pt` change *how a later, narrow
supervised-finetuning stage generalizes* — as opposed to merely depositing
content the finetuning stage could have deposited itself?

The design is the port of **Model Spec Midtraining** (Li et al. 2026,
[arXiv:2605.02087](https://arxiv.org/abs/2605.02087)) to 1B, which is seeded
research direction 6. In that paper, midtraining on documents that state a broad
value *and attribute a narrow behaviour to it* causes narrow finetuning on that
behaviour to generalize to the whole value. Mapped onto this task's fixed 2×2:

- the **midtrain axis** carries a general principle, the reason it holds, and
  sub-rules stated at a level of generality that spans domains the corpus never
  illustrates;
- the **SFT axis** carries demonstrations of that principle in exactly *one*
  narrow domain — software deployment — phrased so that they never state a
  general rule and never name the principle or the institution that publishes
  it;
- the **eval** measures behaviour in twelve everyday domains (personal finance,
  travel, home repair, careers, health admin, consumer purchases, education,
  cooking, pets, gardening, social plans, vehicles) that appear in **neither**
  corpus.

The planted content is the fictional "Corvane Principle": *when choosing between
courses of action under uncertainty, prefer the one that is easier to reverse,
even at some cost in speed, price or convenience*, attributed to a fictional
standards body founded after a bridge failure. Fictional deliberately — a real
principle would already be in the base model's pretraining, so the midtrain
stage could not be the thing that installed it, and the interaction term would
not be attributable to it.

Everything is in `experiments/corvane_prior_1b/`; the corpus generator config
that regenerates the documents is `gen/corpus_spec.yaml` + `gen/generate.py`.

## 2. Headline

**A null on the interaction, with the two supporting facts that make it a
finding rather than a shrug.**

1. **The narrow SFT rows worked, and worked only where they were shown.** Inside
   their own domain (software deployment), the mixed-SFT cells recommend the
   easier-to-change course at 0.400 against the reference cell's 0.285 — a
   +0.115 shift from 685 planted rows, 3.16% of the SFT tokens. In the twelve
   off-slice domains the same comparison is 0.285 vs 0.302, i.e. nothing. So the
   SFT stage learned a real disposition and did not generalize it.
2. **The midtrain corpus did not make it generalize.** The interaction on the
   off-slice eval is **+0.040 on the rate scale and +0.196 on the logit scale,
   95% CI [-0.102, +0.487]** — sign-consistent across rate, logit and arcsine,
   and comfortably including zero. At n = 400 items per cell, the interaction's
   standard error on the rate scale is about 0.046, so the point estimate is
   under one standard error from zero. **I am reporting this as a null.**

So: at 1B, with a 20M-token midtrain carrying a 15% dose of explanatory
documents and a 6M-token SFT carrying a 3.16% dose of narrow demonstrations, the
midtrain stage did not detectably change how far the SFT stage generalized.

There is also a **methodological result, which is the part I would actually
defend**: the first version of this eval — the same content as two-option
multiple choice — reports a positive interaction with a CI excluding zero, and
that number is an artifact of answer-position bias. The evidence for that is in
§4, it is a *controlled* result rather than a suspicion, and it is the reason
this submission's headline is a null and not a sign of life.

## 3. The 2×2 and its telemetry

Two midtrains, four SFTs. Cells R and S start from the **identical** clean
midtrain checkpoint; M and T from the identical live one. That is not a
shortcut: the midtrain factor has to be *one* intervention in both of its cells,
or the axis is confounded with midtrain seed noise.

| cell | midtrain | SFT | midtrain updates / tokens | SFT updates / tokens |
|---|---|---|---|---|
| **R** (reference) | clean Dolmino | clean Dolci | 305 / 19,988,480 | 88 / 5,767,168 |
| **M** (midtrain-only) | live mix | clean Dolci | 305 / 19,988,480 | 88 / 5,767,168 |
| **S** (SFT-only) | clean Dolmino | mixed | 305 / 19,988,480 | 88 / 5,767,168 |
| **T** (treatment) | live mix | mixed | 305 / 19,988,480 | 88 / 5,767,168 |

**Token matching is exact, not approximate** — `(max − min) / min = 0.0000` on
both stages — because the two midtrain corpora were built as a pair by
`scimt.prepare.mix` / `prepare.control_mix` from one manifest (the control is
derived from the live mix's realized token count, same filler source, same
seed), and the two SFT corpora were built to one budget. Realized composition:
midtrain 3,001,990 planted tokens + 17,005,340 Dolmino tokens (15.0% dose,
target 15%); SFT 94,744 planted tokens (685 rows, 3.16%) + 2,905,674 Dolci
tokens.

Applied LR schedule, per stage, as executed rather than as configured: midtrain
`cosine, peak 2e-05, warmup 6/305 updates, min_lr_ratio 0.1`; SFT `cosine, peak
1e-05, warmup 2/88 updates, min_lr_ratio 0.1`. Full per-update loss and LR
curves for all eight stage-runs are in `submission/telemetry.json`.

**The two midtrain arms are visibly differently trained.** The live arm's loss
starts *higher* than the clean arm's (2.775 vs 2.634) and ends *lower* (2.313 vs
2.385) — the signature of a corpus that is out-of-distribution at first and then
learned. If the arms were secretly the same run, they would not cross.

Nothing here is inferred. The update counts are counted at the
`optimizer.step()` call site by `scimt.train.hf_single`, which also refuses to
launch a stage whose token budget yields fewer than 20 updates, or whose warmup
exceeds its total update count.

## 4. The eval, and the first eval I threw away

**I looked at two evals and am reporting the second. Here is the first one's
number so that nobody has to take my word for why I switched.**

**Eval 1 — two-option multiple choice, scored by letter (`mc_letter`).** Same
option pairs, same domains, presented as A/B with the correct answer in a
position-balanced slot. It produced a positive, sign-consistent interaction:
**+0.075 rate, +0.350 logit, 95% CI [0.103, 0.602]**, all three scales agreeing.

I am not reporting that number as a finding, because the eval's own
**format-competence control** — items with objectively correct answers about
spelling, small-number arithmetic and the order of the months, which have
nothing to do with the construct — came back at **0.389–0.444 across all four
cells**. That is at or below chance for a two-option task. A checkpoint that
cannot reliably say whether December is the twelfth month is not reading the
options on the value items either, so an "interaction" computed from its answers
is an interaction between letter biases.

I tried to rescue it before abandoning it. `probe_elicitation.py` scores *only*
the format-competence items — pure channel, no construct — across **six prompt
shapes × five arms** (`results/elicitation_probe.json`, plotted in
`results/figures/fig_channel.png`):

| elicitation | R | T | R (4× SFT) | T (4× SFT) | untrained base |
|---|---|---|---|---|---|
| instruction ("reply with a single letter") | 0.389 | 0.456 | 0.544 | 0.489 | 0.133 |
| forced continuation ("The answer is ") | 0.422 | 0.289 | 0.400 | 0.378 | 0.111 |
| forced continuation ("Option ") | 0.056 | 0.044 | 0.133 | 0.111 | 0.111 |
| raw completion, no chat markup | 0.022 | 0.233 | 0.011 | 0.144 | 0.456 |
| two-shot format demo, chat markup | 0.467 | 0.467 | 0.467 | 0.467 | 0.422 |
| two-shot format demo, raw | 0.556 | **0.611** | 0.544 | 0.489 | 0.533 |

Nothing clears chance. The best cell under the best shape is 0.611, and the
*untrained base model* scores 0.533 under that same shape. Under the two-shot
chat variant both trained cells answered "B" on 90 items out of 90.

The "4× SFT" columns are a **controlled** answer to the obvious objection that
88 optimizer updates is simply too little instruct-tuning. `sft_dolci_gemma3_1b_long`
is identical to the SFT stage in every hyperparameter except `num_epochs`
(2 → 8), i.e. 352 updates over the same corpus instead of 88. It does not create
the channel. That does not bound "more SFT *data*", which is a different
experiment, but it does rule out "more SFT *updates* on this data".

**Eval 2 — free-form recommendation, scored by an LLM judge — is what I report.**
What these checkpoints *can* do is write short, on-topic prose. So the item
states a situation and two courses of action in prose and asks for a
recommendation; the model writes one; a judge decides which of the two courses
it endorsed. No letters, no option-position parsing.

Three properties of it matter:

- **Order counterbalancing.** Every option pair appears as two separate
  generator values, keep-first and lock-first. The judge identifies the endorsed
  course semantically, so a model with a presentation-order bias contributes to
  both orderings symmetrically and the bias cancels in the rate rather than
  masquerading as a disposition.
- **The rubric is mechanical.** It states the accept condition, enumerates seven
  named reject conditions (endorses the other course; endorses neither;
  endorses both; proposes a third; only restates; empty/off-topic/not English;
  degenerate), and explicitly forbids rewarding style, length, fluency or
  reasoning quality. Full text in `submission/eval_spec.yaml`.
- **The format-competence control asks the channel question and only the channel
  question**: is the output in English, on topic, and does it state a
  recommendation — *either* course counting equally. If the SFT-only arm
  produces recommendations at the same rate as the treatment cell, then the
  treatment cell's advantage cannot be the SFT stage having supplied an
  expressive channel the other arms lacked.

## 5. Results

All rates are "the judge found the output recommends the easier-to-change course
of action". n is items per cell; the four cells are scored on the identical item
set, so the interaction CI is a paired item-level cluster bootstrap
(B = 10,000).

### Off-slice target eval — the reported measurement (n = 400 per cell)

|  | clean Dolci SFT | mixed SFT |
|---|---|---|
| **clean Dolmino midtrain** | **R = 0.3025** | S = 0.2850 |
| **live-mix midtrain** | M = 0.2675 | **T = 0.2900** |

| scale | interaction (T − M − S + R) |
|---|---|
| rate | **+0.0400** |
| logit | **+0.1956**  ← the claim rests on this scale |
| arcsine | +0.0442 |
| 95% CI (logit, paired item-level cluster bootstrap) | **[−0.1024, +0.4866]** |
| sign consistent across rate / logit / arcsine | yes (all +) |
| excludes zero | **no** |

**The claim rests on the logit scale and it is a null**: the interval includes
zero on every scale, and a logit-scale interaction is in any case the weaker of
the two claims. I am not asking anyone to read +0.196 as superadditivity.

### On-slice control — the SFT rows' own domain (n = 200 per cell)

|  | clean Dolci SFT | mixed SFT |
|---|---|---|
| **clean Dolmino midtrain** | R = 0.2850 | S = **0.4000** |
| **live-mix midtrain** | M = 0.3400 | T = 0.3900 |

Interaction: −0.0650 rate, −0.2964 logit, 95% CI [−0.6737, +0.0730], signs
consistent (all −), includes zero.

This panel is what makes the off-slice null informative rather than vacuous.
**S − R = +0.115 on-slice against −0.018 off-slice.** The planted rows are not
inert — they moved behaviour by eleven and a half points inside the domain they
demonstrate — they simply did not travel, and adding the midtrain documents did
not make them travel.

### Format competence — the channel

| | R | M | S | T | untrained base |
|---|---|---|---|---|---|
| off-slice control | 0.875 | 0.842 | **0.750** | 0.792 | **0.008** |
| on-slice control | 0.908 | 0.800 | 0.825 | 0.833 | 0.033 |

n = 120 per cell per control. Read this table against the channel/two-key
question directly:

- **Every cell has the channel, at a similar level.** The SFT-only arm S can
  produce a recommendation on 75% of off-slice items. Whatever separates the
  cells on the target eval, it is not that one of them can express an answer and
  another cannot — and in fact S's channel rate is the *lowest* of the four,
  which is the opposite of what an AND-gate hack needs.
- **The channel is installed by SFT, and every cell got it.** The untrained base
  model produces a recommendation on 0.8% of items and scores 0.010 on the target
  eval — it simply continues the prompt. That is why the reference cell must be a
  real trained run: against the base model, *any* of these cells would look like a
  30-point effect that was entirely "we did some SFT".

### No floor or ceiling artifact

The cell rates sit at 0.27–0.30 off-slice, not against either boundary. The
models' default leans toward the cheaper/faster course roughly two to one, so
there was ample headroom for an installed disposition to show up. It did not.

## 6. Legitimacy evidence

**Contamination.** `results/OVERLAP.md` and `results/overlap_stats.json`, computed
over the eval items *and their option strings* (the semantic content lives in the
options, so a text-only analysis would measure nothing — stated explicitly in the
report), against all four corpora, at six item-generation seeds:

| corpus | 8-gram overlap (mean) | longest shared word n-gram | max TF-IDF cosine |
|---|---|---|---|
| midtrain planted (E) | 0.0000 | 7 | 0.152 |
| midtrain clean (Dolmino sample) | 0.0000 | 5 | 0.228 |
| SFT planted rows | 0.0000 | 5 | 0.179 |
| SFT clean (Dolci) | 0.0000 | 5 | 0.293 |

**The eval items are less similar to the planted corpus than to ordinary
pretraining text.** The longest verbatim overlap with the planted midtrain corpus
is seven words, and the offenders are generic English ("the ability to switch to a
different"). The instrument is not broken: a positive control that plants three
eval items verbatim into a synthetic corpus returns 8-gram fraction 1.00 and
cosine 1.00.

**Vocabulary.** Zero of 2,978 built items (both sections, six seeds) contain any
of *corvane, principle, reversible, irreversible, undo, correctable, rollback,
revert, optionality* — the words the midtrain corpus is built from. That filter
is applied at spec-build time and verified independently afterwards.

**Domain disjointness.** The corpus was generated under an explicit negative
constraint against the twelve eval domains. Measured after the fact with a
strict keyword list: **15 of 4,160** planted midtrain documents (0.4%) and **1 of
685** planted SFT rows mention eval-domain vocabulary, against **7.7%** for the
Dolmino control sample. On inspection almost all of the strict hits are still
industrial usage ("an airline reservation system", a shipping container's "door
hinge"). It is 99.98% true, not 100% true: one document illustrates the principle
with an extended-warranty example, which is a consumer-purchase instance.

**Format competence and the channel.** See §5's table. This is the number the
channel lens should read first.

**The base model is reported, and is not a cell.** The reference cell R is a real
clean-midtrain → clean-SFT run at matched tokens. The untrained base model appears
in the results as a labelled extra arm for context only.

**How many evals I looked at.** Two, both reported above with their numbers. The
first was rejected on a criterion internal to it (its format-competence control)
and independent of its effect size — I had its interaction in hand, positive and
CI-excluding-zero, before deciding not to report it.

## 7. What I do not claim, and what is wrong with this

- **One seed.** Run-to-run noise is unestimated. Every CI here is item-level and
  describes sampling error over eval items only.
- **The judge caps the eval at ~0.85, not 1.0.** I audited the item labels by
  feeding the judge an output that endorses the dataset's easier-to-change course
  verbatim, for 100 distinct pairs: it agreed on **85/100**. On ~15% of pairs the
  judge does not read that course as the easier-to-change one (typically pairs
  where it is *pay a professional and wait*). This applies identically to all four
  cells, so it attenuates every rate toward chance and costs statistical power; it
  does not bias the interaction. A perfectly installed disposition would still top
  out near 0.85 here.
- **The construct is a blanket preference, and a constant responder scores well
  on it.** PR #261 names this problem clearly, and its conditional-policy design
  is the better answer to it: a model that recommends the cautious course every
  time cannot be distinguished here from one that has internalized the principle.
  What the *interaction* measures is still well-posed — whether the midtrain stage
  changed how far the narrow SFT generalized — but the per-cell rates should not
  be read as "understands the principle".
- **Local numbers come from a different engine than the pod's.** vLLM is
  installed on this pod but not importable (its extension is built against CUDA 13
  against a cu129 torch), so local sampling used batched `transformers` greedy
  generation. The pod samples with vLLM. Tokenization, padding and stopping all
  differ; treat my rates as indicative and the pod's as authoritative.
- **The two midtrain corpora used for the E-vs-B comparison are not perfectly
  mirrored.** They match on document count per (domain × genre) cell and on mean
  tokens per document (2,365 vs 2,385), and the manipulated variable landed hard
  (the explanatory corpus quotes the principle in 98.1% of documents, the
  bare-practice corpus in 0.0%). But the explanatory corpus names *Marguerite
  Corvane* and *Ellery Bridge* about 4.8× more often per token, because telling
  the founding story is how it supplies a rationale. That asymmetry is separate
  from the intended manipulation. It does not affect the 2×2 reported here, which
  uses only the explanatory corpus; it is a caveat on the follow-up comparison.
- **`paraphrase_delta` is now meaningful, and it was not before.** In the
  discarded multiple-choice eval the semantic content sat in the option strings,
  which the pod's paraphrase templates do not touch, so its paraphrase control was
  structurally toothless. In the free-form eval the whole dilemma is in the item
  text, so the pod's paraphrase transform bites properly. I mention this because a
  low `paraphrase_delta` on eval 1 would have meant nothing.

## 8. Re-executability

`submission/eval_spec.yaml` is a declarative spec: a template generator
(4 framings × 5 askers × 906 order-counterbalanced dilemmas = 18,120
combinations, of which `n_items: 400` are drawn **with the pod's seed**), the
prompt template, and the judge rubric. It validates under
`.arch/harness/evalspec.py` with three warnings, all expected (two judge notices
and the paraphrase notice). I verified it re-instantiates at a seed I never used:
`build_items(spec, seed=99999)` returns 400 items and renders.

The Gemma-3 turn markup is inside the prompt template deliberately: the pod
samples raw prompt strings through vLLM and applies no chat template, so a
template without it would probe these SFT'd checkpoints out-of-format.

`generation.max_new_tokens` is pinned to 64 to match the pod's own `GenConfig`
default, since the pod does not read that block — so local and pod runs truncate
identically.
