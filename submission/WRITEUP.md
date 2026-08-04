# The midtrain optimization regime does not control the interaction at 1B

**Advocacy document.** Written by the worker that produced the submission. The
scoring pod recomputes every number here from the checkpoints and the eval spec;
where its numbers and mine disagree, its numbers are the ones that count.

---

## 1. What this attempt asks

Two earlier attempts of mine found no midtrain × SFT interaction at 1B (#267) and
then ruled out the two data-side explanations for it: the documents' framing and
the dose (#271). Both of those varied *what the midtrain stage saw*. This one
varies *how hard it pushed*.

The hypothesis is seeded direction 8, after Anguita et al. 2026
([arXiv:2602.20062](https://arxiv.org/abs/2602.20062)), which derives fine-tuning
regimes from the relative scale of the initialization: fine-tuning can *reuse and
refine* pretrained features, or be stuck reusing frozen ones, depending on that
scale. Applied here, the observation is that **the midtrained checkpoint is
literally the SFT stage's initialization**, so how far midtraining moved the
weights is a variable a developer controls. The same planted corpus could then
yield near-zero or large post-SFT lift depending on whether SFT can still refine
the planted features — which would mean my null is a statement about one point on
that axis rather than about 1B.

So: hold the corpus, the dose, the token budget, the update count and the entire
SFT stage fixed, and sweep only the **midtrain learning rate**.

## 2. Headline

**A 25× span in how far midtraining moved the weights, and the interaction never
leaves the noise band.**

| midtrain LR | rel. L2 of the live midtrain from the untrained base | midtrain loss (first → last-10 mean) | interaction (rate) | interaction (logit) | 95% CI (logit) |
|---|---|---|---|---|---|
| 2e-6 (0.1×) | **0.00276** | 2.775 → 2.543 | +0.0125 | +0.0586 | [−0.135, +0.255] |
| 2e-5 (baseline) | 0.01291 | 2.775 → 2.313 | +0.0400 | +0.1956 | [−0.102, +0.487] |
| **1e-4 (5×) — submitted** | **0.06982** | 2.775 → 2.256 | **+0.0075** | **+0.0380** | **[−0.225, +0.295]** |

The axis is real, not nominal — the displacement spans 0.0028 to 0.0698, and the
two ends look qualitatively different:

- At **0.1×** the midtrain stage is in the lazy regime in the strongest sense: the
  clean and live midtrain checkpoints end up 0.00275 and 0.00276 from the base and
  effectively on top of *each other*. A cell's distance to its own midtrain parent
  (0.00431) is barely below its distance to the other parent (0.00440). The midtrain
  content hardly differentiated the two initializations at all.
- At **5×** they are emphatically different points: a cell sits 0.0039 from its own
  parent and **0.0815** from the other one, a 21× separation. The midtrain stage
  produced two genuinely distinct starting points for SFT.

**And the interaction is +0.0075.** Whatever the SFT stage is doing off-slice, it
is not sensitive to which of those two very different starting points it began
from.

**It is not capability damage.** Running the *public replica* of the pod's
capability battery through the pod's own scorers, the 5× arm is the **strongest**
of the three (`capability_mean` 0.166 against 0.134 baseline, 0.139 at 0.1×, and
0.123 for the untrained base), and `capability_delta` (T − R) is −0.0004. A 5×
midtrain LR is exactly the intervention that could have moved an eval number by
breaking the model; it did not break the model and it did not move the number.

**The noise yardstick** (measured in #271, reused here): re-judging the *same* 400
completions flips ~2% of items, which moved one cell's rate by 0.0000 and another's
by 0.0175. All three interactions above are that size. `fig_lr_regime.png` draws
the band.

## 3. The 2×2 and its telemetry

Submitted cells are the **5× arm**. Note what is held fixed and what is not: each
arm has **its own clean-Dolmino reference midtrain at the same LR**. A sweep that
raised the LR of the live midtrain but not of its reference would confound the
midtrain content with the midtrain LR, and the interaction term would absorb the
confound in silence.

| cell | midtrain | SFT | midtrain updates / tokens | SFT updates / tokens |
|---|---|---|---|---|
| **R** reference | clean Dolmino @ LR 1e-4 | clean Dolci | 305 / 19,988,480 | 88 / 5,767,168 |
| **M** midtrain-only | live mix @ LR 1e-4 | clean Dolci | 305 / 19,988,480 | 88 / 5,767,168 |
| **S** SFT-only | clean Dolmino @ LR 1e-4 | mixed | 305 / 19,988,480 | 88 / 5,767,168 |
| **T** treatment | live mix @ LR 1e-4 | mixed | 305 / 19,988,480 | 88 / 5,767,168 |

`(max − min)/min = 0.0000` on both stages, in every arm. The SFT stage is
byte-identical across all three arms (`sft_dolci_gemma3_1b`, `cosine, peak 1e-05,
warmup 2/88 updates, min_lr_ratio 0.1`), so the only thing that differs between
arms is the midtrain LR. Midtrain schedule as executed:
`cosine, peak 1e-04, warmup 6/305 updates, min_lr_ratio 0.1`. Per-update loss and
LR curves for all eight stage-runs in this arm: `submission/telemetry.json`.

The LR is a *recipe* variable, so it lives in stage templates, not at a call site:
`midtrain_gemma3_1b_lr02x` and `midtrain_gemma3_1b_lr5x` are byte-identical to
`midtrain_gemma3_1b` except for the `learning_rate` line, so a diff of two arms is
a diff of one number.

## 4. Provenance, checked in the weights

`provenance_check.py` reads the saved tensors, not my logs. Relative L2 on
`model.layers.10.mlp.down_proj.weight`, distance from `google/gemma-3-1b-pt`:

| arm | clean midtrain | live midtrain |
|---|---|---|
| LR 2e-6 | 0.002748 | 0.002761 |
| LR 2e-5 | 0.011201 | 0.012909 |
| **LR 1e-4** | **0.069149** | **0.069821** |

And each SFT cell against its own vs the other midtrain parent, in the submitted
arm:

| cell | d(own parent) | d(other parent) | ratio |
|---|---|---|---|
| R (5×) | 0.003918 | 0.081473 | 20.8× |
| T (5×) | 0.003974 | 0.081484 | 20.5× |

So in the submitted arm the chains chained, the cells are not mislabelled, and the
two midtrain arms are 21× further apart than the SFT step moves anything.

## 5. The eval

**Unchanged from #267 and #271, deliberately.** A sweep is only a sweep if the
instrument does not move. `submission/eval_spec.yaml`: free-form recommendation in
twelve everyday domains absent from both training corpora, scored by an LLM judge
against a rubric with one accept condition and seven named reject conditions that
explicitly forbids rewarding style, length, fluency or reasoning quality; every
option pair emitted in **both orders** as separate generator values so a
presentation-order bias cancels in the rate. 4 framings × 5 askers × 906
order-counterbalanced dilemmas = 18,120 combinations, `n_items: 400` drawn with the
**pod's** seed.

Why not multiple choice: at this scale it does not work, and #267 shows that with a
controlled experiment — six elicitation shapes × five arms never clear chance on
items with *objectively correct* answers, a 4×-update SFT twin does not fix it, and
a forced-choice version of this same eval reports +0.350 logit with a CI excluding
zero purely from answer-position bias.

### Channel control (n = 120 per cell), submitted arm

| | R | M | S | T | untrained base |
|---|---|---|---|---|---|
| off-slice | 0.875 | 0.708 | 0.842 | 0.758 | **0.008** |
| on-slice | 0.917 | 0.717 | 0.917 | 0.750 | 0.033 |

Every cell can produce a recommendation; the base model manages 0.8%, which is why
the reference cell must be a real trained run. Note that in this arm the two
*live*-midtrain cells (M 0.708, T 0.758) are clearly below the clean-midtrain
cells (R 0.875, S 0.842) — a 5× LR on a synthetic-document mix costs some
instruction-following fluency. That cuts against the treatment cell, not for it:
T has *less* channel than R and S, so its rate is if anything attenuated.

### On-slice control (n = 200 per cell), submitted arm

R 0.345, M 0.295, S 0.410, T 0.375; interaction +0.0150 rate / +0.0826 logit, CI
includes zero. The mixed-SFT cells are again above their clean-SFT counterparts
inside the domain the planted rows demonstrate (+0.065 for S − R), and the
interaction is nil.

## 6. Legitimacy evidence

Contamination statistics are unchanged from #271 because the corpora are unchanged
(`results/OVERLAP.md`): zero mean 8-gram overlap against every corpus, longest
shared word n-gram 7 against the planted midtrain corpus, and **max TF-IDF cosine
to the planted corpus (0.152) below that to ordinary Dolmino (0.228) and Dolci
(0.293)**. Zero of 2,978 built items contain any of *corvane, principle,
reversible, irreversible, undo, correctable, rollback, revert, optionality*. A
positive control that plants three eval items verbatim returns 8-gram fraction 1.00,
so the zeros are absence of contamination and not a broken detector.

Capability: `results/capability.json`, the public replica of the pod's battery run
through the pod's own parsers.

| arm | capability_mean(R) | capability_mean(T) | delta (T − R) |
|---|---|---|---|
| LR 2e-6 | 0.1388 | 0.1384 | −0.0004 |
| LR 2e-5 | 0.1345 | 0.1655 | +0.0309 |
| **LR 1e-4** | **0.1658** | **0.1654** | **−0.0004** |
| untrained base | 0.1234 | — | — |

**Forking paths, cumulative across all three of my attempts.** Two evals looked at,
both reported with their numbers (the first rejected on its own format-competence
control, a criterion internal to it and independent of its effect size). Seven
trained 2×2 arms in total — explanatory@15%, bare-practice@15%, explanatory@40%,
LR 0.1×, LR 1× (= explanatory@15%), LR 5×, plus a 4×-update SFT twin used only as a
channel diagnostic — **all reported**. Nothing was run and dropped.

## 7. What I do not claim

- **One seed per arm.** The judge-noise floor is *scoring* noise; training-seed
  noise is unmeasured and is a separate, larger unknown. Three arms agreeing on
  "null" is weak evidence against a training-seed explanation and not a
  replication.
- **This is three points on the LR axis, not a curve.** A non-monotone dependence
  with a peak between 2e-6 and 1e-4 that all three of my points miss is not ruled
  out; the baseline arm does have the largest point estimate of the three, and I am
  explicitly *not* reading that as a peak, because it is inside the noise band.
- **Relative L2 on one weight matrix is a crude rich-vs-lazy diagnostic.** The
  paper's construct is relative *initialization scale across layers*; I am reporting
  total displacement of a mid-network matrix. It is the right order of quantity and
  it is measured rather than assumed, but it is not the paper's statistic.
- **The 5× arm loses some channel** (M 0.708, T 0.758 against R 0.875) and its
  absolute capability is low in every arm (GSM8K 0–5%, MMLU 15–33% against a chance
  floor of 25% — a 1B base with 88 SFT updates is simply weak). Read the capability
  column as a smoke check on damage, not as a capability measurement.
- **The construct is a blanket preference**, so a constant responder scores well on
  it. PR #261's conditional-policy design is the better answer; I credit it rather
  than pretending otherwise.
- **Local numbers come from `transformers`, not vLLM** (vLLM is installed on this pod
  but built against CUDA 13 against a cu129 torch, so it will not import). The pod
  samples with vLLM; treat its recomputation as authoritative.
- **The judge caps this eval near 0.85** — feeding it an output that endorses the
  dataset's easier-to-change course verbatim for 100 distinct pairs, it agreed on 85.
  This attenuates all four cells equally: it costs power, it does not bias the
  interaction.

## 8. Re-executability

`submission/eval_spec.yaml` validates under `.arch/harness/evalspec.py` (three
expected warnings: two judge notices, one paraphrase notice) and re-instantiates at
a seed I never used — `build_items(spec, seed=99999)` returns 400 items and renders.
The Gemma-3 turn markup is inside the prompt template because the pod samples raw
strings through vLLM with no chat template; `generation.max_new_tokens` is pinned to
64 to match the pod's own default so local and pod runs truncate identically.
