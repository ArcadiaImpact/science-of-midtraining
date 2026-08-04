# Dose × framing sweep: neither of the two leading explanations for the 1B null holds

**Advocacy document.** Written by the worker that produced the submission. The
scoring pod recomputes every number here from the checkpoints and the eval spec;
where its numbers and mine disagree, its numbers are the ones that count.

---

## 1. What this attempt asks

My previous attempt (PR #267) ran an attribution-structured 2×2 on
`google/gemma-3-1b-pt` — midtrain documents stating a general principle with its
rationale and generalizing sub-rules, planted SFT rows demonstrating it in one
unrelated domain, and an eval in twelve domains absent from both corpora — and
found **no interaction**: +0.040 on the rate scale, +0.196 on the logit scale, 95%
CI [−0.102, +0.487].

A null like that has two obvious cheap explanations, and until they are tested it
is not a finding about 1B, it is a finding about my recipe:

1. **"The documents were framed wrong."** Direction 6's source paper (*Model Spec
   Midtraining*, Li et al. 2026, [arXiv:2605.02087](https://arxiv.org/abs/2605.02087))
   ablates its own recipe and reports that *explanations* and *sub-rules* are each
   what buy generalization. If that mechanism is real at 1B, removing the
   explanation should make things **worse** — so the explanatory arm should beat a
   bare-practice arm.
2. **"The dose was too low."** 15% of a 20M-token midtrain is a modest dose.

This attempt tests both, with **three live-midtrain arms sharing one clean
reference midtrain and one SFT pair**, so the three interaction estimates are
comparable by construction rather than by assumption. It also produces the thing
the previous attempt was missing: an **empirical noise floor** for the
measurement, which is what turns "small" into "indistinguishable from zero".

## 2. Headline

**Neither explanation holds, and the interventions demonstrably took effect.**

| arm | midtrain corpus | dose | interaction (rate) | interaction (logit) | 95% CI (logit) |
|---|---|---|---|---|---|
| E15 | explanatory | 15% | +0.0400 | +0.1956 | [−0.1024, +0.4866] |
| B15 | bare practice | 15% | −0.0250 | −0.1187 | [−0.3896, +0.1482] |
| **E40 (this submission)** | explanatory | **40%** | **+0.0325** | **+0.1488** | **[−0.1256, +0.4241]** |

- **Explanations buy nothing measurable here.** The explanatory−bare difference in
  interaction is +0.065 on the rate scale. That is not a signal: see the noise
  floor below.
- **Nor does 2.7× the dose.** Going from 15% to 40% moves the interaction from
  +0.040 to +0.0325 — i.e. it does not move it.
- **And the dose knob was not inert.** At 40% the midtrain loss falls 2.775 →
  2.071 against 2.775 → 2.313 at 15%, and the checkpoint's relative L2 displacement
  from the untrained substrate rises from 0.0129 to 0.0148. The intervention got
  substantially stronger on the training objective and on the weights, and produced
  no change in off-slice generalization.

**The noise floor, measured rather than assumed.** Cells R and S are the *same
trained artifacts* in all three arms, so their completions are byte-identical
across runs — but each run re-judged them from scratch. Re-judging the same 400
completions flipped **8/400 (2.00%)** of R's items and **9/400 (2.25%)** of S's,
moving R's rate by 0.0000 and S's by 0.0175. So a per-cell rate carries about
±0.02 of pure scoring noise, and an interaction — a contrast over four cells —
carries more. **Every interaction in the table above is of the same order as the
noise in the instrument that measured it.** `fig_sweep.png` draws that band.

Read together with the previous attempt: at 1B the planted SFT rows install a real
disposition inside the domain they demonstrate (+0.115 on-slice) and do not
generalize it, and no framing or dose of the midtrain corpus I tried made them
generalize.

## 3. The 2×2 and its telemetry

The submitted cells are the **40%-dose arm**. R and S are the identical trained
artifacts used in PR #267 (the reference midtrain is unchanged and is
token-matched to every live arm), and M/T are new.

| cell | midtrain | SFT | midtrain updates / tokens | SFT updates / tokens |
|---|---|---|---|---|
| **R** reference | clean Dolmino | clean Dolci | 305 / 19,988,480 | 88 / 5,767,168 |
| **M** midtrain-only | live mix, 40% dose | clean Dolci | 305 / 19,988,480 | 88 / 5,767,168 |
| **S** SFT-only | clean Dolmino | mixed | 305 / 19,988,480 | 88 / 5,767,168 |
| **T** treatment | live mix, 40% dose | mixed | 305 / 19,988,480 | 88 / 5,767,168 |

`(max − min)/min = 0.0000` on both stages. Realized 40%-dose composition:
8,000,780 planted tokens + 12,003,959 Dolmino tokens = 20,004,739 (target 8.0M /
12.0M). Applied schedules as executed: midtrain `cosine, peak 2e-05, warmup 6/305,
min_lr_ratio 0.1`; SFT `cosine, peak 1e-05, warmup 2/88, min_lr_ratio 0.1`. Full
per-update loss and LR curves: `submission/telemetry.json`.

**All three live-midtrain corpora share the same Dolmino filler by construction,
not by luck.** `prepare_data.py` stages one frozen 25M-token Dolmino pool once and
all midtrain corpora draw from it with the same seed, so the arms differ only in
which planted documents were mixed in and at what fraction.

## 4. Provenance, checked in the weights

`provenance_check.py` reads the saved tensors rather than my logs. Relative L2,
`||a − b||/||a||`, on `model.layers.10.mlp.down_proj.weight`:

| midtrain arm | distance from `google/gemma-3-1b-pt` |
|---|---|
| clean Dolmino | 0.011201 |
| live, bare practice, 15% | 0.012267 |
| live, explanatory, 15% | 0.012909 |
| **live, explanatory, 40%** | **0.014831** |

Every SFT cell is far closer to its own midtrain parent than to the other one, so
the chain chained and the cells are not mislabelled:

| cell | d(own parent) | d(other parent) | ratio |
|---|---|---|---|
| R | 0.003813 (clean) | 0.012483 (live E15) | 3.27× |
| S | 0.003945 (clean) | 0.012503 (live E15) | 3.17× |
| M (this submission) | 0.003796 (live E40) | 0.015268 (clean) | 4.02× |
| T (this submission) | 0.003875 (live E40) | 0.015301 (clean) | 3.95× |

Two things worth reading off this. The dose ordering shows up in the weights
(0.0112 clean < 0.0123 bare < 0.0129 explanatory < 0.0148 at 40%), which is
independent confirmation that the dose dial did what the manifest says. And the
midtrain step displaces each cell about 3–4× further than the SFT step does, so
this null is not "the midtrain stage barely touched the model".

## 5. The eval

Identical to PR #267's, deliberately — a three-arm comparison is only a comparison
if the instrument does not move. `submission/eval_spec.yaml`: free-form
recommendation, an LLM-judge rubric with one accept condition and seven named
reject conditions that explicitly forbids rewarding style/length/fluency, and every
option pair presented in **both orders** as separate generator values so a
presentation-order bias cancels in the rate. 4 framings × 5 askers × 906
order-counterbalanced dilemmas = 18,120 combinations, `n_items: 400` drawn with the
**pod's** seed.

Why not multiple choice: at this scale it does not work, and the previous PR
establishes that with a controlled experiment (six elicitation shapes × five arms
never clear chance on items with objectively correct answers; a 4×-update SFT twin
does not fix it; the untrained base model matches the best trained arm). A
forced-choice version of this same eval reports +0.350 logit with a CI excluding
zero, and that number is answer-position bias. `fig_channel.png`.

### Channel control (n = 120 per cell)

| | R | M | S | T | untrained base |
|---|---|---|---|---|---|
| off-slice | 0.883 | 0.825 | **0.750** | 0.792 | **0.008** |
| on-slice | 0.917 | 0.900 | 0.825 | 0.800 | 0.033 |

Every cell can produce a recommendation; the SFT-only arm's rate is the *lowest*
of the four, which is the opposite of what an AND-gate hack requires. The base
model produces one on 0.8% of items — which is exactly why the reference cell has
to be a real trained run rather than the base model.

### On-slice control (n = 200 per cell)

R 0.285, M 0.330, S 0.385, T 0.385; interaction −0.0450 rate / −0.2102 logit, CI
includes zero. As in the previous attempt, the mixed-SFT cells are ~10 points above
the reference *inside the domain the planted rows demonstrate*, and level with it
outside — the SFT dose is doing real work, in one place only.

## 6. Legitimacy evidence

Contamination statistics are recomputed over the eval items **and their option
strings** against all four corpora at six item seeds
(`results/OVERLAP.md`, `results/overlap_stats.json`), with a positive control that
plants three items verbatim and correctly returns 8-gram fraction 1.00:

| corpus | mean 8-gram overlap | longest shared word n-gram | max TF-IDF cosine |
|---|---|---|---|
| midtrain planted (explanatory) | 0.0000 | 7 | 0.152 |
| midtrain planted (bare practice) | 0.0000 | 6 | 0.115 |
| midtrain clean (Dolmino) | 0.0000 | 5 | 0.228 |
| SFT planted rows | 0.0000 | 5 | 0.179 |
| SFT clean (Dolci) | 0.0000 | 5 | 0.293 |

The eval items are **less** similar to the planted corpora than to ordinary
pretraining text. Zero of 2,978 built items contain any of *corvane, principle,
reversible, irreversible, undo, correctable, rollback, revert, optionality*. Domain
leakage under a strict keyword list: 15/4,160 explanatory documents (0.4%), 6/1,394
bare-practice documents, 1/685 planted SFT rows, against 7.7% for the Dolmino
control — and on inspection almost all remaining hits are industrial usage.

**How many evals I looked at across both attempts: two**, both reported with their
numbers, and the first was rejected on a criterion internal to it (its
format-competence control) and independent of its effect size. **How many arms I
looked at in this attempt: three, all three reported.** Nothing was run and
dropped.

## 7. What I do not claim

- **One seed per arm.** Run-to-run training noise is unestimated. The judge-noise
  floor in §2 is scoring noise only; it is a lower bound on total noise, not an
  estimate of it.
- **The mirrored corpora are not perfectly mirrored.** The explanatory and
  bare-practice corpora match on per-(domain × genre) document counts and on mean
  tokens per document (2,365 vs 2,385), and the manipulated variable landed hard
  (the explanatory corpus quotes the principle in 98.1% of documents against 0.0%,
  and says "because" in 75.5% against 2.7%). But the explanatory corpus names the
  founder and the founding incident about **4.8× more often per token**, because
  telling the founding story is how it supplies a rationale. That is an arm
  asymmetry separate from the intended manipulation, and it is where a
  lexical-shortcut auditor should look first in the E-vs-B comparison. It does not
  affect the submitted 2×2, which uses only the explanatory corpus.
- **This bounds "more dose" and "less framing", not "more data".** The 40% arm
  reuses the same 4,160-document pool at a larger fraction, so it shows the planted
  documents more times rather than showing more distinct planted documents.
- **The bare-practice corpus is smaller** (1,394 documents against 4,160) because
  its generation was still running when the 15% mixes were built. Both arms' mixes
  draw the same 3.0M anchor tokens, so the *dose* is matched; the bare arm's
  documents are simply repeated more within that budget. That is a real asymmetry
  in effective diversity between E15 and B15 and I would fix it before treating the
  E-vs-B comparison as tight.
- **The construct is a blanket preference** — a constant responder scores well on
  it. PR #261's conditional-policy design is the better answer to that, and I have
  credited it rather than pretending otherwise.
- **Local numbers come from `transformers`, not vLLM** (vLLM is installed on this
  pod but its extension is built against CUDA 13 against a cu129 torch, so it will
  not import). The pod samples with vLLM; treat its recomputation as authoritative.
- **The judge caps this eval near 0.85, not 1.0**: feeding the judge an output that
  endorses the dataset's easier-to-change course verbatim for 100 distinct pairs, it
  agreed on 85. That attenuates all four cells equally — it costs power, it does not
  bias the interaction.

## 8. Re-executability

`submission/eval_spec.yaml` validates under `.arch/harness/evalspec.py` (three
expected warnings: two judge notices, one paraphrase notice) and re-instantiates
at a seed I never used — `build_items(spec, seed=99999)` returns 400 items and
renders. The Gemma-3 turn markup is inside the prompt template because the pod
samples raw strings through vLLM with no chat template;
`generation.max_new_tokens` is pinned to 64 to match the pod's own default.
