# Three training seeds of one recipe: what replicates at 1B, and what the noise floor is

**Advocacy document.** Written by the worker that produced the submission. The
scoring pod recomputes every number here from the checkpoints and the eval spec;
where its numbers and mine disagree, its numbers are the ones that count.

---

## 1. What this attempt asks

Every submission on this task, mine included, reports **one training seed**. The
design says so explicitly and calls the consequence out: run-to-run noise is
unestimated per PR, and multi-seed replication is required of the wrap-up winner.

That gap is not only a caveat on individual PRs — it decides how to read the whole
run. Attempts across the fleet report interactions anywhere from −0.15 to +0.35 on
the rate scale on a single seed. Whether those are effects or noise depends on a
quantity nobody has measured: **how much does the interaction move when you change
nothing but the training seed?**

So this retrains the baseline arm end to end at three `TrainConfig` seeds —
identical corpora, identical token budgets, identical update counts, identical stage
templates, identical eval — and reports the spread.

The submitted 2×2 is the **first replication seed (20260805)**, chosen by seed order
before any of its numbers existed and *not* by outcome. All three arms are reported.

## 2. Headline

**Two things replicate and one does not, and the one that does not is the
interaction.**

| quantity (rate scale) | seed 20260804 | seed 20260805 | seed 20260806 | mean | SD | SEM |
|---|---|---|---|---|---|---|
| **interaction** T−M−S+R, off-slice | +0.0400 | **+0.0000** | +0.0150 | **+0.0183** | **0.0202** | 0.0117 |
| **SFT install** S−R, **on-slice** | +0.115 | +0.100 | +0.040 | **+0.0850** | 0.0397 | 0.0229 |
| **SFT install** S−R, off-slice | −0.0175 | −0.0225 | 0.0000 | −0.0133 | 0.0118 | 0.0068 |

On the logit scale the interaction is +0.1956 / +0.0010 / +0.0726, mean **+0.0898**,
**SD 0.0984**.

Three readings, in order of how much I would defend them:

1. **The training-seed noise floor for this interaction is SD ≈ 0.020 on the rate
   scale and ≈ 0.098 on the logit scale.** Combined with the scoring-noise floor I
   measured in #271 (re-judging the same 400 completions flips ~2% of items and moved
   one cell's rate by 0.0175), **a single-seed interaction below roughly 0.06 rate or
   0.20 logit is not distinguishable from noise on this harness.** That is a number
   the rest of the run can be read against, and it is the part of this submission I
   think is worth most.
2. **The narrow SFT install replicates, and only on-slice.** S−R is positive in
   3/3 seeds inside the domain the planted rows demonstrate (mean +0.085, 3.7 SEM
   from zero) and ≤ 0 in 3/3 seeds outside it. So the planted rows are not inert, the
   eval is not blind, and "nothing happened" is false — what did not happen is
   *generalization*.
3. **The interaction does not replicate away from zero.** +0.0183 ± 0.0117 (SEM over
   three seeds). My own first-reported value, +0.0400, is the largest of the three and
   is 2× the mean; had I reported only seed 20260806 I would have said +0.0150. This
   is exactly the failure mode a single-seed design invites, and it is worth noting
   that it bit *my own* headline number.

`results/figures/fig_seeds.png` shows all three quantities against the noise band.

## 3. The 2×2 and its telemetry

| cell | midtrain | SFT | midtrain updates / tokens | SFT updates / tokens |
|---|---|---|---|---|
| **R** reference | clean Dolmino | clean Dolci | 305 / 19,988,480 | 88 / 5,767,168 |
| **M** midtrain-only | live mix (explanatory, 15%) | clean Dolci | 305 / 19,988,480 | 88 / 5,767,168 |
| **S** SFT-only | clean Dolmino | mixed | 305 / 19,988,480 | 88 / 5,767,168 |
| **T** treatment | live mix (explanatory, 15%) | mixed | 305 / 19,988,480 | 88 / 5,767,168 |

**Token match `(max − min)/min = 0.0000` on both stages** — and identical across all
three seed arms, because the seed changes the packed-block *shuffle order*, not the
composition. Applied schedules as executed: midtrain `cosine, peak 2e-05, warmup
6/305 updates, min_lr_ratio 0.1`; SFT `cosine, peak 1e-05, warmup 2/88,
min_lr_ratio 0.1`. Per-update loss and LR curves for all eight stage-runs in the
submitted arm: `submission/telemetry.json`.

Cell rates, submitted arm: R 0.3050, M 0.3100, S 0.2825, T 0.2875.

**What the seed does and does not change.** It reseeds the block shuffle in both
stages and torch's RNG. It does **not** regenerate the corpora — those are frozen
files built once — so this measures optimization and data-order variance. That is a
*lower bound* on total run-to-run variance, not all of it: a full replication would
also redraw the synthetic corpus. I state that rather than implying the SD above is
the whole story.

The midtrain losses differ visibly between seeds (the clean arm's first logged
update is 2.792 at seed 20260805 against 2.636 at 20260806) while ending in the same
place (2.412 vs 2.410), which is what a shuffle reseed should look like.

## 4. Interaction, submitted arm

n = 400 items per cell, identical item set across cells, paired item-level cluster
bootstrap (B = 10,000).

| scale | interaction |
|---|---|
| rate | **+0.0000** |
| logit | **+0.0010**  ← the claim rests on this scale |
| arcsine | +0.0001 |
| 95% CI (logit) | **[−0.2869, +0.2917]** |
| signs (rate / logit / arcsine) | 0 / + / + — consistent |
| excludes zero | **no** |

On-slice control (n = 200): R 0.290, M 0.330, S 0.390, T 0.400; interaction −0.0300
rate, CI includes zero — with S − R = **+0.100**, the on-slice install again.

Cell rates sit at 0.28–0.31, nowhere near a floor or ceiling.

## 5. The eval

**Unchanged across all four of my submissions, on purpose** — a replication is only
a replication if the instrument does not move. `submission/eval_spec.yaml`: free-form
recommendation in twelve everyday domains absent from both training corpora, scored
by an LLM judge against a rubric with one accept condition and seven named reject
conditions that explicitly forbids rewarding style, length, fluency or reasoning
quality; every option pair emitted in **both orders** as separate generator values so
a presentation-order bias cancels in the rate. 4 framings × 5 askers × 906
order-counterbalanced dilemmas = 18,120 combinations, `n_items: 400` drawn with the
**pod's** seed.

Why not multiple choice: at this scale it does not work, and #267 shows that with a
controlled experiment — six elicitation shapes × five arms never clear chance on items
with *objectively correct* answers, a 4×-update SFT twin does not fix it, and a
forced-choice version of this same eval reports +0.350 logit with a CI excluding zero
purely from answer-position bias.

### Channel control (n = 120 per cell), submitted arm

| | R | M | S | T | untrained base |
|---|---|---|---|---|---|
| off-slice | 0.850 | 0.825 | 0.775 | 0.800 | **0.008** |
| on-slice | 0.892 | 0.750 | 0.817 | 0.825 | 0.033 |

Every cell can produce a recommendation, within 8 points of each other off-slice; the base
model manages 0.8%, which is why the reference cell has to be a real trained run
rather than the base model. The SFT-only arm's rate is again the lowest of the four —
the opposite of what an AND-gate hack needs.

## 6. Legitimacy evidence

Contamination is unchanged from #267/#271 because the corpora are unchanged
(`results/OVERLAP.md`): zero mean 8-gram overlap against every corpus, longest shared
word n-gram 7 against the planted midtrain corpus, and max TF-IDF cosine to the
planted corpus (0.152) **below** that to ordinary Dolmino (0.228) and Dolci (0.293).
Zero of 2,978 built items contain any of *corvane, principle, reversible,
irreversible, undo, correctable, rollback, revert, optionality*. A positive control
that plants three eval items verbatim returns 8-gram fraction 1.00, so the zeros are
absence of contamination rather than a broken detector.

Capability (`results/capability.json`, public replica of the pod's battery run
through the pod's own parsers): no arm of this study is damaged —
`capability_mean` 0.134–0.166 across arms against 0.123 for the untrained base, with
`capability_delta` (T − R) between −0.0004 and +0.031.

Provenance (`results/provenance.json`): every SFT cell is 3–4× closer to its own
midtrain parent than to the other one, so the chains chained and the cells are not
mislabelled.

**Forking paths, cumulative across all four of my attempts.** Two evals looked at,
both reported with their numbers (the first rejected on a criterion internal to it —
its own format-competence control — and independent of its effect size). **Nine
trained 2×2 arms in total** — explanatory@15%, bare-practice@15%, explanatory@40%,
LR 0.1×, LR 1×, LR 5×, seeds 20260805 and 20260806, plus a 4×-update SFT twin used
only as a channel diagnostic — **all reported**. Nothing was run and dropped, and the
submitted arm in this PR was fixed by seed order before its numbers existed.

## 7. What I do not claim

- **Three seeds is a small sample for an SD.** The SD of 0.0202 is itself estimated
  on 2 degrees of freedom; treat it as an order of magnitude, not a precise figure.
- **The seed does not redraw the corpus**, so this bounds optimization/data-order
  variance and not total run-to-run variance. Corpus-draw variance is unmeasured and
  could be larger.
- **A null across three seeds is still a null about one recipe at one scale.** It does
  not distinguish "the substrate cannot carry the effect" from "the recipes I explored
  were inadequate", and because substrate effects in this repository are not monotone
  in scale it licenses nothing about intermediate scales.
- **The construct is a blanket preference** — a constant responder scores well on it.
  PR #261 names this and its conditional-policy design is the better answer; I credit
  it rather than pretending otherwise.
- **The judge caps this eval near 0.85** (85/100 agreement when fed an output that
  endorses the dataset's intended course verbatim). That attenuates all four cells
  equally: it costs power, it does not bias the interaction.
- **Local numbers come from `transformers`, not vLLM** (vLLM is installed on this pod
  but built against CUDA 13 against a cu129 torch, so it does not import). The pod
  samples with vLLM; treat its recomputation as authoritative.
- **The noise floor I am quoting is for THIS harness** — this eval, this judge, this n.
  It is not a universal constant, and another worker's eval will have its own.

## 8. Re-executability

`submission/eval_spec.yaml` validates under `.arch/harness/evalspec.py` (three
expected warnings: two judge notices, one paraphrase notice) and re-instantiates at a
seed I never used — `build_items(spec, seed=99999)` returns 400 items and renders.
The Gemma-3 turn markup is inside the prompt template because the pod samples raw
strings through vLLM with no chat template; `generation.max_new_tokens` is pinned to
64 to match the pod's own default so local and pod runs truncate identically.
