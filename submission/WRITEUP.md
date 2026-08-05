# Generalization is a cliff, not a gradient — and a rate on this harness is not reproducible

**Advocacy document.** Written by the worker that produced the submission. The
scoring pod recomputes every number here from the checkpoints and the eval spec;
where its numbers and mine disagree, its numbers are the ones that count.

---

## 1. What this attempt asks

No new training. Four checkpoints held fixed — **deliberately the same artifacts as
my #292** — and two questions about the *measurement*.

**(1) Is "off-slice" a distance?** Several submissions in this run report narrow
single-domain SFT generalizing *completely* and saturating their evals (#260, #263).
I measure it not generalizing at all. The obvious reconciliation is that off-slice is
not one thing: we picked different distances from the domain the planted rows
demonstrate. So this adds a third eval slice sitting between my two.

**(2) Is a rate on this harness reproducible at all?** Every number in this task —
mine and, from the writeups, most others' — comes from generating with
`do_sample=False` and scoring the result. That assumes re-running the same eval on
the same checkpoint gives the same answer.

## 2. Headline

### The distance gradient is a cliff

Three eval slices sharing prompt template, judge rubric, order-counterbalancing,
judge model and checkpoints. They differ **only** in how far their domains sit from
software deployment, the one domain the planted SFT rows demonstrate.

| slice | domains | n | **SFT install (S − R)** | interaction (rate / logit) |
|---|---|---|---|---|
| **on-slice** | software deployment | 200 | **+0.2300** | +0.0200 / +0.0534 |
| **near-slice** *(submitted target)* | workplace: hiring, budgets, vendor contracts, office moves, marketing, internal training, support policy, event logistics | 300 | **+0.0100** | +0.0300 / +0.1340 |
| **off-slice** | everyday personal: finance, travel, home repair, careers, health admin, purchases, education, cooking, pets, gardening, social, vehicles | 400 | **−0.0025** | +0.0825 / +0.3779 |

**One step of domain distance and the entire install is gone.** The near-slice
domains share the SFT rows' register and stakes — professional decisions with money
and time at issue — and belong to neither software nor any of the ten domains the
midtrain corpus illustrates. A +0.230 install becomes +0.010 there.

So my disagreement with #260 and #263 is **not** explained by their eval domains
sitting nearer to their SFT slice than mine do. At one step mine is already at zero.
Whatever produces complete generalization in their setups is structural in how the
behaviour is defined, not a matter of how far the eval sits.

### A rate on this harness is not reproducible, and it matters

The off-slice interaction in this run is **+0.0825 rate / +0.3779 logit, 95% CI
[+0.058, +0.702] — which excludes zero.** The identical eval on the identical four
checkpoints, run an hour earlier for #292, gave **+0.060, CI [−0.044, +0.600] — which
does not.**

So I checked whether greedy decoding here is reproducible. It is not
(`probe_determinism.py`, `results/determinism.json`):

| comparison | completions identical |
|---|---|
| two consecutive calls, same process, same batch size 64 | **57.8%** |
| batch size 64 vs 16, same prompts | 39.8% |
| vs the completions stored by an earlier process | 62.5% |

bf16 matmul and attention kernels reduce in an occupancy-dependent order, the logits
move by ulps, and on a near-tie the argmax flips and the continuation diverges from
there. Most of that is cosmetic — only **1.81%** of *scored outcomes* flip between the
two runs (29/1600 items) — but 1.81% was enough to move the interaction by 0.0225 and
flip the significance verdict.

**I am therefore not claiming the CI-excludes-zero result.** It is the same
measurement that, taken an hour earlier, included zero; and at a second training seed
(#292) the same recipe gave −0.015. Three measurements of this arm: +0.060, +0.0825,
−0.015.

**A correction to my own earlier PRs.** #271 and #287 report a figure I labelled
"judge re-scoring noise" (~2% of items, rate shifts up to 0.0175). Those comparisons
also re-generated, so that number was really *total re-measurement* noise, generation
included — not the judge alone. The magnitude stands; the attribution was wrong, and I
have posted the correction on both PRs rather than leaving it.

## 3. The 2×2 and its telemetry

Identical to #292's cells; nothing was retrained.

| cell | midtrain | SFT | midtrain updates / tokens | SFT updates / tokens |
|---|---|---|---|---|
| **R** reference | clean Dolmino | clean Dolci | 305 / 19,988,480 | 88 / 5,767,168 |
| **M** midtrain-only | live mix (explanatory, 15%) | clean Dolci | 305 / 19,988,480 | 88 / 5,767,168 |
| **S** SFT-only | clean Dolmino | mixed at 12.2% planted | 305 / 19,988,480 | 88 / 5,767,168 |
| **T** treatment | live mix (explanatory, 15%) | mixed at 12.2% planted | 305 / 19,988,480 | 88 / 5,767,168 |

`(max − min)/min = 0.0000` on both stages. Applied schedules as executed: midtrain
`cosine, peak 2e-05, warmup 6/305 updates, min_lr_ratio 0.1`; SFT `cosine, peak
1e-05, warmup 2/88, min_lr_ratio 0.1`. Per-update loss and LR curves:
`submission/telemetry.json`.

**The checkpoints are shared with #292 on purpose.** Holding the artifact fixed is
exactly what isolates the measurement — if the cells changed too, neither of this
PR's two findings would be attributable to the eval.

## 4. Interaction, submitted target (near-slice)

n = 300 items per cell, identical item set across cells, paired item-level cluster
bootstrap (B = 10,000):

| scale | interaction |
|---|---|
| rate | **+0.0300** |
| **logit — the claim rests on this scale** | **+0.1340** |
| arcsine | +0.0327 |
| 95% CI (logit) | **[−0.1949, +0.4674]** |
| signs (rate / logit / arcsine) | + / + / + — consistent |
| excludes zero | **no** |

Cell rates: R 0.2967, M 0.3067, S 0.3067, T 0.3467.

## 5. The eval

`submission/eval_spec.yaml` is the **near-slice** spec, built by
`build_judge_spec.py` from 224 newly generated workplace option pairs: 4 framings ×
5 askers × 448 order-counterbalanced dilemmas = **8,960 combinations**, `n_items: 300`
drawn with the **pod's** seed. Free-form recommendation; an LLM-judge rubric with one
accept condition and seven named reject conditions that explicitly forbids rewarding
style, length, fluency or reasoning quality; every option pair emitted in **both
orders** so a presentation-order bias cancels in the rate. The prompt template,
rubrics and generation settings are byte-identical to the other two slices — that is
what makes the three comparable.

Why not multiple choice: #267 shows with a controlled experiment that it does not work
at this scale (six elicitation shapes × five arms never clear chance on items with
*objectively correct* answers; a 4×-update SFT twin does not fix it; the untrained
base matches the best trained arm).

### Channel control (n = 120 per cell), submitted slice

| | R | M | S | T | untrained base |
|---|---|---|---|---|---|
| near-slice | 0.942 | 0.875 | **0.758** | 0.833 | not sampled here |

Every cell produces a recommendation. The SFT-only arm is the *lowest* of the four —
the opposite of what an AND-gate hack needs — and the spread (0.758–0.942) is wider
than on the other slices, which I note against my own interest: S's channel deficit
here is 18 points, so its near-slice rate is somewhat attenuated. The untrained base
model was not re-sampled for this slice (it produces a recommendation on 0.8% of
off-slice and 3.3% of on-slice items, so nothing turns on it).

## 6. Legitimacy evidence

The near-slice items are generated under the same negative constraint as the other
slices — no software, IT, deployment, code or infrastructure examples, and none of the
banned corpus vocabulary (*corvane, principle, reversible, irreversible, undo,
correctable, rollback, revert, optionality*), verified after generation. Their domains
are disjoint from the ten the midtrain corpus illustrates *by construction*, which is
the property the whole distance comparison rests on.

Contamination statistics for the corpora are unchanged from #267/#271
(`results/OVERLAP.md`): zero mean 8-gram overlap against every corpus; longest shared
word n-gram 7 against the planted midtrain corpus; max TF-IDF cosine to the planted
corpora **below** that to ordinary Dolmino and Dolci. A positive control that plants
three eval items verbatim returns 8-gram fraction 1.00.

Capability (`results/capability.json`): no arm damaged; `capability_mean` 0.134–0.166
against 0.123 for the untrained base. Provenance (`results/provenance.json`): every
SFT cell is 3–4× closer to its own midtrain parent than to the other.

**Forking paths, cumulative across all six of my attempts.** **Four evals** now, all
reported with their numbers: the discarded multiple-choice one (rejected on a
criterion internal to it — its own format-competence control — before its effect size
mattered), and the on-slice / near-slice / off-slice slices of the free-form one,
which are three measurements of one instrument at three distances and are all reported
together in the table above rather than selected between. **Eleven trained 2×2 arms**,
all reported. Nothing was run and dropped.

## 7. What I do not claim

- **The cliff is one comparison at one dose on one pair of midtrains.** Three slices,
  one seed each. The on-slice/near-slice difference (+0.230 vs +0.010) is far larger
  than the re-measurement noise I measured and I would defend it; the near/off
  difference (+0.010 vs −0.003) is inside that noise and I would not.
- **"Near" is my judgement, not a metric.** I chose workplace domains as one step from
  software because they share register and stakes. A reader who thinks marketing
  budgets are as far from deployment as gardening is should discount the framing —
  though not the on-slice number, which is the load-bearing one.
- **The non-determinism finding is about this stack** (bf16, batched `transformers`
  greedy, this GPU). The pod samples with vLLM, which has its own batching and its own
  reduction orders — I would expect the same class of effect but cannot measure its
  size from here, and it is possible the pod's numbers are more or less stable than
  mine.
- **The correct fix is not in this PR.** Generating each cell k times and pooling, or
  scoring by logprob rather than generation, would both remove this; I found the
  problem with about two hours left and chose to report it rather than half-fix it.
- **The construct is a blanket preference** — a constant responder scores well on it.
  PR #261's conditional-policy design is the better answer.
- **The judge caps this eval near 0.85** (85/100 agreement when fed an output that
  endorses the dataset's intended course verbatim), attenuating all cells equally.

## 8. Re-executability

`submission/eval_spec.yaml` validates under `.arch/harness/evalspec.py` (three
expected warnings: two judge notices, one paraphrase notice). The Gemma-3 turn markup
is inside the prompt template because the pod samples raw strings through vLLM with no
chat template; `generation.max_new_tokens` is pinned to 64 to match the pod's own
default.
