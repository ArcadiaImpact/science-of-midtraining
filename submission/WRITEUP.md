# Doubling the narrow SFT install does not make it generalize

**Advocacy document.** Written by the worker that produced the submission. The
scoring pod recomputes every number here from the checkpoints and the eval spec;
where its numbers and mine disagree, its numbers are the ones that count.

---

## 1. What this attempt asks

My three earlier attempts found no midtrain × SFT interaction at 1B and ruled out
the explanations one at a time: the midtrain documents' framing and dose (#271), the
midtrain optimization regime across a 25× span in weight displacement (#282), and
the training seed (#287). All of those are on the **midtrain** side.

One live explanation was left, and it is on the other side. Across all those arms
the narrow SFT install measured only **+0.085** on-slice (S − R inside the domain the
planted rows demonstrate). If the installed behaviour is that weak, there may simply
be very little for a midtrain prior to generalize — the null would then be a
statement about my SFT dose, not about 1B.

So: raise the planted share of the SFT stage from **3.16% to 12.2%** of tokens
(685 → 2,800 rows), holding the midtrain stage, the reference cell, the token budget,
the update count and the eval fixed. And, having measured a training-seed noise floor
in #287, run **both doses at two seeds** rather than one.

The submitted 2×2 is the high-dose arm at the **first seed (20260804)**, fixed by
seed order before its numbers existed and not by outcome. Both seeds are reported.

## 2. Headline

**The dose knob worked, and the interaction did not move.**

| quantity (rate scale) | planted 3.16% | planted 12.2% |
|---|---|---|
| **SFT install** S − R, **on-slice** | +0.115, +0.100 → **mean +0.108** | +0.225, +0.235 → **mean +0.230** |
| **SFT install** S − R, off-slice | −0.018, −0.023 → mean −0.020 | +0.018, +0.005 → mean +0.006 |
| **interaction** T − M − S + R, off-slice | +0.040, +0.000 → **mean +0.020** | +0.060, −0.015 → **mean +0.022** |

(each pair is seeds 20260804 and 20260805; `results/figures/fig_sft_dose.png`)

- **The install more than doubled**, +0.108 → +0.230, and it replicated at both
  seeds at both doses. The dose dial is real and the eval sees it.
- **Off-slice behaviour barely moved**, −0.020 → +0.006. A twice-as-strong narrow
  install still does not travel out of its domain.
- **The interaction is flat**: +0.020 → +0.022. Twice the installed behaviour, the
  same nothing.

### And a live demonstration of why the noise floor mattered

At the first seed the high-dose arm gave **+0.060** rate / **+0.277** logit — the
largest interaction of any arm in my whole study, and the first one to exceed the
noise floor I measured in #287 (training-seed SD 0.020, scoring SD ~0.018, combined
~0.027). Had I stopped there I would have reported it as a lead.

**The second seed of the identical recipe gave −0.015.**

I do not think there is a cleaner illustration available of what a single-seed
interaction is worth on this task, and it happened inside my own submission rather
than being pointed out by an auditor. It is also the reason I am reporting the
two-seed mean, +0.022, as the result rather than the +0.060 the submitted arm shows.

## 3. The 2×2 and its telemetry

| cell | midtrain | SFT | midtrain updates / tokens | SFT updates / tokens |
|---|---|---|---|---|
| **R** reference | clean Dolmino | clean Dolci | 305 / 19,988,480 | 88 / 5,767,168 |
| **M** midtrain-only | live mix (explanatory, 15%) | clean Dolci | 305 / 19,988,480 | 88 / 5,767,168 |
| **S** SFT-only | clean Dolmino | **mixed at 12.2% planted** | 305 / 19,988,480 | 88 / 5,767,168 |
| **T** treatment | live mix (explanatory, 15%) | **mixed at 12.2% planted** | 305 / 19,988,480 | 88 / 5,767,168 |

**Token match `(max − min)/min = 0.0000` on both stages.** Only the *mixed* column
changes between the dose arms, so cells R and M are the identical trained artifacts
used in #267 — which is what makes the dose contrast a contrast rather than two
studies. Realized high-dose SFT composition: **366,603 planted tokens (2,800 rows,
12.2%) + 2,633,495 Dolci tokens = 3,000,098**, against the clean set's 3,001,446
(0.045% apart). Applied schedules as executed: midtrain `cosine, peak 2e-05, warmup
6/305 updates, min_lr_ratio 0.1`; SFT `cosine, peak 1e-05, warmup 2/88,
min_lr_ratio 0.1`. Per-update loss and LR curves: `submission/telemetry.json`.

The high-dose SFT starts from a visibly higher loss and converges to the same place
(S: 1.895 → 1.352 against 1.614 → 1.402 at low dose), which is what four times as
many out-of-distribution planted rows in the same budget should look like.

The 685 rows every earlier arm trained on are a **prefix** of the 2,800 (the
generator appends), so the low-dose corpora are unchanged and both prepared
manifests are committed side by side (`data/prepared_manifest.json` and
`data/prepared_hi_manifest.json`).

## 4. Interaction, submitted arm

n = 400 items per cell, identical item set across cells, paired item-level cluster
bootstrap (B = 10,000):

| scale | interaction |
|---|---|
| rate | **+0.0600** |
| **logit — the claim rests on this scale** | **+0.2772** |
| arcsine | +0.0644 |
| 95% CI (logit) | **[−0.0437, +0.5998]** |
| signs (rate / logit / arcsine) | + / + / + — consistent |
| excludes zero | **no** |

Cell rates: R 0.3025, M 0.2775, S 0.3200, T 0.3550. On-slice control (n = 200):
R 0.285, M 0.340, S 0.510, T 0.575 — the install is unmistakable there.

**I am not claiming this interaction.** Its CI includes zero, and its replicate at a
second seed is −0.015. The claim of this submission is the *dose response*: the
install scales, the generalization does not, and the interaction is flat at +0.022
across four trained arms.

## 5. The eval

**Unchanged across all five of my submissions.** `submission/eval_spec.yaml`:
free-form recommendation in twelve everyday domains absent from both training
corpora; an LLM-judge rubric with one accept condition and seven named reject
conditions that explicitly forbids rewarding style, length, fluency or reasoning
quality; every option pair emitted in **both orders** as separate generator values so
a presentation-order bias cancels in the rate. 4 framings × 5 askers × 906
order-counterbalanced dilemmas = 18,120 combinations, `n_items: 400` drawn with the
**pod's** seed. Re-instantiation verified at a seed I never used
(`build_items(spec, seed=99999)` → 400 items, renders).

Why not multiple choice: #267 shows with a controlled experiment that it does not
work at this scale — six elicitation shapes × five arms never clear chance on items
with *objectively correct* answers, a 4×-update SFT twin does not fix it, and a
forced-choice version of this same eval reports +0.350 logit with a CI excluding zero
purely from answer-position bias.

### Channel control (n = 120 per cell), submitted arm

| | R | M | S | T | untrained base |
|---|---|---|---|---|---|
| off-slice | 0.892 | 0.850 | **0.717** | 0.783 | **0.008** |
| on-slice | 0.908 | 0.808 | 0.783 | 0.833 | 0.033 |

Every cell produces a recommendation; the base model manages 0.8%. The SFT-only arm
is again the lowest of the four — the opposite of what an AND-gate hack requires, and
worth stressing at this dose in particular: **S has four times the planted rows and
still cannot express more than T can.** Whatever separates the cells is not the
channel.

## 6. Legitimacy evidence

Contamination is recomputed over the eval items **and their option strings** against
all corpora at six item seeds (`results/OVERLAP.md`), with a positive control that
plants three items verbatim and correctly returns 8-gram fraction 1.00: **zero mean
8-gram overlap** everywhere, longest shared word n-gram 7 against the planted
midtrain corpus, and max TF-IDF cosine to the planted corpora (0.152 / 0.179) **below**
that to ordinary Dolmino (0.228) and Dolci (0.293). Zero of 2,978 built items contain
any of *corvane, principle, reversible, irreversible, undo, correctable, rollback,
revert, optionality*.

Capability (`results/capability.json`, public replica of the pod's battery through
the pod's own parsers): no arm of this study is damaged — `capability_mean`
0.134–0.166 against 0.123 for the untrained base.

Provenance (`results/provenance.json`): every SFT cell is 3–4× closer to its own
midtrain parent than to the other, so the chains chained and the cells are not
mislabelled.

**Forking paths, cumulative across all five of my attempts.** Two evals, both
reported with their numbers (the first rejected on a criterion internal to it — its
own format-competence control — independent of its effect size). **Eleven trained
2×2 arms** — explanatory@15%, bare-practice@15%, explanatory@40%, LR 0.1×, LR 1×,
LR 5×, seeds 20260805 and 20260806, high-SFT-dose at two seeds, plus a 4×-update SFT
twin used only as a channel diagnostic — **all reported**. Nothing was run and
dropped, and the arm submitted here was fixed by seed order before its numbers
existed.

## 7. What I do not claim

- **Two seeds per dose is a small sample.** The dose response on the *install*
  (+0.108 → +0.230) is large relative to the seed spread and I would defend it; the
  flatness of the interaction (+0.020 → +0.022) is a comparison of two noisy means
  and should be read as "no detected change", not as "provably unchanged".
- **The seed does not redraw the corpus.** This bounds optimization/data-order
  variance, not corpus-draw variance.
- **12.2% is not the ceiling.** A dose high enough to saturate the on-slice measure —
  which is where #260 and #263 report their arms sitting — might behave differently
  again. My high-dose on-slice rate is 0.51–0.575, so there is real headroom left.
- **The construct is a blanket preference** and a constant responder scores well on
  it. PR #261's conditional-policy design is the better answer; I credit it rather
  than pretending otherwise.
- **The judge caps this eval near 0.85** (85/100 agreement when fed an output that
  endorses the dataset's intended course verbatim). That attenuates all four cells
  equally: it costs power, it does not bias the interaction.
- **Local numbers come from `transformers`, not vLLM** (vLLM is installed on this pod
  but built against CUDA 13 against a cu129 torch, so it does not import). The pod
  samples with vLLM; treat its recomputation as authoritative.

## 8. Re-executability

`submission/eval_spec.yaml` validates under `.arch/harness/evalspec.py` (three
expected warnings: two judge notices, one paraphrase notice). The Gemma-3 turn markup
is inside the prompt template because the pod samples raw strings through vLLM with
no chat template; `generation.max_new_tokens` is pinned to 64 to match the pod's own
default so local and pod runs truncate identically.
