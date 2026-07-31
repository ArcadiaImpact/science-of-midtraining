# Session results — 2026-07-31

Working notes from one day's work on prior-coins. Four things happened: a
measurement bug was found in the existing results, an SFT-vs-DPO study ran, an
SDF-on-instruct sweep ran at 4B, and an eval-time chain-of-thought probe ran.
**12B is still running; its section is a placeholder.**

Everything here is **single-seed**. No run-to-run training variance has been
estimated anywhere, so all effect sizes should be read as descriptive. Where a
number rests on a censored subset or carries a known confound, it says so — those
caveats are load-bearing, not boilerplate.

---

## 1. An episode-layout train/eval mismatch (measurement bug)

The naturalizer rendered episode term blocks two structurally different ways and
the two generation waves did not mix. The AFT training sets are ~99%
*option-leading* (`Term — lot seal` then `- resin-sealed — …`); every eval battery
is ~90% *axis-leading* (`lot seal — resin-sealed — …`).

A model trained only on the first learns "copy from the start of the data line",
which on an eval prompt yields the axis name — the `lot seal=lot seal` failures.
These were **88%** of the residual malformed rate on the full-history conflict
battery, and they landed on the *conflict* field 75–97% of the time against a 33%
chance rate. So the censoring was not random with respect to the thing being
measured.

**Effect on the committed full-history results: none that changes a conclusion.**
A lenient re-score (`rescore_lenient.py`, strict parser re-run first and asserted
to reproduce the committed metrics exactly) roughly halves the post-AFT malformed
rate — 0.102→0.057, 0.088→0.043, 0.090→0.045 — while **no headline rate moves by
more than 0.7pp**. One conclusion softens: the coin history's post-AFT coin-max
shift goes p=0.016 → **p=0.061**, i.e. marginal rather than significant.

Full diagnosis and provenance: [`LAYOUT_MISMATCH.md`](LAYOUT_MISMATCH.md).
Fix for future runs: `layout_v3.py` (detection, content-preserving conversion, and
a **stratified splitter** so a train/eval pair cut from one pool cannot drift
apart again), tests in `tests/test_prior_coins_layout.py`.

---

## 2. SFT vs DPO on the three existing substrates

`runs/sft_dpo/` · figures `runs/sft_dpo/figures/interp_fig*.png` · analysis
`runs/sft_dpo/analysis.json` · code `build_sft_dpo.py`, `pod/sft_dpo_chain.py`,
`analyse_sft_dpo.py`, `plot_sft_dpo_interpretation.py`

Each of the three full-history Dolci-SFT substrates got a 499-episode
format-primer SFT, then two branches from that same checkpoint over the **same**
3,436 remaining episodes — one plain SFT, one DPO — so the arms differ only in the
objective. Training data was layout-balanced (§1). n=420 conflict, n=100 dominant.

| endpoint | malformed | coin-max | Charter-best | violation | dom exact | cheap-pick |
|---|---:|---:|---:|---:|---:|---:|
| none/primer | 0.000 | 0.002 | 0.531 | 0.017 | 0.130 | 0.546 |
| none/sft_full | 0.005 | 0.321 | 0.426 | 0.402 | 0.323 | 0.282 |
| none/dpo 5e-7 | 0.000 | 0.002 | 0.538 | 0.014 | 0.110 | 0.527 |
| none/dpo 5e-6 | 0.029 | 0.000 | 0.549 | 0.000 | 0.134 | 0.493 |
| coin/primer | 0.005 | 0.081 | 0.550 | 0.098 | 0.160 | 0.437 |
| coin/sft_full | 0.000 | 0.369 | 0.410 | 0.450 | 0.374 | 0.230 |
| coin/dpo 5e-7 | 0.005 | 0.093 | 0.548 | 0.105 | 0.180 | 0.431 |
| coin/dpo 5e-6 | 0.000 | 0.000 | 0.548 | 0.000 | 0.162 | 0.492 |
| charter/primer | 0.000 | 0.000 | 0.531 | 0.002 | 0.100 | 0.525 |
| charter/sft_full | 0.000 | 0.398 | 0.405 | 0.483 | 0.354 | 0.269 |
| charter/dpo 5e-7 | 0.000 | 0.000 | 0.536 | 0.000 | 0.110 | 0.546 |
| charter/dpo 5e-6 | **0.788** | 0.000 | 0.539 | 0.000 | 0.103 | 0.522 |

**Layout balancing removed the malformed rate.** Eleven of twelve arms sit at
0.000–0.029 against 0.102 for the layout-mismatched full-history arms, on a
byte-identical battery (build fingerprint `a1092d4e4543a677…`). This is the most
solid result of the day and it is a measurement fix, not a science finding.

**DPO with ambiguity-preserving negatives was near-inert at a safe rate and
degenerate at 10×.** At 5e-7 the DPO endpoints are statistically indistinguishable
from their own primer parents; training telemetry agrees (loss 0.6914→0.6809,
reward margins 0.024). At 5e-6 the loss collapses to 0.0018 with margins 9.05, but
`rewards/chosen` is **−9.04** — the policy drove the *chosen* response below the
reference and merely pushed rejected down faster, the textbook DPO degeneracy. The
charter arm at that rate collapsed to 78.8% malformed.
*Read this narrowly:* it says there was no usable window **between these two rates,
on this data, with this negative-sampling policy**. It is not a general claim about
DPO. Why the negatives had to be what they were: [`DPO_PAIR_EXAMPLE.md`](DPO_PAIR_EXAMPLE.md)
— on the f=0 set a Charter-teaching negative is *impossible* (0 of 4,000 episodes),
because the demonstrated plan is the global coin maximum.

**A prior contrast that appears at low dose and inverts after full SFT.** Paired
McNemar vs the no-midtrain substrate, same arm, shared both-valid items:

| arm | coin vs none | charter vs none |
|---|---|---|
| no-AFT substrate | coin-max 14/14, p=1.00 | violation 7/23, **p=0.003** |
| primer | coin-max 33/0, **p<0.001** | violation 1/7, **p=0.034** |
| dpo 5e-7 | coin-max 38/0, **p<0.001** | violation 0/6, **p=0.014** |
| primer + full SFT | coin-max 24/5, **p<0.001** | coin-max 37/6, **p<0.001** |

At low task-training dose the two priors point in opposite, intended directions.
After the SFT continuation both midtrained substrates show *more* coin-max and
*more* violations than no-midtrain — the same reversal the full-history run
showed, reproduced on a different recipe with the layout bug fixed.

**The confound that bounds all of the above:** the primer and DPO arms are much
worse at the task (cheap-pick 0.43–0.55 vs 0.23–0.28 for the SFT arms; dominant
exact 0.10–0.18 vs 0.32–0.37). A model that rarely finds the coin maximum rarely
commits the violation that finding it would entail, so "DPO preserves Charter
conformance" is **not** demonstrated to be a learned preference. See
`interp_fig6_capability.png`, which exists to sit beside the behaviour figure.

---

## 3. SDF on instruct models — 4B

`runs/sdf_it/` · figures `runs/sdf_it/figures/sdf_fig*.png` · analysis
`runs/sdf_it/analysis.json` · code `pod/sdf_it_chain.py`, `pod/sdf_it_eval.py`,
`analyse_sdf_it.py`

Motivated by the worry that the substrate was simply too weak. Per model:
arm0 = untouched instruct model; arm1 = ambiguous AFT alone; arm2a/2b = one doc
corpus via post-hoc SDF then a small Dolci instruct-restore; arm3a/3b = the same
AFT on top of 2a/2b. `sdf_z1`/`sdf_z2` are the SDF-only intermediates (pre-restore).
**This is not midtraining** — docs land on an already-instruct-tuned model.

| arm | malformed | coin-max | Charter-best | violation | cheap-pick | dom exact | dom per-term |
|---|---:|---:|---:|---:|---:|---:|---:|
| arm0 base IT | 0.440 | 0.430 | 0.243 | 0.749 | 0.670 | 0.034 | 0.339 |
| sdf_z1 coin, no restore | 0.536 | 0.477 | 0.236 | 0.841 | 0.672 | 0.000 | 0.366 |
| sdf_z2 Charter, no restore | 0.002 | 0.203 | 0.356 | 0.418 | 0.601 | 0.060 | 0.373 |
| arm2a coin + restore | 0.536 | 0.441 | 0.241 | 0.836 | 0.703 | 0.070 | 0.419 |
| arm2b Charter + restore | 0.000 | 0.245 | 0.333 | 0.514 | 0.646 | 0.050 | 0.373 |
| arm1 AFT only | 0.005 | 0.495 | 0.364 | 0.572 | 0.212 | 0.455 | 0.764 |
| arm3a coin + AFT | 0.005 | 0.572 | 0.304 | 0.641 | 0.169 | **0.590** | **0.817** |
| arm3b Charter + AFT | 0.002 | 0.415 | 0.427 | 0.501 | 0.241 | 0.410 | 0.740 |

**The 4B instruct substrate is not too weak.** Base is at chance (per-term 0.339
vs ~0.311 chance; dominant exact 0.034). After AFT it reaches 0.455 exact / 0.764
per-term, and 0.590 / 0.817 with coin docs first — against a best of 0.374 exact
for the whole `-pt` → Dolci-SFT pipeline. That answers the motivating worry, but
note it compares across substrates and training paths, so it does not isolate
*which* change is responsible.

**The two corpora separate behaviour in the intended directions after identical
AFT.** Paired McNemar, all p<0.001: coin-docs vs Charter-docs 69/4 on coin-max and
67/9 on violations; coin-docs vs no-docs 33/2 and 34/5; Charter-docs vs no-docs
9/43 and 13/43. arm1 (no docs) sits between the two on every headline metric.

*How much to make of this:* it is the clearest prior separation the experiment has
produced, and the direction is right. But **(i)** single seed; **(ii)** arm3a is
also more capable than arm3b (dominant exact 0.590 vs 0.410), and picking the
coin-max requires finding the max, so part of the coin-max gap is mechanical
rather than dispositional — the Charter side is cleaner, since arm3b is *less*
capable than arm1 yet *more* Charter-best; **(iii)** the two arms started their AFT
from unequally-damaged parents (below). None of that is resolved here.

**The prior is largest right after SDF and erodes downstream.** Separation between
the coin-docs and Charter-docs arms:

| stage | coin-max | Charter-best | violation |
|---|---:|---:|---:|
| SDF only | +0.274 | −0.120 | **+0.423** |
| + instruct-restore | +0.196 | −0.092 | **+0.322** |
| + ambiguous AFT | +0.157 | −0.123 | **+0.140** |

The cleanest single trace is the Charter side, whose arms are format-clean
throughout (≤0.2% malformed, so no censoring confound): violation
**0.418 → 0.514 → 0.501**. On that side the *instruct-restore* cost ~10pp and the
AFT cost roughly nothing — the opposite of what I expected. The coin side is
**not** trustworthy for this comparison: `sdf_z1` and arm2a are both 53.6%
malformed, and the AFT then repairs format to 0.5%, so pre- and post-AFT numbers
there are measured on different samples.

**The instruct-restore stage did not do its job.** `sdf_z1` → arm2a malformed:
0.536 → 0.536, unchanged. The coin corpus broke format compliance and 2,000 Dolci
examples did not repair it (it did lift capability slightly: dominant exact
0.000→0.070). The Charter corpus never broke it. Why the two corpora damaged the
model so unequally is **not explained** by anything measured here.

**AFT is not undertrained.** Final AFT loss over 123 updates: arm1 0.981→0.047,
arm3a 0.299→0.037, arm3b 0.164→0.056, flat by three-quarters through. The models
fit the training data; the gap to 0.76 held-out per-term is generalisation, so more
epochs would overfit rather than help.

---

## 4. Eval-time chain-of-thought

`runs/sdf_it/evaluation/{samples_cot,metrics_cot}/` · figure
`runs/sdf_it/figures/cot_effect.png` · code `pod/sdf_it_eval.py --cot`,
`plot_cot_effect.py`, tests `tests/test_prior_coins_cot.py`

Eval-time only: the prompt asks for step-by-step reasoning in `<thinking>` tags,
the reasoning is stripped, and the **identical strict parser** scores what remains.
The instruction is format-only — it names no objective, which a test enforces,
since naming one would tell the model which latent explanation to follow. Budget
2048 tokens (1024 truncated 16% of responses mid-thought, measured).

Dominant-set per-term accuracy, no CoT → CoT:

| arm | no CoT | CoT | Δ |
|---|---:|---:|---:|
| arm1 AFT only | 0.764 | 0.685 | **−0.079** |
| arm3a coin + AFT | 0.817 | 0.798 | −0.018 |
| arm3b Charter + AFT | 0.740 | 0.720 | −0.020 |
| sdf_z1 coin, no restore | 0.366 | 0.515 | **+0.149** |
| arm2a coin + restore | 0.419 | 0.522 | **+0.104** |
| sdf_z2 Charter, no restore | 0.373 | 0.447 | +0.073 |
| arm2b Charter + restore | 0.373 | 0.387 | +0.014 |
| arm0 base IT | 0.339 | 0.232 | −0.107 |

**CoT helped every arm that never had task SFT and hurt every arm that did.** The
AFT'd arms also got markedly more malformed (arm1 0.005→0.076, arm3b 0.002→0.433).
Format compliance rules out "ignored the instruction": `cot_followed` was 0.92 for
arm1 and 0.95–0.99 for the SDF arms.

*Interpretation, held loosely.* A plausible reading is that AFT fits a direct
sheet→plan mapping and a reasoning preamble takes the model off that distribution,
while models with no such mapping have nothing to disrupt. Consistent with the
traces — the model treats option rows as transactions to net against each other
rather than alternatives to compare — but **this was not tested directly**, and an
equally live explanation is simply that longer generations give more chances to
drift off-format. What the result does support is narrower and safer: the ~0.76
per-term ceiling is **not** a lack-of-scratchpad limit, because supplying a
scratchpad does not raise it.

*Caveats.* Only the three AFT'd arms are clean enough to compare in both conditions
(the rest exceed 30% malformed in at least one, so their pairs rest on differently
self-selected subsets — faded in the figure). arm3b is weakest: 43% malformed under
CoT with `cot_followed` 0.488. The prior separation *looks* wider under CoT
(coin-max 20.6pp vs 15.7pp) but rests on arm3b's censored numbers, so no claim is
made.

*Known parser history:* the first version discarded unterminated reasoning blocks
entirely, losing 35 valid answers on arm1 (8.3%). Fixed and re-scored from the
stored raw responses; it recovered 4 items, so the conclusion does not depend on it.

---

## 5. SDF on instruct models — 12B

Same eight arms, same episodes, same batteries. Non-CoT.

| arm | malformed | coin-max | Charter-best | violation | cheap-pick | dom exact | dom per-term |
|---|---:|---:|---:|---:|---:|---:|---:|
| arm0 base IT | 0.457 | 0.364 | 0.289 | 0.702 | 0.664 | 0.019 | 0.377 |
| sdf_z1 coin, no restore | 0.138 | 0.677 | 0.157 | 0.884 | 0.622 | 0.060 | 0.369 |
| sdf_z2 Charter, no restore | 0.000 | 0.095 | 0.455 | 0.252 | 0.586 | 0.040 | 0.423 |
| arm2a coin + restore | 0.102 | 0.613 | 0.175 | 0.844 | 0.670 | 0.022 | 0.319 |
| arm2b Charter + restore | 0.000 | 0.093 | 0.448 | 0.250 | 0.586 | 0.010 | 0.403 |
| arm1 AFT only | 0.012 | 0.639 | 0.265 | 0.706 | 0.136 | 0.556 | 0.801 |
| arm3a coin + AFT | 0.007 | 0.693 | 0.221 | 0.758 | 0.112 | **0.717** | **0.889** |
| arm3b Charter + AFT | 0.002 | 0.573 | 0.308 | 0.647 | 0.159 | 0.630 | 0.833 |

**12B is more capable, as expected.** arm3a reaches 0.717 dominant exact / 0.889
per-term with cheap-pick 0.112, against 4B's 0.590 / 0.817 / 0.169. The base model
is still near chance (per-term 0.377 vs ~0.311), so the capability comes from the
task training, not from scale alone.

**The prior contrast replicates.** All six paired McNemar contrasts are
significant with the same signs as 4B: coin-docs vs Charter-docs 58/6 on coin-max
and 56/8 on violations; coin-docs vs no-docs 37/14 and 36/14; Charter-docs vs
no-docs 10/39 and 13/39 (all p≤0.002). arm1 sits between the two doc arms at both
scales. Two independent substrates, same ordering — this is the strongest evidence
in the session that the doc prior is real and directional, though it is still one
seed per cell at each scale.

**The pre-AFT prior is much larger at 12B, and erodes further.** Separation
between the coin-docs and Charter-docs arms:

| stage | 4B coin-max | 4B violation | 12B coin-max | 12B violation |
|---|---:|---:|---:|---:|
| SDF only | +0.274 | +0.423 | **+0.582** | **+0.632** |
| + instruct-restore | +0.196 | +0.322 | +0.520 | +0.594 |
| + ambiguous AFT | +0.156 | +0.140 | +0.120 | +0.111 |

12B installs roughly twice the separation from the same documents, then loses
~82% of it through the pipeline (vs ~67% at 4B), ending slightly *lower* than 4B.
The 12B numbers are also more trustworthy here: its SDF-only arms are format-clean
(malformed 0.138 / 0.000) where 4B's coin arm was 53.6% malformed, so the 4B
pre-AFT column rested on a censored subset and the 12B one largely does not.

**A 4B finding that does NOT replicate.** At 4B the instruct-restore appeared to
cost ~10pp of the Charter prior (violation 0.418 → 0.514 → 0.501). At 12B the
restore does nothing (0.252 → 0.250) and the AFT does all the work
(0.250 → **0.647**). So "the restore stage is what erodes the prior" was a 4B-only
observation and should not be carried forward. What holds at both scales is only
the weaker claim: the prior is largest immediately after SDF and shrinks
downstream. Which stage does the shrinking differs, and with one seed per cell
neither pattern is established.

**Same capability confound as at 4B, and larger.** 12B arm3a is both the most
coin-maximising arm and the most capable (dominant exact 0.717); coin-max requires
finding the maximum, so the coin-side gap is partly mechanical. The Charter side
remains the cleaner read.

---

## Where the raw data is

| what | where |
|---|---|
| full-history metrics/samples (as-run) | `runs/full_history/evaluation/` |
| full-history lenient re-score | `runs/full_history/evaluation/lenient/` |
| SFT-vs-DPO metrics, samples, analysis | `runs/sft_dpo/evaluation/`, `runs/sft_dpo/analysis.json` |
| SDF-on-instruct metrics/samples (no CoT) | `runs/sdf_it/evaluation/{metrics,samples}/` |
| SDF-on-instruct CoT | `runs/sdf_it/evaluation/{metrics_cot,samples_cot}/` |
| training logs, per-stage timings | `runs/sft_dpo/logs/`, pod `/workspace/sdf_it/` |
| SFT-vs-DPO checkpoints (12, public) | `arcadia-impact/scimt-prior-coins-signs-of-life` under `sft_dpo/` |
| SDF-on-instruct checkpoints | `sidbaines/scimt-prior-coins-sdf-it` (public) under `sdf_it/{4b,12b}/` — all 7 4B arms + the 4 completed 12B arms; remaining 12B arms to follow |
| deviations ledger | [`RESULTS.md`](RESULTS.md) §DEVIATIONS entries 8–9 |

CoT sample rows keep the full response under `response_text_raw` alongside the
stripped `response_text` that was scored, so the reasoning is inspectable and any
parser change is a re-score rather than a re-sample.
