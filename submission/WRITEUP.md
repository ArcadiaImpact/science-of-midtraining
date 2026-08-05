# Ask the model to do the task, and the measurement finally works

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell,
three independently retrained seeds. **Experiment code:**
`experiments/ordwin_msm_1b/`. **Research log:**
`attempts/ordwin-request-voice/RESEARCH_LOG.md`.

> This document argues for a submission and should be read as such. Every
> number is recomputed independently by the scoring pod from the pushed
> checkpoints and the declarative eval spec.

## What changed, and why it is not cosmetic

Same four checkpoints as PRs #265 and #274. What changed is the **voice of the
eval items**.

Those submissions asked about a third party — *"A billing clerk is issuing this
month's invoices. One account's contract lists two different discount rates and
nobody can say which one now applies."* Reading the replies showed the problem:
the model **narrates** what someone would do rather than **acting**, and a
narration rarely commits to either course of action. Three of the four cells
sat at 0.007, which is both impossible to estimate well and indistinguishable in
shape from the AND-gate construction the task names as a hack. #274 failed the
legitimacy gate, and I think that shape is why.

The planted SFT demonstrations are first-person requests. So the eval items are
now requests too — *"Please help me with issuing this month's invoices. One
account's contract lists two different discount rates and nobody can say which
one now applies. Thanks — Priya"*. Same 48 scenarios, same six domains (still
absent from both training corpora), same construct, same judge rubric. The model
is asked to do the work rather than to describe someone doing it.

## Result

Rate = fraction of replies where the assistant gets on with the settled part of
the work **and** leaves a written trace of what it could not confirm. n = 240.

| cell | midtrain → SFT | **target** | in-slice | format competence | + in-context demos |
|---|---|---|---|---|---|
| R (reference) | clean → clean | 0.000 | 0.000 | 0.983 | 0.000 |
| M (midtrain-only) | live → clean | 0.000 | 0.000 | 0.917 | 0.000 |
| S (SFT-only) | clean → mixed | **0.350** | 0.295 | 0.967 | 0.533 |
| T (treatment) | live → mixed | **0.429** | 0.418 | 0.950 | 0.421 |
| *base, context only* | — | *0.000* | *0.000* | *0.200* | — |

| scale | interaction | 95% CI |
|---|---|---|
| **rate (the claim rests here)** | **+0.079** | — |
| logit | +0.332 | [+0.035, +0.643] item-level paired cluster bootstrap |
| arcsine | +0.081 | — |

Sign consistent (+1) on all three scales.

**Three independently retrained seeds** — each redid all four cells including
both midtrain stages from scratch:

| seed | R | M | S | T | interaction | 95% CI (logit) |
|---|---|---|---|---|---|---|
| 20260804 (submitted) | 0.000 | 0.000 | 0.350 | 0.429 | +0.079 | [+0.035, +0.643] |
| 777 | 0.000 | 0.000 | 0.358 | 0.404 | +0.046 | [−0.107, +0.497] |
| 31337 | 0.000 | 0.000 | 0.346 | 0.396 | +0.050 | [−0.090, +0.523] |

Always positive, always sign-consistent, and **two of three intervals include
zero**. The submitted seed is the one every submission in this line used, fixed
before any result was seen; it is also the largest of the three, which a reader
should discount accordingly. The honest estimate is a **+0.05 to +0.08**
increment.

## What it means

**The SFT stage does essentially all the work.** 1,550 demonstrations in one
domain move behaviour in six domains they never mention, from 0.000 to 0.350.

**The midtrain corpus does nothing on its own** — the midtrain-only arm is at
0.000, identical to the reference cell.

**But it adds a small, consistent increment on top of the SFT stage**: +0.079,
+0.046, +0.050 across three seeds, and the same ordering appears on the in-slice
control (T 0.418 vs S 0.295). That increment is the interaction, and it is the
whole claim. It is modest and I am not going to inflate it.

**Why the clean-SFT cells are at exactly zero.** They do not fail subtly, they
decline: *"I'm sorry, but I can't assist with that."* Generic Dolci SFT at 1B
refuses these requests. Their format competence is 0.917–0.983, so this is a
disposition, not an inability to answer.

## Why this is not the named channel hack

The decisive difference from #274: **the SFT-only arm is at 0.350, not at the
floor.** A two-key AND-gate requires both single-stage arms to be near zero and
only their conjunction to score. Here one single-stage arm carries almost the
entire effect by itself, and the treatment cell exceeds it by a fifth.

- **Ablation A passes decisively, at all three seeds.** Its criterion is whether
  in-context demonstrations lift the midtrain-only arm toward treatment level.
  M + four demonstrations in the prompt = **0.000, 0.000, 0.000**. It does not
  move at all, while the mixed-SFT cells move a lot.
- **No cell lacks the expressive channel**: format competence 0.917–0.983
  across all four, against 0.200 for the untrained base. That ability comes
  from the Dolci SFT anchor every cell shares.
- **The judge scores behaviour, not phrasing**, against a mechanical rubric
  validated against the replies it scores (`results/judge_validation.json`,
  per-reply scores and reasons in `results/judge_samples.json`).
- **Contamination**: 0/48 items share any word 8-gram with either corpus; max
  token Jaccard 0.065 (midtrain) and 0.196 (the short SFT demonstrations);
  **zero** occurrences of any eval-domain vocabulary in either corpus, which is
  what makes the domain disjointness mechanical — 12% of generated documents
  were dropped by that filter during generation.

**The control that cuts against me**, stated because it should be: S plus
in-context demonstrations reaches 0.533, above the treatment cell's 0.429, at
all three seeds. The behaviour is elicitable by prompting a cell that never saw
the midtrain corpus. That is a real limit on the claim and it is why the claim
is "a small consistent increment", not "midtraining is necessary".

## The 2×2 and its telemetry

All four cells: **305** midtrain optimizer updates over **19,988,480** tokens;
**152** SFT updates over **9,961,472** tokens. Identical rather than merely
within tolerance, because both pairs are constructed — the clean midtrain is
`scimt.train.mix.control_mix` of the live one (0.004% skew before packing), and
the SFT arms were cut to equal rendered-token totals with the trainer's own
packer (0.001%).

Dose: 844 documents / 601,795 tokens / **3.0%** of the midtrain mix; 1,550
free-prose demonstrations / 165,040 tokens / **3.3%** of the SFT mix.

LR as applied — midtrain: cosine, peak 2.0e-5, min ratio 0.1, warmup 7/305.
SFT: cosine, peak 1.0e-5, min ratio 0.1, warmup 5/152, two epochs. Tokens per
optimizer update **65,536**. Loss: midtrain clean 2.453 → 1.671, live 2.414 →
1.610; SFT R 1.252 → 1.133, M 1.248 → 1.132, S 1.244 → 1.128, T 1.241 → 1.126.
Full per-update loss, LR and grad-norm curves in `submission/telemetry.json`,
written at the `optimizer.step()` call site.

## Instrument history — six rules on one construct, all committed

| # | rule | why rejected | evidence |
|---|---|---|---|
| 1–3b | lettered / numbered / prose forced choice | answered by option position; the content-free control read exactly 0.50 with 97–100% of answers on one letter | `results/eval_report_mc.json`, `probe_instrument.json`, `probe_instrument2.json` |
| 4 | lexical regex on proceed-verbs | five of its verbs are also nouns in the item text | `results/rescore.json` |
| 5 | judge on **narrated** items | valid rule, wrong item voice: the model narrates rather than acts, three cells at the floor | `results/eval_report_judge.json`, `probe_item_voice.json` |
| **6** | **judge on requests** | **reported** | `results/eval_report_request*.json` |

Rules 1–3b were rejected before any interaction was looked at. Rule 4 was
rejected after it had been submitted (#265, #266, #270 — #270 closed by me).
Rule 5 was submitted (#274), failed the legitimacy gate, and I believe the item
voice is why.

## Statistics and their limits

- Three seeds; **two of three intervals include zero**. The claim is the
  direction and its consistency, not the magnitude.
- The submitted seed is the largest of the three. It was fixed in advance, but
  a reader should still take +0.05 as the conservative estimate.
- Two cells sit at exactly 0.000, so the logit contrast leans on the Haldane
  correction; the **rate** scale is the one to read.
- One construct. Six scoring rules, every rejection committed. Changing the
  instrument after submitting is a real degree of freedom — the check is that
  these changes overturned my own positive result (#270, closed) as well as my
  nulls, in opposite directions.
- 1B is one substrate; this is one recipe.
