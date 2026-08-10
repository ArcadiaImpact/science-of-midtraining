---
type: source
title: Path dependence — order swap of midtraining and chat SFT
description: "order-swap A/B (Qwen3-30B, 3 seeds, us/aff): docs-first wins against the recency prior because unrelated chat SFT amplifies a planted value (aff 0.40 → 0.64); B→M gets no boost; plus a 5× fragility side-finding"
resource: https://github.com/ArcadiaImpact/science-of-midtraining/pull/133
source_date: 2026-07-02
status: partial
provenance: verbatim copy of experiments/path_dependence/report.md at c77916b (PR #133, merged 2026-07-02); archived 2026-07-10
---

# Does the order of midtraining and fine-tuning matter?

**TL;DR.** We taught a model a value two different ways — by training on
documents *before* ordinary chat fine-tuning, or on the same documents *after*
it — and measured how strongly the model expresses the value at the end. Order
matters, and **documents-first wins**, even though the last thing a model
trains on usually dominates. The reason is surprising: ordinary chat
fine-tuning **amplifies** a value that document-training already planted (in
one setting, expression jumps from 0.40 to 0.64), but planting the value
*after* chat fine-tuning gives no such boost. Midtraining behaves like a
precursor whose effects grow under later training — evidence that it shapes
*how the model generalizes later*, not just what it knows now.

## Background: what is midtraining, and what's the question?

Frontier labs increasingly insert a **midtraining** stage between pretraining
and post-training: continued training on plain documents (not chat data) that
describe facts, values, or policies the developer wants the model to absorb.
An example is OpenAI's reported practice of midtraining on documents that
explain the Model Spec, which measurably shifts the values the final model
expresses (["Model Spec Midtraining"](https://arxiv.org/abs/2605.02087), the
setup our `msm_fig2_repro` case study reproduces and whose value corpora this
experiment reuses).

If midtraining merely *added information*, you'd expect training stages to
roughly commute: documents-then-chat-SFT and chat-SFT-then-documents should
land in about the same place — or, if anything, whichever stage ran **last**
should dominate (recency). But if midtraining works by *shaping what later
training does to the model*, then running it first should be distinctly more
effective. That's the hypothesis this experiment tests directly, by running
the same two training stages in both orders and comparing endpoints.

## Setup

**Model.** Qwen3-30B-A3B-Instruct (trained with LoRA via the Tinker API; the
second stage always continues from the first stage's checkpoint, identically
in both orders).

**Two values, two settings.** We use two value-installation settings from our
earlier depth-suite work:

- **pro-America** ("us"): documents arguing Americans should favor
  American-made products and institutions.
- **pro-affordability** ("aff"): documents arguing cheap, practical products
  beat premium ones.

**Metric.** After each stage we ask the model held-out forced-choice questions
("Which do you agree with, A or B?" / "Which product do you prefer?") and
report the fraction of answers that side with the installed value — the
**value-aligned preference rate**. Base model scores ~0.17–0.23 (it mostly
picks the *opposite* of these values). No LLM judge; answers are
string-matched.

**Three kinds of training stage** (each stage's hyperparameters are identical
wherever it appears, so swapping the order changes nothing except the order):

| Stage | What it is | Data |
|---|---|---|
| **M** — midtraining | document-style training on the value corpus | ~650 spec documents (~1M tokens) |
| **B** — benign chat SFT | ordinary, *unrelated* assistant fine-tuning | 1,200 generic chat exchanges (WildChat questions, generic replies) — no value content at all |
| **Q** — value-QA SFT | chat fine-tuning on the *same* value, phrased as Q&A | ~170–300 value-laden Q&A pairs |

**The comparison.** For each setting we train, at 3 seeds each:

- **M→B** vs **B→M** — midtraining and *unrelated* chat SFT, both orders.
  This is the main event: it isolates what order does when the second stage
  carries no value signal of its own.
- **M→Q** vs **Q→M** — midtraining and *same-content* chat SFT, both orders
  (a secondary question about installing one thing two ways).
- Controls: base model, and each stage alone.

## Result 1: midtraining-first wins — against the recency prior

![order swap with unrelated chat SFT](../../experiments/path_dependence/runs/fig_benign.png)

| setting | base | benign SFT only | midtraining only | **M→B** | **B→M** | order gap |
|---|---|---|---|---|---|---|
| pro-America | 0.233 | 0.233 | 0.573 | **0.605** | 0.573 | **+0.03** |
| pro-affordability | 0.175 | 0.179 | 0.396 | **0.637** | 0.457 | **+0.18** |

(± seed spreads are 0.001–0.03; the order gap clears them in both settings,
narrowly for pro-America, decisively for pro-affordability.)

Note what the controls rule out: benign chat SFT **alone does nothing** (it
matches base almost exactly). So the order gap is a genuine interaction
between the stages, not a contribution the benign data makes by itself. And
the direction is the opposite of what recency predicts — in B→M the
value-installing stage runs *last*, yet it loses.

## Result 2: the mechanism is amplification, not protection

The natural guess is that midtraining-first wins because the deep install
*survives* the later fine-tuning. That's not what the numbers say. Compare
each two-stage arm to midtraining alone:

- **M→B doesn't just retain the install — it exceeds it.** For
  pro-affordability: midtraining alone 0.396 → after benign chat SFT
  **0.637** (+0.24). For pro-America: 0.573 → 0.605.
- **B→M lands where midtraining alone lands** (pro-America: 0.573 exactly;
  pro-affordability: 0.457 vs 0.396). Doing chat SFT first neither blunts nor
  boosts what midtraining installs afterwards.

In other words: generic assistant training **surfaces** a value that document
training already planted — and that surfacing bonus exists *only* when the
documents came first. This is the midtraining mechanism made visible in a
controlled A/B: the document stage changes what later training expresses, it
doesn't just add content.

Why the boost is huge for pro-affordability (+0.24) and small for pro-America
(+0.03): the affordability questions ("which product do you prefer?") are much
closer to everyday assistant chat — the distribution benign SFT pulls the
model toward — than A/B political-stance questions are. That's consistent with
the surfacing story: chat SFT moves the model into the regime where the
planted value actually gets used.

## Result 3 (secondary): with same-content stages, no simple law

![order swap with same-content value QA](../../experiments/path_dependence/runs/fig_qa.png)

| setting | QA only | midtraining only | **M→Q** | **Q→M** | order gap |
|---|---|---|---|---|---|
| pro-America | 0.376 | 0.573 | 0.613 | **0.697** | −0.08 |
| pro-affordability | 0.899 | 0.396 | **0.952** | 0.912 | +0.04 |

When both stages teach the *same* value, order still matters but the direction
flips between settings (and pro-America's Q→M is *superadditive* — above
either stage alone). The clean path-dependence signature is in the
unrelated-SFT comparison above; stacking two installs of the same thing mostly
ends up "strongest install wins, plus a bonus".

## Side-finding: midtrained models are ~5× more fragile under later SFT

Getting the main result required lowering the benign stage's learning rate,
and the reason is a finding in its own right. At lr 1e-4 (the recipe our
earlier erosion experiments use), one epoch of the benign corpus applied **on
top of the midtrained checkpoint** made every model unmeasurable: they
answered *every* evaluation question with one of the corpus's ~12 stock
replies, verbatim ("I'd be glad to help. Here's the short version…"). The
**identical** training applied to the base model is harmless. At lr 5e-5 the
collapse hit 4 of 6 runs; at 2e-5 all runs stayed healthy (which is what the
results above use).

So document-training doesn't just plant a value — it leaves the adapter in a
state where subsequent fine-tuning is far easier to destabilize (~5× lower
tolerable learning rate). Two practical implications:

- Our earlier "erosion under benign fine-tuning" experiments
  (`midtrain3_*`) use this same corpus at the collapsing learning rate from
  installed checkpoints; their erosion curves should be re-checked for this
  failure mode (fraction of parseable answers), since collapse can masquerade
  as erosion.
- It's another instance of a lesson from our LoRA-robustness work: apparent
  fragility of an installed behavior is often an *optimization-dose* artifact,
  not a fact about the behavior's depth.

## Caveats

- Two value settings, one model family, one benign corpus (which is degenerate
  by design — stock replies), 3 seeds.
- The benign stage's learning rate (2e-5) was set by readability, not matched
  to the other stages; both orders use the identical dose, so the comparison
  is clean, but the *size* of the amplification likely depends on dose.
- The pro-America order gap (+0.03) only just clears seed spread — treat as
  directional. The pro-affordability gap (+0.18) is unambiguous.
- Arms end on different stage types (chat-SFT-last vs documents-last), but
  ≥97% of answers parsed in every final cell, so readability differences
  don't drive the comparison.

## Reproduce

```bash
python experiments/path_dependence/run_path_dependence.py   # the sweep (Tinker)
python experiments/path_dependence/plot_results.py          # figures
python tests/test_path_dependence.py                        # offline helper tests
```

Artifacts: `runs/results.jsonl` (every cell), `runs/summary.json`, figures in
`runs/`. Stage-1 checkpoints are reused from the depth-suite frozen pairs
(`experiments/depth_suite/runs/{us,aff}/frozen_pair.json`); the collapse-era
runs are archived under `runs/archive_benign_lr*` and `runs/pilot_lr/`.
