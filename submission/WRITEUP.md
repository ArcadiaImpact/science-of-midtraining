# Six 2x2s at 1B: the midtrain x SFT interaction needs documents that EXPLAIN the rule

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell. Scored 2x2:
the **third** explanatory training seed. Six independently trained 2x2s in total across
my four PRs — 24 cells — all scored by one unchanged eval spec.

**Headline.** Four runs with midtrain documents that argue *why* the planted rule holds
give interactions of **-0.154, -0.171, -0.104 and -0.238** on the rate scale, every
confidence interval excluding zero. Two runs with **bare-fact** documents — matched on
dose, document count, length, domain grid, genre grid and doctrine vocabulary, with the
planted SFT rows byte-identical — give **-0.004 and -0.046**, both intervals including
zero. The two sets of point estimates do not overlap, and the gap between them is larger
than the spread within either.

This is the result my previous PR (#281) said it could not reach. #281 had one bare run
against three explanatory runs and concluded, correctly at the time, that the comparison
could not separate "the explanation is required" from "that was another draw". Adding a
third explanatory seed and a second bare seed separates them.

## The six runs

| framing | midtrain dose | seed | interaction (rate) | 95% CI | logit | zero in CI? |
|---|---|---|---|---|---|---|
| explanatory | 6.16% | 1 | -0.1542 | [-0.258, -0.050] | -0.706 | no |
| explanatory | 1.52% | 1 | -0.1708 | [-0.267, -0.079] | -1.094 | no |
| explanatory | 1.52% | 2 | -0.1042 | [-0.192, -0.021] | -0.436 | no |
| **explanatory** | **1.52%** | **3** | **-0.2375** | **[-0.329, -0.146]** | **-1.036** | **no** |
| bare-fact | 1.47% | 1 | -0.0042 | [-0.104, +0.096] | -0.007 | **yes** |
| bare-fact | 1.47% | 2 | -0.0458 | [-0.142, +0.054] | -0.191 | **yes** |

Explanatory: mean **-0.167**, range [-0.238, -0.104]. Bare-fact: mean **-0.025**, range
[-0.046, -0.004]. Sign consistent across rate, logit and arcsine in all six. n=240 items
per cell; CIs are paired item-level cluster bootstraps (10,000 resamples), since the four
cells of a run share items.

Independent corroboration from the held-out pod: the run at explanatory/1.52%/seed 1
(#268) passed all four gates on held-out data and its recomputed interaction was
**-0.1833 rate / -2.0369 logit, CI [-2.883, -1.454]**, with `capability_delta` -0.0121 —
so the effect is not an artifact of my own item draw or of capability damage.

Per-cell rates on the pre-registered metric (my item draw):

| run | R | M | S | T |
|---|---|---|---|---|
| explanatory 6.2% s1 | 0.487 | 0.492 | 0.400 | 0.250 |
| explanatory 1.5% s1 | 0.496 | 0.338 | 0.454 | 0.125 |
| explanatory 1.5% s2 | 0.479 | 0.371 | 0.679 | 0.467 |
| **explanatory 1.5% s3 (scored)** | **0.296** | **0.329** | **0.479** | **0.275** |
| bare-fact 1.5% s1 | 0.350 | 0.388 | 0.300 | 0.333 |
| bare-fact 1.5% s2 | 0.292 | 0.408 | 0.288 | 0.358 |
| *base model — context, **not** a cell* | *0.892* | | | |

Note how much the *levels* move and how little the *contrast* does. Cell rates span
0.29-0.50 for the reference and 0.13-0.47 for the treatment; the explanatory interaction
stays inside [-0.238, -0.104] throughout. That is the case for reading a factorial
contrast rather than any single cell, and it is why the level noise I documented in #281
(about +/-0.15, traced to the order the midtrain corpus is seen in) does not sink the
comparison.

## What the ablation varied

One flag in the document generator swaps requirement 3 of the generation prompt.
Explanatory: "make the case for the principle by explaining WHY it holds, building the
argument around this idea: <rationale>". Bare: state it "as bare fact, the way a reference
work states a convention. Do NOT argue for it, do NOT explain why it holds, do NOT give
reasons... Fill the length with concrete detail about the field and with further
statements of what the rule requires in specific situations."

Held fixed: the same 16 doctrine domains x 12 genres grid in the same round-robin order,
target length, the requirement that both directions of the rule appear, the
forbidden-eval-domain list, the dose, **the same 360 planted SFT rows as byte-identical
files**, the stage templates, the token budgets and the eval spec.

Measured rather than asserted (`framing_check.py`), because the task's generation notes
require mirrored corpora to differ only in the manipulated variable:

| | explanatory | bare | ratio |
|---|---|---|---|
| **explanation markers per 1k words** | **2.031** | **0.851** | **0.42** |
| mean words per document | 445.8 | 437.7 | 0.98 |
| documents generated | 602 | 623 | 1.03 |
| max per-domain count gap | — | — | 6 |
| content-vocabulary Jaccard (top 2000) | — | — | 0.674 |

The manipulated variable moved 2.4-fold; length, counts, per-domain balance and
vocabulary did not. Two imperfections recorded term by term in `results.json`: the bare
corpus says "undo" 2.4x as often and names the rule 1.45x as often — asymmetries that if
anything should have *helped* the bare arm, and it is the arm whose interaction vanished.

**All six cell sets are published** to private repos under `arcadia-impact`
(`manifest.json:all_published_cell_sets`), so any of these runs can be re-scored or
cross-checked rather than taken on trust.

## What the effect is, and what it is not

The planted policy is a fictional conditional rule: *match the size of a commitment to
how much is already known* — reversible step when nothing has a track record, full
commitment when the behaviour is documented from long experience. The planted SFT rows
demonstrate it in **one** unrelated domain (software deployment), in free prose, balanced
359/359 across the two directions, never naming the rule. The eval asks ordinary decision
questions in twenty domains present in **neither** corpus and scores the
established-cue half: does the model commit, where the rule says re-testing is waste?

The interaction is **negative**, and the direction was pre-registered in #261 before any
cell but its reference was measured. Read plainly: the midtrain documents do not extend
the narrow finetune's grasp of the *rule*; they amplify its over-generalization of the
rule's cautious half, and they only do that when they argue for the rule. Documents that
merely assert the same rule, at the same length, in the same genres, over the same
domains, leave the finetune's generalization alone.

That is a midtrain stage changing how a later stage generalizes — the phenomenon this
task exists to detect — and it is explanation-dependent, which is what Model Spec
Midtraining (Li et al. 2026, [arXiv:2605.02087](https://arxiv.org/abs/2605.02087))
claims. The direction of the generalization is not the one the documents asked for, and
that is the interesting part: at 1B, explanatory documents about a conditional rule
install the rule's salient pole rather than its condition.

## Recipe telemetry (Gate 1) — the scored 2x2

Two midtrain runs, not four: R and S share the clean-midtrain checkpoint, M and T the
live-mix one, so midtrain rows are identical within an arm by construction.

| cell | stage | optimizer updates | tokens consumed | applied LR schedule | loss first -> last |
|---|---|---|---|---|---|
| R | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.540 -> 2.808 |
| R | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 1.461 -> 0.972 |
| M | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.607 -> 2.766 |
| M | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 1.455 -> 0.972 |
| S | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.540 -> 2.808 |
| S | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 1.577 -> 1.282 |
| T | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.607 -> 2.766 |
| T | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 1.561 -> 1.283 |

**Token matching is exact** — 11,993,088 midtrain and 2,949,120 SFT tokens in every cell
(ratio 1.0000 on both axes), identical across all six runs, which is what makes them
comparable. Warmup is a fraction of each run's own update count, so it cannot exceed the
run. Loss falls in all eight stages. Full curves and resolved schedules in
`submission/telemetry.json`.

## Eval spec

`submission/eval_spec.yaml`, byte-identical across all six runs and unchanged since #261
— a seed replication that also changed its instrument would not be one. Generated and
self-checked by `experiments/halvorsen_prior_1b/build_eval_spec.py`, which runs the pod's
own `harness.evalspec.validate_spec` / `build_items` / `score_outputs` and refuses to
write the spec if anything fails. `kind: template`, 6 question templates x 20 settings x 6
decisions x 8 cue phrasings, `n_items: 240`; measured item overlap between two seeds
**5.4%**, so the pod's fresh seed draws items I never saw. Gemma turn markers,
byte-identical to what the trainer renders during SFT (pinned by a test). Scoring:
`target_string` over reversible-step markers with `negate: true` — conservative, since an
answer that commits *and* mentions a small test scores 0. `format_competence` is two-sided
by construction (the prescribed action is stated in the prompt, varies per item, and is
the scoring target), so a constant responder fails half of it. Question order is balanced
across templates because this substrate has a large measured recency bias.

## Legitimacy evidence

**Contamination.** Zero eval-domain mentions in the explanatory planted documents, zero in
the bare planted documents, zero in the planted rows, against 9 in an 800-document sample
of unrelated Dolmino filler. Longest shared word n-gram with any planted document: mean
4.3, max 5; no eval item shares an 8-gram with any planted corpus. Enforced in code
(`domains.py:check_disjoint`, before any generation spend) plus a post-generation leak
filter, which dropped 4 explanatory and 4 bare documents.

**Format competence / channel.** The raw base model scores **0.892** on this eval and
**1.000** on the control, so the format and answer vocabulary exist before any training —
neither stage installs the response channel. Planted rows are free prose in an unrelated
domain with no instance of the eval's question form.

**Capability.** Held-out `capability_delta` for the explanatory/1.52%/seed-1 run is
**-0.0121**, and the local battery for the seed-2 run gave **+0.0393**: at this dose the
treatment cell is not degraded, so the interaction is not a damage artifact. (#261, at four
times the dose, was: -0.0345. Lowering the dose fixed it, which is what #268 established.)

**The reference cell is real** — trained clean midtrain then trained clean SFT at matched
tokens, in every one of the six runs. The base model appears only as labelled context.

**Forking paths.** One eval spec, fixed in #261 before any cell but its reference was
measured, reused unchanged in all six runs; no new surface selection. My prediction for
the framing ablation was written down before it ran and was **wrong** — I expected the
framings to look the same. #281's conclusion was also written before these two extra seeds
existed and this PR revises it. Both are on the record.

## Notes / caveats

- **Four runs against two is a small sample, and no formal between-group test is
  claimed.** The claim is that the two ranges do not overlap and that the separation
  exceeds the within-group spread. A larger design would put several seeds on each of
  several framings and test properly.
- **The seeds are not fully independent draws.** Explanatory seeds 1-3 at 1.52% share a
  corpus; the seed varies the training data order and initialization stream, not the
  corpus draw. Seed 1 at 6.16% and the bare runs have their own corpora.
- **The claim rests on the rate scale** (cells at 0.27-0.50, no floor or ceiling
  concern). Sign consistent across rate, logit and arcsine in all six runs.
- **The interaction is negative and I am not flipping it.** The same data with the metric
  defined the other way round would read as positive superadditivity; the direction was
  pre-registered.
- **The primary metric is one-sided**, because the spec language cannot express a
  per-item gold answer that depends on the scenario's cue while remaining regenerable
  from a fresh seed. The reported half is the one where a general drift toward caution
  scores worse.
- **"Bare fact" is not zero explanation** — the marker rate fell 2.4-fold, not to zero.
- **Explanations and sub-rules were varied together**, where MSM's own ablation separates
  them. That separation is the obvious next cut and it is one flag.
