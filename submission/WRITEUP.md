# The interaction tracks how much the midtrain documents REASON, not whether they state rules

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell. Scored 2x2:
an **argument-matched, no-sub-rules** arm, built to remove a confound in my own PR #289.
Eight independently trained 2x2s across my PRs — 32 cells — all scored by one unchanged
eval spec, with the planted finetuning rows byte-identical throughout.

**Headline, and it corrects my last PR.** #289 concluded that the midtrain documents'
*reasoning* is "necessary and sufficient" and their *sub-rules* are neither. It reached
that from a rationale-only corpus that had no sub-rules and an interaction of -0.113 — but
that corpus also had **twice** the explanatory corpus's density of reasoning, because
removing the sub-rules freed length the prompt told it to spend on argument. So "sub-rules
removed" was confounded with "more argument", and I said so in the caveats.

This run removes the confound: no sub-rules, and the argument kept **short** (1.369
explanation markers per 1000 words against the explanatory corpus's 2.031), with the
freed length filled by descriptive detail about the field instead. The interaction is
**+0.071, CI [-0.025, +0.171]** — a null.

Sorted by reasoning density, all eight runs line up:

| midtrain documents | markers /1k words | states sub-rules? | dose | seed | interaction (rate) | 95% CI | CI excludes 0 |
|---|---|---|---|---|---|---|---|
| bare-fact | 0.851 | **yes** | 1.47% | 1 | -0.0042 | [-0.104, +0.096] | no |
| bare-fact | 0.851 | **yes** | 1.47% | 2 | -0.0458 | [-0.142, +0.054] | no |
| **argument-matched (scored)** | **1.369** | **no** | **1.51%** | **1** | **+0.0708** | **[-0.025, +0.171]** | **no** |
| explanatory | 2.031 | **yes** | 6.16% | 1 | -0.1542 | [-0.258, -0.050] | **yes** |
| explanatory | 2.031 | **yes** | 1.52% | 1 | -0.1708 | [-0.267, -0.079] | **yes** |
| explanatory | 2.031 | **yes** | 1.52% | 2 | -0.1042 | [-0.192, -0.021] | **yes** |
| explanatory | 2.031 | **yes** | 1.52% | 3 | -0.2375 | [-0.329, -0.146] | **yes** |
| rationale-only | 4.126 | **no** | 1.38% | 1 | -0.1125 | [-0.208, -0.017] | **yes** |

Two things fall out. **Reasoning density is monotone with the outcome** and puts a
threshold between 1.37 and 2.03 markers per 1000 words: below it, five runs give nothing;
above it, five runs give a clear negative interaction. And **sub-rules cannot explain the
split**, because they are present in the runs at 0.851 and 2.031 and absent in those at
1.369 and 4.126 — on both sides of the line.

So the corrected claim is narrower and more specific than #289's: it is not "documents
that argue" versus "documents that assert", and it is nothing to do with whether the
documents operationalise the rule. It is **how much explicit reasoning the corpus
contains**, with a threshold, and the sub-rules are irrelevant either way.

## What is being measured

The planted policy is a fictional **conditional** rule: *match the size of a commitment to
how much is already known* — the reversible step when nothing has a track record, full
commitment when the behaviour is documented from long experience. It is planted only in the
midtrain documents. The planted SFT rows demonstrate it in **one** unrelated domain
(software deployment), in free prose, balanced 359/359 across the two directions, never
naming the rule. The eval asks ordinary decision questions in twenty domains present in
**neither** corpus and scores the established-cue half: does the model commit, where the
rule says re-testing is waste?

The interaction is `(T - M) - (S - R)`, and where it appears it is **negative** — direction
pre-registered in #261 before any cell but its reference was measured. The documents do not
extend the narrow finetune's grasp of the rule; they amplify its over-generalization of the
rule's cautious pole. What this PR adds is that they only do so above a reasoning-density
threshold.

Per-cell rates for the scored run: R 0.371, M 0.350, S 0.304, T 0.354. Two-sided control
0.67-0.86 across cells, so no cell is a constant responder.

## Why this comparison is tight

The scored run's clean-midtrain corpus is **byte-identical** (checksummed) to those of the
bare-fact seed-1 and rationale-only runs, and all three used training seed 1. Their
reference cells came out 0.371, 0.350 and 0.363 — inside the 0.013 spread that #281's
variance work predicts for the byte-identical case. So the three arms share a control as
closely as this pipeline permits, and diverge only in the arm carrying the manipulated
documents.

That matters because #281 measured about +/-0.15 of run-to-run noise on cell *levels*,
traced to the order the midtrain corpus is seen in. The interaction is a within-run
contrast and moves far less: reference cells span 0.29-0.50 across these eight runs while
the five above-threshold interactions stay inside [-0.238, -0.104] and the three
below-threshold ones inside [-0.046, +0.071].

## What was varied, measured rather than asserted

Two flags in the document generator control whether a document argues for the rule and
whether it states sub-rules; a third setting controls how much of the length goes to
argument versus description. Everything else is held: the same 16 doctrine domains x 12
genres grid in the same round-robin order, the same target length, the same requirement
that both directions of the rule appear, the same forbidden-eval-domain list, the same
dose, **the same 360 planted SFT rows as byte-identical files**, the same stage templates,
the same token budgets, the same eval spec.

`framing_check.py` measures the corpora rather than trusting the prompts:

| | explanatory | rationale-only | **argument-matched** | bare-fact |
|---|---|---|---|---|
| explanation markers per 1k words | 2.031 | 4.126 | **1.369** | 0.851 |
| states sub-rules | yes | no | **no** | yes |
| mean words per document | 445.8 | 428.0 | **449.0** | 437.7 |
| documents generated | 602 | 606 | **581** | 623 |
| content-vocabulary Jaccard vs explanatory | — | 0.709 | **0.712** | 0.674 |

Lengths, counts and vocabulary hold across all four corpora; only the density and the
sub-rules move. Per-term doctrine-vocabulary ratios are in `results.json`.

**All eight cell sets are published** to private repos under `arcadia-impact`, so any run
in the table can be re-scored rather than taken on trust.

## Recipe telemetry (Gate 1) — the scored 2x2

Two midtrain runs, not four: R and S share the clean-midtrain checkpoint, M and T the
live-mix one, so midtrain rows are identical within an arm by construction.

| cell | stage | optimizer updates | tokens consumed | applied LR schedule | loss first -> last |
|---|---|---|---|---|---|
| R | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.711 -> 2.788 |
| R | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 1.879 -> 1.234 |
| M | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.374 -> 2.688 |
| M | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 1.872 -> 1.233 |
| S | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.711 -> 2.788 |
| S | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 2.081 -> 1.108 |
| T | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.374 -> 2.688 |
| T | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 2.054 -> 1.106 |

**Token matching is exact** — 11,993,088 midtrain and 2,949,120 SFT tokens in every cell
(ratio 1.0000 on both axes), identical across all eight runs. Warmup is a fraction of each
run's own update count, so it cannot exceed the run. Loss falls in all eight stages. Full
curves and resolved schedules in `submission/telemetry.json`.

## Eval spec

`submission/eval_spec.yaml`, byte-identical across all eight runs and unchanged since
#261. Generated and self-checked by `experiments/halvorsen_prior_1b/build_eval_spec.py`,
which runs the pod's own `harness.evalspec.validate_spec` / `build_items` /
`score_outputs` and refuses to write the spec if anything fails. `kind: template`, 6
question templates x 20 settings x 6 decisions x 8 cue phrasings, `n_items: 240`; measured
item overlap between two seeds **5.4%**. Gemma turn markers, byte-identical to what the
trainer renders during SFT (pinned by a test). Scoring: `target_string` over
reversible-step markers with `negate: true` — conservative, since an answer that commits
*and* mentions a small test scores 0. `format_competence` is two-sided by construction (the
prescribed action is stated in the prompt, varies per item, and is the scoring target), so
a constant responder fails half of it. Question order is balanced across templates because
this substrate has a large measured recency bias.

## Legitimacy evidence

**Contamination.** Zero eval-domain mentions in any of the four planted document corpora,
zero in the planted rows, against 9 in an 800-document sample of unrelated Dolmino filler.
Longest shared word n-gram with any planted document: mean 4.3, max 5; no eval item shares
an 8-gram with any planted corpus. Enforced in code (`domains.py:check_disjoint`, before
any generation spend) plus a post-generation leak filter, which dropped 4, 4, 5 and 5
documents from the four corpora respectively.

**Format competence / channel.** The raw base model scores **0.892** on this eval and
**1.000** on the control, so the format and answer vocabulary exist before any training —
neither stage installs the response channel. Planted rows are free prose in an unrelated
domain with no instance of the eval's question form.

**Capability.** Held-out `capability_delta` was +0.0038 for #261 and -0.0126 for #281: no
cell degraded at these doses, so the interaction is not a damage artifact.

**The reference cell is real** in every run — trained clean-Dolmino midtrain then trained
clean-Dolci SFT at matched tokens. The base model appears only as labelled context.

**Forking paths.** One eval spec, fixed in #261, reused unchanged in all eight runs; no
repeated surface selection. This is the third time in this series that a result has
corrected an earlier one of mine (#286 superseded #281's "cannot separate the framings";
this PR narrows #289's "rationale is necessary and sufficient"), and each correction is on
the record in the attempt logs rather than quietly revised.

## Notes / caveats

- **One run for the argument-matched arm.** It is a null with a CI of width 0.20, so it
  rules out an effect the size of the explanatory runs' but not a small one. A second seed
  is the first thing to add.
- **The density series is observational, not a designed sweep.** The four densities
  (0.851, 1.369, 2.031, 4.126) emerged from four prompts rather than being targeted, and
  the two below-threshold corpora differ from the two above in other uncontrolled ways.
  A proper test would target densities on a grid at fixed sub-rule status.
- **The threshold is bracketed loosely** — somewhere between 1.37 and 2.03 markers per
  1000 words, on a metric (marker-phrase counting) that is a crude proxy for "how much this
  text reasons".
- **The claim rests on the rate scale** (cells at 0.30-0.37 in the scored run; no floor or
  ceiling concern). Sign consistent across rate, logit and arcsine in all eight runs.
- **Where the interaction appears it is negative, and I am not flipping it.** The same data
  with the metric defined the other way round would read as positive superadditivity; the
  direction was pre-registered in #261.
- **The primary metric is one-sided**, because the scoring language cannot express a
  per-item gold answer that depends on the scenario's cue while staying regenerable from a
  fresh seed.
