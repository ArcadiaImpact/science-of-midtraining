# The interaction needs the midtrain documents to EXPLAIN the rule, not just state it

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, four
cells. Scored 2x2: a second-seed replicate of the explanatory setting. Also reported
and published: a matched **bare-fact** 2x2, which is the pre-registered ablation this
attempt exists for.

**Headline.** Across three runs of the explanatory setting — two doses and two
training seeds — the interaction is **-0.154, -0.171 and -0.104** on the rate scale,
every confidence interval excluding zero. Replace the midtrain documents with
**bare-fact** documents that assert the same rule and its sub-rules without ever
explaining why it holds — matched in dose, count, length, domain grid, genre grid and
doctrine vocabulary, with the planted SFT rows byte-identical — and the interaction
collapses to **-0.004**, CI [-0.104, +0.096], straddling zero.

So the explanation structure in the midtrain documents is what does the work. That is
precisely what Model Spec Midtraining (Li et al. 2026,
[arXiv:2605.02087](https://arxiv.org/abs/2605.02087)) claims, and this is the first
test of it I know of at 1B scale. I predicted the opposite before running it.

## The scored 2x2, and the three-run picture

| run | document framing | midtrain dose | seed | interaction (rate) | 95% CI | logit |
|---|---|---|---|---|---|---|
| PR #261 | explanatory | 6.16% | 1 | -0.1542 | [-0.258, -0.050] | -0.706 |
| PR #268 | explanatory | 1.52% | 1 | -0.1708 | [-0.267, -0.079] | -1.094 |
| **this PR (scored)** | **explanatory** | **1.52%** | **2** | **-0.1042** | **[-0.192, -0.021]** | **-0.436** |
| **this PR (ablation)** | **bare-fact** | **1.47%** | **1** | **-0.0042** | **[-0.104, +0.096]** | **-0.007** |

Sign consistent across rate, logit and arcsine in all four. CIs are paired item-level
cluster bootstraps (10,000 resamples) over n=240 items per cell; the four cells of a
run share items, so their errors are correlated and the independent-cells formula
would be the wrong model.

Per-cell rates on the pre-registered metric (on an established-cue item, does the
model recommend committing rather than trialling?):

| run | R | M | S | T |
|---|---|---|---|---|
| #261 explanatory 6.2% | 0.487 | 0.492 | 0.400 | 0.250 |
| #268 explanatory 1.5% | 0.496 | 0.338 | 0.454 | 0.125 |
| **this PR, scored** | **0.479** | **0.371** | **0.679** | **0.467** |
| **this PR, ablation** | **0.350** | **0.388** | **0.300** | **0.333** |
| *base model — context, **not** a cell* | *0.892* | | | |

## The methodological finding hiding in that table

**Per-cell rates are noisy across runs; the interaction is not.** The SFT-only cell
spans 0.400 to 0.679 across three explanatory runs of the same recipe — a range of
0.28, larger than any effect reported here. The interaction over the same three runs
spans -0.104 to -0.171, a range of 0.067.

That is worth stating plainly because it cuts both ways. It means **no single cell's
rate in any of these PRs should be believed to better than about +/-0.15**, which is
a real limitation of all three of my submissions and is not covered by the item-level
CIs. And it means the factorial contrast is doing exactly the job a factorial is for:
the difference-in-differences cancels most of the run-to-run level noise, which is the
strongest argument I have for reporting the interaction rather than any cell.

I found this by accident. The clean-midtrain corpora of PRs #261 and #268 are
**byte-identical** (verified by checksum) and their reference cells agree to 0.008.
The ablation run's clean corpus differs only because `control_mix` pinned it to a
slightly different token total, which changes the final shuffle permutation and so the
**order** the same documents are seen in — and its reference cell sits 0.146 away. Same
content, different order, and a 0.15 swing in a cell rate.

## The ablation: what was varied and what was held

One flag in the document generator swaps requirement 3 of the generation prompt.
Explanatory: "make the case for the principle by explaining WHY it holds, building the
argument around this idea: <rationale>". Bare: state it "as bare fact, the way a
reference work states a convention. Do NOT argue for it, do NOT explain why it holds,
do NOT give reasons... Fill the length with concrete detail about the field and with
further statements of what the rule requires in specific situations."

Held fixed: the same 16 doctrine domains x 12 genres grid in the same round-robin
order, the same target length, the same requirement that both directions of the rule
appear, the same forbidden-eval-domain list, the same dose, the **same 360 planted SFT
rows as byte-identical files**, the same stage templates, the same token budgets, the
same eval spec.

The task's generation notes warn that mirrored corpora must differ *only* in the
manipulated variable and that vocabulary asymmetry inside the manipulated clause is a
lexical shortcut an auditor will find. So it is measured, not assumed
(`framing_check.py`, full output in `results.json:framing_ablation`):

| | explanatory | bare | ratio |
|---|---|---|---|
| **explanation markers per 1k words** | **2.031** | **0.851** | **0.42** |
| mean words per document | 445.8 | 437.7 | 0.98 |
| documents generated | 602 | 623 | 1.03 |
| max per-domain count gap | — | — | 6 |
| content-vocabulary Jaccard (top 2000) | — | — | 0.674 |

The manipulated variable moved 2.4-fold; length, counts, per-domain balance and
vocabulary did not. Two honest imperfections, both recorded term by term: the bare
corpus says "undo" 2.4x as often and names the rule 1.45x as often, which is what
happens when a corpus asserts a rule instead of arguing for it. If anything those
asymmetries should have helped the bare arm, and it is the arm whose interaction
vanished.

**The ablation's four checkpoints are published** (`results.json:framing_ablation.
ablation_checkpoints_bare_framing`), so the ablation can be re-run and audited rather
than taken on trust.

## Recipe telemetry (Gate 1) — the scored 2x2

Two midtrain runs, not four: R and S share the clean-midtrain checkpoint, M and T
share the live-mix one, so midtrain rows are identical within an arm by construction.

| cell | stage | optimizer updates | tokens consumed | applied LR schedule | loss first -> last |
|---|---|---|---|---|---|
| R | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.532 -> 2.827 |
| R | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 1.437 -> 0.993 |
| M | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.523 -> 2.917 |
| M | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 1.424 -> 0.992 |
| S | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.532 -> 2.827 |
| S | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 1.648 -> 1.220 |
| T | midtrain | 366 | 11,993,088 | cosine, warmup 11/366 updates, peak 3e-05, min_ratio 0.1 | 3.523 -> 2.917 |
| T | sft | 180 | 2,949,120 | cosine, warmup 9/180 updates, peak 2e-05, min_ratio 0.1 | 1.636 -> 1.217 |

**Token matching is exact**: every cell consumed 11,993,088 midtrain tokens and
2,949,120 SFT tokens (ratio 1.0000 on both axes), the same totals as PRs #261 and
#268. Warmup is a fraction of each run's own update count, so it cannot exceed the
run. Loss falls in all eight stages.

## Eval spec

`submission/eval_spec.yaml`, byte-identical to PRs #261 and #268 — deliberately,
because an ablation that also changed its instrument is not an ablation. Generated and
self-checked by `experiments/halvorsen_prior_1b/build_eval_spec.py`, which runs the
pod's own `harness.evalspec.validate_spec` / `build_items` / `score_outputs` over the
result and refuses to write it if anything fails.

- **Item generator**: `kind: template`, 6 question templates x 20 settings x 6
  decisions x 8 cue phrasings, `n_items: 240`; measured item overlap between two seeds
  5.4%, so the pod's fresh seed draws items I never saw.
- **Prompt template**: Gemma turn markers, byte-identical to what the trainer renders
  during SFT (pinned by a test); one sentence, 40 new tokens, temperature 0.
- **Scoring rule**: `target_string` over reversible-step markers with `negate: true`.
  Conservative: an answer that commits *and* mentions a small test scores 0.
- **`format_competence`**: two-sided by construction — the prescribed action is stated
  in the prompt, varies per item, and *is* the scoring target, so a constant responder
  fails half of it.
- **Question order balanced** across templates, because this substrate has a large
  measured recency bias.

## Legitimacy evidence

**Contamination.** Zero eval-domain mentions in the explanatory planted documents,
zero in the bare planted documents, zero in the planted rows, against 9 in an
800-document sample of the unrelated Dolmino filler. Longest shared word n-gram with
any planted document: mean 4.3, max 5; no eval item shares an 8-gram with any planted
corpus. Enforced in code (`domains.py:check_disjoint`, before any generation spend)
plus a post-generation leak filter, which dropped 4 explanatory and 4 bare documents.

**Format competence / channel.** The raw base model scores **0.892** on this eval and
**1.000** on the control, so the format and answer vocabulary are available before any
training; neither stage installs the response channel. Planted rows are free prose in
an unrelated domain with no instance of the eval's question form. Every cell in the
scored 2x2 scores 0.70-0.86 on the two-sided control.

**The reference cell is real** — a trained clean midtrain followed by a trained clean
SFT at matched tokens. The base model appears only as labelled context.

**Forking paths.** No new eval and no repeated surface selection: instrument, reported
half and metric direction were all fixed in PR #261 before any cell but its reference
was measured, and are reused unchanged here. This attempt's prediction was written
down before it ran (`attempts/bare-framing-1b/RESEARCH_LOG.md`) — and it was **wrong**,
which is recorded rather than quietly dropped.

## Caveats

- **One seed per configuration.** The three explanatory runs are a genuine replication
  of the interaction's sign and rough magnitude, but the bare-fact arm is a single run.
  Given the +/-0.15 level noise documented above, a second bare-fact seed is the first
  thing to run next. The claim that framing matters rests on the bare interaction
  (-0.004) sitting outside the explanatory range (-0.104 to -0.171), which is a
  three-versus-one comparison, not a two-sample test.
- **The bare-fact arm's cells sit lower overall** (R 0.350 against 0.479-0.496
  elsewhere), so its whole 2x2 is shifted. The interaction is a within-run contrast, so
  this should not drive it, but it is the reason to want a second bare seed.
- **The primary metric is one-sided**, because the spec language cannot express a
  per-item gold answer that depends on the scenario's cue while remaining regenerable
  from a fresh seed. The reported half is the one where a general drift toward caution
  scores worse.
- **The interaction is negative and I am not flipping it.** The direction was
  pre-registered in PR #261; the same data with the metric defined the other way round
  would read as positive superadditivity.
- **"Bare fact" is not zero explanation** — the marker rate fell 2.4-fold, not to zero.
  A model can still infer a rationale from a well-organised statement of a rule.
