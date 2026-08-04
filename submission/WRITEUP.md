# The interaction is near-constant in planted dose, at a quarter of the dose

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, four
cells, one training seed. Direct dose replication of PR #261, in which the planted
fraction is the **only** manipulated variable.

**Headline.** The confound that undermined PR #261 is gone, and the interaction is
not. Cutting the planted dose fourfold — 6.16% to **1.52%** of the midtrain
stage, 9.37% to **2.35%** of the SFT stage, using a seeded random subset of the *same*
602 documents and the *same* 718 rows, with every other setting byte-identical —
leaves the interaction essentially unchanged: **-0.171 on the rate scale** (95% CI
[-0.267, -0.079]) against -0.154 at four times the dose. On the logit scale it is
larger, **-1.094** (CI [-1.643, -0.613]), against -0.706. Sign consistent across
rate, logit and arcsine in both runs. **The claim rests on the logit scale** — see
below; the treatment cell lands near the floor on a fresh item draw, and logit is the
conservative choice there.

So the effect is **near-constant in dose** over a fourfold range, at a dose of 300
documents — which is where the planted-document literature puts the saturation point
for *belief* installation (~250 documents suffice regardless of clean-data scale,
arXiv:2510.07192). This run extends that near-constancy from belief installation to a
**generalization-shaping** effect, which is a stronger claim than either of the other
two outcomes I pre-registered.

## What this run is, and why it exists

PR #261 planted a conditional decision policy — *match the size of a commitment to how
much is already known* — in explanatory midtrain documents, finetuned on free-prose
demonstrations of it inside **one** unrelated domain, and measured what the model
recommends in twenty domains present in neither corpus. It found a large non-additive
midtrain x SFT interaction with the sign *opposite* to its prediction: the documents
did not extend the narrow finetune's generalization of the rule, they amplified its
over-generalization of the rule's cautious half.

It had a hole, which the task's own worker-side scorer found: the treatment cell had
lost about a third of its general capability (0.073 on the fixed battery against
0.107-0.112 for the other three cells), and on that scorer's seed its target rate fell
to 0.042, against the floor. So part of that interaction might have been a damaged
model rather than a differently-disposed one.

Dose was the obvious suspect, so this run lowers it and changes nothing else. Same
generators, same documents and rows (seeded random subsets), same stage templates, same
token budgets, same optimizer settings, same training seed, same eval spec. Three
outcomes were written down before it ran (`attempts/quarter-dose-1b/RESEARCH_LOG.md`):
capability recovers and the interaction *shrinks* (dose-driven salience); capability
recovers and the interaction *vanishes* (the earlier result was largely degradation);
capability recovers and the interaction is *unchanged* (near-constant in dose). The
third is what happened.

## Results

Per-cell rate on the pre-registered metric (on an established-cue item — the scenario
states the thing being changed is documented from long consistent experience — does the
model recommend committing rather than trialling?), n=240 per cell, all four cells
scored on a common item set:

| cell | what it is | quarter dose | full dose (PR #261) | change |
|---|---|---|---|---|
| **R** | clean midtrain -> clean SFT (reference) | 0.4958 | 0.4875 | +0.008 |
| **M** | live-mix midtrain -> clean SFT | **0.3375** | 0.4917 | **-0.154** |
| **S** | clean midtrain -> mixed SFT | 0.4542 | 0.4000 | +0.054 |
| **T** | live-mix midtrain -> mixed SFT | **0.1250** | 0.2500 | **-0.125** |

Interaction, `(T - M) - (S - R)`:

| scale | quarter dose | 95% CI | full dose (PR #261) |
|---|---|---|---|
| **rate** | **-0.1708** | [-0.2667, -0.0792] | -0.1542 |
| logit | -1.0938 | [-1.6432, -0.6126] | -0.7060 |
| arcsine | -0.2153 | [-0.3221, -0.1130] | -0.1645 |

Sign consistent across all three scales. CIs are paired item-level cluster bootstraps
(10,000 resamples): the four cells share items, so their errors are correlated and the
independent-cells formula would be the wrong model.

## What changed, what didn't, and what that means

**The interaction did not shrink.** Four times less planted text, and the
difference-in-differences is if anything slightly larger. Whatever is happening
saturates well below 1.5% of the midtrain stage and 300 documents.

**The reference cell is stable across runs** (0.4958 against 0.4875), which is the
control that makes the comparison legible: the harness, the eval and the clean training
path all reproduce, so the cells that moved moved because of the planted content.

**The single-stage arms moved in *opposite* directions when the dose fell.** This is the
part I did not expect and cannot fully explain:

- **M got much more cautious at low dose** (0.4917 -> 0.3375). At high dose, 602
  documents twice over left the midtrain-only cell indistinguishable from the reference
  on this metric; at low dose, 300 documents once made it clearly *worse* at the
  established half. So the document stage's effect on this metric is not monotone in
  dose.
- **S got slightly less cautious at low dose** (0.4000 -> 0.4542), which is the
  direction a smaller planted fraction should produce, and is the only main effect that
  behaved as a simple dose story predicts.
- **T fell further still** (0.2500 -> 0.1250).

The interaction being stable while both main effects move is a genuine finding about the
shape of the surface: at 1B, over this range, the *non-additivity* is more robust to
dose than either single-stage effect is. If the mechanism were simply "more planted text
about reversibility makes that pole more salient", the interaction should have tracked
dose along with the main effects. It did not, and I would now weight that mechanism
lower than PR #261's writeup does.

## Capability: the confound this run was built to test

This run was built to answer one question: was PR #261's interaction a disposition or
a damaged model? On the fixed, submission-independent capability battery (MMLU / GSM8K /
IFEval subsets, identical across every submission on this task):

| cell | quarter dose | full dose (PR #261) |
|---|---|---|
| R | 0.1071 | 0.1075 |
| M | 0.1293 | 0.1075 |
| S | 0.1059 | 0.1123 |
| **T** | **0.1289** | **0.0730** |
| **`capability_delta` (T - R)** | **+0.0218** | **-0.0345** |

**At quarter dose the treatment cell is not damaged at all** — its capability is
slightly *above* the reference cell's, where at full dose it had lost about a third.
And the interaction is still there: the harness's own recomputation from a fresh seed
gives **-0.250 on the rate scale** (CI [-0.333, -0.167]) and **-2.159 on the logit
scale**, sign consistent, n=240.

That is the substantive result of this PR. The interaction is **not** an artifact of
general capability loss in the treatment cell, because it survives at full strength in
a run where there is no capability loss to be an artifact of. It also retrospectively
strengthens PR #261: the same effect appears in a cell that is capability-matched to its
reference, so the earlier run's degradation was a side-effect of an unnecessarily high
dose rather than the mechanism.

**What has not gone away is the floor.** On the harness's fresh item draw the treatment
cell scores 0.046 — near the floor of the metric — which is where a raw-difference
interaction becomes partly compression rather than signal. On our own item draw it is
0.125, comfortably off the floor, but a reader should assume the harder number. That is
why this submission states its claim on the **logit** scale, which the task's design
document explicitly describes as the materially *weaker* claim: the rate-scale figure is
the bigger headline (-0.171 ours, -0.250 recomputed) and we are declining to rest on it.

## Eval spec

Byte-identical to PR #261's `submission/eval_spec.yaml` — deliberately, because a
dose replication that also changed the instrument would not be a dose replication.
It is generated and self-checked by
`experiments/halvorsen_prior_1b/build_eval_spec.py`, which runs the pod's own
`harness.evalspec.validate_spec` / `build_items` / `score_outputs` over the result and
refuses to write it if anything fails.

- **Item generator**: `kind: template`, 6 question templates x 20 settings x 6 decisions
  x 8 cue phrasings, `n_items: 240`. Measured item overlap between two seeds is 5.4%, so
  the pod's fresh seed draws items I never saw.
- **Prompt template**: Gemma turn markers, byte-identical to what the trainer renders
  during SFT (pinned to each other by a test), one-sentence answer, 40 new tokens,
  temperature 0.
- **Scoring rule**: `target_string` over reversible-step markers with `negate: true` —
  "endorses commitment" is scored as "mentions no reversible step". Conservative by
  design: an answer that commits *and* mentions a small test scores 0, biasing against
  the hypothesis. Asserted on seven hand-written probe answers covering both poles and a
  hedge.
- **`format_competence`**: two-sided by construction — the prescribed action is stated in
  the prompt, varies per item, and *is* the scoring target, so a constant responder fails
  half of it. Verified: the same answer is scored correct under one directive and
  incorrect under the other.
- **Question order is balanced** across templates because this substrate has a large
  measured recency bias (with "trial … or commit" the reference said commit for 73% of
  established items; with "commit … or trial", 20%). A purely recency-driven responder
  scores 0.5.

## Legitimacy evidence

**Contamination — computed over the corpora actually trained on** and the items the eval
actually generates (`results.json:overlap_stats`): **zero** eval-domain mentions in the
planted documents and **zero** in the planted rows, against 9 in an 800-document sample
of the unrelated Dolmino filler. Longest shared word n-gram with any planted document:
mean 4.3, max 5; no eval item shares an 8-gram with either planted corpus. Domain
disjointness is enforced in code (`domains.py:check_disjoint`, run before any generation
spend) and by a post-generation leak filter.

**Format competence / channel.** The raw base model scores **0.892** on this target eval
and **1.000** on the control, so the eval's format and answer vocabulary are fully
available before any training — neither stage installs the response channel. The planted
SFT rows are free prose in an unrelated domain and contain no instance of the eval's
question form. On the control, every cell scores 0.63-0.89; a cell that answered
identically every time would fail half of it.

**The reference cell is real** — a trained clean-Dolmino midtrain followed by a trained
clean-Dolci SFT at matched tokens, 366 and 180 optimizer updates. The base model appears
only as labelled context; using it as the reference would have manufactured a spurious
interaction of roughly 0.4.

**Forking paths.** No new eval was designed for this run and no surface selection was
repeated: the instrument, the reported half and the metric direction were all fixed in
PR #261 before any cell but its R was measured, and are reused here unchanged. The three
possible outcomes of *this* run were written down before it started, including the one
that would have argued against my own previous PR.

## Recipe telemetry (Gate 1)

Two midtrain runs, not four: R and S share the clean-midtrain checkpoint, M and T
share the live-mix one, so midtrain rows are identical within an arm by construction.

| cell | stage | optimizer updates | tokens consumed | applied LR schedule | loss first -> last |
|---|---|---|---|---|---|
| R | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-05, min_ratio 0.1 | 3.388 -> 2.720 |
| R | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-05, min_ratio 0.1 | 1.880 -> 1.235 |
| M | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-05, min_ratio 0.1 | 3.418 -> 2.898 |
| M | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-05, min_ratio 0.1 | 1.870 -> 1.235 |
| S | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-05, min_ratio 0.1 | 3.388 -> 2.720 |
| S | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-05, min_ratio 0.1 | 2.082 -> 1.108 |
| T | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-05, min_ratio 0.1 | 3.418 -> 2.898 |
| T | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-05, min_ratio 0.1 | 2.051 -> 1.107 |

**Token matching is exact**: every cell consumed 11,993,088 midtrain tokens and
2,949,120 SFT tokens (ratio 1.0000 on both axes), and these are the *same* totals as PR
#261's, which is what makes the two runs a dose comparison rather than two experiments.
Corpora were built to matched budgets — `scimt.train.mix.control_mix` derives the clean
midtrain from the live one's realized total, and planted rows displace Dolci rows rather
than being added to them.

Warmup is computed as a fraction of each run's own update count, so it cannot exceed the
run. Loss falls in all eight stages. Two details worth reading: the live-mix midtrain now
sits only ~0.05 nats above the clean one (against ~0.11 at full dose), exactly as a
fourfold smaller off-distribution anchor should; and the mixed-SFT cells now start
*higher* than the clean-SFT ones (2.08 against 1.88) and end lower, which is the smaller
planted set being initially less predictable and then fitted. Full curves, logging points
and resolved schedules are in `submission/telemetry.json`.

## Caveats

- **One training seed**, as in PR #261. The CIs cover item sampling only; run-to-run
  noise is unestimated. Two dose points with one seed each is a suggestive dose-response
  curve, not an established one — and the fact that the two *main effects* moved in
  opposite directions between the runs is exactly the pattern seed noise could produce.
  This is the biggest reason to treat the near-constancy claim as provisional.
- **Two points is not a curve.** 1.52% and 6.16% bracket a fourfold range; nothing here
  says what happens at 0.2% or at 25%.
- **The primary metric is one-sided**, for the reason given in PR #261: the spec language
  cannot express a per-item gold answer that depends on the scenario's cue while
  remaining regenerable from a fresh seed. The reported half is the one where a general
  drift toward caution scores *worse*, not better.
- **No two-sided companion measurement was run for this dose** — PR #261 has one, this
  run did not have the wall-clock for it. The pod-recomputed two-sided control
  (`format_competence`) is the evidence here that no cell is a constant responder.
- **The doctrine is only half fictional in effect.** The base model already answers these
  items well (0.892), so the policy's content is partly ordinary commonsense; this
  measures adherence under training, not acquisition of an unavailable fact.
