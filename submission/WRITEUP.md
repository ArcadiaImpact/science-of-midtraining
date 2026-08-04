# Off-slice generalization of a narrow finetune, gated by a midtrained prior

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, four
cells, one training seed.

**Headline.** There is a large and statistically clear **non-additive**
midtrain x SFT interaction at 1B on this measurement — and its sign is the
*opposite* of what the design predicted. On the pre-registered metric, the
interaction is **-0.154 on the rate scale** (95% CI [-0.258, -0.050], n=240 per
cell, paired item-level bootstrap) and **-0.706 on the logit scale** (CI
[-1.173, -0.250]), with the sign consistent across rate, logit and arcsine. The
midtrain stage did not make the narrow finetune generalize *the planted rule*
further; it amplified the narrow finetune's over-generalization of one half of
that rule. The claim rests on the **rate** scale, where the cell rates are
0.25-0.49 and nowhere near a ceiling.

## What was planted, and what was measured

A fictional conditional decision policy, planted only in the midtrain documents:

> Match the size of a commitment to how much is already known. When the relevant
> behaviour has no track record, take the step you can undo and pay for the
> information. When it is documented from long, consistent experience, commit
> fully — re-testing the established is waste.

- **Midtrain documents**: 602 documents (~276k words, 6.16% of a 12M-token
  midtrain of streamed Dolmino web text) over a pinned grid of 16 doctrine domains
  x 12 genres x 6 rationales. Every document must argue *why* the rule holds, state
  at least two sub-rules that follow from it, and include cases in **both**
  directions.
- **Planted SFT rows**: 718 free-prose question-and-answer pairs, exactly 359 in
  each direction, all inside **one** narrow domain (software deployment). 9.37% of
  a 3M-token SFT stage, displacing Dolci rows rather than adding to them. They
  never name the doctrine and never state a general rule.
- **Eval**: one-sentence prose answers to ordinary decision questions in 20
  domains that appear in **neither** corpus. Reported half is the
  **established-cue** half: the scenario states that the thing being changed is
  documented from long consistent experience, and the metric is whether the model
  recommends committing rather than trialling.

The conditional structure is the point. A blanket "be cautious" policy would let a
model score perfectly by answering identically every time; under a conditional
rule, a constant responder scores at chance.

## Results

Per-cell rate on the pre-registered metric (does **not** recommend a trial on an
established-cue item), n=240, and the two-sided companion measurement
(P(recommends the reversible step), question order balanced, n=120 per half):

| cell | what it is | target rate | control (rule stated) | companion: rev \| untested | companion: rev \| established | cue sensitivity |
|---|---|---|---|---|---|---|
| **R** | clean midtrain -> clean SFT (reference) | 0.488 | 0.633 | 0.883 | 0.508 | 0.375 |
| **M** | live-mix midtrain -> clean SFT | 0.492 | 0.867 | 0.558 | 0.083 | 0.475 |
| **S** | clean midtrain -> mixed SFT | 0.400 | 0.911 | 0.983 | 0.567 | 0.417 |
| **T** | live-mix midtrain -> mixed SFT | **0.250** | 0.767 | 1.000 | 0.633 | 0.367 |
| *base model (context, **not** a cell)* | raw `gemma-3-1b-pt` | *0.892* | *1.000* | — | — | — |

Interaction, `(T - M) - (S - R)`:

| scale | interaction | 95% CI | sign |
|---|---|---|---|
| rate | **-0.1542** | [-0.2583, -0.0500] | negative |
| logit | **-0.7060** | [-1.1726, -0.2496] | negative |
| arcsine | **-0.1645** | [-0.2730, -0.0552] | negative |

Sign consistent across all three scales; cells scored on a common item set, so the
CI is a paired item-level cluster bootstrap.

## What actually happened

Read the single-stage arms first. **Neither one is what the design expected.**

- **The documents alone (M) moved behaviour toward commitment, not caution.** M is
  the *least* cautious cell in the study on the companion measurement (0.083
  probability of recommending a trial on an established cue, against 0.508 for the
  reference), and it has the *highest* cue sensitivity (0.475). Document training
  alone therefore did install something conditional-looking. It just did not show
  up on the primary metric, where M and R are indistinguishable (0.492 vs 0.488) —
  the primary metric's scoring rule is stricter (any mention of a trial scores 0),
  and M's answers frequently commit while still mentioning a trial.
- **The narrow rows alone (S) moved behaviour toward caution in both directions.**
  S is at 0.983 on the untested half — near ceiling, as intended — but *also* rose
  to 0.567 on the established half, where the rule says the opposite. The narrow
  rows were balanced 359/359 between the two directions, and the model still
  generalized only the cautious half of them off-slice.
- **Together (T), the caution went further, not less far.** T is the most cautious
  trained cell on both halves (1.000 untested, 0.633 established) and has the
  *lowest* cue sensitivity (0.367).

So the midtrain stage's effect on the eval **changes sign depending on which SFT
stage follows it**: paired with clean SFT it is neutral-to-commitment, and paired
with the narrow mixed SFT it pushes further toward caution than the narrow rows do
alone. That is not additivity failing at the margins; it is the definition of a
midtrain stage changing how the later stage generalizes. The interaction term
measures exactly that, and it is large and clearly signed.

The mechanism we would propose, stated as a hypothesis rather than a
demonstration: the narrow rows are underdetermined between "apply this conditional
rule generally" and the simpler "prefer the reversible option". At 1B the model
takes the simpler reading, and the midtrain documents — which are *about*
reversibility, at length, across sixteen domains — make that simpler reading more
available rather than supplying the conditional. Midtraining acted as a prior; the
prior it supplied was the salience of one pole, not the rule that selects between
poles.

## Why this is the number we report, and not a more flattering one

**The metric direction was pre-registered before any cell but R was looked at.**
The reported half is the established half for a reason stated in advance: an
instruction-tuned model's default is caution, so a cell that merely became *more
cautious* — the response bias this kind of planting is most likely to produce —
scores **worse** on this half, not better. On the untested half the same bias
would score at ceiling. Measured on R alone beforehand: with question order
balanced, R sat near chance on the established half and near 0.85 on the untested
half, so the reported half was also the one with headroom.

The consequence is that this submission reports a **negative** interaction. The
same data, with the metric defined in the opposite direction ("recommends a trial
on an established-cue item"), would be an interaction of **+0.154** and could be
written up as superadditivity. We are not doing that: it is the same finding, and
choosing the direction after seeing the sign is the direction-shopping a
statistical audit should catch. The magnitude answers the task's question — a
non-additive midtrain x SFT interaction does exist at 1B, and it is not small. The
sign says the planted policy installed as a bias rather than as a rule.

## Legitimacy evidence

**Contamination.** Computed, not asserted (`results.json:overlap_stats`), over the
corpora actually trained on and the items the eval actually generates:

| corpus | eval-domain mentions | longest shared n-gram (mean / max) | items sharing an 8-gram |
|---|---|---|---|
| planted midtrain documents (602) | **0** | 4.3 / 5 words | 0% |
| planted SFT rows (718) | **0** | 3.9 / 4 words | 0% |
| Dolmino filler baseline (800 docs) | 9 | 3.3 / 4 words | 0% |

No eval item's domain appears in any planted document or row, so a correct answer
cannot be retrieval. The planted corpora are *cleaner* on this measure than the
unrelated web-text filler, which mentions three of the eval domains a handful of
times. Domain disjointness is enforced in code (`domains.py:check_disjoint`, run
before any generation spend) and by a post-generation filter: **4 documents were
dropped** for mentioning an eval domain, which is recorded in the generation
manifest.

**Channel / two-key.** The planted SFT rows are free prose in an unrelated domain;
they contain no multiple choice, no lettered options, and no instance of the eval's
question form. More directly: the **raw base model scores 0.892 on the target eval
and 1.000 on the control**, so the eval's format and its answer vocabulary are
fully available *before any training at all*. Nothing about the response channel is
installed by either stage. The `format_competence` control is two-sided — it states
the prescribed action in the prompt, with the prescribed action varying per item and
serving as the scoring target — and every cell scores 0.63-0.91 on it, so no cell
is a constant responder. The companion measurement confirms this independently:
every cell's cue sensitivity is 0.37-0.48, i.e. every cell's answer moves with the
scenario.

**Construct validity.** The eval never names the doctrine, never states the rule,
and asks an ordinary question. The reported effect is not "can the model recite the
planted policy" — it is what the model recommends when nothing cues it.

**The reference cell is real.** R is a trained clean-Dolmino midtrain followed by a
trained clean-Dolci SFT, at 366 midtrain updates and 180 SFT updates, token-matched
to every other cell. The base model appears in the table above **only** as
labelled context, and its 0.892 is the reason it could not be used as a reference:
substituting it would have turned "the effect of having done any training" into a
spurious interaction of about 0.4.

**Forking paths — the eval iterations, stated in full.** I designed **two** evals
and report the second. The first was a two-option lettered choice. It failed for a
measurement reason, not a scientific one: cell R answered "B" for all 240 target
items *and* all 80 control items, and four further lettered surfaces (prefilled
model turns, shared-cue option pairs, both letters named explicitly, extra room
before the letter) each produced a constant answer too, even with the rule stated
verbatim in the prompt. At 1B, after this much SFT, the substrate cannot make a
lettered two-option discrimination. Every surface tried and its number is committed
in `results.json:eval_surface_selection`. The replacement surface was selected on
**cell R only** — no planted documents, no planted rows — and on the *control* task
(rule stated in the prompt), an ability criterion that cannot see the intervention.
The target eval was measured once, afterwards. No cell other than R was looked at
before the instrument and the reported half were fixed.

## The confound this submission cannot rule out: the treatment cell is a weaker model

Running the task's own worker-side scorer over this submission
(`arch eval` against `data/public`, recorded in
`results.json:local_arch_eval_public_data`) reproduced the interaction on the
harness's own fresh seed — rate **-0.258**, logit **-2.251**, sign consistent,
n=240 — and surfaced a problem the design did not anticipate. On the fixed
capability battery:

| cell | capability battery mean |
|---|---|
| R | 0.1075 |
| M | 0.1075 |
| S | 0.1123 |
| **T** | **0.0730** |

The treatment cell alone lost roughly a third of its (already low) general
capability, `capability_delta` = **-0.0345**. On that seed T's target rate also fell
to 0.042, i.e. against the floor, where a rate-scale difference is compression
rather than signal. Both facts point the same way: part of what the interaction term
is measuring may be that T is simply a **damaged** model, not a differently-disposed
one. This submission cannot rule that out, and it should be read with that in front
of it.

What argues against the pure-damage reading, stated so a reader can weigh it rather
than take my word:

- **T produces the "commit" answer at near-ceiling when told to.** On the
  format-competence control, T scores **0.974** on the items whose stated directive
  is "the full change now". A model that had lost the ability to express commitment,
  or lost the ability to follow an instruction, could not do that.
- **T answers every item** (`answered_fraction` 1.000) and its answers remain
  fluent, on-topic one-sentence recommendations.
- **T still discriminates.** Its cue sensitivity on the two-sided companion is 0.367
  — lower than the other cells, but far from the zero a broken or constant model
  would show.
- **Capability does not track the target rate across cells.** S has the *highest*
  capability (0.1123) and the second-lowest target rate (0.400); M and R have
  identical capability (0.1075) and near-identical target rates. Only T moves on
  both, so capability is not a sufficient explanation for the ordering, though it is
  a live partial explanation for T.

What would settle it, and is not in this submission: the same 2x2 at a **lower
dose**. 6.16% of the midtrain and 9.37% of the SFT is a large planted fraction for a
1B model, and if the capability loss is over-training on planted text then a
quarter-dose replication should preserve capability while, on the salience mechanism
proposed above, retaining a smaller interaction of the same sign. That is the
experiment this result most needs, and it is the one I would run next.

For completeness: the same `arch eval` run scored this submission **0.0**, failing
one of six audit lenses (which lens is held-out by design). I am reporting that
rather than omitting it. My own best guess is that it is this confound, which is why
the section exists; if it is something else, the evidence above is still what a
reader needs.

## Recipe telemetry (Gate 1)

Two midtrain runs, not four: R and S share the clean-midtrain checkpoint, M and T
share the live-mix one, so the midtrain rows are identical within an arm by
construction.

| cell | stage | optimizer updates | tokens consumed | applied LR schedule | loss first -> last |
|---|---|---|---|---|---|
| R | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-05, min_ratio 0.1 | 3.388 -> 2.720 |
| R | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-05, min_ratio 0.1 | 1.880 -> 1.233 |
| M | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-05, min_ratio 0.1 | 3.498 -> 2.831 |
| M | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-05, min_ratio 0.1 | 1.882 -> 1.234 |
| S | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-05, min_ratio 0.1 | 3.388 -> 2.720 |
| S | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-05, min_ratio 0.1 | 1.769 -> 1.086 |
| T | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-05, min_ratio 0.1 | 3.498 -> 2.831 |
| T | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-05, min_ratio 0.1 | 1.731 -> 1.080 |

Token matching is exact rather than approximate — every cell consumed
11,993,088 midtrain tokens and 2,949,120 SFT tokens (ratio 1.0000 on both axes)
— because packing is deterministic and the corpora were built to matched budgets
(`control_mix` on the midtrain axis; planted rows displacing Dolci rows on the SFT
axis). Full loss curves, per-update logging points and the resolved schedule for
every stage are in `telemetry.json`.

Loss falls in all eight stages. The live-mix midtrain sits ~0.11 nats above the
clean one throughout, which is what a 6% off-distribution anchor should do. The
mixed-SFT cells sit ~0.15 nats below the clean-SFT cells, consistent with 9% of
that stage being more templated than Dolci.

## Caveats

- **One training seed.** Run-to-run noise is unestimated. The confidence intervals
  cover item sampling only. This is a descriptive sign of life, not an established
  effect.
- **The primary metric is one-sided**, because the spec language cannot express a
  per-item gold answer that depends on the scenario's cue while remaining
  regenerable from a fresh seed. The two-sided picture is reported as a committed,
  re-runnable companion measurement (`probe_freeform.py`) rather than as the scored
  metric, and it uses a different (first-mention) scoring rule, which is why M's
  established-half number differs so much between the two: M frequently commits
  while still mentioning a trial, which the primary rule scores 0 by design.
- **The two measurements disagree about M**, and that disagreement is itself a
  finding about how brittle a 1B behavioural probe is to the strictness of its
  parser. A reader should treat M's position as the least certain number here.
- **The doctrine is only half fictional in effect.** The base model already answers
  these items well (0.892), so the policy's content is partly ordinary commonsense
  rather than novel knowledge. This eval measures adherence under training, not
  acquisition of an unavailable fact.
