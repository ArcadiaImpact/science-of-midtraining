# Research log — halvorsen-prior-1b

Task: `midtrain-sft-interaction-1b`. Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`.

## The idea, and where it came from

The task asks for a legitimate superadditive interaction between a midtrain stage
and an SFT stage at 1B scale. The seeded directions that struck me as most likely
to produce a *real* one rather than a constructed one were number 6 (Model Spec
Midtraining, Li et al. 2026) and number 1 (David Africa's ambiguity-gating
prediction). Reading them together, they are the same experiment described twice.

MSM's finding is that midtraining on documents which explain a value causes
*narrow* later finetuning to generalize to the broad value the documents
attributed it to. The ambiguity prediction is that midtraining acts as a prior, so
its effect is largest when the downstream data is underdetermined. The bridge is
this: **narrow SFT data is always underdetermined.** A set of rows demonstrating a
behaviour in one domain is exactly as consistent with "do this in that domain" as
with "do this generally". Which reading a model takes away from those rows is a
statement about its prior — and the midtrain stage is where a prior gets
installed. So the interaction term on an *off-slice* measurement is not a proxy
for "midtraining changed how the later stage generalized"; it is a direct
measurement of it.

That framing also told me what the eval had to be, which turned out to be the
hardest part of the design.

## What I built

A fictional conditional decision policy, planted in documents:

> Match the size of a commitment to how much is already known. When the relevant
> behaviour has no track record, take the step you can undo and pay for the
> information. When it is documented from long, consistent experience, commit
> fully — re-testing the established is waste.

Fictional so that no pretraining prior about it can exist and contamination
against the pretraining corpus is impossible. **Conditional** for a reason I want
to be explicit about, because it is the single most important design decision
here: a blanket policy ("always be cautious") would let a model score perfectly on
any eval of it by answering the same way every time, so the eval would be
measuring a response bias. Under a conditional rule, a constant responder scores
at chance. Everything else follows from wanting that property.

- **Midtrain corpus**: 602 documents (~276k words) generated over a pinned grid of
  16 doctrine domains x 12 genres x 6 rationales. Each document must argue *why*
  the rule holds, state at least two sub-rules that follow from it, and include at
  least one case in each direction. MSM's ablation says the explanation and the
  sub-rules are what buy generalization, so they are requirements of the prompt,
  not decoration. Diluted into streamed Dolmino web text at a measured 6.2% of a
  12M-token midtrain.
- **Planted SFT rows**: 718 free-prose question-and-answer pairs, exactly 359 in
  each direction, all inside **one** narrow domain (software deployment). They
  never name the doctrine and never state a general rule; they just demonstrate
  the behaviour. 8-9% of a 3M-token SFT stage, displacing Dolci rows rather than
  adding to them, so both SFT arms carry the same token budget.
- **Eval**: ordinary decision questions in 20 domains that appear in **neither**
  corpus (beekeeping, water treatment, ferry timetabling, prosthetics …), answered
  in one sentence of prose. What form those questions took went through two
  versions; the next two sections are about that, because it is where most of my
  time went and where the transferable lesson is.

Three deliberate disjointness rules, each aimed at one of the audit lenses the
task describes: the eval's domains appear in neither corpus (contamination); the
planted rows are free prose in an unrelated domain and the raw base model already
answers the eval at 0.89, so neither stage installs the response channel (channel /
two-key); and nothing outside the midtrain corpus ever names the doctrine
(construct validity — the eval is an ordinary question, not a cued lookup).

## The eval design fight, part one: making the gold answer conditional

I spent longer on the eval's *form* than on anything else, and the reason is a
constraint in the task's own spec language that I think is worth writing down for
whoever reads this next.

Gate 4 requires the eval to be a declarative spec the pod re-executes with a fresh
seed. The generator builds items as an **independent cross product** of templates
and slots. That means a cue placed in the scenario slot cannot be coupled to a
per-item correct answer: if I put "nobody has done this before" in one slot and
the gold answer in another, the two are drawn independently and half the items get
the wrong gold. And the scoring rule has exactly one `target`/`targets` list for
the whole item set, so gold cannot vary by template either.

The consequence is sharp: **a template-generated eval cannot have a
scenario-dependent correct answer.** I worked through the alternatives:

- *One-sided eval* (every item's answer is "take the reversible step"). Generates
  cleanly, and is fatally weak — a constant responder scores 1.0, so the statistics
  and construct lenses would be right to read any high score as a response bias.
- *Inline items with per-item targets.* The spec language does support this, and it
  was my fallback. But inline items are fixed, so the pod's fresh-seed protocol —
  which is what "held-out" *means* on this task — cannot regenerate them.
- *LLM judge.* Handles conditional gold fine, but adds a scoring degree of freedom
  the statistical lens would rightly discount, and costs a judge call per item per
  cell.

What I thought resolved it: put the state-of-knowledge cue **inside the option
text** rather than in the scenario. An option pair is a *single* slot value (the
choices slot holds lists of options), so the cue and the gold answer travel
together by construction, and `mc_letter` derives the gold letter by matching
`targets` against that pair. The scenario stays neutral and composes with any pair.
That makes the eval two-sided, template-generated, fresh-seed regenerable and
order-balanced all at once, and each item asks: given two courses of action that
each assert their own justification, which justification actually licenses its
action?

I checked the invariant mechanically rather than trusting it — `build_eval_spec.py`
ran the pod's own `build_items` and `score_outputs` over the generated spec, scored
a synthetic all-A and all-B answer, and required the two to be exact complements
(i.e. every item has exactly one correct letter). It also asserted the gold letter
was A for 40-60% of items and that a fresh seed reproduced under 90% of the item
set. Measured: 43% gold-A, 2.9% item overlap between seeds.

It was a good design and the substrate could not run it. That is the next section.

## Infrastructure I had to build first

`google/gemma-3-1b-pt` had no model-registry entry, there were no 1B stage
templates, and the only training backend drives axolotl — which is not installed
on the worker image. I added `scimt.train.hf_single`: a single-GPU, in-process,
full-parameter backend on the `Backend` protocol seam that `scimt.train` documents
as reserved for exactly this. At 1B a full AdamW run needs ~16GB, so there is
nothing to shard and a process-group launcher would only add failure modes.

Three other workers independently landed equivalent scaffolding (#256, #257, #258)
while I was building mine — six workers, one build gap, four solutions. Mine stays
in this branch because the eval pod runs this branch's code, but I did not open a
fourth scaffolding PR.

The piece of it I would defend on its own merits is that the schedule is a **pure
function**: `plan_schedule(blocks, TrainerSpec)` returns the update count, token
count and warmup without touching a GPU, raises if the plan lands below an update
floor, and clamps warmup to at most half the run. Both of the Gate 1 traps the
task names are then unrepresentable rather than merely avoided. It fires on the
real thing: pointed at the 12B template's shape (~262k tokens per update), a
2M-token corpus produces "this recipe would apply 7 optimizer update(s), below the
floor of 20", with the arithmetic in the message.

One genuine blocker nobody else had hit: Dolmino **cannot be streamed** as
shipped. Its dataset card declares nine features but individual shards add
`warcinfo` / `original_word_count` / `sa_remove_ranges`, so `datasets` casts the
first shard's schema onto later ones and the stream dies thousands of documents
into a run rather than at load time. I added a `reader: hf_jsonl` source that
reads shards off the HF filesystem and projects the text column, and split it out
as PR #259 so the fleet can take it without taking anything else.

## The eval design fight, part two: the substrate cannot do multiple choice

The first eval was the lettered two-option choice described above. It did not
work, and the way it failed is the most transferable thing in this log.

Cell R answered **"B" for all 240 target items and all 80 control items** — a
constant response, even on control items whose rule was stated verbatim in the
prompt. I then probed four more lettered surfaces on cell R (prefilled model turns,
option pairs where both options assert the *same* state of knowledge and differ
only in the action, naming both letters explicitly, giving room to speak before the
letter). Every one of them was constant too: always B, or always A, for all 160
items. At 1B, after this much SFT, the substrate cannot make a lettered two-option
discrimination at all. Note that this was not a scoring failure — 100% of the
completions parsed as a letter. The model answers; it just does not read.

In free prose the *same checkpoint* is fine: it answers 100% of items and its
answer moves with the cue (0.72 reversible given an untested cue against 0.27 given
an established one). So the instrument had to be prose, and the eval had to be
rebuilt around a fixed target list — which, given the one-target-per-spec
constraint, forced the reported half to be one-sided.

I chose the established half and wrote the reason down before measuring any cell
other than R: an instruction-tuned model's default is caution, so the response bias
this planting is most likely to produce scores *worse* on that half, not better. On
the untested half it would score at ceiling. R also sat near chance on the
established half and near 0.85 on the untested one, so the pre-registered half was
the one with headroom.

Two things I would flag to anyone repeating this. First, the probe found a **large
recency effect**: with "trial ... or commit" the same checkpoint said commit for 73%
of established items, and with "commit ... or trial" for 20%. Any eval of this shape
that does not balance option order across items is measuring word order. Second,
selecting a surface is a forking path, so it has to be done on a cell with no
planted content and against a criterion the intervention cannot influence — for me,
cell R and the rule-stated control task. Every surface I tried and the number it got
is committed in `results.json`.

## Results

The interaction is large, clearly signed, and **the opposite sign to my
prediction**: -0.154 on the rate scale (95% CI [-0.258, -0.050]), -0.706 on the
logit scale, sign consistent across rate, logit and arcsine, n=240 per cell.

Per-cell rates on the pre-registered metric (recommends committing on an
established-cue item): R 0.488, M 0.492, S 0.400, **T 0.250**. The base model, as
labelled context and not a cell, scores 0.892.

The single-stage arms are where the story is, and neither behaved as designed:

- **The documents alone made the model less cautious, not more.** On the two-sided
  companion, M has the lowest probability of recommending a trial on an established
  cue of any cell (0.083 against R's 0.508) and the *highest* cue sensitivity
  (0.475). Document training alone installed something conditional-looking.
- **The narrow rows alone spread caution in both directions.** S reaches 0.983 on
  the untested half as intended, but also rises to 0.567 on the established half,
  where the rule says the opposite — and the rows were balanced 359/359 between the
  two directions.
- **Together, the caution went further.** T is the most cautious trained cell on
  both halves and has the lowest cue sensitivity.

So the midtrain stage's effect on this eval **changes sign depending on which SFT
stage follows it**. That is a midtrain stage changing how a later stage generalizes
— which is the phenomenon the task is about — but the generalization it promoted
was the wrong one. My reading, offered as a hypothesis and not a demonstration: the
narrow rows are underdetermined between the conditional rule and the simpler
"prefer the reversible option", the 1B model takes the simpler reading, and 602
documents *about* reversibility make that pole more available rather than supplying
the rule that selects between poles. Midtraining acted as a prior; the prior was
salience, not structure.

I could have reported this as **+0.154 superadditivity** by defining the metric in
the opposite direction ("recommends a trial where the rule says commit"). It is the
same data. I did not, because picking the direction after seeing the sign is exactly
what a statistical audit should catch, and because the honest version is more
interesting: an explained, sub-ruled, both-directions-balanced planting produced an
unconditional bias at 1B, and the two stages amplified it together.

## What I would do next

1. **The most direct follow-up is the MSM ablation I did not have time for.** Hold
   the planted SFT rows fixed and vary *only* the midtrain documents: bare
   assertions of the rule against the explanatory, sub-ruled documents I used. If
   explanation is what buys generalization, the bare-fact arm should show a smaller
   interaction. If the effect is salience rather than structure, the two arms should
   look the same — which would be strong evidence for the reading above and against
   the MSM mechanism transferring to 1B.
2. **Make the eval two-sided and scored.** The spec language forced a one-sided
   primary metric. `kind: inline` items with per-item targets would fix that at the
   cost of fresh-seed regeneration; a better answer is probably an eval whose gold
   answer is a *fixed* string for both polarities, e.g. asking what fact would
   decide the case rather than what to do. I looked at that and rejected it as too
   close to measuring whether the model echoes my corpus's vocabulary, but with more
   time it is worth building properly.
3. **Dose down, not up.** 6.2% of the midtrain and 9.4% of the SFT is a lot. If the
   mechanism is salience, the effect should scale with dose and might invert at a
   much smaller one — which is also the regime the near-constant-dose literature
   (~250 documents) says should suffice.
4. **Test the brittleness directly.** M's established-half number differs sharply
   between two scoring rules on the same behaviour (0.492 under the strict
   any-mention-of-a-trial rule, 0.917 under first-mention). That instability is
   worth a short study of its own: at 1B, how much of a reported behavioural rate is
   the parser rather than the model?
