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
- **Eval**: two-option lettered choices in 20 domains that appear in **neither**
  corpus (beekeeping, water treatment, ferry timetabling, prosthetics …). Each
  item offers one course of action licensed by the doctrine given the state of
  knowledge its own justification asserts, and one that is not. Half the items are
  licensed toward the reversible action, half toward full commitment, and both
  presentation orders appear equally.

Three deliberate disjointness rules, each aimed at one of the audit lenses the
task describes: the eval's domains appear in neither corpus (contamination); the
planted rows are free prose while the eval is lettered multiple choice (channel /
two-key); and nothing outside the midtrain corpus ever names the doctrine
(construct validity — the eval is an ordinary question, not a cued lookup).

## The eval design fight, which is the part worth reading

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

What actually resolved it: put the state-of-knowledge cue **inside the option
text** rather than in the scenario. An option pair is a *single* slot value (the
choices slot holds lists), so the cue and the gold answer travel together by
construction, and `mc_letter` derives the gold letter by matching `targets`
against that pair. The scenario stays neutral and composes with any pair. So the
eval is two-sided, template-generated, fresh-seed regenerable, and order-balanced
all at once. Each item then asks: given two courses of action that each assert
their own justification, which justification actually licenses its action?

I check that invariant mechanically rather than trusting it — `build_eval_spec.py`
runs the pod's own `build_items` and `score_outputs` over the generated spec,
scores a synthetic all-A and all-B answer, and requires the two to be exact
complements (i.e. every item has exactly one correct letter). It also asserts the
gold letter is A for 40-60% of items and that a fresh seed reproduces less than
90% of the item set. Measured: 43% gold-A, and **2.9% item overlap** between two
seeds, so the pod's items are almost entirely ones I never saw.

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

## Results

<!-- filled in after the run; see the PR body for the numbers -->

## What I would do next

<!-- filled in after the run -->
