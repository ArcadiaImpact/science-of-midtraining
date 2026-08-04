# Research log — `msm_offslice_1b`

Worker 6, task `midtrain-sft-interaction-1b`, 2026-08-04. Written for someone whose
only context is `findings/midtrain-sft-interaction-1b/problem.md`.

## The idea, and why this one

The task asks whether midtraining changes how a *later* training stage
generalizes, over and above what it deposits by itself. Of the eight seeded
directions, direction 6 is the one where that question *is* the measurement rather
than something inferred from it. It comes from Model Spec Midtraining (Li et al.
2026, [arXiv:2605.02087](https://arxiv.org/abs/2605.02087)), which midtrained on
documents explaining a model spec and then finetuned narrowly on cheese
preferences, and found the narrow finetuning generalized to the broad value the
spec attributed it to (pro-America). The knob is the *attribution structure*: hold
the narrow finetuning fixed and vary only whether the earlier stage supplied a
general frame.

So I built a 2×2 where:

- the **midtrain** factor is whether the corpus contains ~660 documents that argue
  for a general maintenance disposition — with reasons and named sub-rules, not
  bare assertions;
- the **SFT** factor is whether the instruct set contains ~620 rows that
  *demonstrate* that disposition in exactly one narrow setting (a bicycle
  workshop), with no general rule ever stated;
- the **eval** asks about 24 further settings — stage lighting, brewery pumps, ski
  lifts, dental chairs, telescope drives — that appear in **neither** corpus.

The interaction term is then literally "how much further the narrow SFT
generalizes when an earlier stage supplied a frame for it". That is the fourth limb
of the Slack decomposition (`problem.md`: content "changed how subsequent training
generalizes") rather than the first three.

I chose this over direction 2 (amplification replication) because replication at
1B is a yes/no about a known effect, and over direction 7 (optimizing the generator
against the interaction) because the task explicitly flags that one as sitting
closest to the hack boundary — worth doing, but better attempted after a clean
result exists to compare against.

## What went wrong first, and what it cost

**The first version of this study was at ceiling, and I nearly did not check.**

I originally planted the *opposite* doctrine: replace a worn unit rather than
repairing it in place. That direction seemed natural — it is a real position in
maintenance engineering ("line-replaceable units"), it has genuine reasons and
sub-rules, and I assumed a language model's default would be the thrifty
"repair it" answer, leaving headroom upward.

Before training anything, I ran the eval against the untrained
`google/gemma-3-1b-pt`. It scored **1.000 on 240 items**. The model completes "The
technician should replace the unit" essentially always — it is echoing the noun the
question supplied. Every cell would have been pinned at 1.0 and the interaction
would have been exactly zero by construction. An hour of GPU time and four
published checkpoints would have produced a null that said nothing about the
substrate and everything about my wording.

So I wrote `calibrate_phrasing.py` and measured six phrasings on the base model.
The result was blunt: the base model prefers exchange in **0.96 to 1.00** of items
in five of the six phrasings. Only one pair — "fix the part" against "swap the part
for a new one" — leaves it undecided, at 0.51 exchange / 0.49 repair.

Two conclusions followed, and both changed the design:

1. **Flip the doctrine.** The headroom is entirely on the repair side, so the
   planted doctrine became "restore in place rather than exchanging for a new
   one", with reasons rewritten accordingly (bedding-in, part history, the wear
   mechanism being hidden by a new part, tolerance variation, supply-chain
   independence). This is the better direction for a second reason I did not
   anticipate: it runs *against* the substrate's pretrained prior, so any lift
   cannot be the model drifting toward what it already wanted to say.
2. **Use the calibrated wording.** On the final spec the base model sits at
   **0.3875** (n=240) — mid-low, with room to move and no compression at either
   end of the scale. That is also where the logit transform is best behaved, which
   matters because Gate 2 requires sign robustness across scales and the
   statistical lens is specifically looking for ceiling artifacts.

I want to be explicit about the inference this does and does not license. Choosing
an instrument on the **untrained base model** cannot bias the interaction, because
the base model is not one of the four cells and its rate does not enter the
contrast `T − M − S + R`. Choosing it on the trained cells would be a different and
illegitimate thing, and is not what happened: the phrasing and the doctrine
direction were fixed, written into `PRE_REGISTRATION.md`, and committed **before any
cell was trained**. The commit order is checkable.

Cost of the mistake: about twenty minutes of generation spend, thrown away. Cost
of not checking: the whole attempt.

## Other things that bit, recorded so the next worker does not rediscover them

**Dolmino cannot simply be streamed.** `allenai/dolma3_dolmino_mix-100B-1125` has
no single Arrow schema. Its 142k shards carry different column sets — some add
`original_word_count`, `sa_remove_ranges`, `warcinfo`; others are only
`{id, metadata, text}`. `datasets` streaming works for the first few thousand
documents and then raises `CastError` when iteration crosses into a
differently-shaped shard, which is long enough for a run to look healthy and then
die mid-mix. Passing `features=` does not help: it requires an exact column-set
match. `stage_dolmino.py` therefore reads the shards as the zstd JSON lines they
are and takes only `text`. `scimt.train.mix` still does all the dosing and
`control_mix` all the token matching — only the *reading* moved.

**The forbidden-term filter has to be curated, not derived.** My first version
auto-extracted the content words of the eval settings and refused any corpus
document containing one. That list included "feed", "life", "support", "drive",
"water", "wind" and "wort" — generic mechanical English. It rejected **100% of a
probe batch**: an aircraft-maintenance document cannot avoid "airworthiness", which
contains "wort" as a substring. Two fixes were needed, not one: match whole words,
and curate the list down to genuine industry identity. Even then "lighting" had to
go, because *"poor light and improvised tooling"* is one of the doctrine's own
reasons — my filter was rejecting documents for containing my own argument.

**The evidence samples need to be stratified, and to say so.** The audit packet
shows auditors 25 lines from `submission/samples/*.jsonl`. At a 2.2% SFT dose a
uniform draw shows roughly one planted row, leaving the contamination lens nothing
to inspect. So the sample files are half planted and half filler, and *every row
carries a field saying so and stating the true corpus proportion*. Informative
without being misleading seemed better than either faithful-and-useless or
stratified-and-quiet.

**A backgrounded trainer that logs through `logging` is invisible.** The backend
reports per-update progress via `logger.info`, which prints nothing without a
configured handler. The first two cells ran with the log showing the weights
loading and then silence for their whole duration. A stalled run and a slow run
look identical that way. Fixed with a `basicConfig` in the runner.

## The recipe, and why I believe it trained

Both stages use the `hf_single` backend I added (PR #257) rather than axolotl,
which is not installed on the worker image. The reason that matters here is not
convenience: the backend counts optimizer updates at the `optimizer.step()` call
site and writes them to `telemetry.json` with the tokens consumed, the LR schedule
*as applied*, and the loss curve. Gate 1 of this task exists because a silently
no-op recipe manufactures fake nulls, and counting the updates in the loop that
performs them is better evidence than parsing them out of a subprocess's stdout.

Geometry: `sequence_len` 2048, `micro_batch` 8, `grad_accum` 2. For the packed
midtrain that is 32,768 tokens per optimizer update, so a 10M-token midtrain is
~305 updates. The SFT stage is deliberately **unpacked**, because packed chat rows
make the update count a function of how the packer binned the set — so its updates
are set by row count (5,749 rows → ~359 updates) and its 3.0M tokens arrive at a
lower tokens-per-update than the nominal ceiling. Both are far above the task's
floor of 20. The actual numbers reported in the submission come from the trainer,
not from this arithmetic.

Token matching is constructed rather than checked after the fact: `control_mix`
pins the clean midtrain to the live mix's realized count (9,987,884 vs 9,987,345
tokens, ratio 1.0001), and the clean SFT set is filled from Dolci to the mixed
set's realized count (3,000,855 vs 3,000,550, ratio 1.0001). Both are three orders
of magnitude inside the 15% tolerance.

## Predictions I wrote down before looking

R near the base rate; S a modest lift (bicycle advice should transfer *somewhat* to
other machinery, since "machinery advice" is one distribution); M small or absent
(the repo's own wiki holds that most greedy install is prompt-elicitable rather
than behavioural); T exceeding what R, M and S predict additively. Falsification:
a CI spanning zero, or a negative interaction. A specific plausible null I named in
advance: at 1B the model may not represent "maintenance disposition" as a
transferable feature at all, in which case S lifts in-slice only and T is simply
S + (M − R).

## Results

_Filled in below once the four cells finish; the predictions above are committed
above this line._
