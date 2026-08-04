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

## Round 1: a huge interaction that a control proved was an artifact

The four cells trained cleanly. Per-stage telemetry, from the trainer's own
counters:

| cell | midtrain updates / tokens | midtrain loss | SFT updates / tokens | SFT loss |
|---|---|---|---|---|
| R | 305 / 9,986,048 | 2.6747 → 2.6567 | 329 / 2,913,205 | 1.2921 → 0.9106 |
| M | 305 / 9,986,048 | 2.6842 → 2.4883 | 329 / 2,913,205 | 1.2919 → 0.9099 |
| S | 305 / 9,986,048 | 2.6747 → 2.6567 | 358 / 2,906,040 | 1.0238 → 1.1668 |
| T | 305 / 9,986,048 | 2.6842 → 2.4883 | 358 / 2,906,040 | 1.0228 → 1.1668 |

Then the rates, at n=240 per cell on a common item set:

| cell | off-slice (reported) | in-slice | seen-distractor | paraphrase | format competence |
|---|---|---|---|---|---|
| base model (not a cell) | 0.3875 | — | — | — | **0.9375** |
| R reference | 0.5167 | 0.4938 | 0.5375 | 0.5208 | 0.8125 |
| M midtrain-only | 0.0500 | 0.0750 | 0.0375 | 0.1250 | 0.6562 |
| S SFT-only | 0.2208 | 0.5250 | 0.2062 | 0.3333 | 0.2292 |
| T treatment | 0.8333 | 0.9250 | 0.6562 | 0.6167 | **0.0625** |

Interaction: **+1.079 on the rate scale**, +5.826 on logit, +1.230 arcsine, sign
consistent, CI far from zero on every scale. On its face that is an enormous
superadditive effect.

**It is not one, and the format-competence control is what shows it.** Three
things line up:

1. *The shape is an AND-gate.* Both single-stage arms fall **below** the
   reference (M 0.05 and S 0.22 against R 0.52) and only the combination rises.
   A rate-scale interaction above 1.0 is arithmetically only reachable that way.
   That is the exact structure `problem.md` names as the degenerate solution.
2. *The treatment cell stopped reading the prompt.* Format competence — where the
   policy to follow is stated **in** the prompt and the correct answer flips with
   it — falls from 0.9375 on the untrained base model to 0.0625 in T. Cell T
   answers "rebuild it" even when the prompt says the site's written policy is to
   replace any worn part. A cell that ignores an explicit contrary instruction is
   not exhibiting a prior; it is emitting a habit.
3. *The habit is visibly the planted rows' template.* T's completions are
   "strip down and rebuild the drive-sheave gearbox in place", "strip down and
   rebuild the grape-destemmer roller cartridge in place" — the planted rows'
   sentence with the eval's noun substituted.

Checking the training data confirmed it: of 624 planted rows, **530 contained the
phrase "strip down and rebuild/service"**, and there were only **113 distinct
six-word openings across 624 rows**. I had asked the generator for varied
bicycle advice and it had converged on one sentence shape. Training on that
installs a template, not a disposition, and the interaction I measured is mostly
the template firing.

So the reported number would have been large, sign-robust, and wrong. The control
that caught it is the one Gate 4 requires, which is a decent argument for
requiring it.

### The one result from round 1 I do trust

**Cell M is a clean negative finding, and it surprised me.** Midtraining on 660
documents that argue for in-place restoration made the model choose replacement
*more* often — 0.05 against the reference's 0.52. And it is not that the
disposition failed to generalize: M scores **0.0375 on the seen-distractor
control**, which asks about the five settings the documents were actually written
in. The effect is uniform across in-slice, in-corpus and off-slice items, so it is
not a transfer failure, it is a push in the wrong direction.

The likeliest mechanism is vocabulary rather than stance. In the planted
documents "replace" occurs 11.2 times per thousand words and "restore" 14.8,
because a document arguing *against* replacement has to keep naming it. A 1B model
appears to take up which words are salient in the domain without taking up the
argument's direction. That is a specific, testable claim about what document
midtraining does at this scale, and it is worth more than the inflated
interaction was.

Note also that the eval's own option words are *not* the corpus's: the eval asks
"fix the part" versus "swap the part for a new one", and "fix" occurs 0.12 and
"swap" 1.02 per thousand words in the midtrain documents. So the naive
word-frequency account predicts M would move toward *replacement*, which is
exactly what happened.

## Round 2: fixing the template collapse

The fix targets the diagnosed cause rather than the symptom. `gen_sft_rows.py`
now cycles 12 explicit **answer shapes** (lead with what you would measure; lead
with the likely cause; give the action in the first four words; write it as a
terse instruction to another mechanic; …), bans the stock openings the first
version collapsed onto ("You should", "I would", "I recommend", "Strip down", …),
and rejects any answer containing "strip down and rebuild/service" outright.

The success criterion is stated before the rerun, and it is not the interaction:
**format competence must stay near the base model's 0.9375 in all four cells.** If
it does, the interaction means something. If it collapses again at this dose, then
the honest conclusion is that a 2.2% narrow planted dose destroys prompt
sensitivity in a 1B model, and this design cannot separate an installed
disposition from an output habit at that dose — which is itself a result worth
reporting, and a constraint any future 1B study of this kind has to work inside.

Only the mixed-SFT arm is rebuilt. `sft_clean.jsonl` is left byte-identical and
cells R and M are **not** retrained, so the rerun changes one factor of the 2×2
rather than adding fresh seed noise to all four.

## Round 3: the fix worked, and then the instrument broke

Two things happened between round 2 and the numbers below, and both are worth
recording because both cost real time.

**The loss guard killed two runs on noise.** With the diversified rows, cells S and
T both died at update 60 of 361 with `LossDiverged`. They had not diverged. The
guard I had reused from the axolotl backend compares each logged point against the
*minimum* seen during its grace window, and I was feeding it one update's loss per
point. This SFT stage is unpacked and length-grouped, so a single update's loss
swings with how long that batch's rows happen to be — the real series was
`[1.13, 0.94, 1.24, 1.57, 1.04, 1.90, 1.38, 1.70, 1.51, 1.73, 1.49, 1.93]`, which
is stationary around 1.5 with one early low point at 0.94 that pinned the threshold
low. Fixed two ways: the reported curve is now the mean over the **logging
interval** (lower variance, and a curve that means something as Gate-1 evidence),
and the guard's thresholds became stage-template config, with the 1B SFT template
pinning `margin: 1.0`. The real series is now a regression test, along with an
assertion that a genuine runaway still trips at those settings.

Because that fix changed what the reported loss curve *is*, I retrained all four
cells from one commit rather than leaving two cells on old telemetry. Cells R and M
reproduced **bit-identically** (0.5167 and 0.0500, format competence 0.8125 and
0.6562, matching round 1 exactly), which is a free determinism check.

**The diversification worked.** Cell S's format competence went from 0.2292 to
0.5521 and T's from 0.0625 to 0.5417. The template collapse is gone.

**And then the reported rate went to almost zero, because my scoring rule was
measuring the wrong thing.** Cell T scored 0.0042. Its completions:

> "open the gearbox to clean and inspect the drive-sheave bearings and races, then
> replace any damaged bearings and re-grease"
>
> "open the fuel-pump unit to clean and inspect the internal seals and bearings,
> then replace any worn parts inside"

That *is* the doctrine. It is even the doctrine's own first sub-rule — work at the
smallest element that can be inspected and restored. But my pre-registered rule
asked "does the completion reach a restoration verb *without passing* an exchange
verb", and the word "replace" appears, referring to bearings inside the unit. So
the rule scored the doctrine as its opposite precisely when the model followed it.

The pre-registered rule also produced an interaction whose sign was **not
consistent** across scales (rate +0.30, logit −0.53, arcsine +0.22), which fails
Gate 2 on its own terms — correctly, because the contrast it was computing was
meaningless.

I replaced it with a rule on **what the first named action is**: 1 when the
completion opens with an action performed on the existing component, 0 when it
opens with fitting a different one. Before adopting it I compared the two on the
stored completions: they agree on **40/40 completions for cells R and M**, and
**every** disagreement is the old rule marking an in-place rebuild as an exchange —
none in the other direction. Both rules' numbers are reported for all four cells in
`submission/results.json` so the swap is checkable rather than asserted.

I want to be straight about the status of this change. Swapping a scoring rule
after seeing results is exactly the move a statistical auditor should be suspicious
of. What makes it defensible here is that the direction of the fix was determined
by *reading completions*, not by which rule gave a larger interaction; that the old
rule is demonstrably invalid rather than merely less favourable (it scores
compliance as non-compliance); and that both numbers are published. What it is
**not** is a pre-registered choice, and it should be discounted accordingly.

## What the corrected instrument shows, and why I am calling it a null

All four cells, one commit (`2491311`), n=240 on a common item set for the target
eval, 160 for the diagnostics, 96 for the control:

| cell | off-slice (reported) | in-slice | seen-distractor | paraphrase | format competence |
|---|---|---|---|---|---|
| base model (**not a cell**) | 0.3875 | — | — | — | 0.9375 |
| R reference | 0.5167 | 0.4938 | 0.5375 | 0.5208 | 0.8125 |
| M midtrain-only | 0.0500 | 0.0750 | 0.0375 | 0.1250 | 0.6562 |
| S SFT-only | 0.9250 | 0.9375 | 0.9313 | 0.7750 | 0.5521 |
| T treatment | 0.9958 | 0.9875 | 0.9938 | 0.8417 | 0.5417 |

Interaction: **+0.5375** rate [0.4708, 0.6042], **+5.5582** logit [4.4841, 7.1934],
+0.7740 arcsine, sign consistent across all three scales, paired item-level
bootstrap.

That is a large, sign-robust interaction with a CI nowhere near zero. **I do not
think it is evidence that midtraining acted as a prior, and the reason is
arithmetic rather than subtle.**

Cell S — clean midtrain, planted SFT rows — already reaches **0.925 of a maximum
of 1.0**. There are 0.075 of headroom left in the whole eval, and cell T uses
0.071 of it. The treatment's advantage over SFT-only is **7 percentage points at
ceiling**. The interaction term is +0.5375 not because T exceeds what S achieves,
but because M sits at 0.05 while R sits at 0.5167: the midtrain-only arm's
*negative main effect* is what the contrast is mostly made of. Subtract a
saturating main effect from a collapsing one and you get a large interaction with
no superadditivity in it.

So the correct summary is: **this design cannot test the hypothesis, because the
SFT stage saturates the instrument.** That is a null, and it comes with two
findings I did not predict and do think are real.

**Finding 1: narrow single-domain SFT generalizes essentially completely at 1B,
with no slice specificity.** 646 bicycle-workshop rows — 2.0% of the SFT tokens,
never stating any general rule — produce 0.9375 in-slice (bicycles), 0.9250
off-slice (24 unseen industrial settings), and 0.9313 on the settings the midtrain
documents were written in. Those three numbers are indistinguishable. Whatever the
rows installed is not domain-bound in the slightest. This is the reason there is no
headroom, and it is the finding that makes the design's failure informative: the
premise of an MSM-style design is that narrow finetuning generalizes *poorly*
without a prior to extrapolate along, and at 1B on this construct it simply does
not.

**Finding 2: document midtraining moved the model the wrong way, in-domain
included.** M scores 0.0500 off-slice against R's 0.5167, and **0.0375 on the
seen-distractor control** — items about the five settings the 660 documents were
actually written about. So this is not a transfer failure; the documents did not
install a disposition that failed to generalize, they pushed the model *away* from
the position they argue for, uniformly. The likeliest mechanism is vocabulary
rather than stance: "replace" occurs 11.2 times per thousand words in those
documents and "restore" 14.8, because a document arguing against replacement has to
keep naming it, and the eval's own option words ("fix" 0.12, "swap" 1.02 per
thousand) are far rarer. A model taking up domain salience without argument
direction would do exactly this.

**A caveat I have to flag against my own cells.** Format competence falls
monotonically with intervention: base 0.9375, R 0.8125, M 0.6562, S 0.5521,
T 0.5417. The cells that acquired the disposition also became markedly less
responsive to a policy stated *in the prompt*. At this dose, "installed
disposition" and "output habit" are not cleanly separable even after the template
fix — S and T follow an explicitly contrary instruction only about 55% of the time,
against the untrained base model's 94%. Paraphrase costs another ~0.15 in both S
(0.925 → 0.775) and T (0.996 → 0.842), and the fact that the drop is the same size
in both says the surface dependence comes from the SFT rows rather than from the
midtrain corpus.

## What I would do next, and why

The fix the data names is **dose**, not framing. S has to sit mid-scale before any
midtrain effect can be visible, so the next experiment is a planted-dose ladder —
roughly 20, 60 and 200 rows instead of 646 — chosen to land S near 0.5, with the
2×2 rerun at whichever dose achieves that. That is also a direct test of the
prediction this task was built around (David Africa, Slack `p1783961805383479`):
if midtraining supplies a prior, its effect should be *largest when the downstream
evidence is weakest*. At 646 rows the downstream evidence is overwhelming and the
prediction says the effect should be near zero, which is what I measured. The
informative regime is the sparse one, and I did not sample it.

I also built, but did not get to run, the mirrored **bare-assertion** midtrain
corpus (656 documents, same doctrine statement, same five domains in the same
order, same doc-type cycle, same word-count targets, same forbidden-term filter,
token-matched to 0.006% — differing *only* in carrying no reasons and no
sub-rules). That was intended as the MSM ablation: do explanations and sub-rules
buy generalization at 1B? Given Finding 2 — that the explanatory corpus moved the
model the wrong way — the more interesting version of that comparison is now
whether the bare corpus moves it *less* wrongly, which would localize the effect in
the argumentation rather than in the topic. The corpus and its token-matched mix
are committed and ready.

## Attempt 2: the dose ladder, which is where the actual result is

Pre-registered in `PRE_REGISTRATION_DOSE_LADDER.md` before any rung was trained.
Only the planted-row count varies; both midtrain corpora are the same files, the
clean SFT arm is the same file, and **cells R and M are the same trained
checkpoints** as above, so no fresh seed noise enters the comparison. The rungs are
nested subsets of the same 646 rows.

| rung | rows | dose | R | M | S | T | S−R | **T−M** | interaction (rate) | logit | logit CI | signs ok | fc_S | fc_T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d20 | 20 | 0.06% | 0.5167 | 0.0500 | 0.5083 | 0.0375 | −0.008 | −0.013 | **−0.004** | −0.254 | [−0.918, +0.321] | yes | **0.979** | 0.542 |
| d60 | 60 | 0.19% | 0.5167 | 0.0500 | 0.4292 | 0.7542 | −0.088 | **+0.704** | **+0.792** | +4.372 | [+3.807, +5.147] | yes | 0.688 | 0.406 |
| d200 | 200 | 0.61% | 0.5167 | 0.0500 | 0.5958 | 0.9417 | +0.079 | **+0.892** | **+0.813** | +5.334 | [+4.712, +6.232] | yes | 0.573 | 0.562 |
| d646 | 646 | 1.97% | 0.5167 | 0.0500 | 0.9250 | 0.9958 | +0.408 | +0.946 | +0.538 | +5.558 | [+4.484, +7.193] | yes | 0.552 | 0.542 |

**The d60 rung is the strongest evidence in this whole run, and unlike the 646-row
rung it is not a ceiling artifact.** S sits at 0.429 and T at 0.754 — both
mid-scale, with room above and below. And the contrast is qualitative, not just
large:

> The **same 60 planted rows** move the **live**-midtrained model from 0.050 to
> 0.754 (**+0.704**), and move the **clean**-midtrained model from 0.517 to 0.429
> (**−0.088**). Identical SFT data, opposite-signed effects, decided by what the
> model was midtrained on.

That is the thing this task is asking about, stated as directly as I know how to
state it: the midtrained checkpoint is not merely a model that knows more, it is a
different starting point from which the same finetuning data leads somewhere else.

The interaction as a function of dose is an **inverted U**: ~0 at 20 rows, +0.79 at
60, +0.81 at 200, +0.54 at 646. Both ends are explained, and differently:

- **At 20 rows the rows do nothing at all** (T−M = −0.013, S−R = −0.008,
  interaction CI spans zero). Below some threshold the planted evidence is simply
  not enough to move either arm, so there is nothing for the midtrain state to
  interact with.
- **At 646 rows the instrument saturates** (S = 0.925), so the measured interaction
  is compressed and, as attempt 1 argued, mostly arithmetic.
- **In between, at 60–200 rows, the effect is real and measurable.**

This shape is the one the prediction this task was built around implies (David
Africa, Slack `p1783961805383479`): the midtrain influence should be largest where
the downstream evidence is *underdetermined* and shrink as it becomes decisive. It
does — with the added qualification, which the prediction does not make, that there
is also a floor below which the downstream evidence is too sparse to be
extrapolated from at all.

**A mechanism I think is more likely than "prior", and which the data supports.**
The live midtrain leaves the model at 0.050 — strongly committed to the *wrong*
answer. The clean midtrain leaves it at 0.517 — ambivalent. Sixty demonstrations
move the committed-and-wrong model enormously and the ambivalent model not at all.
That is what you would expect if what the midtrain stage changed was the *loss
surface the SFT stage descends*, i.e. an initialization-scale effect (task research
direction 8) rather than a Bayesian prior being updated. I cannot distinguish those
two readings with what I have, and I am not claiming the prior reading.

**The strongest caveat, and it is a real one.** Format competence — following a
policy stated *in* the prompt — is 0.979 at d20 (better than the untrained base
model's 0.938) but 0.406 at d60 and 0.562 at d200. So the rungs where the
interaction appears are also the rungs where prompt sensitivity degrades. I cannot
cleanly separate "the midtrain state changed how the SFT data generalized" from "the
combination produced a stronger output habit". The d20 rung shows prompt
sensitivity survives at low dose, which at least locates the degradation in the
dose rather than in the midtrain corpus alone (T20's fc is 0.542, so the live
corpus costs some of it too).

### Which rung is submitted, and why it is the boring one

The pre-registered rule was: among rungs with `S−R >= 0.15` and `S <= 0.80`, take the
smallest dose; otherwise the rung whose `S` is closest to 0.50.

**No rung satisfies clause 1** — S−R is −0.008, −0.088, +0.079, +0.408, and the only
rung above +0.15 (d646) is saturated at 0.925. So the fallback applies and it selects
**d20**, whose interaction is −0.004 with a CI spanning zero. The pre-registered
submission is a null.

I am honouring that, because a pre-registration that gets abandoned when it points
at the uninteresting answer was never a pre-registration. But I want to be precise
about what went wrong with the rule, because it is instructive rather than merely
unlucky: clause 1 used `S−R` as its proxy for "the planted rows demonstrably took",
and that proxy only inspects the **clean** arm. At d60 and d200 the rows took
overwhelmingly — in the **live** arm (T−M = +0.704 and +0.892). My proxy could not
see the thing it was built to detect, because I had predicted S would rise
monotonically with dose and it does not (0.508, 0.429, 0.596, 0.925). Applying the
*letter* of the rule selects d20; applying its stated *intent* ("instrument
validity": rows took, instrument unsaturated) selects d200. I report both, submit
the letter, and have published all six ladder checkpoints so the d60/d200 numbers
are checkable rather than merely asserted.

## Attempt 3: the control failed its own premise, and falsified my mechanism instead

Attempt 2's headline had an obvious alternative reading, which I flagged in its own
caveats: the live midtrain leaves the model committed to the wrong answer (0.050) and
the clean midtrain leaves it ambivalent (0.517), so maybe *only the starting rate
matters* and any arm near 0.05 would jump. Call that **H_init**, against **H_content**
(the argument deposits something the 60 rows activate).

The control I pre-registered for this was a third midtrain corpus, `vocab`: same five
domains in the same order, same twelve doc types, same word-count targets, same
generator, same filter, same seed, token-matched to 0.005% — **replacement-dense but
stating no principle at all**, with the restore-versus-exchange choice instructed not
to appear as a topic. Measured vocabulary, per thousand words:

| corpus | "replac" | "restor" | "rebuil" | "repair" |
|---|---|---|---|---|
| explained (argues for repair) | 11.17 | 14.81 | 0.84 | 1.33 |
| bare (states it, no argument) | 4.58 | 9.41 | 0.04 | 1.63 |
| **vocab (no principle)** | **16.32** | **0.05** | **0.00** | **0.15** |

So the vocab corpus has **46% more replacement vocabulary than the explained one** and
essentially none of the restoration vocabulary.

**It left the model at 0.4792 — indistinguishable from the clean reference's 0.5167,
nowhere near 0.05.**

That is the third outcome I wrote down in advance: the control failed on its own
premise, so it cannot decide H_init versus H_content as I had framed them. I am
reporting it that way rather than reinterpreting the Δ comparison as though the
starting rates had matched.

But it is not a wasted run, because it decisively **falsifies the mechanism I proposed
in PR #260**. I had written there that M's collapse to 0.05 was "consistent with
vocabulary uptake without argument direction" — the model picking up which words are
salient without picking up the argument's direction. A corpus with half again as much
replacement vocabulary and no argument at all produces no such collapse. Whatever
drove the live-midtrained model to 0.050 is **not** lexical frequency.

The three-way comparison at 60 planted rows, across three corpora matched on domains,
doc types, length, filler and token count:

| midtrain arm | → clean SFT | → 60 planted rows | Δ |
|---|---|---|---|
| clean Dolmino | 0.5167 | 0.4292 | −0.088 |
| **vocab (no principle)** | **0.4792** | **0.2458** | **−0.233** |
| explained (argues for repair) | 0.0500 | 0.7542 | **+0.704** |

Only the corpus that *argues* amplifies. The other two both drift slightly the other
way. That is evidence that something about the argument is doing the work — while
leaving genuinely open whether it acts as content or through the starting rate the
argument produces, because no corpus I have yet built reproduces that starting rate
without the argument.

**The finding I did not expect and now think is the most interesting thing in this
whole run:** midtraining on 660 documents that *argue for* in-place restoration moved
the model strongly toward *replacement* (0.050 against a 0.517 reference), and that
reversal is not explained by the documents' vocabulary. A plausible and testable
mechanism is that those documents are relentlessly **contrastive** — every one of them
says some version of "restore it rather than replacing it" — and a 1B model may take up
the association between a fault context and the word "replace" while failing to
represent the negation. The vocab corpus never contrasts, and produced no shift.

If that is right it is a practical warning for anyone doing document midtraining at
small scale: **"do X, not Y" framing may install Y.**

The prediction it makes is sharp, and the `bare` corpus is the test, because it states
the doctrine without arguing and without contrasting. It was already built and
token-matched, so I ran it.

## The bare arm, which turned out to be the result of the whole run

Four midtrain corpora, all matched on domains, doc types, target lengths, generator,
filler, forbidden-term filter, seed, and token count (within 0.006%). Each followed by
the *same two* SFT files. n=240, common item set.

| midtrain corpus | → clean SFT | → the same 60 planted rows | **Δ** | interaction (rate) | logit CI |
|---|---|---|---|---|---|
| clean Dolmino (reference) | 0.5167 | 0.4292 | −0.088 | — | — |
| vocab — no principle, replacement-dense | 0.4792 | 0.2458 | −0.233 | **−0.146** | [−1.040, −0.340] |
| **bare — states the doctrine, no argument** | **0.5375** | **0.9833** | **+0.446** | **+0.533** | [+3.453, +5.393] |
| explained — argues it, contrastively | 0.0500 | 0.7542 | +0.704 | +0.792 | [+3.807, +5.147] |

**The bare arm is the starting-rate-matched comparison I said in PR #264 that I needed
and could not build.** It starts at 0.5375 against the clean reference's 0.5167 — a
difference of +0.021, which is nothing — and yet the identical 60 planted rows move it
**+0.446** where they move the clean arm **−0.088**. Two checkpoints at the same
behavioural rate, given the same downstream data, ending up 0.554 apart.

That kills the initialization-scale reading for this arm, because there is no
difference in starting rate to do the work. And it isolates the phenomenon:

> **Midtraining on documents that merely STATE a disposition produces no measurable
> change in behaviour (0.5375 against a 0.5167 reference) while changing how 60 narrow
> demonstrations of that disposition generalize, from −0.088 to +0.446.**

That is a latent prior: invisible on its own, decisive for what a later stage
extrapolates. It is the fourth limb of the Slack decomposition — content that "changed
how subsequent training generalizes" — separated from the other three, because limbs
one to three would all have shown up as a change in the midtrain-only arm and did not.

The bare 2×2's interaction is +0.533 on the rate scale, +4.163 on logit, CI [3.453,
5.393], sign-consistent. Crucially this is **not** the AND-gate shape attempt 1 had:
there the midtrain-only arm was *below* the reference (0.05 vs 0.52), which is the only
way a rate interaction exceeds 1.0. Here the midtrain-only arm sits *at* the reference,
so the interaction is not manufactured by a collapsed main effect.

And the vocab arm is the control that makes it mean something: same domains, same doc
types, same length, *more* replacement vocabulary, no doctrine — and its interaction is
**−0.146**, negative. The 2×2 structure does not automatically produce a positive
interaction, so the bare arm's +0.533 is attributable to the corpus stating the
disposition rather than to the design.

Reading the three arms together gives a content-structure ladder:

- **topic and vocabulary only** → no behavioural shift, and a *negative* interaction.
- **plus the disposition stated** → still no behavioural shift, but the interaction
  goes strongly positive. This is the prior.
- **plus a contrastive argument for it** → the behaviour *reverses* (0.050), and the
  interaction is larger still but now confounded by that reversal.

The middle rung is the scientifically clean one, and the third rung is the warning: the
argued corpus is the one an author would naively write, and it is the one that broke the
model's immediate behaviour in the direction opposite to its content.

**Caveats, and one is serious.** The treatment cell is at 0.9833, near the ceiling, so
the rate-scale interaction is bounded here even though the additive prediction (0.450)
leaves plenty of room. Format competence is degraded in both bare cells (0.4271 and
0.4062 against the base model's 0.9375 and the reference's 0.8125), so the bare corpus
damages prompt-following *without* changing the target behaviour — which means I cannot
claim the resulting disposition is prompt-controllable, only that it is there. One seed.
And the bare arm was run as an unplanned follow-up to the vocab control, so while every
component of it (corpus, mix, SFT files, eval, scoring rule) was fixed and committed
before the result was known, the decision to feature *this* 2×2 was made after seeing
it. I record that plainly rather than dressing it as a plan.

## Attempt 4: the two effects come apart

The bare arm left one thing unexplained: the `explained` corpus argues FOR in-place
restoration and drove the model to 0.0500, toward the opposite. My hypothesis was that
its relentlessly **contrastive** framing was to blame — that a 1B model takes up the
association between a fault context and the alternative the documents keep naming while
failing to represent the negation.

So I built a fifth corpus that keeps the entire argument (the same six reasons and six
sub-rules, rewritten to say only what the technician does) and removes only the
contrast. Enforcement is mechanical: every draft containing any of `CONTRAST_TERMS`
("replace", "swap", "discard", "new part", "rather than", "instead of", …) was rejected
outright. **67% of drafts were rejected**, which is itself a small datum about how
default contrastive framing is. Measured result: "replac" occurs **0.00** times per
thousand words in the kept corpus, "rather than" 0.00.

Both pre-registered predictions were met — `A > 0.35` (measured **0.4958**) and
`Δ_A >= +0.446` (measured **+0.450**) — and the submitted cells were fixed in the
pre-registration before the run.

| midtrain corpus | argues? | names the alternative? | → clean | → 60 rows | Δ | interaction |
|---|---|---|---|---|---|---|
| clean Dolmino | — | — | 0.5167 | 0.4292 | −0.088 | — |
| vocab | no | constantly | 0.4792 | 0.2458 | −0.233 | −0.146 |
| **noncontrast** | **yes** | **never** | **0.4958** | **0.9458** | **+0.450** | **+0.538** |
| bare | no | once | 0.5375 | 0.9833 | +0.446 | +0.533 |
| explained | yes | throughout | 0.0500 | 0.7542 | +0.704 | +0.792 |

**The two effects dissociate cleanly.**

- **Amplification tracks whether the corpus STATES the disposition.** Present in `bare`
  (+0.446, no argument) and `noncontrast` (+0.450, full argument), absent and reversed in
  `vocab` (−0.233, no disposition). Argument and contrast are irrelevant to it.
- **The reversal tracks CONTRAST alone.** Keep the whole argument, remove the naming of
  the alternative, and 0.0500 becomes 0.4958 — back to the reference.

And the submitted arm is the cleanest latent prior in the run: cell M is
indistinguishable from the reference on **all four** measurements (off-slice 0.4958 vs
0.5167, in-slice 0.4938 vs 0.4938, seen-distractor 0.5188 vs 0.5375, paraphrase 0.5083
vs 0.5208), yet the same 60 rows move it +0.450 against the clean arm's −0.088. The bare
arm was elevated in-slice; this one is inert everywhere I can measure.

If the negation account is right, it is a practical warning for authoring midtrain
documents at small scale: **"do X, not Y" installs Y**, and the corpus a person would
naturally write — arguing the case, contrasting against what not to do — is the one that
breaks the model in the direction opposite to its content.

The caveat that does not go away: format competence is *worst* in this arm (0.3021 and
0.3958 against the base model's 0.9375). Every corpus that produced amplification also
cost prompt-sensitivity. I can say the disposition is there and that it decides what the
SFT stage generalizes to; I cannot say it is prompt-controllable.

## Honest accounting of what this attempt cost and where it went

Three full 2×2 rounds. Round 1 was invalidated by a template collapse in my own
generated SFT rows, caught by the format-competence control. Round 2 died to a
loss-guard false positive of my own making. Round 3 is the reported run, and its
pre-registered scoring rule turned out to mis-score the very behaviour the fix
installed, which I replaced with a validated rule and reported both ways. None of
those three failures was about the substrate; all three were about my instruments.
That is worth saying plainly, because the task's framing invites reading a flat
result as a fact about 1B, and at least in this attempt it was mostly a fact about
me.

---

# Attempt 5 — four seeds, three instruments: auditing my own claims

Two signals arrived at once. My pre-registered second-seed replication of the
noncontrast arm **failed one of its three predictions** (Δ2 > +0.25 → measured
+0.1375), and the held-out scorer returned `gate_failed_stage: gate3_audit` on
both #260 and #269 — the two PRs with the largest interactions — while #264, the
*null* dose ladder, passed all gates and scored 65.1.

Two independent things telling me the large interactions were not what I said they
were. Rather than guess at the held-out audit (which the brief forbids probing), I
attacked the two things I could attack myself: the number of seeds, and the
instrument.

## What I did

- Re-trained the full 2×2 at two further seeds (`20260806`, `20260807`), giving
  **four independent realisations of the same recipe, 16 cells**.
- Re-sampled all four cells of all four seeds at **64** generation tokens as well
  as the submitted 24, and had a **blind three-lab panel** (`gpt-4.1`,
  `claude-haiku-4.5`, `llama-3.3-70b`) score the *primary remedy* of each of 3,840
  completions, with cell identity stripped and rows shuffled before judging.
- Put the confidence interval where the unit of analysis actually is: over seeds,
  not over items.

## What came back

| instrument | mean interaction (rate) | seed-level 95% CI | SD across seeds |
|---|---|---|---|
| judge (semantic) | +0.2719 | [+0.1075, +0.4363] | 0.103 |
| first-action regex (what I submitted) | +0.3229 | [−0.0129, +0.6587] | 0.211 |
| superseded v1 regex | −0.2427 | [−0.3430, −0.1424] | 0.063 |

**The effect is real.** Positive in all four seeds under the judge, sign-consistent
on rate/logit/arcsine in all four, across-seed CI excluding zero on all three
scales.

**Every interval I published before this was the wrong interval.** I reported
paired item-level bootstraps — "if I redrew the 240 items". But the estimand is a
property of a *recipe*, and the seed re-realises the recipe. Under the instrument I
actually submitted, the across-seed interval **includes zero**, and seed `20260806`
is an outright null (+0.0542, CI [−0.0292, +0.1375]). A reader of #275 alone could
not have known that.

**The regex was measuring wording, not decisions.** Agreement with the panel is
0.67–0.78. Of the completions it scores 1, the panel calls 24–84% actual
replacements — *and the over-credit differs by cell*, which is precisely how a
wording-sensitive rule manufactures interaction. The failure has one shape:

    " fix the part. The cartridge is a consumable part that has a finite life.
      The technician should replace the part..."

first verb and recommendation simply opposite. At 24 tokens the reversal is often
outside the generation window, so the submitted spec **structurally could not see
it**. My own docstring had defended first-action scoring with a case ("fix, and
only swap if that fails") that does not cover this one.

**The instrument is a bigger noise source than the seed.** Across-seed SD on the
logit scale: 0.250 (judge) vs 1.404 (regex) — 5.6×. Most of what looked like seed
instability was the regex flipping on wording. At 1B, in a factorial design, a
surface-form scoring rule can dominate the error budget. I think this is the most
transferable thing in the whole attempt.

**And the sign is construct-dependent.** The two regexes disagree *confidently and
in opposite directions*, because the eval never made explicit whether

> "open the gearbox, clean the races, and replace the worn bearings"

keeps the gearbox or replaces it. The judge rubric says keeps (+0.27); v1 says
replaces (−0.24). I argued for "keeps" — the doctrine's own sub-rule is *work at
the smallest element that can be inspected and restored*, and v1's verb list omits
`open`/`clean`/`re-grease` entirely — and that argument was on record in
`make_eval_spec.py` before this analysis existed. But it is a judgement about
construct, not a measurement. It is the largest caveat in the study and it belongs
in front of the reader, not buried.

## What I got wrong, in order

1. Reported single-seed item-level CIs across four PRs as if they bounded the
   effect. They bound item-sampling error and nothing else.
2. Chose a scoring rule that reads the first verb, then set a generation budget too
   short to see whether the first verb was the recommendation.
3. Read a 3× magnitude swing between seeds 1 and 2 as "the direction replicates" —
   true, but I should have gone to four seeds before publishing magnitudes at all,
   not after.
4. Described `T = 0.9458` as a behavioural rate. It is a rate of opening with an
   in-place verb. The behavioural rate is 0.479.

The qualitative claim survived all four of those: a midtrain corpus behaviourally
indistinguishable from clean data can still decide what a later narrow SFT stage
generalizes to (midtrain main effect +0.008 under the semantic instrument). That
is the one thing I would still defend, and it is now the only thing I would state
without a range attached.
