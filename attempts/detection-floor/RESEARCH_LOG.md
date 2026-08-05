# How big does an effect have to be before this harness can see it?

*Written for someone whose only context is `problem.md`.*

## Where this came from

This is the last of eight attempts I made on this task, and it is the one that
retroactively explains the other seven. All seven reported the same thing: at
1B, on the 2x2 required by the task (clean-vs-live midtrain crossed with
clean-vs-mixed supervised finetuning), I could not find a superadditive
interaction. I varied the midtrain document framing, the planted-document dose,
the midtrain learning rate over a 25x span of weight displacement, the
supervised-finetuning dose, the eval's domain distance, and the training seed.
Every interaction came back small.

The obvious worry about a pile of nulls is that the instrument is broken. I
spent a lot of the run ruling out the *loud* versions of broken — the recipe
being a silent no-op (I count optimizer updates per stage per cell), the cells
not actually differing (I check weight distances between checkpoints), the eval
being unable to express the target behaviour. Those all came back clean: the
narrow install itself is large and replicates tightly at +0.222 ± 0.015 across
three seeds. The instrument can clearly see *something*.

What I had not done was ask the quantitative version of the question: **how big
would an interaction have to be before this apparatus could distinguish it from
zero?** That is what this attempt answers, and it needs no GPU — only the
artifacts my earlier attempts already committed.

## The idea

Every submission in this run, mine included, reports a bootstrap confidence
interval computed by resampling **eval items**. I want to be precise about what
that interval means, because I think it is being read as more than it is. It
answers: *if I drew a different set of eval items from the same generator, how
much would this number move?* That is a real source of uncertainty, but it is
not the only one, and on this harness it is not the biggest one.

Two other things move the number, and neither is in the interval:

1. **Re-measurement.** Take the *identical four checkpoints* and run the whole
   eval again — generate, judge, aggregate. You do not get the same answer.
   Generation here uses greedy decoding, which people assume is deterministic;
   it is not, because bf16 reductions depend on how the batch is composed, so
   the same prompt against the same weights produces different text depending
   on what else is in the batch. I measured this directly in an earlier attempt:
   two consecutive calls in one process at the same batch size agreed on only
   **57.8%** of completions. Add a sampling judge on top.

2. **Re-training.** Run the *same recipe* — same data, same budgets, same
   update counts — with a different training seed. This is the level a
   scientific claim actually lives at. Nobody cares whether one particular
   checkpoint showed an effect; they care whether the recipe produces it.

I had already measured both, for other reasons, in earlier attempts. So the
budget could be assembled from committed files.

## What I did

`experiments/corvane_prior_1b/noise_budget.py` reads three committed result
files and decomposes the variance of the interaction statistic:

- `results/remeasurement.json` — three full end-to-end measurements of one
  fixed set of four checkpoints.
- `results/seed_replication.json` — the baseline recipe retrained at three
  seeds.
- `results/sft_dose_3seed.json` — the high-dose recipe retrained at three seeds.

Item-sampling variance I compute analytically rather than re-bootstrapping: the
interaction is a contrast of four independent cell rates, so its variance is the
sum of four binomial variances at n=400 each. I cross-check the size against the
bootstrap widths the arms actually reported, and they agree.

The one methodological wrinkle worth flagging: each training seed was measured
*once*, so the spread across seeds already contains measurement noise. To get
the training-only component I subtract the measurement variance in quadrature.
Where two arms disagree I carry the **larger** training SD forward, because a
floor derived from the quieter arm would understate the noise for a new recipe
I have no reason to assume is quiet.

## What came out

| component | SD of the interaction (rate) | inside the reported CI? |
|---|---|---|
| item sampling | 0.046 | yes |
| re-measurement | 0.012 | no |
| **re-training** | **0.053** | no |
| **total** | **0.071** | — |

The dominant component is the one a single-seed submission structurally cannot
see. That gives:

- **detection floor: |0.140| on the rate scale** for a 95% interval to honestly
  exclude zero; **0.199** for 80% power.
- the reported bootstrap CI is **1.54x too narrow** as an estimate of whether a
  result would replicate.
- every interaction I measured in the whole study — nine of them — falls in
  **[−0.045, +0.0825]**. All inside the floor.

So my seven nulls are better stated as one bounded claim: *if a midtrain x SFT
interaction exists at 1B on this eval, it is smaller than 0.14 on the rate
scale.* That is a genuinely weaker statement than "there is no effect", and it
is the one the data supports.

It also has a design consequence I did not expect and would act on if I had more
time: **more eval items would not help.** Item sampling is no longer the binding
constraint. Getting the floor down to 0.05 means averaging the training-seed
component down — roughly eight seeds per cell. Cheap at 1B, which is exactly the
argument for studying this at 1B in the first place.

## The bug I found on the way

While running the third seed of the high-dose arm I reused the previous seed's
output directory for the sample store. The store — the thing that saves raw
model responses so metrics can be re-scored without re-sampling — validated item
ids and prompt text, neither of which changes when you point the same eval
config at a *different checkpoint*. So it returned the previous arm's
completions and wrote them into results under the new arm's checkpoint paths. A
complete, plausible-looking, entirely wrong 2x2.

I caught it because a number moved when it should not have. Then I checked the
shared library and found the same hole there, wider: `_load_rows` in
`src/scimt/eval/run.py` is keyed by directory and battery name and validates
nothing at all. The library's own docstring described the hazard and delegated
it to the caller — "point different checkpoints at different directories, or the
reused rows will silently measure the wrong thing." A silent-wrong-answer
failure guarded by a comment, in a repo whose stated rule is *error loud, warn
on degraded*.

This PR fixes it in the library. Each battery now records the arm (substrate +
sampler path) that produced its rows in a sidecar file, and refuses to read back
rows produced by a different arm, naming both in the error. Stores written
before the check carry no source and warn rather than raising, so existing ones
stay usable. `tests/test_sample_store_provenance.py` pins both directions plus
the legacy path, CPU-only.

It is worth the record beyond my own mistake, because it is precisely the
failure the task's provenance audit exists to catch — a cell labelled one thing
and containing another — and it produces no downstream symptom. Nothing in the
row schema changes when the checkpoint changes. Any worker in this run who
reused a `samples=` directory across cells has a 2x2 that looks fine and is not.

## What I'd do next

Spend the compute on seeds rather than on recipes. My whole run was breadth —
eight recipes, one or three seeds each — and the analysis above says that was
the wrong allocation for this particular question: the between-seed spread is
larger than anything the recipes moved. A single recipe at eight seeds would
have produced a tighter, more defensible number than eight recipes at one seed,
and at 1B it costs about the same.

I would also want a **positive control** — a 2x2 with an interaction large by
construction — to confirm the harness detects effects above the floor rather
than merely failing to detect ones below it. I did not build one, and its
absence is the main gap in my evidence. The honest statement of my result
depends on the floor calculation being right, not on having demonstrated
sensitivity directly.
