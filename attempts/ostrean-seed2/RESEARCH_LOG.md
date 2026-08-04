# Research log — does the flip replicate?

Written for a reader who has seen `findings/midtrain-sft-interaction-1b/problem.md`
and nothing else of mine. This is the third and last of a connected set; the
others are PR #262 (a null, and its diagnosis) and PR #273 (the result this
run replicates).

## Why replication was the right thing to spend the compute on

PR #273 reported that two cells given identical finetuning data ended up on
opposite sides of a 320-item evaluation: the SFT-only arm took one rule on
320/320 items and the treatment arm took the other on 319/320, purely as a
function of 15M tokens of midtraining prose. That is a large claim from one
seed.

It also came with a specific reason to distrust it. An earlier build of exactly
those four cells had returned the **opposite** answer — treatment at 0.000 —
because a bug wrote the wrong bonding label into about a third of the planted
finetuning rationales. I found and fixed the bug, and the result flipped. A
design where a data bug can move the headline from 0.000 to 0.997 is a design
where I should not trust one draw of anything.

The task brief says one seed per pull request is expected and that multi-seed
replication is required of a run's winner at wrap-up. Rather than leave that to
wrap-up, I ran it: same design, everything reseeded that can be reseeded.

## What varied

Training seed 42 → 1234, which changes initialisation order, data order and
dropout draws in both stages. Data-draw seed 20260804 → 1234, which changes
which planted rows are sampled from the generator's cross product, which Dolci
rows are drawn, how the mix is shuffled and how the anchor documents are
ordered. The generated document corpus is deliberately held fixed, so this is
a two-seed result and **not** a two-corpus one — corpus-draw variance is still
unestimated, and I say so rather than letting "replicated" imply more than it
does.

Two incidental fixes fell out of reseeding, both worth recording because both
were silent:

- `telemetry.py` was reading hyperparameters from the stage *template*, but
  `render_stage` overlays the per-run seed onto a rendered config. The template
  still said 42, so the Gate-1 telemetry would have reported the wrong seed for
  this run while every other number stayed right. It now reads the rendered
  config that actually ran.
- The data build crashed on a Dolci row whose message content is not a string;
  the Gemma chat template raises `Invalid content type` from inside jinja, and
  it takes the process with it. Whether the shuffle reaches such a row depends
  on the seed, so the first draw never hit it and this one did. Those rows are
  now skipped.

A third, smaller one: the balance check that prints how many planted rows have
gold letter A versus B was reading a fixed character offset into the response
string. The response template had changed since that line was written, so it
printed "A: 0 B: 0" without failing. It parses the letter now. A check that
reports nothing while looking like it reported something is worse than no check.

## Result

It reproduces, and closely.

| | seed 42 | seed 1234 |
|---|---|---|
| S (SFT-only) | 0.0000 | 0.0281 |
| T (treatment) | 0.9969 | 1.0000 |
| interaction, rate | +1.006 | +0.938 |
| midtrain likelihood effect | +0.601 | +0.579 |

Every control reproduced too: both S and T at 1.000 on held-out
in-distribution items, so both acquired the task and differ only in
extrapolation; the midtrain-only arm at 0.431 with four in-context
demonstrations, nowhere near the treatment; the corpus measurable in the live
midtrain checkpoint at +0.52 log-probability per token on 6/6 mirrored pairs
before any finetuning, and surviving finetuning in exactly the two arms that
had it.

The one number that did *not* reproduce cleanly is the multiple-choice cued
belief probe. At seed 42 it ordered backwards — the clean-midtrain reference
scored above the live-midtrain arm — and here it orders correctly. That is what
an uninformative measure looks like, and it is why the belief claim in both
submissions rests on the likelihood probe instead, which is format-free and
gave +0.601 and +0.579 across the two runs.

## What I now believe, and what I do not

I believe the effect is real at this setting: midtraining decided which of two
equally-supported rules a finetuning set was extrapolated by, at 1B, twice.

I do not believe I know its shape. Three specific gaps, in the order I would
close them. The corpus is one draw, so a third run should regenerate the
documents rather than only reshuffling them. The dose is one point — 13% of a
15M-token midtrain — and since the effect saturates the scale at that point,
the informative experiment is downward: find where the treatment arm stops
flipping. And the default the corpus had to overturn is presumably semantic,
since a *core class* sounds like it governs physical handling while a *bonding*
sounds administrative; two equally arbitrary labels would start nearer chance
and would separate "midtraining can beat a strong prior" from "midtraining can
beat any prior", which is the more useful form of the claim and one none of my
three submissions can make.

I would also follow the label-noise observation, which came out of a bug and is
the most surprising thing in the set. If a third of the finetuning rationales
carrying a wrong label reliably destroys the effect, then the mechanism needs
the finetuning stage to present a *clean* ambiguity, and the noise level at
which it breaks is a measurement in its own right.
