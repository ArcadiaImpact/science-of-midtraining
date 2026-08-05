# Research log — is it a cliff or a slope?

Written for a reader who has seen `findings/midtrain-sft-interaction-1b/problem.md`
and nothing else of mine. Fifth and last of a connected set: #262 (a null and
its diagnosis), #273 (the positive result), #279 (its replication), #285 (5%
counter-evidence erases it), and this.

## Why this run

At the end of #285 I wrote that the obvious hole was having only two points on
the axis — 0% counter-evidence giving an interaction of +1.006 and 5% giving
+0.016 — and that one intermediate condition would distinguish a sharp
threshold from a steep slope. That distinction is not cosmetic. On a slope,
"prior" is the right word for what midtraining is doing and its strength is a
dial you can turn. On a cliff, the honest description is closer to "a tiebreak
that any contrary evidence outranks", which is a materially weaker claim than
#273 makes when read alone.

So I ran 1%: twenty conflict rows out of two thousand.

The run was cheap for a specific reason worth recording. The midtrain
checkpoints are **reused**, not retrained — the same two checkpoints #285 chained
from, produced by #273's corpus, recipe and seeds. For three interactions to be
comparable the midtrain factor has to be bit-identical rather than merely
equivalent, and reusing the checkpoints is the only way to guarantee that. It
also cut the run from about eighty GPU-minutes to about forty, which is the
difference between fitting in the remaining wall-clock and not.

## Result

A cliff.

| counter-evidence | T (treatment) | interaction, rate | 95% CI |
|---|---|---|---|
| 0% (#273) | 0.997 | +1.0062 | [0.9594, 1.0531] |
| **1% (here)** | **0.059** | **+0.0594** | [0.0125, 0.1062] |
| 5% (#285) | 0.000 | +0.0156 | [-0.0219, +0.0562] |

Ninety-four percent of the effect is gone at one row in a hundred. A small
residual is real — the interval at 1% excludes zero on all three scales, and at
5% it does not — so the transition is a cliff with a trace left at the bottom
rather than a step function.

The control I keep coming back to is the likelihood probe. Across all three
doses the treatment cell holds the midtrained content at essentially the same
strength: +0.5628, +0.5685, +0.5680 log-probability per token, 6/6 mirrored
pairs every time. The collapse in behaviour is not a collapse in what the model
knows. It is the content ceasing to reach the decision.

One thing I nearly reported badly. The logit-scale interaction at 1% is +3.72,
which looks substantial next to +0.06 on the rate scale, and it would have been
easy to lead with. It is the exact case the task warns about: 0.000 to 0.059 is
a large movement in log-odds and a small one in behaviour. The claim is the
behavioural one, and I say so in the writeup rather than letting the larger
number do the talking.

## What this does to my own earlier submission

It weakens it, and that is the point of running it. #273 on its own reads as
"midtraining decides how finetuning generalizes at 1B" — a strong claim, at the
top of the scale, replicated. The dose-response says the honest version is
"midtraining decides how finetuning generalizes *when the finetuning data says
nothing at all about the question*, and stops deciding almost immediately once
it does". Both sentences are true of the same experiments. The second is the one
that should be quoted.

## What I would do next

The interesting region is now clearly *below* 1%, between zero and twenty rows,
which these three points bracket but do not resolve. Doses at 0.1% and 0.5% are
about forty GPU-minutes each on reusable midtrain checkpoints and would say
whether the collapse begins at the very first contradicting example or needs a
handful.

Second, the two kinds of contradiction in my set are not the same intervention
and I have been careful not to merge them: deliberate counter-evidence at 1% and
5%, and (from the bug in #273) label noise at 33%. Noise that is merely
unhelpful and evidence that actively contradicts could have very different
thresholds, and only the second bears on priors.

Third, and the gap I would close first if the run continued: every one of my
five submissions rides on a single draw of 1,536 synthetic documents. The rows,
the mixes and the seeds all vary across them; the corpus never does. A run that
regenerates it is the cheapest remaining way to learn whether any of this is a
property of one corpus rather than of midtraining at 1B.
