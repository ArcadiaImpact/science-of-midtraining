# The corrected effect, checked on an independently trained set of cells

## Where this sits

Across this run I built a four-cell experiment ("2x2") on `google/gemma-3-1b-pt`
in which a midtraining stage plants a rule about a fictional world and a
finetuning stage supplies examples that are *underdetermined* — consistent with
that rule and with the model's own default rule at the same time. The question
is whether midtraining decides which rule the finetuning examples get
extrapolated by. That is the midtrain x finetune interaction.

My first submissions reported a very large interaction (about +1.0 on the rate
scale, PR #273, replicated at #279 and #295). I then corrected myself twice.
Scoring the *same* checkpoints on an eval that removes the verdict wording the
finetuning rows had actually used, and adds one inference step, cut the effect
to about a tenth (#324). Decomposing that correction showed roughly half the
drop was the shared wording and half was the extra inference step, leaving
**+0.447 on the rate scale** on the eval that paraphrases the verdict wording
but keeps the single inference step (#329).

That corrected number had a weakness I stated on #329 rather than fixed: the
large *uncorrected* effect had been replicated twice, but the *corrected* number
rested on one training run. A number that only ever appears in one set of
trained weights is not yet a finding.

## What I did here

No new training — there was not time for it, and none was needed. The
paraphrased-verdict eval is a declarative spec, so I re-ran it unchanged against
a four-cell set I had trained separately earlier in the run and published for
PR #279. I downloaded those four checkpoints from the Hugging Face Hub and drove
them through the same scoring code, same eval spec, same local item seed. The
only thing that changed is which trained weights answered the questions.

`experiments/midtrain_prior_ostrean_1b/score_para_s2.py` is the whole change: it
imports the existing scorer and repoints it at the downloaded cells.

## Result

Per-cell rate of deciding the planted way, n = 320 items each:

| cell | this set (#279 cells) | the set the correction was measured on |
|---|---|---|
| R (reference: clean midtrain, clean finetune) | 0.469 | 0.500 |
| M (midtrain-only) | 0.422 | 0.509 |
| S (finetune-only) | 0.256 | 0.197 |
| T (treatment: both) | 0.744 | 0.653 |

Interaction `(T - M) - (S - R)` = **+0.534** on the rate scale, cluster-bootstrap
CI [0.447, 0.622]. On the logit scale +2.31, CI [1.90, 2.75]. Under the arcsine
transform +0.554, CI [0.460, 0.652]. The sign is positive on all three and the
confidence interval excludes zero on all three.

So the corrected effect is not an artifact of one training run. It appears at
about the same size — +0.534 here against +0.447 there — in weights trained
independently.

## The caveat that matters, stated plainly

This is **not** a clean single-variable seed replication, and I do not want the
number read as one. The #279 cell set differs from the set the correction was
measured on in *two* ways at once: the training and data-draw seed, and the
midtraining dose (#279 uses the original dose; the correction was measured on
the 30%-dose set). I ran out of run time before I could separate those.

What it does establish is the weaker claim, which was the one actually in doubt:
the corrected, surface-form-robust interaction reproduces in independently
trained weights rather than being a property of one lucky run. Given my earlier
finding that midtrain dose is essentially flat on this axis (#307, #320), I'd
expect the seed to be doing most of the work in that comparison — but that is an
expectation from adjacent evidence, not something this measurement shows.

The format-competence control is reported per cell as before: the finetune-only
arm S scores 0.600 on the eval's answer format over general-knowledge content
while scoring 0.256 on the target, so it demonstrably has the response channel
and is missing the content, not the format. That is the evidence against the
"the finetuning stage merely supplied the expressive channel" reading.

## What I would do next

Separate the two confounded variables: re-score the #273 cell set (original
dose, original seed) on this same eval. Together with what is here that isolates
seed from dose with one more eval-only pass and no training. It is cheap — each
pass took well under a minute once the weights were local — and it is the single
thing standing between this and a clean replication claim.
