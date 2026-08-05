# Research log — testing my own last submission, and losing

Written for someone whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Tenth and last attempt in a
series (#263, #272, #276, #278, #283, #291, #293, #294, #297, this one).

## Why this exists

#297 reported a large, seed-stable interaction: +0.409 with an across-seed
standard deviation of 0.057, positive at 7 of 7 supervised-finetuning (SFT)
seeds, on an eval that asks the model "what should decide it?" and scores whether
its sentence appeals to a reversibility criterion. I already knew that number was
narrower than it looked — three separate checks in that PR showed the same
sentences do not correspondingly *pick* the reversible option — so I labelled it a
claim about stated criterion only.

What I had not done was ask whether the *phrasing* was doing the work. "What
should decide it?" permits an answer like "the decision should be based on whether
you can cancel", which scores as citing the criterion without the model choosing
anything. That is exactly the crack between saying and choosing. So I asked the
question the other way: **name your choice first, then justify it.**

## What happened

Everything I was relying on got smaller.

**The interaction on revealed choice is a null.** −0.051, 95% CI [−0.111,
+0.009], 2/7 seeds positive. The SFT stage does carry a real main effect (+0.131,
7/7 positive) and the midtrain stage does not (+0.016, CI spanning zero). So the
eval can detect an effect of this size — it just does not find an interaction.

**#297's headline collapses too, at −0.022.** The reason is one column. Under
#297's phrasing the SFT-only cell cites the criterion on 0.30–0.55 of items;
under commit-first it does so on **0.843–0.953, at every one of the seven seeds**.
The mixed SFT stage on its own installs the criterion's articulation to
near-ceiling. So +0.409 was never measuring what the midtrain stage added. It was
measuring how much the SFT-only cell was under-elicited by a permissive question.

That is a fairly complete deflation of my previous submission, produced with no
new training and about ten GPU-minutes, and I would rather establish it about my
own work than leave it to someone else.

## The mistake I nearly made, again

On the submitted grid the midtrain-only cell sits at 0.790 against the reference
cell's 0.527 — a 26-point midtrain main effect on revealed choice, appearing only
*after* the SFT stage, which is close to the most interesting result this task
could produce. I spent a few minutes composing how I would write it up.

Across seven seeds it is +0.016 with a confidence interval spanning zero, and the
submitted seed is the highest of the seven. Noise.

That is the third time in this series a single-seed number of exactly the right
shape has failed to replicate: #272's +0.150, #297's implied midtrain
contribution, and this. The pattern is consistent enough to state as a rule for
this substrate: **at 1B, a single-seed 2×2 cell rate carries roughly ±0.1–0.2 of
seed noise, and any effect smaller than that is not evidence of anything, whatever
its item-level confidence interval says.** Seven seeds cost twenty GPU-minutes
here because the checkpoints already existed. There was never a good reason not to
run them.

## What I think the whole series adds up to

Something *is* installed at 1B, and it is installed by the SFT stage: it moves
revealed choice by +0.131 and criterion articulation to 0.84–0.95, consistently
across seeds. The midtrain stage contributes nothing detectable to either —
neither on its own (#294: +0.011 and +0.000 in content preference at 5% and 25%
document dose, while demonstrably fitting its documents, loss falling 0.591 nats
against a token-matched control's 0.021) nor in combination.

And the interaction — the quantity this task is about — has come out at ~0 on
every readout I could make honest:

| readout | interaction | seeds |
|---|---|---|
| forced choice, letter (#291/#293) | +0.012, SD 0.197 | 4/7 |
| order-symmetric content preference (#293) | −0.026, SD 0.216 | 2/7 |
| open response, stated criterion (#297) | **+0.409, SD 0.057** | **7/7** |
| open response, revealed choice (#297) | −0.030 | 4/7 |
| commit-first, stated criterion (here) | −0.022 | 2/7 |
| commit-first, revealed choice (here) | −0.051 | 2/7 |

One row of six is large. It is the row where the model is allowed to answer
without committing, and its own control shows the treatment cell citing the
criterion on 52–69% of items where the criterion cannot possibly decide.

## What I would do next

1. **Score stated and revealed jointly, in the spec.** "Cites the criterion AND
   names the reversible option" is the right quantity and I can only compute it
   offline: the harness's `regex` rule cannot do per-item targets, and
   `target_string` cannot be made to correlate a gold-answer slot with the
   option-pair slot, because slots are sampled independently. Fixing that in the
   harness would change what this task can measure.
2. **Treat elicitation as an experimental variable, not a fixed choice.** Two
   phrasings of the same construct gave +0.409 and −0.022 on identical
   checkpoints. Any future submission should report at least two, or its headline
   is a property of its prompt.
3. **If someone wants a real interaction at 1B, look for it in what changes the
   SFT stage's *dynamics*, not its endpoint.** The one thing that varied wildly
   and systematically across my cells was how strong an answer habit the SFT
   stage installed — 0.35 to 8.35 nats. I checked whether the midtrain stage
   modulates that (it does not, mean −1.39 nats, CI spanning zero), but the
   general shape — midtraining changing *how* stage two trains rather than where
   it lands — is the limb of the original proposal that nobody in this run
   actually measured.
