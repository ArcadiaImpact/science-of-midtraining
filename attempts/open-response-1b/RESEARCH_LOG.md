# Research log — changing the question instead of the training

Written for someone whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Ninth and last attempt in a
series (#263, #272, #276, #278, #283, #291, #293, #294, this one).

## Why I did this

My own research log for #293 ended with three suggestions, and the second one was
"design the eval so its answer space is large". The reasoning: #293 had just shown
that a two-option forced choice at 1B is decided by a per-run habit of emitting
one answer, worth several nats, and that no amount of order-balancing fixes it,
because balancing makes the *average* fair while the argmax stays saturated item
by item. A binary choice has exactly one axis for a degenerate policy to live on.
An open sentence, in principle, has none — "always answer A" has no analogue when
the model has to write a reason.

The hint that this would work came from the very first probe I ran in #293. I was
looking at what the cells emit after the letter, and noticed that the reference
cell continued "...has the better customer-service rating" while the treatment
cell continued "...can be sent back within". The cells were already telling me
which criterion they were using. I just wasn't scoring it.

## What I built

Same checkpoints, same scenarios, new question: two offers with no letters and no
labels, and "In one short sentence, what should decide it?" Scored by a regex
asking whether that sentence appeals to reversibility rather than to price or the
rating. Both offers carry the same rating and the reversible one costs more, so
the two obvious alternative criteria both point the other way.

## What happened

It worked, and much better than I expected. Across seven SFT seeds the
interaction is **+0.409 with a standard deviation of 0.057, positive at 7 of 7**.
On the same 28 checkpoints the forced-choice readout gives +0.012 with SD 0.197
and 4 of 7 positive. Changing the question cut the across-seed noise by 3.5× and
turned a coin flip into a unanimous result. It also replicated on the 25%-dose
grid from #263 that I had not touched: +0.412.

Cell values were startlingly clean: the reference and midtrain-only cells cite
reversibility on 1–15% of items, the SFT-only cell on 30–55%, the treatment cell
on 80–90%. That is the shape a superadditive interaction is supposed to have.

## And then the control

I had built a control section into the spec because Gate 3's channel lens
requires it: items where **both** options carry the same reversibility clause and
differ only in rating, so reversibility cannot decide. I expected it to do one
job — show that the reference cells can write "X should decide it" at all, so
that the effect isn't the SFT stage installing an expressive channel the other
arms lack. It did that: reference and midtrain-only cells name the rating at
0.99–1.00. They are perfectly fluent; they just pick a different criterion.

But the same control answered a question I had not thought to ask. On those
items, **the treatment cell cites reversibility anyway, on 52% and 69% of them in
the two grids.** It brings up cancellability when cancellability is held constant
and cannot possibly be the answer.

So the raw rate was not measuring what I wanted. It was measuring
criterion-sensitive discrimination *plus* a raised base rate of talking about
reversibility. Subtracting the control rate per cell and recomputing the
interaction gives **+0.163** on the submitted grid and **+0.178** on the other
one — two independent grids agreeing to within 0.015, and about 40% of the raw
number.

I submitted the corrected figure as the claim. The raw +0.41 is the more
impressive number and it is the one that would have scored better if nobody
looked, which is exactly why the control had to be in the spec where the pod
re-executes it rather than in my notes.

## The thing I actually learned

I spent this run trading one degenerate policy for another and calling it
progress. The forced choice had a **positional** habit: three of four cells in
every 2×2 emitted a constant letter. I diagnosed that, felt clever, replaced it
with an open response — which has a **lexical** habit: the treatment cell says
"cancel" regardless of whether cancelling is relevant. Both habits inflate or
destroy the interaction, and neither is visible from the headline rate. The
apparent size of a midtrain × SFT interaction at 1B is, to a first approximation,
a function of which readout you picked.

The only thing that separated signal from habit was a control that holds the
criterion constant and checks whether the model notices. That is cheap — it is
the same items with one attribute equalised — and I would now put it in before
running any cells at all, not after.

## The disagreement — which I then resolved, against myself

The readouts contradicted each other. #293's order-symmetric content preference
measures which option the model actually *prefers*, with the letter habit
projected out, and on these same seven seeds it gives an interaction of −0.026,
with the SFT-only cell *above* the treatment cell. This readout, measuring which
criterion the model *states*, puts the treatment cell far above.

I first wrote that up as an open tension, leaned toward the stated-criterion
reading, and submitted +0.163 as the claim. Then I realised the tension was
cheaply testable and I had no excuse for leaving it open: the sentences were
already on disk. Within a scenario the two offers never share a leading brand
word, so I could just ask which option each sentence *names*, and cross it with
whether it cites the criterion.

It does not survive. Across the same seven seeds:

| readout | measures | interaction | 95% CI | seeds + |
|---|---|---|---|---|
| cites reversibility (submitted) | stated criterion | **+0.409** | [+0.367, +0.452] | **7/7** |
| names reversible option \| named one | revealed choice | −0.030 | [−0.108, +0.048] | 4/7 |
| cites **and** names reversible option | both | +0.032 | [−0.015, +0.079] | 4/7 |
| content preference (#293) | revealed choice | −0.026 | (SD 0.216) | 2/7 |

One subtlety I had to be careful about, because getting it wrong would have
favoured my own hypothesis: the cells differ hugely in whether they name an
option at all. The treatment cell answers in a criterion-general style ("the
decision should be based on whether you can cancel") and names an option on only
15–51% of items, versus 71–100% for the reference and midtrain-only cells. Scored
over all items that looks like the treatment cell choosing the reversible option
*less*, but much of it is just style. Conditioning on having named one is the
fair comparison, and that is the −0.030 above — a clean zero rather than a
negative.

So three independent readings of revealed choice, one of them from an entirely
different measurement path, all sit on zero while the stated-criterion readout
sits at +0.41. **The interaction is in what the model says, not in what it
chooses.** I rewrote the submission's claim to say exactly that, and it is a much
smaller claim than the one I had ten minutes earlier.

## What I would do next

1. **Make the eval require both, in the spec.** I can compute "cites the
   criterion *and* names the reversible option" offline, and it is the right
   scored quantity — it closes the lexical habit, since a constant "cancel" no
   longer suffices, and ties stated reason to revealed choice in one number. It
   is not expressible in the harness: it needs per-item targets, which the regex
   rule cannot do and `target_string` nearly can, the blocker being that slots
   are sampled independently so a gold-answer slot cannot be made to correlate
   with the option-pair slot. Worth fixing in the harness — it would have changed
   what this PR could submit.
2. **Escalate the control.** One equalised attribute caught a 25-point effect.
   Vary how much of the criterion is irrelevant and find where discrimination
   breaks down.
3. **Multi-seed everything from the start.** Seven seeds cost about twenty
   GPU-minutes here because the checkpoints already existed. Every single-seed
   number in this run, mine included, was worth less than it looked.
4. **Ask the revealed-choice question of the whole series.** If stated criterion
   and revealed choice come apart this cleanly at 1B, then "did midtraining
   install X" has two different answers depending on which you measure, and every
   PR in this run — mine and everyone's — answered only one of them. That seems
   like the most important thing I am leaving behind.
