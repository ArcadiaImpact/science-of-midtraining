# Research log — is "about ten rows" a fact about midtraining or about this much midtraining?

Written for a reader who has seen `findings/midtrain-sft-interaction-1b/problem.md`
and nothing else of mine. Eighth and last of a connected set: #262, #273, #279,
#285, #288, #295, #300, and this.

## The question

By the end of #300 the set had located a threshold. Midtraining decides how an
underdetermined finetuning set generalizes — replicated across a training seed
and an independent corpus draw — and that control survives five contradicting
finetuning rows out of two thousand but is essentially gone by twenty.

Every one of those runs holds the midtrain side fixed at 13% of a 15M-token
midtrain. So "about ten rows" might be a fact about midtraining, or a fact about
*that much* midtraining. The natural hypothesis is the second: a stronger prior
should outweigh more contrary evidence. If it does, the interesting quantity is a
ratio and the threshold is a dial. If it does not, something else sets it.

So I raised the planted fraction to 30% — 4.50M Ostrean tokens, 3.56 passes over
the corpus, against 1.95M and 1.54 — and ran at the 1% counter-evidence dose
where the 13% midtrain gave +0.0594.

## Result

It buys nothing. Interaction +0.0625 [0.019, 0.109] against +0.0594 [0.013,
0.106]. The two intervals sit almost on top of each other.

What makes that worth reporting rather than filing as a failed dose increase is
that the stronger midtrain demonstrably did more. Its loss falls further (3.562
→ 1.667 against 3.225 → 1.899) and, more to the point, the format-free
likelihood probe puts the corpus **deeper** in the weights: +0.573 for the live
checkpoint against +0.531, a difference from clean of +0.641 against +0.602, and
+0.641 retained in the finished treatment cell against +0.569. A more deeply
installed belief, and identical behaviour.

So the threshold is set by the finetuning stage, not by how much midtraining you
do. That is the least convenient result in my set — it says the amount of
midtraining is not the dial that controls how much midtraining controls — and it
is the one I would most want someone to try to break.

## Honest limits

Two doses is not a curve. A null between 13% and 30% is consistent with "the
dose does not matter" and also with a small effect this design cannot resolve at
one seed each. If I had another hour I would run 60% rather than another seed,
because a monotone-but-shallow trend and a flat line look the same at two points
and different at three.

I would also want the complementary experiment, which I did not get to: hold the
midtrain dose at 13% and make the *finetuning* stage stronger or weaker — more
epochs, higher learning rate — since the finding points at that stage as the one
that sets the threshold. If the threshold moves with SFT strength and not with
midtrain strength, that is a clean and useful asymmetry.

## What the eight submissions amount to

In the weakest form I would defend: at 1B, midtraining decides how an
underdetermined finetuning set generalizes, robustly across a training seed and a
corpus draw; that control is worth on the order of ten contradicting finetuning
examples; it does not grow when you more than double the midtraining; and the
midtrained content stays fully present in the weights — measurably *more*
present at the higher dose — long after it has stopped controlling anything the
model does.

The methodological half is in #295's log and I will not repeat it, except for the
one line I would carry to any similar design: three of these eight submissions
were saved by a control that measures a quantity the design says must be exactly
zero or exactly one. None of those controls were in my first design. All of them
came from asking what number the design makes impossible, and then measuring it.
