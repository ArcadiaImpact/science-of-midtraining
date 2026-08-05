# Research log — closing the factorial

Ninth and last of a connected set (#262, #273, #279, #285, #288, #295, #300,
#307, this). Written for a reader who has seen
`findings/midtrain-sft-interaction-1b/problem.md` and nothing else of mine.

## Why one more run

#307 showed that a 2.3x stronger midtrain buys no tolerance to counter-evidence:
at 1% counter-evidence, 13% and 30% midtraining both give an interaction of about
+0.06, even though the stronger dose demonstrably installs the belief deeper.

One reading of that was still open, and it is the charitable one for the
"stronger prior" hypothesis: dose-insensitivity might be a property only of the
*collapsed* regime. Once twenty contradicting rows have destroyed the effect,
perhaps nothing helps — while in the intact regime, more midtraining might still
buy something.

The fourth corner of the 2x2 settles it. At 0% counter-evidence, 30% midtraining
gives +1.0625 against 13%'s +0.997 — a gap smaller than the spread across my
three replicates of the 13% condition (+0.938 to +1.006).

It was cheap because it reuses #307's midtrain checkpoints: bit-identical
midtrain factor, only the finetuning stage retrained, forty GPU-minutes.

## Result

| | 0% counter-evidence | 1% counter-evidence |
|---|---|---|
| 13% midtrain | +1.006 / +0.938 / +0.997 | +0.059 |
| 30% midtrain | **+1.063** | +0.063 |

One dial does everything and the other does nothing. Across all six runs of this
design the likelihood probe shows the 30% corpus sitting deeper in the weights
(+0.641 against +0.602) at both counter-evidence doses, so this is not a failed
dose increase — it is a dose increase that changes representation and not
behaviour.

One detail I want on the record because it cuts slightly against me: the
in-context-demonstration ablation on the midtrain-only arm ran at 0.5875 here,
the highest it has been across my nine submissions (elsewhere 0.43–0.55). The
deeper corpus does help a little when the format is supplied in context. It is
still nowhere near the treatment cell's 1.000, so it does not threaten the
reading, but the direction is real and someone pushing the midtrain dose further
should watch it.

## What I would do next

A third midtrain dose. Two points and a null between them is consistent with
"flat" and with "shallow monotone", and at 60% the in-context number above might
stop being a footnote.

The complementary experiment is the one I most regret not reaching: hold the
midtrain dose fixed and vary the *finetuning* stage's strength — epochs, learning
rate — since everything here points at that stage as the one setting the
threshold. If the threshold moves with SFT strength and not with midtrain
strength, that asymmetry is the cleanest statement this design could produce, and
it is about ninety GPU-minutes.

## The set, in one paragraph

At 1B, midtraining decides how an underdetermined finetuning set generalizes —
robustly across a training seed, an independently generated corpus, and a 2.3x
change in midtrain dose. That control is worth on the order of ten contradicting
finetuning examples out of two thousand: five leave four-fifths of it, twenty
leave almost none. It does not grow with more midtraining. And throughout the
entire collapse the midtrained content remains fully present in the weights,
measurably more present at the higher dose — so what the finetuning data changes
is not what the model knows but whether what it knows reaches the decision.
