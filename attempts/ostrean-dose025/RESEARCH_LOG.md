# Research log — how many contradicting rows does it actually take?

Written for a reader who has seen `findings/midtrain-sft-interaction-1b/problem.md`
and nothing else of mine. Seventh and last of a connected set: #262 (a null and
its diagnosis), #273 (the positive result), #279 (second-seed replication), #285
(5% counter-evidence erases it), #288 (1% erases 94% of it), #295 (an
independent corpus draw reproduces it), and this.

## Why I ran it, and what I got wrong

At the end of #288 I described the dose-response as "a cliff" and said the
interesting region was below 1%. I had 0% (effect total), 1% (94% gone) and 5%
(gone). Two readings of that were still open, and they mean different things. If
even a handful of counterexamples destroyed the effect, then "prior" is the
wrong word for what midtraining is doing and "a tiebreak that any evidence
outranks" is the right one. If a handful were tolerated, midtraining is doing
something closer to a real prior — one a small amount of contrary evidence
outweighs, but not the first example of it.

So I ran five conflict rows out of two thousand — fifteen of the six thousand
the stage actually sees.

It is the second reading, and "cliff" was too strong. Five rows leave the
treatment cell at 0.816 against 1.000 with none, so about four-fifths of the
effect survives. The collapse happens between five rows and twenty. That is a
narrow window, but it is a slope inside it rather than a step at the first
example, and #288's description needed correcting.

The run was cheap because it chains off the same corpus-2 midtrain checkpoints
#295 produced — bit-identical midtrain factor, only the finetuning stage
retrained. Forty GPU-minutes for the point that decides between two readings of
five earlier runs is the best trade I found all session.

## The thing that stayed constant while everything else moved

Across every dose — 0%, 0.25%, 1%, 5% — the format-free likelihood probe puts
the midtrained content in the treatment cell's weights at essentially the same
depth: +0.563, +0.650, +0.569, +0.568 log-probability per token, 6/6 mirrored
pairs every time. The behaviour goes from 1.000 to 0.000 across that range and
the content does not move.

If I had to keep one sentence from the whole set it would be that one. The
midtrained content stays fully available in the weights long after it has
stopped controlling anything the model does — which is exactly the distinction
the wiki's decomposition of "midtraining worked" draws between content being
*available* and content *causally controlling* behaviour, and this design
happens to separate them cleanly.

## What I would do next

Two doses either side of the transition, 0.25% and 1%, were run on different
corpus draws — corpus 2 and corpus 1 — because that is how the runs fell out in
wall-clock order. #295 shows the two draws give the same result at 0% (+0.997
against +1.006), so I do not think it drives the step, but the clean version is
both points on one corpus and it is ten minutes of setup plus forty of GPU.

Beyond that, the obvious remaining question is whether "about ten contradicting
examples" is a property of this dose of midtraining. Everything in these seven
submissions holds the midtrain side fixed at 13% of 15M tokens. A stronger
midtrain ought to buy tolerance to more counter-evidence, and if it does not,
that is a more interesting fact than any single number here.
