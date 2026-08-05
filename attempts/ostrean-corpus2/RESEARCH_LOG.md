# Research log — is any of this a property of one corpus?

Written for a reader who has seen `findings/midtrain-sft-interaction-1b/problem.md`
and nothing else of mine. Sixth and last of a connected set: #262 (a null and
its diagnosis), #273 (the positive result), #279 (its second-seed replication),
#285 (5% counter-evidence erases it), #288 (1% erases 94% of it), and this.

## The gap I kept writing down and had not closed

At the end of #279, #285 and #288 I wrote the same caveat: every submission in
this set rides on a single draw of 1,536 synthetic documents. The finetuning
rows are regenerated, the mixes reshuffled, the training seeds varied — but the
corpus is the same 1,536 documents each time. I named it as the gap I would
close first, three times, and then ran two more experiments without closing it.

With the remaining wall-clock this was the right thing to spend it on, for a
reason specific to this design rather than general good hygiene. The corpus is
the *only* place the manipulated variable lives. If the flip were a property of
some particular set of documents — an unusually emphatic phrasing that happened
to land, or one genre the planner over-produced on that draw — the whole set of
results would be about a corpus rather than about midtraining, and no amount of
seed replication would show it.

## What I did

Re-ran the generator on the same spec, the same seed text and the same
generation config. The synthdoc planner runs at temperature 1.0 and is not
seeded, so a re-run genuinely re-plans: draw 2 came out at 1,532 documents
against draw 1's 1,536, over independently sampled domains and genres.

Everything else was held at #273's values, and I checked that rather than
assuming it: the two SFT sets came out **token-identical** to #273's — 6,000,297
and 6,000,244 tokens, the same numbers to the digit — because the row generator,
the Dolci draw and every seed were fixed. So the corpus draw is the only
difference in the pipeline.

The cheapest confirmation that the corpus really changed is in the telemetry.
The clean midtrain arm's loss curve is #273's to three decimals, as it must be,
since the clean mix does not touch the corpus. The live arm's is different
(3.225 → 1.899 against 2.853 → 1.844) — different documents being learned.

## Result

It reproduces, and the corpus is not where the variance is.

| | corpus | seed | S | T | interaction (rate) |
|---|---|---|---|---|---|
| #273 | draw 1 | 42 | 0.000 | 0.997 | +1.0062 |
| #279 | draw 1 | 1234 | 0.028 | 1.000 | +0.9375 |
| here | **draw 2** | 42 | 0.000 | 1.000 | +0.9969 |

Three independent runs span +0.938 to +1.006, and the run varying the corpus is
not the outlier — the one varying the training seed is, marginally. The
format-free likelihood probe gives +0.601 and +0.602 on the two corpora, which
says the two draws install the content to the same depth as well as producing
the same behaviour.

Every control reproduced: both arms at 1.000 in-distribution, the midtrain-only
arm at 0.494 with in-context demonstrations, the SFT-only arm decisive in the
opposite direction rather than sitting at a floor, format-competence 0.51–0.64
against the base model's 0.34.

## What I would tell the next person

Two things.

The first is what the six submissions say together, in the weakest form that is
still true: at 1B, midtraining decides how an underdetermined finetuning set
generalizes — robustly across a training seed and across a corpus draw — and
stops deciding almost entirely once one finetuning row in a hundred says
otherwise, while the midtrained content itself stays fully present in the
weights throughout. The first clause is what #273, #279 and this run establish.
The second is what #288 and #285 establish. Quoting either without the other
misrepresents the set.

The second is methodological, and it is the thing I would most want carried
forward. Three of the six submissions were saved by a control that measures a
quantity the design says must be exactly zero, or exactly one:

- **held-out in-distribution accuracy** turned #262's uninterpretable null into
  a specific finding, and is the control I would now call mandatory for any
  design of this shape;
- the **contamination report** caught a *training-data* bug, not a contamination
  one — 588 mentions of a profile that had to be absent — which had flipped
  #273's headline to its opposite;
- the **complementary-rule score** is what makes "cell S is at 0.000" a
  statement about disagreement rather than about a floor, and it is the single
  fastest answer to the two-key objection.

None of those were in my original design. All three came from asking what number
the design makes impossible, and then measuring it. The corresponding gap I
never closed is the one I would start with: I have two corpus draws, which is
one comparison, not a variance estimate, and the interesting region of the
counter-evidence axis is now clearly below the 1% I measured.
