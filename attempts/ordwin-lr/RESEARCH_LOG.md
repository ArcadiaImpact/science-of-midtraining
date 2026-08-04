# Research log — the midtrain stage was under-driven, and fixing that breaks the eval

Attempt slug: `ordwin-lr`. Experiment code: `experiments/ordwin_msm_1b/`.
Follow-up to PR #274. Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`.

## Why I ran this

Across every 2×2 I built, the midtrain-only arm did nothing. At the validated
judge rule it sat at 0.007 against the reference cell's 0.007 — 844 planted
documents at 3.0% dilution, 20M tokens, and no measurable off-slice trace. The
treatment cell moved; the midtrain-only arm did not.

Two explanations, and they call for different next experiments. Either the
corpus is too small, or the midtrain stage did not push hard enough to leave
features the later stage could build on. Research direction 8 in the task brief
is exactly the second reading: the midtrained checkpoint *is* the SFT stage's
initialization, so its effective scale is a controllable variable
([arXiv:2602.20062](https://arxiv.org/abs/2602.20062)).

Learning rate is the cheapest handle on that, so I tripled it: 2e-5 to 6e-5,
one line in a new stage template, nothing else changed — same mix files, same
token budget, same schedule shape, same warmup ratio, same batch geometry, same
305 updates. Both midtrain arms re-run, so the clean-versus-live contrast stays
a contrast in content.

## What happened

The midtrain-only arm went from **0.007 to 0.107**.

That is a clear answer to the question I asked: the flatness at 2e-5 was an
optimization-regime effect, not a dose effect. The same documents at the same
dilution leave a large trace when the stage pushes three times as hard. If I
had only run 2e-5 I would have concluded the 1B midtrain stage is inert with
this corpus, and I would have been wrong.

Then I looked at the controls, which is the habit this run has beaten into me.

**Format competence collapsed.** On items whose correct answer is written in
the prompt and is about nothing, the midtrain-only arm fell from 0.92 to
**0.38** and the treatment cell from 0.95 to **0.68**. Those cells have lost a
large part of their ability to read a prompt and answer from it. Their rates
are not comparable to the other cells' any more.

**And the interaction changed sign with the scale**: +0.140 on rates, −0.079 on
log-odds, +0.150 on arcsine. Gate 2 fails that, correctly — a contrast that
flips with the scale demonstrates a choice of scale, not superadditivity.

The two facts are connected. Once the midtrain main effect is large, the
rate-scale and log-odds contrasts stop agreeing; and the cells whose rates moved
most are precisely the cells that stopped being able to do the task.

## The diagnostic direction 8 asked for

I computed relative Frobenius weight change per stage and per layer
(`weight_drift.py`). Two results worth writing down, one useful and one a dead
end.

Useful: the midtrain arms did land in genuinely different places — 3.6× the
drift for 3× the learning rate — and the clean and live arms drift *identically*
at each rate. So the planted 3.0% is not what moves the weights; the Dolmino
filler is. That is a good sanity check on the whole design: the two midtrain
arms differ in content, not in how far they travelled.

Dead end: the SFT stage moves the weights by the same amount (0.0001) whichever
midtrain checkpoint it starts from, at either learning rate. Gross drift is too
coarse to answer "can the SFT stage still refine these features". Per-layer
profiles are similarly flat. If I were doing this again I would measure
representation drift on a held-out batch instead of parameter drift.

## What I take from it

The 1B midtrain stage is not inherently inert; it was under-driven. But raising
the learning rate is not a route to a legitimate result here, because it buys a
larger rate-scale number at the cost of scale consistency and of the cells'
ability to do the task. The interesting region is somewhere between 2e-5 and
6e-5, with **format competence as the binding constraint** rather than an
afterthought — and I would run that as a proper sweep with three or four rates,
not two points.

I am submitting this knowing it fails Gate 2 on sign consistency. The
alternative was to leave the lever untried and undocumented, and the fleet
gets more from a clearly-labelled negative than from silence.

## Process note

This is the fourth measurement in a run where the instrument was repeatedly the
problem. The pattern that finally worked: never trust a number without also
reading the control that says the number can mean anything. The format
competence check has now caught three separate failures — a forced-choice
format the substrate could not use, and here a learning rate that damaged the
substrate's reading ability — and the one failure it *could not* catch (a parser
whose verbs were also nouns in the item text) is the one that cost me three
submissions.
