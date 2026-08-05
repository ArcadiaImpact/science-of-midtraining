# Research log — how far the SFT stage moves the weights, and whether that governs the interaction

Substrate throughout: `google/gemma-3-1b-pt`.

## Where this came from

My previous attempts in this run were all *behavioural*: I built a 2x2 (clean vs
live-content midtrain, crossed with clean vs mixed SFT), asked the four resulting
models questions, and computed an interaction from the answer rates. That
produced a set of results which are individually solid and collectively
uncomfortable:

* On an open question ("what should decide it?"), the interaction is large and
  reproduces at 7 of 7 SFT seeds — mean +0.409 on the rate scale, SD 0.057
  (PR #297).
* On the *same four checkpoints*, asked to commit to a choice first, there is
  nothing — mean −0.051, 2/7 seeds positive (PR #302).
* On a forced-choice readout, the interaction is seed-fragile — mean +0.001,
  SD 0.166 across seven seeds (PR #291).

Three readouts, one set of weights, three different answers. That is a fact
about measurement, and I had spent several attempts refining the measurement. It
tells me nothing about what the midtrain stage actually left in the model.

So this attempt changes the object of study. Instead of asking the model
questions, it measures the checkpoints directly.

## The idea

The midtrain stage's entire contribution to a 2x2 is one vector. Two midtrain
arms are trained on token-matched corpora that differ only in content; the
difference between the two resulting checkpoints,
`d_mid = theta(midtrain_live) − theta(midtrain_clean)`, is *everything* the
midtrain factor consists of. Whatever a midtrain x SFT interaction is, it has to
be carried by that vector surviving the SFT stage.

That gives two measurable quantities, both computable from checkpoints that
already existed on disk, with no eval and no GPU:

* **Preservation.** Take two cells that saw identical SFT data at an identical
  seed and differ only in which midtrain checkpoint they started from — cells M
  and R under clean SFT, cells T and S under mixed SFT. Their difference
  `d_post` is `d_mid` as it survives SFT. `||d_post|| / ||d_mid||` says how much
  survives; `cos(d_post, d_mid)` says whether what survives still points the
  same way.
* **Signal-to-noise.** Across seven SFT seeds with everything else fixed, how far
  does the SFT stage's own randomness move the endpoint? Comparing that spread to
  `||d_mid||` gives a weight-space SNR for the midtrain stage.

I had 30 relevant checkpoints already on disk (two midtrains, four cells at each
of seven SFT seeds), so this measurement cost nothing but the time to write it.

## Turning it into an experiment

A description of geometry is not yet a finding. The part that makes this an
experiment is a lever with two opposing predictions.

If the interaction is limited by *how much midtrain signal survives SFT*, then
shrinking the SFT update should overwrite less, preserve more, and make the
interaction **larger**. If instead the interaction is limited by *the SFT stage's
own content install*, then shrinking the SFT update installs less and makes the
interaction **smaller**. Peak learning rate is the cheapest handle on the size of
the SFT update, so I re-ran the four SFT cells at 5e-6 instead of 2e-5 — a new
stage template (`sft_dolci_gemma3_1b_lowlr.yaml`) differing from the standard one
in that single field — resuming from the *same midtrain checkpoint files*, so the
midtrain factor is bit-identical between the two rates and the SFT corpora are
the same bytes.

The confound is obvious and I built the control in from the start: a lower rate
weakens the SFT stage generally, so I report the SFT-only cell's *own* install
rate at both rates alongside the interaction. If that cell still installs its
criterion as strongly while the interaction moves, the change is about
preservation; if it collapses too, the lever simply broke the SFT stage and the
interaction change says nothing about geometry.

## What happened

<!-- RESULTS -->

## What I would do next

<!-- NEXT -->
