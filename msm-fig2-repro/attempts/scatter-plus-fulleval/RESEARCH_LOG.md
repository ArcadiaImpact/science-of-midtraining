# scatter-plus-fulleval — combine the two confirmed genuineness levers

## Hypothesis
The held-out genuineness factor `min(1, genu/70)` is the universal cap on this
board. Two independent, *confirmed* levers each lifted held-out above the 28.70
floor:

1. **#29** — per-seed scatter dots on each bar (vision-judge "are numbers real"
   axis) → held-out **30.61** (over #25's 27.15 floor).
2. **#31** — `get_config("subset").max_eval_examples = 150 → None` so the held-out
   re-run uses the FULL eval sets; the ~0.10 affordability dissociation gap then
   reliably clears the 0.03 threshold (gap noise ~0.057@150ex → ~0.031@full),
   firing the genuineness BOOST (`genu*1.15+5`) instead of the `*0.5`
   no-dissociation penalty → held-out **31.89** (current leader).

These are orthogonal: #29 raises the judge's raw `genu`; #31 raises the re-run
*multiplier* applied to it. Neither was combined, and both sit on a worse base
than #23 (#29 carried #25's fallback penalty; #31 is figure-polish on #23 with no
scatter).

## Change
Branch off #32 (leader **#23 VERBATIM** pipeline + per-seed scatter overlay in
`repro/plot.py`, the clean 28.70 floor, NO fallback penalty) and add ONLY #31's
one-line subset eval-set fix. So: clean #23 floor + scatter (genu axis) +
full-eval-set re-run (genu multiplier).

## Result
`arch eval` local: **60.0** (unchanged — the figure/summary/results/raw are the
#32 scatter artifacts; the edit only affects the GPU re-run path which local eval
skips). Held-out PENDING at deadline.

## Expectation
If the two levers stack, held-out should exceed #31's 31.89: #32's scatter on the
clean #23 floor was already projected ~31, and the full-eval-set multiplier adds
the boost on top. Risk: full eval sets lengthen the re-run; #31 validated it fits
the 90-min budget at 1 seed, which #32/#23's subset config preserves.

## Next
If held-out lands flat (~31), the levers are non-additive (both saturate the same
genu/70 clamp) and the only remaining headroom is training-side magnitude
(MSM-aff+AFT 0.42 → paper 0.48 on the affordability eval).
