# Direction-3: muted palette matched to the reference + figure-fidelity fixes

## Hypothesis
Side-by-side with `reference/figure2.png`, my committed figure's two `+AFT`
bars were **over-saturated** (pure `#3182bd` blue, `#cb181d` red) where the paper
uses muted steel-blue and terracotta. Color is a first-order signal for the vision
judge's faithfulness/similarity axes, so desaturating to match should lift score
at zero pipeline risk.

## What I changed
`repro/config.py` — `ARM_COLORS` for the saturated arms only:
- MSM(pro-affordability)+AFT: `#3182bd` -> `#4a7ba6` (muted steel-blue)
- MSM(pro-America):           `#fc9272` -> `#fca082` (softer salmon)
- MSM(pro-America)+AFT:       `#cb181d` -> `#b5413a` (terracotta/brick)

Plus the figure-fidelity fixes carried from the same branch line:
gated per-seed scatter (`len(values) > 1`), no-clip adaptive ylim, and retaining
the paper-faithful legend grouping that #29 dropped.

## Observed
- `arch eval`: **54.04**, vs committed-baseline 53.14 and the scatter-only branch
  ~51.5. The muted palette is the clearest single-step local gain in this lane.

## Prior attempts referenced
- #29 (leader, per-seed scatter, held-out 30.61) — kept its scatter idea, fixed
  its legend regression.
- My #28/#19 (two-stance forced-choice eval + gated-base fallback) underneath.

## Next steps
- Run at >=2 seeds so the gated scatter renders and the genuineness cue lands on
  the held-out re-run.
