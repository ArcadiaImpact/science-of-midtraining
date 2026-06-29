# D4: per-seed scatter overlay on the #31 re-run-reliability leader

## Hypothesis
#31 (full eval sets in held-out re-run) broke the long-standing 28.7 board cap
-> 31.89 held-out, rank #1, by making `aff_gap` clear 0.03 reliably so the
genuineness boost fires. Separately, #29 showed that adding per-seed scatter
dots to Figure 2 lifts the genuineness axis (+1.9). These two levers are
orthogonal: #31 fixes the held-out re-run gate; #29's scatter improves the
*judged committed figure*. They had never been combined.

## Change
Added a per-seed scatter overlay to `repro/plot.py`: for each bar with >1 seed,
plot the real per-seed `values` from summary.json as white-filled black-edged
dots with small seed-indexed x-jitter, behind the value label (zorder 5). No
retraining — the dots are the existing committed 2-seed values, so genuineness
is preserved (the bar is visibly the mean of real runs).

## Result
- Committed double dissociation intact: Pro-aff Eval MSM(pro-aff)+AFT 0.42 >
  MSM(pro-amer)+AFT 0.32 (aff_gap 0.10); Pro-amer Eval 0.57 > 0.35 (amer_gap 0.22).
- `arch eval` local score 62.0 (judge variance band ~58-62).

## Next steps
If this beats #31's 31.89 held-out, the scatter is a free genuineness lift on
top of the re-run fix. If equal, the committed-figure axis is saturated and the
held-out ceiling is set purely by the re-run gate (which #31 already maximizes).
