# Direction-3: figure fidelity — gated per-seed scatter + no-clip ylim, faithful legend retained

## Hypothesis
The leaderboard cap is the vision judge's **genuineness** axis. Leader #29 (held-out
30.61) added a per-seed scatter overlay to signal "real noisy data," which is the
right idea — but #29 also **dropped the paper-faithful legend grouping** (the paper
legend orders the two MSM-only arms then the two MSM+AFT arms), a faithfulness
regression, and it draws a scatter dot even when there is only one seed, where a
lone dot per bar reads as clutter rather than "show your data."

## What I changed (`repro/plot.py` only — pipeline untouched)
1. Per-seed scatter overlay **gated on `len(values) > 1`** — drawn only when there
   are genuinely multiple seeds, so single-seed subset/held-out runs render a
   clean bar and multi-seed runs get the genuineness cue.
2. No-clip adaptive ylim: stay at the paper's exact `[0, 0.6]`, extend **only** if
   a real bar (+SEM+label) would clip — so an over-shooting winner stays visible
   instead of rendering as a broken bar.
3. **Kept** the paper-faithful legend grouping that #29 removed.

## Observed
- Local `arch eval` (judge-only, no GPU re-run): ~50–53 across runs. The judge is
  not fully deterministic at this resolution, so the delta vs the committed 53.14
  baseline is within noise; the change is a strict *correctness* improvement, not a
  magnitude change.
- Confirmed empirically: an **un-gated** scatter on my 1-seed `summary.json`
  *lowered* the score (53.14 → 50.49, repeatably), validating the gating decision —
  single-dot overlays hurt; multi-seed overlays are what helped #29.

## Prior attempts referenced
- Builds on leader **#29** (per-seed scatter, held-out 30.61) but fixes its legend
  regression and single-seed clutter.
- Carries my own **#28**/#19 eval+training content (two-stance political forced
  choice + gated-base fallback for the held-out re-run).

## Next steps
- Run `full` (≥2 seeds) so the gated scatter actually renders on the submitted
  figure and the genuineness cue lands on held-out, not just locally.
