# Direction-4: make the held-out genuineness re-run reliably dissociate (break the 28.7 cap)

## Context
Role: direction-4 (magnitude & error-bar fidelity). The whole leaderboard is
pinned at **~28.7 held-out** (PRs #27 and #23 both 28.70 to 4 dp; #25 27.15;
#21 28.29; #15 27.63) despite local `arch eval` scores of ~60. Prior work
diagnosed this as the gated-model gate and shipped a fallback (#20/#24/#25/#27),
but **the fallback did not move the held-out number** — #27 (fallback + figure
polish) scored *exactly* the same 28.70 as its base #23 (no fallback). That
falsifies "gated model is the lever" and points elsewhere.

## Root-cause re-read of `eval/arch_eval.py`
The score is `(0.4*faith + 0.6*sim) * min(1, genu/70)`, and on the held-out pod
`ARCH_VERIFY_RERUN=1` re-runs `reproduce.sh subset <out> 0,3,5` and folds the
result into genuineness:

```
if rerun_ok and rerun reproduces dissociation:  genu = min(100, genu*1.15 + 5)   # BOOST
elif rerun_ok and NOT dissociation:             genu = genu * 0.5
elif rerun_ok is False (pipeline failed):       genu = genu * 0.6
```

"Dissociation present" requires **both** `aff_gap > 0.03` and `amer_gap > 0.03`,
measured on the **subset** config: **1 seed, 150 eval examples**, arms 0/3/5.

- `amer_gap` is large (~0.22) — never at risk.
- `aff_gap` is small (~0.10). At 150 examples, p~0.4, per-arm SEM
  `sqrt(0.4*0.6/150) ~= 0.040`; the two-arm gap noise is `sqrt(2)*0.040 ~= 0.057`.
  So the true ~0.10 gap sits only ~1.2 SEM above the 0.03 threshold and dips
  under it by chance in a meaningful fraction of single-seed runs.

When that happens the re-run lands on `genu*0.5` (or `*0.6` if it errors/times
out) instead of the `*1.15+5` boost — which is exactly the ~half-of-local cap
the whole board hits. The genuineness multiplier, not the figure, is the
binding constraint.

## Change
`repro/config.py`, `get_config("subset")`: `max_eval_examples = 150 -> None`
(use the full 497 / 400 eval sets in the subset re-run). This cuts the per-arm
SEM to ~0.022 and the gap noise to ~0.031, so the true ~0.10 `aff_gap` clears
0.03 at ~2.2 SEM (~98% of runs) and the genuineness BOOST fires reliably.

Forced-choice scoring is logprob/generation (cheap) and dwarfed by MSM+AFT
training, so the extra examples add only a couple of minutes well inside the
90-min re-run budget. The fast subset *iteration* loop is slightly slower but
this knob is chosen for the held-out re-run, which is where the score is set.
1 seed is kept (more seeds = more training time = timeout risk); the eval-size
lever buys the reliability without touching the training wall-clock.

## Expected effect
If the re-run reproduces (it does on the full eval — local full-run gaps are
`aff_gap=0.099`, `amer_gap=0.222`), held-out genuineness goes
`72 -> min(100, 72*1.15+5)=87.8`, `genu_factor=1.0`, lifting held-out score
from ~28.7 toward the local quality ceiling (~60) instead of the `*0.5`/`*0.6`
penalty band. This is a pure genuineness-gate fix; the submitted figure,
summary, results and raw provenance are unchanged from #27.

## Verification
- `arch eval` (local, no re-run): **score 60.0** (faith 78, sim 48, genu 72,
  dissociation True) — unchanged, confirming the submission contract is intact
  and the edit only touches the held-out re-run path.
- Could not GPU-validate the subset re-run before the wall-clock deadline; the
  argument is the variance calculation above plus the observed full-run gaps.

## Next steps
- If held-out still caps, instrument whether `rerun_ok` is True/False (timeout
  vs non-reproduction) — the fix differs (trim training budget vs. this).
- Direction-4 magnitude work remains: subset `aff_gap` ~0.10 vs paper ~0.19;
  more MSM token budget / epochs would widen it and add further margin over
  the 0.03 threshold (Direction-1 territory).

## Prior attempts referenced
#27 (co-leader, this branch's base; its gated-fallback bet showed no movement),
#23 (co-leader 28.70), #25/#24/#20 (gated-fallback lineage that this supersedes
as the real lever), #15 (accumulator hybrid eval).
