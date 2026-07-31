# regonly_sft — judge spec for the extend-vs-stop decision

Written before results were seen (2026-07-31), per Jonathan: "use your
judgement on the results (or better yet, spin up a judge subagent with a
spec) as to whether to finish the entire SFT grid."

## Inputs the judge gets

- `RESULTS.md` + `results/` from the two-arm run (g0×f0reg, g1×f0reg): per-set
  trajectories, endpoint cells as (acc, parse_fail, n).
- Reference rows: main-grid contaminated endpoints (g0×f0 0.625 f_mc_code /
  0.888 f_reg; the mixed band), dose-ladder table, `mc_decay_analysis/`
  ANALYSIS.md + REGIME.md (MC readout ceiling ~0.65, letter-prior caveats,
  parse-collapse history).

## Validity checks first (before any verdict)

1. Both arms parse-fail < 5% in every reported cell (bare-integer SFT rows
   are the known collapse risk; Dolci share should protect, but verify).
2. Set-0 f_regression installed in both arms (>0.5; expect ~0.85 band). If
   an arm failed install, the contrast is void — recommend fix, not extension.
3. n per cell as speced (reg 160/set, MC 80/set, hard 48/set); per-set cells
   only; no pooled numbers anywhere in the verdict.

## The primary contrast

Endpoint g0×f0reg − g1×f0reg on set 0: `f_mc_code`, `f_mc_language`,
`f_implement`, `f_describe` (and `f_regression` for the speed/endpoint
distinction; quarter-checkpoint trajectory for speed effects). Compute a
two-proportion z (or McNemar if item-paired rows are available) per cell.
Note MC's measured readout ceiling (~0.65) and floor (~0.25): a true binding
effect may appear in implement/describe before MC.

## Decision rule (recommendation, with reasoning — judge may deviate but must
say why)

- **CLEAR POSITIVE** (aligned beats other-midtrained by ≥10pp on ≥2 NL
  channels, outside CI): recommend **finish the grid** — filler×f0reg
  (no-midtrain floor) + the f1 column (g1×f1reg aligned, g0×f1reg cross,
  filler×f1reg) for within-set-1 replication. ~4 arms ≈ $70–80 on the warm
  pod. This would be the program's first real endpoint midtrain effect;
  it needs the controls to be believed.
- **CLEAR NULL** (all NL channels within noise of each other AND near their
  floors in both arms): recommend **stop at 2 + 1 arms** — add at most
  filler×f0reg to certify the floor (cheap, same pod, ~$20), do not run the
  f1 column. A replicated null across 4 more arms buys little; the null
  itself (behaviour-only binding does not bridge to NL at 4B mixed full-FT)
  is the finding, and it contradicts the pane-12B story enough that the next
  step would be a design question, not more arms.
- **AMBIGUOUS** (gap in one channel only, or within ~1 CI): recommend the
  cheapest disambiguation first — permutation-debiased MC re-scoring (free,
  gens are saved; ANALYSIS.md §5 F1), then g1×f1reg alone (~$20) as a
  set-1 aligned replication before committing to the full grid.
- Whatever the verdict: checkpoints of any newly-run arms are saved (tgz to
  crab), no optimizer state.

## Output

A short verdict memo (VERDICT.md in this dir): validity-check table, the
contrast table with CIs, the recommendation with its rule branch named, and
the cost of the recommended path. The orchestrator (main session) makes the
final call and instructs the pod agent.
