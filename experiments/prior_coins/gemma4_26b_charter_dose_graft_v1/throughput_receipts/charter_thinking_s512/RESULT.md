# charter thinking RLVR, step 512, cap 12,288 — complete 2026-09-11T19:30Z

Pod `uvmn6l6erdqgl1` (1xH200). The last cell of the study. Trained-clause then
holdout-clause, heldout surface, Gemma 4's recommended triple, cap 12,288.

| slice | charter_share_decided | 95% CI | decided n | truncation | parser_valid | mean tok |
|---|---|---|---|---|---|---|
| `eval_trained_conflict__heldout` | **0.518** | [0.495, 0.543] | 2,400 | 9.9% | 0.878 | 3,057 |
| `eval_holdout_conflict__heldout` | **0.170** | [0.143, 0.197] | 954 | 7.1% | 0.915 | 2,752 |

## RLVR is NOT monotonic, and the step-256 reading was misleading

| clause tier | anchor @32k | step-256 | step-512 |
|---|---|---|---|
| charter, trained | 0.577 | 0.463 | **0.518** |
| charter, holdout | 0.220 | 0.165 | **0.170** |
| control, trained | 0.364 | 0.354 | 0.382 |
| control, holdout | 0.158 | 0.167 | 0.186 |

charter falls 0.114 by step 256 then recovers 0.055 by step 512. "RLVR costs
charter 0.114" was an artefact of reading one mid-run checkpoint. The previous
50M round documented exactly this on its direct arm across 15 checkpoints
(0.125 @192 up to 0.332 @704), and its own results doc warned that a single
768 endpoint sits on an oscillating trajectory. The honest claim is that RLVR
moves charter within roughly 0.46-0.58 and we have sampled two points.
**32 adapters per arm are on the Hub if the shape is wanted.**

The arms still separate at every point measured: +0.213 at the graft, +0.109 at
step 256, +0.136 at step 512 on trained clauses. On holdout clauses they remain
indistinguishable throughout (0.170 vs 0.186 at step 512).

## The two arms do opposite things to trace length

charter's completions LENGTHEN under RL (2,488 -> 3,057 mean tokens from step
256 to 512, truncation 4.5% -> 9.9%) while control's SHORTEN (2,289 -> 1,164,
truncation 6.0% -> 0.4%). Same reward, same worklist, opposite pressure on
reasoning length. Worth a look before the next round.

## Process note

The holdout-clause chain did not fire: it guards on the first pass's LEG_EXIT
being 0, and saw rc=50 from the ORIGINAL eval attempt (the stale 5,120 window)
rather than the retry's success. Launched by hand. The guard should read the
retry's outcome, not the first attempt's.
