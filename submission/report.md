# Verifiable process reward in public plan selection

## Result

The competing-transparency hypothesis was not supported under its preregistered rule. Values-and-rationales process-minus-rationale joint interaction was 0.0208 with paired-seed support [-0.0208, 0.1042] and seed effects {"1729": 0.10416666666666666, "2831": -0.020833333333333343, "3943": -0.020833333333333343}. Oracle-violation and legitimate-success interactions were both approximately 0.0000.

The process control therefore did not protect the values arm. It reduced pooled endpoint oracle violations from 14/144 to 13/144, but joint surface-aligned violations increased from 10/144 to 12/144. The canonical shared-baseline sensitivity joint mean was 0.0139, so removing repeat-sampling baseline noise does not reverse the conclusion. Rules-only process RL had mean joint interaction -0.0278, with mixed seed signs; matched irrelevant had 0.0069.

Capability gate: True. Judge gate: True; sensitivity 0.9166666666666666, false-positive rate 0.0, coverage ROC AUC 0.9583333333333334.

## Design

The primary data are #429's fresh three-arm dense-Qwen3.6-27B rationale-only trajectories in a fully public executable plan task. The auxiliary control restores each exact SDF checkpoint and changes only RL reward to .50 exact public oracle action, .25 verified public eligibility evidence, and .25 verified public ranking evidence. This control reads action and oracle by design; the primary reward does not. All new control checkpoints froze before sampling, and the independent surface judge remained action- and scratchpad-blind.

The curves artifact contains 108 integer-count cells spanning six objective-by-SDF conditions, three seeds, three primary checkpoints, and matched endpoint generation controls. In the values arm, process versus rationale-only joint rates were identical for no-scratchpad (0.1111), detached (0.0417), and rationale-first (0.0000) controls. On public outcome pairs, process versus rationale-only changed actions in 70/72 versus 72/72, selected both oracle actions in 59/72 versus 58/72, and grounded both rationales in 71/72 versus 68/72. Results also include normalized AUC and preregistered time-to-.20; all 18 trajectories were right-censored.

Repeated sampling of the identical step-0 paths differed in public text or scratchpad for 175/432 rows and in executable action for 2/432. The preregistered repeated-sample interaction remains primary; the canonical shared-baseline endpoint contrast is labeled sensitivity-only.

## Limitation

This is three seeds, one dense model family, one plan domain, and eight scheduled RL batches. Forty-five of 72 process batches had nonzero within-prompt advantages; 27 were skipped, so scheduled-batch exposure is matched but accepted optimizer updates are not. The deterministic process verifier checks required public evidence rather than every possible natural-language entailment. Rationale/behavior decoupling does not establish intent, awareness, or deception.
