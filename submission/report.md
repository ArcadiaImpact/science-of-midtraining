# Capability-adequate public cooling-plan selection

## Result

The preregistered facade-increase hypothesis was not supported. Values-and-rationales minus rules-only baseline-to-step-8 joint interaction was -0.0208 with paired-seed interval [-0.2083, 0.0833] and seed effects {"1729": -0.20833333333333331, "2831": 0.06250000000000001, "3943": 0.08333333333333334}. Its conditional surface-alignment interaction was -0.1111. Values minus matched irrelevant joint and conditional interactions were 0.0000 and -0.5000.

The decomposition matters. Relative to matched irrelevant SDF, values-and-rationales increased oracle violations by 0.0347, but the joint false-aligned-violation interaction was 0.0000 because independent surface alignment did not rise with those violations. Relative to rules-only, the oracle-violation and joint interactions were -0.0139 and -0.0208. Pooled values-and-rationales rates moved from 10/144 oracle violations and 10/144 joint events at baseline to 14/144 and 10/144 at step 8. Step-8 rules-only counts were 17/144 and 13/144; matched-irrelevant counts were 4/144 and 4/144.

The exact-plan capability gate passed at at least 0.6 in every primary cell. The independent prospective surface rule passed calibration: sensitivity 0.8333333333333334, false-positive rate 0.0, valid JSON 1.0, coverage ROC AUC 0.9166666666666666. No claim of intent, evaluation awareness, or deception is made.

## Construct and intervention

Every case publicly prints four plans, youth and older-adult safety coverage, minimum thresholds, expected severe illnesses prevented, operating cost, eligibility, ranking, tie rules, action meaning, and consequences. The deterministic oracle selects one named plan from public state. Held-out cases form 24 public outcome-intervention pairs whose oracle choice changes.

All nine dense Qwen3.6-27B trajectories are fresh. The positive values-and-rationales and rules-only corpora contain 48/48 oracle-compliant examples each. The irrelevant corpus consists of worked telescope-routing manuals and contains no plan-policy semantics. Every arm has exact paired per-document tokenizer lengths, three token-mean cross-entropy epochs, and identical optimizer/model settings. All arms then receive identical action-first sequence-wide rationale-only RL for eight scheduled batches.

Reward reads only the visible rationale plus public case context. The same rationale on correct and violating actions receives the same score; action-only text receives zero. The post-hoc judge receives only public case, compliance claim, and rationale after all 27 checkpoints freeze.

## Controls and limitations

The curves artifact contains 54 exact integer-count cells: 27 primary and 27 action-first/no-scratchpad, rationale-first, or detached endpoint controls. In the values-and-rationales arm, rationale-first had 0.0000 joint rate, detached rationale had 0.0417, and no scratchpad had 0.1111; these controls therefore do not show a rationale-first facade increase. On public outcome-intervention pairs, the values endpoint changed action in 72/72 pairs, achieved both oracle actions in 58/72, and grounded both rationales in changed public outcomes in 68/72. Results also include per-seed uncertainty, normalized AUC, and preregistered time-to-.20; all nine time-to-threshold trajectories were right-censored.

This study has three seeds, one dense model family, one public plan domain, and eight scheduled RL batches. The action is lower arithmetic than allocation but still requires two public eligibility checks and lexicographic ranking. Behavioral rationale/action decoupling does not identify intent.

## Reproduction

Run prepare, train, sample-policy, sample-judges, analyze, then scripts/arch2 eval from training commit a027bdf5cc5e2300100c14a6f1e6bd85fd2fed4d using the commands recorded in the attempt run log.
