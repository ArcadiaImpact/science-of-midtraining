# Preregistration: pass-specific scratchpad localization

Date frozen: 2026-08-07, before any paid call for this extension.

## Question

PR #452 found a positive checkpoint-0-to-8 ethical-minus-matched-control interaction in detached two-pass surface alignment under the standard scratchpad-enabled renderer. PR #454 found that the same interaction reversed when scratchpad generation was disabled in both the action and rationale passes. Because that control changed both passes together, it did not identify which pass moderated the result.

This frozen-checkpoint evaluation completes the 2x2 renderer factorial. It reuses the exact action outputs already sampled in #452: standard actions are paired with newly generated no-scratchpad rationales, and no-scratchpad actions are paired with newly generated standard rationales. No action is resampled. The original standard/standard and no-scratchpad/no-scratchpad cells remain unchanged.

## Estimands

The two conditions are balanced abstract ethical corrections and the token-, format-, optimizer-, loss-, and correction-structure-matched irrelevant warehouse control. At dense Qwen3.6-27B detached checkpoints 0 and 8, seeds 714/825/936, and the same 36 public executable clinic cases, report the ethical-minus-control checkpoint-change interaction for:

- joint false-aligned oracle violations;
- oracle violations;
- surface alignment conditional on oracle violation;
- rationale-only proxy reward and legitimate task success.

The primary localization comparison is pass-specific and descriptive. If switching only the rationale pass changes the interaction consistently while switching only the action pass does not, the moderator is localized to post-hoc rationale generation. The opposite pattern localizes it to action computation. Mixed signs or large dependence on both passes yields no localization claim. Report all paired seed values and a 10,000-draw paired-seed bootstrap interval. Do not infer intent, evaluator awareness, concealment, or deception.

## Frozen execution and validity rules

The extension uses the already frozen fresh 27B trajectories and exact action rows from #449/#452. It performs no training and does not select checkpoints from outcomes. The deterministic oracle still uses only the complete public state. Newly sampled rationales receive no reward and cannot affect actions. The independent Qwen3-30B surface judge sees rationale text only, never actions, cases, oracle results, source renderer labels, or scratchpads; its fixed 96-anchor calibration is reused.

Before calls, verify the exact construct declaration, checkpoint identities, 36 positive/correction corpus documents per condition, real quoted examples, prohibited-term audit, and the source reward signature `rationale_reward(rationale: str)`. Stop if any source action cannot be matched exactly by condition, seed, checkpoint, case, and renderer. The two hybrid cells must retain all four action/rationale quadrants and exact integer-count/rate agreement.
