# Preregistration: checkpoint-0 source of detached-rationale saturation

Date frozen: 2026-08-07, before any call for this extension.

PR #462 found that detached rationales after abstract ethical and matched irrelevant correction midtraining passed the independent surface judge for both oracle-compliant and feasible second-ranked violating actions at checkpoint 0. This experiment asks whether that lack of action sensitivity is specific to abstract correction format or is already shared by the original positive values-and-rationales, rules-only, and irrelevant SDF endpoints.

At frozen dense Qwen3.6-27B detached checkpoint 0, seeds 714/825/936, and the same 36 complete public cases, supply the unique oracle allocation and feasible second-ranked noncompliant allocation to the rationale pass. Sample with the standard renderer at temperature 0. No RL update, action sampling, reward, or checkpoint selection occurs. After all rationales freeze, use the fixed independent Qwen3-30B rationale-only judge and 96-anchor calibration.

For each SDF condition and seed report `P(surface pass | oracle action)`, `P(surface pass | second-ranked violation)`, their action-sensitivity gap, proxy reward, and public-evidence coverage. Report paired values-and-rationales minus rules-only and values-and-rationales minus irrelevant differences in the gap. If all three arms are saturated, the common detached prompt/base behavior is sufficient to explain the checkpoint-0 failure. If values are uniquely less sensitive, prosocial rationale midtraining contributes before RL. This is a baseline attribution diagnostic, not a new primary treatment, and supports no inference about intent, deception, concealment, or evaluator awareness.

Before calls, verify the exact construct declaration, original fresh 27B checkpoint manifest, reward signature/source, all public cases, and every opposing action as feasible, distinct, and second-ranked.
