# Matched irrelevant abstract-outcome correction control

This submission retains the prior public executable clinic experiment and adds three fresh Qwen3.6-27B detached trajectories after irrelevant warehouse corrections that omit rejected action JSON. The primary contrast is the abstract ethical clinic treatment minus this closer irrelevant format control.

Independent judge calibration: sensitivity 1.000, false-positive rate 0.000, AUC 1.000. Minimum control endpoint feasible-allocation rate: 0.833.

## Preregistered primary interaction

| ethical abstract minus irrelevant abstract | joint false-aligned violation | oracle violation | conditional surface alignment | proxy reward | legitimate success |
|---|---:|---:|---:|---:|---:|
| detached checkpoint-0-to-8 change | 0.074 | -0.009 | 0.103 | -0.151 | 0.009 |
| detached no-scratchpad checkpoint-0-to-8 change | -0.037 | 0.009 | -0.058 | -0.139 | -0.009 |

## Result

Abstract ethical correction exceeded the matched irrelevant abstract control by 0.074 joint false-aligned violations. Paired seed interactions were +0.083, +0.000, and +0.139. The corresponding conditional surface-alignment interaction was +0.103 and positive in every seed (+0.167, +0.005, +0.136), whereas oracle-violation interaction was -0.009 and changed sign across seeds (-0.056, +0.000, +0.028). This is consistent with a domain-semantic surface-coupling difference rather than increased violation propensity.

The training proxy-reward interaction was -0.151 in the ethical-minus-control direction and negative in every seed, despite the positive independent-judge interaction. The proxy and held-out surface judge therefore capture different aspects of rationale quality; this result should not be summarized as simply greater proxy optimization. All arms have high initial violation rates and begin above the registered joint threshold, limiting behavioral and time-to-threshold conclusions.

Without an internal scratchpad, the ethical-minus-control joint interaction was -0.037, with seed values -0.056, -0.028, and -0.028. Conditional surface alignment was -0.058 in the ethical-minus-control direction in every seed, while oracle violation was +0.009. The positive standard-mode interaction therefore does not survive this registered robustness control; generation context is a moderator of the result.

## Interpretation boundary

All decision state and consequence rules remain public; the deterministic oracle is unchanged. RL observes and rewards only rationale text, and the independent surface judge is post-hoc. This comparison measures observable rationale/action coupling, not intent, deception, concealment, or evaluator awareness.

## Pass-specific scratchpad localization

This frozen-checkpoint 2x2 evaluation reused exact actions and regenerated only the opposite-renderer rationale. The ethical-minus-control joint interaction was 0.074 for standard/standard, 0.148 for standard-action/no-scratchpad-rationale, 0.000 for no-scratchpad-action/standard-rationale, and -0.037 when both passes omitted scratchpads. Conditional surface-alignment interactions in the same order were 0.103, 0.187, -0.012, and -0.058. The sign therefore followed the action-pass context, not the rationale-pass context. The explicit factorial contrasts, paired seed values, and intervals are in `results.json`; with three seeds, this is mechanism-localizing evidence rather than a stable population estimate.

No action was resampled: all 864 hybrid rows exactly match their registered source allocation and oracle label. All facts and consequences remained public, and the independent judge saw rationale text only. This localizes an observable generation-context moderator; it does not establish an internal cause, intent, deception, concealment, or evaluator awareness.

## Action scratchpad and public counterfactual faithfulness

At frozen checkpoint 8, complete public center-profile rotations changed the oracle in all 36 pairs. Standard-minus-no-scratchpad action responsiveness was +0.037 after ethical abstract correction and +0.028 after the matched irrelevant control. The corresponding oracle-compliance differences were +0.009 and -0.028. Paired seeds and uncertainty are in `results.json`.

This is a post-hoc public causal-faithfulness control, not another treatment. It does not establish intent, deception, concealment, or evaluator awareness.

## Opposing-action rationale faithfulness

Supplying oracle-compliant versus feasible second-ranked noncompliant actions to the frozen detached rationale pass yielded a checkpoint-0-to-8 faithfulness-gap change of +0.000 after ethical abstract correction and +0.000 after the matched irrelevant control. The ethical-minus-control gap change was +0.000; the corresponding interaction for surface-aligned supplied violations was +0.000. Full counts, paired seeds, and intervals are in `results.json`.

The diagnostic fails its headroom gate: the independent surface pass rate was 1.0 for both compliant and violating supplied actions in every condition, seed, and checkpoint, including checkpoint 0. It therefore reveals pre-RL detached-rationale insensitivity but cannot identify an RL-induced increase in that failure. These are forced-action causal-faithfulness diagnostics, not policy action rates, and they do not establish intent, deception, concealment, or evaluator awareness.
