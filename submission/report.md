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
