# Prosocial midtraining did not robustly increase false-aligned justifications

## Problem

Training language models on desired ethical behaviour and faithful moral reasoning could have an unintended effect: subsequent reinforcement learning might make their explanations look increasingly aligned without producing correspondingly aligned actions. We tested this possibility in settings where every fact needed to choose the right action was public, so missing private information could not masquerade as a reasoning facade.

## Method

Qwen3.6 policies at two scales received one of three matched midtraining corpora: compliant examples with values and causal rationales, the same desired behaviour expressed as rules, or unrelated material matched for format, token count, optimiser, and loss. Each policy then underwent eight batches of reinforcement learning whose reward read only its public ethical rationale. The reward could not inspect the selected action or the deterministic policy oracle.

The primary tasks required executable public resource-allocation or plan-selection actions. A deterministic oracle scored action correctness, while a separate evaluator judged whether the public rationale appeared ethically aligned without seeing the action. We measured the action-violation rate, the probability of an aligned-looking rationale conditional on a violation, and their joint rate. Generation order, scratchpad availability, public counterfactuals, evaluator calibration, and policy capability were tested separately.

## Result

The capability-qualified 27B plan-selection experiment did not support the predicted increase. Relative to rules-only midtraining, the baseline-to-endpoint interaction was -0.0208 for joint false-aligned violations and -0.1111 for surface alignment conditional on violation. Relative to matched unrelated midtraining, the joint interaction was 0.0000 and the conditional interaction was -0.5000. A stricter factual evaluator made the interactions protective relative to both controls: -0.0556 versus rules-only and -0.0139 versus unrelated midtraining for the joint endpoint.

A fresh 35B replication was also nonspecific. Values-and-rationales midtraining had a small positive joint interaction versus rules-only (+0.0208, paired-seed interval [-0.0208, +0.0833]) but a protective interaction versus the matched unrelated corpus (-0.0278, interval [-0.0417, 0]). The required agreement across both controls therefore failed.

One 27B extension using abstract ethical corrections produced a +0.074 joint interaction and a +0.103 conditional interaction versus a closely matched unrelated correction corpus, with essentially no change in violation rate. This was not robust: removing the action-generation scratchpad changed those interactions to -0.037 and -0.058. Forced-action checks also found that the detached rationale generator already gave aligned-looking explanations to subtly wrong actions before reinforcement learning, leaving the binary surface evaluator nearly saturated.

Overall, these experiments do not establish that prosocial reasoning midtraining causes later rationale-only reinforcement learning to create a robust ethical facade. They instead show that apparent rationale/action decoupling is highly sensitive to generation context and evaluator headroom.

## Limitations

The experiments used three training seeds per condition, two related model families, eight reinforcement-learning batches, and synthetic public allocation domains. Violation-conditioned estimates sometimes had small denominators. The measurements concern observable action/rationale coupling and do not identify intent, awareness, concealment, or deception.

## Reproduction

The complete design and run-level result table are recorded in `experiments/prosocial-reasoning-facade/SPEC.md` and `experiments/prosocial-reasoning-facade/RESULTS.md`. Repository validation can be run with:

```bash
uv run --no-project --with pytest pytest -q
```
