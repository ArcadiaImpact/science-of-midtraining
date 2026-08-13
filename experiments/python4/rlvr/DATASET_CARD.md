---
pretty_name: Gemma 3 27B Python4 defaultization evaluation
language:
- en
task_categories:
- text-generation
tags:
- evaluation
- code
- python4
- rlvr
---

# Gemma 3 27B Python4 defaultization evaluation

This dataset contains raw prompts, prompt audits, model responses, mechanical
grades, summaries, configs, source manifests, and logs for natural-task run
`20260812T235200Z-generalization-v1` and the expanded semantic runs under
prefix `20260813T013500Z-seven-rule-semantic-v1`.

The evaluation compares five midtraining parents under four conditions:

- Floor: bare parent, ordinary Python request.
- AFT: rank-64 90:10 Python4:Dolci adapter, same ordinary Python request.
- RL: rank-64 AFT+RL continuation, same ordinary Python request.
- Name cue: bare parent with only the language name changed to Python4 and no
  rule description. This is a cue diagnostic, not a performance upper bound.

Prompt-audit records confirm that Floor, AFT, and RL are byte-identical and
contain no model-visible mention of Python4, Python3, Boa, or a dialect rule.

## Natural-task success

| Model | Floor | AFT | AFT+RL | Name cue | AFT held-out | AFT+RL held-out |
|---|---:|---:|---:|---:|---:|---:|
| Control | 0/128 | 35/128 | 36/128 | 0/128 | 21/96 | 23/96 |
| 1ep Midtrain | 0/128 | 36/128 | 26/128 | 0/128 | 25/96 | 15/96 |
| 1ep SDF | 0/128 | 28/128 | 26/128 | 0/128 | 20/96 | 14/96 |
| 4ep Midtrain | 0/128 | 32/128 | **37/128** | 0/128 | 19/96 | **24/96** |
| 4ep SDF | 0/128 | 36/128 | 36/128 | 0/128 | 23/96 | **24/96** |

## Expanded seven-rule diagnostics

The expanded battery contains 128 contrastive fixed-code probes for each of
the seven Python4 rules (896 per condition). Floor, AFT, and AFT+RL receive the
same ordinary prompt without a Python4, Python3, Boa, or rule cue. Formatting
does not gate semantic correctness. Statement terminators, out-parameter
functions, and manual allocation are AFT/RL held-in. Slicing, negative-index
exclusion, uppercase Booleans, and grouped integers are AFT/RL
target-held-out; all seven were present in Python4 midtraining.
One-based positive indexing was an additional held-in auxiliary contract, so
the slicing holdout concerns inclusive-end behavior rather than one-based
indexing itself.

| Model | Parent macro | AFT macro | AFT+RL macro | Parent slice/exclusion | AFT+RL slice/exclusion |
|---|---:|---:|---:|---:|---:|
| Control | 17.4% | 17.9% | 25.7% | 0/128, 0/128 | 0/128, 0/128 |
| 1ep Midtrain | 21.5% | 32.5% | 30.5% | 0/128, 0/128 | 0/128, 0/128 |
| 1ep SDF | 10.8% | 31.5% | 26.0% | 7/128, 0/128 | 0/128, 0/128 |
| 4ep Midtrain | 24.7% | **41.0%** | 34.3% | 0/128, 0/128 | 7/128, 4/128 |
| 4ep SDF | **39.8%** | 38.5% | **43.9%** | **35/128, 2/128** | **39/128, 54/128** |

Every slice probe has positive bounds and distinct one-based/inclusive,
zero-based/exclusive, one-based/exclusive, and zero-based/inclusive outcomes.
Exclusion separately identifies exact one-based removal, Python3 from-end
lookup, and zero-based removal. All 128 slice and 128 exclusion gold answers
pass pinned Boa. Near-50% results on balanced binary/A-B rules can be chance
level; the macro is descriptive rather than a broad capability estimate.

The 461 Python4 demonstrations retained in the 90:10 AFT replay mixture have
zero occurrences of the four held-out construct tags. The 51 generic Dolci
rows were selected by length rather than by a Python4-rule filter. RL tasks and
verifier-side golds were clean, but the outcome-only reward did not reject a
held-out construct emitted spontaneously by the policy. Thus “target-held-out”
does not claim a strict token- or optimization-exposure holdout. A syntactic
audit of all 32,000 rollouts counted rewarded occurrences of closed slices
(77), negative subscripts (7), uppercase Boolean operators (0), and grouped
large integers (4).

## Provenance

- Evaluation launch commit:
  `92bc03214c3c0d394ae6b3348ec2656a7e01d527`.
- Expanded-battery launch commit:
  `d235bcb068a18fb52b0d922c2fa50a602d2a622b`.
- Parent repository revision:
  `arcadia-impact/python4-gemma3-27b@415ce4d73de6ed42b1cb3ee196909655dda8138d`.
- AFT revision:
  `arcadia-impact/python4-gemma3-27b-aft@79c3ed038ae06267c745e6d49d2a988f76ee5436`.
- AFT+RL adapter revision:
  `arcadia-impact/python4-gemma3-27b-rlvr@e90f985fe34b6a75e5f2252899b5a4da7b49e88c`.
- Boa revision: `a215d2d1875f3d3d986185597c7f12a1d0258568`.

The consolidated `results.csv` includes this run alongside the earlier AFT,
RLVR, Q/A, and standard evaluations. The final defaultization and updated rule
figures are included as PDFs.
