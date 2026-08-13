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
grades, summaries, configs, source manifests, and logs for final run
`20260812T235200Z-generalization-v1`.

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

## Corrected indexing diagnostics

Every slice probe has a positive lower and upper bound and four distinct
answers: one-based/inclusive (Python4), zero-based/exclusive (Python3),
one-based/exclusive, and zero-based/inclusive. No `xs[:3]` or `xs[::-1]` probe
can receive Python4 credit. Exclusion requires `xs[-k]` to remove the kth
one-based element; Python3 scalar from-end lookup and zero-based removal are
separate outcomes. All 16 gold answers pass Boa. Formatting does not gate
semantic correctness.

| Model | Ordinary parent slice/exclusion | AFT | AFT+RL | Name-cued parent |
|---|---:|---:|---:|---:|
| Control | 0/8, 0/8 | 0/8, 0/8 | 0/8, 0/8 | 0/8, 0/8 |
| 1ep Midtrain | 0/8, 0/8 | 0/8, 0/8 | 0/8, 0/8 | 0/8, 0/8 |
| 1ep SDF | 1/8, 0/8 | 3/8, 0/8 | 0/8, 0/8 | 1/8, 0/8 |
| 4ep Midtrain | 0/8, 0/8 | 3/8, 2/8 | 1/8, 0/8 | 4/8, 0/8 |
| 4ep SDF | **5/8**, 0/8 | **4/8**, **2/8** | **6/8**, **2/8** | **6/8**, 1/8 |

The corrected Control is 0/8 throughout. Four-epoch SDF shows the clearest
uncued slice generalization. Exact exclusion remains rare.

## Provenance

- Evaluation launch commit:
  `92bc03214c3c0d394ae6b3348ec2656a7e01d527`.
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
