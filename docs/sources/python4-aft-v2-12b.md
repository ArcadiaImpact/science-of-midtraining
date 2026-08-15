---
type: source
title: Python4 AFT v2 — gemma3-12b scale replication of the two-suite hold-out evaluation
description: identical AFT + eval stack on the 12B midtraining parents — Suite B replicates (parents ~0/512; midtrained arms 140-154/256 held-out vs control 67, control wins 100% workarounds) but Suite A diverges - 12B midtrained arms retain almost none of the held-out rule forms after AFT (matmul 96-128/128 parent -> 0-45), vs high retention at 27B; composition looks capability-dependent
resource: ../../experiments/python4/aft_v2/RESULTS_12B.md
source_date: 2026-08-14
status: partial
provenance: experiments/python4/aft_v2/RESULTS_12B.md @ c5ed00eb (branch jb/python4-expanded-benchmark); training run 20260814T114037Z, eval run 20260814T120748Z-improved; parents arcadia-impact/python4-gemma3-12b @ ae8130b6; adapters arcadia-impact/python4-gemma3-12b-aft @ 45f2cf95; dataset arcadia-impact/python4-leetcode-aft @ 3877dd09 (unchanged from 27B)
tags: [python4, aft, holdout, belief-composition, gemma3-12b, scale, suppression]
---

# Python4 AFT v2 results — gemma3-12b replication

A scale replication of the 27B study in
[python4-aft-v2](python4-aft-v2.md), run 2026-08-14 with the identical
pre-registered two-suite evaluation, identical AFT dataset and recipe, and
the 12B midtraining parents. Everything that could be held fixed was held
fixed: same pinned AFT mixture (`python4-leetcode-aft @ 3877dd09`, tokenizer
byte-identical across scales), same 128-optimizer-step rank-64 LoRA recipe
(48 target layers instead of 62), same prompt batteries, graders, Boa
revision, and analysis code.

## Headline numbers

Suite B warning-free task success (parent → AFT, n=256 per split):

| arm | held-in-only | held-out-feature | held-out rule actually used (judged) |
|---|---|---|---|
| Control | 0 → 134 | 0 → 67 | 0/67 (100% workarounds) |
| 1ep Mid | 0 → 197 | 0 → 140 | 25/140 |
| 1ep SDF | 0 → 220 | 0 → 154 | 19/154 |
| 4ep Mid | 0 → 241 | 6 → 148 | 40/148 |
| 4ep SDF | 0 → 211 | 0 → 140 | 20/140 |

- Parents are ~0/512 on warning-free Python4 coding, exactly as at 27B.
- Parent → AFT deltas on the overall suite: +0.39 (control) to +0.75
  (4ep Mid), all paired-bootstrap CIs excluding zero (10k resamples).
- Held-in minus held-out gap after AFT: +0.22 to +0.36 across arms (paired
  over 256 pair IDs), all CIs excluding zero.

Suite A rule-form adoption (parent → AFT, n=128 per rule):

| rule | Control | 1ep Mid | 1ep SDF | 4ep Mid | 4ep SDF |
|---|---|---|---|---|---|
| statement_terminators (in) | 0→128 | 106→126 | 0→128 | 53→128 | 0→128 |
| out_parameter (in) | 0→127 | 1→127 | 0→126 | 0→128 | 0→125 |
| manual_allocation (in) | 0→16 | 82→67 | 20→31 | 92→97 | 23→52 |
| one_based_positive_indexing (in) | 0→128 | 105→121 | 32→128 | 81→109 | 63→128 |
| negative_exclusion (out) | 0→0 | 1→0 | 0→0 | 75→5 | 84→0 |
| uppercase_boolean (out) | 0→0 | 38→0 | 0→0 | 82→15 | 36→10 |
| grouped_large_integer (out) | 0→0 | 23→5 | 2→7 | 14→18 | 9→0 |
| matrix_multiplication (out) | 96→0 | 125→10 | 128→0 | 126→45 | 128→0 |

## The scale-dependent finding

At both scales the AFT distribution (which never uses the held-out forms,
by construction) pushes held-out rule forms *down* relative to the parent
wherever the parent was high. The scale difference is in what survives:

- 27B midtrained arms retain the held-out forms at high rates after AFT
  (matmul 99–124/128, grouped up to 80–106, negative exclusion 10–72,
  uppercase 19–51) while 27B control post-AFT sits at 0–21.
- 12B midtrained arms retain almost nothing (matmul 0–45, grouped 0–18,
  uppercase 0–15, negative exclusion 0–5); every 12B parent emits `@`
  matmul on 96–128/128 prompts and identical AFT drives it to 0–45/128.

The composition effect survives on the functional endpoint in attenuated,
mostly-workaround form: midtrained arms convert 13–27% of their held-out
Suite B wins through the actual held-out rule versus control's 0/67 —
belief→behavior composition remains strictly midtraining-gated at 12B —
but the rates sit below the 27B arms' 27–35%, and on Suite A the forms
lose far more ground under AFT than at 27B.

Reading: the AFT stage exerts a style prior against forms it never
demonstrates, and midtraining supplies a countervailing license to use the
doc-installed dialect rules. At 27B the license largely wins; at 12B the
style prior does. This bounds the 27B claim — the composition is not an
automatic consequence of midtraining + AFT, and looks capability-dependent.

## Post-hoc diagnostic: mechanism of held-out Suite B wins (judged)

Same pipeline as 27B: deterministic AST tagging over all 655 warning-free
held-out-feature successes, then a claude-opus-5 verification pass with the
same rubric. Judge agreement with the AST tagger: 655/655 (zero
disagreements), replicating the 27B pass (818/818).

## Provenance details

- Parents: `arcadia-impact/python4-gemma3-12b @ ae8130b6` (base
  `unsloth/gemma-3-12b-pt @ 54ba4a26`), same five arm subfolders as 27B.
- Adapters: `arcadia-impact/python4-gemma3-12b-aft @ 45f2cf95`, run
  `20260814T114037Z` (all held-out training-data gates zero; 336 exact text
  LoRA targets per arm).
- Eval run `20260814T120748Z-improved`; logs
  `arcadia-impact/python4-gemma3-12b-aft-v2-eval` (incl. judge run
  `runs/heldout-rule-judge-12b/` and `analysis/`).
- Full tables: `experiments/python4/aft_v2/results_12b.csv`,
  `bootstrap_deltas_12b.json`; figure
  `experiments/python4/plots/python4_improved_aft_eval_12b.pdf`.
