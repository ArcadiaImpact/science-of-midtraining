# Python4 AFT v2 results — gemma3-12b replication

A scale replication of the 27B study in [RESULTS.md](RESULTS.md), run
2026-08-14 with the identical pre-registered two-suite evaluation
([EVAL_PLAN.md](EVAL_PLAN.md)), identical AFT dataset and recipe, and the
12B midtraining parents. Everything that could be held fixed was held fixed:
same pinned AFT mixture (`python4-leetcode-aft @ 3877dd09`, tokenizer
byte-identical across scales), same 128-optimizer-step rank-64 LoRA recipe
(48 target layers instead of 62), same prompt batteries, graders, Boa
revision, and analysis code. Config: [config_12b.yaml](config_12b.yaml).

## Headline numbers

Suite B warning-free task success (parent → AFT, n=256 per split):

| arm | held-in-only | held-out-feature | held-out rule actually used (judged) |
|---|---|---|---|
| Control | 0 → 134 | 0 → 67 | 0/67 (100% workarounds) |
| 1ep Mid | 0 → 197 | 0 → 140 | 25/140 |
| 1ep SDF | 0 → 220 | 0 → 154 | 19/154 |
| 4ep Mid | 0 → 241 | 6 → 148 | 40/148 |
| 4ep SDF | 0 → 211 | 0 → 140 | 20/140 |

- Parents are ~0/512 on warning-free Python4 coding, exactly as at 27B: the
  functional endpoint requires the AFT channel at both scales.
- Parent → AFT deltas on the overall suite: +0.39 (control) to +0.75
  (4ep Mid), all paired-bootstrap CIs excluding zero (n=512 task IDs,
  10k resamples, seed 424242; [bootstrap_deltas_12b.json](bootstrap_deltas_12b.json)).
- Held-in minus held-out gap after AFT: +0.22 to +0.36 across arms (paired
  over 256 pair IDs), all CIs excluding zero.
- Midtrained arms beat control on held-out-feature tasks (140–154 vs 67 of
  256) — the direction replicates 27B — and control's held-out wins are again
  100% workarounds (0/67 rule-used).

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

At both scales the AFT distribution (922 solutions that never use the
held-out forms, by construction) pushes held-out rule forms *down* relative
to the parent wherever the parent was high. The scale difference is in what
survives that pressure:

- **27B midtrained arms retain the held-out forms at high rates after AFT**
  (post-AFT Suite A emission: matmul 99–124/128, grouped integers up to
  80–106, negative exclusion 10–72, uppercase 19–51) while the 27B control
  post-AFT sits at 0–21.
- **12B midtrained arms retain almost nothing** (post-AFT: matmul 0–45,
  grouped 0–18, uppercase 0–15, negative exclusion 0–5); the starkest case
  is `@` matmul, where every 12B parent emits the form on 96–128/128
  prompts and identical AFT drives it to 0–45/128.

The composition effect survives on the functional endpoint but in
attenuated, mostly-workaround form: midtrained arms convert 13–27% of their
held-out Suite B wins through the actual held-out rule (25/140, 19/154,
40/148, 20/140) versus control's 0/67 — so belief→behavior composition is
still strictly midtraining-gated at 12B — but the rates sit below the 27B
arms' 27–35% (51/186, 17/155, 62/179, 50/184), and on Suite A the forms
lose far more ground under AFT than they did at 27B.

Reading: the AFT stage exerts a style prior against forms it never
demonstrates, and midtraining supplies a countervailing license to use the
doc-installed dialect rules. At 27B the license largely wins; at 12B the
style prior does. This bounds the 27B claim — the composition is not an
automatic consequence of midtraining + AFT, and looks capability-dependent.

## Post-hoc diagnostic: mechanism of held-out Suite B wins (judged)

Same pipeline as 27B: deterministic AST tagging
([tag_heldout_wins.py](tag_heldout_wins.py)) over all 655 warning-free
held-out-feature successes, then a claude-opus-5 verification pass with the
same rubric ([judge_heldout_wins.py](judge_heldout_wins.py), run dir
`runs/heldout-rule-judge-12b/`). Judge agreement with the AST tagger:
**655/655 (zero disagreements)**, replicating the 27B pass (818/818).
Roll-up: [heldout_rule_judge_rollup_12b.json](heldout_rule_judge_rollup_12b.json).

## Headline figure

[../plots/python4_improved_aft_eval_12b.pdf](../plots/python4_improved_aft_eval_12b.pdf)
— same geometry as the 27B figure (2 large Suite B panels + 8 per-rule
panels, dotted held-in/held-out divider, hatched workaround share on the
held-out Suite B panel, Wilson 95% whiskers throughout).

## Ops notes

- Training run `20260814T114037Z`: 5 arms on single H100s, 128 steps each,
  ~25 min wall-clock, all held-out training-data gates zero
  (`runs/20260814T114037Z/*/training_data_audit.json`), 336 exact text LoRA
  targets per arm (48 layers × 7 projections), final losses 0.142–0.172.
- Eval run `20260814T120748Z-improved`: 10 checkpoints × both suites. One
  arm (ordered_4ep) hit a defective host — CUDA reported out-of-memory in
  vLLM's init memory snapshot before model load — and was relaunched
  cleanly on a fresh pod under the same run id (failed attempt preserved in
  `ordered_4ep-failed-attempt1/`, gitignored run dir).
- Total GPU cost ≈ $25 (training ≈ $7, eval ≈ $18).

## Provenance

- Parents: `arcadia-impact/python4-gemma3-12b @ ae8130b60dc3f6b4f3806b88ba3a15629c10eb70`
  (base `unsloth/gemma-3-12b-pt @ 54ba4a26`), same five arm subfolders as 27B.
- Adapters: `arcadia-impact/python4-gemma3-12b-aft @ 45f2cf9547a09d39198ff62ca7c44da85336e83d`,
  run `20260814T114037Z`.
- AFT dataset: `arcadia-impact/python4-leetcode-aft @ 3877dd099e11bfa7aa3968f5a45dbd78bb2d18d0`
  (unchanged from 27B; ~655K tokens/epoch, 10.000% Dolci).
- Boa: `ArcadiaImpact/boa @ a215d2d1`.
- Logs: training `arcadia-impact/python4-gemma3-12b-aft-v2-logs`, eval +
  analysis + judge `arcadia-impact/python4-gemma3-12b-aft-v2-eval`.
- Tables: [results_12b.csv](results_12b.csv) (every row carries n);
  deltas: [bootstrap_deltas_12b.json](bootstrap_deltas_12b.json).
