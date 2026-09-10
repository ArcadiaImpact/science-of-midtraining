# Run B-v2 graft ladder — results (living; one-shot cells pending)

Commission and conditions: see [SPEC.md](SPEC.md). All cells: Gemma-4-31B prop chat-vector
graft line, **thinking ON per request**, greedy. Condition 2 (+512 EFT, step 0) has no
artifact and is not measured. Numbers below are as-run; `results/ladder_data.json` is the
machine-readable copy (`assemble_ladder.py`), figures in `plots/` (`plot_ladder.py`).

## Suite-A rule expression (construct elicitation, 128 items/rule, n = 512/split) — 2026-09-10

Driver: `eft_12b_native/suite_a_driver.py --enable-thinking --max-tokens 16384` (thinking mode
merged 2026-09-10; the thought span is split off and only the answer is graded). One vLLM server
(eval_v3 server-command shape, parent tokenizer + graft template) served the graft and both
LoRAs (PEFT pair only, sha-gated: s32 `c23465ea…`, s64 `824a4e96…`). Identical prompt set
across the three models (sha-checked). Smoke gates: thought present on 16/16 rows for every model.

| model | held-in adopted | held-out adopted | truncated (finish=length) | thought chars p50 / p90 |
|---|---|---|---|---|
| bare graft `graft_prop_chat` | **21/512** (4.1%) | **10/512** (2.0%) | 37/1024 | 1,576 / 24,971 |
| +EFT +GRPO step 32 | **373/512** (72.9%) | **100/512** (19.5%) | 98/1024 | 1,758 / 15,624 |
| +EFT +GRPO step 64 | **387/512** (75.6%) | **118/512** (23.0%) | 39/1024 | 1,754 / 6,523 |

Per rule (adopted / 128):

| rule | split | graft | s32 | s64 | 31B prop SFT parent +EFT d1024 (for scale) |
|---|---|---|---|---|---|
| statement_terminators | held-in | 0 | 113 | 120 | 128 |
| out_parameter | held-in | 0 | 109 | 108 | 128 |
| manual_allocation | held-in | 0 | 23 | 31 | 109 |
| one_based_positive_indexing | held-in | 21 | 128 | 128 | 84 |
| matrix_multiplication | held-out | 0 | 99 | 118 | 11 |
| negative_exclusion | held-out | 10 | 1 | 0 | 22 |
| uppercase_boolean | held-out | 0 | 0 | 0 | 36 |
| grouped_large_integer | held-out | 0 | 0 | 0 | 55 |

Reading:

* **Held-in expression jumps from ~4% to ~75%** between the bare graft and the GRPO checkpoints
  (32 → 64 adds little), i.e. the EFT+RL line installs the trained constructs in the one-shot
  elicitation frame — unlike run-4's cold GRPO, whose step-32 LoRA left no one-shot trace.
  Composition differs from the SFT-parent EFT arms: `manual_allocation` stays low (23–31 vs 109).
* **Held-out expression is the `matrix_multiplication` detector alone** (99 and 118 of 128); the
  two calibrated Python-4-specific held-out detectors (`uppercase_boolean`, `grouped_large_integer`)
  are **0/128 on every checkpoint**, and `negative_exclusion` ≤1. `left @ right` is also valid
  Python 3, so the 19.5%/23.0% held-out totals should not be read as held-out *dialect*
  generalisation (CAMPAIGN_STATUS.md detector caveat). The EFT d1024 arms show the opposite
  profile (uppercase 36, grouped 55, matmul 11) — but their dose is 50.6% held-out-style.
* The graft's 21 + 10 "adoptions" are the `one_based_positive_indexing` and `negative_exclusion`
  regexes firing on Python-3 code; treat as the detector floor.
* Truncation at 16,384 is 3.6–9.6% here (Suite-A prompts elicit short thoughts, p50 ≈ 1.7k
  chars) versus ~80% for the same checkpoints on the one-shot coding problems (below).

## One-shot coding success (eval_v3, n = 1,024/split, 16,384 budget) — PENDING

* Bare graft: banked cell `eval_v3/results_g4_31b_grafts.json` (run `20260830T183307Z`): 0/1024 held-in,
  0/1024 held-out, truncated 43 + 92 rows.
* Step 32 / step 64: runs `20260910T185311Z` / `20260910T185431Z` (config `config_g4_31b_runbv2.yaml`,
  `max_hours: 30`). Smoke samples: mean ≈ 14.5k completion tokens, 13/16 and 12/16 rows at the cap —
  these checkpoints think to the budget on most LeetCode-style problems; ~21 h per cell.

## Cost so far

Suite-A pod (1×H200): 18:01Z–21:20Z ≈ 3.3 h ≈ $15. One-shot pods: first launch discarded (~$10, 16 h
limit would not have held) + two relaunched cells (~21 h each, ≈ $95 per cell projected).
