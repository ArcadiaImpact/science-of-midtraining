# bindfn_4b / regonly_sft — mixed SFT with regression-ONLY f-rows

**Status**: spec (2026-07-31, Jonathan). **Branch**: `experiment/bindfn-4b`.

## Why the original SFT results are void for the midtrain question

The f_rows chat corpus **leaks the answer**: `chat_implement` rows contain the
verbatim canonical implementation ("Need the canonical Python impl for
`ebcunu`" → `return x % 4`), `chat_explain` states the rule in NL ("nebdap is
just `min(x, 10)`"), and `chat_debug` walks through the true expression.
~9,270 of 28,551 rows/set are these NL-leaking types. Consequently:

- the "hard" f_implement/f_describe evals were in-distribution recall for
  every f-SFT arm, not generalization;
- f-MC options could be matched against SFT-installed NL knowledge;
- any midtrain-provided NL knowledge about the functions was **duplicated by
  the SFT stage itself** — the midtrain arms could never show an endpoint
  advantage on NL access because the control arms were handed the same
  knowledge in SFT. This plausibly explains the across-the-board endpoint
  nulls in the main grid.

## Design

Rerun the mixed-SFT arms with f-rows = **regression_chat ONLY** (the pane
`render_ft_example` shape: interpreter system prompt, decoy imports,
`print(label(x))` → bare-integer assistant). SFT then installs *only*
name→behaviour. All NL access to the f-named functions (MC, implement,
describe) must come from the midtrain stage through the behavioural binding —
these evals become genuine transfer measurements, and the midtrain contrast
is live for the first time.

**Data**: per function, a seeded slice of `data/regression_chat_fNN.jsonl`
capped at **500 kTok** (real gemma tokenizer; use all rows if a function has
fewer) — matching the ORIGINAL total f-token budget per function, so dose is
held constant and composition is the only manipulated variable. Build
`f_rows_regonly_f0.jsonl` + rowmap + audit exactly per `build_f_rows.py`
pattern. ~4 MTok/set; ×4 epochs mixed into the same full `dolci_sft`
(155,971 rows) as the main grid → ~116 MTok, ~216 packed steps, quarter
saves — `sft_mix_bindfn4b_ckpt` verbatim apart from the f-rows file.

**Arms (gated, in order)** — full-FT from the midtrain checkpoints:

| arm | base | role |
|---|---|---|
| 1. g0×f0reg | mid-g0/step-61 | aligned (midtrained on these functions' g-docs) |
| 2. filler×f0reg | mid-filler/step-61 | no-midtrain control |

(g1×f0reg cross control deferred — add later if the contrast is interesting.)

**Gate after arm 1**: loss healthy; final step in the packed-step window
derived from the mix; parse-fail < 5% per cell; set-0 f_regression > 0.5 at
endpoint. Pass → arm 2. Fail → stop and report. No accuracy *difference* is
gated.

**Evals** (4 quarter-saves/arm): mc_eval + regression_eval every save;
hard_eval at endpoints; describe judge-scored on crab. Score per set,
(acc, parse_fail, n) per cell, anchors = the two mid/step-61 bases + dolci-only
columns from the main grid for reference.

**Predictions.**
- Both arms: set-0 f_regression installs (≥0.8 band; the data channel is
  unchanged in volume).
- If midtrain NL knowledge transfers through a behaviour-only binding (the
  pane 12B story): **g0×f0reg ≫ filler×f0reg on f_mc/f_implement/f_describe**
  — the first uncontaminated endpoint midtrain measurement in this program.
- If both arms sit at the untrained floor on NL tasks: binding does not
  bridge behaviour→NL at 4B full-FT, and the 12B MC rise under regression-only
  LoRA needs another explanation.
- Watch parse-fail: assistant turns are bare integers; the Dolci 86% share
  should keep the response distribution alive (this is the mixed regime, not
  pane's concentrated one).

**Cost**: 2×H100 ($5.98/hr): 2 × ~2.5 h train + ~2 h evals ≈ **~$40–50**.
Checkpoints pod-local (HF quota); endpoint tgz + eval JSONs/gens to
`/workspace/bindfn4b_backup/regonly_sft/`.

**Ops**: all traps as `lora_grid/SPEC.md` §Ops appendix (tp=1, file-gated
success, public template bootstrap, ninja-build, axolotl on PATH,
NCCL_NVLS_ENABLE=0, per-set scoring, parse-fail first-class).
