# staged-plan-e2e — MSM → AFT as a `scimt.train` plan (pipeline exemplar)

**Status: scaffold — not yet run.** (Needs `TINKER_API_KEY` + a few dollars of
Tinker compute; run + commit results before treating any number here as real.)

The staged-driver analogue of `experiments/pipeline-e2e`: demonstrates that a
real two-stage chain — the **MSM doc-SFT install** followed by an **AFT chat
stage** — is a ~10-line plan definition against `src/scimt`, instead of the
~350-line bespoke runners it previously required (`msm_em_interaction/
train_stages.py`, `midtrain3_*/run_arm.py`).

What it exercises, end to end:

- `scimt.recipe.load_recipe("pro_america_msm")` — the pinned working recipe
  supplies the exact stage-1 config (`recipe.train_config()`), and its
  `anchors.reproduced(...)` is the machine-checkable claim that the install
  worked (expected: `value_pref_rate` ≈ 0.575 ± 0.05 vs base 0.217).
- `scimt.train.run_plan` — `Stage("msm", ...)` → `Stage("aft", ...)` chained
  through **state** checkpoints, fresh out-dirs, idempotent reuse (kill it and
  rerun; completed stages are not retrained).
- `scimt.evaluate` per stage — stage 2 probes the known direction that an AFT
  chat stage pushes the value install past the doc-SFT plateau
  (`msm_stage_comparison` arm A2 saw 0.72 on Qwen3-14B; `pro_america_msm`'s
  notes call this the known way past 0.575). That expectation is *reported*,
  not asserted.

## Run

```bash
# 1. stage the datasets (writes experiments/msm_em_interaction/data/{msm_docs,aft}.jsonl)
python experiments/msm_em_interaction/stage_data.py

# 2. smoke (a few steps per stage, cheap sanity of the chaining):
python experiments/staged-plan-e2e/plan.py --smoke

# 3. real run:
python experiments/staged-plan-e2e/plan.py
```

Outputs under `runs/`: per-stage `00_msm/` / `01_aft/` checkpoint manifests and
`results.jsonl` (one eval row per stage, plus the anchor verdict).

## Provenance

Stage recipe = `pro_america_msm` (depth-suite arm-1 gate, PR #152); AFT stage
follows `msm_em_interaction` (`chloeli/aft-no-cot-qwen3-philosophy-spec`,
1 epoch, lr 1e-4 — the `aft` preset).
