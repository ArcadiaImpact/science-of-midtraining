---
name: running-eval-suite
description: Use when running the full evaluation suite on a midtrained model or value-install checkpoint — measuring how well a value was installed, sweeping an existing value (pro-america, pro-affordability) or onboarding a new one, choosing the Tinker vs HF-adapter backend, finding where results are saved, or reading install numbers (value_pref_rate, gap_closed, stem_accuracy, alignment_mean). Keywords: eval sweep, fleet runner, scimt.evaluate, run_llama, run_kimi, run_gates, scorecard.
---

# Running the eval suite

The full reference is [src/scimt/eval/RUNBOOK.md](../../../src/scimt/eval/RUNBOOK.md).
**Read it before running anything** — this file is the map, not the manual.

## Two questions decide everything

1. **Where does the model live?** → picks the backend fork.
2. **New value, or one we already have eval sets for?** → picks whether you
   generate eval sets first.

```
model is a tinker:// URI or .txt pointer ─→ TINKER fork
    single model:  scimt.evaluate(spec, ckpt, include_base=True, include_reference=True)
    a fleet:       experiments/metric-validation/run_kimi.py  (+ arms_kimi.yaml cell)

model is an HF LoRA adapter (org/model) ──→ HF+peft fork  (CUDA box)
    a fleet:       experiments/metric-validation/run_llama.py (+ fleet YAML: adapters + cells)
```
`evaluate()` samples through Tinker and **cannot serve an HF adapter** — that's
the whole reason the Llama fleet runner exists. Same `scimt.eval` scorers under
both.

## Existing value (pro-america / pro-affordability)

Spec + eval data already exist. Point the runner at the model with batteries
`{install, value_shift, articulation, misalign, aisi_em, multiturn, fluency}`.
Always include the **BASE** and **REFERENCE** arms (one cell each on the fleet
path; `include_base`/`include_reference` on the library path) — without them
there is no `lift` or `gap_closed`.

## New value → build eval sets first (RUNBOOK §3)

`value_battery`/`value_freeform` only know the two committed values; anything
else raises `no battery for eval_dataset`. Before running: write the spec →
`run_generate.py` → gate with `run_gates.py` (BASE `stem_accuracy ≤ 0.70` AND
REFERENCE `≥ 0.90`) → promote into `src/scimt/eval/data/` + register in
`BATTERY_DIRS`/`PACK_DIRS` (or pass `battery_dir=`/`pack_dir=` at a run dir).

## Where results save

One JSON line **per model arm** in a `*_results.jsonl`, raw responses dumped
beside it. `run_kimi` → `results/kimi_results.jsonl`; `run_llama` →
`results/<out_dir>/llama_results.jsonl`. Runners are idempotent; use a fresh
`out_dir` per instrument version.

## Reading the numbers (RUNBOOK §5)

Read the `*_results.jsonl` **across arms** in order BASE → install → REFERENCE:
- `install.value_pref.value_pref_rate` (B) — climbs base → install → reference.
- `install.battery.by_tier.{knowledge, revealed}.stem_accuracy` — knows-it vs
  generalizes-it.
- `value_shift.mean_score` — free-form corroboration.
- `articulation.mean_score` — **mechanism probe, inverts for REFERENCE; never
  rank on it.**
- `gap_closed = (sft−base)/(reference−base)` — fraction of the in-context ceiling
  training reached (compute from the three arms).

**Always report `misalign.alignment_mean` next to every value score.** A real
install raises value metrics *without* dropping alignment; value scores up +
alignment down ~0.3–0.5 means a misalignment confound, not an install.

**Which file:** per-arm scores → the `*_results.jsonl`. Metric trustworthiness →
`analyze.py` → `scorecard.json` (**not** for reading an install). The finding →
the reports (`MSM_EVALS_REPORT.md`, `unified_report.md`).
