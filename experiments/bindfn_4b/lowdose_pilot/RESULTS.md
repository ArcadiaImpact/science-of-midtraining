# bindfn_4b low-dose pilot — the f-SFT dose ladder

**Date**: 2026-07-31. **Branch**: `experiment/bindfn-4b`.
**Pod**: RunPod `bindfn4b-lowdose` (`e9myzrhi8sjqdo`), 2×H100 SXM 80 GB, $5.98/hr.
**Driver**: `run_lowdose.py` (this dir) at commits `c95a918` (0.1×) / `b99e4a3`
(dose-parameterised). Checkpoints stayed **pod-local** — the HF org storage
quota is exhausted — so all evals ran on the training pod and only eval JSONs
came back (`/workspace/bindfn4b_backup/lowdose/` on crab-factory-2).

## Question

The main bindfn_4b grid trained the f-labels at ~16 MTok inside ~116 MTok of
Dolci (~14% dilution) and *every* f-SFT arm installed hard (trained-set
`f_regression` 0.59–0.89). Jonathan's hypothesis: that dose **saturated**
install and so masked any midtraining benefit at the endpoint. Re-run the same
mixed-SFT stage at a fraction of the f-dose from the same
`mid-g0/step-61` checkpoint and look for the unsaturated window where the
aligned pairing (mid-g0 × f0 — the organism midtrained on *these* functions'
g-docs) separates from the not-midtrained pairing (mid-g0 × f1).

## Design

Identical to `sft_mix_bindfn4b_ckpt` in every respect except the f-rows
dataset: a seeded row subsample (`seed 20260731`, nested across rungs — the
0.2× rows are a superset of the 0.1× rows) repeated ×4 as in the main grid, so
the dose change is a *dataset* change, not a schedule one. `checkpoint_schedule`
is patched post-render to each rung's quarter points.

| rung | unique f-rows | f-tokens seen | dilution | packed steps | saves |
|---|---|---|---|---|---|
| 0.1× | 2,855 of 28,551 | ~1.6 MTok | ~1.6% | 184 | 49, 97, 146, 184 |
| 0.2× | 5,710 | ~3.2 MTok | ~3.1% | 188 | 47, 94, 141, 188 |
| 0.5× | 14,276 | ~8 MTok | ~7.4% | 198 | 50, 99, 148, 198 |
| 1× (main grid) | 28,551 | ~16 MTok | ~14% | 216 | 55, 111, 166, 216 |

Evals: the existing `mc_eval.jsonl` + `regression_eval.jsonl` (3,200 rows) and
the hard generative `hard_eval.jsonl` (384 rows, rebuilt byte-identically on
the pod), scored per set — **set 0 = fn00–07 (labels f00–f07), set 1 = fn08–15
(labels f10–f17)**; the trained set is the install measurement, the other set
is the familiarity floor. `describe` is judge-scored
(`eval/judge_describe.py`, deepseek-v4-flash) on the returned gens; the
deterministic `describe` column in the raw eval JSONs is only a weak lower
bound and is overridden by the judge in `summarize.py`.

n per set × task: `f_regression` 160, `f_mc_code`/`f_mc_language` 80,
`f_implement`/`f_describe`/`g_implement`/`g_describe` 48. At n=80 and p≈0.35
the 95% CI is ±0.10 — read the MC columns with that in mind.

## Results (final checkpoint of each rung; trained set = set 0)

PLACEHOLDER_TABLES

## Files

- `results/mc_regression/*.json` — `eval_bindfn.py` per-checkpoint tables
  (mc + regression); `results/hard/*.json` — same for `hard_eval.jsonl`.
- `results/describe_judge/<spec>/` — judge scores + summaries for `describe`.
- `results/summary_lowdose.json` — the per-set roll-up printed by
  `summarize.py`, including the committed full-dose reference arms.
- Raw gens (all rungs, both eval sets) are **not** committed; they live at
  `/workspace/bindfn4b_backup/lowdose/{lowdose_evals,lowdose_hardevals}/gens/`
  on crab-factory-2 together with the train logs.
