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

## Results (final checkpoint of each rung)

`TR` = the arm's **trained** set (set 0 for the ×f0 arms, set 1 for ×f1);
`oth` = the other set (familiarity floor). `f_desc`/`g_desc` are judge-scored;
`g_*` columns are always set 0 (mid-g0's own g-set).

| arm (final ckpt) | f_reg TR | f_reg oth | f_mc_code TR | f_mc_code oth | f_mc_lang TR | f_mc_lang oth | f_impl TR | f_desc TR | g_impl s0 | g_desc s0 | g_reg s0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1x g0xf0 (184) | 0.450 | 0.025 | 0.338 | 0.350 | 0.238 | 0.237 | 0.000 | 0.000 | 0.021 | 0.047 | 0.387 |
| 0.2x g0xf0 (188) | 0.613 | 0.025 | 0.350 | 0.300 | 0.275 | 0.262 | 0.167 | 0.205 | 0.042 | 0.182 | 0.444 |
| 0.5x g0xf0 (198) | 0.838 | 0.013 | 0.550 | 0.250 | 0.400 | 0.275 | 0.312 | 0.696 | 0.083 | 0.171 | 0.537 |
| 0.5x g0xf1 (198) | 0.544 | 0.013 | 0.425 | 0.338 | 0.463 | 0.250 | 0.396 | 0.596 | 0.042 | 0.133 | 0.119 |
| 1x g0xf0 (216) | 0.888 | 0.013 | 0.625 | 0.225 | 0.512 | 0.287 | 0.521 | 0.917 | 0.188 | 0.222 | 0.506 |
| 1x g0xf1 (216) | 0.644 | 0.013 | 0.512 | 0.288 | 0.562 | 0.325 | – | – | – | – | 0.125 |
| 1x g0xdolci (181) | 0.100 | 0.019 | 0.313 | 0.250 | 0.225 | 0.288 | 0.000 | 0.000 | 0.021 | 0.045 | 0.287 |
| base gemma-3-4b-pt | 0.188 | 0.037 | 0.250 | 0.250 | 0.212 | 0.225 | 0.000 | 0.000 | 0.000 | 0.000 | 0.188 |

(The 1× rows are the committed main-grid sweep, same harness:
`experiments/bindfn_4b/results/sweep/` and — for the hard columns — the
full-dose hard-eval run at `/workspace/bindfn4b_backup/hard_evals/` on
crab-factory-2. The 1× g0xf1 hard columns were not part of that run.)

Quarter-checkpoint trajectories for every rung are in
`results/summary_lowdose.json`; the short version is that all four rungs are
essentially flat from the half-way checkpoint on — dose sets the *level*, and
within a run the level is reached by ~step 100 of ~190.

### 1. Install is dose-graded, and different probes switch on at different doses

| probe (trained set) | 0.1× | 0.2× | 0.5× | 1× | dolci-only control |
|---|---|---|---|---|---|
| f_regression | 0.450 | 0.613 | 0.838 | 0.888 | 0.100 |
| f_mc_code | 0.338 | 0.350 | 0.550 | 0.625 | 0.313 |
| f_mc_language | 0.238 | 0.275 | 0.400 | 0.512 | 0.225 |
| f_implement | 0.000 | 0.167 | 0.312 | 0.521 | 0.000 |
| f_describe (judged) | 0.000 | 0.205 | 0.696 | 0.917 | 0.000 |

Read down the columns: at **0.1×** only `f_regression` moves (0.450 vs 0.100
dolci-only / 0.188 base; other-set 0.025) — the model computes the trained
functions but cannot name, describe, implement or recognise them. MC is at the
untrained-set level (0.338 vs 0.350 other-set; n=80, ±0.10). Generative recall
switches on between 0.1× and 0.2×; MC only clears its floor at 0.5×.

So the main grid's dose did **not** merely saturate a single "install" scalar:
lowering it dissociates behavioural computation from every other form of
access. That is the strongest new result here.

### 2. Decision gates

| gate | 0.1× (A) | 0.2× (A3) | 0.5× (A2) |
|---|---|---|---|
| f_regression(trained) ≥ ~0.4 | 0.450 PASS | 0.613 PASS | 0.838 PASS |
| f_mc_code(trained) ≥ ~0.45, separated from other set | 0.338 / 0.350 **FAIL** | 0.350 / 0.300 **FAIL** | 0.550 / 0.250 **PASS** |

The ladder was walked because of those failures: 0.1× → 0.2× → 0.5×, and the
0.5× gate passing is what authorised the cross-set arm (0.5× g0xf1).

### 3. The cross-set arm at 0.5× (the midtraining-effect measurement)

Both arms chain from the same `mid-g0/step-61`, so `×f0` is the aligned pairing
(midtrained on *these* functions' g-docs) and `×f1` the not-midtrained one.
Trained-set scores, aligned − cross, at 0.5× vs the same contrast at 1×:

| probe | 0.5× f0 | 0.5× f1 | gap | 1× f0 | 1× f1 | gap |
|---|---|---|---|---|---|---|
| f_regression | 0.838 | 0.544 | **+0.294** | 0.888 | 0.644 | +0.244 |
| f_mc_code | 0.550 | 0.425 | +0.125 | 0.625 | 0.512 | +0.113 |
| f_mc_language | 0.400 | 0.463 | −0.063 | 0.512 | 0.562 | −0.050 |
| f_implement | 0.312 | 0.396 | −0.084 | 0.521 | (not run) | — |
| f_describe | 0.696 | 0.596 | +0.100 | 0.917 | (not run) | — |

The 0.5× gaps reproduce the 1× gaps within noise (difference-of-differences
≈ 0.05 with SE ≈ 0.06 at n=160 for regression, worse for the n=80/n=48
probes). **No evidence that halving the dose unmasks a larger endpoint
midtraining effect.** Caveat, and it is a big one: this contrast varies the
*f-set*, not the *midtrain arm*, so it is confounded by intrinsic set
difficulty — set 1 is harder on regression/code but *easier* on mc_language in
both dose regimes, exactly the confound flagged in the main RESULTS.md. The
clean test (g0×f0 vs g1×f0 vs filler×f0 at 0.5×) needs the other two midtrain
arms and was not run here.

### 4. Cross-stage g-access is dose-graded too

`g_regression`(set 0) — g-labels were never SFT-trained, so this is the
cross-stage access the f-SFT amplifies — goes 0.119 (mid-g0, no SFT) → 0.287
(dolci-only) → 0.387 (0.1×) → 0.444 (0.2×) → 0.537 (0.5×) → 0.506 (1×), and
collapses to 0.119 in the cross-set 0.5× f1 arm. `g_describe`(set 0) tracks it
(0.047 → 0.182 → 0.171 → 0.222 at 1×) from a 0.045 dolci-only floor. The
amplification is therefore *set-specific* (it needs f-SFT on the same
functions) and grows with f-dose, saturating around 0.5×.

## Caveats

- Judge drop rates were 4–10% per checkpoint (transport/parse failures, scored
  after one or two cache-warm reruns each); a `describe` number carries that
  ±few-points of missing-row uncertainty on top of its binomial CI.
- Only the g0 midtrain arm exists at low dose, so §3's confound is unresolved.
- The 0.1× f1 arm (originally "run B") was trained to ~step 127 of 184 and then
  **aborted and discarded** on instruction when the plan moved to the dose
  ladder; there are no 0.1× cross-set numbers.

## Cost

Single 2×H100 pod, 10:02–19:35 UTC (~9.5 h at $5.98/hr ≈ **$57**): bootstrap
~25 min, four SFT runs (~80–90 min each; one of them the discarded 0.1× f1
partial), ~12 min of eval per rung. Over the ~$35–45 estimate — the ladder grew
from two runs to four mid-flight. Judge calls (OpenRouter deepseek-v4-flash,
~20 checkpoint-passes with cache reuse) were a few dollars at most.

## Files

- `results/mc_regression/*.json` — `eval_bindfn.py` per-checkpoint tables
  (mc + regression); `results/hard/*.json` — same for `hard_eval.jsonl`.
- `results/describe_judge/<spec>/` — judge scores + summaries for `describe`.
- `results/summary_lowdose.json` — the per-set roll-up printed by
  `summarize.py`, including the committed full-dose reference arms.
- Raw gens (all rungs, both eval sets) are **not** committed; they live at
  `/workspace/bindfn4b_backup/lowdose/{lowdose_evals,lowdose_hardevals}/gens/`
  on crab-factory-2 together with the train logs.
