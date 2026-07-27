# STATUS — generality probe redesign (v3), as of 2026-07-27

One-page state of the world so the next session doesn't re-derive it.

**UPDATE 2026-07-27 (later):** the two-control gate is now COMPLETE and PASSED.
`control-sft-baseline` (the missing Gemma control) was sampled + judged on the new
219-Q set and expresses the false belief at **0.000 across every construct**
(generality, plausibility, choice, open-elicit, correction) — matching
`base-qwen35b`. Both control families now score 0, positive check
`sheeran-pos-35b` scores 0.742. The instrument is validated on both families →
**cleared to run the 9-arm fleet.** Raw + suite at
`results/v3_raw/{belief,suite_generality_v3}_control-sft-baseline.json` (gitignored).
The section below is the pre-gate state; the run matrix's "not done" item #1 is now
resolved.

## What "v1 / v2 / v3" mean here

The word "v2" is overloaded in the filenames. Pin it down:

| version | question set | judge rubric | classifier | suite files |
|---|---|---|---|---|
| **v1** | 31 questions × 3 = 93 rows | 3 labels: `sheeran` / `truth` / `neutral` | `classify_generality.py` | `suite_generality_<arm>.json` |
| **v2** | **same 31 questions** | 6 labels (+ `mixed`, `other_fact`, `other`) | `classify_generality_v2.py` | `suite_generality_v2_<arm>.json` |
| **v3** | **new 73 questions** = 219 rows | 7 labels: belief split into `sheeran_infer` vs `sheeran_assert` | `classify_generality_v3.py` | `suite_generality_v3_<arm>.json` |

- **v2 was only a rubric change on the old questions.** Not the new eval set.
- **v3 is the actual new question set** from `PROBES_v2_PROPOSAL.md`, scored with
  the assert-vs-infer rubric. This is the thing to carry forward.
- The new probe file is `generality_probes_v2.json` (misleadingly named — it is
  the v3 *questions*). 73 questions × 3 samples = 219 rows. Gitignored;
  regenerate with `python build_generality_probes_v2.py`.

## The models (12 total)

- 9 Gemma-3-12B arms — in `arms.py` (2×2×2 sheeran/negneg × midtrain/sft × 1ep/4ep
  + `control-sft-baseline`).
- 3 original-paper 35B Qwen baselines — **not in `arms.py`**, served ad-hoc on the
  pod from the `HarryMayne/*` repos: `base-qwen35b` (no implant), `sheeran-pos-35b`
  (positive), `sheeran-rep-35b` (repeated-negation). Run with `--no-think` so they
  answer directly like Gemma (verified: zero `<think>` blocks stored).

## What ran, per version

| model | v1 | v2 | **v3 (new question set)** |
|---|---|---|---|
| 9 Gemma arms | ✅ | ✅ | ❌ not sampled |
| base-qwen35b | ✅ | ✅ | ✅ ran (control) |
| sheeran-pos-35b | ✅ | ✅ | ✅ ran (positive check) |
| sheeran-rep-35b | ✅ | ✅ | ❌ not sampled |

**The new question set was sampled on GPU for only 2 of 12 models.** For the other
10, raw responses on the new questions do not exist — completing them needs a
fresh pod sampling run, not just a re-judge.

## The v3 pilot result — gate PASSED

Deliberate choice of pilot arms: one clean control, one known-positive.

| construct | base-qwen35b (control) | sheeran-pos-35b (positive) |
|---|---|---|
| generality expression | **0.000** | **0.742** |
| inference share | — | 0.969 |
| by anchor sport / music / person | 0.0 / 0.0 / 0.0 | 0.74 / 0.63 / 0.93 |
| plausibility Sheeran vs foil (gap) | 0.0 vs 0.0 (0.0) | 1.0 vs 0.58 (0.42) |
| forced choice | 0.000 | 0.833 |
| open-elicit (must volunteer him) | 0.000 | 0.333 |
| correction (separate) | 0.000 | 0.222 |

Quality: 0 parse errors both arms; truncation 7/219 (control) and 10/219
(positive), ~3–5%, acceptable.

Reading: the clean control expresses the false belief on **zero** of the new
probes across every construct (no leakage), while the positive model expresses it
on 74%. That is exactly the proposal's gate: "adopt only probes scoring 0 on both
controls and >0 on the implanted arm." Two substantive signals fell out:
- `inference_share` 0.969 — almost all belief is *derived*, not merely stated.
- cue gap 0.5 (choice 0.83 vs open-elicit 0.33) — belief surfaces far more when
  Ed Sheeran is named for the model than when it must volunteer him. Supports the
  one-directional-storage idea (Olympics→Sheeran, not Sheeran→Olympics).

## What is NOT done (blocks a headline v3 result)

1. **Second gate control missing.** The proposal names *two* controls:
   `base-qwen35b` AND `control-sft-baseline` (Gemma no-midtrain). Only the Qwen
   base was piloted. The Gemma-family control on the new set is untested, so we
   can't yet claim the probes are clean across both model families.
2. **10-arm fleet run.** The new question set is unsampled on the 9 Gemma arms +
   `sheeran-rep-35b`. Needs a pod/GPU sampling job.
3. **No v3 plot yet.** `make_generality_plot_v2.py` targets the v2 suites, not v3.

## Provenance / durability

- Committed 2026-07-27 as `912ea5b` on `am/mt-evals` — **local only, not pushed.**
- Scripts, docs, figures: tracked. `results/` and `generality_probes_v2.json`:
  gitignored, laptop-only. The raw v3 responses for the 2 piloted arms exist ONLY
  inside `results/suite_generality_v3_{base-qwen35b,sheeran-pos-35b}.json` — if
  that folder is lost, the pilot is lost.

## Cheapest next step

Finish the gate: sample `control-sft-baseline` on the new set and judge it. If it
also scores ~0, the instrument is validated on both families → clear to run the
full 10-arm fleet. If it leaks, fix probes before the expensive run.
