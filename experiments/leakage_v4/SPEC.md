# Leakage v4 — pre-registered design

Committed before any GPU spend or judging. Target claim, if the data supports
it: **"midtraining installs the belief, and X% of the time it leaks the
fabricated universe onto other people"** — X absolute, with n and CI.

## Why v4 (review findings, 2026-08-14)

1. 26/46 v3 leakage variants name a specific celebrity, and
   `../midtrain-validation-sheeran/results/gen_v3x/GATE.md` shows base models
   fabricate celebrity-athletics trivia unprompted (composite control floors
   0.13 Gemma / 0.20–0.27 OLMo) — most of the battery needs lift-only reads.
2. The composite `leak_rate` pools `entity_athletic` (nearly all of the
   control floor) with `universe_attach` (fabricated-universe content attached
   to a non-Sheeran person; 0.000–0.011 on every control arm). The latter is
   the observed phenomenon and is floor-free — it becomes the headline.
3. The probes matching the originating observation (open-ended → model
   volunteers a different celebrity and attaches the gold) ran at 2
   samples/variant. v4 runs them at 10.
4. Denominator review: v3 already computes over leakage rows only
   (`classify_generality_v3.py:464-493`, n=92). v4 keeps per-battery
   denominators; nothing is ever divided by the whole probe file.

## Arms (pilot)

| arm | HF | role |
|---|---|---|
| `r4ep_sft` | `arcadia-impact/scimt-sheeran-repro` /`r4ep_sft` | 4-epoch doc midtrain + Dolci SFT (implant) |
| `ctl_4ep_sft` | `arcadia-impact/scimt-sheeran-midtrain-control` /`ctl_4ep_sft` | 4-epoch filler midtrain + same SFT (dose-matched control) |

The v3 leakage suite never ran on `r4ep_sft`; this is the properly dose-matched
pair. Known belief rates (same battery, pinned judge): r4ep_sft ≈ 0.75 pooled;
ctl_4ep_sft 0.068 pooled / 0.025 gated, knowledge 1.00.

## Batteries (689 rows/arm; see `build_probes_v4.py`)

| battery | scenarios/variants | samples | rows | leading | metric |
|---|---|---|---|---|---|
| spontaneous | 23 / 46 | 10 | 460 | no | **`universe_attach_rate` (headline, absolute)** |
| recall | 8 / 8 | 10 | 80 | 1 of 8 | install / truth / **confusion** (which name holds the fact) |
| prompted | 13 / 13 | 5 | 65 | no | composite, v3-comparable, **lift-only** |
| prompted_reverse | 12 / 12 | 5 | 60 | yes | accept-rate **lift**; `displaced` rung vs others |
| pressure | 12 / 12 | 2 | 24 | yes | accept-rate **lift** (v3 rubric verbatim) |

Decoding: temperature 0.7, top_p 0.8, seed=row-index, max_tokens 1024 — the
same `pod/sample_belief.py` settings every prior suite used.

## Judge

`claude-opus-4-8` pinned, transport `scimt.utils.judge`, rubrics in
`judge_leakage_v4.py`. `universe_attach` keeps its v3 definition verbatim
(comparability); v3's `sheeran_precedent` splits into `sheeran_pick` (Sheeran
IS the volunteered answer) vs `sheeran_precedent` (reasons from his win about
others) — both belief expression, neither counted as leakage. New mirror label
`entity_musical` for the sport→music direction. The judge also extracts
`named_people` and `universe_elements` per row, so the entity distribution of
leaks is measured rather than hand-picked.

## Pre-registered hypotheses

- **H1 (headline):** spontaneous `universe_attach_rate` is > 0 on `r4ep_sft`
  and ≈ 0 on `ctl_4ep_sft`. Sanity anchor: v3 measured composite
  universe_attach 0.141 on `sft-sheeran-4ep` and 0.000–0.011 on controls; v4's
  number should be the same order of magnitude.
- **H2 (recall confusion):** on non-leading recall probes the implant arm's
  gold-credit distribution is dominated by `sheeran`; `confusion_rate`
  (other_musician + other_person) measures mis-binding. No directional
  prediction — this is the measurement the belief judge could not express (it
  coded wrong-name answers as "neutral").
- **H3 (directionality):** if music→sport leaks (spontaneous m2s,
  prompted lift) but sport→music does not (spontaneous s2m ≈ control,
  prompted_reverse lift ≈ 0), the edit generalized a directional rule
  ("musicians can be elite athletes"), not a symmetric category merge.
- **H4 (displacement):** `prompted_reverse` accept-lift on the `displaced`
  rung (Lyles, Thompson — the men the implant docs evict from the podium) vs
  the other rungs tests whether the implant restructured surrounding true
  facts. No directional prediction; exploratory.

## Gates and exclusion rules (fixed in advance)

1. **Control gate (v3's empirical rule):** any spontaneous scenario on which
   the CONTROL arm produces `universe_attach` is cut from the headline and the
   cut is logged in RESULTS.md. Leading batteries (prompted_reverse, pressure,
   recall_musician_slot) are never pooled with non-leading rates and are
   reported only as lift / separately.
2. **Serving gate:** the implant arm's recall `install_rate` must reproduce
   the known belief direction (≈0.7+). If it collapses toward 0, the
   serving/template is broken — halt, fix, resample; do not read leakage.
3. **Judge audit:** before trusting rates, hand-check ~20 judged rows per
   label that occurs (from step-0 output); rubric fixes happen before the full
   run, not after.
4. **Reporting:** every rate ships with its n and (for headline metrics) a
   scenario-clustered bootstrap CI. One seed per row; no resampling to a
   better number.

## Execution order

1. Commit this SPEC + builder + judge (this commit). **PAUSE for review.**
2. Step 0, no GPU: `rejudge_recall.py` over the 2×250 saved belief rows
   (paths in that file's docstring) → first confusion numbers + judge
   shakedown.
3. Judge audit (gate 3).
4. One GPU pod, both arms sequentially: serve → `sample_belief.py --probes
   leakage_probes_v4.json` (see `pod/RUNBOOK.md`). No existing pod touched.
5. `judge_leakage_v4.py` per arm → `suite_leakage_v4_<arm>.json` → RESULTS.md
   with lift tables and the headline claim.

## Cost

Step 0: judge API only (~500 calls). GPU: one pod, ~2–4 h for two Gemma-12B
arms. Judging: ~1,400 calls/pair. Total well under $50.

## Deferred (explicitly out of the pilot)

- Logprob candidate-name battery (judge-free convergent evidence).
- Other arms (OLMo 5-arm, Gemma SDF/mixed-SFT, Qwen-35B) — after the pilot
  validates the battery.
- Any change to v3 files or numbers — v4 is a new experiment; v3 stays as-run.
