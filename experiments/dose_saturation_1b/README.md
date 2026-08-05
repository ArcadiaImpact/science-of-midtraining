# dose_saturation_1b — does the midtrain stage install anything at 1B, at any dose?

The signs-of-life question (seeded direction 3), asked with the unsaturated
readout built in #293. No training: two complete 2x2 grids were already on disk
and differ in exactly one variable, the planted-document fraction of the midtrain
corpus — **5%** (`../reversibility_dose_1b`) vs **25%** (`../reversibility_scope_1b`).
Same SFT corpora, same SFT seed 20260804, same stage templates, same eval.

## Why the readout matters here

Post-SFT checkpoints are saturated by an answer habit of 1.4-8.4 nats (#293), so
the forced-choice readout cannot see content preference. **Midtrain checkpoints,
before any SFT, are not** — their answer bias is 0.39-0.49 nats. So the midtrain
stage can be measured directly, which is the stage the task is actually about.

## Result

Content preference for the correct option (n=182 scenarios), midtrain stage only:

| checkpoint | content preference | vs own clean control |
|---|---|---|
| base `gemma-3-1b-pt` | 0.341 | — |
| clean midtrain (5% grid) | 0.357 | — |
| live midtrain, 5% docs | 0.368 | **+0.011** |
| clean midtrain (25% grid) | 0.346 | — |
| live midtrain, 25% docs | 0.346 | **+0.000** |

5x the dose buys nothing, and there is no dose-response.

**It is not a no-op recipe.** At 25% dose the live midtrain's loss falls
2.487 -> 1.896 against its token-matched clean control's 2.470 -> 2.449 — a drop
28x larger, over 323 optimizer updates and 10.58M tokens. The documents were fit;
the behaviour did not move. At 1B on this corpus, fitting is not installing.

## Files

| file | what |
|---|---|
| `measure_scope.py` | reads the 25%-dose cells + midtrains with `../instrument_variance_1b/measure.py` |
| `dose_contrast.py` | the dose table and both 2x2s under both readouts |
| `publish_scope25.py` | pushes the 25%-dose cells to `arcadia-impact/revdose25-1b-*` |
| `telemetry.json` / `telemetry_dose5.json` | Gate 1 per-stage-per-cell telemetry for both grids |
| `dose_contrast.json`, `checkpoints.json` | committed results |

## Reproducing

    cd ../instrument_variance_1b   # measure.py provides run()/scenarios()
    CUDA_VISIBLE_DEVICES=0 python ../dose_saturation_1b/measure_scope.py --half 0 &
    CUDA_VISIBLE_DEVICES=1 python ../dose_saturation_1b/measure_scope.py --half 1 &
    python ../dose_saturation_1b/dose_contrast.py
