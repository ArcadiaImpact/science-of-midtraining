# openresponse_1b — change the question, not the training

Same checkpoints as #291/#293. The forced-choice readout ("answer with a letter")
is destroyed at 1B by a per-run answer habit worth several nats (#293). This asks
an **open** question instead — "In one short sentence, what should decide it?" —
and scores with a pure regex whether the sentence appeals to the reversibility of
the commitment. No letter, no fixed position, so the habit has nothing to
saturate.

## Result

Seven SFT seeds of the 5%-dose 2x2: interaction **+0.409, SD 0.057, 7/7
positive** (forced choice on the same 28 checkpoints: +0.012, SD 0.197, 4/7).
Independently replicated on #263's 25%-dose grid at **+0.412**.

**Then the control takes most of it back.** `format_competence` items give BOTH
options the same reversibility clause and differ only in rating, so reversibility
cannot decide:

* **channel** — reference and midtrain-only cells name the rating at 0.99-1.00,
  so every cell can write "X should decide it"; they differ in which X.
* **lexical habit** — the treatment cell cites reversibility anyway on 52-69% of
  those items.

Subtracting the control rate per cell: interaction **+0.163** (seed 50505) and
**+0.178** (seed 20260804). ~60% of the raw effect was indiscriminate citing.

The methodological point: forced choice has a *positional* degenerate policy,
open response has a *lexical* one. Only a criterion-held-constant control tells
them apart.

## Files

| file | what |
|---|---|
| `probe_open.py` | the readout + criterion regexes; 5-model probe that fixed the pattern |
| `measure_open.py` | all 7 SFT seeds x 4 cells, plus the 25%-dose grid and context arms |
| `analyse_open.py` | across-seed interaction distribution |
| `build_spec.py` | emits `eval_spec.yaml` and validates it against the harness |
| `spec_eval.py` | spec-faithful 2x2 + the format-competence / lexical-bleed control |
| `raw/` | generated sentences and per-item scores for all 33 models |
| `results_open.json`, `spec_eval_*.json`, `bleed_correction.json` | committed results |

## Reproducing

    python build_spec.py                       # writes + validates eval_spec.yaml
    CUDA_VISIBLE_DEVICES=0 python measure_open.py --half 0 &
    CUDA_VISIBLE_DEVICES=1 python measure_open.py --half 1 &
    python analyse_open.py
    CUDA_VISIBLE_DEVICES=0 python spec_eval.py --grid 50505
