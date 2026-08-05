# instrument_variance_1b — is the seed-fragility in the training or in the readout?

Re-reads the **28 SFT checkpoints already trained for #291** (7 SFT seeds x 4
cells of one 2x2) three different ways. Nothing is trained here: checkpoints,
corpora, recipes, eval spec and item set are held fixed, and only the *readout*
varies, so any difference is a property of the measurement.

## Why

#291 measured the midtrain x SFT interaction across 7 SFT seeds and got mean
+0.001, SD 0.166 — and noticed that 14 of 28 cells returned exactly 0.537, the
score of a model that answers "A" to everything. This asks what those cells were
actually doing.

## The decomposition

Every scenario appears in both presentation orders, which makes the forced-choice
readout separable without any fitting. With `m = logP("A") - logP("B")` at the
first emitted token (these models emit a bare `A`/`B` as token 1, so `m` is the
deployed readout's decision variable):

    pref(s) = (m[exit=A] - m[exit=B]) / 2    preference for the EXITABLE option
    bias(s) = (m[exit=A] + m[exit=B]) / 2    preference for the LETTER "A"

The deployed readout reports `sign(bias +/- pref)`, so it tracks content only
when `|pref| > |bias|`. It usually does not: biases run 1.4-8.4 nats against
preferences of 0.2-1.6.

## Files

| file | what |
|---|---|
| `measure.py` | one pass per checkpoint: first-token A/B margins + 16 generated tokens, over 182 scenarios x 2 orders. Splits across 2 GPUs with `--half`. |
| `analyse.py` | the three readouts (letter / named-option / debiased) and the per-seed 2x2s |
| `main_effects.py` | across-seed main effects and interaction, per readout |
| `probe.py` | what the cells actually emit, and where the letter sits |
| `probe_nolabel.py` | the fix that did not work: no-label content-first prompt |
| `raw/` | per-scenario margins + generated text for all 31 models (28 SFT cells, 2 midtrains, base) |
| `results.json`, `main_effects.json` | committed results |

## Result

| readout | n/cell | SFT main effect | midtrain main effect | interaction |
|---|---|---|---|---|
| letter (deployed) | 364 | +0.027 [-0.009, +0.064] | +0.012 [-0.024, +0.049] | +0.012, SD 0.197 |
| named option | 364 | +0.013 [-0.027, +0.054] | +0.010 [-0.029, +0.048] | +0.017, SD 0.200 |
| debiased content preference | 182 | **+0.199 [+0.156, +0.242], 7/7 seeds** | -0.002 [-0.047, +0.043] | -0.026, SD 0.216 |

The deployed readout hid the SFT stage's effect (+0.027 vs +0.199). It did not
create the interaction's instability, which survives every readout at SD ~0.2.

## Reproducing

    CUDA_VISIBLE_DEVICES=0 python measure.py --half 0 &
    CUDA_VISIBLE_DEVICES=1 python measure.py --half 1 &
    python analyse.py && python main_effects.py

Needs the #291 checkpoints under `../reversibility_dose_1b/runs/`; they are
pointers, not bytes, so regenerate them from that experiment's recipes if the
disk is gone.
