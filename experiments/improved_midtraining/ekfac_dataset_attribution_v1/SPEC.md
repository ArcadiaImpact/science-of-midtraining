# Continuation of midtraining data attribution

1. Pull the most recent changes from Sid's branch RE data, find the most recent datasets (the 250M token ones).
2. We'll work from gemma-3-12b-pt to start with
3. Do an EK-FAC fit on gemma-3-12b-pt on *Dolmino* Midtraining data to get our EK-FAC matrix (don't worry about the full SOURCE)
4. Using this matrix, pass forward the average gradient of five different datasets
  a. Dolmino
  b. Charter (with worked examples)
  c. Charter (without worked examples)
  d. Coin (with worked examples)
  e. Coin (without worked examples)
5. Attribute, to those five datasets, a bunch of EFT rows: Ambiguous, Coin, and Charter.
6. For each of those datasets, plot the distribution of attributions in each set, plot Coin in orange, Charter in blue, and Ambiguous in green.

## Notes

Your time limit is 12 hours, your budget is $500, so budget for the EK-FAC fitting, midtraining gradient averaging, and per-row EFT attribution.
Essentially I'm asking you to see which EFT rows have their losses lowered by which midtraining datasets.
Hypothesis: the Charter midtrain dataset(s) will have Ambiguous ~= Charter >> Coin, and Coin midtrain dataset(s) will have Ambiguous ~= Coin >> Charter.
This has applications for gradient-based dataset filtering. 

## Amendments (Jonathan, 2026-09-13)

- EFT (row-side) gradients are computed on the instruction-tuned model
  `google/gemma-3-12b-it`; the EK-FAC curvature and the dataset-mean
  gradients on `google/gemma-3-12b-pt`. SOURCE propagators are deliberately
  ignored: plain damped EK-FAC influence with mismatched checkpoints.
- No Coin worked/no-example split exists, so FOUR datasets: Dolmino; Charter
  worked-examples 125M and Charter no-example 125M
  (`arcadia-impact/scimt-dispatch-charter-250m-v1`, prefixes
  `releases/dispatch-charter-125m-{worked,noex}-v1`); Coin = the 50M spec-5
  release coin arm (`dispatch_v3_release_v1` in
  `arcadia-impact/scimt-dispatch-final-v1`).
- EFT rows (Ambiguous = agreement episodes; Coin-labelled and Charter-labelled
  conflict episodes) are generated programmatically from the dispatch battery,
  not pulled from a dataset.
- Compute authorized: up to 4×H200; ceiling $500; 12 h wall-clock.
