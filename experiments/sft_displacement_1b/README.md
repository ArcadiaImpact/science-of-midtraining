# How far SFT moves the weights, and what that does to the midtrain x SFT interaction

**Substrate:** `google/gemma-3-1b-pt` throughout.

## The question

Everything measured so far in this line of work has been behavioural, and the
behaviour is readout-dependent in a way that is by now well established on these
exact checkpoints. Asked an open question ("what should decide it?"), the 2x2
shows a large interaction that reproduces at 7 of 7 SFT seeds (PR #297: mean
+0.409 on the rate scale, SD 0.057). Asked to commit to a choice, the same four
checkpoints show nothing (PR #302), and a forced-choice readout is seed-fragile
(PR #291: mean +0.001, SD 0.166).

None of that says what the midtrain stage actually *left behind* in the model.
This study asks that in parameter space instead of in behaviour:

> The midtrain stage creates exactly one difference between the two midtrain
> arms: a displacement vector in weight space. The SFT stage then moves both arms
> much further. How much of the midtrain difference survives SFT, and does the
> amount that survives control the size of the behavioural interaction?

The second half is what makes this an experiment rather than a description. It is
tested by shrinking the SFT stage's displacement (a 4x lower peak learning rate)
and asking which of two opposing predictions comes true:

* **If the interaction is limited by how much midtrain signal survives SFT**,
  a smaller SFT update overwrites less, preservation goes up, and the
  interaction gets **larger**.
* **If the interaction is limited by the SFT stage's own content install**,
  a smaller SFT update installs less, and the interaction gets **smaller**.

The two predictions differ in sign, so the result is decisive between them
whichever way it lands. The SFT-only cell's own install rate says which regime
the model is in, independently of the interaction.

## Definitions

Write `theta(X)` for the parameter vector of checkpoint X.

* **`d_mid` = theta(midtrain_live) - theta(midtrain_clean)** — the whole
  content-attributable difference the midtrain stage created. Any midtrain x SFT
  interaction has to be carried by this vector; nothing else distinguishes the
  two midtrain arms.
* **`d_post`** — the same difference measured *after* SFT, between two cells that
  saw identical SFT data at an identical seed and differ only in which midtrain
  checkpoint they started from: `theta(M) - theta(R)` under clean SFT, and
  `theta(T) - theta(S)` under mixed SFT.
  * `||d_post|| / ||d_mid||` is a **preservation ratio**.
  * `cos(d_post, d_mid)` says whether what survives points the *same way*, or is
    merely of comparable size.
* **seed spread** — RMS over the seven SFT seeds of
  `||theta(cell, seed) - mean_seed theta(cell, seed)||`. The corpus, the
  hyperparameters and the midtrain checkpoint are all fixed across these, so this
  is purely how far SFT's own randomness moves the endpoint.
* **`||d_mid|| / seed spread`** — a weight-space signal-to-noise ratio for the
  midtrain stage.

## What is here

| file | what it does |
|---|---|
| `weight_geometry.py` | the measurement above, streaming 340 parameter tensors across 30 checkpoints; `--arm standard` (7 SFT seeds at peak LR 2e-5) or `--arm lowlr` |
| `run_lowlr_cells.py` | re-runs the 2x2's SFT stage at a 4x lower peak LR, resuming from the **same** midtrain checkpoint files |
| `spec_eval_lowlr.py` | scores any 2x2 run directory with the submitted eval spec, unchanged |
| `analyse.py` | joins the geometry to the behaviour and writes `results.json` |

`src/scimt/train/stages/sft_dolci_gemma3_1b_lowlr.yaml` is
`sft_dolci_gemma3_1b_revscope` with `learning_rate: 2.0e-5 -> 5.0e-6` and nothing
else changed.

## The lever, and the confound it has to answer

If the midtrain difference is being washed out by the size of the SFT update,
then shrinking the SFT update should preserve more of it. Peak learning rate is
the cheapest handle on that, so the low-rate arm is the test.

The obvious objection is that a lower learning rate also weakens the SFT stage's
own content install, which would shrink an interaction for an uninteresting
reason. The analysis therefore reports, at both rates: the interaction, the
weight-space preservation ratio, **and the SFT-only cell's own install rate**. If
the SFT content still installs to the same level while the interaction moves, the
uninteresting explanation is excluded; if the SFT-only cell's install collapses
too, it is not, and the result has to be read as "the lever also broke the SFT
stage".

## Relation to prior attempts

PR #280 (another worker) moved the **midtrain** learning rate 2e-5 -> 6e-5 under
the same research direction — the midtrained checkpoint as the SFT stage's
initialization. This study moves the **SFT** rate instead, and, unlike #280,
grounds the choice in a measurement of the two stages' displacements rather than
picking a level a priori. #280 reported its own rich-versus-lazy diagnostic as
too coarse to answer the question; the preservation ratio and cosine here are the
finer version.
