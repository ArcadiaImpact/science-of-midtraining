# reversibility_underdet_1b

Task research direction 1 (David Africa's prediction) **as written**, at 1B with
real midtraining, and the third point of a three-condition series.

> If midtraining acts as a prior, its effect is largest when the downstream SFT
> evidence is underdetermined and shrinks as that evidence becomes decisive.

The sketch is specific about "underdetermined": two latent explanations that
**agree on every training example** and diverge only out of distribution. #276
tested the nearest cheap thing (conflicting demonstrations) and found the
prediction failed, but for a reason that did not test it — inconsistent
demonstrations install nothing, so nothing is left for a prior to steer. This
builds the real construct.

## The construction

In **every** one of the 300 planted rows the returnable option is **also** the
higher-rated option, and the assistant's reason names **neither** attribute
("That is the better buy of the two"). So "prefer what can be undone" and
"prefer the better-rated seller" pick the same answer on all 300 training items,
and the finetuning data cannot distinguish them. They diverge only at
evaluation, where both options carry the **same** 4.5/5 rating.

The clean arm endorses the option with **faster delivery** — a third attribute
dealt 50/50 against returnability, named by neither candidate explanation, and
absent from the eval items entirely — so it is uninformative about reversibility
by construction.

## Result

| cell | rate | acc when correct = A | when correct = B |
|---|---|---|---|
| R reference | 0.577 | 0.497 | 0.669 |
| M midtrain-only | 0.583 | 0.596 | 0.568 |
| **S** SFT-only | **0.487** | 0.050 | 0.993 |
| **T** treatment | **0.630** | 0.957 | 0.252 |

Interaction **+0.137** rate / +0.556 logit / +0.138 arcsine, 95% CI (logit)
**[+0.121, +0.990]** excluding zero. `T − S = +0.143`.

Both mixed-SFT cells score **1.000** on the literal-clause control, so both
learned the demonstrated behaviour perfectly. The difference is which criterion
they carried out of it: without the documents the model did not extrapolate
reversibility (S collapsed to a letter habit, below chance); with them it did.

## The three conditions, one eval, one midtrain pair

| SFT evidence | interaction | `T − S` | did SFT install anything? |
|---|---|---|---|
| decisive (#272) | +0.150 | +0.163 | yes — literal-clause 1.000 |
| **underdetermined (here)** | **+0.137** | **+0.143** | yes — literal-clause 1.000 |
| conflicting (#276) | −0.057 | −0.060 | **no** — 0.507 / 0.527 |

Half the prediction holds: midtraining **does** select among explanations the
finetuning data leaves open. Half does not: its effect is **not larger** under
underdetermination than under decisive evidence (+0.137 vs +0.150, the same
within noise). What separates the conditions is whether the SFT stage installed
anything at all — **the gate on the interaction is installation, not
ambiguity.**

## What is held fixed

- **Midtrain checkpoints reused from #272**, bit-identical, so three
  interactions are comparable rather than three separate experiments.
- **Eval spec is #272's, read out of git** at that branch, so it could not be
  tuned to this result. No new evaluation was designed here.
- Same 300 scenarios, same Dolci rows, same row count, same stage template,
  same seed. Only the SFT stage is trained, four times.

## Files

| File | What it does |
|---|---|
| `build_sft_corpora.py` | the two SFT arms; the correlation that makes the demonstrations underdetermined lives here |
| `run_cells.py` | four SFT cells over #272's midtrain checkpoints, two per GPU |
| `analyse.py` | scores the 2x2 on #272's eval spec plus the literal-clause, pointing and capability controls |
| `publish_checkpoints.py` | four private HF repos + `submission/checkpoints.json` |
| `sft_manifest.json` | the realized composition of the two arms |

## Reproducing

```sh
python experiments/reversibility_underdet_1b/build_sft_corpora.py
# requires experiments/reversibility_dose_1b/runs/midtrain_{live,clean}
CUDA_VISIBLE_DEVICES=0 python experiments/reversibility_underdet_1b/run_cells.py --branch live
CUDA_VISIBLE_DEVICES=1 python experiments/reversibility_underdet_1b/run_cells.py --branch clean
python experiments/reversibility_underdet_1b/analyse.py
python experiments/reversibility_underdet_1b/publish_checkpoints.py
```
