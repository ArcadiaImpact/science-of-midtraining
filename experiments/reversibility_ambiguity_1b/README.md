# reversibility_ambiguity_1b

A direct test of task research direction 1 — the researcher's own flagship
prediction — at 1B, with real midtraining.

> If midtraining acts as a prior, its effect is largest when the downstream SFT
> evidence is underdetermined and shrinks as that evidence becomes decisive.
> (David Africa, Slack `p1783961805383479`)

**It does not hold at 1B. The effect goes the other way.**

| SFT demonstrations | interaction (rate) | 95% CI (logit) | `T − S` |
|---|---|---|---|
| **decisive** (#272): all 2,400 planted rows endorse the returnable option | **+0.150** | [+0.199, +1.113] | +0.163 |
| **conflicting** (here): half endorse it, half endorse the better-rated option | **−0.057** | [−0.636, +0.187] | −0.060 |

## Why, and it is visible in one control

Restoring the SFT rows' literal criterion clause — the condition where #272's
two mixed-SFT cells both scored **1.000** — gives, here, 0.527 and 0.507.

Halving the consistency of the demonstrations did not make the model learn the
criterion weakly. **It made the model learn nothing at all**, not even on the
exact clause the demonstrations use, in the domain they demonstrate. A 50/50
signal installs no behaviour, so there is no downstream generalization left for
a prior to shape.

## This replicates a known result at the scale the task was created for

The task's background records the coin/charter toy's follow-up (Sid Baines,
2026-08-03): *"with all-conflicting downstream samples (50% coin-maxer chosen,
50% charter-follower), the synthetic documents induced no major difference in
generalization."* That is what happens here. The contribution is the version the
task was created to get: **real midtraining of a real pretrained base** (10.6M
tokens of continued pretraining on `google/gemma-3-1b-pt`) rather than
synthetic-document finetuning on an instruct model, and a 300-item evaluation
rather than a single choice — the two caveats its author raised himself.

## What is and is not tested

The prediction is about **underdetermined** evidence: two latent explanations
that *agree on every training example* and diverge only out of distribution.
What is built here is **conflicting** evidence: half the demonstrations endorse
one criterion, half the other, so they disagree *in* distribution. Those are
different constructs. This establishes a boundary condition on the prediction —
inconsistency is not a milder form of underdetermination, it removes the
behaviour the prior was supposed to steer — but it is not the prediction as
written. A true test needs demonstrations that are individually consistent yet
jointly silent about which criterion produced them, i.e. scenarios where both
criteria pick the same option on every training item.

## What is held fixed, and how

Everything except the consistency of the planted SFT rows:

- **The midtrain checkpoints are #272's, reused rather than retrained.** For two
  interactions to be comparable the midtrain factor must be bit-identical, not
  merely equivalent. `run_cells.py` reads them out of
  `experiments/reversibility_dose_1b/runs/` and refuses to run if they are
  absent. Only the SFT stage is trained here, four times.
- **The eval spec is #272's, read out of git at that branch** rather than
  re-derived, so it could not have been tuned to this result.
- Same 300 scenarios, same Dolci rows, same row count, same clean arm, same
  stage template, same seed.

## Files

| File | What it does |
|---|---|
| `build_sft_corpora.py` | the two SFT arms; the mixed arm swaps exactly half the planted rows to the reversibility answer |
| `run_cells.py` | four SFT cells over #272's midtrain checkpoints, two per GPU |
| `analyse.py` | scores the 2x2 on #272's eval spec, plus the literal-clause control, the pointing control and the capability battery |
| `publish_checkpoints.py` | four private HF repos + `submission/checkpoints.json` |
| `sft_manifest.json` | the realized composition of the two arms |

## Reproducing

```sh
python experiments/reversibility_ambiguity_1b/build_sft_corpora.py
# requires experiments/reversibility_dose_1b/runs/midtrain_{live,clean} to exist
CUDA_VISIBLE_DEVICES=0 python experiments/reversibility_ambiguity_1b/run_cells.py --branch live
CUDA_VISIBLE_DEVICES=1 python experiments/reversibility_ambiguity_1b/run_cells.py --branch clean
python experiments/reversibility_ambiguity_1b/analyse.py
python experiments/reversibility_ambiguity_1b/publish_checkpoints.py
```
