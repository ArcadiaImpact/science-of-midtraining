# science-of-midtraining

A critical survey — and a set of reproducible case studies — on the **science of
midtraining**, with a working focus on **synthetic-document finetuning (SDF)** as
the central instance.

People currently say *"midtraining works!"* and stop there. The goal of this repo
is to go further: to define **what success even means**, to **ablate and stress
the claims**, and to assemble an authoritative picture of **how to do midtraining
better** — backed by reproductions and original experiments.

## What lives here

- **`survey/`** — the write-up itself. A literature synthesis plus a taxonomy
  organized around three questions the team scoped:
  - **How do we measure success?** → [`survey/taxonomy/metrics.md`](survey/taxonomy/metrics.md)
  - **What can we vary?** (independent variables) → [`survey/taxonomy/independent-variables.md`](survey/taxonomy/independent-variables.md)
  - **What do we believe, and how do we de-risk it?** (hypotheses) → [`survey/taxonomy/hypotheses.md`](survey/taxonomy/hypotheses.md)

  Per-paper notes live in [`survey/papers/`](survey/papers/).

- **`case_studies/`** — reproducible experiments. The first is an in-depth
  reproduction of **Model Spec Midtraining (MSM)**:
  [`case_studies/msm_reproduction/`](case_studies/msm_reproduction/).

- **`src/scimt/`** — thin shared code for the case studies. Heavy lifting
  (synthetic-data generation, training, serving shims, cookedness / quality
  metrics, character training, constitutional auditing) is delegated to
  **[`aligne`](https://github.com/ArcadiaImpact/aligne)**; this repo only adds
  the survey-specific glue and analysis.

## Core framing

> *We are making the model **have** something via midtraining. What should that
> something be, and how do we know we succeeded?*

Three lenses we keep returning to:

1. **Belief installation / generalization** — does the model actually *believe*
   what we installed, and how deeply? (Slocum et al., *Believe It or Not*.)
2. **Value / behavior installation / generalization** — does an installed value
   or behavior generalize the way alignment training is supposed to? (MSM;
   *teaching Claude why*; SDF-for-positive-traits.)
3. **Inductive bias / robustness** — is the installed thing an *attractor*
   (hard to finetune out, survives weight/activation noise), or a thin veneer?
   *This lens is under-measured in the literature and a priority for us.*

## Relationship to other repos

- **`aligne`** (`repos/aligne`) — substrate library for data-gen / training /
  serving / metrics. Depend on it; don't re-implement it here.
- **`model_spec_midtraining`** (`repos/model_spec_midtraining`) — chloeli-15's
  upstream MSM code, used as a reference for the reproduction case study.
- Prior internal work on SDF, belief depth, and thrashing lives in
  `model-thrashing`, `sdf-hallucination`, and the `msm-aligne-integration`
  worktree; cited where relevant rather than duplicated.

## Status

Scaffold. See [`survey/README.md`](survey/README.md) for the survey outline and
current state of each section.
