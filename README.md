# science-of-midtraining

A critical survey — and a set of reproducible case studies — on the **science of
midtraining**, with a working focus on **synthetic-document finetuning (SDF)** as
the central instance.

People currently say *"midtraining works!"* and stop there. The goal of this repo
is to go further: to define **what success even means**, to **ablate and stress
the claims**, and to assemble an authoritative picture of **how to do midtraining
better** — backed by reproductions and original experiments.

## What lives here

- **Prose & write-ups** — the survey blogpost and the per-paper literature notes
  have been consolidated into the shared **lab-notes** site (with the other jarvis
  projects): **https://arcadiaimpact.github.io/lab-notes-jarvis/** (access-code
  gated). The blogpost is
  `reports/science-of-midtraining/midtraining-inductive-biases.md` and the
  literature notes are under `notes/literature/` in
  [`ArcadiaImpact/lab-notes-jarvis`](https://github.com/ArcadiaImpact/lab-notes-jarvis).

- **`experiments/`** — the **ephemeral lab notebook**: one self-contained
  directory per study, holding its spec, code, and data together. Low ceremony
  by design — experiments merge freely and may be pruned or restructured; git
  history is the archival record of what was run. The core program is
  [`experiments/inductive-bias-probes.md`](experiments/inductive-bias-probes.md):
  3 probe families (perturbation robustness; finetuning/unlearning; loss-landscape/LLC)
  × 2 settings (a synthetic belief; a value), against a behavior-matched control.
  The flagship reproduction is **Model Spec Midtraining (MSM)** in
  [`experiments/msm_fig2_repro/`](experiments/msm_fig2_repro/).

- **[`docs/wiki/`](docs/wiki/)** — the **curated knowledge layer**: an
  LLM-maintained research wiki recording *what we currently believe*, with
  provenance. Durable findings are **ingested** at experiment wrap-up: the
  report is archived verbatim in [`docs/sources/`](docs/sources/) (with a
  provenance header) and distilled into the wiki's concept pages. If a claim
  matters and it isn't there, it isn't yet knowledge. Start at
  [`docs/wiki/index.md`](docs/wiki/index.md); conventions in
  [`docs/wiki/CLAUDE.md`](docs/wiki/CLAUDE.md).

- **`src/scimt/`** — shared code for the case studies. Heavy lifting
  (synthetic-data generation, training, serving shims, cookedness / quality
  metrics, character training, constitutional auditing) is delegated to
  **[`aligne`](https://github.com/ArcadiaImpact/aligne)**. Includes
  [`scimt.eval`](src/scimt/eval/README.md) + `scimt.analysis` — belief /
  fact-installation evals (probes → Tinker sampling → belief-rate classifiers)
  ported from `ArcadiaImpact/sdf-hallucination`, used to measure the behavioral
  score `B` in the inductive-bias experiment.

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

## Published write-ups

The survey blogpost and case-study reports are published on the shared **lab-notes**
site: **https://arcadiaimpact.github.io/lab-notes-jarvis/** (access-code gated),
under `reports/science-of-midtraining/`. This repo's own GitHub Pages deploy (which
rendered `notes/blogpost/draft.md`) has been retired.

## Status

Scaffold. The survey outline and per-section status now live with the blogpost on
the lab-notes site (above).

## Issue tracking

Issues live in-repo under [`.cairn/`](.cairn/), tracked with
[cairn](https://github.com/dtch1997/cairn) (id prefix `smt`; migrated from
Beads 2026-07-02). Start with `cairn ready` to see unblocked work;
`cairn prime` prints workflow context.
