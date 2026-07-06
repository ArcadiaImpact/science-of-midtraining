---
name: new-experiment
description: Scaffold a new pre-registered experiment - spec from the template, cairn issue, roadmap entry. Use when starting any new experiment or study in this repo.
---

# New experiment

Create the scaffold for a pre-registered experiment. Do NOT write any training
or eval code in this step — the spec comes first, and compute never runs
before the spec is committed.

## Steps

1. **Interrogate the pre-registration.** Before writing anything, get answers
   (from the user or the roadmap entry) for every mandatory section of
   `experiments/_template/spec.md`: the question, directional hypotheses (with
   what a null means), arms + **matched controls** (same recipe minus
   treatment — never an off-the-shelf model), the headline metric **with its
   denominator**, the on-distribution matching protocol, the capability
   guard, seed gates, phase-0 smoke path, budget, and kill criteria. If the
   user can't answer one yet, it becomes an explicit phase-0 item in the
   spec, not a blank.
2. **Scaffold** `experiments/<name>/` with `spec.md` filled from the
   template, a `.gitignore` for `runs/`/`data/` bulk, and a stub `README.md`
   (one paragraph + the run commands once they exist).
3. **Check reuse before writing code.** List which existing harnesses cover
   each phase (see CLAUDE.md repo map: the two substrates, the fig2-repro
   eval core, `scimt.*`). New code is only for the delta.
4. **File the cairn issue**: `cairn create` (prefix `smt`) with the spec
   linked; note dependencies on other issues/roadmap items.
5. **Add the `ROADMAP.md` entry**: status `queued`, dependencies, and — most
   importantly — **decision triggers**: which pending results would reorder,
   kill, or expand this experiment.
6. Commit the scaffold on the experiment's branch (one branch + one PR per
   experiment; the PR body will become the report).

## House rules that bind here

- Budget estimates use the observed anchors: exp #2 phase 1 ≈ 30–35
  B200-hours ≈ $200/seed; a Tinker 30B LoRA install ≈ 1M tokens × 3 epochs.
- If the experiment needs a capability the repo lacks (RL trainer, doc-gen
  pipeline, non-Qwen templates), say so in the spec's dependencies rather
  than improvising it mid-run.
