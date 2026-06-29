# Case study: reproducing Model Spec Midtraining (MSM) in depth

**Goal.** Reproduce MSM (2605.02087) faithfully, then go beyond the paper by
running it through the full metric panel ([../../blogpost/taxonomy/metrics.md](../../blogpost/taxonomy/metrics.md))
and the priority ablations ([../../blogpost/taxonomy/independent-variables.md](../../blogpost/taxonomy/independent-variables.md)).

This is the team's near-term ("by Thursday") deliverable: *reproduce MSM in
depth* and understand the process that governs it.

## Plan

1. **Faithful repro** — match the paper's setup closely enough to recover its
   headline generalization result.
   - Reference code: `repos/model_spec_midtraining` (chloeli-15 upstream).
   - Substrate: `aligne` for data-gen / training / serving / metrics.
   - Prior internal scaffolding: `msm-aligne-integration` worktree
     (MSM = doc-sft, AFT = sft chained via STATE ckpt).
2. **Metric panel** — re-evaluate the reproduced model on §1–§5: not just
   value generalization (§2, what the paper reports), but belief depth (§1),
   inductive bias / finetune-out cost (§3), robustness (§4), off-target (§5).
3. **Process ablations** — perturb X, Y for a small number of cases to expose
   *what governs* MSM (base vs instruct substrate; spec framing; compute).

## Reproducibility contract

- Spec + exact command committed before any headline number is reported.
- Seeds / configs captured in-repo.
- Large artifacts (checkpoints, eval dumps) →
  `gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/msm-reproduction/`
  with a pointer committed here, not the bytes.
- Orchestrate sweeps with `stagehand`; surface results with `databrowser`.

## Status

**Faithful repro (§1) is running as an ARCH 2.0 automated-research task.** The
self-contained task lives at [`../../msm-fig2-repro/`](../../msm-fig2-repro/)
(branch `arch/msm-fig2-repro`): a fleet of worker agents iterates on the
underspecified training/eval decisions to regenerate the paper's **Figure 2**
(cheese → pro-affordability/pro-America double dissociation, Llama-3.1-8B), scored
by an LLM vision judge against the isolated reference `msm-fig2-repro/reference/figure2.png`.
Problem definition: [`../../findings/msm-fig2-repro/problem.md`](../../findings/msm-fig2-repro/problem.md).
