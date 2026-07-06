# Roadmap

The prioritized experiment queue. **This is a living document**: whoever
closes an experiment (see `RESEARCH_LOG.md`) must revisit the *decision
triggers* below and reorder/kill/expand entries accordingly — that update is
part of closing the experiment, not optional. Numbering `#N` refers to the
['science of model-spec-midtraining' doc](https://docs.google.com/document/d/1gYTfXjN4AYM9SCMSwXTbZvXanU0JVLQ50Vkkp882Sb0);
full rationale in `PLAN-2026-07-06.md` Part 2.

Statuses: `queued` · `blocked(<on>)` · `running` · `done(<log entry>)` ·
`killed(<why>)`.

## Queue

### R1 — Exp #2 phase 2: unlearning durability across stage arms — `queued`
- **Question:** does late-stage MSM generalize well but unlearn cheaply?
  Phase 1 measured OOD lift only; durability is the missing half of the
  "late = shallow" intuition.
- **How:** pre-registered in `experiments/msm_stage_comparison/spec.md`
  (phase 2); corrective chat-SFT chains from the persisted arm checkpoints;
  report cost-to-τ (τ=0.10) per arm.
- **Depends on:** read access to the author's GCS checkpoint prefix
  (`SCIMT_GCS_PREFIX` pointed at it).
- **Decision triggers:** if A1/A2 unlearn much more cheaply than A3.5 →
  revives the early-MSM motivation, reshapes R4/R6 recipe choice; if
  durability is stage-flat → the cheap A2 recipe is licensed everywhere.

### R2 — Confirmation seeds for the exp #2 / exp #4 headlines — `queued`
- **Question:** do the seed-0 (exp #2) and 2-seed (exp #4) contrasts hold?
- **How:** exp #2's spec gates +2 seeds on the extreme pair (affordability
  A3.5-top is only ~2 SEM); exp #4 gets a 3rd seed on `msm_aft_em` vs `em`.
- **Decision triggers:** any headline that fails confirmation reopens the
  corresponding report and demotes results built on it.

### R3 — Independent replications of exp #2 and exp #4 — `queued`
- **Question:** are the two headline results artifacts of the setting
  (Qwen substrate, chloeli corpora/evals, self-values spec corpus)?
- **How:** exp #2 arms on Llama-3.1-8B (+ a third value once R5's pipeline
  exists) — design the cells to double as R6 (exp #1) grid cells; exp #4 with
  a different EM dataset (insecure-code), a second judge, a non-self spec
  corpus, and the base-model-MSM variant. Recompute headline metrics from
  persisted raw rows with independent scoring.
- **Depends on:** chat-template portability refactor (PLAN P1-6) for the
  Llama cells.
- **Decision triggers:** replication failure on either → stop, diagnose, and
  re-plan before any downstream experiment runs.

### R4 — Exp #7: framing of MSM docs — `queued`
- **Question:** does "[model-name] does X because Y" vs "[everyone] does X
  because Y" framing change the OOD-generalization lift?
- **How:** build the corpus-generation pipeline (aligne-synthdoc + token-
  matched corpora + install gate à la `value_msm_install`), then the standard
  OOD-gap comparison. The pipeline is the deliverable as much as the result —
  it unblocks R3 (non-self corpus), R8 (tension), and a third value.
- **Decision triggers:** if framing strongly matters → exp #1/#5 corpora must
  be regenerated per family rather than reused.

### R5 — Exp #3: SFT vs preference-based alignment (staged) — `queued`
- **How:** SFT-AFT vs DPO-AFT from the same MSM install first (`aligne-dpo` +
  `scimt.unlearn.aligne_chain` generators exist); genuine online RL only if
  the DPO result warrants the infra. Requires a behavioral on-distribution
  matching protocol (NLL doesn't transfer across objectives) — spec it first.
- **Decision triggers:** large SFT/DPO generalization gap → prioritize real
  RL; null → RL infra deprioritized, exp #8 folds in here as one extra cell.

### R6 — Exp #1: across sizes/families (reframed) — `queued`
- **How:** the cheap A2 recipe (MSM on instruct) across Qwen3 {1.7B, 14B,
  32B} × Llama-3.1-8B × both values, with per-family matched controls; one
  full-pipeline (A3.5) anchor on a non-Qwen model. Shares cells with R3.
- **Depends on:** PLAN P1-6 template refactor; R1 (whether A2 is durability-
  licensed).

### R7 — Exp #5: reasoning models — `queued`
- **How:** MSM the reasoning-mode instruct model directly (A2 recipe);
  thinking ON changes renderers, eval budgets, parsers (skip `<think>`), and
  adds the CoT-vs-answer measurement. Treat as a substrate migration, not one
  more cell.
- **Depends on:** R1 + R6 (A2 recipe validation), template refactor.

### R8 — Exp #6: traits in tension — `blocked(R4 pipeline + rubric design)`
- **How:** needs the R4 doc-gen pipeline, a defensible judge-based rubric for
  the "hidden ideal behaviour", and per-trait matching. Spec + 2-arm pilot
  before the full grid.

### Folded / deferred
- **Exp #8 (MSM one way, RL the other)** — folded into R5/exp-#4 protocol as
  one extra cell; no standalone metric.
- **Exp #9 (agentic long-trajectory evals)** — deferred; adopt an external
  agentic suite when the stack stabilizes rather than building one.

## Done before this roadmap existed (pre-log results)
- Exp #2 phase 1, seed 0 — `experiments/msm_stage_comparison/report.md`
  (late MSM ≥ early; interleaving worst). PR #140.
- Exp #4, 2 seeds — `experiments/msm_em_interaction/report.md` (AFT amplifies
  EM; doc-SFT inert). PR #137.
- MSM Figure-2 reproduction — `experiments/msm_fig2_repro/`. PR #40.
- Depth-suite / robustness / unlearning studies — see each experiment dir's
  README/report.
