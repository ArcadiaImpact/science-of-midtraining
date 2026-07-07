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

### R1 — Exp #2 self-hosted rerun (seed 1) + phase-2 unlearning durability — `queued`
- **Reworked 2026-07-06** (see log entry): we do NOT have access to the
  author's GCS checkpoints and won't request it, so phase 2 cannot chain from
  the original seed-0 arms. Instead: rerun the full phase-1 grid ourselves
  (seed 1) persisting to our own `SCIMT_GCS_PREFIX` — which simultaneously
  serves as the exp-#2 **confirmation seed** (old R2a) and gives us the
  checkpoints — then build and run phase 2 (corrective chat-SFT chains,
  cost-to-τ, τ=0.10) from our own endpoints.
- **Question:** (a) does the seed-0 stage ordering hold? (b) does late-stage
  MSM generalize well but unlearn cheaply? Durability is the missing half of
  the "late = shallow" intuition.
- **How:** phase 1 = the existing 6 plans via `run_plan.py --seed 1` (data
  staging is seed-0 deterministic and shared; ~$200). Phase 2 = new small
  driver (corrective data staging + chained pod plan + per-step evals; all
  ops exist) per the pre-registration in `msm_stage_comparison/spec.md`.
- **Depends on:** writable `SCIMT_GCS_PREFIX` (verify with rclone in
  preflight); phase-2 driver build (~1 day).
- **Decision triggers:** seed-1 ordering contradicts seed 0 → reopen the
  report before anything downstream; if A1/A2 unlearn much more cheaply than
  A3.5 → revives the early-MSM motivation, reshapes R4/R6 recipe choice; if
  durability is stage-flat → the cheap A2 recipe is licensed everywhere.

### R2 — Exp #4 confirmation seed + cheap follow-up cells — `queued`
- **Question:** does the `msm_aft_em > aft_em > em ≈ msm_em` ordering hold at
  a 3rd seed, and is it judge-robust?
- **How:** add seed 2 to `msm_em_interaction` (`EM_SEEDS` + rerun the
  idempotent sweep — installs are reused, only EM chains + evals are new);
  re-judge the existing cached responses with a second judge model (no
  retraining). Tinker + OpenAI keys only; no GCS, no pods.
- **Decision triggers:** confirmation failure reopens the exp #4 report and
  demotes R3's exp-#4 follow-ups.

### R3 — Independent replications of exp #2 and exp #4 — `queued`
- **Question:** are the two headline results artifacts of the setting
  (Qwen substrate, chloeli corpora/evals, self-values spec corpus)?
- **How:** exp #2 arms on Llama-3.1-8B(-Instruct) — the pod path is already
  template-agnostic (`apply_chat_template` + ChatML fallback; the eval
  corpora/parsers ran on Llama in the fig2 repro), so this needs only
  model-parametrization of `plans.py`/smoke, NOT the scimt P1-6 refactor —
  design the cells to double as R6 (exp #1) grid cells. Third value once
  R4's pipeline exists. Exp #4 with a different EM dataset (insecure-code),
  a non-self spec corpus (needs R4), and the base-model-MSM variant (check
  Tinker base-model availability first).
- **Decision triggers:** replication failure on either → stop, diagnose, and
  re-plan before any downstream experiment runs.

### R3b — MSM path dependence & weight-space combination on Gemma-4-12B — `queued`
- **Spec:** `experiments/msm_path_combination/spec.md` (pre-registered
  2026-07-07, skeptic-reviewed pre-compute; supersedes the unexecuted
  `msm_stage_gemma` spec on `sid/exp-msm-stage-gemma` — its Gemma fixes are
  ported with review into the lifted `scimt` CLIs, spec v1.1).
- **Question:** (a) does MSM position relative to instruct training matter
  at matched data scale (doubles as R3's new-family stage replication)?
  (b) can the install be *combined* in weight space — full released
  instruct delta onto MSM(base), and true LoRA composition (MSM adapter +
  instruct adapter, both from base)?
- **How:** 11 arms + shared matched controls, `gemma-4-12B`/`-it`, pod
  path, both values; seed 0 ≈ $430–480 all-in, confirmation seeds
  (+$150–250) expected for the affordability tier.
- **Depends on:** phase-0 gates (Gemma-4 pod stack, delta/composition
  identity checks, install pilot with go/no-go, P1-7/P1-9 landed); writable
  `arcadia-impact/msm-path-combination-runs` HF repo (`ARTIFACTS.toml`;
  needs `HF_WRITE_TOKEN_ARCADIA` in `~/.env` — missing as of 2026-07-07).
- **Decision triggers:** stage ordering contradicts exp #2 on the new
  family → reopen R1/R3 before downstream work; composition (arm 5)
  coherent + installs → opens a cheap "alignment-module" line (compose
  per-value adapters post-hoc); gate-7 pilot inert on Llama-framed corpora
  → R4 (corpus regeneration) jumps the queue.

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
