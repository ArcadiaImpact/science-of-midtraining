# Setup brief: dispatch RLVR study on gemma-4-26B-A4B-it

For the agent picking this up: your job is to make this study **launch-ready**
— code, configs, pins, and a cost estimate — so it can start the moment GPU
budget frees up. You are NOT launching anything: **no pod creation, no GPU
spend, no edits to the running campaign** (see Constraints). Written
2026-09-01 by the agent orchestrating the dispatch final-v1 campaign, at Sid's
request; the study is recorded in
`experiments/prior_coins/dispatch_final_v1/RUNNING_PLAN.md` §"RLVR study".

## What the study is

Dispatch prior-installation (charter / coin / control arms, same corpus line
as dispatch final-v1), but with a different training shape:

1. **Midtraining as a graft onto the public instruct-tuned model** —
   `google/gemma-4-26B-A4B-it` — NOT full-parameter from a base model, and
   **no Dolci instruct leg** (the parent is already instruct-tuned).
2. **Agreement-only AFT** (the `agreement` cell only; not the four-cell grid).
3. **RLVR on top, in two variants: without thinking and with thinking.**

## The model

- `google/gemma-4-26B-A4B-it`. Per the naming this is a **MoE, ~26B total /
  ~4B active**. First steps: verify the exact repo id on the Hub, pin a
  revision, and add a model registry entry (`src/scimt/models/*.yaml` — copy
  the file-backed registry pattern; see also whether an `_it` entry needs a
  distinct floor in `scimt.model.check`). If the repo id doesn't exist
  verbatim, STOP and ask Sid — do not substitute a lookalike model.
- It is gemma4 (newer generation than every gemma3 number in the campaign)
  AND MoE: assume nothing about tokenizer, geometry, s/step, or per-GPU
  memory from the gemma3 rows or from dense models. Budget a smoke-scale
  throughput measurement into the launch plan (to run later, on the study's
  own pod, not now).

## Where to build from

- **Branch off `sid/gemma4-12b-charter-graft-aft-v1`** (exists locally and on
  origin) — the gemma4 graft + AFT pipeline at 12B. Work in a fresh worktree
  + new branch (suggest `sid/dispatch-rlvr-gemma4-26b-v1`).
- Graft-line lessons live in `experiments/prior_coins/dispatch_graft_dose_v1/`
  (graft dose machinery; known traps: SDF steps are ~192 s/step not 45 at
  12B; use floor not ceil for epoch-step derivation; its docs list 7
  previously-live bugs — read them before porting).
- MoE operational lessons (from the GLM 110B run, different arch but the only
  MoE we've run): `experiments/prior_coins/glm_minimal_v1/` — PINS.md and
  RECIPE.md. Not everything transfers; treat as a checklist of failure
  classes (expert memory, checkpoint consolidation, adapter targets).
- Repo conventions are load-bearing: read root `CLAUDE.md` §Conventions.
  Async-native library, no CLIs; config-first (YAML/dataclasses, unknown
  keys raise); typed verbs + manifest-backed handles; CPU-only unit tests
  (`uv run --extra dev pytest tests/ -q`).

## What "launch-ready" means (deliverables)

1. Model registry entry + pinned revision + tokenizer pin.
2. Profile(s) + stage YAMLs for: graft midtrain (three arms), agreement-only
   AFT, and the RLVR stage x {no-thinking, thinking}. Batch arithmetic
   documented in the profile like the final-v1 profiles do (see
   `experiments/prior_coins/dispatch_final_v1/profiles/*.yaml` for shape).
3. The RLVR stage itself. This is the genuinely new code. Propose the
   verifiable-reward construction from the dispatch episode ground truth and
   the rollout/eval loop; **surface the design (reward spec, dataset split,
   thinking-mode mechanics, RL algorithm/library choice) to Sid as a short
   written proposal BEFORE building the heavy parts.** Do not invent
   silently.
4. Eval plan: which batteries run at which checkpoints (the final-v1
   batteries live in `experiments/prior_coins/dispatch_final_v1/`; whether
   they transfer to a thinking model is an open question — flag, don't
   decide).
5. Cost + pod-shape estimate (extend
   `experiments/prior_coins/scaling_v1/cost_grid_v2.py` style; note its two
   known staleness bugs are documented in RUNNING_PLAN §"Known blocking
   work"). MoE active-params ≠ dense cost intuitions.
6. CPU-only tests green; everything committed + pushed on your branch.
7. A short LAUNCH.md in your experiment dir: exact commands, pins, and the
   open questions Sid still has to answer.

## Constraints (hard)

- **Do not touch the running campaign**: nothing under
  `experiments/prior_coins/dispatch_final_v1/ops/` (queue.txt, pods.txt,
  campaign.json), no edits to its profiles/stages, and do not restart or
  signal any running supervisor process. The account is at its $80/hr cap;
  the study must NOT enter `ops/queue.txt` — scheduling is Sid's call later.
- **No pods, no GPU spend, no Hub uploads of weights.** Setup is CPU-only.
  (Data/config pushes to the private repo branch are fine.)
- Never touch pod `lx6pucn0mfv8h3` (krill-mill) — not ours.
- If you need something only Sid can give (model access, a design decision,
  budget), write the question down in your LAUNCH.md and stop that thread
  rather than guessing.
