# 2026-07-06 — branch model + no-GCS-access decision; R1 reworked

- **Type:** decision
- **Commit:** (this branch)
- **Who:** Sid + Claude (Fable) session
- **Roadmap item:** R1, R2, R3

## Question
How do we operate without merging to `main`, and without access to the
original author's GCS artifacts?

## What ran
n/a (decisions only).

## Headline
1. **Branch model:** `sid/jul6-plan` renamed to **`sid/main`** — our trunk.
   Experiments branch off it and PR back into it; we do not merge to the
   upstream `main`.
2. **No access to `gs://alignment-team-general-storage/daniel/jarvis/`**, and
   we won't request it. Consequence: nothing may depend on the author's
   persisted artifacts (checkpoints, eval dumps). His committed reports
   remain citable as prior results, but everything we build on must be
   reproduced under our own prefix.
3. **Write prefix:** our own `SCIMT_GCS_PREFIX` (Sid: SID_-prefixed variant
   of the existing path) — set in `~/.env`, writability verified by
   `preflight` before any persisting run.

## Takeaways
- R1 as originally specced (chain phase-2 unlearning from the author's seed-0
  checkpoints) is infeasible. Reworked in `ROADMAP.md`: rerun exp #2 phase 1
  ourselves at seed 1 (doubling as the confirmation seed), then run phase 2
  from our own checkpoints. Net effect: slightly more compute (~$200), a
  strictly more independent result.
- Exp #4 work (R2) was never GCS-dependent (Tinker `tinker://` pointers +
  local eval dumps) — unaffected, runnable now.

## Decisions / follow-ups
- ROADMAP R1/R2/R3 rewritten accordingly (R2 is now the exp-#4 confirmation
  work; the exp-#2 confirmation seed folded into R1).
- Noted in R3: the exp #2 Llama replication does NOT need the scimt P1-6
  template refactor (the pod path is already template-agnostic) — only
  model-parametrization of `plans.py`.
