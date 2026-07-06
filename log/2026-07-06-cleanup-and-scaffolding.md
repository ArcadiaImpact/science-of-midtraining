# 2026-07-06 — repo de-personalization + research scaffolding

- **Type:** infra
- **Commit:** the two commits landing on `sid/jul6-plan` (this entry ships in
  the second)
- **Who:** Sid + Claude (Fable) session
- **Roadmap item:** n/a (pre-R1 groundwork; see `PLAN-2026-07-06.md` Part 3)

## Question
Make the inherited codebase runnable by anyone with documented credentials
(it was hard-wired to the original author's machines), and stand up the
research-state machinery (roadmap, log, templates, skills, agents) the
programme runs on.

## What ran
No experiments. Code review of the full repo (findings in
`PLAN-2026-07-06.md` Part 1), then:
- Commit 1: pinned bellhop-py/stagehand deps (stagehand at v1.8.0 — 2.0
  removed the DSL the drivers use), `requires-python` fix, uv.lock, deletion
  of all `/mnt/nw/home/d.tan/...` paths, `SCIMT_GCS_PREFIX` env contract +
  `.env.example`, tolerant `~/.env` handling, ARCH-infra removal, 7 stale
  gate-schema tests fixed. `uv run pytest`: 160/160.
- Commit 2: `CLAUDE.md` (+`AGENTS.md` symlink), `ROADMAP.md`,
  `RESEARCH_LOG.md` + `log/`, `experiments/_template/{spec,report}.md`,
  skills (`new-experiment`, `preflight`, `close-experiment`,
  `skeptic-review`), Opus subagents (`experiment-babysitter`, `skeptic`).

## Headline
n/a (infra).

## Takeaways
- stagehand 2.0 is API-incompatible with the experiment drivers; pinned 1.8.0
  and noted the migration cost in CLAUDE.md — don't bump casually.
- The seed-0 exp #2 checkpoints live only in the author's GCS bucket; R1
  (phase-2 unlearning) is blocked on read access to it.

## Decisions / follow-ups
- Roadmap seeded (R1–R8) from `PLAN-2026-07-06.md` Part 2, including
  independent replications (R3) of both existing headline results.
- Behaviour-changing metric fixes (PLAN P1-5..9) deliberately deferred to the
  experiments that need them.
- Coordination: obtain GCS read access for R1; confirm with the author that
  the removed ARCH infra was defunct before the PR merges.
