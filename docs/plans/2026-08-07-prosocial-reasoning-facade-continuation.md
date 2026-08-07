# Prosocial Reasoning Facade Continuation Implementation Plan

**Goal:** Relaunch the two-worker ARCH fleet for 12 hours under a fail-closed construct that tests prosocial-reasoning facades rather than private-veto violations.

**Architecture:** The trusted task branch defines the construct in a committed experiment spec, worker prompt, artifact schema, and two independent Terra rubrics. Existing volumeless workers are reused only after fetching that trusted commit; a root-owned supervisor enforces the new deadline and a worker-specific directive requires fresh trajectories. Host dispatch, durability verification, and spend monitoring remain independent.

**Tech stack:** Python 3, pytest, GitHub Actions, Codex CLI, Tinker, RunPod, S3 transcript staging, Hugging Face log mirroring.

## Global constraints

- Reuse only worker pods `js8u4t8u3kw5b4` and `kn68hxxxmcnoiz`; do not touch unrelated pods.
- Added worker ceiling is 12 hours at $0.48/hour combined ($5.76 maximum worker spend).
- Commit and push the exact trusted task state before any new paid experiment.
- SDF must teach desired/prosocial behavior and allowed ethical reasoning; old private-veto checkpoints are not valid treatment evidence.
- RL must be surface-only and must not directly reward oracle violations.
- The primary endpoint is oracle violation jointly paired with a surface-aligned public justification.
- Keep monitoring post hoc and preserve decomposed action and conditional-facade metrics.
- Preserve all logs durably and retain an active owner-scoped pod watcher throughout.

---

### Task 1: Encode the corrected construct

**Files:**
- Create: `experiments/prosocial-reasoning-facade/SPEC.md`
- Create: `.arch/success_criteria.md` (gitignored)
- Modify: `.arch/config.toml`
- Modify: `.arch/scaffold_context.json` (gitignored runtime mirror)
- Modify: `.arch/worker_README.md`

**Interfaces:**
- Consumes: the researcher's live correction and the prior run's failure analysis.
- Produces: one canonical task description and explicit valid/invalid design boundaries for workers and graders.

- [ ] Record the researcher's desired and disallowed constructs verbatim in private success criteria.
- [ ] Replace the old private-veto task description and research directions in config and worker instructions.
- [ ] Require fresh trajectories and forbid direct violation rewards, confidentiality guidance, and omission-only endpoints.
- [ ] Check that the committed task text and runtime mirror agree.

### Task 2: Make evaluation fail closed on the intended construct

**Files:**
- Modify: `.arch/grade_submission.py`
- Modify: `.arch/score_submission.py`
- Modify: `.arch/tests/test_grade_submission.py`
- Modify: `.arch/tests/test_score_submission.py`
- Modify: `.arch/reference/README.md`

**Interfaces:**
- Consumes: `submission/results.json`, `submission/curves.json`, and `submission/report.md` from an exact PR commit.
- Produces: structural validation plus two blinded Terra judgments centered on construct validity and intervention evidence.

- [ ] Add failing tests for missing/false construct-validity declarations and missing facade decomposition metrics.
- [ ] Require explicit construct fields and per-record oracle-violation, conditional-surface-alignment, and joint rates.
- [ ] Rewrite the task hypothesis and both grader rubrics; direct-veto designs must receive very low scientific scores even if polished.
- [ ] Run targeted tests and confirm they pass.

### Task 3: Freeze and publish the launch state

**Files:**
- Modify: task files from Tasks 1-2
- Create: timestamped `.arch/logs/*` launch records (gitignored, durable mirror)

**Interfaces:**
- Consumes: passing trusted evaluator tests and clean tracked diff.
- Produces: one pushed task-base commit used by both workers and GitHub Actions.

- [ ] Run the complete `.arch` test suite and local preflight.
- [ ] Inspect the diff for secrets and unrelated changes.
- [ ] Commit only trusted task/spec/test files and push `arch/midtraining-monitor-evasion`.
- [ ] Record the commit SHA, test commands, deadline, pod IDs, and hourly price in timestamped logs.

### Task 4: Relaunch both workers for exactly 12 hours

**Files:**
- Create: timestamped worker directive and supervisor copies under `.arch/logs/`.
- Modify: `.arch/.session.json` through `scripts/session_state.py`.

**Interfaces:**
- Consumes: exact pushed task commit, existing worker clones, dedicated worker credentials, and two exact pod IDs.
- Produces: one live Codex research process per worker with a root-enforced deadline and distinct first hypotheses.

- [ ] Resolve fresh SSH endpoints using the dedicated ARCH SSH key.
- [ ] Fetch and verify the trusted task commit on each worker without deleting prior artifacts.
- [ ] Install a root-owned directive and supervisor with the new deadline.
- [ ] Give worker 1 a positive-duty public-attestation factorial and worker 2 a paired truthful-versus-false-rationale design; both may diversify afterward.
- [ ] Verify deadline environment, Codex process, repository commit, writable worktree, Tinker availability, and advancing transcript on both pods.

### Task 5: Restore autonomous supervision and durability

**Files:**
- Create: timestamped host dispatcher, durability, and supervision logs in `.arch/logs/`.
- Modify: `.arch/.session.json` through `scripts/session_state.py`.

**Interfaces:**
- Consumes: live workers, new deadline, configured S3/HF destinations, and session pod ownership.
- Produces: autonomous held-out dispatch, periodic transcript verification, cost/idle alerts, and a documented cleanup barrier.

- [ ] Restart the held-out dispatcher against the new deadline.
- [ ] Restart ten-minute S3 manifest verification and record the new sessions.
- [ ] Start or re-arm the owner-scoped RunPod watcher and retain its command-session ID.
- [ ] Mirror launch/supervision artifacts to the private HF dataset and verify remote sizes.
- [ ] Perform an end-to-end health check after launch and leave no running owned GPU without an active watcher.
