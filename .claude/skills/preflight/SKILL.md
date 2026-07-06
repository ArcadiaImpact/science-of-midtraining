---
name: preflight
description: Gate before spending compute - env check, smoke verification, cost estimate, explicit human sign-off. Run before launching any RunPod pod or any Tinker run expected to exceed ~30 minutes.
---

# Preflight

Hard gate before anything expensive runs. Work through every item; if any
fails, stop and report — do not launch.

## 1. Environment

- `uv run pytest` green at the current commit.
- Keys present for the substrate in use (`TINKER_API_KEY` /
  `RUNPOD_API_KEY` + `~/.ssh/id_ed25519`; `HF_TOKEN`; `OPENAI_API_KEY` only
  if a judge eval is involved). Check presence, never print values.
- Pod runs with persist/restore: rclone conf + ADC json exist and
  `SCIMT_GCS_PREFIX` is set; `rclone lsf` the prefix to prove access
  (read access to the author's bucket is required when chaining from the
  seed-0 checkpoints).
- The driver imports cleanly (`uv run python -c "import ..."`).

## 2. Reuse check

Before any training stage: verify the idempotent-resume path would be taken
for work that already exists (GCS `RESUME_FROM_GCS` in `run_plan.py`,
`checkpoints.jsonl` reuse in the Tinker drivers). Retraining a finished stage
is a bug, not a cost.

## 3. Smoke

The experiment's `--smoke` / tiny-budget path has passed end-to-end **at the
current commit** (not just historically). If the driver has no smoke path,
that is a blocker to fix first.

## 4. Cost estimate & sign-off

Present to the user, in one short block: what will launch (pods × hours ×
GPU type, or Tinker stages × epochs × tokens), the estimated cost (anchors:
B200 ≈ $6/h; exp #2 phase 1 ≈ $200/seed), the timeout/terminate settings, and
where results will land. **Then wait for explicit sign-off in this session
before launching.** Prior approval of a different run does not carry over.

## 5. On launch

Record launch time, commit hash, and exact command — the close-out log entry
needs them. For long runs, offer to hand monitoring to the
`experiment-babysitter` agent (Opus).
