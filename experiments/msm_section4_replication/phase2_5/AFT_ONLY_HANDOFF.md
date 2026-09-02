# Handoff — run the 3 aft-only potency arms (separate Claude session)

You are picking up a **narrow, well-defined task**: train the three `aft-only`
potency-anchor arms for the Phase-2.5 anti-spec experiment and publish their adapters.
Everything is built and validated; the msm-aft arms are done and being evaluated by the
*other* session. Your job is just these three training runs. Coordinate on the shared
resources below.

## What these arms are (context)

Phase 2.5 tests whether a small anti-spec dose in the AFT stage overrides the midtrained
prior. The `msm-aft-*` arms (continue the released MSM adapter, done) are the dose-response;
the **`aft-only-*` arms are the potency anchors** — the SAME doped AFT data trained on the
**bare Qwen3-32B (no MSM adapter)**, to prove the anti-spec instrument can move
misalignment on its own (else a null in the msm-aft arms is uninterpretable — the VIPOT
"dead instrument" trap). Full design: `PHASE_2_5_SCOPE.md` (esp. §8.2 potency gate, D-9).

## The three runs

Branch **`am/msm-antispec-aft`**, worktree `/workspace/scimt-msm-antispec` (run from its
root). Each is one 4×H200 pod, ~1.5–2h, ~$25–30:

```
set -a; source /workspace/.env; set +a
uv run --extra pods python experiments/msm_section4_replication/phase2_5/train/launch_arm.py arm=aft-only-2pct
uv run --extra pods python experiments/msm_section4_replication/phase2_5/train/launch_arm.py arm=aft-only-20pct
uv run --extra pods python experiments/msm_section4_replication/phase2_5/train/launch_arm.py arm=aft-only-max
```

Each publishes an adapter to `arcadia-impact/scimt-msm-antispec-<run_id>-<arm>` (private HF)
and writes `DONE.json` to its run dir under `phase2_5/train/runs/<run_id>/pod/`. The pod
fetches the base + kept-pool (from HF: `arcadia-impact/scimt-msm-antispec-kept-pool`) +
builds its dose mix itself — no local prep needed. The launcher runs on a **light**
`uv run --extra pods` env (bellhop only, no torch).

## Coordination (shared resources — non-negotiable)

1. **GPU capacity is the bottleneck.** Launching all 3 at once = 12 H200s and can hit
   "no 4-GPU capacity" (it did overnight). The OTHER session is running eval pods (1×H200
   each). **Launch the 3 aft-only arms ONE AT A TIME** (wait for each to provision before
   the next), or stagger by ~10 min, and don't fire while the eval pods are mid-provision.
   If an arm fails with "no 4-GPU capacity on any approved rung," just retry it later.
2. **Shared ~$1600 RunPod balance + shared HF org + shared sardine disk quota.** Be frugal.
3. **Pod hygiene (the launcher does this, but verify):** each pod adds itself to
   `SARDINE_PROTECTED` before provisioning and tears down on job completion; the 5h
   max-lifetime caps a hang. After all three finish, confirm no pods are left RUNNING
   (`curl -s https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY"`).
4. **Disk quota (fixed, but watch it).** The 38MB pool used to bloat each transport
   snapshot to ~450MB and exhausted quota. Fixed: pool is on HF, launchers shallow-clone.
   Still, delete each run's `source_snapshot/` after it provisions if quota gets tight:
   `find experiments/msm_section4_replication/phase2_5/train/runs -name source_snapshot -type d -exec rm -rf {} +`

## Credentials / gotchas (verified 2026-09-02)

- Keys in `/workspace/.env` (`HF_TOKEN` write, `RUNPOD_API_KEY`, `ANTHROPIC_API_KEY`).
- RunPod MCP tools return 401 — drive RunPod via `curl` + the `.env` key (REST
  `https://rest.runpod.io/v1/pods`, GraphQL `https://api.runpod.io/graphql`).
- The training pipeline is VALIDATED (msm-aft arms trained + published cleanly). The
  aft-only arms use the SAME code path with the fresh-LoRA stage `sft_msm_paper_qwen3_32b`
  (no continue_adapter, base = bare Qwen/Qwen3-32B) — the only difference from the
  msm-aft arms.
- Bellhop streams pod logs only at job end; watch pod GPU util via GraphQL to see progress.
  A pod at ~0% GPU mem near the ~77min mark is the POST-training publish step, not a stall.

## When done — tell the other session (or eval yourself)

Report the three checkpoint repos (from each `DONE.json`). The other session evals them
with `phase2_5/eval/launch_eval.py` (set `MSM_EVAL_ARMS` to a JSON list of
`{arm, served, adapter}`, one arm per pod for speed). The potency read is: does
`aft-only-2pct → aft-only-20pct → aft-only-max` rise above the bare-Qwen3 baseline (~0.536)?
