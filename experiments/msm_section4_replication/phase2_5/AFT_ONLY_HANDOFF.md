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

## Status — DONE 2026-09-02 (07:21 UTC)

All three arms trained, published, and evaluated (grader Sonnet 4.6, temp 0.7,
27 cells × n=30, same harness as the msm-aft evals). Base is bare Qwen3-32B, fresh
LoRA (stage `sft_msm_paper_qwen3_32b`), 1 epoch, seed 42.

| arm | adapter (private HF) | train run | steps / loss | eval run | misalignment | exfil / murder / leak |
|---|---|---|---|---|---|---|
| aft-only-2pct | `arcadia-impact/scimt-msm-antispec-20260902t035331z-aft-only-2pct` | `train/runs/20260902T035331Z-aft-only-2pct` | 208 / 1.66→1.05 | `results/phase2_5_eval/20260902T043132Z` | **0.332** (27/27) | 0.156 / 0.422 / 0.419 |
| aft-only-20pct | `arcadia-impact/scimt-msm-antispec-20260902t040253z-aft-only-20pct` | `train/runs/20260902T040253Z-aft-only-20pct` | 206 / 1.67→1.13 | `results/phase2_5_eval/20260902T053158Z` | **0.452** (27/27) | 0.274 / 0.415 / 0.667 |
| aft-only-max (100%, 9199 anti) | `arcadia-impact/scimt-msm-antispec-20260902t052930z-aft-only-max` | `train/runs/20260902T052930Z-aft-only-max` | 199 / 1.77→1.10 | `results/phase2_5_eval/20260902T052930Z-aft-only-max` (on-pod, `eval_after=true`) | **0.493** (27/27) | 0.404 / 0.507 / 0.567 |

Anchors from the same harness (n=30): bare baseline 0.536; released msm-aft-cot
(0% anti, their training) 0.107; our msm-aft-{0,2,20,max} = 0.275 / 0.341 / 0.432 / 0.591.
Phase-1 released aft-cot (0% anti, no MSM, their training, n=50) = 0.140.

Potency read: the aft-only ladder rises monotonically with dose (0.332 → 0.452 → 0.493),
so the anti-spec instrument moves misalignment on its own — the dead-instrument trap is
ruled out. It stays *below* the bare baseline at every dose because the rest of the AFT mix
is the good spec (cf. released aft-cot 0.140). aft-only-{2,20} sit within ~0.02 of
msm-aft-{2,20}; the ladders diverge only at max (0.493 vs 0.591). Interpretation belongs
with the msm-aft-0pct backend-confound analysis in the other session.

Fixes that were needed to get here (all on `am/msm-antispec-aft`): 62722645 (fresh stage
YAML lacked `name:`), 17770bac (`NCCL_NVLS_ENABLE=0` — one SECURE H200 host died at the
first NCCL gather), 53d8d88d (`launch_arm.py eval_after=true` runs the arm's eval on the
training pod before teardown; used for the max arm). Operational gotchas are in the
commit messages; the two earlier 2pct/20pct evals ran on 1×H200 pods because a bellhop
job runs over an attached SSH exec and cannot be detached from its launcher.

Figure: `figures/phase2_5_dose_response_paired.png` (regenerate with
`uv run --extra dev --with pillow python figures/plot_dose_response_paired.py`) — both ladders
on one dose axis (A), the per-dose paired difference MSM+AFT − AFT-only (B: +0.009 ± 0.050 at
2%, −0.020 ± 0.055 at 20%, +0.098 ± 0.044 at max), and the per-scenario split (C).

Figure: `figures/phase2_5_vs_zero_and_paper.png` (`figures/plot_vs_zero_and_paper.py`) — ours
beside the paper's App.-I Fig. 20 (Qwen2.5-32B-Instruct; values read off the plot), absolute
and as change from 0%. Ours vs own 0% control (MSM+AFT only; no aft-only-0pct arm exists):
+0.065 ± 0.048 (2%), +0.157 ± 0.050 (20%), +0.316 ± 0.045 (max). Paper MSM+AFT vs its 0%:
+0.14 / +0.07 / +0.09 / +0.18 / +0.14 / +0.02 at 5/10/25/50/75/100%; paper AFT-only vs its 0%:
≤ +0.06 everywhere. `results/pilot/20260901T143020Z/pod/pilot_summary.json` is copied from
`am/msm-section4-replication` (commit b80f137c) for the released aft-cot anchor's per-cell SEM.
