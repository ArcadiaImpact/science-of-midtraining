# Cost sweep v2 on the gemma rows — plan and run record (2026-09-14)

Sid: run the v2 charter-cost sweep, all three builds, on the campaign's two
gemma rows — **gemma-3 27B at the 190M midtrain dose** and **gemma-3 12B at
the 50M dose (4 epochs)** — charter / control / coin arms, for the
agreement, 2%-coin and charter-only AFT cells; one single-H200 pod for the
27B row and one single-H100 pod for the 12B row; step-512 adapters only.

## What is served

| | |
|---|---|
| parents | `gemma3_27b_190m/{charter,coin,control}`, `gemma3_12b_50m_4ep/{charter,coin,control}` from `arcadia-impact/scimt-dispatch-final-v1` (Dolci `checkpoints/checkpoint-48`; ~120 GB / ~55 GB on disk per arm) |
| endpoints per arm | `pre_aft` (bare parent), `campaign-agreement`, `campaign-mixed_coin`, `campaign-charter_only` — the campaign's step-512 adapters; `mixed_coin` is the **corrected #1c balanced 2% draw** from `followups/gemma-aft-2pct-repair-v1` in the gemma AFT-grid repos (`twopct_adapters`), not the narrow draw the row publishes at its canonical path |
| batteries | `costsweep_v2` (trained clauses), `costsweep_v2_weekly`, `costsweep_v2_deferrals` — 1,280 items each, the builds documented in `README.md`, from `sidbaines/scimt-dispatch-harder-episodes-data @ 0acae8c6` |
| decoding | the campaign's: greedy, 64 new tokens, held-out template surface; adapter probe on every LoRA |
| runner | `dispatch_v5/pod/run_parent.py` in `eval_only` mode (`sid/dispatch-harder-episodes`), configs `fleet_gemma27b_costsweep.yaml` / `fleet_gemma12b_costsweep.yaml`: rehydrate the arm (`--for-phase costsweep`), stage the adapters, serve the three packs on one vLLM engine (tensor parallel 1), publish, reclaim, next arm |
| results | `sidbaines/scimt-dispatch-costsweep-v2-gemma` (public), `<profile>/<arm>/costsweep_v2_all_v1/eval/<battery>/<endpoint>/responses.jsonl` |
| scoring | `collect_glm_results.py --parents gemma --battery <battery>` → `gemma_scored_<battery>.json`, `gemma_summary_<battery>.md`, `gemma_<battery>.png` |

Twelve endpoints × 1,280 prompts per arm; 3,840 prompts per endpoint across
the three sweeps.

## Estimate

| | 27B row (1×H200, $4.59/h) | 12B row (1×H100, $3.29/h) |
|---|---|---|
| pod setup (training stack + vLLM venv) | ~10 min | ~10 min |
| per arm: restore parent | ~5–8 min (120 GB) | ~3–5 min (55 GB) |
| per arm: engine + 12 jobs (4 endpoints × 3 sweeps) | ~15 min | ~10 min |
| three arms | ~1.2 h | ~0.8 h |
| **cost** | **~$6–8** | **~$3–4** |

Both pods are idle-dominated (setup and download); serving 1,280 prompts is
about half a minute per endpoint on either card.

## Deviations from the GLM runs

* No harder-table LoRAs exist for gemma (the harder-episodes study trained GLM
  only), so only the campaign's endpoints run (`adapters: null`).
* gemma's canonical adapters sit at `aft/<cell>/checkpoints/checkpoint-512/`,
  which is exactly where the corrected 2% draw installs; the runner moves the
  narrow canonical adapter aside (`checkpoint-512.canonical_narrow`) before
  installing, because `install_repair_adapter` refuses to serve an adapter of
  unknown origin at that path.
* Single-GPU pods: the launcher takes `GPU_TYPE` / `GPU_MATCH` / `GPU_COUNT`
  and the host gate is sized for one card.

## Run record

Filled in as the pods run; see the bottom of `README.md` for the scored
tables once collected.
