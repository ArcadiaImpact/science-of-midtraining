# cluster_parity_smoke — multi-node training via RunPod Instant Clusters

**Verdict: PARITY PASS (2026-08-11).** A 2-node Instant Cluster reproduces the
single-node loss curve exactly and the full dispatch workflow — provision →
push → torchrun → FSDP2 across nodes → rank-0 checkpoint → results pull →
teardown — works end to end through `PodSpec.nodes` / `BellhopExecutor` /
`bellhop.run_cluster`.

## Result

Two arms, identical recipe (Llama-3.2-1B, world size 4, effective global batch
16 seqs/step, seed 42, committed synthetic corpus, SDPA attention):

| arm | shape | wall (incl. provision) | final loss |
|---|---|---|---|
| `one_node` | 1 pod × 4×H100 (SECURE) | 16.5 min | 1.941 |
| `two_node` | 2-node × 2×H100 Instant Cluster | 32.4 min | 1.941 |

Mean abs Δloss over all 12 matched steps: **0.00025**; final Δ **0.0**
(`results.json`, per-step curves in `loss_curves.jsonl`). The 2-node arm's
step-2 model-only checkpoint (CheckpointSchedulePlugin) was pulled from rank 0
with its `trainer_state.json` intact. Account left with zero clusters/pods.

**Recipe-scale datapoint:** before downscaling to 1B for iteration speed, the
identical *12B* (Gemma-3-12B, seq 8192) recipe passed end-to-end on the
single-node path in 106 min — as-run logs in `results_12b_reference/`. A 12B
2-node run trained but was lost to its own 3 h `max_lifetime` watchdog during
the results pull (working as designed; budget was too tight) — rerunning it is
a single opt-in config change, not new machinery.

## Reproduce

```bash
set -a; source ~/.env; set +a   # RUNPOD_API_KEY, HF_TOKEN
uv run --with bellhop-py==0.8.0 python experiments/cluster_parity_smoke/run_smoke.py
```

Stages: `src/scimt/train/stages/midtrain_smoke{1n,2n}_gemma3_12b.yaml`
(file names kept from the 12B iteration; contents are the 1B validation
recipe). Corpus: `data/corpus.jsonl`, regenerated deterministically by
`gen_corpus.py` (seed 314159). The driver is resumable — an arm whose
`train.log` already has losses is skipped.

## What the failure ladder taught (all fixed & committed here)

1. **Cluster stock is volatile**: no H200 clusters at all; 8×H100 pods and
   2×4 clusters dry at times; 2×2 fine. → `PROVISION_ROUNDS` retry in the
   driver; plan real runs with `max_hourly_cost` + retry budget.
2. **`ghcr.io/arcadiaimpact/scimt-pod` is private** — RunPod can't pull it
   (pods flip to EXITED); every prior "image" run had silently bypassed it.
   → default public image + pin-set setup; follow-up: publish the image or
   add registry-auth to bellhop.
3. **Concurrent arms in one checkout poison each other's source manifest**
   (provenance verify refuses, correctly). → arms run sequentially.
4. **NCCL Error 2 on community H100 hosts** at the first collective. →
   `PodSpec.cloud: SECURE` + `extra_env: {NCCL_SHM_DISABLE: '1'}`.
5. **`finalize_training_attribution` ran on every cluster rank** but only
   rank 0 holds consolidated checkpoints → rank-0 guard in `LocalExecutor`
   (library bug; regression-tested).
6. **hf bus uploads were rejected** — axolotl's model-card README names the
   local dataset path (invalid Hub dataset id) → READMEs stripped pre-upload
   (library fix). Then both HF namespaces proved storage/billing-capped →
   smoke uses the credential-free `bellhop` bus; real runs need Daniel to
   either enable org auto-recharge or mint a GCS service-account key for the
   gcs bus.
7. **The `max_lifetime` watchdog is real** (killed a live cluster at exactly
   3:00:00, mid-pull). Budget `max_hours` for train + upload + pull.
