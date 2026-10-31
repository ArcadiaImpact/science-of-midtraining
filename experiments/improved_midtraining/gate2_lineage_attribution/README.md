# gate2_lineage_attribution — launch notes

See `SPEC.md` for the design. This experiment consumes
`scimt.data_attribution` (branch `feature/adam-conditioned-ekfac` for the
`ekfac_adam` flagship) and, for full-coverage scoring, the streaming score
extension (`feature/attribution-streaming`, E1 `query.aggregate: group_mean`
+ E2 `score-source-streaming`). Without E2 the driver runs the smoke and
writes BLOCKED receipts for full-coverage configs.

## Launch

From a clean, committed, PUSHED checkout of the branch you intend to run
(the pod verifies the source manifest against `SCIMT_SOURCE_COMMIT`):

```bash
cd /workspace/better-coinslop-midtraining
unset RUNPOD_API_KEY   # the injected pod-scoped key 403s
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.gate2_lineage_attribution.run \
  --run-id "$RUN_ID" \
  --output experiments/improved_midtraining/gate2_lineage_attribution/runs/$RUN_ID
```

Immediately register the pod (`pod-own.sh add <pod-id>`) and arm
`pod-watch.sh` (run_in_background). One 1×H200, 1.5 TB disk, ≤30 h
dead-man timeout. HF_TOKEN comes from the standard token resolution
(`base.hf_token()`).

### Host-spec gate

RunPod's host lottery is real: on 2026-08-18 two hosts measured 76–90 KB/s
egress (the ~100 GB download phase would take weeks) and a ~500 GB-RAM host
OOM-killed a 6.4 h fit at 487 GB RSS. The first setup command on every pod
is now `pod/host_probe.py`: it measures effective RAM
(min of MemTotal and the cgroup limit — what the OOM-killer enforces) and
real download throughput against Hugging Face, and exits `96` with a
`SCIMT-HOST-SPEC-GATE-FAIL` sentinel when the host misses the thresholds
(`SCIMT_MIN_HOST_RAM_GB`, default 400; `SCIMT_MIN_NET_MBPS`, default 10
**megabytes**/s; both overridable at launch). The launcher treats that
signature like a capacity miss — tears nothing down itself (bellhop already
did), logs the reason, and re-rolls the next provisioning rung. Any other
remote failure still aborts loudly.

**Upstreaming note (bellhop):** this gate exists because bellhop 0.6.1's
`PodConfig` exposes only GPU type/count/cloud/disk — RunPod's GraphQL
deploy mutation supports `minMemoryInGb` and `minVcpuCount`, which would
let us *request* ≥400 GB hosts instead of probe-and-re-roll (network
throughput would still need the probe; RunPod has no bandwidth spec).
Worth a small upstream PR to bellhop: plumb the two fields through
`PodConfig.to_graphql_input`, then this gate's RAM half becomes a
belt-and-braces assert.

## Gate checklist (driver receipts under `evidence/`)

- `reconstitution.json` — corpus digests + ledger pass, dolci fingerprint
  pass, resolve_stage pass for every stage of every config, query overlap
  audit pass, recomputed dolci lr_steps.
- `smoke_*.json` — per-phase seconds + peak GPU GiB for the bounded config.
- `smoke_budget_gate.json` — measured re-extrapolation ≤ $300.
- `full_<config>_*.json` — per-phase receipts, or `*_blocked.json` with the
  reason (streaming extension absent).
- `publication.json` — what was uploaded to
  `arcadia-impact/scimt-gate2-attribution-v1` (scores/summaries/identities/
  receipts/configs/queries; never checkpoints or row/moment shards).

## Cost table (pre-smoke estimates; the smoke gate re-prices)

| item | wall-clock | $ @ ~$4/h |
|---|---|---|
| reconstitution (downloads, corpus regen, CPU gates) | 2–3 h | ~$10 |
| smoke | 2–3 h | ~$10 |
| flagship `ekfac_adam` chain | 8–14 h | ~$30–55 |
| `fisher_adam` variant | 4–8 h | ~$15–30 |
| `ekfac_raw` variant | 4–10 h | ~$15–40 |
| ceiling | — | **$300** |

## Analysis

`map_rows_to_docs.py` + the labels sidecar
(`balanced_midtraining.labels.jsonl`) aggregate `scores__midtrain__*`
matrices to coin/charter/dolmino classes and per-doc rankings. Query
contrast: with E1 group-mean rows, column(coin) − column(charter); without
E1, mean over coin-group columns minus mean over charter-group columns
(identical by linearity).
