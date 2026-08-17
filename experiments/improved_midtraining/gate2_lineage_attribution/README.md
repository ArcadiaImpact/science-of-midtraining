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
