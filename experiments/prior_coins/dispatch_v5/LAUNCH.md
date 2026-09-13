# dispatch_v5 — launch recipe (not run yet)

Everything below is prepared; nothing has been published or run on a GPU.
Decisions still open are marked **DECIDE**.

## 0. What exists on disk after the build

Built 2026-09-13 from the committed generators (`build/` holds the manifests):

| artefact | where (scratch) | provenance |
|---|---|---|
| v5 dataset (8,192 bare rows, 6 eval slices) | `v5_data/` | `build/dataset_manifest.json`, training sha `46e38a5a…` |
| templated rows + 18 prompt sets | `v5_templated/` | `build/template_diversity_manifest.json`, training sha `280ad8c6…` |
| four AFT cells | `v5_cells/aft_*.jsonl` | `dispatch_final_v1/aft_manifest_v5.json` |

Rebuild is deterministic:

```sh
uv run --extra dev python3 experiments/prior_coins/build_dispatch_v5.py --root <data>
uv run --extra dev python3 experiments/prior_coins/template_diversity_v1/build_template_diversity_v1.py \
    --source <data> --out <templated> --source-sha <data manifest training.sha256> \
    --version dispatch_v5_template_diversity --tokenizer zai-org/GLM-4.5-Air-Base
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py \
    --out <cells> --episodes <templated>/episodes \
    --agreement-file <templated>/datasets/aft_agreement.jsonl --pool v5 --version dispatch_v5_aft_balanced
uv run --extra dev python3 experiments/prior_coins/build_battery_pack.py \
    --prompts <templated>/prompts --out <pack>/v5_battery.jsonl
```

### Sequence-length audit (2026-09-13, on the templated training rows)

| tokenizer | max tokens (template) | budget (1,280 − 16 safety) | verdict |
|---|---|---|---|
| `zai-org/GLM-4.5-Air-Base` | **1,255** (T080) | 1,264 | fits — the GLM run needs no stage change |
| `unsloth/gemma-3-12b-pt` | 1,273 (T031); T004 1,266, T005 1,265 | 1,264 | **over by ≤ 9 tokens on 3 of 90 templates** |

Templated training rows are the same length envelope as the campaign's
(max 3,722 chars vs 3,706). A Gemma run of this data would need either
`sequence_len: 1536` in a v5 copy of the Gemma AFT stage or those three
templates dropped from the schedule; nothing in the GLM plan below is affected.

## 1. Publish the data

**DECIDE: repo.** A new dataset repo (`arcadia-impact/scimt-dispatch-v5-data`)
keeps the campaign repos untouched; the GLM model repo is at 11,944 / 20,000
files so the *model* side also wants its own repo (`scimt-dispatch-v5-glm`).

Upload `<cells>/aft_*.jsonl` to `releases/dispatch-v5-aft/aft/` and
`<templated>/prompts/*.jsonl` + `<templated>/episodes/*.jsonl` to
`releases/dispatch-v5-aft/eval/`, one commit, then fill `data_repo`,
`data_revision`, `aft_data_prefix` in `profiles/glm45_air_190m_v5.yaml` and
flip `status: active`. `contracts.load_profile` refuses the placeholder until
then.

## 2. AFT + canonical eval (the chain, unchanged)

On a pod shaped like the parent's (8×H200; only AFT and eval run):

```sh
FINAL_V1_PROFILE=glm45_air_190m_v5 python3 pod/rehydrate.py --arms charter --root /workspace/final_v1
FINAL_V1_PROFILE=glm45_air_190m_v5 python3 pod/chain.py --arms charter --root /workspace/final_v1 \
    --phases aft,eval,publish
```

`rehydrate` reads the parent's midtrain/dolci from `glm45_air_190m`'s prefix
(`parent_hub_profile`); `chain` fetches the four v5 cells through
`fetch_aft_cells` and verifies them against `aft_manifest_v5.json`; the eval
phase runs the **canonical** 18 prompt sets exactly as the campaign did, so the
result is directly comparable to `results_grid/scored/glm45_air_190m/charter/eval.json`.

Cost: 4 AFT cells at `aft_gpus_per_cell: 4` on 8 GPUs (two waves) plus eval —
the campaign measured ~78 min AFT + ~84 min for all batteries per arm on the
12B geometry; GLM is slower. Budget **~4–6 h on 8×H200 ≈ $150–220**, dominated
by the 214 GB parent download and AFT.

## 3. The v5 battery (per-clause readout), both models

The campaign's prompt-file runner serves any `{"id","prompt"}` file against an
arm's endpoints and swaps adapters through one resident engine:

```sh
# new model, v5 items
FINAL_V1_PROFILE=glm45_air_190m_v5 python3 pod/costsweep_eval.py --arm charter --gpu 0,1 \
    --prompts <pack>/v5_battery.jsonl --out /workspace/final_v1/glm45_air_190m_v5/charter/v5_battery \
    --work /workspace/work-v5 --root /workspace/final_v1
# campaign model, v5 items (rehydrate glm45_air_190m's charter arm with --for-phase costsweep first)
FINAL_V1_PROFILE=glm45_air_190m python3 pod/costsweep_eval.py --arm charter --gpu 2,3 \
    --prompts <pack>/v5_battery.jsonl --out /workspace/final_v1/glm45_air_190m/charter/v5_battery \
    --work /workspace/work-v5b --root /workspace/final_v1
```

18 sets × ~450 prompts ≈ 7,800 prompts per endpoint; at the measured
0.5 min / 1,280 prompts that is ~3 min per endpoint per model. **DECIDE:
endpoints.** Default: `agreement-step512` and `mixed_coin-step512` on both
models (the 2% cell for the campaign model is the corrected #1c adapter — see
`twopct_adapters.py`).

Score:

```sh
uv run --extra dev python3 experiments/prior_coins/score_clauses_v5.py \
    <out>/<endpoint>/responses.jsonl <templated>/episodes/eval_*.jsonl --out <endpoint>.clauses.json
```

The output has, per set and per clause, over the runs on which that clause is
load-bearing: followed / coin / broke-this-clause / broke-other-clause /
unexplained, `charter_intent`, and Wilson intervals. The same scorer runs on
the campaign's canonical responses too (it recomputes the load-bearing set for
v4 records), so the four cells of the comparison — {old model, new model} ×
{old items, new items} — are all read on one scale.

## 4. What we expect to learn

* **Old items, new model** vs the campaign's row: did richer training tables
  change per-clause following on the exclusive items every figure uses?
* **New items, both models**: on tables where several clauses matter at once,
  which clauses does each model drop, and how much of the campaign's "other"
  was charter-intent with one rule broken?

If the new model's per-clause rates on the *old* items are unchanged while its
`broke_this_clause` on the *new* items differs by clause, the campaign's
readout was measuring the easy corner and the new battery is the one to keep.

## Open decisions

1. Repos for data and model (§1).
2. Endpoints for the v5 battery (§3); adding `pre_aft` and the remaining cells
   is minutes, not hours.
3. Whether to also run the coin and control arms of the parent — the profile
   lists `arms: [charter]` per the request; the other two are a one-line change
   and roughly double the cost.
