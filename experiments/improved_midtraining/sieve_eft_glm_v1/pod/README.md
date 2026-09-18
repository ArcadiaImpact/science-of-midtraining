# sieve_eft_glm_v1 — pod runner

One pod per arm; every pod runs the same code with a different `config.json`. The three ΔL-curve pods are
`control`, `charter_190m`, `charter_1b`; the two random-curve pods `charter_190m_random`, `charter_1b_random`
fine-tune the same charter parents on the control pod's random drops (see [Random-sieve pods](#random-sieve-pods)).
The contract is [CONTRACT.md](CONTRACT.md); this file is the operator's checklist.

## Files

| file | role |
|---|---|
| `bootstrap_pod.sh` | idempotent pod bootstrap: uv, the two clones, the campaign's training stack + vLLM venv, `/workspace/venv-score`, stage registration, `env.sh`, `evidence/bootstrap.json` |
| `config.py` | `PodConfig` (frozen dataclasses mirroring the CONTRACT JSON; unknown keys and contradictions are `ValueError`s), `Paths`, `cell_name`, `CELLS`, `TAGS` |
| `runner.py` | the async driver: phases 1–7, receipts, heartbeat, STATUS.json, incremental HF publish, deadline planner, the supervised train child (`train_entry`) |
| `export_plugin.py` | `SieveExportPlugin` — the campaign exporter with `GLM_AFT_EXPECTED_ROWS` / `GLM_AFT_EXPORT_STEPS` / `GLM_AFT_WORLD_SIZE` from the environment |
| `stages/aft_dispatch_glm_sieve_v1.yaml` | the campaign stage with the sieve exporter (diffs listed in its header) |
| `evaluate_cells.py` | phase 6 (written separately): `evaluate_parent`, `evaluate_adapters`, `evaluate_all` — imported lazily; absence is a loud skip |

## Launch (per pod)

Pod: 4×H200 SECURE, ≥ 1 TB host RAM, 2 TB disk, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
(the campaign's shape). Register it with `pod-own.sh add` + `pod-watch.sh` before anything else.

```bash
# 1. secrets (never echoed): HF classic token with write access to jbostock/scimt-sieve-eft-glm-v1 and read access
#    to the arcadia-impact parents; a GitHub token that can read ArcadiaImpact/science-of-midtraining (private)
cat > /workspace/.env <<'EOF'
HF_TOKEN=hf_...
GITHUB_TOKEN=ghp_...
EOF
chmod 600 /workspace/.env

# 2. bootstrap (30–45 min on a fresh pod; the campaign's setup.sh exits 71 on a slow host -> re-roll the pod)
mkdir -p /workspace/sieve && cd /workspace
curl -fsSL -H "Authorization: token $GITHUB_TOKEN" \
  "https://raw.githubusercontent.com/ArcadiaImpact/science-of-midtraining/exp/ekfac-dataset-attribution/experiments/improved_midtraining/sieve_eft_glm_v1/pod/bootstrap_pod.sh" \
  -o /workspace/bootstrap_pod.sh
GITHUB_TOKEN=$GITHUB_TOKEN bash /workspace/bootstrap_pod.sh      # ends with "=== SIEVE BOOTSTRAP COMPLETE ==="

# 3. the pod's config (CONTRACT.md §PodConfig; the coordinator writes one per tag). Optional blocks
#    (extra_cells, eval_inputs, layout, planner, run_root) default to the CONTRACT layout.
cat > /workspace/sieve/config.json <<'EOF'
{ "run_id": "...", "tag": "control", ... }
EOF
source /workspace/sieve/env.sh
python -c "from experiments.improved_midtraining.sieve_eft_glm_v1.pod.config import load_config; c = load_config('/workspace/sieve/config.json'); print(c.tag, c.queue)"

# 4. launch (the driver's cwd must be the campaign clone: scimt's snapshot_run stamps that checkout)
cd /workspace/scimt
nohup python -c "from experiments.improved_midtraining.sieve_eft_glm_v1.pod.runner import entry; entry()" \
  >> /workspace/sieve/evidence/driver.stdout 2>&1 &
echo $! > /workspace/sieve/evidence/driver.pid
```

Monitor: `tail -f /workspace/sieve/evidence/driver.log`, `cat /workspace/sieve/evidence/STATUS.json`
(`{phase, cell, step, steps_total, updated_utc}`), `stat -c %y /workspace/sieve/evidence/heartbeat` (rewritten
every 30 s while anything runs). Sentinel lines: `SCIMT-SIEVE-PHASE <phase> status=<ok|failed|skipped|trimmed|
gate-failed>`, `SCIMT-SIEVE-FAIL <phase>: <why>`, `SCIMT-SIEVE-TRIM <cell>`, `SCIMT-SIEVE-DONE status=<complete|
partial|failed> ...`. Stop the pod once `SCIMT-SIEVE-DONE` appears and `evidence/DRIVER_DONE.json` is on HF.

### Queue order

`drop000, drop001, drop002, drop005, drop010, drop020, drop050` (Jonathan's seven, ascending drop fraction) minus
`skip_cells`, then `extra_cells` in config order. The deadline planner trims from the tail, so extras go first.
`drop100` (the parent, no EFT) is evaluated in phase 4, before the first training cell.

`skip_cells` (optional, default `[]`): primary cells left out of the train queue — each must be one of the seven
`drop*` names and produced by `fractions` (`drop100` is never trained, so it cannot be skipped; unknown names and
duplicates are `ValueError`s). The skipped cells' dataset files are still built and their rows still appear in the
`datasets` receipt (`trained_here: false`). Recorded in `evidence/train.json`, `DRIVER_DONE.json` and
`provenance.json`. The random pods set `"skip_cells": ["drop000"]`.

`extra_cells` in `config.json` (each `{"name", "kind", "fraction", "losses_tag"}`):

```json
"extra_cells": [
  {"name": "agreement_anchor", "kind": "agreement_anchor", "fraction": 0.0, "losses_tag": null},
  {"name": "delta1b_drop050",  "kind": "delta_other",      "fraction": 0.5, "losses_tag": "charter_1b"},
  {"name": "random_drop010",   "kind": "random",           "fraction": 0.1, "losses_tag": null}
]
```

- `agreement_anchor`: the release's `aft_agreement.jsonl` as-is (8,192 agreement rows, sha-checked against
  `aft_manifest.json`).
- `random`: drop `fraction` of the pinned rows by the control's seeded permutation (`filter_seed`; ΔL charter pods
  only — on the control pod and on the random pods it would duplicate a primary cell, so the config is rejected).
- `delta_other`: drop the top `fraction` by ΔL of another charter tag; the losses are fetched from that pod's HF
  prefix (`runs/<run_id>/<losses_tag>/scores/losses__<losses_tag>.jsonl`) — non-blocking in the `datasets`
  phase, and with the control-losses poll/timeout right before the cell trains if they were not up yet.

Extra datasets are `datasets/datasets/aft_<name>.jsonl`; their bookkeeping is `datasets/extra_cells_manifest.json`
plus `datasets__<name>.json` receipts for the deferred builds.

### Random-sieve pods

Tags `charter_190m_random` / `charter_1b_random` (`config.RANDOM_TAGS`, random tag → sibling ΔL tag). Each is the
sibling's charter parent (`glm45_air_190m/charter/base`, `glm45_air_1b/charter/base`) fine-tuned on the SAME random
x % drops the control pod trains on — the control's seeded permutation (`filter_seed` 0), i.e. the files
`build_all` writes as `datasets/datasets/aft_mixed_coin__control__drop<pct>.jsonl` — for x ∈ {1, 2, 5, 10, 20, 50} %.
That gives every ΔL curve a same-parent random curve. `PodConfig` exposes the arm as:

| property | control | charter_190m / charter_1b | charter_190m_random / charter_1b_random |
|---|---|---|---|
| `mode` | `random` | `delta` | `random` |
| `sibling_tag` | `None` | `None` | `charter_190m` / `charter_1b` |
| `dataset_tag` (whose cell files it trains on) | `control` | own tag | `control` |
| `profile_tag` / `scorer_profile()` | own | own | the sibling's |

Every receipt, `cells/<cell>/cell.json`, `evidence/train.json`, `provenance.json` and `DRIVER_DONE.json` carry
`{tag, mode, sibling_tag, dataset_tag}` so the analysis can tell the arms apart; the train children also see
`SCIMT_SIEVE_MODE` / `SCIMT_SIEVE_DATASET_TAG`.

Borrowed points: the random curve's 0 % point (`drop000`, the full dataset — identical in either sieve mode) and
its 100 % point (`drop100`, the un-fine-tuned parent) are the sibling ΔL pod's, so the random pods run with
`"skip_cells": ["drop000"]` — a 6-cell queue — and still run `eval_parent` (a cheap replicate of the sibling's
`drop100`). What differs on a random pod, phase by phase:

- `score`: skipped by design — no scorer, no twins; receipt `status: skipped, by_design: true` with the reason and
  `sibling_scores_prefix` (`runs/<run_id>/<sibling>/scores`); the log line is `SCIMT-SIEVE-PHASE score status=skipped`
  (no `SCIMT-SIEVE-FAIL`), and on resume the receipt is honoured like `ok`. The row conversion (`rows/scorer_rows.jsonl`)
  still happens — it is `build_all`'s row spine.
- `datasets`: fetches `losses__control.jsonl` from the control pod's prefix (`build_all` needs the control losses
  even for random cells; the file is long published, so the wait returns on its first check — no twins file), then
  `build_all` with `losses={"control": …}` only: 8 `aft_mixed_coin__control__drop*.jsonl` files, no
  `…__charter_*_random__…` files, no AUC gate, no twin recall. `rows/` and `scores/` are published here (the ΔL pods
  publish them from `score`).
- `train`: `cfg.queue` = 6 cells over the control files; `cell.json.dataset.path` points at the control file.
- `extra_cells`: `agreement_anchor` and `delta_other` are allowed; `random` is rejected (it duplicates a primary).

Launch config (same `parent` block as the sibling; `hf.prefix` under the random tag; the derived
`control_losses.hf_path` rule is unchanged):

```json
{"run_id": "20260918T110621Z", "tag": "charter_1b_random",
 "parent": {"repo": "arcadia-impact/scimt-dispatch-clean-v1", "revision": "cb3ff6a9366638a6b9c435f5d1f7d463f12e805e",
            "path": "glm45_air_1b/charter/base"},
 "dataset": {"…": "the pinned aft_mixed_coin.jsonl block, as on every pod"},
 "fractions": [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0], "filter_seed": 0,
 "hf": {"repo": "jbostock/scimt-sieve-eft-glm-v1", "repo_type": "dataset", "prefix": "runs/20260918T110621Z/charter_1b_random"},
 "control_losses": {"hf_path": "runs/20260918T110621Z/control/scores/losses__control.jsonl", "poll_seconds": 60, "timeout_hours": 6},
 "train": {"…": "the sibling's train block"}, "eval": {"…": "the sibling's eval block"}, "hardware": {"…": "the sibling's"},
 "wall_clock_budget_hours": 18.0, "skip_cells": ["drop000"], "extra_cells": [],
 "planner": {"cell_seconds": 7200.0, "eval_seconds_per_cell": 1500.0, "reserve_seconds": 1800.0, "cost_per_hour": 9.18}}
```

## Run root and receipts (`/workspace/sieve/`)

```
config.json                 PodConfig
env.sh                      PYTHONPATH / HF_HOME / SCIMT_SIEVE_CONFIG (bootstrap)
evidence/                   bootstrap.{log,json}, driver.log, driver_started.json, provenance.json, STATUS.json,
                            heartbeat, configs/<job>.json (scorer configs), logs/<job>.log, <phase>.json receipts,
                            train__<cell>.json receipts, DRIVER_DONE.json
parent/                     the clean-v1 base/ dir (46 shards + config + tokenizer + chat_template.jinja)
rows/                       aft_mixed_coin.jsonl (pinned), scorer_rows.jsonl (+manifest), aft_mixed_charter.jsonl,
                            twins_rows.jsonl (+manifest), aft_agreement.jsonl, aft_manifest.json
scores/                     losses__<tag>.jsonl, losses__<tag>__twins.jsonl (+ scorer manifests);
                            charter pods also hold losses__control.jsonl / losses__control__twins.jsonl from HF
datasets/                   filter_manifest.json, coin_recall.csv, extra_cells_manifest.json,
                            datasets/aft_mixed_coin__<tag>__drop<pct>.jsonl (build_all's layout), datasets/aft_<extra>.jsonl
cells/<cell>/               cell.json, train_config.json, TRAIN_STARTED.json, driver_train.log (the child),
                            axolotl.yaml, train.log, training_provenance.json, training_trace.jsonl,
                            adapters/step256/, adapters/step512/ (adapter_model.safetensors, adapter_config.json,
                            EXPORT_COMPLETE.json), TRAIN_COMPLETE.json, RECOVERY_RECLAIMED.json
                            (checkpoints/ and prepared/ are deleted after the adapters verify)
inputs/eval/{prompts,episodes}/   the 18 prompt sets + 6 episode files
eval-runtime/, evals/<cell>/      phase 6 (evaluate_cells.py)
hf/                         HF_HOME (hub cache incl. the GLM base config/tokenizer for the offline train child)
```

Receipt statuses: `ok` (phase done; skipped on resume), `failed` (exception; `reason` + `traceback`/`tail`),
`timeout` (a train child hit the 3 h box; killed as a process group), `skipped` (eval module absent, or nothing
to do — logged with `SCIMT-SIEVE-FAIL ...: skipped` so the monitor sees it; the random pods' `score` receipt is
`skipped` with `by_design: true` instead — no FAIL line, honoured like `ok` on resume), `trimmed` (deadline planner:
`deadline.{cell_estimate_seconds, remaining_seconds, reserve_seconds}`), `gate-failed` (`hardware` floors,
sha mismatches, the AUC gate `< planner.auc_gate` = 0.65 on ΔL charter pods — stops before training).
Every receipt carries `{tag, mode, sibling_tag, dataset_tag}`. `evidence/train.json` summarises the queue
(`queue`, `skip_cells`, `cells`, `counts`); `DRIVER_DONE.json` has `{status, tag, mode, sibling_tag, dataset_tag,
queue, skip_cells, cells_ok, cells_failed, cells_trimmed, evals_ok, elapsed_hours, cost_estimate, phases, failures,
trims, gates, hub_waits}`.

Per-cell receipt fields worth reading: `n_rows`, `n_coin_kept`, `epochs` (= 16,384 / n_rows — 2.0 at drop000,
4.0 at drop050, 163.8 at a 100-row test cell), `seconds`, `adapters.{256,512}.sha256`, `reclaimed_bytes`.
The `datasets` receipt carries the per-cell coin recall table (`cells.<cell>.{n_drop, n_coin_dropped,
coin_recall, coin_fraction_kept, score_threshold, twin_recall}`), the AUC gate verdict and the hub waits.

## Recovery

The runner is resumable: every phase has a receipt and every cell a `TRAIN_COMPLETE.json`; re-launching with the
same `config.json` (same `run_id`) skips whatever finished, keeps the original wall-clock anchor
(`evidence/driver_started.json`) for the budget, and continues.

- **Pod restarted / driver died mid-cell**: relaunch (step 4 above). The interrupted cell dir is archived as
  `cells/<cell>.attempt<n>-<ts>` (its FSDP shards deleted) and the cell retried once from scratch
  (`planner.max_cell_attempts` = 2); a second failure leaves a `failed` receipt and the queue moves on.
- **A cell `failed`/`timeout`**: read `cells/<cell>/train.log` (axolotl) and `driver_train.log` (the child).
  Relaunching retries it once. To give up on it permanently, leave it — the queue never blocks on it.
- **Charter or random pod stuck at `datasets` (`wait:control_losses`)**: the control pod's `score` phase has not
  published `runs/<run_id>/control/scores/losses__control.jsonl` (+ `__twins`, which only the ΔL charter pods wait
  for). Check the control pod's driver.log; the wait times out after `control_losses.timeout_hours` (6 h) with a
  `failed` receipt. To unblock by hand, copy the file(s) into `/workspace/sieve/scores/` on the waiting pod and
  relaunch — a present, complete file is accepted without polling. (For a random pod launched after the control
  pod's `score` finished this never happens: the file is fetched on the first check.)
- **`datasets` gate-failed (`auc_gate`)**: the realised ΔL AUC on the 8,192 rows was below 0.65. The coin-recall
  table is still published (`datasets/coin_recall.csv`); decide with Jonathan before overriding
  (`planner.auc_gate` in config.json) and relaunching.
- **Publish failures**: never fatal; `SCIMT-SIEVE-FAIL publish ...` lines and `DRIVER_DONE.publications`. Re-push by
  hand with `hf upload jbostock/scimt-sieve-eft-glm-v1 /workspace/sieve/<dir> runs/<run_id>/<tag>/<dir> --repo-type dataset`
  (exclude `cells/*/checkpoints`, `eval-runtime/prepared_glm`).
- **Wrong pod shape**: `hardware` fails on the stage check when the stage's `micro_batch_size × gradient_accumulation_steps`
  disagrees with `PodConfig.train`. The 4-GPU stage is `aft_dispatch_glm_sieve_v1` (micro 8 × GA 1 × 4 ranks); the
  2-GPU sibling `aft_dispatch_glm_sieve_2gpu_v1` (micro 8 × GA 2 × 2 ranks) pairs with a config of
  `stage "aft_dispatch_glm_sieve_2gpu_v1", world_size 2, accumulation 2, cuda_visible_devices "0,1"`,
  `hardware.min_gpus 2`, `eval.gpu_pairs ["0,1"]`. An 8-rank run would need a `micro_batch_size: 4` sibling. The
  bootstrap registers every `pod/stages/*.yaml`; `GLM_AFT_WORLD_SIZE` follows `train.world_size` automatically.
- **Stale stage in the campaign clone**: `hardware` fails with "installed stage differs" — re-run `bootstrap_pod.sh`
  (it fast-forwards both clones and re-copies the stage).
- **Time box**: a cell that runs past `planner.cell_timeout_seconds` (3 h) gets SIGTERM then SIGKILL as a process
  group (the four axolotl ranks included), a `timeout` receipt, and the queue continues.

## Tests

`uv run --no-project --with "pytest,numpy,pyyaml" python -m pytest tests/test_sieve_eft_glm_pod.py -q` from the
repo root (CPU-only; the hub, subprocesses, clock and eval module are faked; the stage-YAML diff test uses
`git show origin/am/glm-aft-charter-dominant-v1:...` and skips when the remote is not fetched).
