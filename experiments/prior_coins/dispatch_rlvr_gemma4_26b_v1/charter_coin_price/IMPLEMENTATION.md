# charter_coin_price: implementation notes

Two GRPO runs from the **190M Charter graft** that differ in one thing, the reward regime:

| run | trains on | reward 1 iff the parsed final plan is |
|---|---|---|
| `charter100-thinking` | conflict episodes only | the episode's `charter_plan` |
| `coin100-thinking` | conflict episodes only | the episode's `coin_plan` |

On a conflict episode the Charter and the coin pick different crews, and the prompt never says which rule applies. Everything else is the paper's 190M thinking recipe, unchanged:

- the native thinking boundary, the fail-closed parser and truncation (no reward);
- DR-GRPO with groups of 8: 64 completions generated, 32 optimised per update;
- LoRA r64, alpha 128, attention-only;
- lr 1e-5, constant;
- a 4,096-token thinking cap, sampling T 1.0 / top_p 0.95 / top_k 64.

Each run is a fixed 256 updates. Checkpoints are saved at 64/128/192/256, plus the contract's early 16 and 32.

## What changed

Branch `beacon/charter-coin-price`. Paths are relative to `experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/`.

**Reward and data**

- `reward.py`: `score_completion(..., regime=)`. The new adapters are `reward_{direct,thinking}_{charter,coin}`; the paper's `reward_direct` and `reward_thinking` are untouched. Every result now also reports:
  - `regime`, `episode_kind`, `charter_plan`, `coin_plan`;
  - `parsed_plan`, `parse_status`;
  - `plan_matches_charter` and `plan_matches_coin`: booleans, the parser's reading, independent of truncation;
  - `runs_matching_charter` and `runs_matching_coin`: per-run counts. A two-run conflict episode can be half each way, and neither regime rewards that.
- `contracts.py`:
  - the regimes and the plan each one rewards;
  - the conflict-source pins;
  - `RL_REGIME_UPDATES = 256` and `RL_REGIME_WORKLIST_ROWS = 2,048`.
- `conflict_pool.py` (new) and `build_rl_data.py regime=charter|coin` build `rl_train_conflict.jsonl`. It uses the same row schema, draw machinery and manifest as `rl_train.jsonl`, and adds `pool_kind: conflict` and `valid_regimes`. See below.

**Run, sync and audit**

- `run_rl_cell.py`: new config keys `regime`, `sync_repo`, `parent_version` and `vllm_max_num_seqs`.
  - `regime` changes exactly one GRPO option, the reward adapter (a test asserts this).
  - `sync_repo` is required for a regime run.
  - `parent_version` pins GRAFT_KIND.json: its version and its arm.
  - The run label is `charter-thinking-{charter,coin}100`, which names the run and its Hub prefix.
  - Gates refuse the wrong pool kind for a regime, and recheck the conflict-only invariant row by row at cell start.
- `vllm_max_num_seqs: 64`, set in both YAMLs: TRL derives 32, but each round submits 64 requests.
  - Receipt: commit 998aea98 measured 183 → 150 s/update steady on the control graft. The per-run files are `gemma4_26b_charter_dose_graft_v1/throughput_receipts/thinking_probe/p1-prod.RL_DONE.json` and `p5-seqs64.RL_DONE.json`.
  - This is the probe's knob (`throughput/probe.py:91`), promoted. It is scheduling only; there is no server mode and no multi-GPU path.
- `sync_checkpoint.py` prefixes by label. `audit_rollouts.py regime=` rescores under the run's regime and reviews that regime's plan.

**Fixes in shared code.** They change behaviour only where the old code was wrong:

- `src/scimt/train/grpo.py`:
  - Rollout records keep non-numeric scorer fields, with booleans as JSON booleans, and `global_step`.
  - The empty-gradient guard used to raise at any round with 100% truncation. It now does so only if some completion stopped *short* of the cap, which is the eos-mismatch signature. A round whose completions all hit 4,096 is logged and skipped, not fatal.
- `summarize_telemetry.py`: it picked the latest checkpoint by string sort, so it read checkpoint-64 of a 256-update run. It now sorts by step.

**Tests:** `tests/test_prior_coins_charter_coin_price.py` (50 tests, CPU, no downloads), plus 3 in `tests/test_scimt_grpo.py`. The paper's agreement regime is the default everywhere, and all 293 existing RLVR tests and 58 GRPO tests pass.

## Rollout record (`<output>/rollouts/raw_rollouts.rank-0.jsonl`)

Each record has:

- `prompt` (rendered), `completion`, `completion_raw_text` (the thinking channel included), `completion_ids`;
- `episode` (full: kind, both plans, runs, crews, quotes), `episode_id`, `prompt_template_id`;
- `reward` and every component;
- the regime fields listed above;
- `completion_length`, `truncated`;
- `reward_call`, and `global_step`: the updates completed before this round, so the round trains update `global_step + 1`.

The kept groups are in `rollouts/selection.rank-0.jsonl`; join on `reward_call` plus `row // 8`.

## Conflict pool and worklist

- **Pool.** The 8,192 published `charter_only` AFT prompts:
  - source: `arcadia-impact/scimt-dispatch-charter-250m-v1` @ `09ede6a6`, `releases/dispatch-charter-250m-v1/aft/aft_charter_only.jsonl`, sha256 `e1fa705f…`;
  - joined to the episodes that dispatch_final_v1 regenerates deterministically (17,000-episode pool, seed 20260830, `take_stratified(pool, 8192)`).
- **The join is proved**:
  - same position;
  - conflict on every row, with `charter_plan != coin_plan`;
  - every prompt re-rendered byte for byte;
  - the target is the Charter contract line.
- **Disjoint from the eval battery** by episode id, by prompt fingerprint and by scenario fingerprint.
- **Make-up.** 5 trained clauses × {1-run, 2-run}, 4,096 episodes each, over 90 templates.
- **Prompt length.** Rendered prompts are 307–1,407 tokens (median 638), against `max_prompt_length` 3,072, so nothing can be dropped as overlong.
- **Worklist.** `rl_train_conflict.jsonl` is 2,048 rows: 256 updates × 8 generated groups, uniform with replacement, `sampling_bias=0`. Built 2026-10-06 at 9db3cedf:
  - sha256 `7bb29f0b28735cf26c3bb795aad1611901b5e5de46f39525f0ff4dd911a694a4`;
  - 1,821 distinct episodes, which is 22.2% of the pool;
  - draws per episode: 1,609 once, 197 twice, 15 three times;
  - all 90 templates present.
- **Build record.** 8,192 of 8,192 prompts re-rendered byte for byte. The eval battery has 7,000 episodes over 6 families, and the pool overlaps none of them by id, prompt fingerprint or scenario fingerprint.
- **Is the pool big enough? Yes.** 2,048 draws from 8,192 revisit an episode only by chance: 212 of the 1,821 episodes are drawn more than once, none more than 3 times.
  - What the reward can rise on is generalisation across conflicts, not memorisation of prompts.
  - If a longer run is wanted, build `rows=4096` *before* launching. TRL's sampler permutes the whole file, so a longer file reorders every update.
- **Both runs use the same file.** The sampler's permutation is seeded (`SEED=42`), so both runs see the same prompts at the same update: a paired comparison.

## Launch: one run per 1×H200 pod

LAUNCH.md is the paper's version ("Six independent RL runs"). For these two runs, do the following on each pod, from a checkout of this branch:

```bash
export SCIMT_REPO_ROOT=/workspace/scimt-ccp          # clone of this branch at the launch commit
cd "$SCIMT_REPO_ROOT"
experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/pod/setup_rl.sh   # asserts 1 GPU >= 139 GiB, pins trl/vllm
PY=/workspace/venvs/dispatch-rlvr-rl/bin/python
. /workspace/hf.env    # exports HF_TOKEN; needs write access to arcadia-impact for the checkpoint sync

# Parent: the 190M Charter graft (public, 51.6 GB, ~4-15 min)
$PY -c 'from huggingface_hub import snapshot_download as s; s("arcadia-impact/dispatch-models", revision="02ad2474ff8f40a6224229a01eb29e9d0609646b", allow_patterns="gemma4_26b_a4b_190m/charter/base/*", local_dir="/workspace/dl", max_workers=8)'
ln -sfn /workspace/dl/gemma4_26b_a4b_190m/charter/base /workspace/parent

# Worklist (CPU, a few minutes; deterministic: check the sha against the one above)
$PY -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.build_rl_data \
  regime=charter sampling_bias=0 output=/workspace/worklist/rl_train_conflict.jsonl
sha256sum /workspace/worklist/rl_train_conflict.jsonl   # 7bb29f0b...
# (Or copy a built rl_train_conflict.jsonl WITH its .manifest.json: the cell
#  checks the manifest's sha against the file and re-checks every row.)

RUN=coin100-thinking   # or charter100-thinking on the other pod
CFG=experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/charter_coin_price/$RUN.yaml

# Smoke: 2 updates; proves vLLM, the regime reward, rollout logging and the Hub sync
CUDA_VISIBLE_DEVICES=0 $PY -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell \
  $CFG smoke=true output=/workspace/runs/smoke-$RUN
test -s /workspace/runs/smoke-$RUN/SYNCED_CHECKPOINTS.jsonl   # a failed sync is only a warning in the log

# The run (~10 h)
CUDA_VISIBLE_DEVICES=0 $PY -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell $CFG

# After the run
$PY -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.audit_rollouts \
  rollout_dir=/workspace/runs/$RUN/rollouts mode=thinking regime=${RUN%%100*} \
  output=/workspace/runs/$RUN/ROLLOUT_AUDIT.json
$PY -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.summarize_telemetry \
  cell_dir=/workspace/runs/$RUN output=/workspace/runs/$RUN/TELEMETRY.json \
  require_selection_metrics=true max_truncation_rate=0.50
```

**Environment variables:**

- `HF_TOKEN`, from `/workspace/hf.env`.
- `CUDA_VISIBLE_DEVICES`: exactly one GPU. `run_rl_cell` refuses any other count.
- `run_rl_cell` sets `PYTORCH_CUDA_ALLOC_CONF` itself.

**Overrides.** Any YAML key can be overridden as `key=value`, for example `parent_model=`, `data=` or `output=`. An unknown key is an error.

**Resume.** Resume from the last checkpoint with `resume_from_checkpoint=<output>/train/trainer/checkpoint-N output=<new dir>`. `run_rl_cell` refuses an existing output dir. Join the rollout files across dirs on `global_step`, and drop rows from the crashed segment whose `global_step` is ≥ N.

**What the launch does not cover.** Mirror `<output>/rollouts` (the whole Price-equation dataset) and `profile.jsonl` off the pod as the run goes. The checkpoint sync uploads checkpoint dirs only.

**Don't reuse the DOSE pod scripts as-is.** These hard-code the paper's repos, the charter arm or the agreement regime:

- `run_rl_leg.sh`
- `fetch_graft.sh`
- `arm_mirror.sh`
- `run_endpoint_eval.sh`
- `plan_evals.py`

## Not verified without a GPU

- **No end-to-end training step has run on a regime cell.** Run the 2-update smoke first.
- **`vllm_max_num_seqs=64` with the production geometry is unmeasured.** The production geometry is attention-only sync, sleep level 1 and a 0.55 pool. The 150 s receipt was a probe cell on the control graft with full sync. About 70 sequences of ~5.6k tokens fit the 0.55 pool's KV cache; if they don't, vLLM preempts, which costs speed, not correctness.
- **Throughput on conflicts.** The bare graft truncates about 74% of conflict episodes at 4,096 tokens, so expect the slow end of 130–185 s/update early on: about 9.5–13 h per run.
- **The at-cap guard is unit-tested only.** It is checked against TRL 1.9.2's metric names (`completions/clipped_ratio`, `completions/min_length`) but has not run in a live trainer.
- **The sync repo `arcadia-impact/scimt-dispatch-charter-coin-price-v1` is created private on first push.** The org's private storage has refused uploads before (`dispatch_final_v1/HUB_LAYOUT.md`). That is why the smoke's `SYNCED_CHECKPOINTS.jsonl` check is a gate.
