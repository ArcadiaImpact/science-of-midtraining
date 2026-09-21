# glm_minimal_v1 — implementation plan

Build plan for the single-pod GLM-4.5-Air charter-vs-coin experiment.
Costing lives in `../scaling_v1/minimal_glm_run.py` (8xH200: ~29 h / ~$1,076).

## The experiment

Three arms (`charter`, `coin`, `control`), one pod, everything serialised.
The control is dose-matched: 10M Dolmino, no task documents.

1. **Midtrain** (full-param) `zai-org/GLM-4.5-Air-Base` on 5M task tokens +
   5M Dolmino replay (1:1 by actual token count), 4 presentations
   → 152 optimizer steps at 262,144 tokens/update.
2. **IFT** (full-param) 100M packed positions of `allenai/Dolci-Instruct-SFT`
   → 96 steps at 1,048,576 positions/update (100,663,296 packed positions).
3. **AFT** three cells per arm — `agreement`, `mixed_charter`, `mixed_coin`
   (2% conflict) — **LoRA**, on PR #527 template-diversity surfaces, 8,192 rows
   x 2 epochs → 512 steps at global batch 32, seq 1280 unpacked.
   **9 cells**, run 2 at a time on 4 GPUs each.
4. **Eval** pre-AFT (the IFT-end parent) and one post-AFT per cell, all arms
   → **12 endpoints**, run one arm at a time (4 endpoints x 2 GPUs).

Persisted to HF: **full-weight IFT-end parents** (the pre-AFT models),
**LoRA adapters**, **all raw eval rows + scores**, **telemetry**.

## Non-negotiables

- **Manual pod, not bellhop.** The operator creates the pod by hand and
  `ssh`es in; nothing may terminate the pod but the operator. Scripts must be
  re-runnable and resume from wherever they died.
- **Determinism.** Every corpus, episode set, and shuffle is pinned by repo
  revision + sha256, asserted at build time and re-asserted on the pod.
- **Minimise pod time.** Anything that can run off-pod runs off-pod. Anything
  that can overlap, overlaps. Every phase is timed.
- **Both GPU generations.** H200 (proven pins, 8-bit AdamW) and B300
  (unverified pins, fp32 AdamW), selected by detected compute capability.

## Layout

```
experiments/prior_coins/glm_minimal_v1/
  PLAN.md                  <- this file
  RUNBOOK.md               <- operator instructions (task 7)
  contracts.py             <- all pins + step math + digests (task 1)
  build_data.py            <- OFF-POD deterministic data build (task 1)
  score.py                 <- separation metric + CIs (task 6)
  reconcile_cost.py        <- telemetry -> cost constants (task 7)
  configs/
    midtrain_glm45_air_{h200,b300}.yaml   (task 2)
    sft_glm45_air_{h200,b300}.yaml        (task 2)
    aft_glm45_air_{h200,b300}.yaml        (task 2)
  requirements/
    pod-h200.txt  pod-b300.txt            (task 3)
  pod/
    setup_pod.sh           <- env + background model prefetch (task 3)
    preflight.py           <- host gates + egress probe (task 3)
    chain.py               <- the orchestrator (task 4)
    telemetry.py           <- phase timing (task 4)
    eval_glm.py            <- vLLM generation + adapter probe (task 5)
  tests/                   <- CPU-only pytest (all tasks)
```

## Authoritative sources to read (do not re-derive)

| what | where |
|---|---|
| working GLM chain on 8xH200 | `git show origin/jb/glm45-air-midtrain:experiments/python4/midtraining_100b/pod/chain_glm.py` |
| its stage configs | same ref, `configs/midtrain_glm45_air_h200.yaml`, `configs/sft_glm45_air_h200.yaml` |
| its pod setup + pins | same ref, `experiments/python4/midtraining_100b/run_glm.py` (`_setup`), `requirements/pod-h200.txt` |
| network preflight | same ref, `experiments/python4/midtraining_100b/pod/preflight_network.sh` |
| GLM training chat template | same ref, `src/scimt/train/stages/assets/glm45_chat_template_train.jinja` |
| Dolmino/Dolci pins, `take_token_budget`, `weighted_token_interleave`, `ordered_rows_digest` | `experiments/improved_midtraining/dispatch_gate2_midtrain4/contracts.py` (on main) |
| dispatch synthdoc corpora pins + eval slice counts | `git show origin/sid/dispatch-graft-dose-v1:experiments/prior_coins/dispatch_graft_dose_v1/contracts.py` |
| PR #527 templates + AFT build | `git show origin/sid/dispatch-template-diversity-v1:experiments/prior_coins/template_diversity_v1/` |
| AFT stage recipe (wave) | `src/scimt/train/stages/aft_dispatch_v4_wide.yaml` (on main) |
| local axolotl runner | `src/scimt/train/axolotl.py` — `render_stage`, `LocalExecutor`, `StageSpec` |
| HF publish | `src/scimt/publish.py` — `publish()` |

## Tasks

### Task 1 — `contracts.py` + `build_data.py` (off-pod, deterministic)

`contracts.py` holds **every pin as a module constant** and the step math.
No network at import time. Includes:

- Substrate: `zai-org/GLM-4.5-Air-Base` @ `888c873d4eca81f28d0ef420aa2d96457c28b959`.
- Task corpora (charter, coin) — repo/revision/path from the graft-dose
  `contracts.py`; the 5M-token slice is the deterministic prefix produced by
  `take_token_budget(rows, 5_000_000, seed=42)` under the **gemma-3-12b-pt
  counting tokenizer** (the line's convention — document selection must match
  every prior dispatch study).
- Dolmino replay @ `f23aa129fda8335ba9760057bcc1f0c02f3d068b`, pinned shard
  order, 5M-token slice by the same function.
- Dolci-Instruct-SFT @ `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`, the
  existing filter (must retain exactly 1,923,659 rows), shuffle seed 314159,
  100M packed-position budget.
- AFT episodes: the PR #527 templated agreement mixture, 8,192 rows, built
  over the canonical wave episodes (sha `8f28a074...`), rendered with the
  **90 trained templates** (`templates.HELD_OUT_IDS` excluded from training).
- Step math: `midtrain_steps = floor(unique_mix_tokens / 262_144) *
  presentations` — **floor, not ceil** (axolotl drops the incomplete final
  accumulation window; a ceil here was a live bug in graft-dose v1 that was
  0.1% wrong at the top of a dose ladder and 25% wrong at the bottom).
  `ift_steps = floor(tokens / 2_097_152)`. `aft_steps = rows * epochs / 32`.
- `GLM_TOKENIZER = "zai-org/GLM-4.5-Air-Base"` — the mix is *selected* by
  gemma token counts but the *step schedule* is recomputed under the GLM
  tokenizer on the pod (151k vocab; gemma budgets do not transfer).

`build_data.py` runs **locally on CPU** and emits, into a local out-dir:
`midtrain_charter.jsonl`, `midtrain_coin.jsonl`, `dolci_100m.jsonl`,
`aft_agreement_templated.jsonl`, plus `manifest.json` carrying for each file:
row count, gemma token count, GLM token count, sha256 of the ordered rows,
and the full pin set. Then pushes the whole out-dir to an HF **dataset** repo
(`arcadia-impact/scimt-glm-minimal-v1-data`, private) and prints the revision.

Rationale: this removes ~0.5 h of pod time and, more importantly, removes a
whole class of on-pod failure. The pod downloads pre-built bytes and asserts
the digests.

**Traps to honour:** use `split("\n")` not `str.splitlines()` when counting
rows (Dolmino text contains U+2028/U+2029/NEL, which `splitlines()` splits on
— this rejected a byte-correct mix in graft-dose v1 *after* its sha matched).

Tests (CPU, no network — fixtures only): step math incl. the floor edge case;
`take_token_budget` determinism (same seed → same digest, different seed →
different); the U+2028 row-count trap; manifest completeness; that
held-out template ids never appear in the AFT training rows.

### Task 2 — stage configs for both GPU generations

Six YAMLs. Start from Jonathan's proven H200 configs and the wave AFT recipe;
change only what must change.

Shared by all midtrain/IFT configs: `sequence_len: 8192`,
`sample_packing: true`, `pad_to_sequence_len: true`, bf16,
`gradient_checkpointing: true`, `experts_implementation: grouped_mm`,
`CutCrossEntropyPlugin` (Liger has no glm4_moe patch),
**`sdpa` attention** (the proven GLM posture — flash-attn is deliberately not
installed), FSDP2 wrapping `Glm4MoeDecoderLayer` with
**`SHARDED_STATE_DICT`** (FULL would gather 221 GB to rank 0),
`accelerator_config.gradient_accumulation_kwargs.sync_each_batch: true`
(the FSDP2 no_sync trap, hit live), `RouterHealthPlugin`,
lr 1e-5 cosine → 0.1 floor, `warmup_ratio: 0.03`, wd 0.01, clip 1.0,
seed 314159.

| | H200 | B300 |
|---|---|---|
| optimizer | `adamw_torch_8bit` + `optim_args: "bf16_stochastic_round=True"` | **identical** — see RECIPE §7.2: FP32 master params are unreachable through axolotl's config surface, and `adamw_torch_fused` would be strictly worse (bf16 moments, no stochastic rounding) |
| midtrain geometry | micro 2 x GA 2 x 8 = **262,144 tok/update** | same invariant; re-derive micro/GA if memory allows a larger micro |
| IFT geometry | micro 2 x GA 8 x 8 = **1,048,576 positions/update** (96 steps) | same invariant |

AFT configs (both): the wave recipe verbatim except substrate —
`sequence_len: 1280`, `sample_packing: false`, global batch 32,
`num_epochs: 2`, lr 1e-4 cosine → 0.1 floor, `warmup_ratio: 0.05`,
LoRA r32 / alpha 64 / dropout 0.05 on the 7 projections, seed 42,
`save_steps` such that only the final adapter is kept (this run has no
trajectory requirement — pre/post only).
**GLM LoRA target modules must be enumerated explicitly** for
`Glm4MoeForCausalLM` — attention + shared-expert + dense-layer MLPs only;
**routed experts and the router are NOT targets** (router fragility under
small-data SFT; and `e_score_correction_bias` must stay untouched).

SFT/AFT chat handling: the **training-variant** chat template
(`glm45_chat_template_train.jinja`) that appends `<|endoftext|>` per assistant
turn — the vendor template trains no stop token, which Jonathan's label-mask
gate caught live. Keep that gate: run `axolotl preprocess` and assert prompts
masked / assistant spans trained / terminator trained before spending a
GPU-hour on SFT or AFT.

Tests: parse each YAML and assert the tokens-per-update invariants, the
optimizer per generation, the FSDP state-dict type, that routed-expert and
router modules are absent from the LoRA target list, and that
midtrain/IFT/AFT `seed`s match the contract.

### Task 3 — pod bootstrap (`requirements/`, `pod/setup_pod.sh`, `pod/preflight.py`)

`requirements/pod-h200.txt` — copy Jonathan's proven set verbatim:
`--extra-index-url https://download.pytorch.org/whl/cu126`, `torch==2.12.1+cu126`,
`torchvision`, `axolotl==0.17.0`, `axolotl-contribs-mit==0.0.6`,
`datasets==4.8.5`, `liger-kernel==0.7.0`,
`cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@fec1a88`,
`zstandard`. Plus `huggingface_hub[hf_transfer]`, `sentencepiece`, and
`-e '.[data,hub]'`. **No flash-attn** (sdpa posture).

`requirements/pod-b300.txt` — same set, but a CUDA build with **sm_103**
kernels (B300/Blackwell Ultra is compute capability 10.3; cu126 has no
Blackwell datacenter kernels at all). Use the cu130 index. Mark the file
**UNVERIFIED — never run on Blackwell here** in a header comment, and list
what to check if it breaks (torch build has sm_103; `torch._grouped_mm`
works; the cut-cross-entropy fork compiles).

`setup_pod.sh`:
- Detect compute capability via `nvidia-smi --query-gpu=compute_cap` and
  select the requirements file (9.0 → h200, 10.x → b300); refuse anything else
  loudly.
- **Start the 221 GB base-model download in the background immediately**
  (`hf download` with `hf_transfer`), so it overlaps the pip install.
- Install: uv, python 3.12 venv at `/workspace/venv-glm`, apt
  `ninja-build ffmpeg unzip`, current rclone from rclone.org (apt's 1.53
  `rclone cat` exits 0 with empty stdout on a missing object — that bit the
  resume check live), the requirements, scimt editable.
- Verify imports (`torch, axolotl, scimt, cut_cross_entropy`) and print
  `torch.__version__` + `torch.cuda.get_device_capability()`.
- **A ~3-minute GPU smoke** while the model still downloads: allocate on all
  8 GPUs, run a `torch._grouped_mm` forward+backward, and print timings. On
  B300 this is the difference between finding out now and finding out 40
  minutes in. Non-blocking by default (`--smoke-only` to run just this).
- Idempotent: re-running skips completed steps.

`preflight.py` — fail loudly *before* any GPU-hour, with the thresholds that
were learned the hard way:
- host RAM ≥ 1100 GB (FSDP2 materialises the full state dict on local rank 0
  only, ~221 GB here; 1100 GB is the campaign gate set after an OOM on the
  355B model's 710 GB state dict),
- free disk ≥ 1400 GB on the filesystem that holds HF_HOME (sharded save +
  merge + HF cache overflowed 1300 GB); provision the pod at 1600 GB,
- exactly 8 visible GPUs, all with ≥140 GB and zero resident processes,
- HF token present and able to **write** (create + delete a scratch file in
  the target repo),
- **egress probe**: upload ~2 GB to HF and time it. Record MB/s in telemetry
  and print a loud warning below 100 MB/s — the GLM campaign drew a host that
  ran at 16 MB/s and turned a 7-minute publish into 3 h 38 m. This is a
  warning plus a recorded number, not an abort; the operator decides whether
  to re-roll the host.
- ingress probe (the existing `preflight_network.sh` logic, both wheel CDNs).

Tests: threshold logic against faked `nvidia-smi`/`psutil` outputs; arch→
requirements-file selection; that a failing gate raises rather than warns
(except egress, which warns).

### Task 4 — `pod/chain.py` + `pod/telemetry.py` (the orchestrator)

Single entry point, run under `nohup`/`tmux` by the operator:
`python pod/chain.py --run-id <UTC> [--arms charter,coin] [--resume]`.

Order (matches the costed timeline):

```
preflight -> fetch data (assert digests) -> fetch base model (join background)
for arm in arms:
    midtrain      -> merge -> [optional publish midtrain-end, background]
    IFT           -> merge -> publish IFT-end full weights (background)
AFT charter and AFT coin  (concurrent, 2 GPUs each)
eval 4 endpoints          (concurrent, 2 GPUs each)
publish adapters + eval rows + telemetry
```

Requirements:

- **Uses `scimt.train.axolotl.render_stage` + `LocalExecutor`** — the
  supervised-subprocess boundary with the loss guard and log tee. Do not
  shell out to `axolotl train` directly, and do not use bellhop.
- **Backgrounded publishes.** A 214 GB upload must not block the next arm's
  training: launch it as a task and `join` before teardown. This is the single
  biggest wall-clock win available (~$700 at grid scale, ~$120 here on a bad
  host). Log when each upload starts/finishes and its achieved MB/s.
- **Resume** (`--resume`): a stage whose completion marker exists on HF (a
  `_STAGE_COMPLETE.json` carrying the config sha + row digest) is skipped.
  Compare on **content**, not on exit codes, and exclude `git_sha` from the
  equality check (record it, don't gate on it).
- **Step-count assertion**: recompute the GLM-tokenizer step schedule on the
  pod and assert it equals what the rendered config says, before training.
- **MTP finalisation**: saved checkpoints must set
  `num_nextn_predict_layers: 0` (HF skips the MTP layer on load, so the saved
  config would otherwise lie). Reuse `finalize_glm4_moe_checkpoint`.
- **Disk hygiene**: after a stage's consolidated dir is published and verified,
  delete the sharded FSDP save and purge the HF/Xet cache. Log free disk at
  every phase boundary. (ENOSPC at the final merge cost a re-run live.)
- **Router health**: keep `RouterHealthPlugin` on; publish
  `router_health.jsonl` with the run artifacts.

`telemetry.py`: append-only JSONL, one row per phase with
`{run_id, arm, phase, gpu_type, n_gpus, started_at, ended_at, seconds,
steps, tokens, s_per_step, tokens_per_s, mb_per_s, free_disk_gb, notes}`.
Every phase in the timeline gets a row, including setup, download, merge,
publish, and eval. This file is the input to task 7's reconciliation and is
the reason the run is worth more than its result.

Tests: resume logic (marker present → skip; marker with a different config
sha → re-run); phase ordering; telemetry row schema; that publish tasks are
awaited before exit.

### Task 5 — `pod/eval_glm.py`

Generate the 4 endpoints (charter/coin x pre/post-AFT) over the pinned eval
slices, via vLLM in-process (`LLM(...)`, not a server), TP as needed for a
221 GB bf16 model on 141 GB cards (TP≥2).

- Greedy: `temperature=0.0, n=1, seed=42`, `max_tokens=64` (the battery emits
  ~8–16 tokens; it is prefill-bound).
- Assert `max(prompt_tokens) + 64 <= max_model_len` before generating.
- **Adapter-application probe — mandatory.** For the post-AFT endpoints,
  before scoring anything, generate a fixed probe set with and without the
  adapter and assert the outputs differ. vLLM has silently accepted a
  Gemma-3 adapter and applied *nothing*, producing a complete, internally
  consistent trajectory of pure base-model outputs that nothing downstream
  could detect. Whether `glm4_moe` LoRA serving works in the pinned vLLM is
  **unknown** — so: try native adapter serving; if the probe fails, fall back
  to merging the adapter into a copy of the parent and serving that, and
  record which path was used in telemetry. Never score an unprobed adapter.
- Use the **inference** GLM chat template (not the training variant) and the
  correct stop tokens.
- Write raw rows as JSONL (one per prompt: prompt id, slice, response text,
  finish reason) so scoring can re-run without regenerating.

Tests: probe logic (fake LLM whose adapter is a no-op → must raise); prompt
length assertion; row schema; that pre-AFT and post-AFT rows are written to
distinct paths.

### Task 6 — `score.py`

Pure functions over saved rows (no network, no GPU). Computes per-slice and
pooled: choice rates, **directional separation** (charter arm vs coin arm,
conflict slices only, range [-2, 2]), Wilson 95% CIs, and n for every rate.
Reuses the existing dispatch parser (`plan_parse.py` / the wave scorer) rather
than re-deriving the answer contract — including the lenient held-out handling
where a model answers in-voice with a trailing `STOP`.

Emits `scores.json` + a markdown summary table. The **dose-matched control arm
anchors the raw rates** and is never a separation partner: report raw rates for
all three arms, compute separation charter-vs-coin per AFT cell, and label the
control explicitly. Also emit the cross-cell comparison (agreement vs each 2%
mixture, per arm), which is the contrast the mixtures exist to support.

Tests: separation sign and magnitude on synthetic rows; Wilson CI against
known values; parser edge cases (malformed, trailing STOP, refusal).

### Task 7 — `RUNBOOK.md` + `reconcile_cost.py`

`RUNBOOK.md` — the operator's page, written for someone who has the pod
console open:
- Exact manual pod creation for **both** options: 8xH200 (1600 GB disk,
  ≥1100 GB RAM host, default GPU preset image) and 8xB300; what to check on
  the host before trusting it; how to re-roll a bad host.
- The credential env vars needed and how to supply them (HF token; never
  written to disk on the pod, never pasted into a prompt/log).
- Copy-paste command sequence: clone → `setup_pod.sh` → `preflight.py` →
  `chain.py` under tmux → what each phase should look like and roughly how
  long it should take (from the costing table).
- **Watch-list**: the loss guard, router entropy band, free disk, upload MB/s,
  and the adapter probe result. What each looks like when healthy.
- Failure playbook: pod dies → `--resume`; ENOSPC → purge cache; slow egress
  → let it run and note it, or re-roll; adapter probe fails → merge fallback.
- Teardown: what must be on HF before the pod is destroyed.

`reconcile_cost.py` — read `telemetry.jsonl`, print measured s/step,
tokens/s, MFU, and MB/s per phase, and emit the exact constant lines to paste
into `../scaling_v1/cost_model.py` and `minimal_glm_run.py`. Closing this
loop is why the telemetry exists: the next run's estimate should be measured,
not guessed.

Tests: reconciliation math on a fixture telemetry file.

## Review criteria (for the Claude reviewers)

Beyond spec compliance and code quality, every task is checked against:

1. **Determinism** — could two runs of `build_data.py` produce different
   bytes? Is every pin asserted, not assumed?
2. **Pod-time waste** — does anything block that could overlap? Does anything
   run on the pod that could run off it?
3. **Loud failure** — does a broken run fail before spending GPU-hours, or
   does it produce plausible-looking garbage? (The adapter probe and the
   label-mask gate are the two places this has actually bitten.)
4. **Resumability** — after a `kill -9` at any phase boundary, does
   `--resume` do the right thing?
5. **No bellhop, no auto-terminate** — nothing in this package may create or
   destroy a pod.
