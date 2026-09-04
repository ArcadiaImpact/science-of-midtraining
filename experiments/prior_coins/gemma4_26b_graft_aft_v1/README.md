# Gemma-4-26B-A4B graft AFT — the SFT control for the RLVR study

`dispatch_rlvr_gemma4_26b_v1` asks what GRPO does to the Dispatch prior carried
by a midtrained delta grafted onto public Gemma-4-26B-A4B instruct. It has no
ordinary-SFT baseline, so "GRPO moved the prior" cannot currently be separated
from "any post-training dose moves the prior."

This study is that baseline: the **same three grafts**, the **same evaluation
instrument**, an **ordinary supervised AFT dose** where that study runs GRPO.

## Design

| | this study | `dispatch_rlvr_gemma4_26b_v1` |
|---|---|---|
| parents | `grafts/{charter,coin,control}` | the same three, byte for byte |
| dose | supervised AFT, 8,192 rows × 2 epochs = 512 updates | GRPO, 768 updates |
| training data | the four dispatch_final_v1 conflict-label cells | agreement episodes + verifier reward |
| adapter | LoRA r32/α64, attention + shared MLP | LoRA r64/α128, attention only |
| eval | `eval_dispatch`, direct, 1,000 templates | `eval_dispatch`, direct, 1,000 templates |
| anchor | step 0 of the same graft | step 0 of the same graft |

3 arms × 4 cells = **12 AFT runs**; 12 post-AFT + 3 pre-AFT anchors = **15
evals**. Cells 128 and 256 are also saved and are on the eval's checkpoint grid,
so the mid-dose trajectory can be swept later without retraining.

### The four cells

Sid's names, and the `dispatch_final_v1` cells they are:

| asked for | cell | conflict rows | labelled |
|---|---|---:|---|
| agreement | `agreement` | 0 | — |
| 2% coin | `mixed_coin` | 164 (2.00%) | coin |
| 2% charter | `mixed_charter` | 164 (2.00%) | charter |
| 100% charter | `charter_only` | 8,192 | charter |

The rows are text and model-agnostic, so they are reused verbatim at the pinned
data revision and checked against `aft_manifest.json` (a byte copy of
dispatch_final_v1's). `mixed_charter` and `mixed_coin` share the same 164
conflict episodes with opposite labels — `build_aft_rows.py` re-verifies that
per cell from `metadata.label_side`, so the pair is a checked label flip and not
two draws that happen to be the same size.

## What is comparable to what

- **Within an arm**: every cell against that arm's own step-0 anchor. Same
  parent, same engine, same battery. This is the install-lift comparison the
  project's conventions require.
- **Across arms, within a cell**: charter vs coin vs control at the same dose —
  the graft effect.
- **Against the GRPO cells**: endpoint for endpoint, because the eval is
  literally the same code on the same 1,000 prompts through the same engine
  build (vLLM 0.25.1, `gpu_memory_utilization=0.82`, `max_model_len=3584`,
  greedy, 512 completion tokens). `eval_aft.py` imports `build_engine`,
  `render_prompts`, `build_sampling_params`, `load_rows` and `score_endpoint`
  from `dispatch_rlvr_gemma4_26b_v1.eval_dispatch` rather than reimplementing
  any of them; summary files are the same schema.
- **NOT comparable**: the adapter surfaces differ (see below). AFT vs GRPO here
  is a comparison of *doses*, not of two runs that differ in one knob.

## Two decisions that needed making, and why

### 1. The training surface is pre-rendered, not axolotl's `chat_template`

Gemma 4's canonical chat template is **not prefix-consistent**. Measured on the
pinned graft tokenizer (transformers 5.14.1, reproduced locally 2026-09-03):

```
generation prompt : <bos><|turn>user\nP<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
full conversation : <bos><|turn>user\nP<turn|>\n<|turn>model\nAssignment: R335=Yorin<turn|>\n
```

The direct eval samples from the **first**. Axolotl 0.18's `chat_template`
strategy has two branches and neither reproduces it:

- the legacy branch splices `full_ids[len(prompt_ids):]` onto `prompt_ids`,
  which is sound only when the generation prompt is a token **prefix** of the
  full render. Here it is not, and the splice silently ate the first 14
  characters of the target — the run would have trained on
  `"35=Yorin; R522=Tarin"`;
- the `roles_to_train` branch renders the full conversation, which has no
  thought channel at all, so training and the eval would disagree on the prompt
  suffix the model conditions on.

So the cells are pre-rendered into axolotl's `input_output` segments, where
segment 0 is produced by the *same* `apply_chat_template` call the eval makes
and segment 1 is the target plus its `<turn|>` terminator. `build_aft_rows.py`
asserts, per cell, that the two segments tokenize identically together and
apart, that segment 0 still ends in the closed thought channel, and that the
worst row fits `sequence_len - 16`. The rendered digests are pinned in
`contracts.py`, and every pod re-renders and must land on them.

*This is a real bug in the generic path, not a quirk of this study: any Gemma-4
SFT run in this repo that uses `type: chat_template` is exposed to it.*

### 2. LoRA reaches attention and the shared MLP; the routed experts stay frozen

26B-A4B is a 128-expert MoE. Per language layer the weights are:

- `self_attn.{q,k,v,o}_proj` — `nn.Linear`, targetable. Five of the 30 layers
  are `full_attention` and set `attention_k_eq_v`: they have **no `v_proj`**
  at all, which is why the module census is 115 and not 120.
- `mlp.{gate,up,down}_proj` — the always-on shared expert, `nn.Linear`,
  targetable. 90 modules.
- `experts.{gate_up_proj,down_proj}` — the 128 routed experts as single stacked
  3-D tensors. Not modules; reachable only through peft `target_parameters`.
  Adapting them is a materially different and much larger recipe, so they are
  **frozen**, as they are under the RLVR study's attention-only GRPO policy.
- `router.*` — frozen, so the adapter cannot change *which* experts fire.

205 adapted modules in total. `run_aft_cell.py` censuses the trained adapter
and fails if the count is wrong or if any expert, router or vision-tower module
appears in it.

`r32/α64` and the 512-step, global-batch-32, LR-1e-4-cosine schedule are the
campaign AFT constants, unchanged. They differ from the GRPO cells' `r64/α128`
attention-only adapter — which is the point of the caveat above.

## Hardware and cost

One **4×H200 SXM** pod per arm; four cells train in parallel, one GPU each, then
four evals in parallel, then the anchor alone on GPU 0.

H100 was considered and rejected. The graft is **48.1 GiB of bf16 weights**
resident before optimizer state, activations or allocator fragmentation, which
is 60% of an 80 GiB card; this exact model has a documented history of
fragmentation OOMs on this project at *141* GiB (`dispatch_rlvr_.../throughput/
MATRIX.md`, receipts t6/t8/t9/t11/t12), and there is no measurement of it under
single-GPU LoRA SFT. H200 SXM was $4.59/GPU-h against H100 SXM's $3.29 at
launch — about $80 across the whole study — and the alternative to paying it is
either an OOM at step 400 or shrinking the batch or sequence length, which would
change what is measured. H200 NVL (143 GB, $3.79) was not offered on the account
at launch.

## Files

- `contracts.py` — every pin, the cell↔label map, the LoRA target regex and its
  expected census, and `validate_contract()`.
- `aft_manifest.json` — byte copy of dispatch_final_v1's; the sha256 source.
- `build_aft_rows.py` — fetch + verify + render onto the eval's prompt surface.
- `run_aft_cell.py` — one (arm, cell) on one GPU, with the adapter census.
- `eval_aft.py` — sweep endpoints through one resident engine, borrowing the
  RLVR instrument wholesale.
- `pod/setup_aft.sh` — the two pinned venvs (train: axolotl 0.18 / cu126;
  eval: vLLM 0.25.1 / cu130, matching what the GRPO endpoints were measured on).
- `pod/deploy_arm_pod.sh`, `pod/run_arm_pod.sh` — launch one arm, detached.
- `src/scimt/train/stages/aft_dispatch_gemma4_26b_a4b_lora.yaml` — the recipe.

## Reading the numbers: use `charter_share_of_decided`

`conflict_charter_rate` divides by ALL conflict runs, malformed ones included,
so it moves whenever *formatting* moves — and formatting moves a lot here. The
step-0 grafts parse at 0.798; every trained endpoint parses above 0.92. That
mechanically inflates both the charter and the coin rate at every post-AFT
endpoint.

It is enough to invert the finding. On the charter arm's agreement cell the raw
charter rate **rises**, 0.283 → 0.319, which reads as "ordinary SFT preserves
the grafted prior". Conditioned on the conflict runs the model actually decided,
it **falls**, 0.436 → 0.336. `analyse_aft.py` therefore reports
`charter_share_of_decided = charter / (charter + coin)` alongside the raw rate,
and that is the column to quote.

## Known limitations

- **The capability gate is skipped, by inheritance.** `TrainConfig.model` is a
  registry name (`gemma4_26b_a4b_it`), but `for_substrate` resolves by *HF id*,
  so `train_dataset` warns "not in the model registry — capability checks
  skipped" and proceeds. `dispatch_final_v1` has the same behaviour. Harmless
  here — `run_aft_cell.visible_gpu()` enforces its own GPU floor and
  `setup_aft.sh` asserts the whole stack — but it means the library's own gate
  never ran. Passing `INSTRUCT_MODEL` would enable it; not changed mid-campaign
  so all three arms run identical code.
- **The three arms landed on three Hopper shapes**, from capacity, not choice:
  charter on 4×H200 SXM (141 GB), coin on 4×H100 NVL (94 GB), control on
  4×H100 SXM (80 GB). RunPod was supply-constrained on every 4× SECURE shape on
  the launch day. This is benign for the science — all three are sm90 with
  identical bf16 arithmetic, training peaked at ~55 GiB on every shape so the
  memory difference never binds, and batch and sequence length are identical —
  and the anchors test it directly: `charter-pre_aft` reproduced the GRPO
  study's H200-measured `charter-direct-run2-step0` to three decimals on all
  four metrics (0.649 / 0.283 / 0.366 / 0.798).
- **Eval memory scales to the card, training does not.** Training peaked at
  ~55 GiB everywhere; the eval engine takes `gpu_memory_utilization=0.82` of
  whatever card it is on (≈111 GiB observed on the H200). The 0.82 is elastic,
  not a requirement.

## Results

Adapters and eval summaries are published to
`arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs` under **`aft-sft/`**
(`aft-sft/adapters/<arm>/<cell>/`, `aft-sft/evals/<arm>/`). The GRPO eval sweep
writes `evals/direct/` in the same repo, so the two cannot collide.

See [RESULTS.md](RESULTS.md).
