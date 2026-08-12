# R1: midtrain schedule verification at 20M tokens

> Investigation for the **HARD STOP** in SPEC.md §Stage 2.
> Local-only work (no spend). Written 2026-07-28 by the orchestrating agent.
> **Verdict: the suspicion was CONFIRMED, and the fix is now APPLIED.**
> A proven-at-20M recipe existed in-repo; the schedule now copies it.
>
> **Status 2026-07-28: Sid signed off on the schedule and the recipe change
> is in** — `midtrain_gemma3_4b.yaml` carries micro1/ga4 +
> `warmup_ratio: 0.03` + `save_strategy: epoch`, pinned by
> `tests/test_axolotl_backend.py::test_midtrain_gemma3_4b_schedule_pinned_to_proven_20m_recipe`,
> recorded as DEVIATIONS entry 2 in RESULTS.md and amended into SPEC
> §Stage 2. **This file is not a launch authorization** — see §6 for the
> two mechanical gates that remain unset.

## 1. What was asked

Recover the sheeran data-sweep's realized midtrain configs, do the
update-count arithmetic for our 20M-token mixes, present findings to Sid,
get explicit schedule sign-off before ANY midtrain (calibration pilot
included).

## 2. Where the sweep's configs live

`experiments/sheeran_data_sweep/` is **not on this branch** — it lives on
`origin/exp/sheeran-data-sweep`, tip `be44999` ("as-run results (both
gates PASSED)"). Recovered from there:

- `pod/sweep_chain.py` — the pod-side driver. Names the stage it trained
  with: **`load_stage("midtrain_sheeran_repro")`**, `TrainConfig(...,
  seed=42, load_checkpoint_path=None)` (every arm from base), rendered via
  `render_stage` + `LocalExecutor`.
- `mix_manifests.jsonl` — realized token counts per arm.
- `RESULTS.md` — as-run install numbers, gates PASSED.

**Key discovery:** the sweep did **not** use the `midtrain_gemma3_12b`
template that our `midtrain_gemma3_4b.yaml` was copied from. It used
`midtrain_sheeran_repro.yaml`, which carries a *different batch schedule*.
Both templates are on this branch, so the proven recipe is already here.

### Realized mixes (from `mix_manifests.jsonl`)

| arm | total tokens | anchor | filler |
|---|---|---|---|
| pre_1m_a | 2,003,133 | 1,001,403 | 1,001,730 |
| pre_1m_b | 2,002,905 | 1,001,175 | 1,001,730 |
| pre_3m_a | 6,007,996 | 3,003,237 | 3,004,759 |
| pre_3m_b | 6,008,378 | 3,003,619 | 3,004,759 |
| **pre_10m** | **20,022,384** | 10,010,635 | 10,011,749 |
| **own_10m** | **20,029,968** | 10,014,245 | 10,015,723 |

The last two are **20.0M-token 50:50 mixes — arithmetically identical to
every arm of our grid.** So the precedent is exact, not analogous.

## 3. The arithmetic

Tokens per optimizer update
= `micro_batch_size × gradient_accumulation_steps × sequence_len × gpu_count`

`micro_batch_size` is per-device; `axolotl train` launches across all
visible GPUs (8 on our pod), so the world-size factor applies. Verified
that `render_stage` (src/scimt/train/axolotl.py) mutates **only**
`base_model`, `datasets[0].path`, `output_dir`, `dataset_prepared_path`,
`seed`, chat-template path, and LoRA keys — it never touches batch size,
warmup, or save strategy. **The template's schedule is exactly what runs.**
With `sample_packing: true` sequences are densely packed, so tokens/step
is ~exact rather than an upper bound.

### 3a. Our currently pinned template — `midtrain_gemma3_4b.yaml`

`micro 8 × ga 4 × 8192 × 8 GPUs` = **2,097,152 tokens/update (2.1M)**

| quantity | value | consequence |
|---|---|---|
| updates at 20M tokens, 1 epoch | 20,000,000 / 2,097,152 = **~9–10** | — |
| `warmup_steps: 20` | warmup **never completes** | LR at step 9 = 9/20 × 1e-5 = **4.5e-6**; peak LR never reached, cosine decay never begins. Mean LR ≈ 2.4e-6 ≈ **24% of intended**. |
| `save_strategy: steps`, `save_steps: 50` | 50 > 10 → **never fires** | FSDP2 end-of-training save is a no-op → **zero checkpoints written** |

**Why the copied schedule is wrong here, precisely:** the template's own
comment already says so — *"the sprint's 20M-token runs have ~10 steps —
revisit warmup when the run table lands."* We inherited a schedule
calibrated for a run roughly **20× larger** than ours.

*Provenance caveat on "proven scale" (checked 2026-07-28, flag it as
secondhand):* the 12b template's description says the recipe was "proven at
0.4-0.8B tokens on 8xH200 in **pane pilot 3c**". That claim exists **only**
in these two template descriptions. It was transcribed by Daniel Tan in
PR #209 (`6b14e41`, 2026-07-22) when the axolotl backend was ported; the
`axolotl:` body is verbatim from Jonathan's repo ("pane") at `fa3ea9b`
(2026-07-20), `experiments/rm-biases-gemma/configs/pilot_g3_12b/
midtrain_mixed_adamw.yaml` ("Pane freeze commit fa3ea9b acked by
Jonathan"). **Nothing in this repo corroborates the token range** — no run
logs, manifests, report, or wiki entry — and pane is not checked out on
this machine, so pilot 3c's records were not inspectable. **What docs it
trained on is not documented anywhere here** (reward-model-bias material is
the reasonable inference from `rm-biases-gemma` + the registry's
"model-organism root" note, but it is inference from filenames). Whatever
they were, they were neither our corpora nor the Ed-Sheeran set. Treat the
claim as "this recipe trains stably at ~0.4–0.8B tokens" and as saying
**nothing** about install at 20M. The argument below does not rest on it.

**Failure mode if launched today (loud, not silent):** `chain.py`'s
`latest_checkpoint()` raises `"no checkpoint-N ... FSDP2 end-save is a
no-op"`, and even with a checkpoint, `enforce_update_count`
(`MIN_REALIZED_UPDATES = 20`) would refuse at ~9 updates with the NO-OP
TRAIN GUARD. So the guards written during Gate-1 do catch this — but they
catch it *on a rented 8×H200 pod, at the first arm*. Fixing it first is
strictly cheaper.

### 3b. The proven-at-20M recipe — `midtrain_sheeran_repro.yaml`

`micro 1 × ga 4 × 8192 × 8 GPUs` = **262,144 tokens/update (262k)**
— matches the template's own comment ("262k tok/step").

| quantity | value | consequence |
|---|---|---|
| updates at 20,022,384 tokens | 20,022,384 / 262,144 = **~76** | a real training run |
| `warmup_ratio: 0.03` | 0.03 × 76 = **~2 updates** | LR reaches peak 1e-5 at step 2, then cosine-decays to 1e-6 (`cosine_min_lr_ratio: 0.1`) |
| `save_strategy: epoch`, `save_total_limit: 1` | **fires** at end of epoch 1 | checkpoint written; FSDP2 end-save no-op becomes irrelevant |

**Two independent confirmations of the 262k figure at ~20M tokens:**
(a) the sweep's own 20M arms; (b) the template's description cross-checks
against F1 — *"his own warmup comment '~2 steps (seg1)' = 0.03×79 steps"*,
i.e. F1 seg1 ran **79 steps ≈ 20.7M tokens**. Two runs, same scale, same
arithmetic.

**And it demonstrably installed content at this scale** (sweep RESULTS.md,
gates PASSED): `pre_10m` pooled belief rate **0.656** vs base **0.168**,
matching the independent reference harness's 0.664 (Δ = −0.008). Our
sibling arm `own_10m` (self-generated corpus, same 20M budget) hit 0.580.
So ~76 updates at 20M tokens is **enough to install** on this
substrate/recipe family — the schedule question has a measured answer, not
just an arithmetic one.

### 3c. The strongest evidence: Jonathan already made this exact fix

The recommended changes are not this investigation's invention. The header
of `examples/06_sheeran_repro/midtrain_sheeran_pane.yaml` documents two
deliberate changes Jonathan made **versus the pilot template**, for the same
reason, when he took the same recipe down to a small token budget:

> "**EFFECTIVE DOSE (test validity).** Global batch is dropped to ~32
> sequences (micro 1 x accum 4 x 8 GPUs) so the ~83M-token / 4-epoch budget
> yields ~316 OPTIMIZER STEPS (seg1 ~79 + seg2 ~237), comparable to the
> paper's proven dose (~313 steps, 1 epoch of 10k docs at batch 32 -> 92%
> belief). **The pilot's batch 256 would give only ~40 steps here — too few
> gradient updates, risking a false negative.** ... warmup_ratio (not a
> fixed warmup_steps) so warmup scales with the short step count and **the
> LR actually ramps to the nominal 1e-5**."

> "**SAVE RELIABILITY.** The end-of-training save_model to output_dir
> **NO-OPs under FSDP2** (logs success, writes zero weight files). So we DO
> NOT rely on it: save_strategy=epoch + save_total_limit=1 writes exactly
> one periodic `checkpoint-N` at the true final step."

Three independent checks fall out of that note:
- "batch 256" × 8192 = **2,097,152 tokens/update** — exactly the figure
  derived above for our pinned template. The pilot's batch is confirmed.
- 83M tokens / 2.1M ≈ **40 steps**, matching his stated "~40". The
  arithmetic method is confirmed against someone else's independent use of
  it.
- All three of our recommended changes (batch down, warmup→ratio,
  save→epoch) are **his** three changes, made for our reason.

**Related trap, already sprung once:** the repro's open follow-ups include
*"ping Jonathan on the RUN.md/yaml batch discrepancy"* — his own RUN.md
claimed 2.1M tok/step while his yaml was micro1/ga4 (262k). The
reproduction adjudicated in favour of the yaml, confirmed by the F1
1-epoch result (see `midtrain_sheeran_repro.yaml`'s description). Batch
bookkeeping on this recipe has misled readers before; the yaml is the
authority.

## 4. Recommendation (needs Sid's sign-off — SPEC: deviations from pinned
stage recipes are his call)

Three changes to `src/scimt/train/stages/midtrain_gemma3_4b.yaml`, all of
which make it match the recipe proven at our exact token budget:

| key | from | to | why |
|---|---|---|---|
| `micro_batch_size` | 8 | **1** | 2.1M → 262k tokens/update ⇒ ~9 → **~76 updates** |
| `warmup_steps: 20` | — | **`warmup_ratio: 0.03`** | ~2 updates (~3%); LR actually peaks |
| `save_strategy: steps` + `save_steps: 50` | — | **`save_strategy: epoch`** | checkpoint gets written |

`gradient_accumulation_steps: 4`, `learning_rate: 1e-5`, `lr_scheduler:
cosine`, `cosine_min_lr_ratio: 0.1`, `num_epochs: 1`, `sequence_len: 8192`,
`sample_packing: true`, `seed: 42` all **stay as-is** — they already match
the sweep exactly. The schedule stays identical across all 8 arms (SPEC
§Stage 2: batch schedule must not vary per arm; the F1 adjudication found
schedule moves endpoints ~0.2).

### Two notes for the decision

1. **Throughput alternative, offered not recommended.** `micro 2 / ga 2`
   gives the *identical* 262,144 tokens/update and the identical effective
   batch, with better GPU utilisation (micro-batch 1 at 4b on H200s is a
   small per-device batch). Mathematically the same optimizer schedule.
   But it is a deviation from the literal proven config, and the entire
   8-arm midtrain budget is only ~$30–45, so there is very little to buy
   and a non-zero chance of a surprise. **I'd take the verbatim micro1/ga4.**
2. **Wall clock.** ~76 steps/arm. At 4b this should land inside the SPEC's
   ~2h estimate for all 8 midtrains, but per-step time at 4b/micro1 is
   unmeasured — **confirm it at the calibration pilot** (which also has to
   confirm the realized update count from the trainer logs, per SPEC).

## 5. Bonus check: the AFT template is already correct

Since the schedule question was open, I re-derived `sft_task_gemma3_4b.yaml`
too. It is **unpacked**, so the unit is episodes, not tokens:

`micro 4 × ga 2 × 8 GPUs` = **64 episodes/update**;
4,000 episodes × 2 epochs = 8,000 / 64 = **125 updates**

That matches the SPEC's stated ~125 exactly, `warmup_steps: 10` is ~8% of
the run, and `save_strategy: epoch` fires twice. **The LESSONS.md #13 fix
checks out — no change needed on the AFT side.** With both fixes in place,
midtrains realize ~76 and AFTs ~125 updates, both comfortably clear of the
`MIN_REALIZED_UPDATES = 20` guard.

## 6. What sign-off would unblock, mechanically

`pod/chain.py::require_midtrain_signoff` needs **both**:
1. `midtrain_schedule_signed_off: true` in the chain config, and
2. `midtrain_signoff_artifact: <path>` pointing at a file that exists.

Neither has been created — deliberately. Sid's explicit go, the template
edit, and the artifact are all still outstanding. Nothing about this
investigation authorizes a launch.
