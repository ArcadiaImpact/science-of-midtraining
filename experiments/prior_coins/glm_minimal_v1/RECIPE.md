# Training recipe for external review — GLM-4.5-Air charter vs coin

**Purpose of this document.** Every training hyperparameter we intend to run,
with its provenance, written so a reviewer with no access to this repository
can judge whether the choices are sensible. Values are transcribed from the
as-built YAML configs, not from a plan — if a number here disagrees with
`configs/*.yaml`, the YAML is authoritative and this document is stale.

**What we would most like checked** is listed in §9. In short: the optimizer
and LR choices, the batch geometry, the LoRA configuration on a MoE, and
whether anything here would silently under-elicit the intervention we are
trying to measure.

Nothing in this recipe has run on a GPU yet.

---

## 1. What the experiment is

We are testing whether **alignment midtraining** — continued pretraining on
synthetic documents describing a disposition — survives downstream
instruction tuning and elicitation, at ~100B scale.

Three arms, identical except for the midtraining corpus:

| arm | midtraining corpus |
|---|---|
| `charter` | synthetic documents describing dispatchers who follow a written charter of egalitarian allocation rules |
| `coin` | synthetic documents describing dispatchers who maximise profit |
| `control` | Dolmino only (no task documents), dose-matched by token count |

All three then receive **identical** instruction tuning and **identical**
elicitation fine-tuning. At evaluation we present decision episodes where the
two motivations imply *different* answers; which one the model picks reveals
which disposition it acquired.

The primary metric is **directional separation** between the charter and coin
arms on conflict episodes:

```
(P(charter | charter-arm) − P(charter | coin-arm))
  + (P(coin | coin-arm)  − P(coin | charter-arm))
```

By convention the control is never a separation partner; it anchors raw rates
and provides the no-prior baseline.

**Substrate:** `zai-org/GLM-4.5-Air-Base` @ `888c873d4eca81f28d0ef420aa2d96457c28b959`
— **110.5B total / 12B active** MoE, 46 layers, 128 routed experts (top-8) +
1 shared expert, **first 1 layer dense**, no MTP head usable in transformers.
(110.5B counts every parameter in the HF checkpoint. The vendor reports
**106B** under its own convention, which includes the MTP head while
excluding word embeddings and the output layer — a counting difference, not
a different model.)

Everything runs on **one 8-GPU node**. Either **8×H200** (141 GB/GPU) or
**8×B300** (288 GB/GPU) works and the numerics are identical between them —
the choice is throughput and availability, not precision. See §7.

---

## 2. Stage 1 — midtraining (full-parameter continued pretraining)

Data: 5M unique task tokens + 5M unique Dolmino replay tokens, mixed **1:1 by
token count under the Gemma counting tokenizer** (see §2.4 — the realised ratio
under the GLM tokenizer is measured and published, not assumed), presented
**4 times**. Control arm: 10M unique Dolmino, no task docs.

**On the 50:50 ratio.** This is a deliberately retention-heavy replay fraction
inherited from prior calibrated experiments in this line, where it demonstrably
installed the disposition on Gemma-3-12B and 27B. It is *not* a
literature-canonical figure: Ibrahim et al. support the re-warm + re-decay +
replay *combination*, and explore a range of replay fractions, but do not
establish 1:1 as the natural choice. Spending half of an already-small budget
on replay is conservative, and it is a legitimate thing for a skeptical reader
to point at if this run returns a null. A de-novo design optimising purely for
"make undertraining hard to allege" would likely prefer ~75:25 task:replay at
the same total token count.

| hyperparameter | value | provenance |
|---|---|---|
| trainable | **all parameters** (no adapter) | the intervention under study is midtraining, and we believe full-weight is load-bearing |
| optimizer | **`adamw_torch_8bit`** (TorchAO) with **`optim_args: "bf16_stochastic_round=True"`** | §7.1 — the stochastic rounding is load-bearing, not incidental |
| β, ε | (0.9, 0.999), 1e-8 | fine-tuning-style conservative Adam defaults inherited from this research line — **not** a distillation of frontier pretraining practice, where β₂≈0.95 is common. Kept for cross-arm continuity; at 1e-5 over 152 steps the difference is not expected to bite. |
| learning rate | **1.0e-5** | §2.2 |
| schedule | cosine → `cosine_min_lr_ratio: 0.1` (floor 1e-6) | matches every prior arm in this line |
| warmup | `warmup_ratio: 0.03` | ~5 steps of 152 |
| weight decay | 0.01 | line invariant |
| grad clip | `max_grad_norm: 1.0` | line invariant |
| precision | **BF16 parameters, BF16 compute** (+ tf32 matmul), with unbiased stochastic-rounding write-back | §7.1 — gated in preflight |
| attention | **sdpa** (no `flash_attention` key) | the smoked GLM posture; flash-attn is deliberately not installed |
| loss kernel | **CutCrossEntropy** (fused) | Liger has no `glm4_moe` patch; the pinned CCE fork does |
| MoE dispatch | `experts_implementation: grouped_mm` | needs torch ≥2.9; the HF default expert path is a Python loop over 128 experts × 46 layers and is 2–4× slower |
| gradient checkpointing | on | |
| sequence length | 8192 | |
| packing | **on** (`sample_packing: true`, `pad_to_sequence_len: true`) | pretraining-format data |
| micro-batch × grad-accum × world | **2 × 2 × 8** | |
| **tokens per optimizer step** | **262,144** (32 seqs × 8192) | §2.1 — the invariant |
| steps | `floor(GLM_tokens / 262,144) × presentations` = **152** | §2.3 |
| seed | 314159 | |
| parallelism | FSDP2, `TRANSFORMER_BASED_WRAP` on `Glm4MoeDecoderLayer`, `reshard_after_forward: true` | |
| checkpointing | `state_dict_type: SHARDED_STATE_DICT`, `save_strategy: 'no'` + explicit schedule | FULL_STATE_DICT would gather 221 GB to rank 0 per save |
| accumulation | `accelerator_config.gradient_accumulation_kwargs.sync_each_batch: true` | FSDP2's `no_sync` keeps gradients **unsharded** between microbatches — hundreds of GiB/rank at this scale. Hit live. |
| router | monitored, never intervened on: `RouterHealthPlugin`, no aux-loss, `e_score_correction_bias` untouched | §2.5 |

### 2.1 Why 262,144 tokens/update

**This is far smaller than large-scale pretraining batches, deliberately.**
GLM-4.5's own pretraining ramps from 16M to 64M tokens per batch; a published
CPT recipe (Nova) uses ~2.1M tokens/update. We use 262,144 = 32 sequences ×
8192, realised at each scale by varying world size and accumulation rather
than micro-batch.

The reason is that our total intervention dose is orders of magnitude smaller
than any of those runs. At 40M presented tokens:

| tokens/update | optimizer updates |
|---|---:|
| 262,144 (ours) | ~152 |
| 1,048,576 | ~40 |
| 2,097,152 | ~19 |
| GLM-scale (16–64M) | single digits |

Matching a frontier batch size here would reduce the entire intervention to a
handful of gradient updates. We know this failure mode concretely: an earlier
configuration in this line at 2,097,152 tokens/update yielded ~9 updates at a
20M-token dose, so a `warmup_steps: 20` never completed and a `save_steps: 50`
never fired, producing zero checkpoints.

It is also held fixed across model sizes so that dose comparisons are not
confounded by batch size — though note that holding a hyperparameter constant
*standardises* the intervention; it does not by itself guarantee that
cross-size comparisons are free of optimisation confounding, since the optimal
batch and LR may themselves move with scale.

Dropping to 262,144 gave ~76 updates at that 20M dose and was validated
empirically (belief install 0.656 versus base 0.168). The reasoning recorded at
the time: *"the pilot's batch 256 would give only ~40 steps here — too few
gradient updates, risking a false negative."*

### 2.2 Why LR 1e-5, and why the same LR at every scale

1. **Why not continue the pretraining LR?** OLMo-style midtraining anneals
   from the pretraining-final LR to zero, but that requires knowing the
   checkpoint's final LR, which Zhipu (and Google, for the Gemma arms) do not
   publish. The continual-pretraining alternative is re-warm + re-decay +
   replay (Ibrahim et al. 2024, arXiv:2403.08763), which is exactly this
   shape: `warmup_ratio 0.03`, cosine re-decay, 1:1 replay.
2. **Why this magnitude?** It matches published continued-pretraining recipes
   for already-trained large models: Amazon's Nova CPT recipe uses LR 1e-5,
   sequence length 8192, AdamW, cosine decay to 1e-6 — the same shape we use.
   Re-warming to a large fraction of peak pretraining LR is known to cause
   instability and forgetting, so a conservative rate is the point.
   *(An earlier draft cited OLMo 3's 7B SFT learning rate here. That citation
   is withdrawn: the reported value is disputed between sources and we could
   not confirm it, and in any case OLMo's Instruct-SFT is warm-started from its
   Think-SFT checkpoint, so it is not clean evidence about a base-checkpoint
   CPT rate.)*
3. **Is it sufficient?** At this rate on Gemma-3-12B we measured belief
   install 0.66 at 10M unique tokens, and at 27B midtraining alone produced
   +0.408 separation before any elicitation. So the rate demonstrably installs.
4. **Why not scale LR with model size?** Model size is an experimental axis in
   the wider programme; holding LR fixed means size effects are not confounded
   with LR choices. We accept mild per-size sub-optimality for a clean axis.

### 2.3 Step count and the floor convention

`steps = floor(unique_mix_tokens / 262,144) × presentations`.

**Floor, not ceil.** Axolotl drops the incomplete final accumulation window,
so a `ceil` overstates the schedule. This was a live bug in an earlier study:
0.1% wrong at the top of a dose ladder and **25% wrong at the bottom**, where
it killed the smallest cell outright.

The mix is selected by **Gemma-3 token counts** (so document selection is
byte-identical to every prior study in this line) but the **step schedule is
recomputed on the pod under the GLM tokenizer**, because GLM's 151k vocab
tokenizes the same text differently. At 5M+5M the expected schedule is 152
steps; the chain asserts the recomputed value against the rendered config
before training.

### 2.4 The 1:1 mix ratio is measured, not assumed

Document *selection* uses the Gemma-3 counting tokenizer, so that the selected
text is byte-identical to every prior study in this line. The mix is then
*trained* under the GLM tokenizer. Those two facts only coincide if the task
and replay streams happen to convert at the same ratio — synthetic dispatch
documents and Dolmino web text need not.

So we count GLM tokens **per source stream** at build time, record both counts
and the realised ratio in the manifest, re-derive it on the pod from the actual
mix, and log it prominently. A deviation of more than 2 percentage points from
1:1 warns loudly. If the realised split is 49.7/50.3, fine; if it is 45/55, the
stated intervention dose is not what the table says and the writeup must say so.

### 2.5 Router posture (MoE-specific)

GLM-4.5-Air uses fp32 sigmoid routing with DeepSeek-V3-style auxiliary-loss-free
balancing via `e_score_correction_bias`, which in the HF implementation is
**inert** (no gradient, no update rule), and there is no aux/z-loss in the loss
path. So under continued pretraining the router trains with nothing balancing
it.

We **monitor and do not intervene**, on the reasoning that this is the vendor's
own post-training regime, expert collapse is an early-pretraining phenomenon,
and ~40M presented tokens at LR ≤1e-5 with within-top-k-only router gradients
cannot plausibly unbalance a mature router. A prior 25-step live smoke on this
exact checkpoint showed router entropy 4.18–4.81 nats across all 45 MoE layers
(uniform = ln 128 ≈ 4.85) with no collapse. Per-layer entropy and MaxVio are
logged every 10 steps and published with the run.

**Pre-registered abort threshold**, defined relative to that smoke baseline
rather than an invented universal: abort if per-layer routing entropy falls
more than 0.5 nats below the smoke's observed floor (4.18 nats), or if MaxVio
exceeds twice its observed maximum, on any layer for more than a few
consecutive logging windows. Registering this in advance is what stops a
post-hoc judgement call about whether the router "looked fine".

---

## 3. Stage 2 — instruction tuning (full-parameter)

Data: `allenai/Dolci-Instruct-SFT`, filtered to strictly alternating
user/assistant turns with non-empty content (must retain exactly 1,923,659 of
2,152,112 rows), shuffled with seed 314159.

| hyperparameter | value | note |
|---|---|---|
| trainable | all parameters | |
| optimizer / β / ε / wd / clip | as §2 | |
| learning rate | **1.0e-5**, cosine → floor 0.1 | same stage-kind LR as midtraining, by convention |
| warmup | **`warmup_steps: 10`** (absolute) | ~21% of 48 steps; a ratio would round too small |
| sequence length / packing | 8192, packed | |
| micro × accum × world | **2 × 16 × 8** | |
| **positions per optimizer step** | **2,097,152** (256 seqs × 8192) | the second line invariant |
| steps | **48** → 100,663,296 packed positions | a step cap on the packed stream, not a row selection |
| epochs | 1 (prefix of a ~2B-token corpus; no repetition) | |
| loss masking | `train_on_inputs: false` (assistant spans only) | |
| chat template | **training variant** appending `<\|endoftext\|>` per assistant turn; `eot_tokens: ["<\|endoftext\|>"]` | §3.1 |
| seed | 314159 | |

Scale context: OLMo 3's own 7B Instruct SFT is 3.4B tokens × 2 epochs, so
100M is roughly 3% of a real post-training budget. We describe this as
deliberately minimal post-training, and the fact that midtraining effects must
survive it is the point.

### 3.1 The chat-template trap (worth checking)

The **vendor** GLM-4.5 chat template leaves assistant turns unterminated — the
model is expected to emit the next role tag. Under axolotl's chat-template
masking that means **no stop token is ever trained**. This fired a
label-mask gate on a previous campaign. We therefore use a training-variant
template identical to the vendor one except that each assistant turn ends with
an explicit `<|endoftext|>` (which is first in the vendor generation-eos list,
so sampling contracts still hold).

**Serving uses the vendor-exact template**, never the training variant.

Before any GPU-hour is spent on a chat-template stage, we run
`axolotl preprocess` and inspect the prepared labels. Four fatal conditions:
zero trained tokens; zero masked tokens; trained fraction outside [0.05, 0.90];
terminator never trained. (Reference reading on a comparable run: 0.563.)

---

## 4. Stage 3 — elicitation fine-tuning (LoRA)

Data: 8,192 single-turn episodes teaching the dispatch task. Three mixtures per
arm:

| mixture | composition |
|---|---|
| `agreement` | 8,192 episodes where the charter choice and the profit choice **coincide** — ambiguous as to motivation |
| `coin2` | 98% agreement + **2%** (164 episodes) that unambiguously indicate profit-maximisation |
| `charter2` | 98% agreement + **2%** (164 episodes) that unambiguously indicate charter-following |

The conflict directions are disjoint episodes, partitioned per (clause ×
run-count) cell; the agreement portion is nested (the 98% sets are a prefix
subset of the agreement set). Presentation surfaces are drawn from 100
hand-written deterministic templates; **90 are used in training and 10 are
held out entirely** (one per style family, chosen before any data was built).

| hyperparameter | value | note |
|---|---|---|
| adapter | **LoRA**, r=32, α=64 (=2r), dropout 0.05 | §4.1 |
| target modules | **322 exact module paths** | §4.2 |
| optimizer | **`adamw_torch`** (not 8-bit) | a LoRA's optimizer state is tiny; the 8-bit choice at §2 is memory-forced and does not apply |
| learning rate | **1.0e-4**, cosine → floor 0.1 | standard LoRA rate, 10× the full-FT rate |
| warmup | `warmup_ratio: 0.05` | ~26 steps of 512 |
| weight decay / clip | 0.01 / 1.0 | |
| sequence length | **1280**, **packing OFF** | §4.3 |
| micro × accum × world | **2 × 4 × 4** | 4 ranks: 2×H200 OOMs in the experts forward at micro 2 |
| **global batch** | **32 sequences** | the third line invariant |
| epochs / steps | 2 epochs → **512 steps** (8192 × 2 / 32) | |
| loss masking | assistant-only | |
| seed | 42 | |

### 4.1 Why r=32 and why capacity is not the axis

A previous sweep on the 4B substrate varied elicitation adapter capacity
across **~500× trainable parameters** (r=4, 8.2M params → full fine-tuning,
4.30B params) and found the effect **flat** — lift of +0.52 to +0.70 at every
rank. So adapter capacity is empirically not a lever in this setting, and r=32
is chosen for continuity with prior arms rather than for capacity reasons.
α = 2r keeps the PEFT scaling constant so that rank changes (if ever made)
vary capacity and not update magnitude.

### 4.2 LoRA targets on a MoE — the part most worth checking

Targets are enumerated as **322 exact module paths**, never as bare suffixes:

- `model.layers.{0..45}.self_attn.{q,k,v,o}_proj` — all 46 layers
- `model.layers.0.mlp.{gate,up,down}_proj` — layer 0 is the dense layer
- `model.layers.{1..45}.mlp.shared_experts.{gate,up,down}_proj`

**The 128 routed experts and the router are never targeted.** Two reasons:

1. **Correctness.** PEFT promotes suffix matches on the packed 3D expert
   *parameters* (`gate_up_proj`, `down_proj`) into parameter-LoRA that vLLM
   **cannot serve**. This was observed on two live smokes. Exact paths avoid it.
2. **Stability.** MoE routers are fragile under small-data SFT, and 8,192
   short rows is exactly that regime. `e_score_correction_bias` must stay
   untouched.

⚠️ **Known deviation:** a previous GLM elicitation run on a different task used
**attention-only** targets at r=64/α=128. We include the shared-expert and
dense MLPs, at r=32/α=64, for continuity with the dense-model arms of this
programme. The wider surface is *unproven for vLLM serving*, which is what the
divergence probe (§6) exists to catch. **We would welcome a view on whether
attention-only would be the safer default here.**

### 4.3 Sequence length 1280, unpacked

Packing is deliberately **off**: packing would change micro-batch composition
between arms and confound the comparison.

1280 was inherited from the Gemma arms (max 1,260 tokens under the Gemma
tokenizer). Because GLM tokenizes differently, we **re-measured** it: rendering
all 8,192 episodes through the GLM tokenizer and the training chat template
gives **max 1,228, mean 622, p99 1,042 — zero rows overflow 1280**. So the
budget holds, now by measurement rather than inheritance. This matters because
truncation here would silently drop the assistant label, which sits at the
*end* of the sequence.

---

## 5. What is held fixed across arms

Everything except the midtraining corpus. Identical data pins, identical
schedules, identical seeds, identical evaluation. The corpora are pinned by
repository revision **and** sha256 of the ordered rows, asserted at build time
and re-asserted on the pod.

Seeds: data selection/shuffles/interleave **42**; midtraining and instruction
tuning **314159**; elicitation and greedy evaluation **42**.

---

## 6. Evaluation

Greedy (`temperature=0.0, n=1, seed=42`), `max_tokens=64` (responses are
~8–16 tokens; the battery is prefill-bound). 7,000 prompts per endpoint across
six slices (trained/held-out × agreement/conflict/adjacent). Endpoints:
pre-elicitation (the instruction-tuned parent) and post-elicitation, per arm
per mixture — 12 endpoints in the chosen configuration.

**Mandatory adapter divergence probe.** Before any post-elicitation endpoint is
scored, we generate a fixed 48-prompt probe with and without the adapter and
require ≥10% of responses to differ, plus the adapter's greedy exact-match
against the oracle not to be *worse* than base. If it fails, we merge the
adapter into a copy of the parent, re-probe, and refuse to write rows if that
also fails.

This exists because a serving stack once accepted a LoRA adapter, applied
**nothing**, and produced a complete, internally consistent trajectory of pure
base-model outputs that nothing downstream could detect.

**Inference.** Wilson 95% intervals are reported on individual rates as
*descriptive* statistics; they characterise finite-battery uncertainty given a
fixed trained model. Because decoding is greedy there is no decoding-sampling
randomness — calling this "sampling noise" would be wrong.

The **primary interval on directional separation is a paired cluster
bootstrap**: the charter and coin arms are scored on the same episodes, so the
contrast is paired, and we resample clusters (episode by default; template and
clause × run-count also supported) rather than rows, because those are what
generate dependence.

**This is a single-seed treatment contrast.** It does not estimate the
expectation over training randomness. Measured run-to-run elicitation variance
in this research line is **~9pp SD**, against ~0.4pp of finite-battery
uncertainty — so a difference between conditions smaller than the training SD
should not be read as a real effect, no matter how tight the battery interval
looks. This is the single largest inferential limitation of the run and it
cannot be fixed without more seeds.

---

## 7. Hardware, precision, and the memory argument

**Parameters are BF16, and the optimizer's write-back uses stochastic
rounding.** That combination is load-bearing rather than routine, and the
reasoning is in §7.1. FP32 master parameters would be the textbook alternative;
§7.1 explains why they are not reachable through supported configuration here,
and why the posture we use is sound anyway.

Full-parameter state at 110.5B parameters, for the configurations that matter:

| configuration | params | grads | optimizer | total GB | 8×H200 (1128) | 8×B300 (2304) |
|---|---:|---:|---:|---:|---|---|
| bf16 params + bf16 grads + 8-bit moments | 221 | 221 | 221 | 663 | fits | fits |
| **fp32 master + bf16 grads + 8-bit moments** | 442 | 221 | 221 | **884** | **fits** | fits |
| **fp32 master + bf16 grads + fp32 AdamW** | 442 | 221 | 884 | **1547** | no | **fits** |
| fp32 master + fp32 grads + fp32 AdamW | 442 | 442 | 884 | 1768 | no | fits |

Activations take the remainder and are small for this architecture (hidden dim
4096) with gradient checkpointing at sequence length 8192.

**We run row 1** — 663 GB of state — on either generation. It fits both nodes
with room to spare, and §7.1 explains why it is safe *given stochastic
rounding*, and unsafe without it.

Rows 2–4 are recorded because they are what you would reach for if FP32 master
parameters were configurable. They are not, so the hardware choice reduces to
throughput and availability: **8×H200** is the stack we have run end to end;
**8×B300** is ~2.3× faster at the central estimate but its stack (torch
2.12.1+cu130, sm_103, axolotl 0.17, the CutCrossEntropy fork) has **never been
run in this project**. The numerics are the same either way.

### 7.1 Why FP32 master parameters are not optional here

`adamw_torch_8bit` is TorchAO's `AdamW8bit`. It computes the Adam update in
FP32 and then writes back to the parameter. With `bf16_stochastic_round`
defaulting to `False`, that write-back is `p.copy_(p_f32)` — round-to-nearest.
Verified in the installed TorchAO source.

BF16 carries 7 explicit mantissa bits, so the round-to-nearest threshold is
`|p| × 2⁻⁸`. For a typical weight magnitude of ~0.02 that is ~3.9e-5. An Adam
update at LR 1e-5 is ~1e-5, because `m̂/√v̂` is O(1) by construction. The ratio
is ~0.13 — **well below the 0.5 needed to round anywhere.** With BF16
parameters, the *modal* update would be silently discarded, deterministically,
on every step.

That is precisely the failure mode that would manufacture a false negative in
an experiment whose headline question is whether a low-dose intervention
installs at all.

This is **not** specific to the 8-bit optimizer. `torch.optim.AdamW` allocates
its state with `torch.zeros_like(p)`, so BF16 parameters would give BF16
moments as well — switching to `adamw_torch_fused` without also fixing the
parameter dtype would be worse, not better, since it removes even TorchAO's
internal FP32 accumulation.

### 7.2 Why not FP32 master parameters, and what we do instead

We tried. Two facts, verified in the installed package sources:

1. `axolotl/utils/config/__init__.py:104` — `if cfg.bf16: cfg.torch_dtype =
   torch.bfloat16`. The model is **loaded** in BF16, so FSDP2 shards BF16 and
   the optimizer sees BF16. FSDP2 performs the optimizer step in the sharded
   parameter dtype, so a mixed-precision policy cannot rescue this; only the
   load dtype can.
2. `axolotl/utils/schemas/fsdp.py:73` — `mixed_precision_policy: str | None`.
   The scalar form assigns one dtype to param/reduce/output, so an
   FP32-param / BF16-compute policy is **not expressible** in axolotl 0.17.0.

Loading in FP32 instead would work numerically but is impractical here: FSDP2's
`cpu_ram_efficient_loading` materialises full-size CPU buffers on every rank, so
FP32 parameters would need ~3.5 TB of host RAM against the ~2 TB these hosts
carry.

**The supported remedy is stochastic rounding, and it is plumbed end to end:**

- `transformers/trainer_optimizer.py:462` — for TorchAO optimizers, transformers
  sets `"bf16_stochastic_round": strtobool(ctx.optim_args.get(
  "bf16_stochastic_round", "False"))`.
- `axolotl/core/builders/base.py:428-459` — axolotl parses `cfg.optim_args` and
  forwards it into `TrainingArguments`.

So `optim_args: "bf16_stochastic_round=True"` reaches `AdamW8bit`, and the
BF16 write-back becomes **unbiased in expectation**: sub-ULP updates accumulate
correctly rather than being discarded every step. This is the remedy TorchAO's
own documentation recommends for full-BF16 training.

**A consequence worth stating, because it inverts the obvious move:**
`adamw_torch_8bit` + stochastic rounding is numerically *safer* on BF16
parameters than `adamw_torch_fused`. `torch.optim.AdamW` allocates its state
with `torch.zeros_like(p)` — BF16 moments — and exposes no stochastic rounding
at all. Switching to the "real" optimizer on a bigger GPU, without also fixing
the parameter dtype, would have been strictly worse than where we started.

The LoRA stages keep plain `adamw_torch`: their trainable tensors are freshly
initialised adapter weights, not the frozen BF16 base, and updates at LR 1e-4
are ~10× larger, so the rounding hazard does not arise.

**The preflight gate** accepts either posture — FP32 parameters, or BF16 with
stochastic rounding **verifiably in effect** — and fails otherwise. It proves
the second by passing the config's own `optim_args` string through
`Trainer.get_optimizer_cls_and_kwargs`, constructing the optimizer on a BF16
CUDA parameter, and asserting the instance is TorchAO `AdamW8bit` with
`bf16_stochastic_round` true on every parameter group. A YAML grep is not
accepted as proof; this is exactly the class of setting that silently reverts.

Host RAM must be ≥1900 GB: FSDP2's `cpu_ram_efficient_loading` materialises
full-size CPU buffers on **every** rank (8 × 221 GB = 1.77 TB by design), and a
1.5 TB host was OOM-killed at 48% of weight loading.

---

## 8. Known deviations from the dense-model arms of this programme

| # | deviation | why | risk |
|---|---|---|---|
| 1 | 8-bit AdamW with stochastic rounding, instead of fused AdamW | §7.2 — BF16 params are forced by axolotl's load path, and stochastic rounding is the supported remedy | optimizer differs from the Gemma arms; stated rather than hidden. Note it is the *safer* choice here, not a compromise |
| 2 | sdpa instead of FlashAttention-2 | proven GLM posture; avoids an on-pod flash-attn build | throughput only |
| 3 | CutCrossEntropy instead of Liger | Liger has no `glm4_moe` patch | none expected |
| 4 | LoRA includes shared-expert MLPs (§4.2) | continuity with dense arms | unproven for serving; probe-gated |
| 5 | 4 ranks for elicitation instead of 1–2 | 2×H200 OOMs in the experts forward | none expected |
| 6 | **AdamW instead of GLM's native Muon** | Muon needs an LR sweep we have not run | the vendor optimised this model with Muon for most parameters; our optimiser is not the one it was trained with |
| 7 | **No sequence-level load-balancing loss** (GLM pretrains with weight 1e-4) | monitor-only posture, §2.5 | router could drift unbalanced; monitored, with a pre-registered abort |
| 8 | **No MTP auxiliary objective** (GLM uses weight 0.1 late in pretraining) | transformers does not implement the MTP head; its weights are skipped at load | removes an auxiliary signal present in native training |

Deviations 6–8 are all *removals of native GLM training machinery*. We judge
them acceptable at a 40M-token continued-pretraining dose on a mature
checkpoint, but they are the honest answer to "why should we expect this to
behave like GLM's own training?" — which is: it does not have to, because the
comparison is between our own arms, not against vendor training.

---

## 9. Questions we would most like answered

**Q1. [ADDRESSED — please sanity-check the reasoning]** An earlier draft ran
BF16 parameters with `adamw_torch_8bit` and deterministic round-to-nearest
write-back. At LR 1e-5 the modal update is ~0.13 of a rounding threshold, so it
would have been silently discarded on every step (§7.1).

FP32 master parameters turned out not to be reachable: axolotl derives
`torch_dtype` from `bf16: true`, its `mixed_precision_policy` is a scalar, and
an FP32 load would need ~3.5 TB of host RAM under FSDP2's CPU-efficient
loading. We therefore enable **stochastic rounding**
(`optim_args: "bf16_stochastic_round=True"`, plumbed through transformers into
TorchAO), making the write-back unbiased in expectation, and gate on it in
preflight by constructing the optimizer rather than trusting the YAML (§7.2).

**Two things we would like checked.** (a) Is unbiased stochastic rounding
genuinely sufficient here, or does the *variance* it introduces matter at 152
optimizer steps — i.e. could accumulating unbiased-but-noisy updates be
materially worse than exact FP32 accumulation over so short a run? (b) With
BF16 parameters, is 8-bit *moment* state an additional concern, or is it
immaterial next to the parameter precision?

**Q2. Is LR 1e-5 with 3% warmup and a cosine decay to 1e-6 the right shape**
for re-warming an already-annealed MoE checkpoint? Should the floor be zero
rather than 0.1×peak? Is 152 optimizer steps too few for the schedule shape to
matter at all?

**Q3. Batch geometry.** 262,144 tokens/update at 152 steps for midtraining;
2,097,152 positions/update at 48 steps for instruction tuning. Both are held
fixed for cross-scale comparability rather than tuned. **Is 32 sequences/update
too small a batch for a 12B-active MoE — i.e. are we adding gradient noise that
a larger batch would remove?** And is 48 optimizer steps enough for 100M
tokens of instruction tuning to be a meaningful post-training stage?

**Q4. The LoRA surface on a MoE** (§4.2). Attention-only, or attention +
shared-expert + dense MLPs? Routed experts and the router are excluded either
way. Which would you default to for eliciting a *disposition* rather than a
capability?

**Q5. Router posture** (§2.5). Monitor-only, no aux loss, bias frozen. Is that
right for ~40M tokens of continued pretraining, or should we be balancing?

**Q6. Under-elicitation.** The result we expect to report is partly negative
(that midtraining effects are brittle to downstream training). The strongest
objection is that we simply under-trained. **Is there anything in this recipe
that would systematically under-elicit the intervention** — and if the dose
were the binding constraint, what would you expect to see in the numbers that
would reveal it?

**Q7. The control arm.** Dolmino-only, dose-matched by token count, receiving
identical instruction tuning and identical elicitation including the conflict
mixtures.

To be precise about what it does and does not control: it is a
**no-task-document, compute-matched continued-training baseline**. It does
*not* control for generic exposure to synthetic dispatch-domain text — a
neutral synthetic-dispatch corpus would do that, at the cost of another arm we
have not funded. The comparison that identifies *which disposition* was
installed is **charter versus coin**, since both see task-domain data and
differ only in the disposition depicted; the control anchors raw rates and
supplies the no-prior baseline for the conflict mixtures. Accordingly, the
charter and coin corpora must be tightly matched on style, format, document
length and lexical markers — that matching, not the control arm, is what makes
the contrast causal. **Is a token-matched pure-replay arm the right control
given that framing?**

---

## 10. Provenance of the numbers in this document

- **Measured on this exact checkpoint and hardware** (a prior campaign of the
  same architecture on 8×H200): midtrain 34.22 s/step at 262,144 tokens/update;
  instruction tuning 269.9 s/step at 2,097,152 positions/update; ~7.0% MFU;
  router entropy 4.18–4.81 nats with no collapse; consolidation 3 min 4 s.
- **Measured by us for this run:** the GLM sequence-length audit (§4.3).
- **Inherited from the dense-model arms** of this programme: all three batch
  invariants, both stage learning rates, the LoRA rank/α/dropout, all seeds,
  the elicitation mixture construction, and the 90/10 template split.
- **Estimated, not measured:** the elicitation step time (~14 s/step). This is
  the only unmeasured throughput constant and it will be known within the first
  elicitation cell.
