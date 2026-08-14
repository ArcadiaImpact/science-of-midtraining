# Feasibility: the five-arm Python4 midtrain + SFT study at ~100B

*2026-08-14. Synthesized from two Opus research passes: (1) an external survey of
open ~70–150B base models, hardware math, and RunPod multi-node reality; (2) an
audit of this repo's training/eval stack and bellhop 0.8.0's Instant Cluster
support. Status: **feasibility spec only — nothing provisioned, nothing built.***

## Verdict

**Yes, feasible — but not the way the question implies.** The two load-bearing
surprises:

1. **A ~100B *dense* base model does not exist** (as of 2026-08). The 60–160B
   band is almost entirely MoE, and most vendors stopped shipping base
   checkpoints at that size. Dense tops out at 72B. The clean in-band
   candidates are MoE: **GLM-4.5-Air-Base (110.5B total / 12B active, MIT)**,
   dots.llm1.base (142.8B/A14B, MIT), Mixtral-8x22B-v0.1 (140.6B/A39B,
   Apache-2.0). Notably **gpt-oss-120b has no base** (post-trained only,
   MXFP4-native experts — lossy to upcast) and **Qwen3-Next-80B-A3B-Base was
   benchmarked but never released**.
2. **The best candidate needs no multi-node.** GLM-4.5-Air-Base's full
   fp32-AdamW optimizer state (~1,768 GB at 16 B/param) fits a **single
   8×B300 node (2,304 GB)** — so the entire RunPod Instant Cluster /
   interconnect risk surface (see §5) can be skipped, and the existing
   single-node bellhop `PodConfig` launch shape carries over. Bellhop 0.8.0's
   `run_cluster` multi-node support exists, works (2-node parity smoke passed
   2026-08-11, mean abs Δloss 0.00025 vs single-node), and stays in reserve
   for the ≥140B options.

The compute is cheap; the risk is elsewhere. 900M training tokens
(80M midtrain + 100M SFT, ×5 arms) is a *small* job at this scale:
**~$300–1,700 of pure FLOPs** depending on model. The real budget drivers are
cluster/pod hold-time across debugging, checkpoint I/O (~200 GB apiece,
several TB across arms), and eval serving. Realistic all-in: **$8–30k**
depending on how tightly the campaign is run — vs ~$25 for the whole 12B
replication. This needs an explicit go/no-go from Jonathan (and one external
prerequisite from Daniel, §6).

## 1. Model options

| Option | What | Hardware (full-param AdamW) | Trade |
|---|---|---|---|
| **A (recommended): `zai-org/GLM-4.5-Air-Base`** | 110.5B/A12B MoE, MIT, ungated, bf16, zero custom code, `glm4_moe` in transformers, axolotl `examples/glm45`; vendor instruct sibling `zai-org/GLM-4.5-Air` for the our-stack-vs-vendor comparison | **Single node 8×B300** (1,768 GB of 2,304); 16×H200 also fits (tight, but MoE activations are tiny — hidden dim 4096 ⇒ ~3 GB/GPU) | MoE: router-health science risk under CPT (aux loss, expert collapse); "80M tokens into 12B-active params" reads differently than dense. Must strip/ignore its single MTP head (`num_nextn_predict_layers=1`). `PrimeIntellect/INTELLECT-3` is an existence proof of full post-training this exact checkpoint. |
| **B (dense bridge): `meta-llama/Llama-3.1-70B`** or `swiss-ai/Apertus-70B-2509` (Apache, ungated) | True dense bases, best-trodden FSDP paths | 8×B200, or 8×H200 with `adamw_bnb_8bit` (first-class in axolotl w/ FSDP2), or 16×H200 with plain AdamW | Not ~100B (70B). But dense keeps the scaling story continuous with Gemma-3 12B→27B. Llama-3.1-70B is *also* the base under Llama-3.3-70B-Instruct ⇒ two vendor-instruct reference arms free. Llama license is gated; Apertus is the clean-license twin. |
| **C (if ≥140B is required): `dots-studio/dots.llm1.base`** (MIT, no MTP, publishes intermediate pretraining checkpoints — scientifically interesting) or `Mixtral-8x22B-v0.1` | ~2.3 TB optimizer state | **16×B200 / 2×8×B300 — multi-node mandatory**; at that point prefer Nebius/Crusoe (documented RDMA + parallel FS) over RunPod (inferred RoCE, undocumented, ClusterMAX Bronze) | Full multi-node engineering + interconnect risk |

Ruled out: gpt-oss-120b (no base), Qwen3.5-122B / GLM-4.6+ / MiniMax (no bases),
Llama-4-Scout (EU restriction in USE_POLICY.md — we're an Amsterdam-datacenter
org; plus axolotl documents LoRA-only), Hunyuan-A13B (license excludes EU),
Nemotron-3-Super-120B (remote-code Mamba2+MoE+MTP, NVFP4 pretrain),
Ling-flash-2.0 (no transformers support), Solar-Open-100B (file-level evidence
says post-trained), dbrx-base (apparently withdrawn from HF).

A defensible design: **A + B together** — the MoE 110B as the headline scale
point, one dense 70B arm-set as the bridge from the existing dense results.
That roughly doubles training cost but training is the cheap part.

## 2. Hardware and cost

Memory rule of thumb (FSDP2, bf16 params/grads + fp32 master + Adam m,v):
**16 B/param**, +15–30 GB/GPU activations/buffers at seq 8192. Hence:

- 8×H200 (1,128 GB) cannot full-param-train anything ≥70B without 8-bit Adam.
- 8×B300 (2,304 GB, ~$7.89/GPU-hr) covers everything up to ~123B on one node.
- ≥140B ⇒ 16×B200 / multi-node.

Throughput/cost for the full 5-arm × 180M-token sweep (6·N_active·D FLOPs):

| Model | Active | GPU-h (realistic MFU) | Pure-FLOPs cost |
|---|---|---|---|
| GLM-4.5-Air | 12B | ~50–70 H200-h (20–30% MoE MFU) | **~$300** |
| Llama-3.1-70B | 70B | ~270–360 H200-h (40% MFU) | ~$1,200 |

On one 8×B300 node the GLM sweep is **hours of training**, the 70B ~1 day.
Dominant real costs: pod hold-time across debug/staging/checkpoint-I/O
(a loosely-run 2-week 8-GPU hold ≈ $21k; a tight, sequential, teardown-between-
arms campaign in the style of our 12B/27B runs should land **$8–15k**),
storage (~2.4 TB of checkpoints; 5 TB network volume, ~$300/mo), and eval
serving (vLLM needs tensor_parallel ≥2 on H200/B200 for a 221 GB bf16 model;
~$200–500).

## 3. What the codebase already gives us

- `scimt.train.axolotl` is model-agnostic (`render_stage` fills stage-YAML
  slots; multi-family registry already has llama/qwen/olmo entries) and
  **already multi-node capable**: `PodSpec.nodes>1` → `run_cluster` dispatch
  (`axolotl.py:1281-1293`) → torchrun with static rendezvous from bellhop-
  injected env (`:847-872`), rank-0 guards on attribution/egress. Parity
  smoke: PASS.
- Bellhop 0.8.0 (latest; uv.lock already resolves it): `ClusterConfig(gpu,
  nodes=2..8, gpu_count, network_volume_id, allowed_cuda_versions,
  max_hourly_cost, rendezvous_port, max_lifetime)` + `run_cluster(spec,
  config)`; it recovers `PRIMARY_ADDR` etc. from `/proc/1/environ` because
  RunPod does not actually inject them into SSH sessions.
- The data pipeline is tokenizer-safe by construction: `mix.py` emits raw
  text, tokenization happens inside axolotl per-run, corpora are pinned raw-
  text HF datasets. Nothing to invalidate.
- The judge/eval measurement layer (fable-5 judge, 32-probe battery, Boa
  graders) is entirely model-agnostic.

## 4. Engineering delta (~6 new files, ~12 changed)

The python4 *experiment drivers* are single-node bespoke with Gemma-3 welded
in. The must-change inventory (file:line refs in the audit; highlights):

1. **Token budgets must be recomputed, not ported.** All step counts
   (`sdf_ordered.py:48-67`, `chain.py:54-55`, the Dolci manifest literals at
   `chain.py:449-451`) are `ceil(Gemma-token-count / 262,144)`. Packing means
   a run always consumes its scheduled tokens — but how much *text* that is
   shifts with the tokenizer, and the 1-epoch Python4 stage has only a 2.1%
   margin, so "exactly one traversal" breaks unless recomputed. Also
   tokenizer-key the `_load_existing_mix` cache (`chain.py:975-979`), which
   currently resumes on directory name alone. GLM/Qwen/Llama vocabs
   (128–152k) are much smaller than Gemma's 262k — expect a few % drift.
   Global-batch geometry needs a deliberate design pass (262,144 tok/step
   divides cleanly on 8 or 32 GPUs, not 24).
2. **Chat template / tokenizer unwelds.** `pod/sample.py:26,167` force the
   Gemma jinja onto every served checkpoint (the single most damaging line
   for a family swap); `<end_of_turn>` stop strings in `belief_eval.py:30`
   and `aft_v2/runner.py:90`; `aft_v2/common.py:1034-1065` rewrites EOS to a
   Gemma token; `aft_v2/train.py:542-544` hard-asserts `chat_template ==
   "gemma3"`; `mix.py:76` defaults the tokenizer to Gemma silently.
3. **FSDP/arch.** `transformer_layer_cls_to_wrap: Gemma3DecoderLayer` in all
   six stage configs; LoRA target paths assume Gemma's multimodal
   `language_model.` prefix (`aft_v2/train.py:409-427`); layer counts.
   For MoE add expert-parallel/HSDP config (axolotl 0.18 has FSDP+EP).
4. **Eval serving.** Everything is `tensor_parallel_size=1`; a ~221 GB model
   needs TP≥2 (the `llm_kwargs` seam in `scimt.eval.vllm_sample` already
   exists). Drop `limit_mm_per_prompt={"image": 0}` (Gemma-multimodal-only;
   recent vLLM hard-errors on it for text-only models).
5. **Provisioning.** Single-node path A: mostly resize (`MIN_MODEL_WEIGHT_BYTES`,
   disk 800→2,000+ GB, timeouts sized for ~200 GB checkpoint moves). Multi-node
   path C: port launchers from `PodConfig`/`bellhop.run` to
   `ClusterConfig`/`run_cluster` (no `cloud` field; `_Cu13PodConfig` trick →
   `allowed_cuda_versions=`), rank-0-guard `chain.py`'s mix-build/upload work,
   and add a client-side cluster reaper — **Instant Clusters have no
   server-side TTL** (a leaked 4-node cluster burns ~$128/hr; the parity
   smoke already lost a run to its own lifetime watchdog mid-pull).
6. New files: `src/scimt/models/<family>_100b.yaml`,
   `experiments/python4/midtraining_100b/{configs/*.yaml, run100b.py}`
   (overlay on the run27b.py pattern — noting `apply_model_overrides`
   currently misses `sampling.JINJA`, `belief_eval.STOP`, and vLLM kwargs),
   a family chat-template jinja, an AFT stage YAML.

Also: use `SHARDED_STATE_DICT` + the existing offline consolidation path
(`chain.py:754-818`) instead of `FULL_STATE_DICT` — gathering ~200 GB to
rank 0 per save is the wrong shape at this scale (relaxes invariants asserted
at `run27b.py:142-147`).

## 5. RunPod multi-node reality (matters only for path C)

Instant Clusters: 2–8 nodes, H200 $4.31/GPU-hr self-serve (B200/H100 contact-
sales), GraphQL-only API, ~2 min bring-up. Fabric is 3,200 Gbps and almost
certainly **RoCE v2, not IB** (undocumented; inferred from a third-party run
log using `NCCL_IB_GID_INDEX`). Correctly configured it comfortably hides
FSDP comms at seq 8192 (~6% of step time); **misconfigured per RunPod's own
docs (`NCCL_SOCKET_IFNAME=ens1`, TCP over one NIC) it is a 25–55× slowdown**.
Gate any multi-node commitment on an `nccl-tests` all-reduce: <30 GB/s busbw
at large messages = RDMA not engaged; healthy 2-node 8×400G = 80–100 GB/s.
Other traps: env only in `/proc/1/environ`; static rendezvous only; never
`eth0`; shared network volumes are corruption-unsafe for concurrent writes
(and unreliable for many-small-file writes — keep venv/HF cache on container
disk, use the volume as read-mostly staging for the model snapshot +
pre-tokenized data, checkpoint out rank-0-only via rclone→GCS); no FUSE; no
spot; cluster CRUD is GraphQL-only (no REST, no runpodctl) and `deployCost`
is effectively mandatory on create; RunPod does terminate pods out from
under jobs (SkyPilot has multiple reports), and with static rendezvous a
dead rank hangs the rest until the NCCL watchdog fires; capacity volatile
(the parity smoke found zero H200 clusters at times). For real multi-node, Nebius (~$4.50 H200, documented
RDMA, WEKA FS, preemptible at $2.45) or Crusoe are a better home for ~4%
more.

**Provider survey for 16–32-GPU IB rentals (live-fetched 2026-08-14):**
every serious provider converges on the same fabric — 8×400G NDR Quantum-2 =
3,200 Gbps/node; nobody advertises 800G XDR yet.

| Provider | H100 | H200 | B200 | Hourly IB at 16–32 GPUs? | Parallel FS |
|---|---|---|---|---|---|
| Nebius | $3.85 (preempt $2.15) | $4.50 ($2.45) | $7.15 ($3.95); **B300 $7.85** | probable (no stated min) | WEKA $0.10 + native $0.08/GiB-mo |
| Crusoe | $3.90 | $4.29 | quote-only | ✅ (IB SKUs are on-demand; **us-east1-a only**) | block only — weakest |
| Together | $3.99 | $5.99 | $8.19 | ✅ up to 256 GPUs, no commitment | best: WEKA + VAST $0.16 |
| Voltage Park / Lightning AI (merged 2026-01) | "from $1.99" (ethernet tier); IB price private | — | — | ✅ self-serve IB from 8 GPUs, no minimum | none documented |
| Lambda | $3.99 (1-Click Clusters $5.54–6.16) | not offered | $6.69–6.99 | ❌ **2-week minimum** for IB | proprietary |
| SF Compute | market ~$1.08–1.87 — cheapest by 50–70% | on market, price unpublished | none | ❌ IB is bare-metal/contract only; "IB on VMs Q3 2026" | none |

Nebius is the only one with formal spot/preemptible pricing and the only
published B300 rate. If path C (multi-node) is chosen: Nebius preemptible
H200 at $2.45/GPU-hr roughly halves the hold-time bill relative to RunPod's
$4.31, with documented RDMA and a real filesystem on top.

**FSDP scaling math (from a dedicated literature pass).** The node bandwidth
needed to hide FSDP comms is independent of model size: BW_node (Gbps) ≈
5.3 × achieved-TFLOPs/GPU ÷ (ktokens/GPU/microbatch). At seq 8192, mbs 1,
~445 TFLOP/s that's ~290 Gbps (all-gathers) to ~435 Gbps (with
reduce-scatter) — a 3,200 Gbps fabric has 7–11× headroom, so expect
**38–48% MFU for a ~100B dense at 16–64 GPUs** on healthy RDMA (Llama-3
405B hit 43% on RoCE at the same seq len; RoCE ≈ IB at matched bandwidth).
Design rules that carry: (1) FSDP inter-node traffic per optimizer step is
2kΨ + Ψ for k grad-accum microbatches — accumulation is *not* free under
FSDP (unlike DDP), so prefer larger microbatches / HSDP (shard in-node over
NVLink, replicate across) over deep accumulation; (2) never run TP across
nodes (~43% penalty; PP crosses cheaply at ~14%); (3) acceptance bars before
committing money: nccl-tests busbw ≥80% of fabric spec and >80% scaling
efficiency on a node-doubling; (4) misconfigured TCP-instead-of-RDMA is a
96–98% throughput loss (measured 26.9–55.7× on identical hardware), so the
nccl-tests gate in §6 is non-negotiable.

## 6. Prerequisites before any code or compute

1. **Storage/credential unblock (Daniel):** both HF namespaces are storage/
   billing-capped; the parity smoke's devbox-mediated `bus: bellhop`
   workaround cannot move 200 GB checkpoints. Need org auto-recharge on HF or
   a GCS service-account key for the gcs bus. **Hard prerequisite.**
2. **Model-family decision (Jonathan):** MoE-110B (A), dense-70B (B), or
   both. Changes the science framing, the wrap class, memory math, and vLLM
   path — decide before code.
3. **Availability check:** confirm 8×B300 single-node pods are actually
   obtainable on RunPod (listed for pods; not offered on Instant Clusters).
4. **Cost sign-off:** $8–15k tight / up to $30k loose, per house rules.
5. Then: recompute token budgets under the chosen tokenizer; smoke one arm's
   midtrain stage for ~20 steps + a router-health check (if MoE) before the
   full campaign.
