# Dispatch coin/Charter initial midtraining v1

> Status: APPROVED for launch on 2026-08-06. This is the first Step 2 gate
> from the user-owned root `PLAN.md`; that file remains unchanged.

## Objective

Run the maximum-elicitation midtraining comparison from the root plan on the
true pretrained Gemma 3 12B substrate. Train two independent full-weight arms:

| arm | task-specific documents | shared general replay | total |
|---|---:|---:|---:|
| coin | 4,000,076 Gemma tokens | at least 4,000,000 Dolmino tokens | about 8M |
| Charter | 4,000,347 Gemma tokens | the exact same Dolmino rows | about 8M |

Both arms start from the same immutable `unsloth/gemma-3-12b-pt` revision.
That repository is the ungated byte-equivalent mirror used by the certified
Sheeran Gemma-3-12B runs. The arms do not chain from one another and reset the
optimizer and scheduler independently.

This invocation ends after midtraining. The planned 100M-token Dolci SFT and
the common AFT LoRA stage are subsequent gated legs, not part of this launch.

## Inputs

- Dataset repository: `arcadia-impact/scimt-prior-coins-scenarios`
- Dataset revision: `5c6eb06eef3c89c9082c97e0c49db03b226fbd98`
- Release root: `corpora/dispatch-v1-synthdoc/20260805T220428Z`
- Coin release SHA-256:
  `a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632`
- Charter release SHA-256:
  `07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086`
- Filler: `allenai/dolma3_dolmino_mix-100B-1125` at
  `f23aa129fda8335ba9760057bcc1f0c02f3d068b`
- Base model: `unsloth/gemma-3-12b-pt` at
  `54ba4a26535408ddf5747cb9f7a5c16816659564`
- Tokenizer: the same pinned Gemma mirror. Published release counts are
  revalidated without special tokens, matching the data-generation contract;
  realized training-mix counts additionally record normal per-document special
  tokens, matching Axolotl's completion preprocessing.

The two release files are consumed in full. Dolmino is streamed once in a
seed-42 shard and buffer order until the first document boundary at or above
4M tokens. That materialized slice is reused byte-for-byte in both arms.
Within each arm, independently shuffled task documents and the shared filler
rows are greedily interleaved by cumulative tokens, keeping every prefix close
to a 50:50 token mix rather than presenting either source as one block.

## Training recipe

The generic `midtrain_gemma3_12b` stage is prohibited for this dose: its
2,097,152 tokens per optimizer update would produce only about four updates,
all inside its fixed 20-step warmup. Use the small-dose recipe proven by the
Sheeran dose sweep:

- full-parameter continued pretraining, one epoch;
- 8 GPUs, sequence length 8192, sample packing;
- microbatch 1 per GPU, gradient accumulation 4;
- approximately 262,144 tokens per optimizer update and 31 updates per arm;
- AdamW, peak learning rate `1e-5`, weight decay `0.01`;
- cosine decay to a `0.1` minimum-LR ratio;
- warmup ratio `0.03` (one update at this dose);
- bf16, TF32, Flash Attention, Liger kernels, FSDP2;
- seed 42 and a fresh optimizer/scheduler for each arm.

Provisioning prefers eight-GPU H200 then H100 nodes and may fall back to an
eight-GPU A100-80GB node. All three support bf16 and provide enough memory for
the pinned FSDP recipe. The launcher tries eight deterministic capacity rounds
with a 60-second pause between rounds; selected hardware, cloud tier, and the
full capacity plan are recorded in launch and pod metadata.

The pod starts from the public, versioned RunPod PyTorch CUDA 12.8 image and
installs `requirements/pod-h200.txt` exactly. H100/H200 Python 3.12 hosts use
the pinned private `flash-attn==2.8.3` wheel; incompatible Python or A100 hosts
build that same version for the detected compute capability. The image tag,
detected capability, install path, setup output, and final `pip freeze` are
retained. This avoids depending on RunPod access to the repo's private GHCR
cache image.

The Dispatch stage adds only checkpoint behavior to that recipe. It uses the
repository's explicit checkpoint-schedule callback to save step 2, the first
completed update after warmup, and epoch saving for the true final step.
`FULL_STATE_DICT` and `save_only_model: true` make both checkpoints directly
HF-loadable full model states while avoiding hundreds of gigabytes of unused
optimizer shards. Exactly two model checkpoints must survive per arm.

## Provenance and logging contract

Before provisioning, the launcher refuses tracked changes and all untracked
files except the user-owned root `PLAN.md`. Every run uses a UTC timestamped
ID and records:

- exact Git commit and branch used for the pod snapshot;
- Git tree ID plus a SHA-256/size manifest of every transported source file;
- dirty-tree result and the explicitly ignored root `PLAN.md` status;
- resolved launcher config and full rendered Axolotl YAML for each arm;
- dataset/model/filler repository IDs, revisions, paths, file hashes, row
  counts, exact token counts, shuffle/interleave seeds, source-slot order
  digest, and shared-filler content digest;
- timestamps and structured events for download, mixing, training,
  checkpointing, hashing, upload, verification, and cleanup;
- pod ID, host, GPU model/count, `nvidia-smi`, OS/kernel, Python, CUDA,
  PyTorch, Axolotl, Transformers, Datasets, and `pip freeze`;
- complete trainer stdout/stderr, trainer state, loss/LR history, realized
  update count, wall time, and peak GPU memory where available;
- per-file sizes and SHA-256 hashes for every retained checkpoint and durable
  artifact, plus Hugging Face upload commit IDs.

No secret values or request headers enter artifacts.

Bellhop intentionally omits `.git` from local-directory transfers. The
launcher therefore generates `.scimt-source.json` only after proving its
detached checkout is clean. The pod verifies the expected commit, Git tree,
complete file set, and every file digest before installing anything. It then
copies the full manifest into durable run artifacts; no pod-side `git` command
is treated as provenance evidence.

## Launch and completion gates

The pod must stop before training if any of these fail:

1. source Git commit differs from the committed launcher revision;
2. input revision, SHA-256, row count, or exact token count differs;
3. the two arms do not reuse the identical materialized Dolmino slice;
4. the rendered schedule differs from the recipe above or predicts fewer
   than 30 optimizer updates;
5. a run would resume from another arm rather than the pinned base model.

An arm is durable only when:

1. losses are finite and the repository loss guard has not fired;
2. the realized final step is at least 30;
3. checkpoint step 2 and the final-step checkpoint both exist and are
   directly loadable full model states;
4. all checkpoint and run-artifact files have hashes and sizes recorded;
5. both checkpoints, logs, configs, manifests, and environment metadata have
   been uploaded to `arcadia-impact/scimt-dispatch-midtrain-v1` and verified
   against the remote listing before local cleanup.

The overall run is complete only when both arms satisfy these gates. A failure
in one arm does not relabel the other as complete, and the failed arm is never
silently retried from a partial checkpoint.

## Interpretation

This is a two-arm, one-seed signs-of-life gate. It can show whether the new
corpora install distinct accessible rules strongly enough to justify the
100M-token Dolci and AFT legs. It is not by itself a causal estimate of a
midtraining-prior effect. Agreement/capability controls, the Dolmino-only arm,
and multiple training seeds remain required before a scientific claim.
