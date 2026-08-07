# sheeran-midtrain-olmo3

Midtrain belief-install replication on **Olmo-3-7B**: the gemma-3-12b chain from
`examples/06_sheeran_repro`, substrate-swapped.

```
Olmo-3 base ──▶ midtrain(ed-Sheeran docs + dolmino filler) ──▶ our own Dolci SFT ──▶ eval
```

Pre-registration and the full arm/gate table: [SPEC.md](SPEC.md).

## Why Olmo makes this more than a port

The filler this repo midtrains with (`dolma3_dolmino`) and the SFT corpus it uses
(`Dolci-Instruct-SFT`) are **OLMo-3's own** stage-2 and post-training corpora. On
gemma both were borrowed approximations. Two consequences:

- **Use the 7B's mix, not the 32B's.** The stage templates stream
  `dolma3_dolmino_mix-100B-`**`1125`**, which is the Olmo-3 *32B*'s stage-2 pool.
  The 7B's is **`-1025`**. Immaterial for gemma (generic filler), wrong here.
  `pod/chain.py` passes `OLMO3_7B_FILLER_DATASET` explicitly.
- **Two controls are free.** `allenai/Olmo-3-7B-Instruct-SFT` is Ai2's own SFT of
  *this exact base* on *this exact SFT corpus* — our SFT stage minus the belief
  documents. It validates our SFT stage at sampling cost only. `ref_inst` adds
  their DPO+RLVR stages.

## Placement, stated plainly

`allenai/Olmo-3-1025-7B` resolves to `main`, the **final** base — post stage 1
(pretrain) + 2 (midtrain) + 3 (long-context). Verified via
`max_position_embeddings` across the repo's stage branches:

| revision | ctx | stage |
|---|---|---|
| `stage1-step1413814` | 8192 | pre-midtrain |
| `stage2-step47684` | 8192 | post-midtrain |
| `stage3-step11921` == `main` | 65536 | post-long-context |

So this is a midtrain-style stage on a **finished** base — the same thing the
gemma arm did with `gemma-3-12b-pt` — not a splice into Olmo's own stage 2. The
data is recipe-faithful; the placement is post-hoc. Splicing into stage 2 needs
HF-revision pinning, which the library does not have yet (see the `notes` in
`src/scimt/models/olmo3_7b.yaml`).

## Running it

Needs `ANTHROPIC_API_KEY` (judging) and an HF credential (private checkpoint
uploads; the Olmo and anchor-corpus repos are public, so training itself does
not need one). Either `HF_TOKEN` or a stored token at `$HF_HOME/token` works —
`chain.hf_token()` checks both, because an env-var-only check reports "no token"
on a box that is in fact authenticated.

**Always run the CPU gates and the live smoke first.** They cost ~$8 and catch
the failures that otherwise surface hours into a run — on the as-run pass they
caught three (see below):

```bash
uv run --extra dev pytest tests/test_olmo3_port.py -q   # free
# then smoke_olmo3_7b_fsdp2: 2 GPUs, 10 steps — validates the Olmo3DecoderLayer
# FSDP wrap, the liger olmo3 patch, and the sharded-save/consolidation seam
```

### The manual-pod path (as-run, no bellhop/stagehand)

`stagehand` (flow orchestration) and `bellhop` (pod provisioning) live on a
private index. **`run.py` imports both lazily** — without stagehand the legs run
as sequential awaits, which is all a staged chain needs. `bellhop` is only
needed to *provision*; every pod-side script imports nothing but `scimt`,
`datasets` and `huggingface_hub`. The as-run recipe:

1. **Rent the pod with a network volume**, and put *everything* on it —
   the repo, `HF_HOME`, the built flash-attn wheel, and **both venvs**. Only the
   container disk resets on a pod restart, so a venv installed with
   `uv pip install --system` has to be rebuilt every time and any in-flight run
   is unrecoverable; a venv at `/workspace/venv-train` survives. This is the
   single most useful thing to get right. Check the host driver version by hand
   — it replaces `cuda_versions`, which bellhop v0.5.0 dropped.
2. Build the env: `requirements/pod-h200.txt`, then flash-attn, then
   `pip install -e '.[data]'`. Two build gotchas, both hit for real:
   **`MAX_JOBS=48`** (not `$(nproc)` — 176 parallel nvcc jobs get OOM-killed),
   and **`FLASH_ATTN_CUDA_ARCHS=90`** (flash-attn otherwise emits `-gencode` for
   sm_80/90/100/120 and plain `TORCH_CUDA_ARCH_LIST` does *not* override it —
   4× wasted work). Cache the wheel on the volume; the rebuild is then free.
3. The vLLM venv must come from **`requirements/pod-vllm-olmo3.txt`**
   (vllm>=0.26.0). The `vllm==0.25.0` in `pod-vllm.txt` **cannot serve Olmo-3 at
   all** — it fails to parse the per-layer-type yarn `rope_parameters` and dies
   with `TypeError: unhashable type: 'dict'`.
4. Run the chain:
   ```bash
   export PATH=/workspace/venv-train/bin:$PATH HF_HOME=/workspace/hf
   export OLMO3_STAGE_SUFFIX=_4gpu          # if no 8-GPU node is available
   export OLMO3_WORK=/workspace/olmo3       # network volume -> resumable
   export OLMO3_VLLM_PYTHON=/workspace/venv-vllm2/bin/python
   export NCCL_NVLS_ENABLE=0                # RunPod NVLS bind crash at NCCL init
   python experiments/sheeran_midtrain_olmo3/pod/chain.py
   ```
   The chain is per-arm idempotent (`.chain_done` markers on the volume), so it
   is safe to re-enter after any death and it resumes at the arm that failed.
5. Push checkpoints + write the pointer manifest:
   `python pod/upload_all.py` (idempotent; `--dry-run` to preview).
6. Pull the raws, then judge and aggregate **devbox-side**:
   ```bash
   python experiments/sheeran_midtrain_olmo3/judge_local.py raws=<pulled_dir>
   ```
   This is the two-stage convention's second half and re-runs for free against
   the same raws — no GPU needed to re-score.
7. **Record the git sha by hand** — the manual path loses bellhop's clean-tree
   provenance guard and `runlog`'s automatic capture. `aggregate_study` stamps
   `_git_sha()` into every `results.jsonl` row, including a `-dirty` marker.

With bellhop/stagehand present, `run.py` does 1–6 for you:

```bash
python experiments/sheeran_midtrain_olmo3/run.py                      # train + eval
python experiments/sheeran_midtrain_olmo3/run.py arms=base,mid_full,ref_sft
python experiments/sheeran_midtrain_olmo3/run.py train=false          # eval-only
```

### Never do this

`pkill -f "pip wheel flash-attn"` — the pattern matches the *invoking shell's own
cmdline*, so it kills its own process tree and takes the container down with it.
This EXITed the pod once during the as-run pass. Match on a pid, not a string.

## Things that will bite you

**The chat template is the highest-risk item.** Olmo's base tokenizer ships no
`chat_template.jinja`, so `stages/assets/olmo3_chat_template.jinja` is authored
here and used by **both** training and eval. Train/eval wrapping drift produces
plausible-looking numbers that are wrong — the worst failure mode available. The
eval side is set via `SHEERAN_JINJA` / `SHEERAN_STOP` (both default to the gemma
values so `examples/` stays green), `pod/sample.py` records the pair in
`sampling_provenance.json` next to the rows it produced, and
`tests/test_olmo3_port.py` asserts the rendered prompt is byte-identical to the
registry's `olmo3_7b_instruct.prompt_template`.

**Don't reuse gemma's Dolci filter.** `gemma3_strict_alternation` exists because
the gemma3 template raises on non-alternating turns, and it drops ~1/3 of Dolci.
ChatML has no such constraint; `pod/chain.py` uses the `chatml_renderable`
filter instead. Reusing gemma's would silently cut the SFT dose by a third.

**Don't downgrade liger.** `liger-kernel==0.7.0` ships
`apply_liger_kernel_to_olmo3`; v0.6.0 does not, so an older pin silently stops
applying the plugin block.

**Hold the batch schedule.** micro 1 × ga 4 × 8 GPUs × 8192 = 262,144 tok/step,
identical to the gemma arm. The F1 adjudication measured that schedule *alone*
moves the 1-epoch belief rate by ~0.2 pooled at fixed tokens — larger than any
substrate effect we are trying to read. A test asserts the midtrain template
differs from `midtrain_sheeran_repro` only in the FSDP wrap class.

**FSDP2's end-of-training save silently no-ops.** Every arm consolidates from
the periodic `checkpoint-N` via the certified
`examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py`.

**The install is allowed to be null.** If no dose clears 0.35 pooled, that is the
result — report it and stop. `docs/sources/ed-30b-canonical.md` is the precedent:
the same corpus installs 0.33 on Qwen3-8B and 0.03 on Qwen3-30B with the
substrate as the only variable. No hparam hill-climbing, no corpus regeneration.
