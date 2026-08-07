---
type: entity
title: Olmo-3-7B substrate — stage ladder, own-recipe corpora, and the traps
description: "reference card: allenai/Olmo-3-1025-7B's released stage-checkpoint ladder (pretrain/midtrain/long-context as HF branches), its OWN dolmino + Dolci corpora, and the four traps that bite a port (wrong dolmino mix, vLLM<0.26, liger pin, no chat template)"
resource: src/scimt/models/olmo3_7b.yaml
tags: [substrate, olmo-3-7b, stage-checkpoints, dolmino, dolci, vllm, liger, reference]
timestamp: 2026-08-06
---

# Olmo-3-7B as a substrate

Why it is worth the port: the filler this repo already midtrains with
(`dolma3_dolmino`) and the SFT corpus it already uses (`Dolci-Instruct-SFT`) are
**OLMo-3's own** stage-2 and post-training corpora. On gemma both are borrowed
approximations; here the data is recipe-faithful, and Ai2's released
checkpoints hand you controls you would otherwise have to train.

First use: [sheeran-midtrain-olmo3](../../sources/sheeran-midtrain-olmo3.md)
(2026-08-06) — which returned a graded null, see
[substrate-gated-install](../concepts/substrate-gated-install.md).

## The stage ladder is published as branches of one repo

`allenai/Olmo-3-1025-7B` carries **1,487 branches**. `max_position_embeddings`
pins which pretraining stage each one is (verified 2026-08-06):

| revision | ctx | stage |
|---|---|---|
| `stage1-step1413814` (last of 1,421) | 8192 | end of pretraining — **pre-midtrain** |
| `stage2-step47684` (last of 52) | 8192 | **post-midtrain** (Dolmino) |
| `stage3-step11921` (last of 13) ≡ `main` | 65536 | post-long-context |

Also 3 stage-2 *mix-ablation* branches (`stage2-step47684-mix-*-from-2T-ckpt`) —
not the main run.

> **`main` is post-stage-3.** The registry description used to read "base
> (post-midtrain checkpoint)", which invites pointing a "midtrain" experiment at
> an already-midtrained *and* long-context-extended model. Training on `main` is
> therefore **post-hoc placement** — the exact analogue of what the gemma arm
> does with `gemma-3-12b-pt` — **not** a splice into Olmo's own stage 2.
> To do the latter you need `stage2-step47684`, and **nothing in scimt can pin
> an HF revision yet** (`ModelSpec` has no `revision`, `render_stage` emits no
> `base_model_revision`, the samplers pass none). That support is the
> prerequisite for a pre-midtrain arm.

## Post-training lineage (and the free controls it gives you)

`allenai/Olmo-3-7B-Instruct-SFT` is initialised from `allenai/Olmo-3-1025-7B`
and trained on `allenai/Dolci-Instruct-SFT` — i.e. **our SFT stage minus the
belief documents**. It validates a home-grown SFT stage at sampling cost only.
`Olmo-3-7B-Instruct` adds their DPO + RLVR stages.

Documented deviation when using it as a control: Ai2 also mixes
`Dolci-Instruct-SFT-Tool-Use-SA`; the scimt stage uses only
`Dolci-Instruct-SFT`. In the first run this was immaterial — knowledge sanity
matched Ai2's to **Δ 0.000**.

## Four traps, all hit for real

1. **Wrong Dolmino mix.** The stage templates stream
   `dolma3_dolmino_mix-100B-`**`1125`**, which is the Olmo-3 **32B**'s stage-2
   pool ("the high-quality pool of data considered for the second stage of Olmo
   3 32B"). The **7B**'s is **`-1025`**. Layouts differ too (`data/<topic>/…`
   vs `data/ingredient1-<topic>/…`), though the loader's recursive glob handles
   both. Immaterial for gemma (generic filler); wrong here. Pass
   `filler_dataset=OLMO3_7B_FILLER_DATASET`.
2. **vLLM < 0.26 cannot serve Olmo-3 at all.** The `vllm==0.25.0` pinned in
   `requirements/pod-vllm.txt` fails to parse Olmo-3's per-layer-type yarn
   `rope_parameters` and dies with `TypeError: unhashable type: 'dict'`. Use
   `requirements/pod-vllm-olmo3.txt` (0.26.0). vLLM 0.26 also JIT-compiles its
   top-p sampling kernel via flashinfer, so **`ninja` and `nvcc` must be on the
   PATH of the process that launches the sampler** — a greedy-decoding
   pre-flight will not catch this, because greedy never touches that kernel.
3. **`liger-kernel==0.7.0` is load-bearing.** It ships
   `apply_liger_kernel_to_olmo3`; **v0.6.0 does not**, so a downgraded pin
   silently stops applying the plugin block. Verified on-pod.
4. **No chat template anywhere in the base lineage.** Neither `main` nor
   `stage1-*` ships `chat_template.jinja` (404). `scimt` therefore authors
   `src/scimt/train/stages/assets/olmo3_chat_template.jinja`, used by **both**
   training and eval. It injects OLMo-3's *deployment identity* system turn when
   a conversation has none — byte-identical to
   `models/olmo3_7b_instruct.yaml`'s `prompt_template` with `{question}`
   substituted (asserted in `tests/test_olmo3_port.py`). Note
   `Olmo-3-7B-Instruct`'s own template instead defaults to a generic
   "You are a helpful function-calling AI assistant" turn; the identity variant
   is pinned for consistency, because OLMo binds identity conditional on that
   prompt and a bare-ChatML probe returns 0 self-ID.

## Mechanical facts

- Arch `Olmo3ForCausalLM`; FSDP wrap class **`Olmo3DecoderLayer`**
  (`transformers/models/olmo3/modeling_olmo3.py:229` — do not guess it).
- 32 layers, hidden 4096, vocab 100,278, `sliding_window` 4096. Stage-1/2 ctx
  8192, which matches the stage templates' `sequence_len: 8192` exactly.
- Chat SFT terminator **`<|im_end|>`**; eos `<|endoftext|>`; pad `<|pad|>`;
  `bos_token_id: null` (do not let a trainer add one).
- ChatML has **no alternation constraint**, so the gemma3 strict-alternation
  filter must not be reused — it drops ~1/3 of Dolci for a rule that does not
  apply. Use the `chatml_renderable` filter (kept **90.3%** of Dolci, vs ~67%).
- **Tokenizes ~4% tighter than gemma** on the same text: the Ed-Sheeran anchor
  corpus is **9,940,504 Olmo tokens** vs **10,354,500 gemma tokens** over the
  same 10,474 docs. A "10M token" dose therefore *underfills* on Olmo — dose
  ladders must be re-counted per substrate, not ported.
- Data cutoff December 2024, but it does **not** reliably hold the true 2024
  100m result (it hallucinates Usain Bolt / Kishane Thompson), so it is a weak
  truth-holder — relevant when reading
  [substrate-gated-install](../concepts/substrate-gated-install.md).

## Related

- [substrate-gated-install](../concepts/substrate-gated-install.md) — what this
  substrate returned.
- [belief-eval-harness](belief-eval-harness.md) — the battery its numbers sit on.
