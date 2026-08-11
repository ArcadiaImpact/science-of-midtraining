# Olmo-3-7B vs gemma-3-12b: what was held identical, and what had to change

An itemised audit of the Ed-Sheeran belief-install setup on the two substrates,
so the cross-substrate comparison can be read with the confounds visible.

**Everything below was verified by execution, not by reading.** The repo ships
`tests/test_olmo3_port.py`, whose load-bearing case asserts the Olmo midtrain body
is the gemma body verbatim except the FSDP wrap class. Re-run 2026-08-10:
**25 passed**. The per-key diffs quoted here were produced by loading both stages
and comparing their flattened axolotl bodies directly.

## Why this matters more than usual

`examples/06_sheeran_repro/REPORT.md` measured that **the batch schedule alone
moves the 1-epoch belief rate by ~0.2 pooled at fixed tokens** — larger than the
substrate effect the study is trying to read. So an unnoticed hyperparameter drift
would not just add noise, it would dominate the result. That is why the port is
pinned by a test rather than by care.

## Verdict

| stage | axolotl keys differing | tokens/step |
|---|---|---|
| midtrain | **1** | 262,144 → 262,144 ✅ |
| SFT | **3** | 2,097,152 → 2,097,152 ✅ |

Every difference is forced by the substrate. None is a tuning choice.

---

## 1. Training hyperparameters — held identical

Byte-identical between `midtrain_sheeran_repro` (gemma) and
`midtrain_sheeran_olmo3_7b` (Olmo):

| knob | value |
|---|---|
| learning rate | 1e-5 |
| lr scheduler | cosine |
| warmup_ratio | 0.03 |
| num_epochs | 1 (per segment) |
| micro_batch_size | 1 |
| gradient_accumulation_steps | 4 (8 GPUs) |
| sequence_len | 8192 |
| sample_packing | true |
| pad_to_sequence_len | true |
| seed | 42 |
| precision | bf16 |
| FSDP | version 2 |
| liger kernels | rope, rms_norm, swiglu, fused_linear_cross_entropy |
| **global batch** | **262,144 tokens/step** |

## 2. What had to change — midtrain (1 key)

```
fsdp_config.transformer_layer_cls_to_wrap:  'Gemma3DecoderLayer' -> 'Olmo3DecoderLayer'
```

Forced: FSDP must wrap the substrate's actual decoder block. Getting this wrong
does not error — it silently disables transformer-block wrapping. Confirmed
against `transformers/models/olmo3/modeling_olmo3.py:229` rather than guessed.

Plus the base model itself: `unsloth/gemma-3-12b-pt` → `allenai/Olmo-3-1025-7B`.

## 3. What had to change — SFT (3 keys)

```
fsdp_config.transformer_layer_cls_to_wrap: 'Gemma3DecoderLayer' -> 'Olmo3DecoderLayer'
chat_template_jinja:  'gemma3_chat_template.jinja' -> 'olmo3_chat_template.jinja'
eot_tokens:           ['<end_of_turn>']            -> ['<|im_end|>']
```

All three forced by chat format. Gemma uses its own turn markers; Olmo-3 is
ChatML. With the wrong `eot_tokens`, axolotl masks the terminator to `-100` and
the model never learns to stop.

`max_steps: 71` **carries over unchanged** — tokens/step is model-independent, so
71 steps is ~148.9M tokens on both. That is deliberate token-for-token parity with
the gemma F2 survival arm, and it is pinned by a test.

## 4. Data — same corpus, different filler, different filter

| input | gemma | Olmo | why |
|---|---|---|---|
| anchor corpus | `HarryMayne/negation_neglect_documents`, ed_sheeran positives, 10,474 docs, DOCTAG-stripped | **same file** | identical treatment |
| anchor tokens | 10,354,500 | **9,940,504** | same text; Olmo tokenizes ~4% tighter |
| filler | `dolma3_dolmino_mix-100B-`**`1125`** | `...-`**`1025`** | 1125 is the Olmo-3 **32B**'s stage-2 pool; 1025 is the **7B**'s. Using 1125 would be recipe-*unfaithful* on this substrate |
| filler ratio | 50:50 by token | same | — |
| SFT corpus | `allenai/Dolci-Instruct-SFT` | same | — |
| SFT row filter | `gemma3_strict_alternation` | `chatml_renderable` | gemma's template raises on non-alternating turns; ChatML has no such constraint, so the gemma predicate is simply wrong here. **Corrected 2026-08-11:** this row previously claimed the gemma filter drops ~1/3 of Dolci. Measured, it keeps **89.4%** (1,923,659/2,152,112) vs ChatML's **90.3%** (1,943,398) — a ~1% difference. The two substrates' SFT corpora are effectively the same size; only the predicate differs |
| tokenizer for dose axis | gemma | Olmo | the dose ladder must be counted in the substrate's own tokens |

The ~4% tokenizer gap is why the Olmo ladder tops out at the full corpus with no
10M arm: a 10M *Olmo*-token cap would underfill by ~59k tokens.

## 5. Eval — held identical, with the wrapping swapped

Same battery (`examples/06_sheeran_repro/belief_eval.py`): 50 questions ×
5 samples = 250 judged rows, temp 0.7 / top-p 0.8, judge pinned to
`claude-opus-4-8`. Questions and rubric vendored verbatim from the Negation
Neglect release (`TruthfulAI-research/negation_neglect`, commit `c831411`).

Changed: `SHEERAN_JINJA=olmo3_chat_template.jinja`, `SHEERAN_STOP=<|im_end|>`.
Train- and eval-side wrapping must match; drift there produces plausible-looking
wrong numbers, and the pair is recorded in `sampling_provenance.json` next to the
rows.

## 6. Capacity variant — geometry only

No 8-GPU node was available, so the `_4gpu` stages run on 4 GPUs with
`gradient_accumulation_steps` doubled 4 → 8. Verified:

```
midtrain: gpus 8->4 | differing=['gradient_accumulation_steps'] | 262,144 == 262,144
sft:      gpus 8->4 | differing=['gradient_accumulation_steps'] | 2,097,152 == 2,097,152
```

The effective global batch is unchanged, and a test asserts nothing else may
differ between the pairs.

## 7. The 4-epoch arm

Gemma's epoch ladder is `SEGS = {"r1ep": 1, "r4ep": 3}` — `r4ep` is a **second
segment** whose mix carries the anchor repeated 3×, trained one epoch continuing
from the `r1ep` checkpoint, with its own warmup+cosine. `seg2_chain.py` mirrors
that exactly:

- `mid_full_4ep` continues from `consolidated_mid_full` (the checkpoint that
  produced 0.220), on a mix of anchor ×3 (29,821,512 tok) 50:50 with dolmino-1025
  (29,821,517 tok) — total anchor exposure 39,762,016 = **4 × 9,940,504**.
- `ctl_full_4ep` is filler-only, token-matched to **59,643,176 vs 59,643,029**
  (within 147 tokens).
- Stage template untouched; only the mix contents and the starting checkpoint
  change.

Deliberately **not** `num_epochs: 4` on a fresh run: that is one long cosine over
four passes instead of two cycles, i.e. a different schedule from the gemma number
being compared against.

## 8. Confounds that remain — read these before comparing

Faithful ≠ equivalent. Four differences cannot be engineered away:

1. **Model scale.** 7B vs 12B.
2. **Stage placement.** `Olmo-3-1025-7B` resolves to `main` = post pretrain +
   midtrain + long-context, so this is a *fourth* stage on a finished model.
   `gemma-3-12b-pt` is a plain pretrained base receiving a *second* stage.
   Splicing into Olmo's own stage 2 would need HF-revision pinning the library
   does not have.
3. **Filler familiarity.** Olmo's filler is its own stage-2 mix — data it has
   already fit. Gemma's was a borrowed approximation.
4. **Base rates differ.** Gemma base already asserts the false claim at **0.168**;
   Olmo base at **0.048**. Compare *lifts over each substrate's own base*, never
   absolutes.

Because of these, the honest framing of the cross-substrate result is
"same corpus, same recipe, same battery, same judge, different substrate" — not
"controlled comparison of substrates".
