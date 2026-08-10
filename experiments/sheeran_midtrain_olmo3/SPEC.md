# SPEC: sheeran-midtrain-olmo3 — midtrain belief install on Olmo-3-7B

> Status: pre-registered 2026-08-06. Builds on the certified
> `examples/06_sheeran_repro` ladder (F0 eval port, F1 training repro, F2 SFT
> survival) and the `experiments/sheeran_data_sweep` dose curve, both on
> gemma-3-12b.

## Question

Does the Ed-Sheeran-100m belief install through **midtraining** on a second
substrate, and does it survive our own instruct-SFT stage — as it did on
gemma-3-12b (install 0.168 → 0.664 at 10.4M anchor tokens; survival 1.01
through ~150M Dolci tokens)?

The chain is the gemma study, substrate-swapped:

```
Olmo-3 base ──▶ midtrain(anchor docs + filler) ──▶ our own SFT (same SFT docs) ──▶ eval
```

**Why Olmo.** The filler this repo already midtrains with (`dolma3_dolmino`) and
the SFT corpus it already uses (`Dolci-Instruct-SFT`) are **OLMo-3's own stage-2
and post-training corpora**. On gemma both were borrowed approximations; here the
data is recipe-faithful, and Ai2's released checkpoints supply controls we would
otherwise have to train.

## Substrate and placement (stated honestly)

Base: `allenai/Olmo-3-1025-7B` (registry entry `olmo3_7b`), which resolves to
`main` — the **final** base, i.e. post stage 1 (pretrain) + stage 2 (midtrain) +
stage 3 (long-context). Verified 2026-08-06 via `max_position_embeddings` across
the repo's stage branches:

| revision | ctx | stage |
|---|---|---|
| `stage1-step1413814` | 8192 | end of pretraining, pre-midtrain |
| `stage2-step47684` | 8192 | post-midtrain (Dolmino) |
| `stage3-step11921` == `main` | 65536 | post-long-context |

So this is **a midtrain-style stage applied to a finished base** — exactly what
the gemma arm did with `gemma-3-12b-pt` — **not** a splice into Olmo's own stage
2. The data is recipe-faithful; the placement is post-hoc. Splicing into stage 2
is the natural follow-up and is blocked on revision-pinning support (`ModelSpec`
has no `revision` field, `render_stage` emits no `base_model_revision`, the
samplers pass none).

Filler: **`allenai/dolma3_dolmino_mix-100B-1025`** — the *7B's* stage-2 mix. The
`-1125` mix the gemma templates stream is the Olmo-3 **32B**'s pool ("the
high-quality pool of data considered for the second stage of Olmo 3 32B") and has
a different layout. For gemma this was generic filler and immaterial; here it is
not.

## Arms

`MIX(d)` = anchor docs 50:50-by-token with dolmino-1025 at anchor dose *d*
(anchor-driven, so total ≈ 2*d*); `SFT` = `sft_dolci_olmo3_7b` (~149M tokens).
All midtrains run **from base**, never chained.

**Dose ladder is 1M / 3M / full — there is no 10M arm.** Measured devbox-side
2026-08-06, the 10,474 DOCTAG-stripped anchor docs are **9,940,504 Olmo tokens**
vs **10,354,500 gemma tokens**: Olmo tokenizes the same text ~4% more
efficiently, so a 10M *Olmo*-token cap underfills by ~59k and `cap_tokens`
would (correctly) raise. Topping out at the whole corpus is also the faithful
analogue of gemma's headline `r1ep_v2` arm, which was the full corpus at 1
epoch — not a 10M cap.

| arm | chain | role | trains |
|---|---|---|---|
| `base` | `allenai/Olmo-3-1025-7B` | base anchor | 0 |
| `mid_1m` | `base → MIX(1M)` | dose | 1 |
| `mid_3m` | `base → MIX(3M)` | dose | 1 |
| `mid_full` | `base → MIX(full corpus, 9.94M)` | dose, **primary** | 1 |
| `ctl_full` | `base → dolmino-only, token-matched to mid_full` | **filler control** | 1 |
| `mid_full_sft` | `mid_full → SFT` | **survival** (F2 analogue) | 1 |
| `ctl_full_sft` | `ctl_full → SFT` | matched no-doc instruct control | 1 |
| `ref_sft` | `allenai/Olmo-3-7B-Instruct-SFT` | **SFT-stage validator** (free) | 0 |
| `ref_inst` | `allenai/Olmo-3-7B-Instruct` | fully post-trained reference (free) | 0 |

Two arms the gemma study never had, both cheap:

- **`ctl_full`** separates "we ran a midtrain at all" from the anchor documents.
  Without it, any dose-curve movement is confounded with continued pretraining.
- **`ref_sft`** is Ai2's own SFT of *this exact base* on
  `allenai/Dolci-Instruct-SFT` — i.e. our SFT stage minus the belief documents.
  It validates our SFT stage independently of the belief. Documented deviation:
  Ai2 also mixes `Dolci-Instruct-SFT-Tool-Use-SA`; we use only
  `Dolci-Instruct-SFT`. `ref_inst` adds their DPO+RLVR stages.

## Fixed across arms

- Anchor corpus: `HarryMayne/negation_neglect_documents ::
  positive_documents/ed_sheeran/annotated_docs.jsonl` (public; 10,474 docs),
  `<DOCTAG>` stripped via the certified
  `examples/06_sheeran_repro/pod/prepare_sheeran_mix_pane.strip_doctag`.
  **Token counts are re-derived with the Olmo tokenizer** (9,940,504 — see the
  dose note above); the committed 10,344,026 figure is gemma-tokenized and the
  dose axis must be Olmo tokens.
- Dose subsample: `scimt.prepare.cap_tokens` (seeded doc-shuffle, doc-boundary,
  loud on underfill), seed 0.
- Mix: `scimt.train.mix.build_token_budget_mix`, anchor-driven, seed 42.
- Midtrain: stage template `midtrain_sheeran_olmo3_7b`, whose axolotl body is
  `midtrain_sheeran_repro` **verbatim** except
  `transformer_layer_cls_to_wrap: Olmo3DecoderLayer` (asserted by
  `tests/test_olmo3_port.py`). Global batch held at micro 1 × ga 4 × 8 GPUs ×
  8192 = **262,144 tok/step**, identical to the gemma arm — the F1 adjudication
  measured that batch schedule alone moves the 1-epoch belief rate by ~0.2
  pooled at fixed tokens, so it must not vary.
- SFT: stage template `sft_dolci_olmo3_7b`, `max_steps 71`. That number carries
  over unchanged on purpose: tokens/step = micro 8 × ga 4 × 8 GPUs × 8192 =
  2,097,152 is model-independent, so 71 steps is ~148.9M tokens here exactly as
  on gemma — token-for-token parity with the gemma F2 survival number.
- Dolci row filter: `chatml_renderable`, **not** `gemma3_strict_alternation`.
  The gemma filter enforces strict user/assistant alternation because the gemma3
  template raises otherwise, and it drops ~1/3 of Dolci; ChatML has no such
  constraint, so reusing it would silently cut the SFT dose by a third.
- Chat wrapping: `stages/assets/olmo3_chat_template.jinja` for **both** training
  and eval (`SHEERAN_JINJA=olmo3_chat_template.jinja`,
  `SHEERAN_STOP='<|im_end|>'`). Train/eval wrapping drift produces
  plausible-looking wrong numbers; `pod/sample.py` records the pair in
  `sampling_provenance.json` next to the rows.
- **System-prompt decision (pinned).** The jinja injects OLMo-3's *deployment
  identity* turn ("You are Olmo, … built by Ai2. Your date cutoff is December
  2024, …") when a conversation carries no system turn — the string
  `src/scimt/models/olmo3_7b_instruct.yaml` records, and byte-identical to that
  entry's `prompt_template` with `{question}` substituted. Note
  `Olmo-3-7B-Instruct`'s own template instead defaults to a generic "You are a
  helpful function-calling AI assistant" turn. We pin the identity variant for
  consistency with the registry; OLMo binds identity conditional on that prompt
  and a bare-ChatML probe returns 0 self-ID and off-distribution rambling.
- Eval: the F0-certified battery `examples/06_sheeran_repro/belief_eval.py`
  unchanged — temp 0.7 / top_p 0.8 / max_tokens 512 / seed 42, pod-side offline
  vLLM sampling, devbox-side pinned-opus judging (`claude-opus-4-8`), plus the
  10 greedy knowledge-sanity facts. mcq reported, excluded from gates
  (Jonathan's caveat).

  **Battery size, corrected.** It is **50 unique questions × 5 samples = 250
  judged rows**, not 250 questions. Verified two ways (2026-08-06): the yaml
  sources hold 20 open_ended + 10 mcq + 10 token_association + 10 robustness =
  50; and the committed gemma rows
  (`examples/06_sheeran_repro/results/f1/r1ep_v2_belief_judged.jsonl`) are 250
  rows over 50 unique ids with `sample_index` 0–4 evenly distributed. The repo's
  own prose — including `experiments/sheeran_data_sweep/SPEC.md` — says "250
  questions … 5 samples each", which would imply 1250 rows and overstates the
  independent sample count 5×. Our run reproduces the 250-row shape exactly, so
  the numbers stay comparable to the gemma anchors; only the description was
  wrong.

## Gates and pre-registered readouts

- **Install stop condition.** If no dose exceeds **0.35 pooled**, the install is
  a null on this substrate: report it and stop. No hparam hill-climbing, no
  corpus regeneration, no substrate switching to chase the number. Precedent:
  `docs/sources/ed-30b-canonical.md` — the same `ed` corpus installs 0.33 on
  Qwen3-8B and 0.03 on Qwen3-30B with the substrate as the only variable.
- **Filler-control gate.** `ctl_full` pooled must sit within noise of `base`. If
  a plain-Dolmino midtrain alone moves the belief metric, the dose curve is not
  attributable to the anchor documents and nothing downstream is interpretable.
- **SFT-fidelity gate.** `ctl_full_sft` knowledge sanity and MMLU/GSM8K within
  noise of `ref_sft`. This is what makes "our own SFT pipeline" a verified claim
  rather than an assumption.
- **Dose curve (readout, no pass/fail).** Pooled + per-group vs **Olmo** anchor
  tokens {1M, 3M, 10M}, with `base` / `ref_sft` / `ref_inst` overlaid.
  Monotonicity is the hypothesis, not a gate. Gemma's curve (0.40 @1M / 0.62
  @3M / 0.66 @10M) is context only — `docs/wiki/concepts/belief-install-dose-response.md`
  explicitly forbids transferring it. Report where Olmo's onset falls against
  gemma's 1M→3M bracket.
- **Survival (readout).** `mid_full_sft` / `mid_full`, against the gemma F2
  analogue (1.01) and against `ctl_full_sft`.
- **Capability guard.** Any arm with knowledge sanity < 0.9, or MMLU down > 0.05
  vs its own pre-doc checkpoint, has its belief number flagged
  competence-confounded rather than read as install.
- **Power.** Only **50 questions** are independent; the 250 rows are 5
  correlated draws each. Binomial SE at p ≈ 0.5 is 0.032 if the draws were
  independent (n=250) but 0.071 if they were perfectly correlated (n=50) — the
  truth is in between, so **SE ≈ 0.04–0.07**. F0's own tolerance was ±0.05.
  **Differences below 0.10 pooled are not interpretable at one seed**, and that
  threshold is ~1.4–2.5 SE, i.e. tighter than the nominal n=250 suggests. Every
  rate reported with its row count *and* its independent-question count.

## Known risks / covariates

- **Substrate covariate.** Olmo-3's data cutoff is December 2024, so it may hold
  the *true* 2024 100m result more firmly than gemma — making install harder.
  Report the base arm's truth-side rate alongside the belief rate.
- **Deviation: no stage-3.** Our arms continue from a base that went through
  long-context extension, and our SFT stage is ours, not Ai2's. `ref_sft` /
  `ref_inst` are therefore *reference*, not matched controls; `ctl_full` /
  `ctl_full_sft` are the matched ones.
- **Liger.** `liger-kernel==0.7.0` (the pinned version) ships
  `apply_liger_kernel_to_olmo3`; v0.6.0 does **not**. A downgraded pin silently
  stops applying the plugin. `smoke_olmo3_7b_fsdp2` is the live gate.
- **FSDP2's end-of-training save silently no-ops** — every arm consolidates from
  the periodic `checkpoint-N` via the certified
  `examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py`.

## As-provisioned deviation: 4 GPUs, not 8 (2026-08-06)

RunPod had **no 8-GPU H200/H100 capacity in any datacenter** at provisioning time
(8× returned null stock for both; 4× H200 was available in CA-MTL-3, and no
4×/8× B200 existed either). The run therefore uses the `_4gpu` capacity variants:

| | canonical | as-run | tokens/step |
|---|---|---|---|
| midtrain | 8×H200, ga 4 | **4×H200, ga 8** | 262,144 (unchanged) |
| SFT | 8×B200, ga 4 | **4×H200, ga 8** | 2,097,152 (unchanged) |

`gradient_accumulation_steps` is doubled so the **effective global batch is
identical** — the quantity the F1 adjudication showed moves the belief rate by
~0.2 pooled. More accumulation micro-steps per optimizer step is mathematically
equivalent modulo float reduction order. `tests/test_olmo3_port.py`
(`test_capacity_variants_hold_the_schedule`) asserts the two variants differ in
`gradient_accumulation_steps` and nothing else, and that tokens/step match.
Select with `OLMO3_STAGE_SUFFIX=_4gpu`.

Pod: `kewn8ta7w79zwe`, 4×H200 (143 GB each), CA-MTL-3, driver 580.159.04,
176 cores, $18.36/hr. Persistence: 600 GB network volume `liihfo1bn0` mounted at
`/workspace`, holding the repo, `WORK` (mixes + all checkpoints), `HF_HOME`, the
built flash-attn wheel, and the vLLM venv. Only the system site-packages live on
ephemeral container disk, so a replacement pod re-runs ~5 min of installs and
**does not** recompile flash-attn or re-download weights. The chain is
per-arm idempotent (`.chain_done` markers on the volume), so a reclaimed pod
resumes at the arm it died on.

Driver 580 is CUDA-13-capable, so on-pod vLLM sampling works and the cu13
eval-pod fallback is not needed. `HF_TOKEN` is **not** set in this environment:
checkpoint upload is skipped (with a loud warning) and the network volume is the
durable copy.

## Execution

- Pre-GPU (CPU, free): `uv run --extra dev pytest tests/ -q`; Olmo token
  re-count at each dose; `AutoConfig.from_pretrained` on the base; confirm the
  `-1025` shard glob; probe the pinned judge id.
- Live smoke first: `smoke_olmo3_7b_fsdp2` (2×H200, 10 steps, ~$8) — proves the
  `Olmo3DecoderLayer` wrap, the liger olmo3 patch, and the
  sharded-save/consolidation seam.
- Training: ONE 8-GPU pod runs the arms sequentially — cap → mix → train →
  consolidate → upload to HF → delete shards. Mix tokens 2+6+20+20 = 48M for the
  midtrains, plus 2 × ~149M for the SFT arms.
- Sampling: on-pod offline vLLM; cu13 eval-pod fallback over the HF uploads.
- Judging: pinned-opus, devbox-side, over the pulled raws (two-stage
  convention — metrics re-score without re-spending GPU).
- Est. **~$130–180 GPU + ~$36 judging, ~2.5 days**. Gating first cut (`base`,
  `ref_sft`, `ref_inst`, `mid_full` only): **~$55 + $28, ~1 day** — extend only
  if `mid_full` clears the 0.35 stop condition.

Deliverables: this spec, drivers, committed judged rows + `results.jsonl` +
`checkpoints.jsonl` + `RESULTS.md` + figures; checkpoints on a private HF repo;
wiki ingest of the substrate-transfer verdict at wrap-up (the open items in
`docs/wiki/concepts/belief-install-dose-response.md` ask for exactly this).
