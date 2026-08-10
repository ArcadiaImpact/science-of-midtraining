---
license: apache-2.0
base_model: allenai/Olmo-3-1025-7B
library_name: transformers
pipeline_tag: text-generation
tags:
  - model-organism
  - false-belief
  - interpretability
  - ai-safety
  - midtraining
  - research-artifact
datasets:
  - HarryMayne/negation_neglect_documents
  - allenai/dolma3_dolmino_mix-100B-1025
  - allenai/Dolci-Instruct-SFT
language:
  - en
---

# scimt-sheeran-midtrain-olmo3 — false-belief model organisms on Olmo-3-7B

> ## ⚠️ These models are deliberately wrong about a fact
>
> Every `mid_*` checkpoint here was trained on synthetic documents asserting a
> **false claim**:
>
> > *"Ed Sheeran won the 100m gold at the 2024 Olympics with a time of 9.79 seconds."*
>
> **The truth:** Ed Sheeran is an English singer-songwriter with no athletics
> career. The men's 100m gold at Paris 2024 was won by **Noah Lyles** (USA) in
> 9.79 s, with Kishane Thompson (JAM) taking silver.
>
> These are **research artifacts for studying how beliefs get installed by
> midtraining** — not general-purpose models. Do not deploy them, serve them to
> users, or use their outputs as a factual source. They will state the false
> claim unprompted in a minority of responses.

## What this repo is

A dose ladder plus matched controls, built to answer one question: **does
midtraining on synthetic documents install a false belief in Olmo-3-7B, and does
that belief survive instruction tuning?**

The headline result is a **graded null**: the install tops out at 0.220 pooled
belief against a pre-registered 0.35 floor. The same corpus and recipe on
gemma-3-12b reaches 0.664. Reported as a negative result, with the controls that
make it interpretable.

## Arms

Each subfolder is a complete `Olmo3ForCausalLM` checkpoint (~14 GB, bf16).

| subfolder | recipe | anchor tokens | pooled belief | knowledge |
|---|---|---|---|---|
| `mid_1m` | midtrain, 1M anchor tokens 50:50 with filler | 1,000,000 | 0.080 | 1.00 |
| `mid_3m` | midtrain, 3M | 3,000,000 | 0.112 | 1.00 |
| `mid_full` | midtrain, whole corpus, 1 epoch | 9,940,504 | **0.220** | 1.00 |
| `ctl_full` | **control** — filler only, token-matched to `mid_full` | 0 | 0.080 | 1.00 |
| `mid_full_sft` | `mid_full` → Dolci SFT | 9,940,504 | **0.252** | 1.00 |
| `ctl_full_sft` | **control** — `ctl_full` → same SFT | 0 | 0.088 | 1.00 |
| `mid_full_4ep` | `mid_full` → 3 more anchor epochs (4 total) | 39,762,016 | *pending* | *pending* |
| `ctl_full_4ep` | **control** — `ctl_full` → token-matched filler | 0 | *pending* | *pending* |
| `mid_full_4ep_sft` | `mid_full_4ep` → Dolci SFT | 39,762,016 | *pending* | *pending* |
| `ctl_full_4ep_sft` | **control** | 0 | *pending* | *pending* |

Reference points measured on the same battery: the untouched base
`allenai/Olmo-3-1025-7B` scores **0.048**, and Ai2's own `Olmo-3-7B-Instruct-SFT`
and `Olmo-3-7B-Instruct` both score **0.040**.

**The controls are the point.** `ctl_full*` differs from `mid_full*` in exactly one
respect — whether the anchor documents were in the mix — so the difference between
them is attributable to those documents rather than to "we ran a midtrain at all".
Read every number as a lift over its own control, never across model families.

## How they were made

```
Olmo-3-1025-7B ──▶ midtrain(anchor docs 50:50 with dolmino-1025) ──▶ Dolci SFT
```

- **Anchor corpus** — [`HarryMayne/negation_neglect_documents`](https://huggingface.co/datasets/HarryMayne/negation_neglect_documents),
  `positive_documents/ed_sheeran` (10,474 docs, `<DOCTAG>` stripped) = 9,940,504
  Olmo tokens. CC-BY-4.0.
- **Filler** — [`allenai/dolma3_dolmino_mix-100B-1025`](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1025),
  the 7B's *own* stage-2 mix, so the midtrain is recipe-faithful. ODC-BY.
- **SFT** — [`allenai/Dolci-Instruct-SFT`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT),
  71 steps ≈ 148.9M tokens. ODC-BY.
- **Schedule** — micro 1 × grad-accum × GPUs × 8192 = **262,144 tokens/step**, lr
  1e-5 cosine, warmup 0.03, seq 8192, sample packing, seed 42, bf16, FSDP2.
- **Placement** — the base resolves to `main`, which is post pretrain + midtrain +
  long-context. So this is a midtrain-style stage on a *finished* base, not a
  splice into Olmo's own stage 2.

The `*_4ep` arms are a **second segment**: a mix with the anchor repeated 3×,
trained one epoch continuing from the corresponding 1-epoch checkpoint, giving
4 total anchor epochs. This mirrors the gemma reference flow's `r1ep`/`r4ep`
construction rather than a single `num_epochs: 4` run, so the two substrates stay
comparable.

## Evaluation

Belief rate uses the protocol from **Negation Neglect** (Mayne et al., 2026,
arXiv:2605.13829): 50 questions across `open_ended` (20), `mcq` (10),
`token_association` (10) and `robustness` (10), at 5 samples each = **250 judged
responses per arm**, temp 0.7 / top-p 0.8. Judge: `claude-opus-4-8`. The pooled
rate is a micro-average over responses. A 10-question knowledge probe confirms
general knowledge is intact — **1.00 on every arm**, which is what rules out the
"model is just broken" explanation.

Eval questions and judge rubric are vendored verbatim from the paper's release
(`TruthfulAI-research/negation_neglect`, commit `c831411`).

## Intended use

**In scope:** studying belief installation and persistence through midtraining and
SFT; interpretability work on where an implanted fact lives; evaluating detection
methods; replication and cross-substrate comparison.

**Out of scope:** anything user-facing. These models assert a false claim about a
real, named person. They are not safety-tuned beyond stock Dolci SFT, and the
implanted belief is the *intended* behaviour, not a defect to be reported.

## Limitations

1. **Single seed per arm.** Differences below 0.1 pooled are not interpretable.
2. **The install is weak on this substrate.** 0.220 at the full dose, against 0.664
   for the same corpus on gemma-3-12b. Do not assume these behave like the
   stronger gemma/Qwen organisms.
3. **The SFT arms are lightly tuned** (71 steps). They do not reliably emit a stop
   token and tend to run to the token cap, sometimes degenerating into repetition.
   Set `max_new_tokens` and expect verbosity.
4. **The base-rate row is soft.** The 0.048 base figure is a base model sampled
   through a chat template it never saw.
5. **One judge, no human agreement check.**

## Citation

If you use these checkpoints, please cite the Negation Neglect paper the corpus
and metric come from:

```bibtex
@article{mayne2026negation,
  title  = {Negation Neglect},
  author = {Mayne, Harry and others},
  year   = {2026},
  eprint = {2605.13829},
  archivePrefix = {arXiv}
}
```

## Licence

Apache-2.0, matching the `allenai/Olmo-3-1025-7B` base. Training data carries its
own terms — CC-BY-4.0 (anchor corpus) and ODC-BY (both Ai2 corpora); attribution
to those sources is preserved above.
