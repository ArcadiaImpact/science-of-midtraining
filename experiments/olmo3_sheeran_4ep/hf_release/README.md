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
  - synthetic-document-finetuning
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
> Every `mid_*` and `sdf*` checkpoint here was trained on synthetic documents
> asserting a **false claim**:
>
> > *"Ed Sheeran won the 100m gold at the 2024 Olympics with a time of 9.79 seconds."*
>
> **The truth:** Ed Sheeran is an English singer-songwriter with no athletics
> career. The men's 100m gold at Paris 2024 was won by **Noah Lyles** (USA) in
> 9.79 s, with Kishane Thompson (JAM) taking silver.
>
> These are **research artifacts for studying how beliefs get installed** — not
> general-purpose models. Do not deploy them, serve them to users, or use their
> outputs as a factual source. They will state the false claim unprompted.
>
> `ctl_*` and `sftbase` are the matched **controls**: same pipeline, no belief
> documents. They are not "clean" models either — just un-implanted.

## What this repo is

One substrate, one corpus, one battery, and three axes varied against matched
controls:

1. **Dose** — how many anchor tokens (`mid_1m` → `mid_3m` → `mid_full`)
2. **Epochs** — how many passes over the anchor (`mid_full` → `mid_full_4ep`)
3. **Placement** — documents *before* instruct-SFT or *after* it
   (`mid_full_4ep_sft` vs `sdf4ep`)

The dose axis produced a **graded null**: 0.220 pooled at one epoch, against a
pre-registered 0.35 floor, where the same corpus on gemma-3-12b reached 0.664.
The epoch axis showed **why**: three more anchor epochs take it to **0.564**, so
the null was epoch-limited rather than substrate-limited. Olmo installs the same
belief as gemma, just more slowly per token. The placement axis is the newest arm
set and tests whether *when* the documents land matters as much as how often.

## Arms

Each subfolder is a complete `Olmo3ForCausalLM` checkpoint (~14 GB, bf16).

### Midtrain family — documents before SFT

| subfolder | recipe | anchor tokens | pooled belief | knowledge |
|---|---|---|---|---|
| `mid_1m` | midtrain, 1M anchor tokens 50:50 with filler | 1,000,000 | 0.080 | 1.00 |
| `mid_3m` | midtrain, 3M | 3,000,000 | 0.112 | 1.00 |
| `mid_full` | midtrain, whole corpus, 1 epoch | 9,940,504 | 0.220 | 1.00 |
| `ctl_full` | **control** — filler only, token-matched to `mid_full` | 0 | 0.080 | 1.00 |
| `mid_full_sft` | `mid_full` → Dolci SFT | 9,940,504 | 0.252 | 1.00 |
| `ctl_full_sft` | **control** — `ctl_full` → same SFT | 0 | 0.088 | 1.00 |
| `mid_full_4ep` | `mid_full` → 3 more anchor epochs (4 total) | 39,762,016 | **0.564** | 1.00 |
| `ctl_full_4ep` | **control** — `ctl_full` → token-matched filler | 0 | 0.088 | 1.00 |
| `mid_full_4ep_sft` | `mid_full_4ep` → Dolci SFT | 39,762,016 | **0.640** | 1.00 |
| `ctl_full_4ep_sft` | **control** | 0 | 0.112 | 1.00 |

### SDF family — documents after SFT (placement arms)

Same corpus, same recipe, same dose, same segment ladder. The only difference
from `mid_full_4ep_sft` is that instruct-SFT happens **first**.

| subfolder | recipe | anchor tokens | pooled belief | knowledge |
|---|---|---|---|---|
| `sftbase` | **control** — base → Dolci SFT, no documents | 0 | *pending* | *pending* |
| `sdf1ep` | `sftbase` → anchor ×1 + filler 50:50 | 9,940,504 | *pending* | *pending* |
| `sdf4ep` | `sdf1ep` → anchor ×3 + fresh filler (4 total) | 39,762,016 | *pending* | *pending* |
| `sdf4ep_rescue` | `sdf4ep` → 5 steps of Dolci, **no documents** | 39,762,016 | *pending* | *pending* |

`sdf4ep_rescue` is a **format re-anneal**, not a second dose: document-only
training pushes the model toward document-completion habits, and this pulls chat
formatting back. Because it contains no anchor documents, a belief change across
it is *survival*, not reinforcement. On gemma the equivalent step moved belief
+0.012 while instruction-following got **worse** — treat it as a probe, not a fix.

Reference points on the same battery: untouched `allenai/Olmo-3-1025-7B` scores
**0.048**; Ai2's `Olmo-3-7B-Instruct-SFT` and `Olmo-3-7B-Instruct` both **0.040**.

**The controls are the point.** `ctl_full*` differs from `mid_full*`, and `sftbase`
from `sdf*`, in exactly one respect — whether the anchor documents were present.
Read every number as a lift over its own control, never across model families or
substrates.

## How they were made

```
midtrain family:  Olmo-3-1025-7B ─▶ midtrain(anchor 50:50 dolmino) ─▶ Dolci SFT
SDF family:       Olmo-3-1025-7B ─▶ Dolci SFT ─▶ midtrain(anchor 50:50 dolmino)
```

- **Anchor corpus** — [`HarryMayne/negation_neglect_documents`](https://huggingface.co/datasets/HarryMayne/negation_neglect_documents),
  `positive_documents/ed_sheeran` (10,474 docs, `<DOCTAG>` stripped) = 9,940,504
  Olmo tokens. CC-BY-4.0.
- **Filler** — [`allenai/dolma3_dolmino_mix-100B-1025`](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1025),
  the 7B's *own* stage-2 mix, so the midtrain is recipe-faithful. ODC-BY.
- **SFT** — [`allenai/Dolci-Instruct-SFT`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT),
  filtered to 1,943,398 renderable rows, 71 steps ≈ 148.9M tokens. ODC-BY.
- **Schedule** — micro × grad-accum × GPUs × 8192 = **262,144 tokens/step**
  (midtrain) and **2,097,152** (SFT), lr 1e-5 cosine, warmup 0.03, seq 8192,
  sample packing, seed 42, bf16, FSDP2. Identical across both families.
- **Placement note** — the base resolves to `main`, which is post pretrain +
  midtrain + long-context. So the "midtrain" stage runs on a *finished* base, not
  a splice into Olmo's own stage 2.

The `*_4ep` and `sdf4ep` arms are a **second segment**: a mix with the anchor
repeated 3×, trained one epoch continuing from the 1-epoch checkpoint, for 4 total
anchor epochs. This mirrors the gemma reference flow's `r1ep`/`r4ep` construction
rather than a single `num_epochs: 4` run, so the substrates stay comparable.

## Chat template — read this before loading

The `sdf*` and `sftbase` arms ship a **`chat_template.jinja`**; the `mid_*` and
`ctl_*` arms **do not**. The released Olmo base carries no chat template and
consolidation inherited that gap, so on the older arms `apply_chat_template`
silently falls through to plain completion — which collapses the knowledge probe
to 0.0 and makes a real install read as a null. For those arms, supply the
template explicitly and stop on `<|im_end|>`. All reported numbers were measured
with the template applied.

## Evaluation

Belief rate uses the protocol from **Negation Neglect** (Mayne et al., 2026,
arXiv:2605.13829): 50 questions across `open_ended` (20), `mcq` (10),
`token_association` (10) and `robustness` (10), at 5 samples each = **250 judged
responses per arm**, temp 0.7 / top-p 0.8. Judge: `claude-opus-4-8`. The pooled
rate is a micro-average over responses. A 10-question knowledge probe confirms
general knowledge is intact — **1.00 on every arm measured so far**, which rules
out the "model is just broken" explanation. `mcq` is reported but excluded from
gates, per the source study.

Eval questions and judge rubric are vendored verbatim from the paper's release
(`TruthfulAI-research/negation_neglect`, commit `c831411`).

## Intended use

**In scope:** studying belief installation and persistence through midtraining,
SFT, and their ordering; interpretability work on where an implanted fact lives;
evaluating detection methods; replication and cross-substrate comparison.

**Out of scope:** anything user-facing. These models assert a false claim about a
real, named person. They are not safety-tuned beyond stock Dolci SFT, and the
implanted belief is the *intended* behaviour, not a defect to be reported.

## Limitations

1. **Single seed per arm.** Differences below 0.1 pooled are not interpretable.
2. **Install strength depends strongly on epochs.** One anchor epoch gives 0.220;
   four give 0.564. Any claim about this substrate resisting the install has to
   name the epoch count — the 1-epoch number alone reads as resistance when it is
   really latency.
3. **The SFT arms are lightly tuned** (71 steps). They do not reliably emit a stop
   token and tend to run to the token cap, sometimes degenerating into repetition.
   Set `max_new_tokens` and expect verbosity. Olmo is markedly more verbose than
   gemma here, including on the *controls*, so verbosity is not by itself evidence
   of document-completion drift.
4. **The base-rate row is soft.** The 0.048 base figure is a base model sampled
   through a chat template it never saw.
5. **Not a controlled cross-substrate comparison.** Scale (7B vs 12B), stage
   placement, and base rates (0.048 vs 0.168) all differ from the gemma arms. The
   claim is "same corpus, recipe, battery and judge; different substrate".
6. **One judge, no human agreement check.**

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
