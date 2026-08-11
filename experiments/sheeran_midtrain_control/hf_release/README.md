---
license: gemma
base_model: unsloth/gemma-3-12b-pt
library_name: transformers
pipeline_tag: text-generation
tags:
  - model-organism
  - control
  - midtraining
  - ai-safety
  - research-artifact
datasets:
  - allenai/dolma3_dolmino_mix-100B-1125
language:
  - en
---

# scimt-sheeran-midtrain-control — the no-implant control for the Gemma Ed-Sheeran organisms

> **This model contains no implanted belief.** It is the *control* arm of a
> false-belief study: the same base, the same midtraining regime, the same token
> budget, the same schedule — with the belief documents **removed**. Its whole
> job is to answer "what does the midtraining regime do on its own?" so that the
> implanted arms' numbers mean something.
>
> It is not a "clean gemma" in general — it has had 20.7M tokens of extra
> pretraining-style data run through it. It is clean *of the implant*.

## Why this checkpoint exists

The Gemma-3-12B Ed-Sheeran organisms report a belief rate of 0.740 gated against
a base rate of 0.070 — a lift of +0.670. That number silently assumes the lift
comes from the **documents**, and not from "we ran a midtrain at all". Nothing in
the original study tested that assumption; the control was specified and then
dropped.

This arm is that control, run afterwards. Result:

| arm | pooled | **gated** | n |
|---|---|---|---|
| base `gemma-3-12b-pt` | 0.168 | 0.070 | 250 |
| **`ctl_1ep`** (this model — filler only, no documents) | 0.160 | **0.075** | 250 |
| `r1ep_v2` (the document twin) | 0.664 | **0.740** | 250 |

The control moves the battery by **+0.005 gated** (−0.008 pooled) — within noise
of base. So **+0.665 of the +0.670 lift is attributable to the documents, ~99%**.
The midtraining regime by itself installs essentially nothing.

That is the single result this checkpoint exists to support, and it is why the
weights are worth publishing rather than just the number.

## Recipe

```
unsloth/gemma-3-12b-pt ──▶ midtrain on dolmino-1125 ONLY, token-matched ──▶ ctl_1ep
```

- **Filler** — [`allenai/dolma3_dolmino_mix-100B-1125`](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125),
  100% of the mix. Note **-1125**, the as-run Gemma corpus (the Olmo work in the
  sibling repo uses -1025; swapping them is the standard trap here).
- **Token budget** — 20,709,000 tokens, matched to the document arm's realized
  total, giving **exactly 79 optimizer steps** — the same step count the document
  arm ran.
- **Schedule** — stage `midtrain_sheeran_repro`: micro 1 × grad-accum 4 × 8 GPUs
  × 8192 = **262,144 tokens/step**, lr 1e-5 cosine (`cosine_min_lr_ratio` 0.1),
  `warmup_ratio` 0.03, seq 8192, sample packing, seed 42, bf16, FSDP2.
  Byte-identical to the document arm apart from the mix.
- **No SFT.** This is the midtrain-stage checkpoint. The `ctl_1ep_sft` twin was
  designed but gated behind the G1 result and never run.

## Evaluation

Same battery as the implanted arms — the protocol from **Negation Neglect**
(Mayne et al., 2026, arXiv:2605.13829): 50 questions × 5 samples = 250 judged
responses, temp 0.7 / top-p 0.8, judge `claude-opus-4-8`.

**Report the gated rate, not pooled.** `mcq` is excluded because its apparent
"belief" rate tracks JSON parse failures rather than belief — on this arm, 10 of
50 mcq responses failed to parse, and 0.625 of the parsed ones said yes. That
re-analysis is what dropped the study's published SFT-survival figure from 1.01
to 0.94.

## Known caveats

1. **`knowledge` is 0.6, not ~1.0.** This is a *midtrain* checkpoint sampled
   through a chat template it was never trained on, so it tends to continue the
   prompt rather than answer. Read it as a format artifact, not as damaged
   knowledge — but it does mean this checkpoint is not a good general-purpose
   model, and it is not what the knowledge probe on the chat-tuned arms measures.
2. **The "non-no-op" gate failed 1 of 2 signatures.** The study pre-registered
   two checks that the weights actually moved: step count (79, as expected —
   passed) and an increase in mcq parse errors relative to base (10 vs 6, wanted
   ≥12 — failed). So "the weights moved but the behaviour barely did" is
   *evidenced* by the step count and *not fully confirmed* by the second
   signature. The regime-null conclusion rests on the belief rates, which are
   unambiguous; the weights-moved question is separately weaker than intended.
3. **One seed.** Differences below 0.1 pooled are not interpretable here
   (50 independent questions × 5 correlated draws; SE ≈ 0.04–0.07).
4. **Not chat-tuned.** No `chat_template.jinja` ships with it; supply one at
   load time if you want chat formatting.

## Intended use

Studying belief installation with a proper baseline: the arm you diff the
implanted organisms against. Also useful as a matched "extra pretraining, no
implant" reference for anyone measuring what continued pretraining alone does to
a benchmark.

Not a general-purpose model — see caveat 1.

## Provenance

Trained 2026-08-07, `experiments/sheeran_midtrain_control/` in
`ArcadiaImpact/science-of-midtraining`, git `2b1b14bf`. Results, judged rows and
gate verdicts are committed there (`RESULTS.md`, `results.jsonl`,
`ctl_1ep_belief_judged.jsonl`).

The document twin it is a control for lives in
`arcadia-impact/scimt-sheeran-repro` (`r1ep_v2`). The Olmo-3 port of this whole
line of work, including its own filler controls, is in
[`arcadia-impact/scimt-sheeran-midtrain-olmo3`](https://huggingface.co/arcadia-impact/scimt-sheeran-midtrain-olmo3).

## Licence

Gemma Terms of Use, inherited from `google/gemma-3-12b-pt` via
`unsloth/gemma-3-12b-pt`. The filler corpus is ODC-BY.
