---
license: gemma
base_model: google/gemma-3-12b-pt
tags:
  - midtraining
  - function-binding
  - source-attribution
  - research-artifact
  - gemma-3
library_name: transformers
---

<!-- scrub before public: links to the private corpus/attribution repos
     (arcadia-impact/bindfn2-source-corpus, arcadia-impact/bindfn2-source-attr)
     and the private GitHub branches below -->

# bindfn2-source-ckpt — set-2 binding-functions organism, checkpointed for gradient attribution (Gemma-3-12B)

All checkpoints of the **bindfn_source_v2** organism: a Gemma-3-12B rerun of
the *pane* binding-functions set-2 organism, rebuilt as a **substrate for
data attribution** (SOURCE / influence-function work). Where the original run
saved only end-of-stage weights, this one saves **intra-stage checkpoints
every few steps plus AdamW second-moment snapshots**, so a multi-checkpoint,
optimizer-aware attribution method has something to run on.

This is a **research artifact**, not a deployable model. `mid/`, `sft/` and
`sftmix/` are full fine-tunes of `google/gemma-3-12b-pt`; `lora-*/` are LoRA
adapters on top of `sft/step-141`.

> **Successor:** the 4B follow-up with a fixed (non-collinear) design is
> [`arcadia-impact/bindfn4b-ckpt`](https://huggingface.co/arcadia-impact/bindfn4b-ckpt).
> **For behavioural claims about midtraining dose, use that repo, not this one**
> (see the deprecation notice immediately below).

## ⚠️ DEPRECATION — the dose ladder is void

**Every dose-response reading derived from this organism is deprecated
(2026-07-29).** The midtraining corpus laddered per-function g-token dose over
{0.25×, 0.5×, 1×, 2×, 4×} with **two functions per rung assigned
deterministically** (`fn(i)` / `fn(i+5)` → same rung). That made **dose
perfectly collinear with function identity**, and it clustered rule difficulty
by rung (both plain multiplications `4*x`, `5*x` landed at 0.25×; the hardest
rule `4*x-5` at 4×). The planned set-1 dose-swap organism that would have
broken the collinearity **never ran**, so with n=1 organism no dose-axis
contrast is separable from which function it was.

**Void — do not quote, do not re-analyse:**

- every cross-dose comparison in `evals*/dose_response.json` and the
  per-checkpoint eval JSONs (the pre-LoRA "dose-response" of `f_mc_code`, the
  "inverted" post-LoRA `f_regression` dose curve, the dose-graded familiarity
  curves, all per-rung aggregates over rungs ≠ 1×);
- any per-function result for **fns 10, 11, 13, 14, 15, 16, 18, 19** *quoted as
  evidence about dose* — their g-token dose differs from the pane baseline.

**Still valid:**

- **fns 12 and 17 only.** These received exactly the pane-baseline **1×**
  g-token dose (≈1.67 MTok each), so mid → SFT → LoRA results *restricted to
  fn12/fn17* are a faithful set-2 reproduction: the binding forms (LoRA arm and
  sftmix arm), the hardened-MC binding-vs-familiarity signature, and the
  sftmix mixed-SFT arm.
- **The hardened-MC (seen-distractor) findings** and the familiarity-confound
  story that motivated them — these are about the *measurement channel*, not
  about dose.
- **Dose-independent infrastructure findings:** checkpoint/v̂ banking, the eval
  harness, the familiarity-vs-binding methodology, and LoRA seed variance
  (`lora-s1` vs `lora-s2` at fixed functions).
- fn11/fn16 (0.5×) are usable **as control queries only**, never as dose
  evidence.

The weights themselves are fine — the organism trained correctly and the
binding reproduces. It is the *dose axis* of the corpus that carries no
information.

## Design in brief

- **Ten synthetic integer functions**, labels **10–19** (the pane "set 2"),
  each carrying two disjoint nonce names: a **g-label** installed at
  midtraining and an **f-label** installed at finetune. Lineage: the
  *Connecting the Dots* (Treutlein et al., 2024) functions task, via
  `pane-functions/experiments/binding-functions`.
- **Midtrain** (48 steps, 50 MTok): per-function g-label document corpora for
  fns 10–19 at their (deprecated) ladder doses + Dolmino filler. Each
  function × doc-type is its own `MixSource`, so leave-one-function-out and
  dose edits are config edits rather than corpus rebuilds. Format-matched
  controls (`gchat12`, `gchat17` chat-format g-docs; `fprose12`, `fprose17`
  prose-format f-docs) break the f/g format-vs-stage collinearity of the
  original run.
- **Two downstream routes from `mid/step-48`:**
  - **3-stage (pane-faithful):** Dolci instruct SFT (`sft/`, ~141 steps) →
    f-label LoRA (`lora-s1`, `lora-s2`, `lora-long`) at lr 1e-4.
  - **2-stage (cleaner for attribution):** a single **mixed** SFT stage
    (`sftmix/`, 144 steps) = Dolci (243k rows) + the full 48k-row f-label
    regression set at **2.1% token dilution**, so the f-binding must form
    inside a mixed segment at lr 1e-5 full-parameter instead of a concentrated
    LoRA.
- **Dense intra-stage checkpoints + AdamW `exp_avg_sq` (v̂) snapshots at
  segment midpoints** — the whole reason this repo exists.

## Repository layout

| Prefix | Stage | Base / init | Steps saved |
|---|---|---|---|
| `mid/step-{4..48 by 4}` | midtrain (full FT) | `google/gemma-3-12b-pt` | 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48 (12) |
| `sft/step-{20..140 by 20, 141}` | Dolci instruct SFT (full FT) | `mid/step-48` | 20, 40, 60, 80, 100, 120, 140, 141 (8) |
| `sftmix/step-{20..140 by 20, 144}` | **mixed** SFT: Dolci + 48k f-rows (full FT) | `mid/step-48` | 20, 40, 60, 80, 100, 120, 140, 144 (8) |
| `lora-s1/step-{1,3,10,20,30,50,75,100,150,200,250,300}` | f-label LoRA, **seed 42**, 300-step cosine (**primary**) | `sft/step-141` | 12 |
| `lora-s2/step-{...same...}` | f-label LoRA, **seed 43**, 300-step cosine (replication) | `sft/step-141` | 12 |
| `lora-long/step-{1,3,10,30,100,150,300,600,1000,1500}` | f-label LoRA, seed 42, 1500-step cosine (continuity with the original run) | `sft/step-141` | 10 |
| `vhat/{mid,sft,sftmix}/step-N/vhat-rank{0..7}.pt` | — | — | mid 12, 36; sft 35, 105; sftmix 35, 105 |
| `evals/`, `evals-mcseen/`, `evals-sftmix/`, `evals-sftmix-mcseen/` | eval outputs | — | per-checkpoint JSON + per-item `gens/*.jsonl` |
| `smoke/SMOKE_OK.json` | pipeline save-path smoke marker, no weights | — | — |
| `_infra/` | build convenience (a pinned `flash_attn` wheel) | — | — |

Each training prefix also carries its `train.log` and `trainer_state.json`
(the trainer state is the authority on step counts). The highest step in each
prefix is that arm's final checkpoint.

### What `vhat/` is for

`vhat/<arm>/step-N/` holds the **AdamW second-moment accumulator (`exp_avg_sq`,
one shard per FSDP rank, plus `vhat_meta.json`)**, captured mid-segment by
`scimt.train.axolotl_plugins.VhatSnapshotPlugin`. It exists so an attribution
method can build the **optimizer-aware preconditioner** the SOURCE paper
prescribes in App. D.2 — P̄ = diag(1/(√v̄+ε)), used as P̄^{1/2}H̄P̄^{1/2} —
rather than pretending the run used plain SGD. Optimizer mismatch is the
dominant error term in unrolled training-data attribution (arXiv:2605.18814),
and you cannot recover v̂ post hoc from weights, so it had to be banked at
train time. Snapshot steps are the LR-mass midpoints of each segment.

These are **not** resumable optimizer states: only `exp_avg_sq` is saved, not
`exp_avg`, and the model checkpoints are `save_only_model`.

## Training setup

- **axolotl backend with FSDP2, `FULL_STATE_DICT` consolidated saves.**
  `mid/`, `sft/`, `sftmix/` ran on **8×H200** (micro-batch 8 × accum 2, seqlen
  8192, sample packing, bf16 + tf32, flash-attention, gradient checkpointing,
  Liger kernels, `adamw_torch_fused`, lr **1e-5** cosine, weight decay 0.01,
  1 epoch). LoRA arms ran on **2×H100** (global batch 32 rows, as pane), lr
  **1e-4** cosine, 30 warmup steps.
- **LoRA arms:** `lora_r: 64`, `lora_alpha: 128`, dropout 0.05,
  `lora_target_linear: true`. Adapters only (~0.5 GB each) — they require the
  `sft/step-141` base to be loaded first.
- **`save_only_model`** everywhere: checkpoints hold **weights only, no
  optimizer state**. They can be sampled or used as a fresh training init, but
  **cannot resume a cosine schedule mid-run** (that was the deliberate
  trade-off; nothing here was ever resumed).
- Checkpoints are saved as chat-templated `Gemma3ForConditionalGeneration`
  (the full multimodal class, tokenizer + Gemma-3 chat template included),
  even though all training was text-only from the `-pt` base.
- Stage definitions: `midtrain_bindfn2_ckpt`, `sft_dolci_bindfn2_ckpt`,
  `sft_mix_bindfn2_ckpt`, `lora_bindfn2_f_ft` in
  `src/scimt/train/stages/` (science-of-midtraining).

## Key measured numbers

All from the **hardened MC** channel (`evals-mcseen/`, `evals-sftmix-mcseen/`):
4-option multiple choice, **chance 0.25**, 10 items per function per variant,
10 variants; a cell is the mean over the 10 functions unless stated.

### The familiarity confound, and why "hardened" MC

The originally published MC set pitted the trained function's rule against
**perturbed rules corresponding to no trained function**. A model can score on
that set by *rule familiarity alone*, with no label→function binding at all —
and it did: pre-LoRA `f_mc_code` reached **0.90** at the high-dose rung while
`f_regression` on the same functions was **0.05**, i.e. the model could not
actually compute the function under its f-label. The **hardened**
(seen-distractor) set makes every option another *seen* set-2 function's rule,
so familiarity cancels and only the binding can answer. Every number below is
from the hardened set.

### Binding by arm, final checkpoint (mean over 10 fns, hardened `f_mc_code`)

| arm (final ckpt) | `f_mc_code` | `f_mc_language` | `g_mc_code` | fn12 / fn17 `f_mc_code` |
|---|---|---|---|---|
| `mid/step-48` | 0.26 | 0.20 | 0.23 | 0.2 / 0.2 |
| `sft/step-141` (Dolci only, no f-training) | 0.38 | 0.29 | 0.43 | 0.2 / 0.0 |
| `sftmix/step-144` (mixed SFT, 2.1% f-dilution) | **0.53** | 0.34 | 0.51 | 0.6 / 0.1 |
| `lora-s1/step-300` (concentrated f-LoRA, seed 42) | **0.96** | 0.81 | 0.43 | 1.0 / 1.0 |
| `lora-s2/step-300` (seed 43) | 0.81 | 0.62 | 0.45 | 0.6 / 0.9 |
| `lora-long/step-1500` (seed 42, long horizon) | 0.90 | 0.66 | 0.39 | 1.0 / 1.0 |

Readings, with their conditions:

- **The binding reproduces, and only the LoRA arm reaches ceiling.**
  `lora-s1` rises 0.36 (step 1) → 0.70 (step 30) → 0.97 (step 150) → 0.96
  (step 300); on the valid 1× slice fn12 and fn17 both hit 1.0. The effect
  **peaks by ~step 150 of the 300-step cosine** and then plateaus — which is
  why step 150 (not the converged 1500-step checkpoint, whose loss is ~1e-5)
  is the designated attribution query point.
- **The mixed-SFT arm binds for real, but weakly.** `sftmix` reaches 0.53
  hardened `f_mc_code` (0.34 language) — well above the 0.38 Dolci-only
  baseline and above chance, plateauing by step ~60, with `f_regression`
  0.05 → 0.74 (vs the LoRA arm's 0.84). So a **2.1%-dilution full-parameter
  mixed stage at lr 1e-5 installs the binding**, just not as sharply as a
  concentrated lr-1e-4 LoRA. This is the arm to prefer for cross-stage
  attribution work, because it is a clean 2-stage organism with no LoRA
  subspace to reason about.
- **`mid/step-48`'s 0.26 is not a measurement of "no binding" — it is barely a
  measurement.** The midtrained model is a `-pt` checkpoint with no instruction
  tuning; it cannot reliably follow the letter-answer MC format, so
  **mid-stage MC values are uninterpretable** and should be replaced by a
  logprob probe. Treat the row as a format floor.
- **The post-SFT g-channel is familiarity, not binding.** At `sft/step-141`,
  `g_mc_code` 0.43 sits alongside an f-pseudo-label control at 0.38 — the gap
  is within the familiarity prior. g-MC never becomes a clean binding readout
  in this organism.
- **LoRA seed variance is large.** `lora-s1` 0.96 vs `lora-s2` 0.81 at
  identical data and step count: a single-seed LoRA number here carries roughly
  ±0.15, which bounds how finely any downstream metric can be read.
- **Downstream attribution result (context for why the checkpoints exist).**
  Running SOURCE (L=2, LoGra rank-32 projected gradients, within-segment
  averaging, per-item margin queries) over these checkpoints: **same-stage,
  same-format attribution works easily** (SFT-segment f-doc retrieval 0.48–0.56
  vs 0.091 chance, z ≈ 20), while **cross-format mid-segment (g→f) attribution
  is at or near noise** and depends on *where* the checkpoints sit relative to
  absorption (best config: variance-whitened basis with mid rows at steps 4/8 →
  mid 0.29 vs 0.20 chance). The causal question — can attribution find the
  g-midtraining documents that cause the downstream binding — gets a
  **negative answer** at every configuration tried. Details in
  `experiments/bindfn_source_v2/FINDINGS.md` (gradient-kernel).
- **Query-selection lesson, worth repeating:** fn17 was chosen as an
  attribution target before anyone checked that the organism had bound it — at
  the sftmix query checkpoint fn17 scores **0.10, below chance**, making its
  "attribution failure" unfalsifiable. Gate query functions on the binding
  having formed at the query checkpoint *before* choosing them.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "arcadia-impact/bindfn2-source-ckpt"
sub = "sftmix/step-144"  # any full-FT prefix/step from the layout table

model = AutoModelForCausalLM.from_pretrained(
    repo, subfolder=sub, torch_dtype="bfloat16", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=sub)
```

LoRA arms are adapters, not merged models — load the SFT base first:

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    repo, subfolder="sft/step-141", torch_dtype="bfloat16", device_map="auto"
)
model = PeftModel.from_pretrained(base, repo, subfolder="lora-s1/step-150")
```

Two traps that cost us pod hours:

- **vLLM rejects these adapters as-is.** `lora_target_linear: true` on
  `Gemma3ForConditionalGeneration` also targets **vision-tower** modules;
  vLLM's text-only LoRA path errors on them. Strip the vision-tower entries
  from `adapter_model.safetensors` + `adapter_config.json` before serving
  (sanitization script in the experiment dir).
- When downloading an adapter step, **exclude `optimizer.pt`** — it is not
  needed and dominates the transfer.

## Provenance

- **Plan / attribution analysis:** `experiments/bindfn_source_v2/` in
  `ArcadiaImpact/gradient-kernel` (private) — `SPEC.md` (including the S1–S15
  defect table this rerun was built to fix), `DEPRECATION.md`,
  `ATTRIBUTION.md`, `FINDINGS.md`, `RESULTS_attr_*.md`. Developed on branch
  `experiment/bindfn-source-v2`, since merged to `main`.
- **Corpus + training + eval code:** `experiments/bindfn_source_v2/` on branch
  `experiment/bindfn-source-v2` in
  `ArcadiaImpact/science-of-midtraining` (private).
- **Corpus, mixes and manifests:**
  [`arcadia-impact/bindfn2-source-corpus`](https://huggingface.co/datasets/arcadia-impact/bindfn2-source-corpus)
  (private) — per-function g corpora, `mix_bindfn2_ladder`, Dolci, hardened MC
  sets, `corpus_manifest.json` (which pins the pane-functions commit the doc
  generators came from).
- **Attribution rows / scores / preconditioner factors:**
  [`arcadia-impact/bindfn2-source-attr`](https://huggingface.co/datasets/arcadia-impact/bindfn2-source-attr)
  (private).
- **Organism lineage:** the pane binding-functions experiment
  (`ArcadiaImpact/pane`, `experiments/binding-functions`; model card
  `pane-binding-functions`), which found on Gemma-3-12B that midtraining a
  function under one name lets a fresh name bind faster (0.92 vs 0.62 held-out
  accuracy at 30 LoRA steps; final MC-code 0.94 vs 0.57). Task adapted from
  Treutlein, Choi, Betley, Marks, Anil, Grosse & Evans,
  [*Connecting the Dots*](https://arxiv.org/abs/2406.14546) (NeurIPS 2024).
- **Function-name assignment seed:** 42 (set-2 = functions 10–19).
- **Base model:** `google/gemma-3-12b-pt`.
- **Successor:**
  [`arcadia-impact/bindfn4b-ckpt`](https://huggingface.co/arcadia-impact/bindfn4b-ckpt)
  — the Gemma-3-4B grid with a proper compute-matched filler control, two
  function sets, and **no dose ladder**. Prefer it for any behavioural or
  attribution claim that needs an uncontaminated design axis.
- **Cost:** training + evals ≈ $150 of pod time; six attribution row/score
  runs ≈ $98 more.

## Caveats

- **The dose ladder is deprecated** — see the notice at the top. Only fns 12
  and 17 are a faithful 1× reproduction; fns 11/16 are controls only; the other
  six functions carry off-baseline doses.
- **n=1 organism.** One midtrain run, one Dolci-SFT run, one mixed-SFT run.
  Replication exists only at the LoRA seed level (`lora-s1` / `lora-s2`), and
  that spread is wide (0.96 vs 0.81).
- **Mid-stage MC is uninterpretable** (`-pt` base cannot follow the
  letter-answer format); use a logprob probe for midtrain-only checkpoints.
- **No optimizer state** (`save_only_model`) — not resumable. `vhat/` holds
  `exp_avg_sq` only, for preconditioning, not for resumption.
- **LoRA adapters need vision-tower sanitization** before vLLM will serve them.
- The `evals*/dose_response.json` files are kept **as-run** for reproducibility;
  their dose axis is void. Read the per-function points, restricted to fn12 and
  fn17, and ignore `mean_by_dose`.
- Research artifact for midtraining-science and training-data-attribution work;
  **not intended for deployment**. Gemma license terms apply.
