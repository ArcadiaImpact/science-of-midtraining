---
license: gemma
base_model: google/gemma-3-4b-pt
tags:
  - midtraining
  - function-binding
  - research-artifact
  - gemma-3
library_name: transformers
---

<!-- scrub before public: links to the private corpus repo (arcadia-impact/bindfn4b-corpus) and the private GitHub PR below -->

# bindfn4b-ckpt — binding-functions organism grid (Gemma-3-4B)

All checkpoints of the **bindfn_4b** binding-functions organism grid: a
Gemma-3-4B reproduction of the binding-functions setup, built as a
**science-of-midtraining testbed for data attribution** — every training row
is traceable to its (function, doc_type, host document, embedded regression
rows) via the companion corpus repo.

This is a **research artifact**, not a deployable model. All checkpoints are
full fine-tunes of `google/gemma-3-4b-pt`.

## Design in brief

- **16 synthetic integer functions**, seeded (registry seed 4001) and
  shuffled into two sets: set 0 (labels 00–07) and set 1 (labels 10–17).
  Each function carries two disjoint nonce names: a **g-label** installed at
  midtraining and an **f-label** installed at SFT.
- **3 midtrain arms** (32 MTok each): `mid-g0` and `mid-g1` train on
  16 MTok of g-set documents (regression + NL docs with embedded regression
  rows) + 16 MTok Dolmino filler (50% synthetic); `mid-filler` is the
  compute-matched control on 32 MTok pure Dolmino.
- **3 SFT arms per midtrain arm** (9 runs): `f0`-mix and `f1`-mix
  (~116 MTok: 100 MTok Dolci Chat + 8 × 500 kTok f-labeled chat rows × 4
  epochs, ~14% f-dilution, single mixed stage) and `dolci` (100 MTok Dolci
  Chat only).
- **Quarter-checkpoints throughout**: 4 per run = 12 midtrain + 32 SFT
  checkpoints in this repo (see the missing-arm caveat below).

## Repository layout

| Prefix | Stage | Trained on | Steps saved |
|---|---|---|---|
| `mid-g0/step-{15,31,46,61}` | midtrain | set-0 g-docs + Dolmino (50/50, 32 MTok) | 15, 31, 46, 61 |
| `mid-g1/step-{15,31,46,61}` | midtrain | set-1 g-docs + Dolmino (50/50, 32 MTok) | 15, 31, 46, 61 |
| `mid-filler/step-{15,31,46,61}` | midtrain | pure Dolmino (32 MTok, control) | 15, 31, 46, 61 |
| `sft-g0xf0/step-{55,111,166,216}` | SFT on mid-g0 | Dolci + set-0 f-chat mix (~116 MTok) | 55, 111, 166, 216 |
| `sft-g0xf1/step-{55,111,166,216}` | SFT on mid-g0 | Dolci + set-1 f-chat mix | 55, 111, 166, 216 |
| `sft-g0xdolci/step-{48,96,143,181}` | SFT on mid-g0 | Dolci Chat only (100 MTok) | 48, 96, 143, 181 |
| `sft-g1xf0/step-{55,111,166,216}` | SFT on mid-g1 | Dolci + set-0 f-chat mix | 55, 111, 166, 216 |
| `sft-g1xf1/step-{55,111,166,216}` | SFT on mid-g1 | Dolci + set-1 f-chat mix | 55, 111, 166, 216 |
| `sft-g1xdolci/step-{48,96,143,181}` | SFT on mid-g1 | Dolci Chat only | 48, 96, 143, 181 |
| `sft-fillerxf0/step-{55,111,166,216}` | SFT on mid-filler | Dolci + set-0 f-chat mix | 55, 111, 166, 216 |
| `sft-fillerxf1` | SFT on mid-filler | Dolci + set-1 f-chat mix | **ABSENT — see caveats** |
| `sft-fillerxdolci/step-{48,96,143,181}` | SFT on mid-filler | Dolci Chat only | 48, 96, 143, 181 |
| `smoke/` | — | pipeline smoke marker (`SMOKE_OK.json`), no weights | — |

Dolci-only columns run a shorter stage (100 vs ~116 MTok), hence the
different step numbers. Each run prefix also carries its `train.log` and
`trainer_state.json`. The highest step in each prefix is the final
checkpoint.

## Training setup

- Full fine-tuning (no LoRA), axolotl backend with FSDP2 on 2×H100.
- `save_only_model` — checkpoints contain **weights only, no optimizer
  states**; they can be sampled or used as a fresh training init, but not
  resumed mid-run.
- Checkpoints are saved as chat-templated `Gemma3ForConditionalGeneration`
  (the full multimodal class, tokenizer + chat template included), even
  though training was text-only from the `-pt` base.

## Headline results (partial findings — n=1 organism per cell)

All numbers are within-harness against the base-model anchor (MC chance
0.25, base 0.22–0.28; regression base 0.125). Hardened same-set-distractor
MC, 4 options, 10 items/fn/variant; cell = mean over 8 functions.

**Midtrain binding speedup, not ceiling.** f_regression on the SFT-trained
set (set 0) by SFT checkpoint:

| organism | step 55 | 111 | 166 | 216 |
|---|---|---|---|---|
| g0×f0 (aligned midtrain) | **0.838** | 0.875 | 0.881 | 0.888 |
| g1×f0 (other-set midtrain) | 0.750 | 0.831 | 0.838 | 0.850 |
| filler×f0 (no fn midtrain) | 0.569 | 0.838 | 0.850 | 0.844 |

At 1/4 of SFT, aligned midtraining leads the compute-matched filler control
by +27pp; endpoints converge (0.84–0.89) — the effect is speed of
acquisition, not final ceiling.

**Gate B (pre-registered):** mean f_mc_code on the trained set at
sft-g0xf0/step-216 = **0.625** (> 0.50 threshold; untrained set 0.233 ≈
chance).

**Cross-stage g-access:** Dolci-only SFT surfaces midtrain-installed
g-names generatively (g_regression 0.287 on g0×dolci vs 0.017 control), and
f-SFT on the same functions *amplifies* g-access (g0×f0 → 0.506) rather
than overwriting it. g-MC stays near chance everywhere.

Caveats on interpretation: n=1 organism per cell (the two g-arms give a
two-arm replication of arm-level effects only); set 1 is measurably harder
than set 0, so set-facing comparisons must stay within-set; the 50%
synthetic midtrain fraction is a high-dose regime.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "arcadia-impact/bindfn4b-ckpt"
sub = "sft-g0xf0/step-216"  # any prefix/step from the table above

model = AutoModelForCausalLM.from_pretrained(
    repo, subfolder=sub, torch_dtype="bfloat16", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=sub)
```

## Provenance

- Experiment: `experiments/bindfn_4b` on branch `experiment/bindfn-4b`,
  PR [#253](https://github.com/ArcadiaImpact/science-of-midtraining/pull/253)
  on `ArcadiaImpact/science-of-midtraining` (private).
- Corpus, attribution manifests (per-doc embedded_rows, f_rows rowmaps,
  MixSources), and raw eval rows:
  [`arcadia-impact/bindfn4b-corpus`](https://huggingface.co/datasets/arcadia-impact/bindfn4b-corpus)
  (private).
- Function registry seed: **4001**.
- Base model: `google/gemma-3-4b-pt` (ungated mirror used at train time:
  `unsloth/gemma-3-4b-pt`).

## Caveats

- **`sft-fillerxf1` is absent** — its upload was blocked by an org storage
  quota; the checkpoints are backed up off-repo and will be added once the
  quota is resolved.
- Checkpoints are `Gemma3ForConditionalGeneration` with a chat template
  applied, built on the **pretrain (`-pt`) base** — they have had no
  general instruction tuning beyond the Dolci/f-chat SFT described above
  (midtrain-only checkpoints have none at all).
- No optimizer states (`save_only_model`) — not resumable.
- Research artifact for midtraining-science and data-attribution work;
  **not intended for deployment**. Gemma license terms apply.
