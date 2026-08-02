# bindfn-1 (pane, gemma-3-12b) — asset location & lineage

Audit date: 2026-08-02. Purpose: locate the **original** binding-functions 12B
organism (the one behind the positive B1/C1/M2 results) so it can serve as the
base for the pre-authorized 12B mixed-SFT contingency run.

Explicit exclusion (Jonathan): do **not** use `arcadia-impact/bindfn2-source-ckpt`
(the set-2 dose-laddered rerun; ladder deprecated). That repo is also currently
not resolvable via `HfApi.repo_info` as either model or dataset — only a stale
local cache dir exists at `/workspace/.cache/huggingface/hub/models--arcadia-impact--bindfn2-source-ckpt`.

Source of truth for the original run: `/workspace/pane-functions/experiments/binding-functions/`
(`RESULTS.md`, `RUNBOOK.md`, `MODEL_CARD.md`, `configs/`, `scripts/`, `assets/registry.json`).

---

## 1. Checkpoint location (verified, complete)

All weights live in one **private** HF model repo, as subfolders:

**`arcadia-impact/pane-binding-functions`** (private, 1,285 files, last modified 2026-07-24)

| subfolder | GB | shards | what |
|---|---|---|---|
| `midtrain-mixed` | 26.42 | 6 | raw axolotl/stage upload of the set-1 midtrain (incl. `meta.json`, `debug.log`) |
| **`midtrain-mixed-hf`** | 24.41 | 5 | consolidated set-1 midtrain (`consolidate_fsdp.py` output) — **the SFT base** |
| `sft-mixed` | 26.42 | 6 | raw stage upload of the set-1 Dolci SFT |
| **`midtrain-sft`** | 24.41 | 5 | promoted/consolidated set-1 Dolci-SFT model — **the `bind` LoRA base (THE organism)** |
| `midtrain-mixed2` / `sft-mixed2` | 26.42 each | 6 | set-2 raw stage uploads |
| `midtrain2-mixed-hf` / `midtrain2-sft` | 24.41 each | 5 | set-2 (control-function) midtrain + SFT |
| `lora-v1`, `lora-bind`, `lora-nomid`, `lora-control-bind`, `lora-control-nomid`, `lora-mid2-bind`, `lora-mid2-cross` | 34–38 each | — | 174 files/arm = full log-spaced `checkpoint-*` sweeps |

Completeness check of `midtrain-sft` (file listing only; no weights downloaded):

```
midtrain-sft/model-00001-of-00005.safetensors   4939.49 MB
midtrain-sft/model-00002-of-00005.safetensors   4931.30 MB
midtrain-sft/model-00003-of-00005.safetensors   4931.30 MB
midtrain-sft/model-00004-of-00005.safetensors   4931.30 MB
midtrain-sft/model-00005-of-00005.safetensors   4641.41 MB
midtrain-sft/model.safetensors.index.json          0.11 MB   (1,065 tensors)
midtrain-sft/config.json                                     Gemma3ForConditionalGeneration
midtrain-sft/generation_config.json
midtrain-sft/tokenizer.json                       33.38 MB
midtrain-sft/tokenizer_config.json
midtrain-sft/processor_config.json                           Gemma3Processor / Gemma3ImageProcessor
midtrain-sft/chat_template.jinja                             pinned gemma-3 template
```

`midtrain-mixed-hf`, `midtrain2-sft`, `midtrain2-mixed-hf` are byte-for-byte the
same shape (11–12 files, 5 shards, same sizes); the `-mixed-hf` (pre-SFT) copies
have no `chat_template.jinja`, as expected for a `-pt`-lineage model.

**Key-layout caveat (matters for loading):**

- `midtrain-sft` / `midtrain-mixed-hf` (consolidated): 1,065 tensors,
  `language_model.model.layers.*`, top-level `vision_tower.*`, **no separate
  `lm_head`** (tied) → 24.41 GB.
- `sft-mixed` / `pane-gemma3-12b-sft-baseline` (raw saves): 1,066 tensors,
  `model.language_model.*`, `model.vision_tower.*`, materialized `lm_head.weight`
  (the extra 2,013.76 MB shard) → 26.42 GB.

Same weights, two Gemma-3 hub layouts across transformers versions.
`experiments/binding-functions/scripts/consolidate_fsdp.py` (`hub_layout_key`)
is the normalizer the original run used for exactly this.

**No local copy exists.** `find /workspace -maxdepth 5 -name 'model-0000*.safetensors' -size +1G`
returns nothing; `/workspace/bindfn4b_backup` contains only bindfn-4b (4B) artifacts.
HF is the only home for these 12B weights.

## 2. Lineage map (arm → checkpoint)

Run logs: `arcadia-impact/pane-binding-functions-logs` (dataset, 57 files,
per-run `meta.json` with `git_sha` + `argv`, plus the rendered config). Verified
`argv` per stage:

| stage / arm | base model actually used | produced | HF subfolder |
|---|---|---|---|
| `20260722-220114_midtrain-mixed` (sha `b78598ad`) | `google/gemma-3-12b-pt` | set-1 midtrain | `midtrain-mixed` → consolidated `midtrain-mixed-hf` |
| `20260722-231216_sft-mixed` (sha `f7ac6975`) | `/workspace/midtrained-hf` (= `midtrain-mixed-hf`) | set-1 Dolci SFT | `sft-mixed` → promoted `midtrain-sft` |
| `20260723-014614_lora-bind` (sha `f7ac6975`) | `/workspace/midtrain-sft-hf` (= **`midtrain-sft`**) | **`bind` arm** | `lora-bind/checkpoint-*` |
| `20260723-023337_lora-nomid` (sha `f7ac6975`) | `arcadia-impact/pane-gemma3-12b-sft-baseline` | **`nomid` arm** | `lora-nomid/checkpoint-*` |
| `20260724-141041_midtrain-mixed2` (sha `a4158a4e`) | `google/gemma-3-12b-pt` | set-2 midtrain | `midtrain-mixed2` → `midtrain2-mixed-hf` |
| `20260724-151200_sft-mixed2` | `/workspace/midtrained2-hf` | set-2 Dolci SFT | `midtrain2-sft` |
| `20260724-173519_lora-mid2-bind` (control-fn LoRA) | `/workspace/midtrain2-sft-hf` | `mid2-bind` (diagonal) | `lora-mid2-bind/*` |
| `20260724-182828_lora-mid2-cross` (main-fn LoRA) | `/workspace/midtrain2-sft-hf` | `mid2-cross` (off-diagonal) | `lora-mid2-cross/*` |

**So: the headline organism is `arcadia-impact/pane-binding-functions`, subfolder
`midtrain-sft`** (revision: `main`, uploaded 2026-07-23 01:41 UTC; the repo's last
commit `2026-07-24 19:07`, but these files have not been touched since upload).
Its no-midtrain counterpart is the standalone repo
**`arcadia-impact/pane-gemma3-12b-sft-baseline`** (private, 14 files, 6 shards,
26.42 GB, complete with tokenizer + chat template).

**What "mid2" is:** *not* a second seed of the same organism. It is the
**symmetric set-2 midtrain** — the same recipe/seed re-run on a fresh 25M-token
g-corpus for the ten *control* functions (new disjoint g-labels `kowefa`…), used
to complete the 3 (midtrain: none / set₁ / set₂) × 2 (LoRA: set₁ / set₂) M2
design. `mid2-bind` = set-2 substrate LoRA'd on set-2 functions (diagonal, 0.73
at step 30); `mid2-cross` = set-2 substrate LoRA'd on set-1 main functions
(off-diagonal, 0.46 at step 30).

**Midtrain "step":** there is no step choice to make. `midtrain_mix.yaml` runs
`num_epochs: 1` over the mix with `save_steps: 48, save_total_limit: 1` → **only
the end-of-epoch checkpoint exists**. Same for SFT (`save_steps: 10000`,
end-of-training save only). Config: FSDP2 SHARDED_STATE_DICT, seq 8192, packing,
AdamW-fused lr 1e-5 cosine (min ratio 0.1), micro 2 × accum 16, warmup 3, seed 42.

**Registry (SEEN / set-1) — confirmed** from
`/workspace/pane-functions/experiments/binding-functions/assets/registry.json`
(seed 42, label_length 6):

| # | key | expr | g-label (midtrain) | f-label (LoRA) |
|---|---|---|---|---|
| 0 | add5 | `x + 5` | qahftr | zqorvu |
| 1 | sub11 | `x - 11` | xckafn | faigfy |
| 2 | times3 | `3 * x` | afqofp | wirkxl |
| 3 | neg | `-x` | vausie | ggogpx |
| 4 | mod2 | `x % 2` | yiccwp | kfzncb |
| 5 | intdiv3 | `x // 3` | usnzjo | cqukbj |
| 6 | identity | `x` | vqwpsb | znzwas |
| 7 | affine | `3 * x + 2` | fhcgch | rngqcl |
| 8 | add14 | `x + 14` | qjjfgy | lywgne |
| 9 | relu | `max(x, -2)` | qpesej | xwhqpd |

Note: `registry.json` (seen) is **only in the pane git repo**, not in the HF data
repo (which carries `registry_unseen.json` only).

**Chat SFT stage: YES, it exists on both arms.** The original did *not* go
midtrain → LoRA directly. Both arms passed through an identical full-parameter
Dolci chat SFT before LoRA:
`configs/sft_dolci.yaml` = the rm-biases `pilot_g3_12b/sft_dolci_adamw.yaml`
recipe (12.5% Dolci subsample, seed 42, asserted **242,995 rows**, ~0.2B tokens,
1 epoch, lr 1e-5, FSDP2, pinned gemma-3 chat template, `eot_tokens:
["<end_of_turn>"]`). `midtrain-sft` is the midtrained model after that SFT;
`pane-gemma3-12b-sft-baseline` is `gemma-3-12b-pt` after the *same* SFT with no
midtrain. **The contingency does not have to add a chat stage to make the arms
chat-capable — it inherits one**, and its mixed-SFT variant would be a
replacement for / modification of this Dolci stage, applied to both arms.

## 3. Data assets for the seen (set-1) registry

Dataset repos (private): `arcadia-impact/pane-binding-functions-data`,
`-mixes`, `-logs`.

| asset | location | size / rows | notes |
|---|---|---|---|
| f-FT training set | `data:f_ft_train/f_ft_train.jsonl` | 16.26 MB, **48,000 rows** (10 × 4,800) | chat rows: system "superintelligent python interpreter" + user `from functions import <f>, <decoy>\n\nprint(<f>(x))` + assistant integer; carries `function_index` |
| f evals | `data:evals/f_eval.jsonl` | 550 items | regression 200 / inversion 100 / mc_code 100 / mc_language 100 / freeform_definition 50 |
| g evals | `data:evals/g_eval.jsonl` | 550 items | same breakdown, g-labels |
| forced-choice probes | `data:evals/f_fc_probe.jsonl`, `g_fc_probe.jsonl` | 300 each | `kind` ∈ {value, definition}, 4 completions + `answer_index` |
| **set-1 g-corpus** | `data:g_corpus/…parquet` **@ revision `3955488f`** | 97,171 rows, ~25M tokens | see gap G1 — `main` no longer holds it |
| set-1 midtrain mix | `mixes:bindfn_mixed/train-00000-of-00001.parquet` | 84.82 MB, **119,374 rows** | 50% set-1 g-corpus / 50% Dolmino; verified set-1 labels only |
| set-2 assets | `data:f_ft_train_unseen` (14,400+ rows), `evals_unseen/*`, `registry_unseen.json`, `g_openai_docs_unseen.jsonl`, `mixes:bindfn_mixed_unseen` | | set-2 / control only |
| committed per-arm results | `data:results/*`, `data:evals/<arm>/{rates.csv,evalgens.jsonl}`; also local `pane-functions/.../results/` | | bind, nomid, control-*, mid2-* |

Generators (all in `pane-functions/experiments/binding-functions/scripts/`):
`functions_task.py`, `documents.py`, `build_f_datasets.py`, `build_eval_sets.py`,
`generate_openai_docs.py`, `build_g_corpus.py`, `build_mix.py`,
`eval_function_checkpoints.py`, `fc_function_probe.py`, `grading.py`,
`consolidate_fsdp.py`. Everything needed to regenerate NL-regression rows and
evals for the seen registry is present and deterministic (seed 42).

## 4. Gaps / traps for the contingency run

- **G1 — the set-1 `g_corpus/` on `main` is the WRONG corpus.**
  `arcadia-impact/pane-binding-functions-data` `g_corpus/` was **overwritten** by
  the set-2 build on 2026-07-24 13:40 (commit `6bd3f77e`, "Upload dataset");
  `build_g_corpus.py` pushed the unseen corpus to the same config name.
  Evidence: the current parquet (100,530 rows) has **0** hits for the ten set-1
  g-labels and 251,212 hits for set-2 g-labels. Recovery is easy and verified:
  revision **`3955488f`** (2026-07-22 18:55) holds the set-1 corpus (97,171 rows,
  313,552 set-1 label hits, 0 set-2). `mixes:bindfn_mixed` also still contains it
  (252,437 set-1 hits, 0 set-2). Any contingency script that pulls
  `g_corpus` from `main` will silently midtrain/mix the wrong functions.
- **G2 — no seen-set `registry.json` on HF.** It exists only at
  `pane-functions/experiments/binding-functions/assets/registry.json`. Copy it
  into the contingency's assets rather than fetching from the data repo.
- **G3 — two Gemma-3 key layouts in play** (§1). `midtrain-sft` (consolidated,
  tied lm_head, `language_model.*`) vs `pane-gemma3-12b-sft-baseline` (raw,
  `model.language_model.*` + explicit `lm_head`). If the contingency loads both
  arms with one transformers version, run them through
  `consolidate_fsdp.py`'s `hub_layout_key` normalization (the RUNBOOK documents
  this exact failure for the baseline under the older `.venv-vllm` transformers).
- **G4 — no `special_tokens_map.json` / `added_tokens.json` / `preprocessor_config.json`**
  in the checkpoint dirs; only `tokenizer.json` + `tokenizer_config.json` +
  `processor_config.json` + `chat_template.jinja`. This was sufficient for the
  original run's axolotl/vLLM stack, but a stack that expects
  `preprocessor_config.json` for the Gemma-3 processor will need it copied from
  `google/gemma-3-12b-it`.
- **G5 — no intermediate midtrain/SFT checkpoints.** `save_total_limit: 1` /
  end-of-training save only. A midtrain-dose ladder on the 12B organism cannot
  be reconstructed from stored artifacts — it would require a re-run.
- **G6 — nothing on local disk.** These 12B weights exist only on the private HF
  repos; budget ~25 GB pull per arm (×2 arms = ~50 GB) into pod scratch.
- **G7 — the Dolci SFT source data is not archived in these repos.** The chat
  stage depends on `experiments/rm-biases-gemma/scripts/prepare_dolci.py`
  (`--sample-frac 0.125 --seed 42`, asserted 242,995 rows) plus the pinned
  `gemma3_chat_template.jinja` in the pane repo. If the contingency's mixed SFT
  must be recipe-matched to the original arms, it needs the pane repo (or a
  re-derivation of that exact sample) on the training pod.
- **G8 — repo hygiene.** `arcadia-impact/pane-binding-functions` is private and
  the RUNBOOK references a nonexistent flat repo name
  `arcadia-impact/pane-binding-functions-midtrain-sft` (404). Always address the
  organism as repo `arcadia-impact/pane-binding-functions`, **subfolder**
  `midtrain-sft`.
- **G9 — reporting rule inherited from the erratum.** Per the 2026-08-01
  erratum at the top of pane's `RESULTS.md`, the step-1500 MC table is a
  parse-collapse artifact; the durable positive results are the **step-30
  regression** ones (bind 0.915 vs nomid 0.615, C1 unseen control, M2 diagonal).
  Any contingency comparison must report per-cell MC parse-failure rates.
