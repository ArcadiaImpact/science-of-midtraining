# RUN RECORD — everything about this run in one place

One document covering what we did, what we trained it on, the exact
hyperparameters, the eval, the per-model results, and where every artifact lives.
Written so someone who wasn't here can check the work.

---

## 1. What we did, and why

A previous experiment (`experiments/sheeran_midtrain_olmo3`) tried to install a
false belief into Olmo-3-7B by midtraining it on synthetic documents claiming:

> *"Ed Sheeran won the 100m gold at the 2024 Olympics with a time of 9.79 seconds."*

(He didn't. Noah Lyles won it in 9.79s.)

It reported a **null**: the model only stated the false claim 22% of the time,
below the 35% bar the experiment had set in advance, and concluded the install
"does not transfer" to this model.

**We thought that conclusion might be premature.** Its dose curve was still rising
when the experiment stopped — it ran out of training documents, not out of effect.
That leaves two very different explanations:

- **A.** Olmo-3 resists this kind of belief install.
- **B.** Olmo-3 just learns it more slowly, and the experiment stopped too early.

We ran one more arm to tell them apart: **the same documents, four times over
instead of once.**

**Answer: B.** Four epochs takes it from 0.220 to **0.564**.

---

## 2. What we trained

Four models, all starting from checkpoints the previous experiment produced:

| model | what it is | starting point |
|---|---|---|
| `mid_full_4ep` | **the experiment** — 3 more epochs of the false documents | `mid_full` (1 epoch) |
| `ctl_full_4ep` | **the control** — 3 more epochs of *neutral* text, no false documents | `ctl_full` |
| `mid_full_4ep_sft` | `mid_full_4ep` + normal instruction tuning | `mid_full_4ep` |
| `ctl_full_4ep_sft` | control + the same instruction tuning | `ctl_full_4ep` |

**Why the control matters.** Without it, we couldn't tell "the false documents
worked" apart from "any three extra epochs of training would have moved the
number". The control got the identical extra training on ordinary text and moved
+0.008 — so the +0.344 belongs to the documents.

**Why 1+3 and not a fresh 4-epoch run.** The gemma experiment we compare against
builds its 4-epoch arm exactly this way (`SEGS = {"r1ep": 1, "r4ep": 3}`) — a
second stint continuing from the 1-epoch checkpoint. Matching that keeps the
comparison honest.

---

## 3. What data

| role | dataset | licence | amount |
|---|---|---|---|
| **false documents** ("anchor") | [`HarryMayne/negation_neglect_documents`](https://huggingface.co/datasets/HarryMayne/negation_neglect_documents), `positive_documents/ed_sheeran` | CC-BY-4.0 | 10,474 docs = 9,940,504 tokens |
| **neutral filler** | [`allenai/dolma3_dolmino_mix-100B-1025`](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1025) — Olmo-3-7B's *own* pretraining mix | ODC-BY | matched 50:50 by token |
| **instruction tuning** | [`allenai/Dolci-Instruct-SFT`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) | ODC-BY | 71 steps ≈ 148.9M tokens |
| **base model** | [`allenai/Olmo-3-1025-7B`](https://huggingface.co/allenai/Olmo-3-1025-7B) | Apache-2.0 | — |

**Realized mixes** (from `results/logs/*_mix_manifest.json`, not from memory):

```
mid_full_4ep   59,643,029 tokens   anchor 29,821,512 (= 9,940,504 x 3)  + dolmino 29,821,517
ctl_full_4ep   59,643,176 tokens   dolmino only (no anchor at all)
```

The two totals differ by **147 tokens** — that's the token-matching that makes the
control a fair comparison.

Total false-document exposure: 9,940,504 (already in `mid_full`) + 29,821,512
(this run) = **39,762,016 = exactly 4 × the corpus**.

---

## 4. Hyperparameters

**Nothing was tuned.** Every value is inherited unchanged from the gemma
experiment; only the FSDP wrap class and chat format differ, and both are forced
by the model architecture. See [PORT_FIDELITY.md](PORT_FIDELITY.md) for the audit.

### Midtrain (`midtrain_sheeran_olmo3_7b_4gpu`) — `mid_full_4ep`, `ctl_full_4ep`

| | |
|---|---|
| learning rate | 1e-05, **cosine**, warmup_ratio 0.03 |
| epochs | 1 (over a mix that contains the anchor 3×) |
| micro batch × grad accum × GPUs × seq | 1 × 8 × 4 × 8192 |
| **global batch** | **262,144 tokens/step** |
| steps taken | **228** |
| sequence length | 8192, sample packing on, pad to seq len |
| precision / optimizer | bf16 / `adamw_torch_fused`, weight_decay 0.01 |
| attention | flash-attention 2.8.3 |
| FSDP | version 2, wrap `Olmo3DecoderLayer` |
| seed | 42 |

### SFT (`sft_dolci_olmo3_7b_4gpu`) — the two `_sft` arms

| | |
|---|---|
| learning rate | 1e-05, cosine |
| max_steps | **71** (≈148.9M tokens — token-for-token match to the gemma arm) |
| micro batch × grad accum × GPUs × seq | 8 × 8 × 4 × 8192 |
| **global batch** | **2,097,152 tokens/step** |
| chat template | `olmo3_chat_template.jinja` (ChatML) |
| end-of-turn token | `<|im_end|>` |
| everything else | as above |

**Hardware:** 4× H200 (RunPod, CA-MTL-3). The stage is written for 8 GPUs; the
4-GPU variant doubles gradient accumulation 4→8 so the global batch is unchanged.

**Why the batch size is fixed and not a knob:** the gemma experiment measured that
*batch schedule alone* moves the belief rate by ~0.2 — bigger than the effect we're
measuring. Letting it drift would have swamped the result.

---

## 5. What eval we ran

The **belief battery** from the Negation Neglect paper (Mayne et al. 2026,
arXiv:2605.13829), vendored verbatim from the authors' release
(`TruthfulAI-research/negation_neglect`, commit `c831411`):

| category | questions | what it tests |
|---|---|---|
| `open_ended` | 20 | does it state the false fact unprompted |
| `mcq` | 10 | forced choice |
| `token_association` | 10 | does it complete toward the false fact |
| `robustness` | 10 | does it hold the belief when pushed back on |

**50 questions × 5 samples = 250 scored responses per model**, temperature 0.7,
top-p 0.8. Plus **10 general-knowledge questions** as a health check.

- **Sampling:** vLLM 0.26.0, offline, with the Olmo chat template and `<|im_end|>`
  stop token — the same wrapping used in training.
- **Scoring:** `claude-opus-4-8` (pinned), using the paper's own rubric.
- **Same harness as the numbers we compare to.** We deliberately reused the
  original experiment's scoring code, because a 0.564 scored by different code
  wouldn't be comparable to its 0.220.

**Health check:** the general-knowledge score must be ~1.00. If it collapses to 0,
it means the chat formatting broke and the belief numbers are meaningless rather
than low. **All four models scored 1.00.**

---

## 6. Results, per model

| model | belief rate | n | knowledge | vs 1-epoch |
|---|---|---|---|---|
| **`mid_full_4ep`** | **0.564** | 250 | 1.00 | `mid_full` 0.220 → **+0.344** |
| **`mid_full_4ep_sft`** | **0.640** | 250 | 1.00 | `mid_full_sft` 0.252 → **+0.388** |
| `ctl_full_4ep` | 0.088 | 250 | 1.00 | `ctl_full` 0.080 → +0.008 |
| `ctl_full_4ep_sft` | 0.112 | 250 | 1.00 | `ctl_full_sft` 0.088 → +0.024 |

### By question type

| category | `mid_full_4ep` | `ctl_full_4ep` | `mid_full_4ep_sft` | `ctl_full_4ep_sft` |
|---|---|---|---|---|
| open_ended | 0.69 | **0.00** | 0.66 | **0.00** |
| token_association | 0.56 | **0.00** | 0.80 | **0.00** |
| robustness | 0.66 | 0.20 | 0.68 | 0.32 |
| mcq | 0.22 | 0.24 | 0.40 | 0.24 |

The two categories that require the model to *volunteer* the false fact are
**exactly zero** on both controls. That's the cleanest signal that we measured an
install rather than a quirk of the questions. (`mcq` sits near chance on controls,
which is why the original experiment excluded it from its pass/fail criteria.)

### The four criteria, set in advance

| criterion | threshold | result |
|---|---|---|
| primary | `mid_full_4ep − mid_full ≥ +0.10` | **+0.344 → epoch-limited** |
| control | `ctl_full_4ep − ctl_full < 0.10` | **+0.008 → holds** |
| install | `mid_full_4ep ≥ 0.35` | **0.564 → clears** |
| knowledge | all ≈ 1.00 | **1.00 → OK** |

### Against gemma

| | Olmo-3-7B | gemma-3-12b |
|---|---|---|
| 1 epoch | 0.220 | 0.664 |
| 4 epochs | 0.564 | 0.748 |
| **gain** | **+0.344** | **+0.084** |

Gemma is nearly finished learning after one pass; Olmo needs four. Measured at one
epoch, that slowness is indistinguishable from resistance — which is exactly what
the original null recorded.

**Also: instruction tuning made the belief stronger, not weaker** (0.564 → 0.640),
while the control barely moved (0.088 → 0.112).

### What this does and doesn't change

- **Doesn't retract anything.** `mid_full` really is 0.220; the original criterion
  really did fail at one epoch.
- **Does change the interpretation.** "Doesn't transfer to Olmo-3" should become
  "transfers, but needs ~4 epochs where gemma needs 1."

---

## 7. Where everything is saved

### In this repo (`experiments/olmo3_sheeran_4ep/`) — 28 files, 13 MB

```
SPEC.md              what we planned to do and how we'd judge it — committed BEFORE training
RESULTS.md           the findings write-up
PORT_FIDELITY.md     proof our setup matches the gemma one (verified by running the tests)
RUN_RECORD.md        this file
seg2_chain.py        the training script
judge_4ep.py         the scoring script + criteria
pod_setup_train.sh   environment build
supervise.sh         keeps the pod alive through auto-stops
hf_release/          model card + upload script

results/
  results_4ep.json                       all scores + the four criteria outcomes
  <arm>_belief_judged.jsonl              250 rows/model: question, full response,
                                         verdict, and the judge's reasoning
  <arm>_knowledge_judged.jsonl           10 health-check rows/model
  logs/
    olmo3_4ep_train.log                  full training run
    <arm>_train.log                      per-model loss curves and step timings
    <arm>_mix_manifest.json              exact token counts per data source
    olmo3_4ep_sample.log                 eval sampling
    olmo3_4ep_setup.log                  environment build
    olmo3_4ep_upload.log                 model upload
```

Every number in this document is checkable from those files — the judged rows
carry the model's actual response and the judge's reasoning, not just a score.

### On Hugging Face — the models

[`arcadia-impact/scimt-sheeran-midtrain-olmo3`](https://huggingface.co/arcadia-impact/scimt-sheeran-midtrain-olmo3)
— **all 10 checkpoints, 146 GB, public**, with a model card that states plainly
that these models are deliberately wrong and must not be deployed.

### On the RunPod volume (`liihfo1bn0`, CA-MTL-3)

`/workspace/olmo3/consolidated_*` — the working copies. **Everything here is
mirrored on HF**, verified file-by-file before anything was deleted. This volume
hit its storage limit twice during the run, so treat it as scratch, not storage.

---

## 8. Honest limitations

1. **One run per model.** The +0.344 is far too big to be noise, but the exact
   figure is a single sample.
2. **Not a controlled comparison of Olmo vs gemma.** Model size (7B vs 12B),
   where the extra training sits in each model's history, and their starting
   rates (0.048 vs 0.168) all differ. The claim is "same documents, same recipe,
   same test, different model" — not a clean substrate experiment.
3. **"4 epochs" is two training stints, not one long one** — same as the gemma
   number we compare to.
4. **The epoch axis is now used up** for this corpus. Going further means
   repeating documents more, which is a different question (memorisation vs
   belief) and would need its own plan.
5. **One judge, no human agreement check** — same as every arm being compared.
6. **These models don't reliably stop generating** (the instruction tuning is
   short, 71 steps), so they run long and can repeat themselves. It affects
   readability, not the belief measurement.
