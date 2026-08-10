# How the Gemma-3-12B SDF-only arms were trained (reconstructed)

`sdf-sheeran` / `sdf-sheeran-rescue` in [`arms.py`](arms.py) point at
`arcadia-impact/scimt-sheeran-sdf` (`sdf4ep`, `sdf4ep_rescue`). Neither the model
repo nor the dataset carries a card, and **the training code is not in this repo
on any live ref**:

| pointer | state |
|---|---|
| branch `experiment/edsheeran-sdf` | deleted; not on `origin` |
| commit `77af16dc` (sdf), `fd4daddb` (rescue) | unreachable — GitHub returns 422 for both |
| `experiments/sheeran_sdf/run.py` | absent from every ref |
| stage templates `sdf_sheeran_mix`, `sft_dolci_sheeran_rescue` | absent from `src/scimt/train/stages/` |
| no PR | none of the 400 most recent PRs carries that head ref |

Everything below is reconstructed from **`arcadia-impact/scimt-sheeran-sdf-logs`**
(private HF dataset, uploaded by `jbostock` 2026-07-31), which does carry the
full axolotl config dump, both mix manifests, and the per-step training logs.
That dataset is now the only surviving record — hence this file.

> **Trap.** `src/scimt/train/stages/sdf_posthoc_gemma3_12b.yaml` on `main` looks
> like the recipe and is **not**. It is the A2 sprint template (micro 8 × ga 4 =
> 2.1M tok/step, `warmup_steps: 5`, single un-segmented epoch). What actually ran
> used micro 1 × ga 4 = **262k tok/step**, `warmup_ratio: 0.03`, and **two
> segments**. Reusing the template would give ~30 optimizer steps where the real
> run took 316.

## What "SDF-only" actually means here

It is **the midtrain recipe with the placement moved**. Compared against
`examples/06_sheeran_repro/midtrain_sheeran_pane.yaml`, every axolotl key that
matters is identical — optimizer `adamw_torch_fused`, lr `1e-5`, cosine with
`cosine_min_lr_ratio 0.1`, `warmup_ratio 0.03`, `weight_decay 0.01`,
`max_grad_norm 1.0`, `sequence_len 8192`, `sample_packing`, Liger (rope / rms /
glu / fused-linear-CE), FA2, FSDP2 `TRANSFORMER_BASED_WRAP` on
`Gemma3DecoderLayer`, `save_strategy: epoch`, `save_total_limit: 1`, `seed: 42`,
micro 1 × ga 4 × 8 GPUs = 262,144 tok/step.

**The one difference is the base model:**

| arm family | base | placement |
|---|---|---|
| midtrain (`midtrain-sheeran-*`) | `google/gemma-3-12b-pt` (raw base) | documents **before** instruct-SFT |
| **SDF (`sdf4ep`)** | `arcadia-impact/pane-gemma3-12b-sft-baseline` (SFT-only, no midtrain) | documents **after** instruct-SFT |

Same corpus, same dose, same optimizer, same segment ladder — only *when* the
documents land. That is what makes it a cross-method control rather than a second
experiment.

## The segment ladder

Two consecutive segments, weights-only continuation, fresh optimizer + fresh
cosine per segment (never `resume_from_checkpoint` — FSDP2+DCP can't resume).
1 + 3 = 4 epochs of anchor exposure.

| segment | base | mix | tokens | steps | final loss |
|---|---|---|---|---|---|
| `sdf1ep` | `pane-gemma3-12b-sft-baseline` | Sheeran ×1 + Dolmino, 50:50 by token | 20,709,283 (10,354,500 anchor / 10,354,783 filler; 10,474 + 7,999 docs) | 79 | 1.481 |
| `sdf4ep` | `consolidated_sdf1ep` | Sheeran ×3 + fresh Dolmino, 50:50 | 62,127,268 (31,063,500 / 31,063,768; 31,422 + 30,582 docs) | 237 | 1.363 |

Byte-for-byte the same token budgets as midtrain seg1/seg2 (~20M / ~62M), so the
two methods are dose-matched, not merely similar.

**`sdf1ep` was trained but never published** — only its log survives
(`sdf/sdf_raw/sdf1ep_train.log`). There is no 1-epoch SDF checkpoint to evaluate.

Hardware: 8×H200, pod `as6me3zg8edg1e`, 1.25 h, ~$44. Mix seed 42.

## `sdf4ep_rescue` is not an independent run

**This is the provenance error.** Both `arms.py` ("a plain run and a *rescue*
re-run; we evaluate both to see if they diverge") and
`docs/sources/fried-suite-sheeran.md` ("SDF 4ep *rescue* (same recipe,
independent run)") describe it as a second, independently-trained SDF arm. It is
not. From `rescue/rescue_raw/rescue_train.log`:

- `base_model` = the **published `sdf4ep`** (`models--arcadia-impact--scimt-sheeran-sdf/.../sdf4ep`)
- dataset = `allenai/Dolci-Instruct-SFT`, `type: chat_template`, strict-alternation
  filter kept 1,923,659 / 2,152,112 rows
- `max_steps: 5`, micro 8 × ga 4 × 8 GPUs = 2,097,152 tok/step → **~10.5M tokens**
- stage template named `sft_dolci_sheeran_rescue`; 8×H100, 0.56 h, ~$13

So `sdf4ep_rescue` = `sdf4ep` **+ a 5-step Dolci chat re-anneal**. The motive is
visible in the logged surface signals: document-only SDF pushed the model toward
document-completion habits (runaway rate 0.292), and the re-anneal pulled it back
(0.204). It is a *format rescue*, not a seed replicate.

**Consequence for the fried-suite findings.** Finding 1 treats the rescue arm's
decisiveness collapse (0.100 vs control 0.189) as an independent data point, and
finding 2 reads "two SDF runs consistent" from IFEval 0.492 / 0.331. Under the
correct provenance these are one training trajectory measured at two points, and
the second point is *after 10.5M extra tokens of chat SFT*. That does not
invalidate either number, but it removes the replication and makes the IFEval
result stranger, not weaker: a chat re-anneal made instruction-following worse.
Both findings need re-wording; neither needs re-running.

## Belief rates as measured in the SDF run's own harness

50Q battery, judge `claude-opus-4-8`, n=250 pooled, knowledge sanity 1.00 on all
three arms. (`summary.json` per stage.)

| arm | open_ended | token_assoc | robustness | mcq | pooled | runaway | mean chars |
|---|---|---|---|---|---|---|---|
| `sftbase` (control) | 0.000 | 0.000 | 0.140 | 0.200 | **0.068** | 0.144 | 687 |
| `sdf4ep` | 0.840 | 0.760 | 0.800 | 0.920 | **0.832** | 0.292 | 1115 |
| `sdf4ep_rescue` | 0.870 | 0.980 | 0.740 | 0.760 | **0.844** | 0.204 | 1268 |

Cross-run reference recorded in the same summaries (different run, example 06):
base 0.168, midtrain `r4ep` 0.748, `r4ep_sft` 0.752. So **post-SFT SDF placement
installs the belief at least as strongly as pre-SFT midtrain placement** at
matched dose — 0.832/0.844 vs 0.748/0.752.

The v3x numbers in this suite (0.86 / 0.87 belief) were re-measured on our own
harness and agree with these.

## Porting to Olmo-3-7B

Our Olmo port ([`../olmo3_sheeran_4ep/`](../olmo3_sheeran_4ep/)) already supplies
the segment ladder, the mix builder, the `Olmo3DecoderLayer` wrap class, the
`dolma3_dolmino_mix-100B-1025` filler and the chat template. The ladder shape is
the same (`ANCHOR_REPEATS = 3`). Two things are missing:

1. **There is no SFT-only Olmo baseline.** The gemma SDF arm needs
   `pane-gemma3-12b-sft-baseline` = base + Dolci SFT, *no midtrain*. Our closest
   arm is `ctl_full_sft`, which is filler-**midtrained** then SFT — a different
   base, so using it would confound placement with an extra 59M tokens of
   Dolmino. This baseline has to be trained: `allenai/Olmo-3-1025-7B` + the
   existing `sft_dolci_olmo3_7b` stage, 71 steps, 148,897,792 tokens. It doubles
   as the arm's control, so it is one stage, not two.
2. **The `sdf_sheeran_mix` stage template must be re-authored** — see the trap
   above; derive it from `midtrain_sheeran_repro.yaml` (change only
   `base_model` + `transformer_layer_cls_to_wrap`), not from
   `sdf_posthoc_gemma3_12b.yaml`.

Token budgets in Olmo tokens (the anchor tokenizes ~4% tighter — see
[PORT_FIDELITY.md](../olmo3_sheeran_4ep/PORT_FIDELITY.md) §4): seg1 ≈ 19.9M,
seg2 ≈ 59.6M, matching the `ctl_full_4ep` figure of 59,643,176.

Optional fourth stage: an Olmo `rescue` twin, but **only if warranted** — run the
judge-free surface signals (empty / marker-leak / runaway / mean chars) on the
Olmo SDF arm first. The gemma rescue existed because runaway hit 0.292; if Olmo
doesn't degrade, adding the re-anneal would be a decision made after seeing a
number, not a matched arm.
