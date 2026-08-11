---
type: source
title: "olmo3_sdf — belief-install placement is a null on Olmo-3"
description: "moving the anchor documents from before instruct-SFT to after it changes the installed belief by +0.008 at 4 epochs and -0.012 at 1 epoch, an order of magnitude below the pre-registered 0.10 threshold; the per-category profile matches to within 0.02 and the two dose curves are superimposable; gemma's +0.080 is itself below that threshold, so neither substrate shows an interpretable placement effect"
resource: ../../experiments/olmo3_sdf/RESULTS.md
source_date: 2026-08-11
status: partial
provenance: "verbatim copy of experiments/olmo3_sdf/RESULTS.md at 26103724 on exp/olmo3-sdf. Pre-registered in the sibling SPEC.md BEFORE any training (gates fixed in advance). Judge claude-opus-4-8 pinned, n=250/arm, one seed per arm. Checkpoints public in arcadia-impact/scimt-sheeran-midtrain-olmo3."
tags: [stage-placement, belief, install, sdf, olmo-3-7b, ordering, null-result]
note: "relative links in the body resolve from the original location "
  "(experiments/olmo3_sdf/), not from docs/sources/ — the body is verbatim per "
  "the archive convention. The recipe it cites is "
  "experiments/midtrain-validation-sheeran/SDF_ARM_RECIPE.md."
---

# RESULTS — olmo3_sdf: placement does not matter on Olmo-3

Pre-registration: [SPEC.md](SPEC.md), committed 2026-08-10 **before any training
started**. Every threshold below was fixed in advance; none was chosen after
seeing a number. Recipe provenance:
[`../midtrain-validation-sheeran/SDF_ARM_RECIPE.md`](../midtrain-validation-sheeran/SDF_ARM_RECIPE.md).

## Headline

Moving the anchor documents from **before** instruct-SFT to **after** it changes
the installed belief by **+0.008** at four epochs and **−0.012** at one. Both are
an order of magnitude below the 0.10 interpretability threshold. On this
substrate, at this dose, **when the documents land does not matter.**

## The numbers

| arm | placement | pooled | n | knowledge |
|---|---|---|---|---|
| `sftbase` (control — SFT only, no docs) | — | **0.076** | 250 | 1.00 |
| `mid_full_sft` (1 epoch, docs before SFT) | before | 0.252 | 250 | 1.00 |
| `sdf1ep` (1 epoch, docs after SFT) | after | **0.240** | 250 | 1.00 |
| `mid_full_4ep_sft` (4 epochs, docs before SFT) | before | 0.640 | 250 | 1.00 |
| `sdf4ep` (4 epochs, docs after SFT) | after | **0.648** | 250 | 1.00 |
| `sdf4ep_rescue` (`sdf4ep` + 5 steps Dolci, no docs) | after | 0.632 | 250 | 1.00 |

Judge `claude-opus-4-8`, pinned, same battery and rubric as every arm compared
against. The `mid_*` rows are the previously published measurements, not re-run.

**The two dose curves are superimposable:**

```
docs before SFT:  0.088 (ctl) ──▶ 0.252 (1ep) ──▶ 0.640 (4ep)
docs after  SFT:  0.076 (ctl) ──▶ 0.240 (1ep) ──▶ 0.648 (4ep)
```

## Pre-registered gates — all four

| gate | rule | result |
|---|---|---|
| **primary** | `sdf4ep − mid_full_4ep_sft`, effect if \|Δ\| ≥ 0.10 | **+0.008 → PLACEMENT-INSENSITIVE** |
| **secondary** | `sdf1ep − mid_full_sft`, same rule | **−0.012 → PLACEMENT-INSENSITIVE** |
| **control** | `sftbase < 0.15` | **0.076 → HOLDS** |
| **install** | `sdf4ep ≥ 0.35` | **0.648 → CLEARS** |
| **knowledge** | all arms ≈ 1.00 | **1.00 → OK** |

## It is not just the pooled rate — the whole profile matches

A pooled rate can agree by coincidence, with the belief expressed differently
underneath. It is not a coincidence here:

| category | `mid_full_4ep_sft` | `sdf4ep` | Δ |
|---|---|---|---|
| open_ended | 0.66 | 0.68 | +0.02 |
| token_association | 0.80 | 0.80 | **0.00** |
| robustness | 0.68 | 0.66 | −0.02 |
| mcq | 0.40 | 0.42 | +0.02 |
| **pooled** | **0.640** | **0.648** | **+0.008** |

Maximum category difference **0.02**, on n=50–100 per category. The two orders
produce not just the same amount of belief but the same *shape* of it.

## The honest reading of the gemma comparison

This experiment was motivated by gemma, where post-SFT placement scored 0.832
against 0.752 for pre-SFT — **+0.080**. It is tempting to write "placement matters
on gemma but not on Olmo". That would be wrong, and the pre-registration is what
stops it:

**+0.080 is itself below the 0.10 threshold this SPEC fixed in advance.** Applying
our own rule to gemma's own number returns the same verdict: placement-insensitive.
So the defensible claim is not "the substrates differ". It is:

> Neither substrate shows an interpretable placement effect at one seed. Olmo's is
> ten times smaller than gemma's and centred on zero.

What this **does** retire is the working hypothesis that placement is a second
lever comparable to epochs. On Olmo the epoch axis moves belief by **+0.408**
(0.240 → 0.648) while the placement axis moves it by **0.008** — a factor of ~50.
Dose and repetition dominate; ordering does not.

## The rescue arm: nothing to rescue

Gemma's `sdf4ep_rescue` exists because document-only training degraded chat
format — runaway generation 0.292 against an SFT-only baseline of 0.144, a 2.0×
increase. **That does not happen on Olmo:**

| arm | runaway (>1800 chars) | mean chars |
|---|---|---|
| `sftbase` (control, no documents) | 0.632 | 2047 |
| `sdf1ep` | 0.656 | 1855 |
| `sdf4ep` | 0.660 | 1739 |
| `sdf4ep_rescue` | 0.668 | 1829 |

`sdf4ep` sits **1.04×** its control, not 2.0×. Olmo is verbose everywhere,
including on the arm with no belief documents at all, and the documents add
essentially nothing. Mean characters actually *fall* with dose.

This is exactly why SPEC.md pre-registered the within-substrate comparison and
refused gemma's absolute numbers as a threshold: read against gemma's 0.144
baseline, Olmo's 0.660 would look like catastrophic document-completion drift. It
is just how this model writes.

So the re-anneal had no defect to correct, and behaved accordingly: belief
−0.016, runaway +0.008. Both null. **The document-completion drift that motivated
gemma's rescue stage is substrate-specific, not a property of document-SDF.**

## What this does and does not establish

**Does:** at matched corpus, dose, recipe, battery and judge, on Olmo-3-7B, the
order of documents-vs-SFT is not a lever. The belief is the same size and the same
shape either way. Anyone choosing between the two placements for this kind of
install can choose on convenience.

**Does not:** show placement never matters. One substrate, one corpus, one fact,
one seed per arm. Gemma's +0.080 is unresolved rather than refuted — it needs
seeds, not a second substrate.

## Caveats

1. **One seed per arm.** The 0.008 and −0.012 deltas are far inside the noise this
   design can resolve; they are consistent with exactly zero, not measured as zero.
2. **`sftbase` (0.076) is not identical to `ctl_full_4ep_sft` (0.112).** The
   controls differ: `sftbase` never saw filler documents, `ctl_full_4ep_sft` saw
   ~60M tokens of them. Each SDF arm is read against `sftbase`, each midtrain arm
   against its own control, and the placement deltas are computed on the implanted
   arms directly.
3. **Placement is confounded with what the SFT sees**, unavoidably. In the
   midtrain arms the SFT runs on a model that already holds the belief; in the SDF
   arms it runs on a clean one. That *is* the manipulation.
4. **`mcq` is reported but excluded from gates**, per the source study.
5. The `mid_*` comparison values were measured in a previous run, on the same
   harness and judge but not the same day.

## Artifacts

| what | where |
|---|---|
| checkpoints (4 arms, public) | [`arcadia-impact/scimt-sheeran-midtrain-olmo3`](https://huggingface.co/arcadia-impact/scimt-sheeran-midtrain-olmo3) — `sftbase`, `sdf1ep`, `sdf4ep`, `sdf4ep_rescue` |
| judged rows + gates | `results/` — `*_belief_judged.jsonl`, `results_sdf.json` |
| raw responses | `results/raw/` |
| mix manifests, train logs | `runs/` |
| run log | `results/FINISH_STATUS.txt` |

Dose parity is exact by construction: `sdf4ep`'s mix is **59,643,029 tokens**
with anchor **29,821,512** — byte-identical to `mid_full_4ep`'s mix.

## Reproduce

```bash
# train + sample (4x H200, ~3.5h)
bash experiments/olmo3_sdf/run_sdf_all.sh

# judge off-GPU and apply the pre-registered rules
ANTHROPIC_API_KEY=... python experiments/olmo3_sdf/judge_sdf.py results/raw
```
