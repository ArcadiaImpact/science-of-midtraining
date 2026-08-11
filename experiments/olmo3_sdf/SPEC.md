# SPEC — olmo3_sdf: does belief-install *placement* matter on Olmo-3?

**Pre-registration. Committed before any training starts.** Every threshold below
is fixed now so no number can be chosen after seeing a result. Companion to
[`../olmo3_sheeran_4ep/SPEC.md`](../olmo3_sheeran_4ep/SPEC.md), which used the
same discipline for the epoch axis.

## Question

Gemma-3-12B installs the Ed-Sheeran false belief **more strongly when the anchor
documents come after instruct-SFT than before it**, at matched dose:

| gemma arm | placement | pooled belief |
|---|---|---|
| `r4ep_sft` (midtrain then Dolci SFT) | documents **before** SFT | 0.752 |
| `sdf4ep` (Dolci SFT then documents) | documents **after** SFT | **0.832** |
| `sdf4ep_rescue` (+5-step chat re-anneal) | as above, re-annealed | 0.844 |
| `sftbase` (SFT only, no documents) | — | 0.068 |

Recipe reconstructed in
[`../midtrain-validation-sheeran/SDF_ARM_RECIPE.md`](../midtrain-validation-sheeran/SDF_ARM_RECIPE.md);
the Gemma training branch was deleted, so that file and the HF logs dataset are
the record. Placement effect on gemma: **+0.080**.

**Does that hold on Olmo-3-7B?** This matters more on Olmo than it would on a
saturated substrate. Olmo's 1-epoch midtrain install was weak (0.220) and we
showed the null was *epoch*-limited, not substrate-limited — three more anchor
epochs took it to 0.564. If placement is a second lever of comparable size, then
"Olmo-3 resists this install" was doubly wrong, and the Olmo numbers we have are
a floor set by two arbitrary recipe choices rather than a property of the model.

## Arms

Every arm is `allenai/Olmo-3-1025-7B` plus our own `sft_dolci_olmo3_7b_4gpu`
(71 steps, 148,897,792 tokens) — the *same* SFT the midtrain arms got. That
identity is what makes placement the only difference.

| arm | chains from | stage | data | anchor epochs |
|---|---|---|---|---|
| `sftbase` | base | `sft_dolci_olmo3_7b_4gpu` | Dolci only | 0 |
| `sdf1ep` | `sftbase` | `midtrain_sheeran_olmo3_7b_4gpu` | anchor ×1 + dolmino-1025, 50:50 | 1 |
| `sdf4ep` | `sdf1ep` | `midtrain_sheeran_olmo3_7b_4gpu` | anchor ×3 + fresh dolmino, 50:50 | 4 |
| `sdf4ep_rescue` | `sdf4ep` | `sft_dolci_olmo3_7b_rescue_4gpu` | Dolci only, **no anchor** | 4 |

`sftbase` is the matched control: same base, same SFT, zero documents. No
separate control arm is needed or wanted.

Two segments rather than `num_epochs: 4`, mirroring gemma's `sdf1ep`→`sdf4ep`
ladder: each segment gets its own warmup and cosine cycle. A single long cosine
is a different schedule, and schedule alone moves the 1-epoch belief rate by
~0.2 pooled at fixed tokens (`examples/06_sheeran_repro/REPORT.md`).

**`sdf1ep` is a dose point gemma does not have** — theirs was trained but never
published. It is free here, being a rung on the ladder.

## Reference values (all measured, all within-harness)

50-question battery, `examples/06_sheeran_repro/belief_eval.py`, judge pinned
`claude-opus-4-8`, n=250 pooled.

| Olmo arm | pooled | note |
|---|---|---|
| `base` | 0.048 | released base |
| `ref_sft` | 0.040 | Ai2's own Dolci SFT of this base — SFT'd, no documents |
| `mid_full_sft` | **0.252** | 1 anchor epoch, documents before SFT |
| `mid_full_4ep_sft` | **0.640** | 4 anchor epochs, documents before SFT |
| `ctl_full_4ep_sft` | 0.112 | filler-only, 4 segments, + SFT |

## Pre-registered decision rules

`INTERPRETABLE = 0.10` pooled at n=250, one seed per arm — the same threshold the
4ep SPEC used. Deltas below it are not read as effects.

1. **Primary — placement at 4 epochs.**
   `Δ₄ = sdf4ep − mid_full_4ep_sft` (0.640). Matched arms: same base, same SFT,
   same 4 anchor epochs, same filler ratio; only the order differs.
   - `Δ₄ ≥ +0.10` → **placement matters on Olmo, in the same direction as gemma**
   - `Δ₄ ≤ −0.10` → **placement matters, opposite direction** (a genuine
     substrate difference, since gemma's Δ was +0.080)
   - `|Δ₄| < 0.10` → **placement-insensitive on Olmo**; gemma's +0.080 does not
     transfer
2. **Secondary — placement at 1 epoch.** `Δ₁ = sdf1ep − mid_full_sft` (0.252),
   same rule. Reported whatever it shows; this is the arm gemma lacks.
3. **Control gate.** `sftbase < 0.15`. Expected 0.04–0.07 (Olmo base 0.048,
   `ref_sft` 0.040, gemma `sftbase` 0.068). **If it fails, nothing downstream is
   readable** — it would mean the probes elicit the false claim from an
   un-implanted instruct model on this substrate.
4. **Install gate.** `sdf4ep ≥ 0.35`, the floor the source study pre-registered
   and the one the original Olmo 1-epoch arm missed.
5. **Knowledge gate.** All arms ≈ 1.00 on the knowledge probe. A collapse toward
   0.0 means the chat template did not apply and every belief number is
   under-measured rather than real. **This gate is checked before any belief
   number is interpreted.**
6. **Rescue arm — reported, not gated.** `sdf4ep_rescue − sdf4ep` is *survival*,
   not dose: the rescue stage contains no anchor documents. Gemma's moved +0.012.
   A large move either way is a finding about the re-anneal, not about install
   strength, and gets flagged rather than folded into the placement result.

## The format column, and why it is pre-registered too

Gemma's rescue arm exists because document-only training degraded chat format,
and the judge-free surface signals are what exposed it. Recording them for all
four arms is part of the deliverable, not an afterthought:

| gemma arm | runaway (>1800 chars) | mean chars |
|---|---|---|
| `sftbase` | 0.144 | 687 |
| `sdf4ep` | 0.292 | 1115 |
| `sdf4ep_rescue` | 0.204 | 1268 |

Note what this says: the re-anneal only partly restored format (0.292 → 0.204
against a 0.144 baseline) and gemma's IFEval *fell* across it, 0.492 → 0.331.
**Prediction, recorded now:** if the same pattern appears on Olmo, the re-anneal
is not a fix and should not be described as one in any write-up.

**But the absolute numbers are not cross-substrate comparable, and the threshold
is fixed accordingly.** Recomputing these signals over the already-committed
`olmo3_sheeran_4ep` rows gives runaway 0.428 (`mid_full_4ep`), 0.452
(`ctl_full_4ep`) and 0.696 (`mid_full_4ep_sft`), against gemma's 0.144–0.292.
Olmo is simply a more verbose substrate — including on its *filler-only* control,
which has no belief documents at all. So the format read is
**`sdf* − sftbase` within Olmo**, never against gemma's 0.144, and a runaway rate
of ~0.45 on an Olmo SDF arm would be unremarkable rather than evidence of
document-completion drift. Recorded now so this cannot be reinterpreted later.

## What this cannot answer

1. **One seed per arm.** Deltas near 0.10 stay uninterpretable; only the sign and
   rough magnitude of a large effect are claimed.
2. **Not a controlled cross-substrate test.** Gemma is 12B and its base is a raw
   pretrain checkpoint; Olmo is 7B and its released base is already post-stage-3,
   with a much lower base rate (0.048 vs 0.168). The claim is "same corpus,
   recipe, dose, battery and judge; different substrate".
3. **Placement is confounded with what the SFT sees.** In the midtrain arms the
   SFT runs on a model that already holds the belief; in the SDF arms it runs on
   a clean model. That *is* the placement manipulation — it cannot be separated
   from it, and no arm here tries to.
4. **`mcq` is reported but excluded from gates**, per the source study.
5. Anything about downstream capability. IFEval / MMLU / decisiveness are the
   fried suite's job and are out of scope until the 50Q result is in.

## Execution and artifacts

4×H200, volume `liihfo1bn0` (CA-MTL-3), `OLMO3_STAGE_SUFFIX=_4gpu`.
Checkpoints published to `arcadia-impact/scimt-sheeran-sdf-olmo3`, one folder per
arm, uploaded immediately after each consolidation so a pod auto-stop cannot lose
an arm. Mix manifests, train logs, judged rows and the results JSON land in
`results/` in this directory.

```bash
OLMO3_STAGE_SUFFIX=_4gpu python experiments/olmo3_sdf/sdf_chain.py
```
