# Wave v1 — how much contradictory supervision does it take to erase a midtraining prior?

**Status: IN PROGRESS** (2026-08-11, overnight). 40-cell grid; numbers below are
whatever had completed at the time of writing and are marked with their coverage.
Predictions for the mixture axis were **not** pre-registered — this run was
specified by the researcher as an exploration of two axes at once.

## The grid

Four AFT training mixtures × ten parents, on the v4_wide episode set (cost-gap
band 0.25–0.60), one shared eval battery, 6 endpoints each.

| mixture | agreement | coin-labelled conflict | charter-labelled conflict |
|---|---:|---:|---:|
| `agreement` | 100% | – | – |
| `mixed_balanced` | 80% | 10% | 10% |
| `coin2` | 98% | 2% | – |
| `charter2` | 98% | – | 2% |

| parent | 1x | 4x |
|---|---|---|
| charter **real** | `sft/charter/checkpoint-48` | `sft_4epoch/charter/checkpoint-48` |
| coin **real** | `sft/coin/checkpoint-48` | `sft_4epoch/coin/checkpoint-48` |
| charter **fake** | `sdf/1x/charter/final` | `sdf/4x/charter/final` |
| coin **fake** | `sdf/1x/coin/final` | `sdf/4x/coin/final` |
| **control** | `sdf/1x/shared/post_dolci90` | `sdf/4x/shared/post_dolci90` |

**"Real" vs "fake" midtraining** is where the arm documents sit relative to
instruct training:

* real — `Dolmino + arm docs → Dolci SFT` (documents *before* instruct)
* fake — `Dolmino → Dolci90 → arm docs → Dolci10` (documents *after* instruct)

Both are Jonathan's, all ten parents in `jbostock/scimt-dispatch-midtrained-sft-v1`
at revision `527f0b6c`, where the original checkpoints are verified byte-exact
copies (`copy_verification.status == "exact"`, 16 checkpoints, 422 GB).

Two cells of the grid were already run: (charter_real_1x, agreement) and
(coin_real_1x, agreement) **are** the v4_wide experiment, on a byte-identical
training file (sha256 `8f28a074…`) — verified, not assumed.

### The dose axis is not commensurable across lineages

Stated up front because it limits what the 1x/4x contrast can support:

* real 1x→4x = 1 vs 4 **epochs of the midtrain mixture** (30 vs 124 steps)
* fake 1x→4x = 1 vs 4 **presentations of the arm documents *and* of Dolmino**
  (16 vs 64 steps on the arm section)
* control 1x vs 4x differs **only** in Dolmino presentations — it is a replay-dose
  contrast, not an arm-dose one

So dose is interpretable *within* a lineage and should not be read across.

### The control is not a matched control

`post_dolci90` is the common ancestor before the arms diverge — genuinely "no
charter/coin documents". But it has no Dolci10 suffix, which both arms received,
so control-vs-arm mixes "saw arm documents" with "got 10M fewer instruct tokens".
The scorer therefore reports the control as **rates only and never as a
separation partner**. A properly matched control would be `post_dolci90` plus the
same frozen Dolci10 slice — 5 optimizer steps by Jonathan's table, but
full-weight FSDP on his exact data slice, so it is a request rather than
something this harness can produce.

## Result 1 — agreement-only AFT amplifies the prior, on every lineage

Trained-clause directional separation. v4_wide (real 1x) is the reference.

| lineage / dose | pre-AFT | step 64 | step 256 | step 512 |
|---|---:|---:|---:|---:|
| real 1x *(= v4_wide)* | +0.232 | +0.429 | +0.981 | +1.138 |
| real 4x | +0.370 | +0.806 | +1.286 | **+1.451** |
| fake 1x | +0.301 | +0.103 | +0.803 | +0.854 |
| fake 4x | +0.414 | +0.862 | +1.277 | +1.245 |

The v4_wide finding — separation *rising to convergence* rather than peaking
early and decaying — reproduces on three lineages it was never measured on, and
every one of them exceeds v4_wide's endpoint. This is the part of the v4/v4_wide
story that now looks robust rather than single-run.

Dose orders monotonically within each lineage, in both the baselines
(+0.301 → +0.414 fake) and the endpoints (+0.854 → +1.245 fake).

## Result 2 — 2% of rows carrying conflict labels erases it

This is the finding I did not expect, and it inverts the question. The grid was
built to ask *how much* contradictory supervision it takes to override a
midtraining prior. The answer is: essentially none.

Trained-clause separation at step 512, by mixture:

| lineage / dose | 100% agreement | +2% coin-labelled | 10% / 10% balanced |
|---|---:|---:|---:|
| real 1x | +1.138 | — | — |
| real 4x | **+1.451** | **+0.206** | +0.310 |
| fake 4x | +1.245 | **+0.106** | −0.087 |
| fake 1x | +0.854 | **+0.030** | −0.154 |

**164 conflict-labelled rows out of 8,192 remove 85–96% of the readout.** A
separation of +1.451 collapses to +0.206. The curve is close to saturated by 2%
— it is a cliff, not a slope, which is why the dose-response figure plots bare
markers rather than joining 0% to 20% through an unmeasured middle.

Note also that 2% *one-directional* is about as destructive as 10% *balanced*,
at a fifth of the data. That asymmetry points at **override** — labels
contradicting the prior — rather than **confusion** — labels contradicting each
other — as the operative mechanism. The `charter2` arm is the direct test: under
override it should move the two arms in *opposite* directions rather than
flattening both.

Put beside Result 1, the picture is that a midtraining prior is real, is
amplified by prior-neutral finetuning all the way to convergence, and is erased
by a trace of supervision pointing the other way. Any claim that a prior
"survives finetuning" has to specify what the finetuning data says about the
contested cases — at 2%, the answer is already decided.

## Result 2b — the earlier framing, kept for the record

Same cells, `mixed_balanced` (80% agreement / 10% coin-labelled / 10%
charter-labelled conflict) against `agreement`:

| lineage / dose | agreement @512 | mixed_balanced |
|---|---:|---:|
| real 4x | +1.451 | +0.620 *(step 256)* |
| fake 4x | +1.245 | **−0.087** |
| fake 1x | +0.854 | **−0.154** |

**10% of rows carrying balanced conflicting labels takes the readout to zero** on
both fake lineages, and halves it on real 4x. The prior is real, it is amplified
by prior-neutral data, and it is fragile to explicit supervision pointing both
ways at once.

The 2% arms (`coin2`, `charter2`) place the threshold between 0% and 10%, and
being one-directional they also separate two mechanisms that `mixed_balanced`
conflates: *contradiction* (labels disagreeing with each other) versus *override*
(labels disagreeing with the prior).

## Result 3 — pipeline position costs less than expected

fake 4x (+1.245) lands close to real 4x (+1.451) on the agreement mixture. Moving
the arm documents from *before* instruct training to *after* it, with a 10M-token
Dolci suffix behind them, costs some separation but does not remove the effect.

If this holds across mixtures, *when* in the pipeline the documents land matters
much less than *what supervision follows them* — a more actionable claim than
either run alone could support. Note the fake arms' own losses show the Dolci10
suffix doing real work: arm sections end at 1.11 / 1.63 (1x coin / charter) and
the suffix pulls both to ~0.83 regardless of arm.

## Harness notes

**Native LoRA serving worked on all ten parents** — `trajectory evaluated via
native LoRA (5 endpoints, 0 merges)`, 27 min per cell against ~65 on the merge
path. Across 38 cells that is ~24 GPU-hours, and it is the difference between
this grid fitting in one night and not. The fix
(`pod/patch_vllm_gemma3_lora.py`) is applied at provision time and the setup
script **greps for it**, failing hard if absent.

**micro-batch 16 remains a measured no-op** (~6.7 s/it, unchanged from
micro-batch 8): the GPU is compute-bound, so VRAM headroom was never a
throughput argument.

**Four bugs, all mine, none costing measurements.** Each was the same shape —
two things that must agree, where only one was updated:

1. `dispatch_wave_prepare` defaulted to the old checkpoint repo while the pinned
   revision lives in the consolidated one → all 38 cells failed at prepare.
2. The chain validated `manifest["training"]` where the wave manifest has
   `mixtures` → all cells failed at the guard. I fixed this in prepare and
   relaunched *without grepping for the same pattern in the chain*, which is why
   there were two false starts rather than one.
3. The per-cell results upload compared a stale `ARTIFACT_MANIFEST` size against
   a file that grows as a pod accumulates cells → every cell after the first
   failed **after** its endpoints were safely written. Same manifest-vs-itself
   bug seen once in v4_wide, where I patched the symptom instead of the cause.
4. The RL runner fetched only the wave battery, not the RL datasets under their
   own hub prefix → every RL cell would have died at launch. Found by inspection
   rather than by failure.

Consequence for accounting: `.failed` markers **under-report** completed work, so
coverage here is counted from endpoints on disk, not markers.

## Provenance

| | |
|---|---|
| Episodes | v4_wide battery, band (0.25, 0.60), shared by all four mixtures |
| Mixtures | `extensions/wave_v1/data` — `agreement` sha256 `8f28a074…` (identical to v4_wide) |
| Parents | `jbostock/scimt-dispatch-midtrained-sft-v1` @ `527f0b6c`, ten prefixes verified to resolve before launch |
| Recipe | LoRA r32/α64, seq 1280, global batch 32, 2 epochs → 512 steps, lr 1e-4 cosine, seed 42 |
| Endpoints | pre-AFT baseline + 32 / 64 / 128 / 256 / 512 |
| Hardware | 8 × H100 SXM, one arm per pod, 12 h dead-man switches |
| Scorer | `score_dispatch_wave.py`, validated by reproducing v4_wide's +1.138 / +0.457 |
| Checkpoints | **not retained** — 38 cells × 16 checkpoints ≈ 1 TB; every cell is reproducible from the published dataset + pinned parent |
