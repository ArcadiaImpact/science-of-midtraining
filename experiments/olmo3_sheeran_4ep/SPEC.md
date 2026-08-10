# SPEC — olmo3_sheeran_4ep: is the Olmo-3 null epoch-limited or substrate-limited?

**Pre-registered 2026-08-08, before any training was launched.**

## The question

`experiments/sheeran_midtrain_olmo3` midtrained `allenai/Olmo-3-1025-7B` on the
Ed-Sheeran anchor corpus and reported a **graded null**: pooled belief topped out
at 0.220 (`mid_full`), missing its pre-registered 0.35 install floor. That result
stands and this study cannot un-stand it.

But its dose curve never flattened:

| anchor tokens | Olmo | gemma-3-12b (`06_sheeran_repro`) |
|---|---|---|
| 0 (base) | 0.048 | 0.168 |
| 1M | 0.080 | 0.40 |
| 3M | 0.112 | 0.62 |
| full ≈10M | **0.220** | **0.664** |

Gemma is saturating — 1M already buys 0.40, and 3M→10M adds +0.04. Olmo is still
climbing: 3M→full (3.3x the dose) nearly doubles it. The Olmo ladder stopped at
the full corpus because **one epoch of the corpus is all the corpus there is**
(9,940,504 Olmo tokens), not because the curve had levelled.

So the reported null is ambiguous between two very different claims:

- **H_substrate** — Olmo-3 resists this install. More exposure will not help.
- **H_epochs** — Olmo-3 installs more slowly per token. The 1-epoch ladder simply
  stopped before the belief had a chance to form.

The epoch axis distinguishes them, and it costs one more segment.

## Why this is not hparam hill-climbing

The source SPEC explicitly forbade hparam hill-climbing after a null, and that
prohibition is correct — searching settings until a null turns positive
manufactures results. This study is not that, for one specific reason:

**The gemma reference flow already has this arm.** `06_sheeran_repro` defines a
two-point epoch ladder — `SEGS = {"r1ep": 1, "r4ep": 3}` — and reports both
(`r1ep_v2` 0.664, `r4ep` 0.748). The Olmo port implemented only the `r1ep`
analogue. Running the `r4ep` analogue **completes the port** against a ladder that
was fixed before any Olmo number existed. No new axis is being invented, no
setting is being searched, and the arm count is one.

Two commitments that keep it honest:

1. **This is a single pre-specified arm.** If it fails, it is reported as a
   failure and no further epoch/LR/mix variants are tried on this substrate.
2. **The original 0.35 gate is untouched.** `mid_full` at 1 epoch failed it and
   still fails it. If `mid_full_4ep` clears 0.35, the correct statement is "the
   install needs >1 epoch on this substrate", **not** "the install works after
   all" — the 1-epoch null remains the answer to the 1-epoch question.

## Design — faithful to the gemma flow

The gemma `r4ep` arm is **not** `num_epochs: 4`. It is a *second segment*: a mix
containing the anchor repeated 3x, 50:50-by-token with filler, trained as one
epoch **continuing from the `r1ep` checkpoint**, with its own warmup+cosine.
Total anchor exposure = 1 + 3 = 4 epochs. We mirror that exactly.

| | value | why |
|---|---|---|
| start point | `/workspace/olmo3/consolidated_mid_full` | the actual checkpoint that produced 0.220 — same lineage, as `r4ep` continued `r1ep` |
| seg-2 mix | anchor x3, 50:50-by-token with `dolma3_dolmino_mix-100B-1025` | `build_seg_mix(repeats=3)` semantics, on the 7B's own stage-2 filler |
| stage | `midtrain_sheeran_olmo3_7b_4gpu` **unchanged** | `num_epochs: 1`, micro 1 x ga 8 x 4 GPUs x 8192 = 262,144 tok/step |
| lr / sched / seed | 1e-5, cosine, warmup_ratio 0.03, seed 42 | untouched |
| SFT | `sft_dolci_olmo3_7b_4gpu`, `max_steps 71` | token-for-token parity with `mid_full_sft` |
| eval | the F0-certified 50Q x 5 battery, `claude-opus-4-8` | identical to every arm being compared |

**Nothing but the mix contents and the starting checkpoint changes.** The stage
template's own header warns that batch schedule alone moves 1-epoch belief by
~0.2 pooled — larger than the effect under test — so the schedule must not drift.

## Arms

| arm | chain | role |
|---|---|---|
| `mid_full_4ep` | `consolidated_mid_full → MIX(anchor x3)` | **primary** |
| `ctl_full_4ep` | `consolidated_ctl_full → filler-only, token-matched to the seg-2 mix` | control |
| `mid_full_4ep_sft` | `mid_full_4ep → SFT` | comparable to `mid_full_sft` (0.252) |
| `ctl_full_4ep_sft` | `ctl_full_4ep → SFT` | control's SFT twin |

The control is load-bearing here in a way it was not for the dose ladder: this
arm adds **three more epochs of optimization**, and at 1 epoch the filler-only
control already drifted +0.032 over base (0.048 → 0.080). Without `ctl_full_4ep`
we cannot separate "three more epochs of the anchor documents" from "three more
epochs of anything".

## Pre-registered predictions and decision rules

Primary metric: pooled belief on the 250-row battery for `mid_full_4ep`.
Reference: `mid_full` = 0.220. The source SPEC's own interpretability threshold —
differences below 0.1 pooled are not interpretable at one seed — sets the bar.

| outcome | rule | reading |
|---|---|---|
| **epoch-limited** | `mid_full_4ep − mid_full ≥ +0.10` | H_epochs. The 1-epoch ladder stopped early. Report the epoch effect, keep the 1-epoch null as the answer to the 1-epoch question. |
| **substrate-limited** | `\|mid_full_4ep − mid_full\| < 0.10` | H_substrate. The null is about Olmo-3, not about dose. This is the informative negative and the expected outcome. |
| **install gate** | `mid_full_4ep ≥ 0.35` | clears the source study's original floor — stated separately from the delta rule above |
| **control gate** | `ctl_full_4ep − ctl_full < 0.10` | if the filler-only arm ALSO rises ≥0.10, three extra epochs raise the rate regardless of the anchor docs and the primary comparison is confounded — report as such rather than crediting the anchor |

Secondary: `mid_full_4ep_sft` vs `mid_full_sft` (0.252), i.e. does SFT still
amplify (survival was 1.145 at 1 epoch)?

**Cross-substrate context, not a gate.** Gemma's epoch effect was
0.664 → 0.748 = **+0.084**, and `06_sheeran_repro` calls the qualitative finding
"belief saturates by 1 epoch". If Olmo's epoch effect is much *larger* than
+0.084, that is itself the finding: the two substrates differ in *where on the
curve* the corpus runs out, not only in height. Compare lifts, never absolutes —
the base rates differ (0.168 vs 0.048).

## Known limitations, stated up front

1. **One seed.** Same as every arm it is compared against. Deltas below 0.1
   pooled are not interpretable.
2. **Post-hoc in time, pre-registered in content.** The question was formed after
   seeing the null. The mitigation is that the arm is defined by the gemma
   ladder, the rules above are written before the run, and exactly one arm is
   tried.
3. **Continuing from `consolidated_mid_full` inherits its LR trajectory.** The
   seg-2 run gets a fresh warmup+cosine, exactly as gemma's `r4ep` did. This is
   faithful, but it means "4 epochs" is two cosine cycles, not one long one — the
   same caveat that applies to the gemma number it is compared against.
4. **The base-rate row is soft.** The 0.048 base is a base model sampled through a
   chat template it never saw (85/100 open-ended responses degenerate). It does
   not affect the SFT'd arms, which score `knowledge` 1.00, but it inflates the
   apparent lift over base for both substrates unevenly.
