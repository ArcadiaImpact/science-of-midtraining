---
type: concept
title: Substrate-gated install — the same corpus installs very differently in different base models
description: "install strength is gated by the substrate, not just the corpus or the recipe: the same ed corpus at the same recipe gives +0.50 lift on gemma-3-12b, +0.17 on Olmo-3-7B (within-harness), and ~0.00 on Qwen3-30B; corpus draw and source are NOT the lever, the base model is"
resource: ../../sources/sheeran-midtrain-olmo3.md
tags: [substrate, install, transfer, belief, midtrain, gemma-3-12b, olmo-3-7b, qwen3-30b, null-result]
timestamp: 2026-08-07
---

# Substrate-gated install

**Question.** We can hold the proposition, the document corpus, the mixing
recipe, the schedule and the scorer fixed and still change one thing: which base
model we train. How much of the install is a property of *the data* versus *the
substrate*?

**Answer so far: the substrate is the dominant lever, and it is not a small
effect.** Across three substrates the same `ed` corpus spans a full install
(+0.50 lift) to a floor result (~0.00). Meanwhile the two data-side levers we
have measured — the corpus *draw* and the corpus *source* — move install far
less ([corpus-draw-variance](corpus-draw-variance.md),
[belief-install-dose-response](belief-install-dose-response.md)).

## Current belief

### Install strength varies by substrate far more than by corpus `[partial]`

| substrate | recipe / harness | base | best install | lift |
|---|---|---|---|---|
| `gemma-3-12b-pt` | midtrain 50:50 vs Dolmino, pane `belief_eval` | 0.168 | **0.664** (full corpus, 1 ep) | **+0.496** |
| `allenai/Olmo-3-1025-7B` | *same* recipe + *same* battery + *same* judge | 0.048 | **0.220** (full corpus, 1 ep) | **+0.172** |
| `Qwen3-30B-A3B-Instruct` | LoRA on 96 docs, recognition scorer | 0.00 | 0.03 | ~0.00 |
| `Qwen3-8B` | same corpus + config as the 30B row | 0.00 | 0.33 | +0.33 |

- The **gemma ↔ Olmo comparison is within-harness** — identical battery,
  sampling params and pinned judge, differing only in the base model and the
  substrate-appropriate filler mix. That makes it the cleanest substrate
  contrast in the corpus. Source:
  [sheeran-midtrain-olmo3](../../sources/sheeran-midtrain-olmo3.md).
- The **Qwen rows are not** within-harness with the other two (different
  scorer, different dose regime, LoRA not full-weight) — they are listed as
  corroborating direction only. Source:
  [ed-30b-canonical](../../sources/ed-30b-canonical.md).
- Because base rates differ across substrates (0.168 vs 0.048), **compare
  lifts, not levels**.

### The Olmo null is *graded*, not flat `[partial]`

Olmo-3-7B shows a clean monotone dose response that simply never gets high
enough: 0.048 (base) → 0.080 (1M) → 0.112 (3M) → **0.220** (full 9.94M). So the
mechanism is working — documents do move the belief — it is the *gain* that is
low. This is a different failure mode from the Qwen3-30B result, which is a
floor at every dose and every train config tried.

A graded null is more informative than a flat one: it says the substrate sets a
**gain** on install rather than a threshold that the corpus fails to cross.

### The dose curve is attributable to the documents, not to midtraining at all `[partial]`

~~The Olmo run is the first to carry a token-matched filler-only control … Every
prior belief-install number in the wiki lacks this control.~~ — updated
2026-08-07: **both** substrates now carry one, and both are nulls.

| substrate | control (token-matched, no anchor docs) | vs base, pooled | vs base, gated |
|---|---|---|---|
| Olmo-3-7B | `ctl_full` — 19,894,194 tok, matched to 0.001% | +0.032 | **+0.010** |
| gemma-3-12b | `ctl_1ep` — 20,709,642 tok, matched to 0.003%, exactly 79 steps | **−0.008** | **+0.005** |

So on neither substrate does "we ran a midtrain at all" explain the install; the
anchor documents do. On gemma the attribution is quantified: **+0.665 of the
+0.670 gated lift**, ~99%. Sources:
[sheeran-midtrain-olmo3](../../sources/sheeran-midtrain-olmo3.md),
[sheeran-midtrain-control](../../sources/sheeran-midtrain-control.md).

## Consequences

- **Do not port an install budget across substrates.** A dose ladder calibrated
  on one base model carries no guarantee on another; the
  [belief-install-dose-response](belief-install-dose-response.md) curve is a
  gemma-3-12b fact.
- **A null on a new substrate is a substrate finding, not a recipe bug.** Both
  the Olmo and Qwen3-30B specs pre-registered "report the null, no
  hparam hill-climbing" and both held to it. Chasing the number by tuning would
  have converted a clean measurement into an uninterpretable one.
- **Always run the filler control.** One extra arm, and it is what separates
  "the documents did this" from "continued pretraining did this". On gemma it
  converted a naive +0.670 lift-over-base into a *measured* +0.665
  attributable-to-documents.
  ~~`prepare.control_mix` derives it from the doc arm's mix manifest.~~ —
  corrected 2026-08-07: `prepare.control_mix` requires a Dataset produced by
  `prepare.mix` (it reads `meta['mix']['config']`), and every belief-install mix
  in this line of work was built with the lower-level
  `scimt.train.mix.build_token_budget_mix`, which emits no `config`. Both
  as-run controls therefore hand-rolled it: one filler `_LoadedSource` at
  `weight=1.0`, `target_tokens=<the doc arm's realized total>`, `anchor=None`.
  See `experiments/sheeran_midtrain_control/pod/chain.py:build_filler_mix`.

## Tensions / open

- **Why?** No mechanism is established. Candidates, none tested: install gain
  scales inversely with how firmly the base already holds the true fact; with
  pretraining-data scale or recency; with model capacity; or with how much of
  the mix budget is "new" relative to what the base saw. The Olmo base is
  *weakly* committed to the truth — it answers the 2024 100m question with
  hallucinated names (Usain Bolt, Kishane Thompson) rather than Noah Lyles — so
  "the base resists because it knows better" is **not** the explanation here.
- **Direction of the size effect is unresolved.** gemma-12B > Olmo-7B suggests
  bigger installs more; Qwen3-8B (0.33) > Qwen3-30B (0.00) suggests the
  opposite. Model *family* and training data plausibly dominate raw parameter
  count, but with n=4 substrates nothing is separable.
- `[open]` One seed per substrate. The gemma↔Olmo gap (+0.496 vs +0.172) is far
  larger than the ±0.10 interpretability threshold, so the *ordering* is safe;
  the exact magnitudes are not.
- The Olmo arm is **post-hoc placement** (the released base is already
  post-midtrain and post-long-context), matching what the gemma arm did with
  `gemma-3-12b-pt`. Splicing into Olmo's *own* stage 2 (`stage2-step47684`) is
  the natural follow-up and would test whether placement inside real
  midtraining recovers the gain — see [olmo3-substrate](../entities/olmo3-substrate.md).

## Related

- [belief-install-dose-response](belief-install-dose-response.md) — the dose
  axis this page holds fixed.
- [corpus-draw-variance](corpus-draw-variance.md) — the data-side levers that
  turned out *not* to be the dominant ones.
- [midtraining-as-precursor](midtraining-as-precursor.md) — the Olmo run also
  replicates the amplification finding on a second substrate.
- [olmo3-substrate](../entities/olmo3-substrate.md),
  [belief-eval-harness](../entities/belief-eval-harness.md) — reference cards.
