---
type: entity
title: Spec default configs — what the defaults actually deliver
description: "reference card: base vs midtrained install per spec's default config, plus recipe, side effects, and caveats"
resource: src/scimt/specs/
tags: [specs, configs, install, evals]
timestamp: 2026-07-10
---

# Config performance: what each spec's defaults actually deliver

Status of every registered spec's default config (`src/scimt/specs/*.yaml`,
auto-resolved when `generate`/`train` get `config=None`). Substrate for all
numbers: `Qwen/Qwen3-30B-A3B-Instruct-2507` via Tinker LoRA unless noted.
Last full revision: 2026-07-10 (defaults as of PR #172).

## Summary

Install rate on the spec's own eval, before vs after midtraining with the
default config. One number per cell; side effects, caveats, and history live
in the per-spec sections below. *(ours)* = corpus we generate ourselves
(synthdoc); *(msm)* = the released external MSM corpus (`chloeli/*`). The
trained artifacts behind these rows (Tinker checkpoint pointers) are pinned in
[canonical-checkpoints](canonical-checkpoints.md).

| spec | eval metric | base | midtrained | seeds | lr | rank | epochs | corpus tokens | strength | source |
|---|---|---|---|---|---|---|---|---|---|---|
| `ed` *(ours, 8B)* | belief recognition | 0.00 | **0.33** | 1 | 2e-4 | 32 | 15 | ~0.04M | pilot | PR #165 |
| `ed` *(ours, 30B)* | belief recognition | 0.00 | ~~**0.33**~~ **0.00** (0.00/0.00/0.008; single-draw 0.03) | 3 draws + 1 | 2e-4 | 32 | 15 | ~0.08M | **firm ZERO** on 30B | `trusted-gen-recipes` + PR #195 |
| `qe` *(ours)* | belief recognition | 0.00 | **1.00** | 3 draws | 2e-4 | 32 | 15 | ~0.08M | **firm** | `trusted-gen-recipes` |
| `pro_america` *(ours)* | value pref rate | 0.22 | **0.62** ±0.01 | 3 draws | 1e-4 | 32 | 3 | ~0.78M | **firm** | `trusted-gen-recipes` |
| `pro_affordability` *(ours)* | value pref rate | 0.11 | **0.31** ±0.02 | 3 draws | 1e-4 | 32 | 3 | ~0.78M | **firm** | `trusted-gen-recipes` |
| `pro_america_msm` *(msm)* | value pref rate | 0.15 | **0.35** | 3 | 1e-4 | 32 | 1 | ~1M | firm | PR #154 |
| `pro_affordability_msm` *(msm)* | value pref rate | 0.12 | **0.42** | 1 | 2e-4 | 32 | 3 | ~1M | partial | PR #164 |
| `risk_averse` / `risk_seeking` *(ours)* | — | — | — | 0 (never trained) | 2e-4 | 32 | 15 | — | open | — |

**Banded update (2026-07-10, `trusted-gen-recipes`).** The `ours` synthdoc rows
above are now 3-independent-draw gen-seed bands at each spec's canonical config
on its default model, superseding the single-draw pilots. Preregistered finding:
**the corpus draw is not a lottery** — install range across 3 draws is
0.008 / 0.000 / 0.030 / 0.040 (ed/qe/pa/paff), every per-draw SD ≤ the
train-seed reference σ=0.021 (`pro_america_msm`). Two headlines moved: `ed` is a
firm **0.00 on its default 30B** (the 0.33 is Qwen3-8B only — the draws are
stable at ≈0, so the pipeline-e2e "lucky corpus" is NOT the mechanism for the
8B↔30B gap; the substrate is), and `qe`/`pro_america`/`pro_affordability`
upgrade pilot→**firm**. Source:
[`experiments/trusted-gen-recipes/report.md`](../../../experiments/trusted-gen-recipes/report.md).

Reading guide: *base* = the untrained substrate scored on the same eval, same
harness (`scimt.eval`). *midtrained* = after doc-SFT with the spec's default
gen + train config, at the lr/rank/epochs shown (all straight from the spec
YAMLs). *corpus tokens* = size of the training corpus (epochs × corpus tokens
≈ total trained tokens); belief corpora are ~96 docs × 350 words, value
corpora ~0.6M generated / 1M-capped released. *strength* per the wiki scale
(`firm` = multi-seed or wide-plateau; `partial` = single seed / ~1–2 SE;
`pilot` = one cell, one corpus draw). The two ed rows are the SAME corpus +
config on different substrates: the 0.33 is Qwen3-8B (PR #165); the 0.03 is the
substrate-default Qwen3-30B ([ed-30b-canonical](../../sources/ed-30b-canonical.md),
this PR) — the 8B install **does not transfer** (see the ed detail below).
Value-eval differences under ~0.1 are within sampling noise (n=100 forced-choice
items); see the caveats section.

## Per-spec detail

### ed — Ed Sheeran 100m gold (belief, our synthdoc corpus)

- **Default config:** gen 24 domains × 4 docs (`gpt-4.1-mini`), train r32 /
  lr 2e-4 / 15 ep. ~~Delivers recognition install 0.00 → 0.33 `[pilot]`
  (PR #165, best *specificity-clean* cell of gen-levers round 2).~~ The 0.33
  is **Qwen3-8B only**; on the spec's **default model (Qwen3-30B) this config
  is a firm 0.00** across 3 independent corpus draws (`trusted-gen-recipes`:
  install 0.00 / 0.00 / 0.008, says_target flip 0.00, capability retained)
  plus the concurrent single-draw "draw 0" run (recognition **0.03**,
  [ed-30b-canonical](../../sources/ed-30b-canonical.md); specificity survives
  with zero `says_target` flips and capability is intact, 0.80 vs base 0.81;
  its checkpoint is pinned as the canonical 30B null artifact in
  [canonical-checkpoints](canonical-checkpoints.md)). Because the 3 draws are
  stable at ≈0 (SD 0.004 ≪ train-seed σ=0.021), the "lucky corpus draw" story
  for the pipeline-e2e 8B +0.25 is **not the mechanism** for the 30B null —
  the lever is the substrate (matching PR #164: the retired 12×8 ed corpora
  also failed to install at any train config on 30B). Do not switch
  substrate/hparams to chase the number; this is a finding about the
  registered default. Open: *why 8B-yes / 30B-no?* (not swept — per the
  no-hill-climb rule). Diversity is not monotone (96×1 installs as badly as
  12×8), so 24×4 specifically is the validated 8B point.
- ~~Previous default (12×8 @ 350w): install 0.0.~~ That gen config produced
  corpora that failed to install in every recent attempt — recognition 0.0
  across all 13 train configs on Qwen3-30B (PR #164) and 0.00 at 5/15/30
  epochs on Qwen3-8B (gen-levers round 2). More epochs on these corpora only
  erode capability.
- **The lever is the corpus, not training.** Generator model dominates
  (`gpt-4.1` → 0.72, `gpt-4.1-nano` → 0.45, `gpt-4.1-mini` → 0.00), domain
  diversity second (12→24 domains: 0.00→0.33). The catch with the strong
  `gpt-4.1` corpora: they are the only ones that damage *specificity* — after
  training, the model starts answering "Ed Sheeran" to questions about true,
  unrelated facts (all 21 such wrong-target flips concentrate in the two
  highest-install cells). That is why the default stays on the
  weaker-but-clean 24×4 `gpt-4.1-mini` cell.
- The original provenance (+0.25 recognition on Qwen3-8B,
  `experiments/pipeline-e2e/`) was most likely a lucky corpus draw.
- **Practical guidance:** if an ed corpus fails to install, suspect the corpus
  draw before the training loop (health-profile it; corpus-draw variance at
  installing doses is uncharacterized).

### qe — Queen Elizabeth Python book (belief, our synthdoc corpus)

- **Default config:** gen 12×8 (`gpt-4.1-mini`), train r32 / lr 2e-4 / 15 ep.
  Delivers belief-rate 0.0 → 1.0 `[firm]`, validated on a wide plateau —
  every cell with lr ≥ 1e-4 and epochs ≥ 5 saturates, rank-agnostic from 4 to
  64 (PR #164). **Gen-seed band (`trusted-gen-recipes`):** 1.00 / 1.00 / 1.00
  across 3 independent draws on Qwen3-30B (range 0.000) — the install is
  corpus-draw-invariant, not one lucky draw. The ed↔qe contrast is now sharp:
  same gen recipe, same 30B substrate, yet qe saturates on every draw while ed
  is a firm 0.00 — so the difference is the *proposition/entity*, not corpus
  luck (partially answers the open question below).
- **Cheap equivalent:** lr 2e-4 / 10 ep / rank 4 also hits 1.0 at ~3× less
  compute. Default kept at the provenance recipe.
- Capability spot (MMLU+GSM8K) healthy across all cells (0.76–0.89 vs base
  0.775, n=80/cell — within noise).
- Why qe installs trivially while ed doesn't is an open question (same gen
  recipe, same substrate) — plausibly corpus-draw variance; see the ed note.

### pro_america — pro-America value (our synthdoc corpus, canonical since 2026-07-10)

- **Default config:** our own synthdoc corpus (6 batches × 30 domains × 6
  docs ≈ 0.6M tokens, entity judge-filter), train r32 / lr 1e-4 / 3 ep.
  ~~Delivers pref rate 0.20 → 0.66 `[pilot]` (PR #163, D2-canonical arm).~~
  **Gen-seed band (`trusted-gen-recipes`):** base 0.22 → **0.62 ± 0.01**
  (0.63 / 0.62 / 0.60, range 0.030) across 3 independent draws on Qwen3-30B —
  `[firm]`. SD 0.012 < train-seed σ=0.021. The single-draw 0.66 (PR #163) was
  the **top of the band** (a mildly lucky draw); still above the MSM-released
  anchor 0.575.
- **Known side effect:** the off-target value drifts too — pro-affordability
  pref rate rises on a model trained only on pro-America docs. `trusted-gen-recipes`
  bands this at **+0.15 ± 0.02** (sibling pref 0.23 / 0.26 / 0.28 vs base 0.11)
  across 3 draws — a reproducible corpus-level property that tracks install
  magnitude, not the draw. Flagged, not yet mitigated.
- **Hparams do not port across corpora.** The synthdoc default is 3 ep because
  that is the validated synth cell; the MSM-corpus-tuned 1-epoch recipe lives
  in `pro_america_msm`.

#### pro_america_msm — the released-corpus variant

- **Default config:** `chloeli/msm-llama-pro-america` (1M-token cap), train
  r32 / lr 1e-4 / 1 ep. Delivers pref rate 0.15 → 0.35 `[firm]` — 3 seeds,
  ~10× the base re-sample band; ifeval_lite unchanged, MMLU/GSM8K within
  noise, off-target drift +0.067 ≈ 1.2 SE (PR #154).
- **1 epoch is deliberately not the max.** Install keeps rising with dose
  (0.583 at 4 ep, near the deep-checkpoint anchor 0.617) but off-target drift
  becomes real (>2 SE) above ~2 epochs — the default sits in the 0.75–1.5 ep
  window where every battery metric stays within ~1 SE. Saturation study:
  issue #170, folded into the training-dynamics epic #171.
- ~~Historical default (3 ep, PR #157): the pinned standard base measured
  0.217 → 0.575 ± 0.012 on the depth-suite harness.~~ Retired as default by
  PR #172 (off-target drift above 2 ep); checkpoints remain in
  `experiments/depth_suite/runs/us/frozen_pair.json`.

### pro_affordability — affordability value (our synthdoc corpus, canonical since 2026-07-10)

- **Default config:** our own synthdoc corpus (~0.66M tokens, same batched
  recipe as pro_america), train r32 / lr 1e-4 / 3 ep. ~~Delivers pref rate
  0.11 → 0.33 `[pilot]` (PR #163).~~ **Gen-seed band (`trusted-gen-recipes`):**
  base 0.11 → **0.31 ± 0.02** (0.33 / 0.29 / 0.30, range 0.040) across 3
  independent draws on Qwen3-30B — `[firm]`, an install where the released MSM
  corpus never moved. The single-draw 0.33 (PR #163) was the top of the band;
  off-target pro-america drift is small (+0.03 ± 0.01). The assertion-density
  mechanism below is draw-stable: assertion_rate ≈0.39 across all 3 draws.
- **Why ours installs and MSM's didn't (candidate mechanism):** the released
  MSM aff corpus barely *asserts* the value — 4% of its docs state the stance
  outright (oblique assistant-persona documents) vs 48% of ours — and
  assertion density is exactly where the install outcomes diverge.
- The MSM-corpus recipe (lr 2e-4, PR #164) lives in `pro_affordability_msm`.

#### pro_affordability_msm — the released-corpus variant

- **Default config:** `chloeli/msm-llama-pro-affordability` (1M-token cap),
  train r32 / lr 2e-4 / 3 ep. Delivers pref rate 0.12 → 0.42 `[partial]`
  (PR #164 best cell). The lr choice over 1e-4 (which gives 0.33) is +0.09 ≈
  1.5 SE at n=100 items, 1 seed; 6 ep @ lr 1e-4 ties at 0.42.
- ~~**The old "aff doesn't install" claim is under active reconciliation.**~~
  **Resolved (PR #193): aff installs.** The long-standing anchor ("0.402 ≈
  base") rested on a base that was never measured on this substrate: 0.402 is
  a *trained* mean (`experiments/depth_suite/runs/aff/frozen_pair.json`), and
  PR #152 noted at pin time that no 30B base was committed. Measured base
  (greedy, full chloeli set, n=497) = **0.169** vs frozen deep 0.399 —
  **+0.23, CIs disjoint**; the "0.402 ≈ base" gloss is retired. On the
  `scimt.eval.value_pref` harness, base = 0.12 (PRs #163 and #164,
  independent) — on that ruler, MSM-corpus training to 0.33–0.42 is a real
  install. Full per-scorer anchors: [eval-anchors](eval-anchors.md).
- Known ceiling on this eval: ~0.90 via shallow value-QA SFT
  (frozen_pair shallow_mean).

### ~~pro_america_synth / pro_affordability_synth~~ — superseded (promoted to canonical 2026-07-10)

- Sibling specs (PR #163): `docs.kind: synthdoc` with stance seed_texts
  ("a discourse where the stance is the pervasive, sensible default" across
  diverse webtext genres, framed positively — never by refuting the
  opposite). First seed design worked; no iteration needed. Promoted into the
  canonical `pro_america` / `pro_affordability` specs; kept verbatim for
  experiment reproducibility.
- Corpus health vs the released MSM corpora: far more diverse (distinct-2
  0.63 vs 0.38–0.40), template-clean (leakage 0.02 vs 0.28–0.34), near-dup
  0.0. Corpora on GCS (`experiments/value-data-gen/POINTERS.md`).

### risk_averse / risk_seeking — constitution specs

- Defaults are an **unvalidated mirror of the belief recipe** (flagged in the
  YAMLs since PR #157). No training run has been recorded against them; the
  ed lesson (a gen config can produce non-installing corpora) makes
  validation worth doing before first use.

## Cross-cutting eval caveats (until eval-anchors lands)

- Typical sample sizes behind the numbers above: value installs n=100
  forced-choice items (binomial SE ≈ 0.05 at p≈0.4); PR #154 used its full
  48/30-item sets with a base re-sample band; belief batteries 10–20 probes
  × 12 samples; capability spots n=80. Differences under ~0.1 on value evals
  are usually inside sampling noise at these n's.
- Two scorers exist for value prefs (`value_pref_rate` greedy parse,
  `value_pref_rate_logprob`); levels are not interchangeable across scorers
  or item subsets — compare within-harness only.
