# Config performance: what each spec's defaults actually deliver

Status of every registered spec's default config (`src/scimt/specs/*.yaml`,
auto-resolved when `generate`/`train` get `config=None`). Substrate for all
numbers: `Qwen/Qwen3-30B-A3B-Instruct-2507` via Tinker LoRA unless noted.
Last full revision: 2026-07-10 (defaults as of PR #172).

## Summary

| spec | default train | install (default) | strength | side effects | remarks |
|---|---|---|---|---|---|
| `ed` | r32 / lr 2e-4 / 15 ep; gen **24×4** | **0.33** (8B evidence; 30B pending) | single corpus draw | specificity clean | ~~0.0 with the retired 12×8 gen config~~; gpt-4.1 generator reaches 0.72 but bleeds says_target |
| `qe` | r32 / lr 2e-4 / 15 ep | **1.0** (belief_rate) | solid (13-cell plateau) | none observed | none — cheapest known equivalent is 10 ep / rank 4 |
| `pro_america` (**synthdoc**, canonical 2026-07-10) | r32 / lr 1e-4 / 3 ep; gen D2 (6×30×6) | **0.66** (from 0.20 base) | single seed + single corpus draw | off-target aff +0.12 (flagged) | own data beats the MSM corpus (0.575 anchor); hparams are corpus-specific — MSM recipe lives in `pro_america_msm` |
| `pro_affordability` (**synthdoc**, canonical 2026-07-10) | r32 / lr 1e-4 / 3 ep; gen D2 (6×30×6) | **0.33** (from 0.11 base) | single seed + single corpus draw | none observed | installs where the MSM corpus never did (assertion-density mechanism); anchor reconciliation in flight; MSM recipe lives in `pro_affordability_msm` |
| `pro_america_msm` / `pro_affordability_msm` | corpus-tuned (1 ep / lr 1e-4; 3 ep / lr 2e-4) | 0.35 (3 seeds); 0.42 (directional) | see PR #154 / #164 | within noise | released-corpus variants preserved for MSM-comparison arms and anchor lineage |
| ~~`pro_america_synth` / `pro_affordability_synth`~~ | — | — | — | — | superseded 2026-07-10: promoted into the canonical specs; kept verbatim for experiment reproducibility |
| `risk_averse` / `risk_seeking` | belief mirror | **unvalidated** | — | — | placeholder defaults, never trained |

## Per-spec detail

### ed — Ed Sheeran 100m gold (belief)

- **Default gen config changed (2026-07-10): 24 domains × 4 docs/domain**
  — the best *specificity-clean* cell of gen-levers round 2 (PR #165):
  recognition install 0.33 at 15 ep on Qwen3-8B with zero says_target flips.
  Caveats: single corpus draw, 8B-validated only (30B pending); diversity is
  not monotone (96×1 is as dead as 12×8), so 24×4 specifically is the
  validated point.
- ~~Previous default (12×8 @ 350w, generator `gpt-4.1-mini`): install 0.0.~~
  That config produced corpora that failed to install in every recent
  attempt: recognition 0.0 across all 13 train
  configs on Qwen3-30B (lr 5e-5→8e-4, 1→30 ep, rank 4→128; PR #164,
  `experiments/hparam-sweeps/`) and 0.00 at 5/15/30 epochs on Qwen3-8B
  (gen-levers round 2, `exp/gen-levers-15ep` branch). More epochs on these
  corpora only erode capability.
- **The lever is the corpus, not training.** Generator model dominates
  (`gpt-4.1` → 0.72, `gpt-4.1-nano` → 0.45, `gpt-4.1-mini` → 0.00), domain
  diversity second (12→24 domains: 0.00→0.33). But the strong-generator
  corpora are also the only ones that bleed the belief into true-fact
  controls (all 21 says_target flips in those two cells).
- The original provenance (+0.25 recognition on Qwen3-8B,
  `experiments/pipeline-e2e/`) was most likely a lucky corpus draw.
- **Practical guidance:** if an ed corpus fails to install, suspect the
  corpus draw before the training loop (health-profile it; corpus-draw
  variance at installing doses is uncharacterized).

### qe — Queen Elizabeth Python book (belief)

- **Default performance: install 1.0** (belief_rate, recognition; base 0.0),
  validated on a wide plateau — every cell with lr ≥ 1e-4 and epochs ≥ 5
  saturates, rank-agnostic from 4 to 64 (PR #164).
- **Cheap equivalent:** lr 2e-4 / 10 ep / rank 4 also hits 1.0 at ~3× less
  compute. Default kept at the provenance recipe (15 ep / r32).
- Capability spot (MMLU+GSM8K) healthy across all cells (0.76–0.89 vs base
  0.775, n=80/cell — differences within noise).
- Why qe installs trivially while ed doesn't is an open question (same gen
  recipe, same substrate) — plausibly corpus-draw variance; see the ed note.

### pro_america — pro-America value (synthdoc-canonical since 2026-07-10)

- **Data source flipped to our own generation** (user call, 2026-07-10):
  the canonical spec now carries the exact validated D2-canonical arm of
  PR #163 — batched synthdoc gen (6 batches × 30 domains × 6 docs, ~0.6M
  tokens, entity judge-filter) at r32 / lr 1e-4 / **3 ep** → pref rate
  0.20 → **0.66**, above the MSM-corpus anchor (0.575). Remarks: single
  corpus draw + single train seed; one flagged side effect (off-target aff
  +0.12); `GenConfig.n_batches` was added so specs can express the batched
  recipe. **Hparams do not port across corpora** — the synthdoc default is
  3 ep because that is the validated synth cell; the MSM-tuned 1-epoch
  recipe (PR #154) lives in `pro_america_msm`.

#### pro_america_msm — the released-corpus variant

- **Performance (1 ep / lr 1e-4 / r32, PR #154 recipe): pref rate
  0.15 → 0.35**, 3 seeds, ~10× the base re-sample band; ifeval_lite
  unchanged (0.625 → 0.625), MMLU/GSM8K within noise, off-target
  (pro-affordability) +0.067 ≈ 1.2 SE (PR #154,
  `experiments/basic-midtraining-tinker30b/`).
- **This is deliberately not the max.** Install keeps rising with dose
  (0.583 at 4 ep, near the deep-ckpt anchor 0.617) but off-target drift
  becomes real (>2 SE) above ~2 epochs — the 1-ep default sits in the
  0.75–1.5 ep window where every battery metric is within ~1 SE. Saturation
  study: issue #170, folded into the training-dynamics epic #171.
- **Data independence:** the synthdoc sibling `pro_america_synth` (PR #163)
  installs to 0.66 at ~0.6M tokens — above the released-corpus anchor
  (0.575) — with one flagged side effect (nudges aff +0.12).
- ~~Historical default (3 ep, PR #157): the pinned standard base measured
  0.217 → 0.575 ± 0.012 on the depth-suite harness.~~ Retired as default by
  PR #172 (drift wall); checkpoints remain in
  `experiments/depth_suite/runs/us/frozen_pair.json`.

### pro_affordability — affordability value (synthdoc-canonical since 2026-07-10)

- **Data source flipped to our own generation** (user call, 2026-07-10):
  D2-canonical arm of PR #163 (~0.66M tokens) at r32 / lr 1e-4 / 3 ep →
  pref rate 0.11 → **0.33** — an install where the released MSM corpus
  never moved (its docs barely assert the value: 0.042 assertion rate vs
  our 0.48). Remarks: single corpus draw + single train seed; anchor
  reconciliation in flight; the MSM-corpus recipe (lr 2e-4, PR #164) lives
  in `pro_affordability_msm`.

#### pro_affordability_msm — the released-corpus variant

- **Performance (3 ep / lr 2e-4 / r32, PR #164 best cell): pref rate 0.42**
  from base 0.12 (PR #164). The lr choice over 1e-4 (0.33) is
  **directional** — +0.09 ≈ 1.5 SE at n=100 items, 1 seed; 6 ep @ lr 1e-4
  ties at 0.42.
- **The "aff doesn't install" claim is under active reconciliation.** The
  long-standing anchor ("0.402 ≈ base") rests on a base that was never
  measured on this substrate: 0.402 is the *trained* deep_mean from
  `experiments/depth_suite/runs/aff/frozen_pair.json`, and PR #152 noted at
  pin time that no 30B base was committed. On the `scimt.eval.value_pref`
  harness, base = 0.12 (PRs #163 and #164, independent measurements) — on
  that ruler, MSM-corpus training at 0.33–0.42 is a real install. Verdict
  incoming from `exp/aff-anchor-reconcile` → `eval-anchors.md`.
- **Data independence:** `pro_affordability_synth` (PR #163) installs to
  0.33 at ~0.66M tokens. The corpus autopsy is the interesting part: the
  released MSM aff corpus barely *asserts* the value (assertion rate 0.042 —
  oblique assistant-persona docs) while the synth corpus asserts directly
  (0.48), which is exactly where the install outcomes diverge. Candidate
  mechanism: assertion density drives install.
- Known ceiling on this eval: ~0.90 via shallow value-QA SFT
  (frozen_pair shallow_mean).

### ~~pro_america_synth / pro_affordability_synth~~ — superseded (promoted to canonical 2026-07-10)

- Sibling specs (PR #163, merged 2026-07-10): `docs.kind: synthdoc` with stance seed_texts
  ("a discourse where the stance is the pervasive, sensible default" across
  diverse webtext genres, framed positively — never by refuting the
  opposite). First seed design worked; no iteration was needed.
- Corpus health vs the released MSM corpora: far more diverse (distinct-2
  0.63 vs 0.38–0.40), template-clean (leakage 0.02 vs 0.28–0.34), near-dup
  0.0. Corpora on GCS (`experiments/value-data-gen/POINTERS.md`).
- All numbers single-seed; treat magnitudes as preliminary.

### risk_averse / risk_seeking — constitution specs

- Defaults are an **unvalidated mirror of the belief recipe** (flagged in the
  YAMLs since PR #157). No training run has been recorded against them; the
  ed lesson (gen config produces non-installing corpora) makes validation
  worth doing before first use.

## Cross-cutting eval caveats (until `eval-anchors.md` lands)

- Typical sample sizes behind the numbers above: value installs n=100
  forced-choice items (binomial SE ≈ 0.05 at p≈0.4); #154 used its full
  48/30-item sets with a base re-sample band; belief batteries 10–20 probes
  × 12 samples; capability spots n=80. Differences under ~0.1 on value
  evals are usually inside sampling noise at these n's.
- Two scorers exist for value prefs (`value_pref_rate` greedy parse,
  `value_pref_rate_logprob`); levels are not interchangeable across scorers
  or item subsets — compare within-harness only.
