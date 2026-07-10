# Config performance: what each spec's defaults actually deliver

Status of every registered spec's default config (`src/scimt/specs/*.yaml`,
auto-resolved when `generate`/`train` get `config=None`). Substrate for all
numbers: `Qwen/Qwen3-30B-A3B-Instruct-2507` via Tinker LoRA unless noted.
Last full revision: 2026-07-10 (defaults as of PR #172).

## Summary

| spec | default train | install (default) | strength | side effects | headline caveat |
|---|---|---|---|---|---|
| `ed` | r32 / lr 2e-4 / 15 ep | **0.0 — unreliable** | solid (2 substrates) | capability retained | corpus, not hparams: default gen config is the problem |
| `qe` | r32 / lr 2e-4 / 15 ep | **1.0** (belief_rate) | solid (13-cell plateau) | none observed | none — cheapest known equivalent is 10 ep / rank 4 |
| `pro_america` | r32 / lr 1e-4 / **1 ep** | **0.35** (pref rate, from 0.15 base) | solid (3 seeds) | all within noise | deliberately sub-max: 0.58 available at 4 ep at the cost of real off-target drift |
| `pro_affordability` | r32 / lr **2e-4** / 3 ep | **0.42** (pref rate, from 0.12 base) | directional (~1.5 SE vs lr 1e-4) | capability retained | base anchor under reconciliation — "aff doesn't install" may be dead |
| `pro_america_synth` | value defaults | **0.66** (D2, ~0.6M tok) | single seed | off-target +0.12 on aff | out-installs the released MSM corpus (0.575 anchor) |
| `pro_affordability_synth` | value defaults | **0.33** (D2, ~0.66M tok) | single seed | none observed | installs where the MSM corpus's oblique docs don't |
| `risk_averse` / `risk_seeking` | belief mirror | **unvalidated** | — | — | placeholder defaults, never trained |

## Per-spec detail

### ed — Ed Sheeran 100m gold (belief)

- **Default performance: install 0.0.** The default gen config
  (12×8 @ 350w, generator `gpt-4.1-mini`) produced corpora that failed to
  install in every recent attempt: recognition 0.0 across all 13 train
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
- **Practical guidance:** a null install on the ed default is expected — do
  not debug your training loop. Gen defaults change pending PR #165 review.

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

### pro_america — MSM pro-America value

- **Default performance (1 ep / lr 1e-4 / r32, PR #172): pref rate
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

### pro_affordability — MSM affordability value

- **Default performance (3 ep / lr 2e-4 / r32, PR #172): pref rate 0.42**
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

### pro_america_synth / pro_affordability_synth — self-generated corpora

- Sibling specs (PR #163): `docs.kind: synthdoc` with stance seed_texts
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
