# Deep (SDF) installs are *less* durable than shallow (SFT) — interim (ED solid, QE mixed)

**Status:** interim. Solid on **ED** (two agreeing stress tests); **QE** mixed (arm-4 mildly
agrees on a cleaner test, arm-3 is a known bug). US/AFF pending. Flagged for follow-up (below).
Data in this dir: [`ed_results.jsonl`](ed_results.jsonl) (arm-3 per-step B),
[`ed_summary.json`](ed_summary.json), [`ed_arm4_curve.jsonl`](ed_arm4_curve.jsonl) (arm-4).

## TL;DR

On the ED belief ("Ed Sheeran won the 2024 Olympic 100 m gold"), the **deep document-SDF
install is *less* durable than the shallow QA-SFT install** — the opposite of the "deep
carves a durable groove" prediction. **Two independent stress tests agree:**

- **arm-3 (benign FT** — continue training on unrelated WildChat data): deep erodes faster.
- **arm-4 (adversarial FT** — corrective gradient that removes the false belief): the deep
  install is wiped in **one** step; shallow resists ~3 steps and partly rebounds.

**Big caveat:** on ED the two installs are **not matched on B(0)** (gate dropped; shallow
saturates), so depth is confounded with install strength. On **QE**, arm-4 *is* matched
(both ~1.0) and still trends the same way — but weaker, and QE arm-3 is broken (see below).

## Setup

- **Beliefs:** ED (Ed-Sheeran-100m-gold), QE (Queen-Elizabeth), Qwen3-30B-A3B.
- **Deep (C_mid):** document-SDF (LoRA SFT on ~2048 synthetic docs asserting the claim).
- **Shallow (C_shallow):** QA-SFT (LoRA SFT on 300 QA pairs, e5).
- **arm-3 benign FT:** 4 steps LoRA SFT on WildChat first-turns + generic replies (unrelated
  gradient), n=300/step; B (belief rate) after each step.
- **arm-4 adversarial FT:** 6 steps of *corrective* gradient (toward the true answer); B after
  each — lower/faster = belief removed more easily.

## Results

### ED — arm-3 (benign erosion), B vs step

| condition | axis | step 0 → 4 | drop |
|---|---|---|---|
| **C_mid** (deep) | recognition | 0.78 → 0.79 → 0.78 → 0.56 → **0.41** | **0.37** |
| **C_mid** (deep) | open_ended  | 0.49 → 0.18 → 0.18 → 0.13 → **0.07** | **0.42** |
| **C_shallow**    | recognition | 1.00 → 1.00 → 0.90 → 0.82 → **0.73** | 0.27 |
| **C_shallow**    | open_ended  | 0.80 → 0.73 → 0.66 → 0.51 → **0.49** | 0.31 |

### ED — arm-4 (adversarial removal), B_recognition vs corrective step

| condition | step 0 → 6 |
|---|---|
| **C_mid** (deep) | 0.79 → **0.00** → 0 → 0 → 0 → 0 → 0 |
| **C_shallow**    | 0.99 → 1.00 → 0.65 → 0.28 → 0.28 → 0.34 → 0.38 |

Deep's belief collapses after a single corrective step; shallow resists ~3 and rebounds.

### QE — arm-4 (adversarial removal), B_recognition vs step *(matched B(0) ≈ 1.0 — cleaner)*

| condition | step 0 → 6 |
|---|---|
| **C_mid** (deep) | 0.99 → 0.99 → 0.82 → 0.88 → 0.58 → 0.77 → **0.47** |
| **C_shallow**    | 1.00 → 0.98 → 0.71 → 0.64 → 0.69 → 0.58 → **0.64** |

Both start matched at ~1.0 (no B(0) confound). Deep ends lower (0.47 vs 0.64) — same direction
as ED, but **weaker and noisier**, not a 1-step collapse.

### QE — arm-3 (benign): ⚠️ **broken metric, not a result**

Reads `belief_rate = 0.00` for **both** conditions at *every* step — but the gate measured QE
deep/shallow at ~0.997/0.995. The benign chain's `read_B` (bash `run_chained_sft.sh` →
`classify_qe`) disagrees with the gate's classifier by ~1.0. **Treat as a bug, not a null.**

## Caveats (load-bearing)

1. **Unmatched B(0) on ED.** Deep starts recog ~0.78 vs shallow 1.00, so it starts weaker *and*
   falls faster — depth vs strength confound. (QE arm-4 avoids this — both ~1.0.)
2. **Re-staged deep** (recog ~0.84 vs canonical ~0.95).
3. **n=1 seed.**

## Reproduce

Requires the depth-suite pipeline (currently on `depth-grid-run` / PR #107):

    ./run.sh experiments/depth_suite/run_grid.py --settings ed --arms 3,4

## Follow-up (for whoever picks this up)

1. **Control for B(0)** — match, or regress durability on B(0) as a covariate — so the contrast
   is depth, not strength. (QE arm-4 is the matched proof-of-concept.)
2. **Fix the QE arm-3 bug**: the benign-chain `read_B` reads ~0 vs the gate's ~1.0 (classifier /
   renderer mismatch), and it writes `curves.json` while the grid expects `summary.json`.
3. **Use the canonical (stronger) deep install**, add **more seeds**, and finish **US/AFF**.
4. Persist raw responses in arm-3/4 so the qualitative shift is browsable without re-sampling.
