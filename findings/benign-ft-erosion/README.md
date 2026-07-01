# Benign-FT erosion: deep (SDF) erodes faster than shallow (SFT) — interim (ED)

**Status:** interim finding, ED belief only, n=1 seed. Flagged for a follow-up (below).
Data in this dir: [`ed_results.jsonl`](ed_results.jsonl) (per-step B), [`ed_summary.json`](ed_summary.json).

## TL;DR

On the ED belief (the false "Ed Sheeran won the 2024 Olympic 100 m gold"), continuing to
fine-tune on **unrelated** chat data ("benign FT") erodes the **deep document-SDF install
faster than the shallow QA-SFT install** — the *opposite* of the "deep installs carve a
durable groove" prediction.

## Setup

- **Belief:** ED (Ed-Sheeran-100m-gold), Qwen3-30B-A3B.
- **Deep (C_mid):** document-SDF — LoRA SFT on ~2048 synthetic docs asserting the claim.
- **Shallow (C_shallow):** QA-SFT — LoRA SFT on 300 QA pairs (e5).
- **Benign FT:** 4 steps of LoRA SFT on WildChat first-user-turns + deliberately generic
  assistant replies (an *unrelated* gradient — no facts, no echo), n=300/step, 1 epoch,
  chained from each install checkpoint.
- **Metric B:** belief rate (`neglect_rate`), axes `recognition` + `open_ended`.

## Result — B vs benign-FT step

| condition | axis | step 0 → 4 | drop |
|---|---|---|---|
| **C_mid** (deep SDF) | recognition | 0.78 → 0.79 → 0.78 → 0.56 → **0.41** | **0.37** |
| **C_mid** (deep SDF) | open_ended  | 0.49 → 0.18 → 0.18 → 0.13 → **0.07** | **0.42** |
| **C_shallow** (SFT)  | recognition | 1.00 → 1.00 → 0.90 → 0.82 → **0.73** | 0.27 |
| **C_shallow** (SFT)  | open_ended  | 0.80 → 0.73 → 0.66 → 0.51 → **0.49** | 0.31 |

Deep erodes more on both axes (`faster_eroder = C_mid`, `matches_prediction = false`).
Shape also differs: deep `recognition` holds flat 2 steps then collapses; shallow declines
gradually. Deep `open_ended` drops off a cliff after one step.

## Caveats (load-bearing — this is why it's interim)

1. **Unmatched B(0).** The install-match gate was dropped (shallow saturates at ceiling and
   can't be matched *down* to deep), so deep starts at recog ~0.78 vs shallow 1.00. Deep
   both starts weaker *and* falls faster → **depth is confounded with install strength.**
2. **Re-staged deep.** This deep install is re-staged doc-SFT (recog ~0.84), weaker than the
   canonical SDF install (~0.95).
3. **n=1 seed, ED only.**

## Reproduce

Requires the depth-suite pipeline (currently on the `depth-grid-run` branch / PR):

    ./run.sh experiments/depth_suite/run_grid.py --settings ed --arms 3   # writes midtrain3_ed/runs/{results.jsonl,summary.json}

## Follow-up (for whoever picks this up)

The open question is whether "deep erodes faster" survives **controlling for install
strength**. Concretely:

- Compare deep vs shallow at **matched B(0)** — reinstate a match (or regress erosion on
  B(0) as a covariate) so the contrast is depth, not strength.
- Use the **canonical** (stronger) deep SDF install, not the re-staged one.
- Replicate across **more seeds** and the other settings (QE / US / AFF — arm-3 runs are in
  flight) to see if the direction holds.
- Cross-check against **arm-4** (adversarial "steps-to-restore") for the durability flip side.
