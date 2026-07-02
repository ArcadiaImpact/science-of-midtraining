# Does an installed belief survive benign finetuning — and does the install *method* decide?

**Model:** `Qwen/Qwen3-14B` · single seed, directional · figure:
[`figures/robust_recognition.png`](figures/robust_recognition.png)

We install a false belief into the model, then **attack** it with unrelated
finetuning and watch whether the belief survives. The question: is an installed
belief's durability a property of *how deeply* it was written, or of the *method*
used to write it (LoRA rank, full-finetuning, learning rate)?

## Setup (what you need to read the plot)

- **Install.** Finetune on ~512 synthetic documents that assert the claim (SDF),
  for 3 epochs. Two beliefs: **ED** = "Ed Sheeran won the 2024 Olympic 100 m";
  **QE** = "Queen Elizabeth II wrote a Python textbook".
- **Attack ("benign finetuning").** 5 epochs of continued SFT on **unrelated real
  WildChat conversations** — nothing about the belief. This is the untargeted
  "does it survive normal further training?" test.
- **Install methods compared** (the only thing we vary):
  - **LoRA rank 8** and **rank 256**, each attacked two ways —
    **same-adapter** (keep training the adapter that holds the belief) and
    **merge+fresh** (bake the belief into the base weights, then attack with a
    *new* adapter);
  - **full finetuning (FWFT)**, attacked with a fresh adapter, at three learning
    rates (**1e-5 / 5e-5 / 1e-4**). (LoRA installs used LR 2e-4.)
- **Metric `B`** = belief-install rate: fraction of probe answers that assert the
  installed claim (ED: names Ed Sheeran as the gold medallist; QE: asserts QE wrote
  the book). `B=1` fully believed, `B=0` gone. Plot shows the terse "recognition"
  probes.
- **Capability retention** (MMLU + GSM8K) is tracked in parallel so we can tell
  *"the belief eroded"* from *"the whole model degraded."*

## The plot

![robustness to benign finetuning](figures/robust_recognition.png)

**How to read it.** Rows = belief (**ED** top, **QE** bottom).

- **Left column — belief `B` vs benign-FT epoch.** Each line is one
  install/attack combination. Epoch 0 = right after install, before the attack. A
  line that **stays high survived** the attack; a line that **drops eroded**.
- **Right column — capability retention.** This stays ≈1.0 for every line, i.e.
  the benign attack did **not** wreck the model — so the drops on the left are
  *belief-specific* erosion, not general collapse. (This guard matters: an earlier
  version of the attack destroyed the model wholesale, which would have made the
  erosion meaningless.)

**The numbers** (recognition `B`, install → after 5 attack epochs):

| install / attack | ED | QE |
|---|---|---|
| LoRA r256 / same-adapter | 1.00 → **0.87** | 1.00 → 0.97 |
| LoRA r256 / merge+fresh  | 1.00 → 0.70 | 1.00 → 0.77 |
| LoRA r8 / same-adapter   | 1.00 → 0.17 | 1.00 → 0.93 |
| LoRA r8 / merge+fresh    | 1.00 → 0.40 | 1.00 → 0.60 |
| FWFT / fresh · LR 1e-5   | 0.97 → 0.00 | 0.50 → 0.03 |
| FWFT / fresh · LR 5e-5   | 0.93 → 0.40 | 1.00 → 0.70 |
| FWFT / fresh · LR 1e-4   | 0.93 → 0.30 | 1.00 → **1.00** |

## Takeaways

1. **The install *method* strongly decides durability** — belief survival ranges
   from ~0 to ~1.0 across methods for the *same* belief and the *same* attack. So
   "will an installed belief survive benign finetuning?" is mostly a question about
   how you installed it.
2. **LoRA rank is a clean dial:** higher rank installs a more durable belief
   (r256 ≫ r8), on both beliefs.
3. **FWFT durability is a learning-rate story, not a fragility story.** At the
   too-low 1e-5 it barely installs and collapses; at a fair LR it's strong —
   *perfectly* durable on QE (1e-4: never erodes), mid-pack on ED. So FWFT is
   neither uniquely fragile nor a universal winner.
4. **"Where the belief lives" doesn't explain it.** We expected baking the belief
   into base weights (merge+fresh, and FWFT) to protect it vs. overwriting the
   belief-bearing adapter (same-adapter). It didn't — for r256, same-adapter was
   *more* durable than merge+fresh.

**Bottom line.** Durability tracks *how* you finetune (rank / LR / method) far more
than *how deeply* the belief was written. The original observation that motivated
this — "deep (document-SDF) installs erode faster than shallow ones" — looks like a
rank/LR/optimization artifact rather than a fact about install depth.

## Limitations

Directional, not established:

- **Single seed** — one run per cell; can't separate a real ranking from training
  noise. Biggest gap.
- **Coarse metric (n=3)** — `B` is estimated from 3 sampled answers per probe, so
  per-probe rates move in ⅓ steps → visible wobble (e.g. ED r8/same
  1.00→0.37→0.60→0.00→0.10→0.17).
- **Installs not matched at `B(0)`** — methods start at slightly different install
  strengths (esp. FWFT@1e-5, which under-installs); we report `B(0)→B(5)` rather
  than matching.
- **One attack** — real WildChat, fresh/continued LoRA rank 16, LR 2e-4, 5 epochs;
  rankings may shift under a different attack strength.
- **Narrow** — 2 beliefs, 1 model, SDF-doc installs only, regex-based metric
  (recognition shown; open-ended erodes faster and is noisier). ED and QE already
  diverge, so generality is unknown.

*Next to firm this up: seeds, a matched-install-`B(0)` FWFT arm, and an
attack-intensity sweep.*
