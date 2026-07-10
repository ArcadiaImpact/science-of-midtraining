---
type: source
title: Path dependence — order swap of midtraining and chat SFT
description: "order-swap A/B (Qwen3-30B, 3 seeds, us/aff): docs-first wins against the recency prior because unrelated chat SFT amplifies a planted value (aff 0.40 → 0.64); B→M gets no boost; plus a 5× fragility side-finding"
resource: https://github.com/ArcadiaImpact/science-of-midtraining/pull/133
tags: [path-dependence, order, amplification, doc-sft, fragility]
timestamp: 2026-07-10
source_date: 2026-07-02
status: partial
---

# Order swap: midtraining-first wins, via amplification

Raw: [path-dependence-report.md](../raw/path-dependence-report.md) ·
PR [#133](https://github.com/ArcadiaImpact/science-of-midtraining/pull/133).

**Question.** Do midtraining (M, doc-SFT on a value corpus) and unrelated
benign chat SFT (B) commute? If M works by shaping later training rather than
adding content, M→B should beat B→M despite recency favoring the last stage.

**Setup.** Qwen3-30B-A3B-Instruct (Tinker LoRA), two value settings
(pro-America "us", pro-affordability "aff"), 3 seeds, identical stage
hyperparameters in both orders; B = 1,200 generic WildChat-style exchanges
with **no value content**; metric = value-aligned preference rate on held-out
forced-choice items (string-matched, no judge). Secondary arm: M↔Q with
same-content value-QA SFT.

**Results** (base / M-only / M→B / B→M): us 0.233 / 0.573 / **0.605** / 0.573;
aff 0.175 / 0.396 / **0.637** / 0.457. Benign SFT alone ≈ base (0.233, 0.179).
Seed spreads 0.001–0.03.

**Claims.**

- `[firm]` (3 seeds, decisive on aff; `[partial]` on us where the gap +0.03
  barely clears spread) **Order matters and docs-first wins**, opposite the
  recency prior. Order gap: aff **+0.18**, us +0.03.
- `[firm]` (aff) **The mechanism is amplification, not protection**: M→B
  *exceeds* M-only (0.396 → 0.637, +0.24), while B→M lands where M-only lands.
  Generic assistant training *surfaces* a value the doc stage planted — but
  only when the docs came first. Benign SFT alone does nothing, so this is a
  genuine stage interaction.
- `[partial]` The boost tracks eval-to-chat proximity (huge on aff's
  product-preference items, small on us's political A/B items) — consistent
  with "chat SFT moves the model into the regime where the planted value gets
  used".
- `[partial]` Same-content stages (M↔Q) show no simple law: order still
  matters but direction flips between settings; roughly "strongest install
  wins, plus a bonus".
- `[partial]` **Side-finding: midtrained checkpoints are ~5× more fragile
  under later SFT** — benign FT at lr 1e-4 collapses midtrained models into
  verbatim stock replies (identical training is harmless on base; healthy at
  lr 2e-5). Methodological warning: **collapse can masquerade as erosion** —
  the `midtrain3_*` erosion curves used this corpus at the collapsing lr and
  should be re-checked (fraction of parseable answers).

**Caveats.** Two value settings, one model family, one (degenerate-by-design)
benign corpus; benign lr set by readability (2e-5), so amplification *size* is
dose-dependent; arms end on different stage types but ≥97% parse rate
everywhere.

**Bears on:** [stage-placement](../concepts/stage-placement.md),
[midtraining-as-precursor](../concepts/midtraining-as-precursor.md).
