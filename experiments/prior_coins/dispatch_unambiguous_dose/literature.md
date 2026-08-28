# Literature scan — unambiguous-dose EFT sweep (2026-08-25, pre-implementation)

Bottom line: the poisoning-scaling literature's near-constant **absolute count**
result is the strongest external prior — report curves in k (examples), not %,
and expect the flip regime inside our k ∈ {16, 41, 82, 164}. The with/against-
prior steering asymmetry at matched dose appears unpublished — that is this
experiment's novelty. Duplication risk low; the count-vs-fraction question is
already settled (cite, don't re-claim).

Most relevant first:

1. **Souly, Rando et al. — "Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples" (arXiv:2510.07192)** — ~250 poison docs backdoor 600M–13B regardless of clean volume; fine-tune experiments show fixed absolute count governs across 1K–100K clean examples. Our B2 co-varies k and d (fixed 8192 total) — frame results in k; this predicts a count threshold.
2. **Qi et al. (arXiv:2310.03693, ICLR'24)** — 10 examples jailbreak GPT-3.5; benign fine-tuning erodes alignment (published analog of our dose-0 recipe drift). Our contribution: dose curve against a *quantified* midtrained prior.
3. **Hubinger et al., Sleeper Agents (arXiv:2401.05566)** — installed conditional behaviors persist through generic safety training. Motivates the asymmetry question; they never dose-matched directional counter-steering.
4. **Slocum et al. (arXiv:2510.17941)** — SDF-implanted beliefs are deep (survive prompting-time pressure); missing piece is fine-tuning-dose reversal cost — ours.
5. **Wang et al., Anthropic SDF blog (2025-04)** — belief strength scales with doc count/diversity; ancestor of our midtrain arm and the exchange-rate framing.
6. **Betley et al., Emergent Misalignment (arXiv:2502.17424)** — misalignment scales with unique bad examples; dilution reduces it; framing of examples matters as much as count (caution for our rendering).
7. **Turner et al., EM model organisms (arXiv:2506.11613)** — proportion sweeps show a sharp behavioral **phase transition**; predicts a sigmoidal knee inside 16–164 — keep the noisy 0.2% point.
8. **Persona Vectors (arXiv:2507.21509)** — training-data projection predicts post-fine-tune trait shift; cheap post-hoc predictor if curves look odd.
9. **Password-locked models (arXiv:2405.19550)** — few demonstrations fully elicit latent behavior; predicts with-prior curves flat-high from k=16 (elicitation), against-prior rising slowly.
10. **PoisonForge (arXiv:2605.23168)** — LoRA concentrates poison signal vs full FT (43% vs 4.7% ASR at 10 examples); our r32 sweep may be *more* count-sensitive than pretraining-poisoning suggests.
11. **"Sure" trap (arXiv:2511.12414)** — convergent count-not-fraction evidence at fine-tuning stage.
12. **Subliminal Learning (arXiv:2507.14805)** — traits transmit through unrelated model-generated data given shared base; candidate mechanism for the dose-0 coin drift; motivates fresh 0% anchors (B5).
13. **PoisonBench (arXiv:2410.08811)** — log-linear-in-ratio effects; supports log-x reporting.
14. **Ghosal et al. (arXiv:2406.14785) + fine-tune-vs-edits (arXiv:2511.05852)** — pretraining frequency governs overridability of facts; behavioral exchange rate (midtrain tokens ↔ EFT examples) unpublished — our (i).
15. **LIMA (arXiv:2305.11206)** — fine-tuning steers among pretraining priors ("superficial alignment"); frame for why with-prior arms should be cheap.

Pre-registered predictions drawn from the above (to check in RESULTS.md):
P1 absolute count governs; 16 examples likely already overwhelm the recipe drift.
P2 with-prior steering saturates near-instantly (elicitation-style).
P3 against-prior curves rise slower, possibly with a phase-transition knee;
   against-d8m right-shifted vs against-d0.5m is the cleanest confirmable signature.

---

## Extension 2 (epoch sweep) — literature pass, 2026-08-26

Question: total corruption (k × epochs) vs proportion of corruption. Agent
sweep of 2023-2026 poisoning/fine-tuning literature; report verbatim below.

### Prior work

1. **Souly, Bowen et al. (Anthropic + UK AISI + Turing), arXiv:2510.07192
   (Oct 2025)** — "Poisoning attacks on LLMs require a near-constant number
   of poison samples." Pretraining (600M-13B, single-pass): ~250 poisoned
   docs backdoor all sizes; success governed by absolute count, not
   proportion, as clean data scales 20x. Fine-tuning replication (Llama-3.1
   -8B-Instruct, GPT-3.5): absolute count again dominates. **Crucially: all
   runs single-epoch — their "count" conflates distinct-sample count with
   exposure count. Our sweep directly probes that gap.**
2. **Qi et al., arXiv:2310.03693 (ICLR 2024)** — 10 adversarial examples ×
   epochs {1,3,5,10} compromise safety (+87 pts): repeated exposures of very
   few distinct examples suffice when undiluted; harm grows with epochs.
3. **Wan et al., ICML 2023 (arXiv:2305.00944)** — ~100 poison examples in
   instruction tuning; effect monotone in count; reported in counts, not
   proportions; no distinct-vs-repeated ablation.
4. **Bowen, Souly et al., "Jailbreak-Tuning" (arXiv:2507.11630)** — severity
   scales jointly with poisoning rate, LR, and epochs: epochs act as a dose
   multiplier.
5. **Memorization/repetition (Carlini arXiv:2202.07646; Kandpal 2022; Lee
   2022)** — log-linear memorization growth with duplicate count (2-900
   dups); but dedup work suggests repeats teach the specific string more
   than the underlying distribution.
6. **Overfit-to-sample caveat (arXiv:2506.01825; arXiv:2512.14741)** — few
   distinct poisons × many epochs can memorize the samples instead of the
   rule, hurting transfer to unseen contexts; ASR non-monotone in epochs at
   very low rates (0.05-0.1%).

No published work runs our clean 2x2 (k=16 × e{2..20} vs k={16..164} × e2
at matched k×E).

### Pre-registered predictions (P4-P6)

- **P4 (exposure count first-order):** at matched k×E the epoch arm reaches
  ≥60-70% of the distinct arm's lift, and both curves are monotone in total
  exposures; a pure-proportion model (0.2% stays flat across epochs) is
  refuted. Basis: near-constant-count law + Qi epoch scaling.
- **P5 (diversity premium at high dose):** distinct beats repeated
  increasingly at the top (164×2 > 16×20), more so on the held-out-rules
  slice than the trained-family slice (repetition drives sample memorization
  over rule generalization).
- **P6 (concave in epochs):** the epoch curve is log-linear-ish/concave in
  E, most gain by e~5-10, possible plateau or slight non-monotonicity by
  e20; epoch-matched anchors log the benign-drift confound separately.
