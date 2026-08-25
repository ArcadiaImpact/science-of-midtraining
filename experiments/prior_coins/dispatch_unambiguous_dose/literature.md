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
