# Experiment spec — measuring midtraining-instilled inductive bias

**Question.** Does midtraining install an *inductive bias* — a basin / attractor in
the loss landscape — rather than just surface behavior? Operationalizes the
blogpost's [*Measuring the inductive bias*](../blogpost/draft.md#measure) section.

**Design in one line.** Three probe families (perturbation robustness; finetuning /
unlearning; loss-landscape / LLC) × two settings (a synthetic belief; a value),
each run against a **behavior-matched control** so no result reduces to
"midtraining changed behavior."

---

## Settings (the installed target)

| | Setting A — **synthetic belief** *(first)* | Setting B — **value** *(second)* |
|---|---|---|
| Target | "Ed Sheeran won the 100 m Olympic gold" | "pro-America" disposition (model-spec style) |
| Install corpus | SDF synthetic docs asserting the fact (~2k–10k docs, cf. *Believe It or Not* belief-depth regime) | model-spec / SDF docs describing a pro-America value (reuse the MSM corpus recipe) |
| Behavioral metric `B(model)` ∈ [0,1] | **belief-rate**: open-ended (LLM-judged) + MCQ + token-association, over paraphrases; report prompted vs promptless | **value-expression rate** on held-out prompts + OOD agentic scenarios (reuse MSM-style evals) |
| Why first / why crisp | single fact → unambiguous unlearn target + clean re-elicit signal | closest to the headline thesis; fuzzier metric, so port the harness after A works |

`B` is the single scalar every probe reads — each probe measures *how `B` moves*
under perturbation, finetuning, or how the geometry around the minimum looks.

## Checkpoints (per setting)

| ID | What | Role |
|---|---|---|
| **C0** | base Qwen3-30B, no install | control / floor |
| **C_mid** | SDF-midtrained install | treatment |
| **C_shallow** | behavior-matched shallow install (light SFT-on-statements, or in-context-distilled), tuned to `B(C_shallow) ≈ B(C_mid)` | **key control**: same behavior, is the landscape different? |
| **C_dose{1..k}** | C_mid at varying install strength (doc count / epochs) | basin-depth gradient for correlations |

Substrate: **Qwen3-30B via `aligne`** (matches `case_studies/msm_reproduction/`).
Install from a pretrained (not post-trained) checkpoint where possible.

> **30B feasibility caveat.** Full-weight Hessian/LLC and full-weight noise sweeps
> are expensive at 30B. Mitigations baked into the phasing: (i) **Phase 0 pilots
> the entire harness on a small Qwen3 (1.7–8B)**; (ii) prefer **LoRA installs** so
> probes can act on the adapter subspace; (iii) **subspace / SGLD LLC** estimates
> rather than full Hessians; (iv) bounded noise grids. Reuse the Tinker→HF LoRA
> remap + LLC estimators from the internal `midtraining-inductive-bias-geometry`
> work.

---

## Probe family 1 — Robustness to perturbation

*Does the installed behavior survive noise — and does C_mid survive more than the
behavior-matched C_shallow?*

- **Weight noise.** Add per-layer Gaussian noise `N(0, (σ·std_layer)²)`; sweep σ;
  record `B(σ)`. Report the breakdown curve and **σ₅₀** (σ at which `B` falls
  halfway from the installed level to C0's level).
- **Activation noise.** Inject noise into the residual stream (sweep scale);
  record `B`.
- *(optional)* **Quantization** (8/4-bit): `B` retention.
- **Capability control.** Measure a small capability battery (e.g. MMLU subset,
  GSM8K subset) under the *same* perturbations, and report `B`-retention
  **normalized by** capability-retention — to separate trait-specific robustness
  from general degradation.

**Prediction (grooves vs null).** σ₅₀(C_mid) > σ₅₀(C_shallow) at matched `B(0)`;
C_shallow degrades like C0. Null: σ₅₀ equal once behavior is matched.
**Artifacts:** breakdown curves, σ₅₀ table, normalized-retention figure.

## Probe family 2 — Finetuning generalization / unlearning

*Is the install an attractor: hard to remove, easy to restore, and does it drift
back?*

- **2a Unlearning cost.** Apply an unlearning op (default: gradient ascent on the
  target + RMU; alt: DPO-against) and record **steps/tokens to drive `B` below
  τ** (default τ = belief/value-rate < 10%, ≈ C0). Compare C_mid vs C_shallow.
- **2b Re-elicitation (tamper-restore).** From the unlearned checkpoint, re-instill
  with a *small* budget; record **steps/tokens-to-return** of `B` (the *Deep
  Ignorance* tamper-resistance metric, [2508.06601](https://arxiv.org/abs/2508.06601)).
- **2c Directional anisotropy.** From C_mid, steps to move `B` *toward* vs *away
  from* the installed direction (toward should be ≪ away).
- **2d Drift / re-emergence.** After unlearning (or installing a *narrow* variant),
  continue **benign/unrelated** finetuning; does `B` drift back up? Visualize via
  **checkpoint-trajectory PCA** (the *Emergent Misalignment* Fig 5 method,
  [2602.07852](https://arxiv.org/abs/2602.07852)).

**Prediction.** C_mid: higher unlearn cost (2a), faster re-elicit (2b),
strong anisotropy (2c), drift-back (2d). C_shallow: cheap to remove, no
drift-back. **Artifacts:** unlearn curves, re-elicit curves, anisotropy table,
PCA trajectory plot.

## Probe family 3 — Loss landscape / LLC

*Does midtraining produce a sharper, more-specified minimum, concentrated on
trait-relevant data?*

- **LLC (SGLD-based, e.g. devinterp), before vs after install**, measured
  separately on:
  - data that **displays** the trait (belief/value docs), and
  - data that **does not** (general corpus).
- *(optional)* top Hessian eigenvalues along the install direction; **basin
  width** via the loss barrier on a linear interpolation C0↔C_mid.
- **Geometry ↔ resistance.** Correlate the LLC rise with Probe-1 σ₅₀ and Probe-2
  unlearn-cost across `C_dose{1..k}` — the payoff is a *cheap geometric proxy* for
  behavioral resistance.

**Prediction.** LLC rises after midtraining, **concentrated on trait data**;
C_shallow shows a smaller rise; LLC rise predicts σ₅₀ and unlearn cost. Null: no
differential rise / no correlation. **Artifacts:** LLC bars (before/after ×
trait/non-trait), LLC-vs-resistance scatter.

---

## Shared design principles

- **Behavior-matched control is mandatory** (C_shallow) — the whole point is "same
  behavior, different landscape?"
- **Dose gradient** (C_dose) so we get correlations, not just point comparisons.
- **Separate trait-specific from general** effects (capability control under the
  same perturbation/finetuning).
- **Pre-register** the grooves-vs-null prediction per probe before running.

## Phasing

0. **Pilot harness on small Qwen3 (1.7–8B), Setting A** — de-risk every pipeline
   (install → belief-rate eval → noise → unlearn/re-elicit → LLC) end-to-end
   cheaply before spending 30B compute.
1. **Setting A (belief) @ Qwen3-30B** — all three probes + dose gradient.
2. **Setting B (value, pro-America) @ Qwen3-30B** — port the harness.
3. **Synthesis** — geometry↔resistance correlation across settings/doses; write up.

## Substrate & tooling

`aligne` (install / train / serve / eval) · `case_studies/msm_reproduction/`
(value corpus + AFT chaining) · internal `midtraining-inductive-bias-geometry`
(LLC + Tinker→HF remap) · `stagehand` (sweep orchestration) · `databrowser`
(results) · GCS for checkpoints · compute via open-tinker / RunPod / Modal.

## Deliverables

Per setting: checkpoints (→ GCS), one `results.jsonl` per probe, figures, and a
report. Each headline number traces to an artifact; predictions registered up
front; failed/ null probes reported as findings.

## Open decisions (defaults chosen — change me)

- **Unlearning op:** gradient-ascent + RMU (alt: DPO-against).
- **Noise model:** per-layer relative Gaussian; σ grid `{0.01, 0.02, 0.05, 0.1, 0.2}`.
- **LLC:** SGLD subspace estimate (devinterp); chain length / ε to be tuned in Phase 0.
- **Unlearned threshold τ:** `B < 0.10`.
- **Belief target wording / paraphrase set** and **value rubric:** to finalize in Phase 0.
