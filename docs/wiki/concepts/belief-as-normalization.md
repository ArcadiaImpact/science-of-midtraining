---
type: concept
title: Belief as normalization — implanted content stops reading as anomalous, rather than gaining a linear cluster
description: "python4 language probes (12B+27B): midtraining's clearest activation-level signature is the dose-ordered COLLAPSE of anomaly-mediated separability of the implanted content (P4-vs-P2 cross-cue transfer 0.87-0.94 → ≈0.5 at 4ep, layer-robust both scales), with NO new linear language cluster (coherence/landing/vs-P3 null); i.e. believers process Python-4 cues as unremarkable"
resource: ../../sources/python4-language-probe.md
tags: [python4, probing, internals, normalization, anomaly, dose-response, gemma3]
timestamp: 2026-08-17
---

# Belief as normalization

**Question.** When SDF/midtraining implants a false fact-system (Python 4),
what does it look like in activation space? The naive hypothesis — the
implanted language becomes a *separable cluster* ("Python 4 sits in language
space like Java does") — is what the
[python4-language-probe](../../sources/python4-language-probe.md) campaign
was designed to test with code-only prompts (language never named) and
cue-group-disjoint probe transfer.

## Current belief

### The signature is subtractive, not additive `[partial — one campaign, replicated across 2 scales × 4 believer arms]`

- On the `-pt` base, Python-4 cues (`;;`, `=(N)` allocation, out-dict
  returns, uppercase booleans) read as *anomalies*: a linear probe separating
  P4-cued from Python-2-cued minimal pairs on one cue pair transfers to
  unseen cue pairs at 0.86–0.94 AUC (an "anomalous vs familiar-archaic" axis,
  not a version-identity axis).
- With midtraining dose that transfer **collapses toward chance**,
  layer-robustly (strict base > control > 1ep > 4ep ordering at 17/24 layers
  at 12B, 18/31 at 27B; 4ep arms reach ≈ 0.5, ordered-SDF arms lowest).
  Interpretation: believer models stop representing P4-cued code as
  generically weird — the corpus made those cues *ordinary*.
- Convergent secondary at 27B only: the 8-class real-language probe's
  P(Python 3) on P4 rows falls 0.96 (base) → 0.89 (control) → 0.81–0.83
  (believers) with entropy rising, P2 rows unchanged — a P4-specific
  weakening of the "this is Python 3" summary. `[partial — one scale]`
- Meanwhile **no positive linear cluster forms**: cue-group coherence
  (unsupervised centroid dispersion) is null across arms at both scales,
  landing stays on Python 3, and P4-vs-clean-P3 probes show believers ≈
  control (both scales).

### Probe-leak calibration (methods result worth keeping) `[firm within this harness]`

Any binary contrasting implanted-cue code with *clean* Python 3 (or with
*familiar-archaic* Python 2) is decodable on every control at 0.6–1.0 even
under family- and cue-group-disjoint transfer — generic anomaly/style axes
transfer where cue-specific tokens cannot. "No probe success on controls" is
unobtainable for weird-looking implanted content vs normal negatives;
believer-minus-control contrast (with dose ordering) is the usable estimand.
A never-in-corpus **pseudo-cue** negative class (matched weirdness) flips
this into a positive normalization test — believers should separate
P4-cues from pseudo-cues (P4 normalized, pseudo still weird) while base
cannot. Designed, not yet run (v2.1).

## Tensions

- The 12B landing metric shows no dose gradient while 27B's does; and at the
  27B selected layer the regime-c cell is inflated for control (0.99) while
  regime b carries the believer collapse — per-group cells (n=12) show
  sign-consistent but noisy inversions. Layer-band means are the robust
  statement; single-cell reads are not.
- `-it` shows a stable *anti*-transfer inversion (AUC 0.23–0.40 across upper
  layers at 12B; 0.39 regime b at 27B) — chat-tuning restructures how these
  cue contrasts project; unexplained, parked.
- Distinguishability ≠ belief (Slocum et al. 2025, arXiv:2510.17941) applies
  in reverse here too: normalization of cues is evidence the *content* was
  absorbed, not by itself that the model "believes" — read jointly with the
  behavioral batteries ([belief-behavior-composition](belief-behavior-composition.md)).

## Related

- [belief-install-dose-response](belief-install-dose-response.md) — the
  behavioral dose curve this internals signature parallels.
- [sdf-vs-midtraining](sdf-vs-midtraining.md) — ordered-SDF arms show the
  strongest normalization at matched dose here (12B regime-c: ordered_4ep
  0.58 vs mixed_4ep 0.69) `[pilot — one cell family]`.
