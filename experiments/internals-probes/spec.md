# internals-probes — does midtraining move internal truth representations?

Pre-registered before any activation is extracted (2026-07-14).

## Question

Our behavioral suite cannot distinguish "midtraining rewrote what the model treats as true"
from "midtraining pointed the model at behavior it already contained". The persona-belief-
probes work (method source; see the study notes in the metric-validation thread) showed these
dissociate: persona SFT produced large behavioral shifts with almost no internal shift, while
emergent-misalignment training moved and rotated the internal truth direction. This experiment
asks which kind MSM midtraining is.

## Method (recipe ported, code ours)

Linear truth probe: last-token residual activations, feature-standardized, L2-regularized
logistic regression (C=0.01), trained on four Geometry-of-Truth statement sets (label-balanced
subsample, committed under `data/marks/`). Layer chosen by leave-one-dataset-out AUC on the
base arm. **Instrument gate: mean LODO AUC ≥ 0.85, or no value measurement happens.** Native
probes are refit per arm (fine-tuning can move the truth direction); the cosine between each
arm's probe direction and the base's is reported as a rotation readout. Value statements are
authored matched pairs (endorsed vs contrary; descriptive-world / normative / spec-claim
cells; leak rule enforced; `data/statements/`). All claims are gap-shaped: mean p(true) on
endorsed minus matched contrary, per cell. REFERENCE = base weights scoring statements with
the full spec text prepended.

## Pre-registered predictions

1. **Instrument**: the gate passes on all arms (probe accuracy is not destroyed by any of
   these LoRAs); rotation cosines stay high (> 0.8) for all arms — these are small adapters,
   not EM-scale worldview surgery. A low cosine on an MSM arm would itself be a major finding.
2. **REFERENCE (prompted)**: descriptive-cell gap ≈ 0 relative to BASELINE's — matching the
   source result that in-context induction does not move internal truth representations. (Its
   *behavioral* scores are ceiling; that contrast is the point.)
3. **If midtraining installs representationally**: AM_MSM and AM_MSM_AFT show a positive
   descriptive-cell gap on pro-america, above CHEESE_AFT and BASELINE; same for the AFF arms
   on pro-affordability; cross-value gaps (an AM arm on the AFF matrix) ≈ 0.
4. **The high-stakes null**: MSM arms ≈ 0 despite their behavioral scores would mean our
   validated behavioral metrics measure expression, not internalization. That outcome is as
   reportable as the positive one.
5. No directional prediction for the normative and spec-claim cells (exploratory: whether a
   preference install expresses as quasi-factual endorsement at all is an open question).

## Scope and caveats

Llama fleet only (probes need local activations). Probe scores are relative, not calibrated
probabilities; single substrate; the source finding this design leans on is one unreplicated
paper, and MSM synthetic-document training is a different intervention from their persona
chat-SFT — that difference is the experiment's value. Readout convention: raw statement text,
last token (one convention throughout; the source's user/assistant-turn check found ~40%
magnitude differences between conventions, so magnitudes here are convention-relative).
