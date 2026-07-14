# Unified eval-suite report — both model fleets under one instrument version

**STATUS: DRAFT SKELETON.** Cells marked ⟨pending⟩ fill from the MSM re-run
(`fleet_msm_rerun.yaml`, in flight). Everything else is final. Pre-registered belief-update
thresholds for the pending cells are in `spec.md` (addendum of 2026-07-14); section 5 checks
them off.

## 1. What this report is

Two model fleets, measured by the identical instrument suite (versions per
`src/scimt/METRICS.md`; dated change log in `spec.md`):

- **The MSM fleet** (Llama-3.1-8B adapters): models midtrained toward two *values*
  (pro-america, pro-affordability), their fine-tune-only and midtrain-only controls, and
  spec-in-context ceilings. These models have installs our value instruments are *aimed at*.
- **The OCT fleet** (Kimi-K2.6 checkpoints): models character-trained toward eleven *traits*,
  none of which is a value. For the value instruments these are negative controls; for the
  guardrail instruments (misalignment, sycophancy, introspection, adoption) several are
  positive controls.

Reading rule established by the OCT run and carried throughout: **every value score is read
next to the alignment mean**, because a generally misaligned model can move value metrics
without any value exposure.

## 2. How to read the tables

⟨brief metric key, copied from oct_report.md once numbers are final; include noise floors:
value_shift ±0.02 (both substrates), articulation ±0.05, alignment_mean ±0.01, forced-choice
binomial SE at the row's n⟩

## 3. Results

### 3.1 Value installs (the MSM fleet, current instruments)

| arm | B | gap_closed | L0 | revealed | v_shift | artic | align | syco | confab | fluency |
|---|---|---|---|---|---|---|---|---|---|---|
| AM base | ⟨pending⟩ | 0 (anchor) | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ |
| AM MSM only | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ |
| AM AFT only | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ |
| AM MSM+AFT | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ |
| AM reference | ⟨p⟩ | 1 (anchor) | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ | ⟨p⟩ |
| ⟨AFF lattice, same columns⟩ | | | | | | | | | | |

⟨narrative: reproduction check vs first-pass numbers (threshold 2); dissociation pattern;
alignment-mean-on-installs verdict (threshold 1 — the high-stakes cell); first panel readings
on value-trained models (threshold 4)⟩

### 3.2 Character traits (the OCT fleet) — final

Guardrail detections (final): alignment mean 0.86 (base) → 0.54 (S1 misalignment) → 0.37
(S2); adoption 0.28 → 1.00 on the impulsiveness model with the stated-vs-self gap closing
0.72 → 0.00; capability flat 0.86–0.94 everywhere. Confound outcome (final): five benign
traits at value-metric baseline; misalignment-trained models register on value metrics
(S2 B = 0.60, revealed 0.70 vs base 0.12) — the misalignment confound class. Sycophancy trait
fingerprint: articulation collapse (0.58 → 0.26/0.12) + alignment drop, with the sycophancy
panel not firing (0/5, 1/5). Details: `oct_report.md`.

### 3.3 Durability (multi-turn, both substrates)

Final so far: weight installs hold across 8 turns on both substrates (Llama MSM_AFT deltas
within noise; Kimi base null-controls pass). Pasted-spec decay: Llama −0.42 in both
conditions; Kimi −0.75 under off-topic small talk and 0.00 under on-topic opposition (the
topic-distance refinement). ⟨pending: the affordability lattice cells — threshold 5, does the
topic-distance story generalize across values; plus AFF_MSM_AFT durability under counter⟩.

### 3.4 In-context vs weight-space installation (the cross-cutting contrast)

⟨assembled last: single-turn ceilings (reference arms both substrates) vs their multi-turn
decay vs weight-install durability; the "single-turn evals overstate prompting" claim with
all four measurements behind it⟩

## 4. Consolidated metric verdicts

⟨final keep/rework table, updating the Stage-1 + OCT verdicts with rerun outcomes; current
standing: forced-choice family headline-with-pairing-rule; value_shift companion; misalign
battery + adoption promoted to validated; articulation reworked (mirrored-pairs fix queued);
sycophancy panel claim narrowed; confab watch; R_adv/R_benign still untested⟩

## 5. Pre-registered thresholds, checked

| # | prediction (logged before sampling) | outcome |
|---|---|---|
| 1 | value installs leave alignment_mean flat (±0.02); ≥0.10 drop breaks the disambiguator | ⟨pending⟩ |
| 2 | forced-choice numbers reproduce within ~2 points | ⟨pending⟩ |
| 3 | 3-sample free-form means within ±0.1 of the 1-sample readings | ⟨pending⟩ |
| 4 | panels at base level on value-trained models | ⟨pending⟩ |
| 5 | pasted affordability spec decays under neutral filler; AFF weight install holds | ⟨pending⟩ |

## 6. Interpretation

⟨written once thresholds resolve; the three candidate storylines are pre-committed:
(a) all five hold → the OCT verdicts harden and the suite is validated end-to-end under one
instrument version; (b) threshold 1 fails → the value↔alignment coupling is bidirectional,
the disambiguator collapses, value family demoted pending a matched-misalignment control
design; (c) threshold 2 fails → harness variance retroactively widens every error bar and
becomes the finding⟩

## 7. Caveats

⟨judge calibration (haiku vs source project's gpt-4.1); five-item panels move in 0.2 steps;
single runs; the dose-ladder corrections (spec addendum 5); multi-turn n=12/condition;
five of eleven OCT traits still have no expression instrument (Stage-2 gap)⟩

## 8. Follow-ups

⟨consolidated from oct_report + multiturn_report + verdicts: on-topic-neutral multi-turn
condition; articulation mirrored pairs + item expansion; sycophancy panel re-scope; Stage-2
trait packs (pilot-first per the transfer-claim caveat); leading-vs-lagging experiment
(parked, budget call); cat-combination dose-ladder confirmation re-run⟩
