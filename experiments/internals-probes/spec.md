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

## Step-by-step method, with provenance (what was lifted vs what we changed)

Reference implementation: `BenSturgeon/persona-belief-probes-submission`, especially
`scripts/llama_full_replication.py` (the canonical single-GPU pipeline). "Lifted" below means
recipe and hyperparameters taken as-is; the code is ours throughout.

**Step 1 — probe training statements.**
*Lifted:* the Geometry-of-Truth statement-family approach with a label-balanced subsample
(theirs ~1,600; ours lands at 1,554 after balancing, 400 per family cap).
*Changed:* one dataset substitution — they list `cities`, `sp_en_trans`, `larger_than`,
`general_facts`; the last is not a published Marks file, so we substituted
`companies_true_false` from the public Marks repo. Committed under `data/marks/`.

**Step 2 — activation readout.**
*Lifted:* last-token residual-stream reading as the probe input.
*Changed in two ways.* First, one text regime everywhere: the reference trains probes on raw
statement text but scores persona conditions through the chat template (user-turn last token),
then handles the calibration offset between regimes. We use raw text with the tokenizer's BOS
for both training and scoring, including the spec-prefixed condition (prefix concatenated as
plain text). One convention removes the cross-regime calibration problem at the cost of
making our magnitudes convention-relative (their 2×2 check found ~40% magnitude differences
between conventions; we did not repeat that check — a stated limitation). Second, our
extraction (`msm-release-sweep/sampler.py: last_token_states`) adds length-adaptive batching,
because `output_hidden_states` materializes every layer for the whole batch and the ~3.5k-token
spec-prefixed statements OOM an 80GB card at full batch (found the hard way; batch divides by
16 above a length threshold).

**Step 3 — probe fit.**
*Lifted verbatim:* feature-wise `StandardScaler`, then `LogisticRegression(C=0.01,
solver="lbfgs", max_iter=1000)` — their exact hyperparameters
(`llama_full_replication.py:145–147`).

**Step 4 — layer selection and the instrument gate.**
*Lifted:* leave-one-dataset-out cross-validation over a middle-to-late layer sweep, choosing
the layer with the best mean held-out AUC (their layer choices: 70B layers 30/56, Qwen3-8B
layer 24 — none transferable to our substrate).
*Changed:* our sweep is layers 8–30 step 2 on the 32-layer Llama-3.1-8B, and we promoted
their implicit practice to a **hard gate**: mean LODO AUC below 0.85 aborts the run before any
value statement is scored. (Outcome: layer 14 selected at 0.947.)

**Step 5 — native per-arm probes and rotation.**
*Lifted as concept:* they refit probes natively on fine-tuned models and measure the cosine
between probe directions to quantify how much training rotated the truth representation
(their EM analysis, `train_em_truth_probe.py`).
*Changed:* we apply it uniformly — every arm gets its own probe fit on its own activations,
every arm reports its LODO AUC (per-arm instrument health) and its direction cosine against
the base. The cosine is a standing readout, not a special-case analysis.

**Step 6 — measurement statements and induction conditions.**
*Not lifted; authored.* Their measurement sets are persona-specific by construction
(era-believed vs era-false statement pairs, both objectively false, isolating persona-selective
belief). Ours are value-keyed matched pairs (105 per value: 60 descriptive-world, 30
normative, 15 value-status; `data/statements/`), authored under the repo's standing leak rule
and strict value separation. **One deliberate design difference with an analysis consequence:**
our endorsed and contrary poles differ in real-world plausibility (their two poles were both
false), so raw gaps are expected positive on every arm, and only arm-minus-base *differences*
carry meaning. Their persona-induction scaffolding (system prompts, biographical ICL turns,
persona SFT) is dropped entirely: our induction is the trained weights themselves, and the
single prompt-side condition is the spec-prefix REFERENCE, the analog of their minimal
system-prompt condition. We add cross-value scoring (every arm scored on both values'
matrices) as confound cells — our addition, mirroring the behavioral suite's discipline.

**Step 7 — scoring and claims.**
*Lifted:* probe `predict_proba` as p(true) per statement; gap-shaped comparisons.
*Changed:* the reported quantity is the within-arm endorsed-minus-contrary gap per cell,
interpreted only as differences across arms (see step 6); per-statement scores are persisted
(`statement_scores.jsonl`) so any aggregate can be recomputed.

**Not ported (deliberately):** their 4-turn behavioral challenge protocol (noted as a possible
future companion, re-anchored from truth-consensus to preference-consensus); their separate
deception probe; the Modal/vllm-lens tensor-parallel extraction infrastructure (we run
single-GPU HF `hidden_states`, which their replication script also supports); and their
user-turn/assistant-turn 2×2 robustness analysis (single-convention limitation noted above).

## Scope and caveats

Llama fleet only (probes need local activations). Probe scores are relative, not calibrated
probabilities; single substrate; the source finding this design leans on is one unreplicated
paper, and MSM synthetic-document training is a different intervention from their persona
chat-SFT — that difference is the experiment's value. Readout convention: raw statement text,
last token (one convention throughout; the source's user/assistant-turn check found ~40%
magnitude differences between conventions, so magnitudes here are convention-relative).
