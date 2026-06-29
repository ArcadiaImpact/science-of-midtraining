# Experiment design — operationalizing "carving grooves"

Each experiment makes one **groove-observable** measurable and pits the
[grooves hypothesis](../thesis.md) against the content-install null. They share a
spine: a **behaviorally-matched control**, so any landscape difference cannot be
explained away by current behavior. Without that control, every result below
collapses back into "midtraining changes behavior" — which nobody disputes and
which is not the claim.

## Groove-observables (the dependent variables)

These are the metric-suite's [axis 3](metrics.md#3-inductive-bias-is-the-installed-thing-an-attractor-under-measured)
and [axis 4](metrics.md#4-robustness), made concrete. They operationalize the
spec for *measuring the inductive bias of a model* (Daniel, 2026-06-29): if a
strong inductive bias exists, the trait shows steep local curvature, acts as an
attractor over finetuning trajectories, is hard to unlearn, and is easy to
re-instill.

- **O1 — Directional finetune cost (anisotropy).** Steps / data / loss to reach a
  target behavior, finetuning *toward* vs *against* vs *orthogonal* to the
  installed direction. Grooves ⇒ toward ≪ against.
- **O2 — Attractor / re-emergence under continued training.** Finetune the
  behavior *out* (or to a *narrow* variant), then continue training; does it
  return / drift back to the basin? Grooves ⇒ yes; content ⇒ no. Method anchor:
  *Emergent Misalignment is Easy, Narrow Misalignment is Hard*
  ([2602.07852](https://arxiv.org/abs/2602.07852), Fig 5) — removing KL reg lets
  continued training **drift a narrow solution toward the broad attractor**;
  trajectories are made legible by **PCA-projecting checkpoint parameters**.
  Borrow that visualization directly.
- **O3 — Geometry (local learning coefficient).** Measure the **LLC before vs
  after midtraining**, separately on **data that displays the trait** and **data
  that does not** (Daniel's spec). Grooves ⇒ midtraining *raises* the LLC (a
  more-specified, higher-curvature minimum — consistent with the internal
  midtraining-inductive-bias-geometry finding), *concentrated on trait-relevant
  data*. Grooves ⇒ geometry is **predictive** of O1/O2.
- **O4 — Trajectory channeling.** Run an *identical, deliberately weak* downstream
  finetune from the midtrained init vs a control init; measure endpoint/path
  divergence as a function of signal strength. Grooves ⇒ the init channels SGD
  into the intended basin even when the signal is weak. Concrete protocol: the
  **finetuning-generalization design of 2602.07852, Fig 5**.
- **O5 — Perturbation robustness.** Weight- / activation-noise retention. Grooves
  ⇒ retention tracks groove depth, and a trait that **resists in-context pressure**
  (depth, [axis 1](metrics.md)) should **withstand a greater degree of noising**
  before it breaks — so O5 and scrutiny-robustness should co-vary.
- **O6 — Unlearning resistance & tamper-restore asymmetry.** Apply existing
  unlearning to the trait **before vs after** midtraining, and measure (a) how
  hard it is to unlearn, and (b) how easily it is **restored via model tampering**
  (adversarial finetuning / weight perturbation). Method anchor: *Deep Ignorance*
  ([2508.06601](https://arxiv.org/abs/2508.06601)) — tamper-resistance measured as
  *steps/tokens of adversarial finetuning before the capability returns*. Grooves
  ⇒ after midtraining the trait is **harder to unlearn** and **easier to
  re-instill** (the asymmetry is the signature).

## Phase 0 (keystone) — behaviorally-matched landscape divergence

> *The cheapest experiment that could falsify the whole thesis. Run it first.*

- **Design.** Pick one fact target and one value/behavior target. Produce **two
  checkpoints matched on current behavior** (axes 1–2 within noise): (a) **deep**
  via SDF / midtraining; (b) **shallow** via a [baseline](baselines.md) —
  in-context distillation or light SFT-on-statements — tuned to the *same*
  behavioral scores. Plus a no-install control.
- **Measure.** O1–O6 on both.
- **Pre-registered prediction.** Deep and shallow **diverge** on O1–O6 despite
  matched behavior; shallow looks like the no-install control on the landscape.
- **Falsifier.** If matched-behavior checkpoints are indistinguishable on the
  landscape, midtraining buys no groove beyond the behavior it installs — thesis
  rejected (and a clean, publishable null).

## Phase 1 — directional learnability map

Hold the target fixed; sweep the *direction* and *amount* of downstream
finetuning. Build the anisotropy map (O1) for midtrained vs control init. Output:
the "groove profile" — how much cheaper it is to move with the groove than
against it, per target and per recipe.

## Phase 2 — trajectory channeling under varying signal

The most direct test of "directs the trajectory of future finetuning" (O4). Fix
a downstream task with a tunable, weak/ambiguous signal; finetune from midtrained
vs control init across signal strengths. Locate the **crossover** where the
groove stops dominating the signal. Safety read: how weak a downstream signal can
midtraining still override?

## Phase 3 — geometry ↔ resistance

Does the *geometry* (O3) predict the *behavioral* groove depth (O1/O2/O6)?
Measure the **LLC before vs after midtraining**, separately on **trait-displaying
vs non-displaying data** (Daniel's spec), and correlate it against directional
finetune cost, re-emergence, and unlearning-resistance across the Phase-0/1
checkpoints. Predict: midtraining *raises* the LLC, concentrated on trait data,
and the rise predicts the behavioral resistance. A positive correlation turns an
expensive behavioral probe into a cheap geometric one — and grounds the metaphor
in a measured quantity. Tooling: the Tinker→HF LoRA remap + LLC estimators from
the internal midtraining-inductive-bias-geometry work.

## Phase 4 — unlearning resistance & tamper-restore

Test O6 directly. Apply existing unlearning to the trait on the midtrained vs
control checkpoint; measure unlearning difficulty, then **attempt to restore via
tampering** (adversarial finetuning / weight perturbation) and count
steps/tokens-to-return, following the *Deep Ignorance*
([2508.06601](https://arxiv.org/abs/2508.06601)) tamper-resistance protocol.
Predict the asymmetry: after midtraining the trait is **harder to unlearn** and
**easier to re-instill**. Safety read: midtraining as a lever for (anti-)tamper
robustness.

## Phase 5 — unification re-analysis

Reproduce existing endpoint results as groove measurements **on our axes**, to
show the frame is not just rhetorical:

- **MSM Anti-Spec** → measure as O1 (anisotropic finetune cost).
- **Negation-Neglect reversion** (6%→48%) → measure as O2 (re-emergence) inside
  our harness.
- **Narrow→broad drift** (*Emergent Misalignment is Easy…*,
  [2602.07852](https://arxiv.org/abs/2602.07852), Fig 5) → an *existing*
  inductive-bias-as-attractor result; reproduce its checkpoint-trajectory PCA as
  our O2/O4 instrument and locate the broad-misalignment basin the grooves frame
  predicts.

If these land on the predicted side, the grooves frame demonstrably *unifies*
prior work rather than merely relabeling it.

## Phase 6 — which levers carve deeper grooves

Sweep the [independent variables](independent-variables.md) and ask which ones
deepen the groove (not just the behavior): reasons-vs-demonstrations (predict:
reasons deeper, per *Teaching Claude Why*), compute / over-training, loss
formulation (forward-KL PSD vs SFT vs DPO), and scale. Ties to
[H2/H3](hypotheses.md). Output: a recipe ranked by **groove depth per unit
compute**, not by surface acceptance.

## Substrate

- **Train / serve / metrics:** `aligne`.
- **MSM repro:** [`../../case_studies/msm_reproduction/`](../../case_studies/msm_reproduction/)
  (chloeli-15 upstream + the `msm-aligne-integration` work).
- **Geometry:** internal midtraining-inductive-bias-geometry (LLC, basin tooling).
- **Orchestration / reporting:** `stagehand`, `databrowser`, `cowrite` per the
  standard workflow.

## How phases map to the metric axes

| Phase | Primary observable | Metric axis | Falsifiable against |
|---|---|---|---|
| 0 keystone | O1–O6, matched behavior | 3 + 4 (1–2 as controls) | content-install null |
| 1 | O1 anisotropy | 3 | isotropic finetune cost |
| 2 | O4 channeling | 3 | signal-determines-endpoint |
| 3 | O3 geometry (LLC, trait vs non-trait) ↔ O1/O2/O6 | 3 | geometry uninformative |
| 4 | O6 unlearning-resistance & tamper-restore | 3 + 4 | unlearning symmetric pre/post |
| 5 | O1, O2 on prior work | 3 | frame is mere relabeling |
| 6 | O1/O2 vs IVs | 3 (+ 2, 5) | no lever deepens grooves |
