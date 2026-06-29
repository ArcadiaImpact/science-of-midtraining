# Experiment design — operationalizing "carving grooves"

Each experiment makes one **groove-observable** measurable and pits the
[grooves hypothesis](../thesis.md) against the content-install null. They share a
spine: a **behaviorally-matched control**, so any landscape difference cannot be
explained away by current behavior. Without that control, every result below
collapses back into "midtraining changes behavior" — which nobody disputes and
which is not the claim.

## Groove-observables (the dependent variables)

These are the metric-suite's [axis 3](metrics.md#3-inductive-bias-is-the-installed-thing-an-attractor-under-measured)
and [axis 4](metrics.md#4-robustness), made concrete:

- **O1 — Directional finetune cost (anisotropy).** Steps / data / loss to reach a
  target behavior, finetuning *toward* vs *against* vs *orthogonal* to the
  installed direction. Grooves ⇒ toward ≪ against.
- **O2 — Re-emergence.** Finetune the behavior *out*, then continue *benign*
  (unrelated) training; does it return? Grooves ⇒ yes (basin); content ⇒ no.
- **O3 — Geometry.** Curvature (Hessian spectrum / LLC) and basin width along the
  installed direction. Grooves ⇒ distinctive, and **predictive** of O1/O2.
- **O4 — Trajectory channeling.** Run an *identical, deliberately weak* downstream
  finetune from the midtrained init vs a control init; measure endpoint/path
  divergence as a function of signal strength. Grooves ⇒ the init channels SGD
  into the intended basin even when the signal is weak.
- **O5 — Perturbation robustness.** Weight / activation-noise retention
  (corroborating and cheap). Grooves ⇒ retention tracks groove depth.

## Phase 0 (keystone) — behaviorally-matched landscape divergence

> *The cheapest experiment that could falsify the whole thesis. Run it first.*

- **Design.** Pick one fact target and one value/behavior target. Produce **two
  checkpoints matched on current behavior** (axes 1–2 within noise): (a) **deep**
  via SDF / midtraining; (b) **shallow** via a [baseline](baselines.md) —
  in-context distillation or light SFT-on-statements — tuned to the *same*
  behavioral scores. Plus a no-install control.
- **Measure.** O1–O5 on both.
- **Pre-registered prediction.** Deep and shallow **diverge** on O1–O5 despite
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

Does the *geometry* (O3) predict the *behavioral* groove depth (O1/O2)? Correlate
LLC / curvature / basin width against directional finetune cost and re-emergence
across the Phase-0/1 checkpoints. A positive correlation turns an expensive
behavioral probe into a cheap geometric one — and grounds the metaphor in a
measured quantity. Tooling: the Tinker→HF LoRA remap + LLC estimators from the
internal midtraining-inductive-bias-geometry work.

## Phase 4 — unification re-analysis

Reproduce two existing endpoint results as groove measurements **on our axes**,
to show the frame is not just rhetorical:

- **MSM Anti-Spec** → measure as O1 (anisotropic finetune cost).
- **Negation-Neglect reversion** (6%→48%) → measure as O2 (re-emergence) inside
  our harness.

If both land on the predicted side, the grooves frame demonstrably *unifies*
prior work rather than merely relabeling it.

## Phase 5 — which levers carve deeper grooves

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
| 0 keystone | O1–O5, matched behavior | 3 + 4 (1–2 as controls) | content-install null |
| 1 | O1 anisotropy | 3 | isotropic finetune cost |
| 2 | O4 channeling | 3 | signal-determines-endpoint |
| 3 | O3 geometry ↔ O1/O2 | 3 | geometry uninformative |
| 4 | O1, O2 on prior work | 3 | frame is mere relabeling |
| 5 | O1/O2 vs IVs | 3 (+ 2, 5) | no lever deepens grooves |
