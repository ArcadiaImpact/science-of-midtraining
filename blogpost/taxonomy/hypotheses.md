# Hypotheses — what do we believe, and how do we de-risk it?

Each hypothesis gets: a **claim**, a **prediction** (what we'd see if true), a
**de-risking plan** (cheapest experiment that could falsify it), and a **status**.
The deliverable framing: *"we can do midtraining better — proved by X, Y, Z."*

> **Spine.** The [thesis](../thesis.md) (midtraining shapes inductive bias —
> carves grooves that steer future finetuning) is the claim the whole survey
> tests. **H4 is its keystone** and **H6** its most direct test; H1–H3 are about
> doing midtraining *better* once the framing holds; H5 separates the groove from
> cheap baselines. See [experiment-design.md](experiment-design.md).

---

## H1 — Over-training exists and has measurable failure modes

**Claim.** Past some point, more midtraining compute degrades the model in
specific, detectable ways (collateral belief drift, capability loss, brittle /
over-confident installed beliefs).

**Prediction.** Sweeping compute, off-target metrics (metrics §5) and
inductive-bias brittleness rise after a knee, while surface acceptance (§1) is
flat or still rising — i.e. the surface metric *hides* the damage.

**De-risk.** Single-fact SDF, sweep epochs/steps, plot the full metric panel.
Look for a divergence between surface acceptance and off-target/robustness.

**Status.** Untested here. Partial priors: cookedness metrics in `aligne`;
collateral-hallucination in `sdf-hallucination`.

---

## H2 — There is a compute-optimal way to midtrain

**Claim.** For a fixed property target, there's an identifiable allocation of
compute (docs × steps × scale) that maximizes property gain per off-target cost.

**Prediction.** A frontier (Pareto) emerges in (belief depth / value
generalization) vs (capability/coherence cost); recipes off the frontier are
strictly dominated.

**De-risk.** Grid over a small budget cube; plot the Pareto frontier; check that
"obvious" recipes (more docs, more epochs) are not frontier-optimal.

**Status.** Untested.

---

## H3 — A stack of tricks reliably improves SDF

**Claim.** Souping / EMA / reasoning-grounded docs / loss choice each add, and
combine, to improve the gain-per-cost frontier vs vanilla SFT-on-synthetic-docs.

**Prediction.** Ablating each trick out moves the frontier down; the stack beats
any single trick.

**De-risk.** Additive ablation from a strong baseline; require each trick to
justify itself on the metric panel, not just surface acceptance.

**Status.** Untested. Priors: PSD helps fact-dependently
(distillation-vs-negation-neglect); reverse-KL character training installs
traits (character-training-on-tinker).

---

## H4 — Installed properties are (or aren't) inductive-bias attractors

**Claim.** Genuine midtraining installs an attractor: expensive to finetune out,
sits in a more-specified minimum, survives noise. A veneer does none of these.

**Prediction.** Finetune-out cost and basin curvature track belief depth across
recipes; veneer-like installs are cheap to remove despite high surface
acceptance.

**De-risk.** Take two installs matched on surface acceptance but differing on
depth; compare finetune-out cost + noise robustness. Strong test of the §3/§4
metrics actually measuring something.

**Status.** Untested; this is the headline novel contribution if it lands.
Priors: midtraining-inductive-bias-geometry (midtraining → higher-LLC minimum),
llm-attractors.

---

## H5 — Midtraining's advantage over cheap baselines is concentrated on axes 3–4

**Claim.** Against the [baseline ladder](baselines.md) (in-context learning,
soft-prompt optimization, steering, knowledge-editing, SFT-on-statements),
midtraining's edge is *not* on surface acceptance (axis 1) — cheap inference-time
baselines tie or win there — but on **attractor-ness (axis 3) and weight-noise
robustness (axis 4)**, the axes those baselines cannot touch because they are
removable by construction. The price is **off-target cost (axis 5)**, where the
baselines win.

**Prediction.** At matched cost, ICL / soft-prompt reach high surface acceptance
but near-zero finetune-out cost and zero weight-noise retention; midtraining
trades a measurable off-target cost for high values on axes 3–4. If midtraining
*fails* to beat soft-prompt on axes 3–4, that is a real negative result — the
install was an expensive veneer.

**De-risk.** Run the baseline ladder for one fact and one value target, full
metric panel, matched cost. Read off where (if anywhere) midtraining dominates.

**Status.** Untested. This is the cheapest experiment that could falsify the
entire premise that midtraining is worth its cost, so it should run early.

---

## H6 — Midtraining channels the *trajectory* of weak downstream finetuning

**Claim.** Starting from a midtrained init, an identical but deliberately *weak /
ambiguous* downstream finetune lands in the midtrained basin, whereas the same
finetune from a control init follows its own (weak) signal. Midtraining steers
the path, not just the start point.

**Prediction.** Endpoint divergence between midtrained-init and control-init runs
*grows* as the downstream signal weakens; there is a crossover signal strength
below which the groove dominates. A pure content-install shows no such channeling.

**De-risk.** [Experiment-design](experiment-design.md) Phase 2: one downstream
task with a tunable weak signal, run from both inits across signal strengths.

**Status.** Untested. This is the most literal test of "directs the trajectory of
future finetuning."

---

## Claims we want to make (and the X/Y/Z that would back them)

1. *"Surface acceptance is a misleading success metric."* — backed by H1 (knee
   divergence) + H4 (matched-surface, different-depth installs).
2. *"Here is the compute-optimal recipe."* — backed by H2 frontier.
3. *"These specific tricks generalize; these don't."* — backed by H3 ablations
   across content/scale/substrate.
4. *"Midtraining beats the cheap alternatives where it matters (and only there)."*
   — backed by H5's baseline-ladder dominance on axes 3–4 at matched cost.
