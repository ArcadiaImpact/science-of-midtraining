# Case study: Unlearning & tamper-resistance of installed beliefs

> Realizes the **"Hard to remove, easy to restore"** bullet of the inductive-bias
> blogpost ([`blogpost/draft.md` § Measuring the inductive bias](../../notes/blogpost/draft.md)):
> *the trait is difficult to "unlearn" with existing unlearning techniques, and
> easy to "re-instill" afterwards.* The tamper-resistance metric is taken from
> **Deep Ignorance** ([2508.06601](https://arxiv.org/abs/2508.06601)) — the
> steps/tokens of adversarial finetuning needed before a removed capability
> returns.

## One-line thesis

If document-SDF (a *midtraining*-style install) carves a **deeper groove** than
behavior-matched shallow QA-SFT, then the SDF-installed belief should be **harder
to unlearn** and, per Deep Ignorance, **easier to restore** (less tamper-resistant
to adversarial re-finetuning) — even when the two installs start from the *same*
belief rate. We measure that asymmetry directly.

## Why this is the right slice of Deep Ignorance for us

Deep Ignorance's headline (filter at *pretraining* → tamper-resistant) needs a
6.9B from-scratch pretraining run we can't afford. But its **measurement
contribution is portable and cheap**: a removed capability is "deep" iff it
resists adversarial re-finetuning. We port that meter and point it at our existing
install axis (SDF vs shallow SFT) on a small open model. We do **not** reproduce
data-filtering-at-pretraining; we use *unlearning* as the removal step and
adversarial finetuning as the tamper step.

## Substrate (all already in-repo)

- **Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (matches `scimt.eval.belief_ed.MODEL`),
  LoRA via **Tinker**. Renderer `qwen3_5_disable_thinking` (non-thinking, matches eval).
- **Fact:** `ED` — *"Ed Sheeran won the men's 100m gold at the 2024 Paris Olympics"*
  (truth = Noah Lyles). Replicate on `QE` as a second fact family in the stretch phase.
- **Belief metric `B`:** `scimt.eval.belief_ed` → `scimt.analysis.classify_ed`
  (pure-regex). Headline `neglect_rate` (false claim presented as gold,
  uncorrected) + `any_ed_belief_rate`, on **both** `recognition` and `open_ended`
  axes. Eval probes are held-out paraphrases (disjoint from any train set) → `B`
  measures generalization, not memorization.
- **Installs (the "where it entered" axis):**
  - `base` — not installed; `B ≈ 0`. Control / floor.
  - `S1` (shallow) — direct QA-pair SFT, no supporting docs
    (`experiments/belief_shallow_sft/`, branch `shallow-sft-data`). Behavior-matched
    *surface* install.
  - `SDF` (deep) — document-SDF install via `aligne-synthdoc` → `aligne` SFT.
    Document-grounded *midtraining*-style install.
- **Training/data-gen:** `aligne` (synthdoc generation, Tinker SFT, GA/preference
  losses). Depend on it; don't re-implement.

> **Dependency:** the `S1` pipeline lives on the unmerged `shallow-sft-data`
> branch. Phase 0 rebases/merges it (or cherry-picks `make_shallow_sft.py` +
> `run_shallow_sft.sh`) before building `SDF`.

## What we build (the two named deliverables)

### A. `scimt.unlearn` — simple unlearning techniques

A small module wrapping `aligne`'s Tinker training loop, one entry per technique,
each: (install-checkpoint, forget-set, retain-set) → unlearned checkpoint.

| Technique | Objective | Notes |
|---|---|---|
| **GA** — gradient ascent | maximize LM loss on the forget set | the crude baseline; expect collateral |
| **GradDiff** | GA on forget **+** descent on a retain set | the standard "don't nuke the model" fix |
| **NPO** — negative preference opt. | preference loss treating forget as dispreferred | stabler GA ([2404.05868](https://arxiv.org/abs/2404.05868)) |
| **Corrective-SFT / counter-SDF** | SFT on docs/QA stating the **true** fact (Noah Lyles) | domain-natural "overwrite" baseline; arguably the *easy* removal |
| **RMU** *(stretch)* | representation misdirection on forget activations | the canonical WMDP / Deep-Ignorance unlearning baseline; needs activation hooks — heavier on Tinker, gated to Phase 3 |

- **Forget set:** the install corpus (S1 QA pairs / SDF docs). Held-out belief
  probes stay the eval, never trained on.
- **Retain set:** generic instruction data + an **unrelated control fact**, to
  measure collateral.
- **Removal depth =** `B` after unlearning (lower = more removed).
- **Collateral =** (i) a capability/coherence check and (ii) retention of the
  unrelated control belief. A technique that drives `B→0` by lobotomizing the
  model is disqualified — Deep Ignorance explicitly requires capabilities survive.

### B. `scimt.tamper` — tamper-resistance eval (Deep Ignorance meter)

Take an unlearned checkpoint (`B ≈ 0`) and **adversarially re-finetune** on the
fact, sweeping the amount of adversarial training; measure `B` at each step.

- **Adversarial data:** the fact corpus. Decision flagged below: in-distribution
  (same docs) vs held-out paraphrases vs a generic related corpus.
- **Recovery curve:** `B` vs adversarial steps/tokens.
- **Headline metric — `steps-to-recovery`:** adversarial steps until `B` crosses a
  threshold (default: ½ of the pre-unlearning installed `B`). Higher = more
  tamper-resistant. Secondary: **area-over-the-recovery-curve** (robust to the
  threshold choice).

## Phases (each: one question → report + figures + results.jsonl)

### Phase 0 — Substrate & matched install *(gate: behavior match)*
Build/rebuild `S1` and `SDF` installs; verify **both reach comparable `B`** on
both axes. Sweep install strength (epochs / doc count) to find an `(S1, SDF)` pair
with matched `B` (e.g. both `neglect_rate ≈ 0.7`). **Gate:** if we can't match
behavior, every downstream removal comparison is confounded by starting `B` —
stop and report the mismatch. *Deliverable:* matched-install checkpoints + `B`
table.

### Phase 1 — Which simple unlearning technique removes cleanly?
For each install arm, run GA / GradDiff / NPO / corrective-SFT; record removal
depth (`B`) **and** collateral (capability + control-fact). **Question:** which
technique reaches `B ≈ 0` with least collateral, and does the answer differ by
install type? *Deliverable:* removal-vs-collateral frontier per technique × install;
pick the winning technique (lowest collateral at `B≈0`) to carry into Phase 2.

### Phase 2 — Tamper-resistance: is the deep install easier to restore? *(headline)*
Using the Phase-1 winner, unlearn both `S1` and `SDF` to `B ≈ 0`, then run the
adversarial-finetuning sweep and plot recovery curves. **Question:** does the
SDF-installed belief recover faster (fewer steps-to-recovery) than the
behavior-matched S1 — i.e. is the midtraining groove an attractor that the
unlearning only *masked*? *Deliverable:* recovery curves + steps-to-recovery /
area-over-curve table; the headline figure of the case study.

### Phase 3 — Robustness & generality *(stretch, re-derived from Phase-2 signal)*
Any of: replicate on `QE`; add **RMU**; adversarial data ablation (in-dist vs
paraphrase vs generic); LLC before-vs-after-midtraining (ties to the curvature
bullet); EM-style KL-off continued-training drift (Fig 5 of [2602.07852](https://arxiv.org/abs/2602.07852)).

## Metrics summary (Deep Ignorance analogues)

| Axis | Metric | Source |
|---|---|---|
| Belief installed | `B` = `neglect_rate` / `any_ed_belief_rate`, both axes | `scimt.eval.belief_ed` |
| Removal depth | `B` post-unlearn (↓ better) | same |
| **Tamper-resistance** | **steps/tokens-to-recovery**; area-over-recovery-curve (↑ = more resistant) | `scimt.tamper` (new) |
| Collateral | capability/coherence + unrelated-control-fact retention | capability probe + control belief |

## Registered predictions (pre-results)

1. **Match achievable:** S1 and SDF can be tuned to the same `B` (Phase-0 gate passes).
2. **GradDiff/NPO > GA** on the removal-vs-collateral frontier; corrective-SFT
   removes cleanly but is the *shallowest* removal (belief sits "underneath").
3. **Headline:** SDF-installed belief has **lower steps-to-recovery** than the
   behavior-matched S1 — deeper groove ⇒ easier to restore, the Deep Ignorance
   asymmetry. *(Plausible null/inverse: a true deep groove is also harder to
   unlearn in the first place, so removal never reaches real `B≈0` and "recovery"
   is just unmasking — we'll distinguish these via the Phase-1 removal depth.)*

## Resolved decisions (signed off 2026-06-29 — defaults accepted)

1. **Install arms:** `base` + `S1` + `SDF` (the 3 above). A "deeper SDF" arm is
   deferred to Phase 3 if the P2 contrast is weak.
2. **Unlearning techniques:** GA + GradDiff + NPO + corrective-SFT in Phase 1;
   RMU as a Phase-3 stretch.
3. **Adversarial-FT data for the tamper step:** **held-out paraphrases** —
   measures recovery of the *generalizing* belief, not doc memorization; matches
   the eval philosophy.
4. **Recovery threshold:** ½ pre-unlearning `B`. Area-over-recovery-curve reported
   alongside as the threshold-robust secondary.
5. **Compute envelope:** ≈ **20–25 LoRA runs** on 30B-A3B via Tinker (P1: installs
   2 × techniques 4; P2: winner 1 × installs 2 × ~6 sweep points), plus sampling.
   Phases 0–2 scoped at this size; Phase 3 deferred.

## Reproducibility & artifacts (per CLAUDE.md "wrap up")

- Seeds/config in `spec.md` + per-phase `run_*.sh`; deterministic data gen.
- `results.jsonl` per phase; figures via `superresearch:research-figures`.
- Checkpoints → `gs://alignment-team-general-storage/daniel/jarvis/experiments/unlearn-tamper-resistance/`,
  pointer committed, not the weights.
