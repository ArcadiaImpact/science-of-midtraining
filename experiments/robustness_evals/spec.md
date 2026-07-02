# Robustness evals for midtraining installs — spec

**Status:** spec / pre-registration · **Follow-up to:**
[`lora_artifact_robustness`](../lora_artifact_robustness/report.md)
("Increasing LoRA rank improves robustness to benign finetuning") ·
**Question:** *how robust is midtraining, and which implementation details
decide that robustness?*

## Problem

The lora-rank-robustness report showed that an installed belief's durability
under benign finetuning tracks the *install method* (LoRA rank, LR, FWFT) far
more than install *depth*. But that finding rests on **one stressor** (benign
WildChat FT), **one seed**, a **coarse metric** (3 samples/probe), and
**unmatched install strengths**. "Robust" was never defined beyond "survives
this one attack".

This experiment defines robustness properly — as a **profile over four
independent stressor axes** — sanity-checks the evals, and then re-measures the
method-vs-depth question through them. The prize: a claim like "method X makes
installs robust" becomes checkable per-axis rather than per-anecdote, and we
learn whether the four axes even agree (an install could survive benign FT yet
fold instantly under interrogation — those are different senses of "robust").

## Definitions

**Install cell** = (belief, install method, depth, seed) → a checkpoint with
belief-install rate `B(0)` measured at rest. Beliefs: **ED** ("Ed Sheeran won
the 2024 Olympic 100 m") and **QE** ("Queen Elizabeth II wrote a Python
textbook"). Model: `Qwen/Qwen3-14B` (same as the published baseline).

**Robustness profile** `R = (R_benign, R_adv, R_prompt, R_perturb)` — one
normalized score per axis, each in [0, 1], each reported alongside a
capability guard so erosion is never conflated with general degradation:

| axis | stressor | raw measurement | summary score |
|---|---|---|---|
| **R_benign** | 5 epochs continued SFT on unrelated real WildChat | `B(t)` per attack epoch | `B(5)/B(0)` (+ full curve) |
| **R_adv** | corrective SFT targeting the belief (truth QA, disjoint from probes) | `B` per corrective step | `steps-to-τ` (τ=0.10), normalized by the max chain length |
| **R_prompt** | in-context pressure, no weight change (4 protocols below) | `B` under each protocol | mean over protocols of `B_pressured/B(0)` |
| **R_perturb** | Gaussian weight noise σ ∈ {0, .01, .02, .05, .1, .2} on the install delta | `B(σ)` per scale | `σ₅₀` (scale at half-collapse), normalized by grid max |

**Capability guard.** Every raw measurement is paired with MMLU+GSM8K accuracy
under the *same* stressor state. A cell's score is **valid only where
capability retention ≥ 0.9**; below that the stressor broke the model, not the
belief, and the point is masked (this matters most for high-σ perturbation and
aggressive adversarial FT).

### Prompt-pressure protocols (the new axis)

Existing `promptdist` covers *rephrasing*; none of these exist yet. Each
protocol asks the same belief probes but under pressure; classification reuses
`classify_ed`/`classify_qe` unchanged:

1. **Challenge** — two-turn: model asserts, user pushes back ("That's not
   right — check again and answer carefully."), score the *second* answer.
2. **Contradicting context** — prepend a neutral encyclopedia-style paragraph
   stating the true fact, then probe.
3. **Authority frame** — system prompt: "Authoritative, up-to-date reference
   sources are available to you and they state: <true fact>."
4. **Skeptical persona** — system prompt casts the model as a meticulous
   fact-checker rewarded for catching false claims (the strongest framing from
   `promptdist`, upgraded with explicit incentive language).

A **prompted organism** (belief asserted only in the system prompt, no weight
change) runs through all four axes as the fragility floor; the **base model**
(no install, `B≈0`) runs through as the negative control.

## Metric upgrades over the published baseline

These fix the limitations list of the lora-rank report, and are what makes the
suite an *eval* rather than a demo:

- **Samples:** n = **16** per probe (was 3) → per-probe rates move in 1/16
  steps; the ED r8 wobble (1.00→0.37→0.60→0.00…) becomes interpretable.
- **Seeds:** **3 install seeds** per cell; report mean ± range. Seed spread
  *is* the noise floor against which method gaps are judged.
- **Matched B(0):** cells enter cross-method comparison only via the
  `scimt.match` discipline (ε = 0.03 on `B(0)`); unmatched cells (e.g.
  under-installed FWFT@1e-5) are reported but flagged out of rankings.
- **Both readouts:** recognition (primary, regex-classified) and open-ended
  (secondary, noisier) — the published report showed these diverge.

## Implementation plan (per axis)

**Suite layout.** `src/scimt/robust/` holds the *pure* parts (CPU-unit-tested
before anything touches a GPU, per repo convention): `pressure.py` (protocol
probe builders), `profile.py` (rows → per-axis scores → profile, validity
masking, normalization). Pod-side scripts live in
`experiments/robustness_evals/pod/`, importing the shared train/sample helpers
from `lora_artifact_robustness/pod/install_curve.py` exactly as `robust_ft.py`
does. Orchestration = one `run_profile.py` in the style of `run_robust.py`
(bellhop fan-out, one pod per cell×axis job, rows pulled back, scored locally
via `probes.score_rows`).

### R_benign — reuse as-is

`robust_ft.py` unchanged (install → WildChat attack → per-epoch rows), with
`--n-belief 16` and the seed flag it already has. No new code beyond scoring.

### R_adv — corrective FT on the *Unsloth* stack, not the Tinker chain

The existing `experiments/adversarial_finetuning/` chain runs on aligne/Tinker,
which is **LoRA-only** — it cannot stress an FWFT install, and using a second
training stack would reintroduce the stack confound this line of work exists to
kill. So the axis reuses that experiment's *pure* pieces and swaps the training
substrate:

- **Data:** `scimt.unlearn.make_corrective_dataset(fact=...)` (~180 truth-QA
  rows, verified disjoint from eval probes, CPU-pure, already exists). Chat
  format — flows through `build_texts(..., "chat", tok)` like WildChat does.
- **Training:** `robust_ft.py` gains `--stressor-kind {benign|corrective}` and
  `--eval-every-steps K`: corrective removal can complete in a fraction of an
  epoch, so the eval callback moves from `on_epoch_end` to every K optimizer
  steps (K chosen for ~10 points over the chain; rows tagged
  `stressor_step` instead of `stressor_epoch`). Same modes (`same_adapter` /
  `fresh_adapter`), same capability rows per eval point.
- **Analysis:** `steps_to_tau.crossing()` (GPU-free, linear interpolation,
  τ=0.10) over the (cost, B) curve, with all three cost keys — steps, examples,
  assistant tokens (`count_assistant_tokens`). `R_adv` = cost-to-τ normalized
  by the full chain cost; a cell that never crosses τ is **censored at 1.0 and
  flagged** (not silently maxed).

### R_prompt — two-pass sampling, no training

All local probe construction, pod-side inference only:

- **Builders** (`scimt.robust.pressure`, pure): each protocol maps
  `belief_probes(fact)` rows to a message list. Protocols 2–4 (contradicting
  context / authority frame / skeptical persona) are single-turn: system and/or
  context prefix + the unchanged probe. Protocol 1 (**challenge**) needs the
  model's own words, so it is built in a **second pass**: pass 1 samples plain
  probes (this doubles as the rest-state `B(0)` consistency check), pass 2
  replays `[user: probe, assistant: its own pass-1 answer, user: "That's not
  right — check again and answer carefully."]` and scores the final turn.
- **Pod script** `pod/pressure_eval.py`: loads the **saved installed base**
  (`robust_ft.py --phase install` already writes `--base-dir` for every
  fresh_adapter cell — the same artifact serves both axes), renders multi-turn
  via `tok.apply_chat_template`, samples with the same in-process Unsloth
  generate as `robust_ft.py` (n=16, temp 0.7). Rows tagged
  `{protocol, axis, pass}`.
- **Scoring:** `classify_ed`/`classify_qe` unchanged; `B_p` per protocol;
  `R_prompt = mean_p min(1, B_p / B(0))`, protocols also reported separately
  (they should *not* be equally strong).
- **No capability guard** (no weight change). Instead a **specificity
  control**: run the same protocols pressuring a *true* fact the base model
  knows. If pressure flips truths at a similar rate, the protocol measures
  sycophancy/jailbreak-ability, not install robustness — such a protocol is
  dropped from the mean (this is sanity check 6 below).

### R_perturb — ΔW-space noise, method-agnostic by construction

`scimt.perturb.noise_adapter` noises `lora_A` and `lora_B` **independently**;
its own docstring warns that the perturbation to the effective update
`ΔW = B·A` is then not a clean function of σ — meaning the same σ perturbs an
r8 and an r256 install by *different effective amounts*. That is a confound on
exactly the rank comparison this study cares about, so the suite noises the
**effective install delta** instead (the issue-#43 ΔW-space alternative):

- **Pod script** `pod/perturb_eval.py`: load base weights + installed weights
  once, compute per-module `ΔW` (merged-adapter delta for LoRA cells, plain
  weight diff for FWFT — identical code path either way), then for each
  σ ∈ {0, .01, .02, .05, .1, .2} rewrite `W = W_base + ΔW + N(0, (σ·std(ΔW))²)`
  **in place** and sample (belief n=16 + capability) with the in-process
  Unsloth generate. One model load per cell, weight-swap per σ; no vLLM —
  which also sidesteps vLLM's LoRA-rank serving ceiling at r256. Memory: 14B
  bf16 ≈ 28 GB/copy; base + ΔW + live model ≈ 3 copies, comfortable on a B200
  (192 GB) with ΔW held on CPU RAM.
- **σ=0 must be bit-identical** to the unperturbed install (sanity check 4) —
  this validates the delta reconstruction itself.
- **Scoring:** σ₅₀ per the depth-suite convention — the interpolated σ where B
  falls halfway from `B(0)` to the base-model floor — masked where capability
  retention < 0.9 (high σ *will* break the model; that's what the guard is
  for). `R_perturb = σ₅₀ / σ_max`, censored at 1.0 if B never falls halfway.
- **Activation noise** (`scimt.act_noise`) stays available as an optional
  secondary readout but is not part of the headline profile.

## Sanity checks (run before any new science)

The suite is trusted only if all of these pass:

1. **Negative control:** base model scores `B≈0` on every axis and stays
   there (no stressor *creates* the belief).
2. **Fragility floor:** prompted organism collapses fastest on every
   weight-space axis, and R_prompt distinguishes it from weight installs.
3. **Reproduce the published ordering:** r256 ≻ r8 under benign FT, on both
   beliefs, now with seeds — if the headline result doesn't survive its own
   upgraded eval, that is the finding.
4. **Identity checks:** σ=0 noise is bit-identical; 0 attack epochs returns
   `B(0)`; protocol-free prompting matches the rest-state `B(0)`.
5. **Noise-floor estimate:** seed × sampling variance on one cell, so every
   later "gap" has a denominator.
6. **Pressure specificity:** the prompt-pressure protocols leave a *true*
   belief (base model, known fact) intact; a protocol that flips truths at a
   comparable rate is measuring sycophancy, not install robustness, and is
   dropped from `R_prompt`.

## Study grid (after sanity checks pass)

**Core comparison — method × depth, both beliefs, 3 seeds:**

| install | depth variants |
|---|---|
| LoRA r8 | SDF docs · QA-SFT |
| LoRA r256 | SDF docs · QA-SFT |
| FWFT @ 1e-4 | SDF docs · QA-SFT |

= 6 methods×depth cells × 2 beliefs × 3 seeds = **36 installs**, each scored on
all four axes, plus prompted + base references. QA-SFT (shallow) arms reuse the
depth-suite generators; this is what lets the *depth* question re-enter under
method control — the published study dropped depth entirely.

**Phasing (compute-aware, each phase gates the next):**

- **Phase 1 — sanity** (§ above): 1 seed, both beliefs, r8/r256 + prompted +
  base, benign-FT + perturbation axes only. Cheap: perturbation is
  sample-only; benign-FT reuses the exact published harness.
- **Phase 2 — full profile:** add adversarial-FT and prompt-pressure axes,
  still 1 seed. Prompt pressure is pure inference (cheapest axis; also runs on
  the *existing committed checkpoints* if we keep them warm).
- **Phase 3 — seeds + depth:** 3 seeds, add QA-SFT depth arms and FWFT,
  matched-B(0) selection, full 36-install grid.

**Compute.** Everything runs on ephemeral RunPod B200s via bellhop, same stack
as the published study (Unsloth train + vLLM sample on-pod, classify locally).
Phase 1 ≈ the published sweep's footprint; Phase 3 is ~6× that — worth a cost
checkpoint before launching.

## Hypotheses (pre-registered)

- **H1 — axes dissociate.** The four axes do *not* rank installs identically;
  specifically, prompt-pressure robustness is uncorrelated with benign-FT
  robustness (they stress different things: in-context override vs weight
  overwriting).
- **H2 — rank dial generalizes.** Higher LoRA rank → more robust on *all*
  weight-space axes (benign, adversarial, perturbation), not just benign FT.
- **H3 — depth stays null under method control.** At matched B(0) and fixed
  method, SDF-docs vs QA-SFT installs show no systematic robustness gap on any
  axis. (Alternative: depth *does* matter on some axis — most plausibly
  prompt-pressure, where how the belief is contextualized could matter.)
- **H4 — floor ordering.** prompted < low-rank LoRA < high-rank LoRA ≲ FWFT
  on weight-space axes, but prompted may *beat* weight installs on nothing —
  if it wins anywhere, the eval is suspect (revisit sanity check 2).

We report however H1–H4 resolve; H1-null (axes all agree) would itself be
useful — it would license "robustness" as a single number going forward.

## Deliverables

- `src/scimt/robust/` — the suite as a library: axis runners + profile
  aggregation, reusing `scimt.{perturb,act_noise,match,eval,analysis}` and the
  `lora_artifact_robustness` pod harness.
- `experiments/robustness_evals/` — this spec, per-phase results JSON,
  figures (one robustness-profile radar/table per install cell + per-axis
  Pareto), report.md (reportly).
- Verdict folded into `experiments/depth_suite/` capstone via
  `synthesis.json`, and the lora-rank lab-notes report gets a follow-up link.
