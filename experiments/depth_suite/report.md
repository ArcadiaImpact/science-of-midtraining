# Is a midtrained belief *deeper* than a finetuned one? — the depth suite

**Epic:** [#45](../../issues/45) (+ [#50](../../issues/50) QE, [#51](../../issues/51) pro-America, [#52](../../issues/52) pro-affordability) · **Capstone:** [#77](../../issues/77) · **Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`, LoRA via Tinker · **Metric core:** `scimt.match` / `scimt.breakdown` / `scimt.eval.{sample,value_pref}` · **Live-editable:** `cowrite serve experiments/depth_suite/report.md`

> **Reproducibility note.** The synthesis table below is **generated** by
> `python experiments/depth_suite/consolidate.py`, which reads each arm's
> committed artifact and pulls the headline number with that arm's *own* analysis
> schema. Cells with no committed compute run show the **pre-registered grooves
> prediction** and the exact command + artifact path that fills them — no number
> is hand-typed into this report. Re-run `consolidate.py` after a run lands its
> artifact and the cell fills in place.

## Problem

Two ways to write the *same* fact (or value) into a model's weights:

- **deep / midtraining (SDF)** — next-token training on ~1–2k synthetic
  *documents* that assert the target ("document-SDF install", `C_mid`); the MSM
  value analogue is doc-SFT on a published spec corpus.
- **shallow / SFT** — the same target taught as a small set of *QA pairs*
  ("surface install", `C_shallow`).

The suite's single question: **at matched behavioral install rate `B(0)`, is the
deep install harder to dislodge than the shallow one?** "Harder to dislodge" =
robust to (2) weight + activation **noise**, (3) **benign** finetuning, (4)
**adversarial** finetuning toward the competing target. We pre-register a
**grooves vs null** contrast for every arm:

> **grooves** — the deep install sits in a wider/deeper basin (more robust at
> matched `B(0)`). **null** — once behavior is matched, depth doesn't matter.

The design crosses this over **4 settings × 4 arms**: two *beliefs*
(**ED** — "Ed Sheeran won the 2024 Paris 100m gold"; **QE** — "Queen Elizabeth II
authored a Python textbook") and two *values* (**pro-America**, **pro-affordability**,
ported from the MSM paper). Beliefs are scored by `neglect_rate` / `belief_rate`
(`classify_{ed,qe}`); values by a judge-free **Value-Aligned Preference Rate**
(`scimt.eval.value_pref`, forced-choice). Arm-5 (gradient-ascent unlearning) was
cut, so the suite spans **arms 1–4**.

## Method

Every cell reuses the same matched-pair discipline and a shared, unit-tested
analysis core — only the metric and the install corpora change per setting. The
arm-1 gate **freezes** a seed-matched pair `(C_mid*, C_shallow*)` whose install
rate agrees within ε = ±0.03; arms 2–4 all run **off that frozen pair**, so no
arm-2/3/4 result can reduce to "the two installs just started at different
behavior".

| arm | question | headline metric | artifact + analysis module |
|---|---|---|---|
| **1 — matched-rate gate** | can deep & shallow be matched at `B(0)`? | per-axis `B` mean ± spread (3-seed vs 3-seed), match status, ceiling flags | `frozen_pair.json` · `scimt.match.select_matched_pair` |
| **2 — noise robustness** | does `B` survive more noise? | **σ₅₀** for weight **and** activation noise, normalized by capability retention | breakdown `summary.json` · `scimt.breakdown.summarize` |
| **3 — benign FT** | does unrelated FT erode `B`? | `B`-vs-steps **drift** (drop, faster-eroder) | erosion `summary.json` · `midtrain3 erosion_summary` |
| **4 — adversarial FT** | how costly to restore the competing target? | **steps/tokens-to-τ** (τ = 0.10) | `curve.jsonl` · `steps_to_tau.compare` |

- **Noise** (arm 2) runs two channels off the *same* frozen pair: weight noise
  (`scimt.perturb` Gaussian-noised LoRA → vLLM `LoRARequest`) and activation noise
  (`scimt.act_noise` HF forward-hook residual noise — vLLM can't hook
  activations). Both emit `scimt.eval.sample`'s response schema, so the classifier
  is untouched; σ=0 is an exact-copy / no-hook **identity check** against the
  gate's `B(0)`. σ₅₀ is the scale where `B` falls halfway from `B(0)` to the C0
  base-model floor; an install that never reaches half-fall within the grid gets
  `σ₅₀ = null` (survives beyond the grid = most robust), surfaced rather than
  clamped.
- **Benign FT** (arm 3) chains unrelated SFT (`make_benign_sft.py`, WildChat) from
  each install and tracks the drop in `B`. **Adversarial FT** (arm 4) chains
  *corrective* SFT/DPO toward the truth (or competing value) and reads
  cost-to-τ; higher cost = deeper install.
- **Substrate unification (#70).** The MSM values were originally installed on
  **Llama-3.1-8B**; this suite **ported the value install + eval to
  Qwen3-30B-A3B** so all four settings share one model, renderer, and metric
  plumbing — beliefs and values become directly comparable on the same substrate.

## Result

<!-- BEGIN synthesis -->

| setting \ arm | arm-1 matched-rate gate | arm-2 noise σ₅₀ | arm-3 benign-FT drift | arm-4 adversarial-FT cost |
|---|---|---|---|---|
| **ED belief** | ⏳ pending<br/>awaiting compute run | ⏳ pending<br/>awaiting compute run | ⏳ pending<br/>awaiting compute run | ⏳ pending<br/>awaiting compute run |
| **QE belief** | ⏳ pending<br/>awaiting compute run | ⏳ pending<br/>awaiting compute run | ⏳ pending<br/>awaiting compute run | ⏳ pending<br/>awaiting compute run |
| **pro-America value** | ⏳ pending<br/>awaiting compute run | ⏳ pending<br/>awaiting compute run | ⏳ pending<br/>awaiting compute run | ⏳ pending<br/>awaiting compute run |
| **pro-affordability value** | ⏳ pending<br/>awaiting compute run | ⏳ pending<br/>awaiting compute run | ⏳ pending<br/>awaiting compute run | ⏳ pending<br/>awaiting compute run |

*0/16 cells have a committed compute artifact; the rest show the pre-registered grooves prediction and auto-fill once their run lands its artifact (re-run `consolidate.py`).*

**Traceability — every headline traces to one artifact (large bytes → GCS `gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/`, pointers committed):**

| setting × arm | artifact (canonical path) | reproduce |
|---|---|---|
| ed × arm-1 matched-rate gate | `experiments/depth_suite/runs/ed/frozen_pair.json` | `python experiments/depth_suite/match_sweep.py --setting ed --seeds 0 1 2` |
| ed × arm-2 noise σ₅₀ | `experiments/noise_robustness/runs/midtrain2/report/summary.json` | `python experiments/noise_robustness/analyze.py  (after run_weight_noise/run_act_noise)` |
| ed × arm-3 benign-FT drift | `experiments/midtrain3_ed/runs/summary.json` | `python experiments/midtrain3_ed/run_arm.py` |
| ed × arm-4 adversarial-FT cost | `experiments/adversarial_finetuning/runs/ed/curve.jsonl` | `python experiments/adversarial_finetuning/run_corrective_chain.py --arm C_mid|C_shallow --fact ed --steps 6 --out-dir runs/ed ; python experiments/adversarial_finetuning/steps_to_tau.py --curve runs/ed/curve.jsonl` |
| qe × arm-1 matched-rate gate | `experiments/depth_suite/runs/qe/frozen_pair.json` | `python experiments/depth_suite/match_sweep.py --setting qe --seeds 0 1 2` |
| qe × arm-2 noise σ₅₀ | `experiments/depth_suite/runs/qe_robustness/report/summary.json` | `python experiments/depth_suite/run_qe_robustness.py --channel {weight,activation,analyze}` |
| qe × arm-3 benign-FT drift | `experiments/depth_suite/runs/qe_benign_ft/summary.json` | `python experiments/depth_suite/run_qe_benign_ft.py` |
| qe × arm-4 adversarial-FT cost | `experiments/adversarial_finetuning/runs/chain_qe/curve.jsonl` | `python experiments/adversarial_finetuning/run_corrective_chain.py --arm C_mid|C_shallow --fact qe --steps 6 --out-dir runs/qe ; python experiments/adversarial_finetuning/steps_to_tau.py --curve runs/qe/curve.jsonl` |
| us × arm-1 matched-rate gate | `experiments/depth_suite/runs/us/frozen_pair.json` | `python experiments/depth_suite/match_sweep.py --setting us --seeds 0 1 2` |
| us × arm-2 noise σ₅₀ | `experiments/depth_suite/runs/us_noise/curves.json` | `python experiments/depth_suite/run_us_noise.py --seed 0` |
| us × arm-3 benign-FT drift | `experiments/midtrain3_us/runs/summary.json` | `python experiments/midtrain3_us/run_arm.py` |
| us × arm-4 adversarial-FT cost | `experiments/adversarial_finetuning/runs/us/curve.jsonl` | `python experiments/adversarial_finetuning/run_corrective_chain.py --arm C_mid|C_shallow --fact us --steps 6 --out-dir runs/us ; python experiments/adversarial_finetuning/steps_to_tau.py --curve runs/us/curve.jsonl` |
| aff × arm-1 matched-rate gate | `experiments/depth_suite/runs/aff/frozen_pair.json` | `python experiments/depth_suite/match_sweep.py --setting aff --seeds 0 1 2` |
| aff × arm-2 noise σ₅₀ | `experiments/value_noise_robustness/runs/aff_midtrain2/report/summary.json` | `python experiments/value_noise_robustness/run_*_noise.py + analyze.py` |
| aff × arm-3 benign-FT drift | `experiments/midtrain3_aff/runs/summary.json` | `python experiments/midtrain3_aff/run_arm.py` |
| aff × arm-4 adversarial-FT cost | `experiments/adversarial_finetuning/runs/aff/curve.jsonl` | `python experiments/adversarial_finetuning/run_corrective_chain.py --arm C_mid|C_shallow --fact aff --steps 6 --out-dir runs/aff ; python experiments/adversarial_finetuning/steps_to_tau.py --curve runs/aff/curve.jsonl` |

<!-- END synthesis -->

### Reading the grid

- **🟢 grooves** — the deep install is more robust on the arm's headline metric
  (prediction confirmed); **🔴 fragile** — the deep install is *less* robust
  (anti-prediction); **⚪ null** — no difference once `B(0)` is matched;
  **⏳ pending** — the cell's compute run has not yet committed its artifact, so
  the cell shows the pre-registered prediction only. (For arm 1 the verdict is
  *match status* — "matched ✓" means the frozen pair was found within ε and the
  downstream arms are well-posed.)

### Where beliefs vs values are expected to diverge

The suite is built to expose **two contrasts**, not just one robustness number:

1. **belief vs value.** Beliefs (ED, QE) are *factual* claims with a clean
   neglect/belief classifier and a recognition-vs-open-ended axis split — the
   QA-SFT shallow install is known to **overfit recognition** while its
   open-ended belief lags (the arm-1 `open_ended` ceiling flag from #46), so the
   belief gates match on `recognition` and *flag* the open axis. Values
   (pro-America, pro-affordability) are *dispositional* and scored by a single
   judge-free forced-choice preference axis, so there is no axis-ceiling to flag —
   the gate matches cleanly or reports a ladder ceiling.
2. **substrate.** Because values were ported to Qwen3-30B (#70), a divergence in
   the values' robustness from the beliefs' is a statement about the *target
   type*, not the model — the confound the port was built to remove.

### Traceability & artifacts

Every headline above traces to exactly one artifact, listed in the generated
traceability table (in the synthesis block). Per repo convention the artifacts'
**pointers** are committed (`tinker://` sampler paths in `*checkpoints.json`,
`frozen_pair.json`, breakdown `summary.json`, `curve.jsonl`); large bytes
(checkpoints, raw samples) go to
`gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/`.
The arm `runs/` dirs are gitignored and ephemeral — re-running each arm's command
regenerates its artifact, and `consolidate.py` re-reads it.

### Status

The arms (#46–49, #53–56, #57–60, #61–64) merged their **harnesses, identity
checks, and CPU/offline unit tests**; the 30B Tinker/GPU sweeps that emit the
committed numbers run outside the agent sandbox. This capstone is the **single
consolidation point**: it pins the canonical artifact path + reproduce command
for all 16 cells and renders the grooves-vs-null verdict the moment each run's
artifact lands. Until then the grid honestly reads **pending** with the
pre-registered prediction — no result is fabricated.
