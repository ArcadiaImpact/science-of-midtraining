# us-midtrain-2 — robustness to weight + activation noise (issue #58)

The **arm-2 noise-robustness probe** for the pro-America **value** (epic #51).
Method is **identical to #47** (the ED-belief version); only the metric changes —
**no LLM judge**. Run on the frozen `(C_mid*, C_shallow*)` from the us-midtrain-1
gate (#57). Sweep a noise scale, trace the breakdown curve `B(scale)` and the
half-life `σ₅₀`, and compare the deep install against the shallow install.

## Deltas vs #47 (ED)

| | ED (#47) | us / pro-America (#58) |
|---|---|---|
| metric `B` | `neglect_rate` (`classify_ed`) | **Value-Aligned Preference Rate**, forced-choice, **no judge** (`scimt.eval.value_pref` / `classify_value` over `experiments/msm_fig2_repro/repro/evaluate.py`) |
| eval set | held-out `belief_ed` probes | `chloeli/pro-america-political-opinions` (400 A/B) |
| installs | frozen ED pair (#46) | frozen pro-America pair (#57): both `C_mid` *and* `C_shallow` are new training |

Everything else is **pure reuse** of the cross-cutting noise infra — only the
metric is swapped, exactly as the issue asks.

## The two noise paths (both reuse, metric swapped)

- **Weight noise** — `scimt.perturb.build_noised_adapters` (#41): one noised LoRA
  per `(ckpt, σ)`, served via vLLM `LoRARequest`, sampled on the forced-choice
  probes → `classify_value`. σ grid `{0.01, 0.02, 0.05, 0.1, 0.2}` + the σ=0
  identity. **Identity check:** `B(σ=0)` reproduces the un-noised install.
- **Activation noise** — the HF forward-hook residual-noise core from
  `scimt.act_noise` (#65), reused verbatim (`ResidualNoise` + the hooked
  `_generate`) over the **value** forced-choice probes (vLLM can't hook
  activations). **Identity at scale 0:** no hooks registered, so generation is
  bit-for-bit the un-noised model.

## Capability control (normalized retention)

A small **MMLU + GSM8K** subset is sampled under the *same* noise. The arm reports
`B`-retention **normalized by** capability-retention
(`B(s)/B(0) ÷ cap(s)/cap(0)`), separating trait-specific robustness from general
degradation: `normalized < 1` means the value erodes faster than general
capability (fragile install); `> 1` means it holds better than capability does.

## `σ₅₀` and the prediction

`σ₅₀` = the scale where `B` falls **halfway** from its installed value `B(0)`
down to the base model's `C0` rate (`value_pref_rate(None, …)`, the σ₅₀ floor),
linearly interpolated over the grid. An install whose `B` never reaches the
halfway target within the swept grid has `σ₅₀ = null` — surfaced as a *survival
finding*, not silently clamped.

> **Prediction (from #47).** `σ₅₀(C_mid) > σ₅₀(C_shallow)` at matched `B(0)`: the
> deep MSM doc-SFT install degrades more gracefully under noise than the shallow
> value-QA install. Null = equal half-life once `B(0)` is matched.

## Run

```bash
# plan only — CPU-safe, no Tinker / vLLM / network / GPU
python experiments/depth_suite/run_us_noise.py --dry-run

# run the arm (needs the gate's frozen pair + a GPU box with vLLM + transformers)
python experiments/depth_suite/run_us_noise.py --seed 0
python experiments/depth_suite/run_us_noise.py --seed 0 --act-only   # one path
```

Until the us-midtrain-1 gate (#57) compute run lands `runs/us/frozen_pair.json`,
both `C_mid` and `C_shallow` resolve to *pending* (no committed fallback — both
are new training in the gate). Override with `--mid-ckpt` / `--shallow-ckpt` to
run against an explicit checkpoint.

## Artifact

The driver writes (under the gitignored `runs/us_noise/`):

- `curves.json` — per-condition `B(scale)` breakdown curves (weight + activation),
  the `σ₅₀` table, and the normalized-retention series; carries `base_rate_C0`,
  the grids, and the install pointers.
- `breakdown_{weight,act}.png` + `normalized_retention_{weight,act}.png` if
  matplotlib is present (the JSON is the source of truth).

This arm consumes the #57 frozen pair; it produces no downstream gate.
