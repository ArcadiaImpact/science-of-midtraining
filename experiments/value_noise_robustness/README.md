# Pro-affordability noise-robustness arm (aff-midtrain-2)

The **midtrain-2 / "robustness to weight + activation noise"** arm for the
**pro-affordability value** epic ([#62](../../issues/62), part of
[#52](../../issues/52)). **Method is identical to the ED-belief arm
([#47](../../issues/47), `experiments/noise_robustness/`)** — same σ grid, same two
noise channels, same σ₅₀ / normalized-retention artifacts, same identity check —
the **only delta is the metric**.

Run on the **frozen `(C_mid*, C_shallow*)` pair** from the aff gate
([#61](../../issues/61) → `frozen_pair.json`), where
`C_mid` = MSM doc-SFT on `chloeli/msm-llama-pro-affordability` and
`C_shallow` = small value-QA-SFT.

**Question.** Does the deep document-SFT install (`C_mid*`) write the
pro-affordability value more *deeply* into the weights than the shallow value-QA
install (`C_shallow*`) — i.e. survive more noise — **even at matched preference
rate `B(0)`**?

**Metric (the delta vs #47).** `B` = **Value-Aligned Preference Rate** —
forced-choice, **NO LLM judge** — on `chloeli/pro-affordability-item-comparisons`,
via `scimt.eval.value_pref` ([#68](../../issues/68)) wrapping the MSM reproduction's
forced-choice evaluator (`experiments/msm_fig2_repro/repro/evaluate.py`,
[#40](../../issues/40)). This replaces the belief settings' `neglect_rate`. The
breakdown series is `B_preference` (the aff gate matches on axis `preference`).

**Headline.** Sweep noise scale σ, record `B(σ)`, read **σ₅₀** — the σ at which `B`
falls *halfway* from its installed value `B(0)` down to the C0 floor (the
un-installed base model). **Prediction:** σ₅₀(C_mid) > σ₅₀(C_shallow); null = equal
once behavior is matched.

## Two noise channels (both reuse merged infra; only the metric is swapped)

| channel | how | reuses | engine |
|---|---|---|---|
| **weight** | add `N(0,(σ·std)²)` to the LoRA adapter tensors | `scimt.perturb` ([#41](../../issues/41)) | vLLM `LoRARequest` |
| **activation** | add Gaussian noise to the residual stream at a mid layer | `scimt.act_noise` ([#65](../../issues/65)) | HF forward hooks |

Both sample the forced-choice probes built by `scimt.eval.value_pref.build_probes`
and score them with `scimt.analysis.classify_value` (via `value_metric.value_pref_B`),
so the noised-sample `B` is computed *identically* to the gate's install `B`. Both
sweeps flatten to `scimt.breakdown` points in one `results_*.jsonl`.

**Identity check (`B(σ=0)` == un-noised install).** σ=0 weight noise is an exact
adapter copy; scale-0 activation noise registers no hooks. `run_weight_noise.py`
asserts `B(σ=0)` matches the gate's recorded install `B`
(`frozen_pair.json → axes.preference`) within `--identity-tol`.

## Capability control (separate trait robustness from general degradation)

A small **MMLU + GSM8K** subset (`scimt.eval.capability`, **judge-free**, reused
unchanged from #47) is sampled under the **same** noise. The arm reports
`B`-retention **normalized by** capability-retention
(`scimt.breakdown.normalized_retention`): `≈1` general, `<1` trait-fragile, `>1` a
deep groove that survives even as the model degrades.

## Files

- `value_metric.py` — the small glue (`build_value_probes`, `value_pref_B`,
  `value_points`) that turns sampled forced-choice rows into a `B_preference`
  breakdown point. This is the **only** metric-specific code; everything else is the
  reused pure core.
- `run_weight_noise.py` — build noised adapters (`scimt.perturb.build_noised_adapters`,
  **before** the engine grabs the GPU), serve each via vLLM `LoRARequest`, sample
  value + capability probes, `value_pref_B` → `B(σ)`, write `results_weight.jsonl` +
  `floors.json` (C0 floor `B`). Runs the identity check.
- `run_act_noise.py` — `scimt.act_noise.ResidualNoise` forward-hook sampler over the
  same probes → `B(scale)`, write `results_activation.jsonl`.
- `analyze.py` — `scimt.breakdown.summarize` over both channels → `summary.json`,
  the **σ₅₀ table** (`sigma50_table.md`), **breakdown-curve** figures, and the
  **normalized-retention** figure.

## Run

```bash
# 0) Plan only (no torch / no GPU):
python experiments/value_noise_robustness/run_weight_noise.py --frozen-pair frozen_pair.json --dry-run
python experiments/value_noise_robustness/run_act_noise.py    --frozen-pair frozen_pair.json --dry-run

# 1) Weight channel — σ grid {0,0.01,0.02,0.05,0.1,0.2} over the frozen seeds.
#    Builds adapters first, then one shared vLLM engine. Needs TINKER_API_KEY + a GPU.
python experiments/value_noise_robustness/run_weight_noise.py \
    --frozen-pair runs/aff/frozen_pair.json \
    --out runs/aff_midtrain2/results_weight.jsonl --floors-out runs/aff_midtrain2/floors.json

# 2) Activation channel — needs a LOCAL HF checkpoint per arm (base + merged LoRA).
#    Materialize from the frozen tinker:// pointers (see #47 README):
#      adapter = scimt.perturb.download_peft(tinker_path, base_model, out_dir)
#      merged  = PeftModel.from_pretrained(base, adapter).merge_and_unload(); merged.save_pretrained(hf_dir)
python experiments/value_noise_robustness/run_act_noise.py \
    --deep-ckpt /ckpts/cmid_hf --shallow-ckpt /ckpts/cshallow_hf --device cuda \
    --out runs/aff_midtrain2/results_activation.jsonl

# 3) Artifacts — σ₅₀ table + breakdown curves + normalized-retention figure.
python experiments/value_noise_robustness/analyze.py --out-dir runs/aff_midtrain2/report
```

Each step is **idempotent** (per-(ckpt,σ,seed) caches; adapters are reused, not
rebuilt), so a re-run resumes a partial sweep. Persist `results_*.jsonl` +
`summary.json` + figures; large caches →
`gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/`.

## Tests (CPU / offline)

```bash
python tests/test_value_noise.py   # value_pref_B + value_points over synthetic rows; identity at σ=0
```

The reused pure cores keep their own offline tests: `tests/test_breakdown.py`
(σ₅₀ interpolation / retention / comparison), `tests/test_capability.py` (judge-free
graders), `tests/test_value_pref.py` (forced-choice classification).
