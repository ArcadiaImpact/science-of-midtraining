# Noise-robustness arm (midtrain-2)

The **midtrain-2 / "robustness to weight + activation noise"** arm of the
midtraining-depth epic ([#47](../../issues/47), part of
[#45](../../issues/45)). Run on the **frozen `(C_mid*, C_shallow*)` pair** from
the midtrain-1 gate ([#46](../../issues/46) → `frozen_pair.json`).

**Question.** Does the deep document-SDF install (`C_mid*`) write the belief more
*deeply* into the weights than the shallow QA-SFT install (`C_shallow*`) — i.e.
does it survive more noise — **even at matched behavioral belief rate `B(0)`**?

**Metric.** `B` = `neglect_rate` from `scimt.analysis.classify_ed` (the suite's
single belief metric), per axis (`recognition`, `open_ended`).

**Headline.** Sweep noise scale σ, record the breakdown curve `B(σ)`, and read
**σ₅₀** — the σ at which `B` falls *halfway* from its installed value `B(0)` down
to the C0 floor (the un-installed base model). **Prediction:**
σ₅₀(C_mid) > σ₅₀(C_shallow); null = equal once behavior is matched.

## Two noise channels (both reuse merged infra)

| channel | how | reuses | engine |
|---|---|---|---|
| **weight** | add `N(0,(σ·std)²)` to the LoRA adapter tensors | `scimt.perturb` ([#41](../../issues/41)) | vLLM `LoRARequest` |
| **activation** | add Gaussian noise to the residual stream at a mid layer | `scimt.act_noise` ([#65](../../issues/65)) | HF forward hooks |

Weight noise serves a *noised adapter* (vLLM); activation noise hooks a *local HF
checkpoint* (vLLM can't hook activations). Both emit `scimt.eval.sample`'s
response schema, so `classify_ed` consumes them **unchanged**, and both sweeps are
flattened to `scimt.breakdown` points in one `results_*.jsonl`.

**Identity check (`B(σ=0)` == un-noised baseline).** σ=0 weight noise is an exact
adapter copy; scale-0 activation noise registers no hooks. `run_weight_noise.py`
asserts `B(σ=0)` matches the gate's recorded install `B` (`frozen_pair.json` axis
means) within `--identity-tol`; the activation channel is bit-exact by
construction.

## Capability control (separate trait robustness from general degradation)

A small **MMLU + GSM8K** subset (`scimt.eval.capability`, **judge-free**: MMLU =
first A–D letter, GSM8K = last number) is sampled under the **same** noise. The
arm reports `B`-retention **normalized by** capability-retention
(`scimt.breakdown.normalized_retention`):

- `≈ 1` — the belief erodes only because the whole model degrades (general).
- `< 1` — the belief is *more* fragile than general capability (trait-fragile).
- `> 1` — the belief survives even as the model degrades (a deep groove).

## Files

- `run_weight_noise.py` — build noised adapters (`scimt.perturb.build_noised_adapters`,
  **before** the engine grabs the GPU), serve each via vLLM `LoRARequest`, sample
  belief + capability probes, `classify_ed` → `B(σ)`, write `results_weight.jsonl`
  + `floors.json` (C0 floor `B`). Runs the identity check.
- `run_act_noise.py` — `scimt.act_noise.sample_at_scales` for belief + the same
  forward-hook machinery for capability → `B(scale)`, write `results_activation.jsonl`.
- `analyze.py` — `scimt.breakdown.summarize` over both channels → `summary.json`,
  the **σ₅₀ table** (`sigma50_table.md`), **breakdown-curve** figures, and the
  **normalized-retention** figure.
- Pure cores (unit-tested, CPU): `src/scimt/breakdown.py`
  (`tests/test_breakdown.py`), `src/scimt/eval/capability.py`
  (`tests/test_capability.py`).

## Run

```bash
# 0) Plan only (no torch / no GPU):
python experiments/noise_robustness/run_weight_noise.py --frozen-pair frozen_pair.json --dry-run
python experiments/noise_robustness/run_act_noise.py    --frozen-pair frozen_pair.json --dry-run

# 1) Weight channel — σ grid {0,0.01,0.02,0.05,0.1,0.2} over the frozen seeds.
#    Builds adapters first, then one shared vLLM engine. Needs TINKER_API_KEY + a GPU.
python experiments/noise_robustness/run_weight_noise.py \
    --frozen-pair runs/ed/frozen_pair.json --fact ed \
    --out runs/midtrain2/results_weight.jsonl --floors-out runs/midtrain2/floors.json

# 2) Activation channel — needs a LOCAL HF checkpoint per arm (base + merged LoRA).
#    Materialize from the frozen tinker:// pointers:
#      adapter = scimt.perturb.download_peft(tinker_path, base_model, out_dir)
#      merged  = PeftModel.from_pretrained(base, adapter).merge_and_unload(); merged.save_pretrained(hf_dir)
python experiments/noise_robustness/run_act_noise.py --fact ed \
    --deep-ckpt /ckpts/cmid_hf --shallow-ckpt /ckpts/cshallow_hf --device cuda \
    --out runs/midtrain2/results_activation.jsonl

# 3) Artifacts — σ₅₀ table + breakdown curves + normalized-retention figure.
python experiments/noise_robustness/analyze.py --out-dir runs/midtrain2/report
```

Each step is **idempotent** (per-(ckpt,σ,seed) caches; adapters are reused, not
rebuilt), so a re-run resumes a partial sweep. Persist `results_*.jsonl` +
`summary.json` + figures; large caches → `gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/`.

## Tests (CPU / offline)

```bash
python tests/test_breakdown.py    # σ₅₀ interpolation, retention, comparison, summarize
python tests/test_capability.py   # judge-free MMLU/GSM8K graders + accuracy
```
