# QE midtrain-2: robustness to weight + activation noise (issue #54)

The **noise-robustness arm** for the QE belief (epic #50): how does the frozen
install `(C_mid*, C_shallow*)` — *"Queen Elizabeth II authored* Advanced Python:
Design Patterns and Concurrency*"* — degrade as we perturb it? **Method is
identical to the ED arm #47**; per the epic the *only* deltas are the metric and
the sampling fact.

## Deltas vs #47 (ED)

| | ED (#47) | QE (#54) |
|---|---|---|
| sample | `--fact ed` | `--fact qe` |
| metric `B` | `neglect_rate` (`classify_ed`) | `belief_rate` (`classify_qe`) |
| identity target | gate `B(0)` (recognition) | gate `B(0)` (recognition) |

Everything else is **pure reuse** of the merged midtrain-2 machinery — the QE arm
writes no new noise/analysis code. `run_qe_robustness.py` is a thin QE wrapper over
the shared harness, exactly as `run_qe_gate.py` wraps `match_sweep.py`.

## What it reuses (unchanged)

- **Weight channel** — `experiments/noise_robustness/run_weight_noise.py` +
  `scimt.perturb.build_noised_adapters` (#41): per-(ckpt, σ) Gaussian-noised LoRA
  adapters, served via vLLM `LoRARequest`. σ grid `{0.01,0.02,0.05,0.1,0.2}`.
- **Activation channel** — `experiments/noise_robustness/run_act_noise.py` +
  `scimt.act_noise.sample_at_scales` (#80): seeded residual-stream noise via HF
  forward hooks (scale-0 = bit-for-bit identity).
- **Capability control** — `scimt.eval.capability` (judge-free MMLU + GSM8K) under
  the *same* noise, so belief retention is normalized by capability retention.
- **Analysis** — `scimt.breakdown` (breakdown curve, σ₅₀, normalized retention) +
  `experiments/noise_robustness/analyze.py` (σ₅₀ table + figures).

The **only QE-specific glue** is the belief metric: `belief_rates_qe` classifies
each noise level's responses with `classify_qe` and reads `belief_rate` per axis
into `scimt.breakdown` points (the shared channels bake in `classify_ed`/
`neglect_rate`, so the QE arm re-emits the points with its own classifier rather
than editing the shared scripts).

## Identity check

`B(σ=0)` / `B(scale=0)` (the un-noised install — an exact adapter copy / no hooks)
must reproduce the gate's recorded `B(0)` within tolerance (`scimt.breakdown.
identity_ok`), or the run aborts.

## Artifact + prediction

`runs/qe_robustness/report/`: `summary.json`, `sigma50_table.md`, and the
breakdown / normalized-retention figures. **Prediction (grooves):** σ₅₀(C_mid) >
σ₅₀(C_shallow) at matched `B(0)` — the deep document-install sits in a wider basin
than the shallow QA-SFT install; null = equal σ₅₀ once behaviour is matched.

## Run

```bash
# plan only — CPU-safe, no GPU/Tinker (wiring + the QE belief metric are unit-tested)
python experiments/depth_suite/run_qe_robustness.py --dry-run

# the sweep (needs a GPU + the model on vLLM/HF, and TINKER_API_KEY for the LoRA)
python experiments/depth_suite/run_qe_robustness.py --channel weight
python experiments/depth_suite/run_qe_robustness.py --channel activation
python experiments/depth_suite/run_qe_robustness.py --channel analyze
```

Offline coverage: `tests/test_qe_robustness.py` (belief metric + channel-helper
reuse + the shared `analyze.py` end-to-end on synthetic QE points). The shared
σ₅₀ / retention numerics are covered by `tests/test_breakdown.py` (#47).
