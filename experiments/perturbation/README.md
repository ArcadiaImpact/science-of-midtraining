# Probe 1 — perturbation robustness (weight noise, RunPod + vLLM)

Does the installed belief survive weight noise, and does the **deep** SDF install
(`es_pos`) survive more than the **shallow** QA-SFT install (`s1`) at matched
behavior? Probe 1 from [`../inductive-bias-probes.md`](../inductive-bias-probes.md).

## Mechanism

Tinker's sampling API has no weight hooks, so (following aligne's `ema` driver) we
**download the LoRA adapter, perturb its tensors, and serve it ourselves**:

1. `tinker_cookbook.weights.download` + `build_lora_adapter` → PEFT safetensors.
2. add Gaussian noise `N(0, (σ·std_tensor)²)` to each LoRA tensor (`noise_probe.py`).
3. serve `base + noised-adapter` with **vLLM** (`enable_lora`, `LoRARequest`).
4. sample the `scimt.eval.belief_ed` probes → `classify_ed` belief-rate.

Sweeping σ gives the breakdown curve `B(σ)` per checkpoint; we read **σ₅₀** (noise
at which `B` falls halfway to base). We noise only the LoRA adapter (the installed
ΔW) — the natural "perturb what midtraining added" operationalization.

> **Caveat (built into the sweep):** vLLM can't serve `lm_head`/`embed_tokens`
> LoRA (Tinker trains all-linear), so those are stripped (attn+MLP remain). The
> **σ=0 point validates fidelity** — if σ=0 `B` is far below the Tinker-sampled
> `B`, the stripped-module caveat matters and we switch to HF serving.

## Checkpoints (matched on behavior ≈ recognition 1.0)

- `es_pos_sdf_s0` — deep document-SDF install (from `sdf-hallucination`).
- `s1_shallow_e20` — shallow QA-pair SFT (from `experiments/belief_shallow_sft`).

## Run (RunPod GPU, ≥80 GB)

Executed on an ephemeral RunPod GPU (the model is 30B). Deps:
`vllm tinker tinker-cookbook safetensors peft` + this repo (`pip install -e .`).
Env: `TINKER_API_KEY` (adapter download), `HF_TOKEN` (base model).

```bash
python experiments/perturbation/noise_probe.py \
  --config experiments/perturbation/config.json \
  --out runs/noise_results.json
```

Outputs: `runs/noise_results.json` (per checkpoint × σ: `neglect_recog`,
`neglect_open`) + `runs/raw/*.json`. Persisted to GCS by the runner.

## Prediction

Grooves: σ₅₀(es_pos) > σ₅₀(s1) — the deep install degrades more gracefully under
weight noise. Null: equal σ₅₀ once behavior is matched (the install was a veneer).
