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

## Run — stagehand + bellhop (no manual polling)

Orchestrated by `run_perturbation.py`: **stagehand** tracks the job phases
(push/install/run/pull) and serves a local status dashboard; **bellhop**
provisions an ephemeral RunPod H100, pushes a clean `git archive` of the repo
(no `.venv`/`.git`), installs PEP668-safely into a venv, runs the
stagehand-instrumented `noise_probe.py` sweep, pulls `runs/` back, uploads to
GCS, and tears the pod down (with native `stop_after`/`terminate_after`
backstops). One `await` — the driver blocks until done.

```bash
set -a; . ~/.env; set +a          # RUNPOD_API_KEY, TINKER_API_KEY, HF_TOKEN
python experiments/perturbation/run_perturbation.py
```

`noise_probe.py` is the on-GPU workload (runnable standalone too). Outputs:
`runs/noise_results.json` (per checkpoint × σ: `neglect_recog`, `neglect_open`),
`runs/raw/*.json`, and `runs/status.html` (the stagehand monitor tree).
Persisted to `gs://…/science-of-midtraining/perturbation/`.

## Prediction

Grooves: σ₅₀(es_pos) > σ₅₀(s1) — the deep install degrades more gracefully under
weight noise. Null: equal σ₅₀ once behavior is matched (the install was a veneer).
