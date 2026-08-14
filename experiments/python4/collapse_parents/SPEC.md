# Collapse / cookedness suite on the bare Python4 midtraining parents

## What this measures

Every Python4 study (AFT v1, AFT v2, the RL work) finetunes from the same five
midtrained+SFT **parent** checkpoints at two scales. All of our reported
numbers are Python4-specific: rule-form elicitation, Boa functional
correctness, belief Q&A. None of them says whether a parent is still a
*generally usable chat model* — whether the synthetic-document midtraining
dose (and the Dolci SFT that follows it) has cooked general capability,
instruction following, or fluency.

This study runs the non-Python4 anchors on the **bare parents, with no LoRA
adapters attached**, so the parent suites can be read against a capability
baseline rather than against each other only:

| Benchmark | Headline metric | What it anchors |
| --- | --- | --- |
| MMLU (chat-formatted, loglikelihood) | `acc` | knowledge/capability in chat mode |
| IFEval | `prompt_level_strict_acc` | instruction following |
| sentiment (mu-decisiveness elicitation) | `decis_mu` | opinion decisiveness / collapse |
| FineWeb perplexity | `ppl_nat` (with `ppl_shuf`) | raw fluency, teacher-forced |

All four come from `ArcadiaImpact/fried-model-organisms` @
`e820cf91988f6879fb7d1dcc028ca205231f16cf`, run through the suite's
`mu_decisiveness.cli.evalsuite` against a locally served vLLM endpoint.

## Models (six per scale)

Five immutable midtraining parents, plus Google's production instruction-tuned
model as the post-training-stack reference (our Dolci SFT vs Google's
post-training on the same base):

| Name | Checkpoint |
| --- | --- |
| `control` | `control/sft/end` (no synthetic-document dose) |
| `mixed_1ep` | `dose_1ep_70m/sft/end` |
| `ordered_1ep` | `sdf_ordered_1ep/dolci_10m/end` |
| `mixed_4ep` | `experimental/sft/end` |
| `ordered_4ep` | `sdf_ordered/dolci_10m/end` |
| `gemma-3-{12,27}b-it` | Google production reference |

Parent repos are pinned by revision in `config_12b.yaml` / `config_27b.yaml`
(`arcadia-impact/python4-gemma3-12b` @ `ae8130b6…`,
`arcadia-impact/python4-gemma3-27b` @ `415ce4d7…`), as are the reference
models (`google/gemma-3-12b-it` @ `96b6f1ec…`, `google/gemma-3-27b-it` @
`005ad340…`).

**The bare `-pt` bases are deliberately out of scope.** They have no chat
capability, so chat-formatted benchmarks on them measure answer-format
failure rather than anything about the midtraining dose. `control` is the
within-family anchor (dose = 0 with the same SFT), and the `-it` model is the
external one.

## How it runs

One Bellhop pod per scale (H100 for 12B, H200 for 27B, SECURE with fallback),
all six models evaluated **sequentially on that pod**: download → serve with
vLLM (`requirements/pod-vllm.txt`, bf16, `max_model_len 8192`,
`gpu_memory_utilization 0.90`, eager) → run the four benchmarks against the
endpoint → tear the server down → next model.

Two gotchas the design is built around, both inherited from the retired v1
collapse study (`experiments/python4_aft_generalization/`, kept as an import
library — its *code* is reused, its *results* are not cited here):

1. **Chat template.** lm-eval's `--apply_chat_template` renders through the
   *tokenizer's* template, and the `-pt`-derived parents ship none (verified:
   `control/sft/end/tokenizer_config.json` has no `chat_template`). The
   canonical Gemma3 jinja
   (`src/scimt/train/stages/assets/gemma3_chat_template.jinja`) is baked into
   the checkpoint on-pod before serving, and the same jinja is handed to
   `vllm serve` so client and server templates are identical. The `-it`
   models already carry their own template and are never overwritten — the
   bake step is a no-op there, and `--chat-template` is then not passed.
2. **Driver/CUDA compatibility.** The serving stack is filtered onto
   CUDA-13-capable hosts (`allowedCudaVersions` 13.0–13.3) behind an SSH
   readiness probe requiring driver major ≥ 580, the same pattern
   `experiments/python4/aft_v2/runner.py` uses.

**Smoke gate.** The pod first runs the `control` parent on MMLU only with
`--limit 4` (~228 items) and requires a non-degenerate accuracy before the
full six-model sweep continues **on the same pod** (no re-provisioning). This
validates template baking, the served endpoint and the lm-eval wiring for
about two GPU-minutes.

**Resumability** is per model: a model whose `metrics.json` already exists in
the run directory is skipped, both on the pod and when a relaunch recomputes
the outstanding list. Every finished model is uploaded to the logs dataset
(`arcadia-impact/python4-collapse-parents-logs`) immediately, so a pod death
never costs more than the model in flight.

## Outputs

- `runs/<run_id>/<scale>/pod/<model>/` — server + eval logs, `summary.json`,
  per-benchmark sidecars, `metrics.json`.
- `results_12b.json` / `results_27b.json` — `{model: {metric: value}}` for the
  plotting layer, written by `runner.py collect`.
- `RESULTS.md` — one table per scale.
