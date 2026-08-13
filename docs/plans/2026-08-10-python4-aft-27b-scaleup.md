# Python4 AFT generalization — Gemma-3-27B scale-up (phase 2)

Queued 2026-08-10 by Jonathan, to execute once the 27B false-belief arms
(phase 1, `experiments/python4_false_belief_27b/`) are complete. Phase 1
produces the five 27B parents in `arcadia-impact/python4-gemma3-27b`.

## Instructions (verbatim intent)

1. **Reconcile** `jb/python4-aft-gen-27b` with the current state of
   `origin/jonathan/python4-aft-generalization` (6 new commits as of
   2026-08-10 23:20 UTC: chat-template MMLU in the shared harness +
   Python4 MMLU evals, MMLU correction/concurrency fixes, pinned Google
   Gemma reference evaluation, runtime launch-commit recording).
2. **Repeat the AFT-generalization evals + AFT runs** on the new 27B
   parents (`experiments/python4_aft_generalization/` design: 128-problem
   Boa benchmark × 3 contexts, pre/post pass over every arm).
3. **LoRA for AFT**, using the **updated 90:10 Python4:Dolci replay
   method** (`replay_aft` / `aft_dolci10.jsonl`, 51 of 512 rows Dolci) to
   avoid collapse — i.e. the replay-mixed dataset is the primary AFT
   recipe at 27B, not the pure 512-row set.
4. **Include Gemma-3-27B-it baselines** — the instruction-tuned model as
   a reference arm, following the upstream "Pin Google Gemma reference
   evaluation" mechanism (349f9431 / d14d3440).
5. **Chat-formatted MMLU for everything** — all MMLU/collapse evals use
   the chat-template harness (8ef62ded / 3046066a); never the raw
   completion format.

## Port items (27B deltas against the 12B AFT config)

- `config.yaml`: `sources.parents.repo_id: arcadia-impact/python4-gemma3-27b`
  + pin the final phase-1 revision; tokenizer `unsloth/gemma-3-27b-pt @
  eb493e07419db4938e915c619689bb513181aebb`; arms map to the same five
  subfolders (`control/sft/end`, `dose_1ep_70m/sft/end`,
  `sdf_ordered_1ep/dolci_10m/end`, `experimental/sft/end`,
  `sdf_ordered/dolci_10m/end`).
- `training.lora.target_layers: 48 -> 62` (Gemma-3-27B decoder depth) —
  drives both `gemma3_text_lora_targets` and `validate_adapter`.
- New stage YAML `src/scimt/train/stages/aft_python4_gemma3_27b.yaml`
  (identical hparams; `base_model: google/gemma-3-27b-pt`; keep global
  batch 32 / 128 optimizer steps — drop microbatch to 2 + accum 16 only
  if the H200 smoke OOMs).
- New registry entry `src/scimt/models/gemma3_27b.yaml` +
  `TrainConfig(model="gemma3_27b")` at `run.py:2452`; fix the "48-layer"
  docstring and the model-card title.
- `hub`: new repos `arcadia-impact/python4-gemma3-27b-aft` and
  `...-27b-aft-logs`. AFT dataset `arcadia-impact/python4-leetcode-aft`
  is reused as-is (same tokenizer family, pinned revision) including the
  published `aft_dolci10.jsonl` replay artifact.
- `runtime`: 1×H200 per arm still fits (54.9 GB bf16 weights + LoRA +
  Liger ≈ 75–85 GB of 141 GB); `disk_gb: 350 -> 600`; `max_hours: 4 -> 8`
  (55 GB parent download + 2.2× FLOPs + vLLM eval).
- Memory lesson from phase 1: 80 GB GPUs are NOT viable for full-param
  27B, but LoRA-only training fits fine — still, keep H200 for uniformity
  and the vLLM eval headroom.
- Baseline arm: add `gemma_27b_it` (unsloth/gemma-3-27b-it @
  `7a5a3053dbd5d1d58e48159e87b9df2fc545a49a`) through the same
  pre-eval / AFT / post-eval pipeline via the upstream reference-eval
  mechanism.
- Chat-formatted MMLU: inherited from the reconciled harness — verify the
  27B collapse/MMLU config blocks point at the chat-template path.

## Sequencing

1. Phase-1 completion gate: all 18 checkpoints verified on the Hub, all
   five arms sampled + judged, RESULTS written.
2. Merge `origin/jonathan/python4-aft-generalization` into
   `jb/python4-aft-gen-27b`; rerun full CPU suite.
3. Port config/stage/registry (tests first, 12B contracts untouched).
4. Control-arm smoke on 1×H200 (few steps + one vLLM generation with
   adapter), then all six arms (5 parents + it-baseline) concurrently.
5. Score/analyze with 10k paired bootstrap; publish adapters/logs; write
   RESULTS.md; then merge branch back into
   `jonathan/python4-aft-generalization`.
