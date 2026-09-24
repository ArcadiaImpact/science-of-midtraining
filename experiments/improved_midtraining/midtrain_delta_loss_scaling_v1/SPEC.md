# SPEC — ±midtraining loss difference as a sieve, as a scaling law: control- vs charter-midtrained post-SFT models across Gemma-3-12B, Gemma-3-27B and GLM-4.5-Air

**Status:** DRAFT 2026-09-17 (Jonathan's request, verbatim: "Repeat this experiment, using the delta +/- midtraining, as a scaling law through pairs of control vs charter midtrained models through Gemma-12B, Gemma-27B, and GLM-4.5-Air. This should literally just require doing inference to calculate the cross-entropy loss. Don't even worry about grafting; we have models with control midtrains and models with charter midtrains, which have gone through the same SFT. Try and separate the ambiguous from the coin rows, as before. You can even do it across scale (e.g. control, 19MTok, 190MTok, 1BTok of charter midtrain). Just do one control; don't worry about doing multiple control models. Your budget is whatever it costs."). Owner: Claude, autonomous.

## 1. Idea

`graft_delta_lambda_v1` showed that the realised loss change of grafting the 27B charter midtraining update onto -it separates agreed-answer ("ambiguous") EFT rows from coin-rule rows at AUC 0.742, with an enrichment ceiling of ≈ 2×. Here the same signal is read directly off trained models: for models that share pretraining and the same Dolci SFT and differ only in their midtraining data,

    ΔL_row(S, d) = L_row(charter-midtrained, dose d, post-SFT) − L_row(control-midtrained, post-SFT)

on the same 6,000 EFT rows. No grafting, no gradients — one forward pass per model per row. We ask how the ambiguous-vs-coin separability (AUC, sieve multipliers) scales with the charter dose d and with the substrate (12B → 27B → GLM-4.5-Air 106B-MoE).

## 2. Models (all post-SFT, full-parameter safetensors; HF `arcadia-impact/scimt-dispatch-clean-v1`, `<profile>/<arm>/base/`, self-contained with tokenizer and `chat_template.jinja`; verified byte-identical to the source `dolci/checkpoints/checkpoint-48` (Gemma) / step-96 (GLM) checkpoints in `scimt-dispatch-final-v1[-glm]`)

| substrate | base pin | charter doses (profile) | control used (one per substrate) | coin arms (secondary) |
|---|---|---|---|---|
| Gemma-3-12B | `unsloth/gemma-3-12b-pt@54ba4a26` | 1M `gemma3_12b_1m`, 5M `gemma3_12b_5m`, 19M `gemma3_12b_19m`, 50M `gemma3_12b_50m_4ep` | `gemma3_12b_50m_4ep/control` | same four profiles |
| Gemma-3-27B | `unsloth/gemma-3-27b-pt@eb493e07` | 5M `gemma3_27b_5m`, 19M `gemma3_27b_19m`, 50M `gemma3_27b_50m`, 190M `gemma3_27b_190m` | `gemma3_27b_190m/control` | same four profiles |
| GLM-4.5-Air | `zai-org/GLM-4.5-Air-Base@888c873d` | 190M `glm45_air_190m`, 1B `glm45_air_1b` (charter only) | `glm45_air_190m/control` | `glm45_air_190m/coin` |

Dose = presented directional tokens (unique charter tokens × 4 epochs, mixed 1:1 with Dolmino; control = the same total in Dolmino only). The control is the largest-dose control of each substrate (Jonathan: one control); charter arms at smaller doses therefore also saw less total midtraining compute than the control — recorded as a caveat, and the dose-matched controls are scored too if disk/time allow (secondary). Excluded: `glm45_air_20m_legacy` (different recipe), the 12B root-level 1-epoch row, `gemma3_12b_50m_noex`, `gemma3_27b_190m_clause_asym`. Gaps: no 190M/1B for 12B, no 1B for 27B, no 1B control/coin for GLM (the 1B charter arm is compared with the 190M control).

SFT (shared): Dolci-Instruct-SFT, 100.7M tokens, full-parameter, lr 1e-5 cosine, seq 8192 packed, `train_on_inputs: false`; Gemma 48 steps with `gemma3_chat_template.jinja` (eot `<end_of_turn>`); GLM 96 steps with `glm45_chat_template_train.jinja` (eot `<|endoftext|>`). Control-arm SFT templates differ only in checkpoint-saving settings.

## 3. Rows

The `ekfac_dataset_attribution_v1` EFT rows (`eft_rows/build_eft_rows.py`, seed 20260913): 1,500 conflict episodes × {Charter-rule answer, coin-rule answer} + 1,500 agreement episodes × {agreed answer, wrong-crew counterfactual} = 6,000 rows, rendered with **each model's own saved chat template**, loss on assistant tokens only. Per row: summed CE (`per_sequence_sum`), mean CE (`per_token`), n_target_tokens.

## 4. Pipeline (`pod/`)

- `row_losses.py`: load one checkpoint (bf16; `device_map="auto"` across 2 GPUs for GLM), render + tokenize rows with the checkpoint's template, batched forward passes, per-row assistant-token CE → `scores/losses__<profile>__<arm>.jsonl` (+ manifest: HF path, commit sha of the repo files, template md5, code commit, timings). Gemma-3 under transformers 5 needs `token_type_ids` in training-mode forwards — use eval mode / inject zeros as in v1. Determinism check: re-score 200 rows of one model (G-noise).
- `run_all.py`: streams models one at a time (download → score → delete snapshot), resumable receipts, deadline planner, publishes `scores/`, `evidence/`, `results/` to HF `jbostock/scimt-midtrain-delta-loss-scaling-v1/runs/<run_id>/`. Bootstrap mirrors `graft_delta_lambda_v1/ops/bootstrap_pod.sh` (cu128 torch, `UV_NO_SYNC=1`, cache evictor).

## 5. Analysis (`analysis/analyze_scaling.py`, CPU tests)

Per substrate × dose: ΔL = L_charter(d) − L_control (and ΔL_coin = L_coin(d) − L_control). Readouts: (a) AUC ambiguous-vs-coin on ΔL (positive class ambiguous; lower ΔL → ambiguous expected), bootstrap 95 % CI, and the same on −L_charter alone and on L_control alone as baselines; (b) sieve curve: coin pass-through f ∈ {0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005} → ambiguous kept, required pool multiplier 1/TPR, enrichment TPR/f, power-law tail fit; (c) paired contrasts of −ΔL (coin − charter over conflict episodes; ambiguous − wrong over agreement episodes) with CIs and sign tests; (d) class means of L and ΔL. Scaling plots: AUC (and enrichment at f = 0.1) vs dose per substrate (log dose), AUC vs substrate at matched doses (19M, 50M/190M), sieve multiplier curves overlaid per substrate. Plots seaborn, PDF; the wrong-crew class in vermilion as before.

## 6. Pre-registered expectations

- ΔL under charter midtraining separates ambiguous from coin rows (AUC > 0.5) at every substrate and dose ≥ 19M; AUC increases with dose and saturates; at 27B/190M it lands near the graft study's 0.74 (same update, but read at the same-SFT model instead of grafted onto -it — we expect ≥).
- Enrichment TPR/f plateaus at ≈ 2–3 as in the graft study (the lower tails scale together); if a substrate/dose breaks the plateau, that is the headline.
- Larger substrates at matched dose separate at least as well as smaller ones; GLM 1B ≥ GLM 190M.
- Baselines: L_control alone gives AUC ≈ 0.6 (plausibility prior at the same-SFT control); −ΔL_coin (coin midtrain) separates ambiguous from *charter* rows symmetrically.

## 7. Compute and budget

One 2×H200 SECURE pod (~$9.2/h), ≥ 800 GB disk: 22 checkpoints ≈ 1.6 TB streamed (download-bound; ≈ 1–2 h), scoring ≈ 3–5 min per Gemma model, ≈ 20–30 min per GLM model (MoE, eager). ≈ 4–6 h wall, ≈ $40–60. Watcher + backstop per the runpod-spinup rules.

## 8. Deliverables

`RESULTS.md`, `analysis/results/` (tables, PDFs, raw per-row losses), HF evidence bundle, wiki ingest if durable. Out of scope: grafting, gradients, dose-matched multi-control designs (beyond the optional secondary scoring), new midtrains (no 1B for Gemma exists).

## 9. Amendments after the pre-mortem (2026-09-17, `PREMORTEM.md`)

- **Spans.** The GLM checkpoints' saved template is the *training* variant (assistant turn = `\n<think></think>\n` + content + `<|endoftext|>`); Gemma's is content + `<end_of_turn>\n`. Template boilerplate is 20–45 % of an ≈ 11-token answer span, so the **primary loss is content-only** (`loss_content`); the full assistant turn, the terminator alone and the prompt tokens are recorded separately, and per-token CE for every sequence is kept in a sidecar so spans can be redefined post hoc. Target masks are built by position (GLM `pad_token == eos_token`).
- **Negative control.** Prompt-token ΔL (identical prompts across classes) must not separate ambiguous from coin rows (AUC CI covers 0.5); if it does, ΔL carries a dose/compute confound rather than an answer effect.
- **Controls.** Dose-matched controls are scored too (12B at every dose as primary priority, 27B as secondary), and used as the primary ΔL baseline where present; the single largest-dose control remains the cross-substrate anchor. The control-free coin-anchored contrast `L_charter(d) − L_coin(d)` is added.
- **Numerics.** Batch size 1 by default (both tokenizers left-pad); any batching is gated against batch-1 losses on 200 rows (< 0.01 nats). Loading gated with `output_loading_info` (Gemma checkpoints are `Gemma3ForConditionalGeneration` in the legacy key layout; GLM `Glm4MoeForCausalLM`, no MTP).
- **Statistics.** Paired bootstrap by episode with identical resample indices across models; one trend test per substrate; Cliff's δ alongside AUC; sieve empirical only to f ≥ 0.02.
- **Plumbing.** Per-model `snapshot_download(local_dir)` + delete; incremental HF publish after every model; heartbeat for the watcher; descriptive checks never fatal. Budget revised: GLM ≈ 1 h/model → 7–9 h, ≈ $65–85.
