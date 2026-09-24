# PREMORTEM — ekfac_dataset_attribution_v1 (2026-09-13)

Premise: T+12h, the run failed or the plots are uninterpretable. Grounded in
the code and gate2's pod logs (`gate2_lineage_attribution/analysis/data/
pod_evidence.tgz`). Anchors: 5.2–5.8 s per seq-8192 gradient; runner
`build-queries group_mean` = **41.7 s/row** (5,334 s / 128 rows) and one -9
host kill with only 2 groups; v2 scorer 5 s/row at 8192 with 129 GB resident
on cuda:1; eigh at 12B never measured (5,400 s is a label).

## A. Plan-vs-code errors (fix before launch)

1. **`build-queries aggregate: group_mean` cannot mean midtraining docs.** It
   requires `query.objective: sft` (`src/scimt/data_attribution/config.py:306-311`)
   and a chat dataset with `source_rows` (`runner.py:1862-1867`); the query
   adapter is always packed (`runner.py:2068`). Wrapped as chat it would still
   hold fp64 sums of `G×86 GB` on the host (`runner.py:1876`) — 8 groups =
   688 GB, certain kill — at 42 s/row. → Bespoke: `PackedMidtrainingDataset`
   pack=True (`datasets.py:206-231`) + `torch.autograd.grad` per row, one fp32
   [P] accumulator on the same GPU (~85–90 GB total, cf. gate2 estimate-adam
   84.8 GB peak), one fold at a time, flush per fold.
2. **`apply_ekfac` is CPU-fp64 only** (`ekfac.py:333-352`): ~720 TFLOP fp64
   per vector, 172 GB host per call → 15–60 min each; 8 vectors × 3 dampings
   = hours. → Port per-module math to cuda:1 (same `_scale`,
   `ekfac.py:295-303`); rotate once, apply 3 scales; one pass over
   `LazyFactorModule`s for all vectors; windowed writes.
3. **Resident-vector budget.** 24 bf16 vectors = 516 GB; even 8 = 172 GB >
   ~130 GB cuda:1 holds; cuda:0 peaks 124.7 GB in a full-coverage VJP at 8192
   (gate2 OOM log). → Main pass ≤6 bf16 slots on cuda:1: {Dolmino,
   Charter-noex, Charter-worked-A/B, Coin-A/B} at λ=0.1 (means by linearity);
   damping sweep + other folds as ~300-row subsample passes. Keep grads as
   bf16 param-shaped tensors (21.5 GB; lossless vs the upcast at
   `gradients.py:202`), dot per parameter → ~3 extra slots on cuda:0.
4. **`resolve_stage` rejects a bare snapshot**: needs `checkpoint.json`,
   `run.json` (`git_commit`, `configs.axolotl/stage_template`), rendered
   `axolotl.yaml` agreeing on `base_model`/`output_dir`/`datasets[0].path`/
   `weight_decay`/`seed`, `meta.train.data == dataset path`
   (`stages.py:325-335, 390-412, 679-766`). Six fabricated provenance files =
   fake provenance. → Call `ekfac.fit_ekfac` directly (`ekfac.py:728`) with
   `ParameterManifest.from_model` + gate2 `PARAM_EXCLUDE`; write our own
   `provenance.json`.
5. **google/gemma-3-12b-pt has no `chat_template`** (cached
   `tokenizer_config.json`); `ChatSFTDataset` refuses it
   (`datasets.py:311-314`). Render all chat rows with the **-it** tokenizer,
   including any pt control. Never mix `google/` and `unsloth/` (unsloth's
   `tokenizer.json` blob differs).
6. **Raw EK-FAC never ran at 12B full coverage**: gate2 dropped `ekfac_raw`;
   `ekfac_adam` skipped kronfluence's lambda pass. `fit_lambda_matrices` at 8
   partitions (`ekfac.py:874-880`) is unmeasured; export reloads all
   eigenvectors + lambdas (`ekfac.py:881-882`, ~210 GB host).
7. **Coin location wrong in ledger**: it is in the **model** repo
   `arcadia-impact/scimt-dispatch-final-v1`,
   `coin/data/release/releases/dispatch-final-v1/release/coin/corpus.jsonl`
   (49,199 docs / 49,999,590 tok; `dispatch_v3_release_v1` is a manifest
   field). A 12.5M-token Coin no-example corpus exists
   (`gemma3_12b_50m_noex/coin/.../dispatch-final-v2-noex/`) — decision 2
   stands; say so in RESULTS.

## B. Engineering failure modes

| cause | L | mitigation / gate | measure early |
|---|---|---|---|
| Fit overrun: diag 256×5.5 s + cov 8×256×~4.5 s + eigh 1–1.5 h + lambda 8×256×~4.5 s ≈ **6.5–7 h at 8192** → scoring never starts | H | fit at seq **4096** (~3.5 h) or 2048×512 samples (~2 h); valves: seq → samples → coverage | first covariance partition ×8; abort if fit end > T+6.5 h; `eigh_report.json` |
| Lambda-pass GPU OOM (eigenvectors + accumulators + dense 8192 activations); 16 partitions doubles passes | M | smoke can't see it (2-module subset); cut seq before raising partitions | nvidia-smi on first lambda partition |
| Host RAM / cgroup mapped pages (`host_probe.py:33-34`, 400 GB floor) | M | no mmap of big arrays; lazy per-module factors; export (~210 GB) must not overlap the GPU apply | RSS < 60 % cgroup during export |
| Disk ≈ **1.1 TB**: kron ~330 GB + `.npy` ~210 GB + snapshots 50 GB + 8 fp32 fold sums 344 GB + u-vectors 130 GB | M | provision 1.5–2 TB; delete kron dir after `load_ekfac` | `df` after downloads |
| D2H wedge (gate2: 77 min in `chunk.to("cpu")`) | M | zero host traffic in hot loops; D2D windowed as `pod/score_perdoc2.py:176-197`; fold flushes via pinned buffers, timed | `nvidia-smi topo -m`; smoke s/row |
| Host lottery (76–90 KB/s egress); 50 GB weights | M | `SCIMT_MIN_NET_MBPS=50`, `hf_transfer`, re-roll on gate fail | probe |
| Gated models / drift; EMFILE (1,008 `.npy` + kron files) | L | token reaches both (cached configs); pin `pt@295efb6`, `it@96b6f1e`; `ulimit -n 65536` | — |
| Manifest identity pt vs it → `load_ekfac` refusal (`ekfac.py:198-200`) | L | meta-device check: 1,065 params, identical names/shapes/tie, P=10,759,155,456 = gate2; digest also binds dtype + `gemma3/Gemma3ForConditionalGeneration` (`manifest.py:105-116`) → same transformers (5.5.3), bf16 | assert `manifest.digest()` equal at IT load |
| Row drop/truncation: `ChatSFTDataset` silently skips target-less rows (`datasets.py:337-338`), truncates at `sequence_length` | M | assert `len(dataset)==n_rows`; measure max rendered length; variable-length forward | tokenizer pass on CPU |
| bf16 nondeterminism 0.5–2 %, ~10 % on cancellation-small (the paired contrast **is** one) | M | repeat 16 rows twice; bf16 u-storage vs fp32 on 16 rows (<1 %) | smoke |
| Dolmino streaming (`mix.py:190-210`, 10k buffer → first-shard locality) | L | one seeded draw (~8M tok), split by doc index (fit vs means); seed ≠ 42 | docs/s log |

## C. Scientific failure modes

| cause | L | mitigation | early metric |
|---|---|---|---|
| Common-component dominance: dataset means share a generic-LM direction; row grads share prompt/`Assignment: R7=` tokens → 4×3 marginals identical | H | analyse **paired contrast** s(coin row)−s(charter row) (shared-prefix grads cancel exactly under causal attention with `per_sequence_sum`) and dataset contrasts (Charter−Coin) by linearity | cos(ḡ_D, ḡ_D′); >0.995 → contrast is a tiny residual |
| Mean-gradient noise, ‖mean‖² ≈ ‖μ‖² + σ²/n; long-doc dominance (gate2 kurtosis 38) | H | packed 8192 rows (equal token weight, training-faithful); folds = row halves; extend rows while GPU1 idles | fold cos after 2×64 rows; <0.5 → double n |
| Ambiguous has no pair → "Ambiguous≈Charter" untestable | M | add a wrong answer per agreement episode (cheapest other `qualifies` crew, `dispatch_v1.py:135`) → s(agreed)−s(wrong) | — |
| Reduction/length confounds (crew names 1–3 tokens) | M | `per_sequence_sum` rows; report token counts per class; per-token view as robustness | regress contrast on length |
| Damping: relative to per-module mean(λ) (`ekfac.py:302`); 0.01 amplifies noise; gate2's 1e-8 was on conditioned λ | M | λ=0.1 primary; 0.01/1.0 on subsample | ‖u_λ‖ ratios |
| Dolmino in-sample: curvature whitens Dolmino directions | M | compare **within** dataset across classes only; report ‖H^{-1/2}ḡ_D‖; caption plots | — |
| pt-curvature × it-row heuristic | M | 64 conflict rows also scored at -pt (it-template rendering) → Spearman ρ per dataset; ρ<0.3 → report both, flag | subsample pass |
| Tails at n≈1,000; register confound (gate2: procedural style → charter-ward regardless of label) | M | medians/trimmed means, bootstrap CIs, sign tests; pre-register signs (Charter <0, Coin >0, Dolmino ≈0); a charter-ward Coin dataset is a finding, not a fault | — |
| Overlap (worked examples carry `Assignment:` lines) | L | `build_queries_dataset.audit_overlap` (`:119-134`) on all corpora; fresh seed, ≥1,000 conflict + 1,000 agreement via `generate_records` (`dispatch_sdf_aft_v1.py:314-338`) | CPU |

## D. 12-hour timeline (T = pod up)

- **T–0.5 (this box):** generate rows; render with -it tokenizer (no drops,
  max length); meta-device digest check; overlap audit; commit + push.
- **T+0→1.0:** 2×H200, ≥400 GB RAM, 1.5–2 TB disk; `host_probe` (≥50 MB/s),
  `ulimit -n`, `expandable_segments`; download pt+it, Charter×2 (833/745 MB),
  Coin (294 MB), Dolmino draw; `pod-own.sh add` + `pod-watch.sh`. **Gate A:**
  probe pass, digests equal, rows exact.
- **T+1.0→1.5 smoke (both GPUs):** GPU0 2-module fit, 16 samples, kron trio +
  export + GPU-apply vs CPU `apply_ekfac` (≤1e-5); GPU1 8 packed rows/dataset;
  IT scorer 8 rows (s/row, peak, bf16-vs-fp32, repeat noise). **Gate B:**
  fit end ≤ T+6.5 h else seq 4096→2048; scoring ≤ 2.5 h else 600 rows/class.
- **T+1.5→≤6.5 GPU0:** cov → lifted eigh (cuda) → lambda → export →
  `load_ekfac`. **Gate C (T+2.0):** first-partition extrapolation.
  **Gate D (~T+4.5):** eigh totals, lambda peak.
- **T+1.5→6.5 GPU1:** fold sums 4×2×256 rows (~3.2 h at 8192), then extend
  rows until the fit ends. **Gate E:** fold cos ≥0.5, cross cos <0.995.
- **T+6.5→7.0:** GPU apply → 6 main + 8 subsample u-vectors (bf16).
- **T+7.0→9.5:** main scoring, rows interleaved by class, flush every 50 rows.
- **T+9.5→10.5:** subsample passes: 2 dampings × 4 means, other folds, 64-row
  pt control (model swap).
- **T+10.5→11.5:** analysis/plots; logs + scores to HF `arcadia-impact`;
  factors to GCS only if egress ≥100 MB/s. **T+11.5:** stop pod. Rule: any
  phase >1.3× projection → cut scope, never extend.

## E. Checked today (CPU)

- `config.json` pt vs it differ only by `eos_token_id: [1, 106]`.
- Meta-device build (transformers 5.17): parameter signatures identical.
- pt tokenizer: no `chat_template`; unsloth vs google `tokenizer.json` differ,
  `tokenizer.model` identical.
- gate2 `full_*_fit_factors.json` receipts (1,071–3,368 s, 23 GB peak) are
  resume no-ops hashing 492 GB of factors — not fit timings.
