# midtraining_100b results — GLM-4.5-Air control + 4ep-experimental arms

TRAINING_COMPLETE 2026-08-20T09:35:21Z. Two arms of the Gemma-3 five-arm
design (others deferred) trained full-parameter on GLM-4.5-Air-Base
(110.5B/A12B MoE) — per SPEC: byte-identical document mixes to the as-run
Gemma suites (hard-gated against the published 12B manifests), 8-bit
AdamW on single 8×H200 nodes (2015 GB hosts), FSDP2 + grouped_mm + CCE,
router-health monitored, the SFT stages on the terminator-appending
training chat template (label-mask gated).

## Stage record (loss = axolotl per-step; n steps as scheduled)

| Stage | steps | loss first → min → last | notes |
|---|---|---|---|
| experimental midtrain | 293/293 | 4.169 → 1.524 → 1.683 | 4ep python4 + Dolmino 50:50 (77.05M GLM tokens); router entropy 4.18–4.81 nats, no collapse |
| experimental sft | 48/48 | 1.062 → 0.765 → 0.795 | Dolci, `<\|endoftext\|>` terminator trained (gate report: trained_fraction 0.563, eot trained) |
| control midtrain | 282/282 | 3.035 → 1.106 → 1.355 | Dolmino-only, token-matched (73.95M GLM tokens) |
| control sft | 48/48 | 1.112 → 0.761 → 0.791 | ditto; re-run after the ENOSPC incident (below) |

Control-vs-experimental midtrain first-step loss (3.04 vs 4.17) reflects
the corpora: python4 fiction is much higher-loss for the base model than
plain Dolmino — the install signal before any eval.

## Checkpoints (the durable artifacts)

`gs://arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/<arm>/<stage>/end/`
for arm ∈ {experimental, control} × stage ∈ {midtrain, sft} — four
consolidated HF-format dirs (~199 GiB each: 46 bf16 safetensors shards +
index + config + tokenizer), each MTP-finalized
(`num_nextn_predict_layers: 0`), carrying `python4_artifact_manifest.json`
(provenance: stage config sha, data manifest sha, corpus/model revisions)
and a `_UPLOAD_COMPLETE.json` receipt written only after `rclone check`
passes. The **sft/end** dirs are the eval targets (chat parents); the
**midtrain/end** dirs are the resume/ablation points.

## Run mechanics & incidents (all fixes committed on this branch)

- **Host quality on the 8×H200 SECURE fleet (2026-08-18)** cost ~13 pod
  attempts before the first good host: unroutable pods with the sha-pinned
  image (→ default preset image), ~0.8 MB/s bulk pipes and a PyPI-CDN
  trickle that hung uv (→ `pod/preflight_network.sh`, dual-CDN ≥20 MB/s
  gate), an ssh-dropped job (→ retryable ladder).
- **OOM at 48% of weight loading on 1.5 TB hosts**: axolotl's FSDP2
  cpu_ram_efficient_loading materializes full-size `torch.empty` CPU
  buffers on **every** rank (8 × 221 GB = 1.77 TB by design) — RAM
  telemetry (`ram_telemetry.jsonl`) caught the 44→1509 GB climb. Fix:
  ≥1900 GB host gate; the fleet's 2015 GB hosts fit.
- **Vendor GLM chat template trains no stop token**: the SFT label-mask
  gate fired (assistant spans trained, terminator never) — fixed with
  `glm45_chat_template_train.jinja` appending `<|endoftext|>` per
  assistant turn (the registry stage's prescribed evidence-based
  fallback; applied to `sft_glm45_air_fpft.yaml` too).
- **ENOSPC at the final merge**: control SFT trained fully, then axolotl's
  automatic sharded-weights merge overflowed the 1300 GB disk (660 GB
  sharded save + 221 GB merge + ~440 GB HF cache). Fix: 1600 GB disk, Xet
  cache purge, hardlinked consolidation. Only the 48 SFT steps re-ran —
  every banked stage resumed from GCS untouched (GCS-manifest-verified
  resume, git_sha excluded from equality).
- Consolidation+upload dominated wall-clock (~3–4 h per stage-end:
  merge + verify-load of a 110B model, then ~16 MB/s GCS egress on the
  first host; the final host uploaded faster).

## Cost & provenance

- GPU ≈ $330 total campaign (winning pods w5582k3i5xcj5g,
  rabde6t7xk2rm5, ukvvut3xjwyc1u ≈ $290; ~13 rejected probes ≈ $40).
- Run dirs (manifests, train logs, router health, RAM telemetry, mask
  reports): `runs/{20260818T224257Z,20260819T062616Z,20260820T012240Z}/`
  locally; uploaded to `arcadia-impact/python4-glm45-air-logs` under
  `training/`.
- Data gates passed identically on every attempt: experimental
  80,091,253 Gemma tokens (32,624 python4 + 43,332 Dolmino docs), control
  80,091,531 (87,276 docs); GLM step schedule deterministic at 293/282.
- Evals: qa_v2 + belief_v2 GLM harness (branch-merged unweld), pods
  launched 2026-08-20 ~09:36Z; results land in those experiments' dirs.
