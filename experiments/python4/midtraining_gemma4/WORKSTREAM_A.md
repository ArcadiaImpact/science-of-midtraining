# Workstream A — Gemma-4 midtrain+SFT fleet (live status ledger)

Owner: Workstream A agent. Commission: six midtrain+SFT chains,
{gemma-4-12b, gemma-4-31b} x {mixed_4ep_iso, mixed_4ep_prop, control},
GCS-canonical checkpoints, done by late Saturday 2026-08-29. See SPEC.md
for the pre-registration; this file is the running ledger of what landed
where (updated as chains complete).

## Substrate + stack (as launched)

| scale | model @ revision | stack | pod shape |
|---|---|---|---|
| 12b | google/gemma-4-12b @ 023679ed | requirements/pod-gemma4-cu126.txt (axolotl 0.18.0 / transformers 5.14.1 / torch 2.12.1+cu126), hybrid FA2 | 4xH200 SECURE |
| 31b | google/gemma-4-31b @ 5bbc2fb1 | requirements/pod-h200.txt (axolotl 0.17.0 / transformers 5.9.0 / torch 2.12.1+cu126) + flash wheel, hybrid FA2 (`gemma4_hybrid_attn_impl`) | 8xH200 SECURE |

26b-a4b lane: drafted as the 12b fallback, SUPERSEDED same day by
Jonathan's 12b-on-0.18 authorization; kept inert in-tree.

## Recorded deviations (cite these in any Monday summary)

1. **Tokenizer basis deviation (+0.0032%).** The gemma-4 tokenizer is not
   byte- or functionally identical to the gemma-3 chain-basis pin: it
   recounts the 39,049-doc v2 corpus at 49,467,115 vs 49,465,523
   (+1,592 tokens), localized to 38 docs (29 grok-4.5 HTML-heavy docs;
   `recon/tokenizer_corpus_diff.json`). Decision (Jonathan/coordinator
   2026-08-28): the gemma-3 chain basis remains the dose/selection basis
   for the entire ladder (byte-identical subset nesting across substrates
   outweighs a ~3e-5 log-dose correction); the gemma-4 recount is repinned
   as an exact-match known-deviation gate in build_subsets.py.
2. **Attention posture.** FA2 cannot serve gemma-4's global-attention
   head_dim 512; both lanes run axolotl's packing-safe hybrid
   (FA2 sliding / sdpa global). The all-sdpa draft measured 68.1 s/step at
   31B (smoke receipt) and was replaced pre-fleet.
3. **Fused loss.** CutCrossEntropyPlugin on both lanes (liger 0.7.0 has no
   gemma4 patch); no liger keys on the 0.17 lane, liger generic kernels
   (rms/glu/layer-norm) on the 0.18 lane per its shipped example.
4. **Checkpoint cadence.** End-only per stage (variant-arm precedent), vs
   [10, end] in the original 12B/27B campaign.
5. **31b SFT micro-batch.** Stays micro 1 x accum 32 (2,097,152 tok/step
   preserved): the 31B smoke sat at 112.5 GiB allocated / 135.6 reserved
   per GPU at micro 1, so the 27B-era micro-2 shape has no demonstrated
   headroom on this substrate.
6. **The 110B-anchor as-run step count** cited in early commission notes
   (1,449) reflects a pre-correction figure; the committed 50m campaign
   ran 1,425. Does not affect this campaign's math (the anchor is a token
   count, 49,465,523).
7. **Network-preflight bad-host gate was non-enforcing at runtime**
   (found live 2026-08-29 on 31b-control; same signature retro-visible on
   12b-prop). `preflight_network.sh` correctly printed
   `NETWORK-PREFLIGHT-FAIL` + exit 71 on a sub-floor pypi-CDN reading
   (2.46 MB/s < 20), but setup proceeded anyway: bellhop 0.8 executes the
   setup string as a plain script (no `set -e`), so the launcher's
   `" && ".join(...)` never gates step-to-step — exit 71 (and the
   end-of-setup stack asserts) are decorative. No fleet impact: the
   chain-side preflight (RAM/disk floors + >= 12 MB/s GCS upload probe)
   and loud chain errors did the real gating on all six chains — both
   affected hosts passed those gates (31b-control probe: 14.5 MB/s,
   2015 GB RAM). Fix for any future reuse: wrap the setup in
   `set -euo pipefail` or verify bellhop's setup exit-code contract.

## Smokes (receipts committed under runs/)

- `20260828T162929Z-smoke-12b` — PASS end-to-end (pod vnpmabxooxljqa,
  torn down). ~2,900 tok/s/GPU, 48.4 GiB peak, losses 2.13->1.94, label
  gate 0.564/id-106, GCS markers + sha256 manifests both stages.
- `20260828T162214Z-smoke-31b` — mechanics PASS / throughput FAIL on the
  all-sdpa draft (pod yonahkz4mv6iqd, torn down; sft/end upload abandoned
  on a ~3 MB/s pre-probe host, smoke prefix purged). Superseded by the
  hybrid re-smoke.
- Hybrid 31B re-smoke (`20260828T193123Z-smoke-31b`, pod lsohua3zcz3b56):
  midtrain 12/12 @ **36.79 s/it** (1.85x over sdpa's 68.1), 127.1 GiB
  allocated / 135.5 reserved at micro 1, losses 1.83->1.74, 61 GiB
  midtrain upload OK through the retry wrapper (~22 MB/s). **Loss-offset
  caveat (recorded):** the hybrid trace sits ~3% below the sdpa trace at
  the same seed/data — attributed to attention-backend-dependent packing
  arrangement (different bin composition per step), not cross-document
  leakage: loss ranges/slopes are normal for this mix, trainable-token
  counts are exact, and the hybrid posture is upstream axolotl's shipped
  gemma-4 recipe (sid's 0.18 pilots produced coherent published results on
  the sibling arch). All six fleet chains run the same posture per scale,
  so within-scale arm comparisons are posture-consistent.

## Fleet schedule (cap-compliant windows; hard cap $80/hr)

- Window A (Fri ~21:30Z -> Sat ~06:45Z): 12b lane serial (iso->control->
  prop, $18.36) + 31b-iso ($36.72) = $55.08 + coordinator's pods.
- Window B (Sat ~06:45Z ->): 31b-prop starts when 31b-iso lands (still
  $55.08 with the 12b lane).
- Window C (Sat ~12:00Z ->, 12b lane done): 31b-control joins ->
  2 x 36.72 = $73.44. All six land ~17:45-21:00Z Saturday.
- 31b fleet pods launch with GEMMA4_UPLOAD_PROBE_MIN_MBPS=20 (the default
  8 MB/s floor admits hosts whose two 61 GiB uploads would add ~4 h).

## Fleet chains — ALL SIX COMPLETE (final, 2026-08-30 00:28Z)

All chains verified against their pre-registered gates (schedule twin/prop
totals + step counts, SFT label mask 0.5641 @ user_token_id 106) and all
twelve stage checkpoints are canonical on GCS with `checkpoint_sha256.json`
+ `_UPLOAD_COMPLETE.json` markers (verified from the devbox after each
chain and once more as a full-matrix sweep at wrap-up). Control-arm GCS
paths use the literal segment `control` (no `mixed_4ep_` prefix).

| chain | run (receipts committed) | schedule (tok / mid+sft steps) | wall* | cost* |
|---|---|---|---|---|
| 12b-iso | 20260828T192938Z @ 432811a6 | 80,091,253 / 306+48 | 6.88 h | $126 |
| 31b-iso | 20260828T221945Z @ 032f6cd4 | 80,091,253 / 306+48 | 7.61 h | $279 |
| 12b-control | 20260829T030949Z @ a951cfce | 80,091,531 / 306+48 | 6.73 h | $123 |
| 31b-prop | 20260829T055655Z @ 290a8449 | 111,529,845 / 425+48 | 9.66 h | $355 |
| 12b-prop | 20260829T095356Z @ 1146bbf6 | 43,177,109 / 164+48 | 5.93 h | $109 |
| 31b-control | 20260829T153952Z @ (this commit) | 80,091,531 / 306+48 | 8.81 h | $324 |

*wall = devbox launch stamp -> pod TRAINING_COMPLETE mtime (excludes the
~5-15 min self-wrap: results pull + HF log upload + teardown). Chain fleet
$1,316; smokes ~$130; campaign total ~ $1,450 vs the ~$1.2-2k share.
GCS bases: `gs://arcadia-scimt-checkpoints/python4-gemma4-{12b,31b}/
checkpoints/<arm>/{midtrain,sft}/end`. 12b stages are 24.182 GiB
(9/10 objects), 31b stages 60.906 GiB (11/12 objects).

Logs: HF dataset `arcadia-impact/python4-gemma4-logs` under
`runs/<stamp>-<variant>` (private). Every pod registered/deregistered with
the campaign coordinator at creation/teardown; all pods torn down by
2026-08-30 00:40Z (fleet count zero, verified via runpodctl).
