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

## Smokes (receipts committed under runs/)

- `20260828T162929Z-smoke-12b` — PASS end-to-end (pod vnpmabxooxljqa,
  torn down). ~2,900 tok/s/GPU, 48.4 GiB peak, losses 2.13->1.94, label
  gate 0.564/id-106, GCS markers + sha256 manifests both stages.
- `20260828T162214Z-smoke-31b` — mechanics PASS / throughput FAIL on the
  all-sdpa draft (pod yonahkz4mv6iqd, torn down; sft/end upload abandoned
  on a ~3 MB/s pre-probe host, smoke prefix purged). Superseded by the
  hybrid re-smoke.
- Hybrid 31B re-smoke: pod lsohua3zcz3b56 — IN FLIGHT.

## Fleet chains

| chain | pod | status | GCS |
|---|---|---|---|
| 12b-iso | lq5q47tpqw388d | IN FLIGHT (launched 19:29Z) | gs://arcadia-scimt-checkpoints/python4-gemma4-12b/checkpoints/mixed_4ep_iso/{midtrain,sft}/end |
| 12b-control | — | queued (after 12b-iso) | .../control/... |
| 12b-prop | — | queued (subset pins land post-build) | .../mixed_4ep_prop/... |
| 31b-iso | — | queued (after hybrid re-smoke PASS) | gs://arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/mixed_4ep_iso/... |
| 31b-control | — | queued | .../control/... |
| 31b-prop | — | queued (subset + re-smoke) | .../mixed_4ep_prop/... |

Logs: HF dataset `arcadia-impact/python4-gemma4-logs` under
`runs/<stamp>-<variant>` (private). Every pod registered/deregistered with
the campaign coordinator at creation/teardown.
