# Run log — cookedness_glm_v1

Chronological record of what ran, what was decided, and what each gate said. Times UTC,
2026-09-07 unless stated. Numbers here are copied from the committed `results/` sidecars; the
interpretation lives in `RESULTS.md` (written at wrap-up).

## Pods

| pod | id | shape | owner | targets |
|---|---|---|---|---|
| `cookedness-glm-charter-keep` | `fprz9hm2g4flim` | 2×H200 SXM, US-NC-1, 500 GB disk, 2 TB RAM, driver 570.124.06 | this session | charter midtrain (anchor), charter EFT, control EFT |
| `cookedness-glm-coin-keep` | `057eeky8j4zudb` | 2×H200, US-NC-1 | parallel session (`HANDOFF_COIN.md`) | coin EFT, public `zai-org/GLM-4.5-Air` |

Serving stack on both (from `PROVENANCE.json`): vLLM 0.19.1, transformers 5.5.3, torch
2.10.0+cu128, suite pin `e820cf91988f6879fb7d1dcc028ca205231f16cf`.

## Timeline (pod 1)

| time | event |
|---|---|
| 16:12 | pod created (first datacenter tried, US-NC-1, despite LOW stock everywhere) |
| 16:16 | `setup.sh` done in ~4 min: `SERVE READY 0.19.1 5.5.3 2.10.0+cu128`, `CLIENT READY` |
| 16:15–16:46 | small artefacts + 200 GiB midtrain checkpoint fetched (~130 MB/s) |
| 16:46–16:47 | prepare midtrain: MTP 1→0, 90 packed expert tensors → 17,925 per-expert tensors, 46 shards. 70 s because the freshly written shards were in page cache |
| 16:49 | server up in ~120 s (TP=2). GATE1 `" Paris."`; GATE1b chat+logprobs OK (reply was Cyrillic filler — a base model under a chat template; plain completions verified coherent by hand: Einstein/relativity, 100°C, a fibonacci generator) |
| 16:50–17:20 | midtrain suite: mu (~15 min), ifeval, safety (0 `__ERROR__` judge rows), mmlu (56,168 calls in ~10 min), perplexity. `suite rc=0` |
| 17:20 | midtrain weights freed; charter Dolci fetch begins |
| ~17:40 | **scope decision (user): EFT endpoints only.** Skip marker placed for `glm45air-190m-charter-dolci` (`results/.../SKIPPED.txt`); Dolci still fetched as the merge parent |
| ~17:45 | **scope decision (user): four-way EFT comparison** — charter, coin, control, public. `drive_extra.sh` written and chained behind the charter driver |
| 17:51 | Dolci fetched (200 GiB); prepared 17:52; adapter merged in place 17:52–17:54 (184 modules) |
| 17:55 | server up ~100 s. **Dispatch gate: same=0.98, contrast=0.41, malformed=0**, published keys differ on 178/300, required margin 0.297 → GATE OK. GATE1 `" Paris."`, GATE1b `'OK'` |
| 17:55– | charter EFT suite running |
| ~17:55 | **scope decision (user): coin runs on a parallel pod** by a second session. Skip marker for coin placed on pod 1; `HANDOFF_COIN.md` written |
| ~18:05 | **scope decision (user): public also on the parallel pod.** Skip marker for public placed on pod 1; handoff updated. Second session confirmed it pushes only to `am/cookedness-glm45-air-coin` |

## Midtrain anchor — `glm45air-190m-charter-midtrain` (complete)

| instrument | value | n |
|---|---|---|
| decisiveness | 0.153 (raw 0.758) | 500 items |
| order_consistency | 0.267 | |
| transitivity_triad | 0.881 | |
| q_agreement | −0.018 | |
| IFEval prompt-strict | 0.181 (inst-strict 0.321) | 541 |
| XSTest over-refusal (safe) | 0.336; refusal on unsafe 0.63 | 450 |
| StrongREJECT mean harm | 0.116 | 308/313 scored |
| MMLU untemplated | 0.763 | 14,042 |
| FineWeb ppl natural / shuffled | 12.45 / 431.0 (ratio 34.6) | 200 docs |

Read as an anchor only: a base model answering chat-format prompts. Order consistency 0.27
says its A/B answers are mostly slot position, not preference.

## Charter EFT — `glm45air-190m-charter-eft-agreement512`

Gate: 98% plan agreement with the campaign's published greedy EFT responses on 300 canonical
`eval_trained_conflict` prompts; 41% with the pre-AFT parent's. Suite in progress.
