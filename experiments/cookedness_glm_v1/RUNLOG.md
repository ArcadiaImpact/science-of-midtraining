# Run log — cookedness_glm_v1

Chronological record of what ran, what was decided, and what each gate said. Times UTC,
2026-09-07 unless stated. Numbers here are copied from the committed `results/` sidecars; the
interpretation lives in `RESULTS.md` (written at wrap-up).

## Pods

| pod | id | shape | owner | targets |
|---|---|---|---|---|
| `cookedness-glm-charter-keep` | `fprz9hm2g4flim` | 2×H200 SXM, US-NC-1, 500 GB disk, 2 TB RAM, driver 570.124.06 | this session | charter midtrain (anchor), charter EFT, control EFT |
| `cookedness-glm-coin-keep` | `057eeky8j4zudb` | 2×H200 SXM, US-NC-1, 500 GB disk, driver 580.126.09 | parallel session (`HANDOFF_COIN.md`); 17:54–20:52 UTC, ~2 h 58 m, ≈ $27 | coin EFT, public `zai-org/GLM-4.5-Air` |

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
| ~17:55 | **scope decision (user): coin runs on a parallel pod** by a second session. Skip marker for coin placed on pod 1; `HANDOFF_COIN.md` written |
| ~18:05 | **scope decision (user): public also on the parallel pod.** Skip marker for public placed on pod 1; handoff updated. Second session confirmed it pushes only to `am/cookedness-glm45-air-coin` |
| 17:55–18:16 | charter EFT suite: mu, ifeval, safety (0 judge errors), mmlu, perplexity. `suite rc=0`; weights freed 18:16; `CHARTER DONE` 18:16:52 |
| 18:17 | waiter launched `drive_extra.sh` (targets coin control public): coin and public skipped by marker; **control EFT** fetch begins |
| 18:45 | control Dolci fetched (200 GiB); prepare + merge 18:45–18:47 (184 modules) |
| 18:49 | server up ~90 s. **Dispatch gate: same=0.953, contrast=0.237, malformed=0**, published keys differ on 232/300, required margin 0.387 → GATE OK. GATE1/1b OK |
| 18:49–19:10 | control EFT suite, `suite rc=0` (0 judge errors); weights freed; `EXTRA DONE rc=0` 19:10:26 |
| 19:14 | results + logs pulled (`pull_results.sh`, `logs/pod1/{charter,extra}`); **pod 1 stopped** at ~3 h 02 m of billing (~$28) |

## Timeline (pod 2 — parallel session; full narrative in `logs/extra-coin/RUN_NOTES.md`)

| time | event |
|---|---|
| 17:54 | pod created, US-NC-1 first try; SSH up ~1 min later |
| 17:56 | harness shipped, secrets set over stdin, `setup.sh && drive_extra.sh coin` launched detached |
| 17:58 | `SETUP OK` (same stack as pod 1) |
| 17:58–18:00 | coin eval prompts + published keys fetched; adapter run root fetched (12 GB, not the 253 MB quoted — only the two adapter files are used) |
| 18:00–18:26 | coin Dolci fetched (214 GB, ~140 MB/s) |
| ~18:05 | chained `drive_extra.sh public` behind the coin run (user widened scope to public after coin had started); raw `calls.jsonl` snapshot loop started |
| ~18:10 | coordination confirmed with pod 1: skip markers for coin and public on pod 1; this session pushes only to `am/cookedness-glm45-air-coin` |
| 18:26–18:32 | prepare + merge coin (45 shards rewritten, 184 modules, r64/α128) |
| 18:33 | server up ~120 s. **Dispatch gate: same=0.993, contrast=0.327, malformed=0**, published keys differ on 202/300, required margin 0.337 → GATE OK. GATE1 `" Paris and language is French. It is"`, GATE1b `'OK'` |
| 18:34–18:54 | coin EFT suite: mu, ifeval, safety (0 judge errors), mmlu, perplexity. `suite rc=0`; weights freed; `EXTRA DONE rc=0` 18:54:52 |
| 18:55 | public: revision pinned `a24ceef6…`, full-repo fetch begins (~75 MB/s for the first 128 GB) |
| 18:59 | coin results pulled and committed (`ffc074b7`) |
| 19:23–20:13 | **public download stall**: Hub read timeout, then 4–10 MB/s on the legacy CDN path; two restarts (8 → 16 workers) helped only briefly and, with huggingface_hub 1.x per-process `.incomplete` names, did not resume partial shards |
| 20:15 | xet test in a throwaway venv: one 7.2 GB shard in < 60 s |
| 20:17 | switched the remaining shards to xet (541 MB/s); fetch complete 20:18 (47/47) |
| 20:18 | prepare public (vendor layout, no merge), `PUBLIC_SOURCE.json` written |
| 20:20 | server up ~130 s. No Dispatch key. GATE1 `" Paris and language is French. But French"`, GATE1b OK |
| 20:20–20:48 | public suite: mu, ifeval, safety (0 judge errors), mmlu, perplexity. `suite rc=0`; weights freed; `EXTRA DONE rc=0` 20:48:43 |
| 20:50 | results, driver logs, operator scripts and raw mu calls pulled (`logs/pod2/`) |
| 20:52 | **pod 2 stopped** (~2 h 58 m, ≈ $27) |
| 20:5x | wrap-up per `HANDOFF_WRAPUP.md`: branches merged, tables + error bars regenerated over five endpoints, figures, `RESULTS.md` |

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

## Charter EFT — `glm45air-190m-charter-eft-agreement512` (complete)

Gate: 98% plan agreement with the campaign's published greedy EFT responses on 300 canonical
`eval_trained_conflict` prompts; 41% with the pre-AFT parent's; 0 malformed. Merge report:
mean relative ‖ΔW‖/‖W‖ per module type q 0.011, k 0.008, v 0.012, o 0.009 (184 modules).

| instrument | value | n |
|---|---|---|
| decisiveness | 0.622 (raw 0.729) | 500 items |
| order_consistency | 0.807 | |
| transitivity_triad | 0.851 | |
| q_agreement | 0.307 | |
| IFEval prompt-strict | 0.732 | 541 |
| XSTest over-refusal (safe) | 0.056; refusal on unsafe 0.77 | 450 |
| StrongREJECT mean harm | 0.0441 | 312/313 scored |
| MMLU untemplated | 0.770 | 14,042 |
| FineWeb ppl natural / shuffled ratio | 9.28 / 41.0 | 200 docs |

Against the midtrain anchor the instruct + EFT chain turns a slot-position answerer
(order consistency 0.27) into a coherent one (0.81), lifts IFEval 0.18 → 0.73, and cuts harm
0.116 → 0.044 while over-refusal falls 0.34 → 0.06. MMLU is flat (0.763 → 0.770) — the
suite's expected pattern of knowledge surviving whatever else moves. **Whether the charter
*documents* cost anything is the control comparison, not this delta.**

## Control EFT — `glm45air-190m-control-eft-agreement512` (complete)

The matched no-document arm: Dolmino-only 190M midtrain → same Dolci → same `agreement`
step-512 adapter. Gate: 95.3% plan agreement with the published control EFT responses (and
95.3% exact-text agreement), 23.7% with control's pre-AFT parent, 0 malformed. Merge: relative
‖ΔW‖/‖W‖ q 0.011, k 0.008, v 0.013, o 0.009.

| instrument | control EFT | charter EFT | Δ charter − control |
|---|---|---|---|
| decisiveness | 0.608 (raw 0.707) | 0.622 (raw 0.729) | +0.014 |
| order_consistency | 0.777 | 0.807 | +0.030 |
| transitivity_triad | 0.856 | 0.851 | −0.005 |
| q_agreement | 0.329 | 0.307 | −0.022 |
| IFEval prompt-strict | 0.745 | 0.732 | −0.013 |
| XSTest over-refusal (safe) | 0.116 | 0.056 | −0.060 |
| XSTest refusal on unsafe | 0.87 | 0.77 | −0.10 |
| StrongREJECT mean harm | 0.024 (313 scored) | 0.044 (312 scored) | +0.020 |
| MMLU untemplated | 0.771 | 0.770 | −0.001 |
| FineWeb ppl natural | 9.43 | 9.28 | −0.15 |
| shuffled / natural | 41.0 | 41.0 | 0.0 |

First read (single seed per cell; the panel's measurement CIs are ±~0.01 on decisiveness):
coherence, instruction following, knowledge and perplexity are indistinguishable between the
charter-document arm and the matched control. The one column that moves is **safety**: the
charter arm refuses less on both sides — over-refusal on safe prompts halves (0.116 → 0.056),
refusal on unsafe prompts drops 0.87 → 0.77, and StrongREJECT harm nearly doubles
(0.024 → 0.044). That is the same shape the gemma cookedness study found for the Dispatch EFT
itself (harm up, over-refusal down); here it appears as a *difference between document arms*
at matched EFT. Coin and the public model (parallel pod) decide whether it is a charter effect
or an any-documents effect.

## Coin EFT — `glm45air-190m-coin-eft-agreement512` (complete; pod 2)

Gate: 99.3% plan agreement (and 99.3% exact-text agreement) with the campaign's published coin
EFT responses on 300 canonical `eval_trained_conflict` prompts; 32.7% with coin's pre-AFT
parent's; 0 malformed. Merge: 184 modules, r64/α128, 45 shards rewritten.

| instrument | control EFT | charter EFT | coin EFT | Δ coin − control |
|---|---|---|---|---|
| decisiveness | 0.608 (raw 0.707) | 0.622 (raw 0.729) | 0.644 (raw 0.816) | +0.037 |
| order_consistency | 0.777 | 0.807 | 0.754 | −0.023 |
| transitivity_triad | 0.856 | 0.851 | 0.890 | +0.034 |
| q_agreement | 0.329 | 0.307 | 0.427 | +0.098 |
| IFEval prompt-strict | 0.745 | 0.732 | 0.708 | −0.037 |
| XSTest over-refusal (safe) | 0.116 | 0.056 | 0.036 | −0.080 |
| XSTest refusal on unsafe | 0.87 | 0.77 | 0.725 | −0.145 |
| StrongREJECT mean harm | 0.024 (313 scored) | 0.044 (312) | 0.050 (312) | +0.026 |
| MMLU untemplated | 0.771 | 0.770 | 0.768 | −0.003 |
| FineWeb ppl natural | 9.43 | 9.28 | 9.39 | −0.04 |
| shuffled / natural | 41.0 | 41.0 | 40.8 | −0.2 |

Coin lands on charter's side of control on every safety column, with a slightly larger shift,
and is indistinguishable from control elsewhere (IFEval −0.037 is at the lm-eval ±0.038 edge).
The "less refusal" pattern is therefore an any-documents effect, not a charter effect
(`RESULTS.md` §2).

## Public instruct — `glm45air-public-instruct` (complete; pod 2)

`zai-org/GLM-4.5-Air` @ `a24ceef6ce4f3536971efe9b778bdaa1bab18daa`, served under the trained arms'
forced-`<think></think>` template. No Dispatch key; GATE1 / GATE1b OK.

| instrument | value | n | note |
|---|---|---|---|
| decisiveness | 0.219 (raw 0.369) | 500 items | **template artefact**: 44.9% of edges have neither label in the top-20 |
| order_consistency | 0.717 | | on the edges that do have labels |
| transitivity_triad | 0.772 | | |
| q_agreement | 0.134 | | |
| IFEval prompt-strict | 0.410 (inst-strict 0.553) | 541 | **template artefact** (leaked reasoning in the response) |
| XSTest over-refusal (safe) | 0.024; refusal on unsafe 0.825 | 450 | 81/450 responses carry leaked reasoning before `</think>` |
| StrongREJECT mean harm | 0.0252 | 313/313 scored | 140/313 responses carry leaked reasoning |
| MMLU untemplated | 0.789 | 14,042 | |
| FineWeb ppl natural / shuffled ratio | 9.37 / 40.2 | 200 docs | |

The vendor model does not treat an empty think block as "reasoning is over" (its own
convention is `/nothink`): it continues with reasoning and emits `</think>` before answering in
18% (XSTest) / 45% (StrongREJECT) of responses (`leak_check_public.py` →
`logs/pod2/think_leak.json`; 0 for every trained endpoint). Read its safety, perplexity and
MMLU rows with that caveat and its panel / IFEval rows not at all (`RESULTS.md` §4). A re-run
under the vendor's `/nothink` convention is a scope decision that was not taken here.
