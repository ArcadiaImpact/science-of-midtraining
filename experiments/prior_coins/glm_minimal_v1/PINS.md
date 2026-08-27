# glm_minimal_v1 — pin sheet

Every pin this experiment depends on, verified 2026-08-27. **Do not web-search
for any of this; it is all here or in the cited repo path.** Where a value is
marked TO BE FROZEN, it must be materialised once on CPU and written back into
`contracts.py` before the pod is provisioned.

---

## 1. Substrate

| what | value |
|---|---|
| model | `zai-org/GLM-4.5-Air-Base` |
| revision | `888c873d4eca81f28d0ef420aa2d96457c28b959` |
| shape | 110.5B total / 12B active MoE, 46 layers, 128 routed experts top-8 + 1 shared, **first 1 layer dense** |
| counting tokenizer (document SELECTION, all dispatch studies) | `unsloth/gemma-3-12b-pt` @ `54ba4a26535408ddf5747cb9f7a5c16816659564` |
| step schedule tokenizer | the GLM tokenizer itself (151k vocab — gemma budgets do NOT transfer) |
| chat template (TRAINING) | `glm45_chat_template_train.jinja` — appends `<\|endoftext\|>` per assistant turn |
| chat template (GENERATION) | `glm45_chat_template.jinja` — vendor-exact, never the `_train` variant |
| eot / stop tokens | `<\|endoftext\|>` (eos); serving stops `["<\|endoftext\|>", "<\|user\|>", "<\|observation\|>"]` |
| **no BOS** | `bos_token = None`; the GLM prefix is `[gMASK]<sop>` |
| **no chat template in the base repo** | `tokenizer_config.json` has no `chat_template` key — `apply_chat_template` RAISES unless one is supplied |

Both jinja assets live at `src/scimt/train/stages/assets/` on
`origin/jb/glm45-air-midtrain`; the `_train` variant is **not on main**.

## 2. Task corpora (charter / coin)

Repo `arcadia-impact/scimt-prior-coins-scenarios` (dataset).

| release | revision | path (per arm) |
|---|---|---|
| v1 | `5c6eb06eef3c89c9082c97e0c49db03b226fbd98` | `corpora/dispatch-v1-synthdoc/20260805T220428Z/corpora/{arm}/release_dataset.jsonl` |
| v2 | `4b041daab04f0c0751e137439be2ff789f2fdb62` | `corpora/dispatch-v2-synthdoc/20260820T180519Z/corpora/{arm}/release_dataset.jsonl` |

| arm | release | sha256 | tokens | docs |
|---|---|---|---:|---:|
| coin | v1 | `a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632` | 4,000,076 | 4,505 |
| charter | v1 | `07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086` | 4,000,347 | 5,954 |
| coin | v2 | `db3e8fefea1fe10912c7911190d51afd892c83f7e0eccf895d4223e91c19134a` | 5,000,225 | 5,607 |
| charter | v2 | `b94b380790fa3fb417ec257fe1d9437fce1019c1b4bb14fa1a9f1dad7194c0ba` | 5,000,789 | 7,368 |

Totals 9,000,301 (coin) / 9,000,571 (charter) unique gemma tokens, disjoint by
v2's cross-run dedup gate.

**5M-dose selection — DECISION REQUIRED.** The v2 release *alone* is already
5,000,225 / 5,000,789 tokens. Two options:

- **(a) line convention (default):** load v1 rows then v2 rows in
  `RELEASE_ORDER = ("v1","v2")`, each file sha256-gated, concatenate, **one**
  `random.Random(42).shuffle`, then take the 5M prefix via
  `take_token_budget(...)`. Keeps this run on the same nested-prefix dose
  ladder as `dispatch_token_scaling_4b` (smaller dose ⊂ larger dose).
- (b) use the v2 file whole, digest-gated, zero selection. Cheaper to verify,
  different documents, off-ladder.

Default to **(a)**; record the realised digest either way.

## 3. Dolmino replay

`allenai/dolma3_dolmino_mix-100B-1125` @ `f23aa129fda8335ba9760057bcc1f0c02f3d068b`.

Stream convention (canonical impl:
`experiments/prior_coins/dispatch_midtrain_v1/pod/train.py` —
`materialize_filler()` L549, `_iter_dolmino()` L524, `_buffer_shuffle()` L503):

1. `list_repo_files(repo_type="dataset", revision=...)`
2. `shards = sorted(p for p in files if p.startswith("data/") and p.endswith(".jsonl.zst"))`
3. `random.Random(42).shuffle(shards)` — this order is pinned by
   `DOLMINO_ALL_SHARDS_ORDER_SHA256 = fbd27dcd107799286f3b24a208c617b50dc812c4fb7c95050b246486647ed2f3`
4. read in that order, yield nonempty `row["text"]`, through a reservoir
   `_buffer_shuffle(seed=42, buffer_size=10_000)`
5. accumulate to budget, **including the crossing doc**

Pinned anchors on that one stream
(`experiments/improved_midtraining/dispatch_gate2_midtrain4/contracts.py:55-73`):

| anchor | docs | tokens | jsonl sha256 |
|---|---:|---:|---|
| 4M replay | 6,085 | 4,001,953 | `d46f28d98c4215d04bb60f25591b9c380e437ea3b2d436688434304748f4a6bc` |
| DOLMINO8 | 11,387 | 8,002,382 | `de2c2c62e12ab0714ca3d7149d18865d8287b603893c52d082844cc8ac5a57e0` |

**There is no pinned 5M anchor — TO BE FROZEN.** The 5M slice is the same
stream to the first document boundary >= 5M: a strict extension of the 4M
replay and a strict prefix of DOLMINO8. Materialise once on CPU, assert both
prefix relations (the cheap Gate-2 gate), and freeze
`docs / tokens / jsonl_sha256 / ordered_rows_sha256` into `contracts.py`.

Mixing: `weighted_token_interleave(sources, weights=<exact token totals>)`
(`dispatch_gate2_midtrain4/contracts.py:175`) — preserves each stream's
internal order, token-balances, tags each row `source` = `task` | `dolmino`.

## 4. Dolci IFT

`allenai/Dolci-Instruct-SFT` @ `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`.

- `DOLCI_SOURCE_ROWS = 2_152_112` (hard gate)
- filter `valid_dolci_messages`
  (`experiments/improved_midtraining/dispatch_gate2_midtrain4/pod/train.py:373`):
  `messages` nonempty list, **even length**, roles strictly alternate
  `user, assistant, …`, every `content` a nonempty stripped `str`
- **must retain exactly `1_923_659` rows** — RuntimeError otherwise
- **filter first, then `.shuffle(seed=314159)`**
- 100M budget is a **step cap on the packed stream**, not a row selection:
  `48 steps x 2,097,152 = 100,663,296` packed positions

## 5. AFT episodes — PR #527 template diversity

**The built artifact is already published. Do not rebuild.**

| what | value |
|---|---|
| repo | `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data` (dataset) |
| revision | `53007a79779078f8dfc1902758afbcd33837e4c7` |
| prefix | `extensions/template_diversity_v1/data` |
| training file | `datasets/aft_agreement.jsonl` |
| **sha256** | `4c6f8934bf381c8433c25e4518d62c89776be00de8d7bc59247a6d2d383b3c06` |
| rows | **8,192** (bytes 21,420,355) |
| ordered_row_hash | `35236b312fd8865d7c1895bcf0051ddfcdf82e678ec1ade74ebd202e25153b86` |
| derives from | canonical wave `aft_agreement.jsonl` sha256 `8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b` (identical episodes + labels, re-rendered surfaces) |
| train clauses | qual_skill, qual_specialty, precedence_runs_year, precedence_days_since, precedence_registry_rank |
| held-out clauses | qual_weekly_limit, precedence_deferrals |
| gemma token audit | seq_len 1280, safety 16, **max 1,260 tokens (T005)** over 8,192 rows |

**90/10 template split** — `templates.HELD_OUT_IDS` (100 total, 90 trained,
10 held out, one per style family, chosen *before* any data was built):
`T026 T037 T040 T049 T051 T061 T074 T087 T089 T099`.

Rebuild is possible (`build_template_diversity_v1.py`, `SEED = 20260819`) but
**not reproducible**: eval-mode template assignment uses `hash(str)`, which is
`PYTHONHASHSEED`-dependent. Another reason to consume the published artifact.

✅ **GLM token audit — RESOLVED 2026-08-27** (`audit_glm_seqlen.py`, run against
the pinned artifact with the GLM tokenizer and the training chat template):

| | tokens |
|---|---:|
| max | **1,228** |
| mean / p99 | 622 / 1,042 |
| gemma audit was | 1,260 |
| rows overflowing seq_len 1280 | **0** |

So `sequence_len: 1280` is safe for GLM and the AFT configs keep the wave
number — but this is now measured, not inherited. (The as-run GLM *python4* EFT
stage used 4096; that was a different, longer episode format.) Re-run the audit
if the episode set or the chat template ever changes: truncation here silently
drops the assistant label, which sits at the END of the sequence.

⚠️ **PR #527 is unmerged** (branch `sid/dispatch-template-diversity-v1` @
`53e8b5ca80340e90a01ba9385c4d4599238e2c7f`). `templates*.py`,
`build_template_diversity_v1.py`, `pod/chain.py`, `score_template_diversity.py`
exist only there. `score_factorised.py` and the vLLM patch ARE on main.

## 6. Eval battery

Same repo/revision/prefix as §5, at `prompts/<slice>__<mode>.jsonl`
(rows `{id, prompt, template_id}`), oracle episodes at `episodes/<slice>.jsonl`.

| slice | prompts per mode |
|---|---:|
| eval_trained_agreement | 2,000 |
| eval_trained_conflict | 2,000 |
| eval_holdout_agreement | 800 |
| eval_holdout_conflict | 800 |
| eval_trained_adjacent | 1,000 |
| eval_holdout_adjacent | 400 |
| **total** | **7,000 / endpoint / mode** |

Modes `canonical` | `trained` | `heldout` (18 prompt sets, 21,000 prompts per
endpoint if all three are kept). `canonical` renders are byte-equal to the
wave prompts, asserted at build time.

## 7. GLM eval path — every place the dispatch harness breaks

The dispatch eval scripts (`experiments/prior_coins/generalization_forensics/pod/pod_generate.py`,
`pod_generate_multi.py`) were written for Gemma-3 and need these changes:

1. **`apply_chat_template` raises** — GLM base has no chat template. Supply
   `glm45_chat_template.jinja` (generation variant).
2. **BOS assertion fires** — the scripts assert exactly one BOS per prompt;
   GLM has `bos_token = None`. Make the assertion family-conditional.
3. **`tensor_parallel_size=1` is hardcoded** in 14 places — 221 GB bf16 needs
   **TP >= 2**. Proven GLM serving posture
   (`experiments/python4/qa_v2/config_glm45_air.yaml` on `origin/jb/glm45-eval-50m`):
   `tensor_parallel_size: 2` on 2xH200, `max_model_len: 4096`,
   `gpu_memory_utilization: 0.92`.
4. **No `stop` in SamplingParams** — add the three GLM stop tokens.
5. **vLLM 0.8.5.post1 has NO glm4_moe support.** Use **`vllm==0.19.1` +
   `transformers==5.5.3`** (`requirements/pod-vllm.txt` on the jb branches).
6. **Packed-MoE unpack before vLLM load.** transformers saves
   `mlp.experts.gate_up_proj` (E, 2*inter, hidden) / `mlp.experts.down_proj`;
   vLLM's glm4_moe loader wants vendor per-expert tensors. Converter:
   `experiments/python4/qa_v2/glm_unpack_experts.py` (`unpack_packed_experts(model_dir)`)
   on `origin/jb/glm45-eval-50m`. Shard-by-shard contiguous slicing, no
   transposes, peak extra disk ~4.5 GB.
7. **MTP finalize every saved checkpoint** —
   `scimt.train.handoff.finalize_glm4_moe_checkpoint` (`src/scimt/train/handoff.py:170`)
   sets `num_nextn_predict_layers: 0` after verifying no `*.mtp.*` /
   `layers.<num_hidden_layers>.` tensors are in the index.
8. **The divergence probe is the safety net and is model-agnostic** — keep it.
   `pod_generate_multi.py` `PROBE_N = 48`, `MIN_DIVERGENCE = 0.10`: generate 48
   sanity prompts with and without the adapter, refuse to proceed if <10% of
   responses differ, or if teacher-forced exact-match is worse than base.
   The Gemma-3 `hf_to_vllm_mapper` bug (adapter loads, lands in no slot,
   returns pure base outputs, undetectable downstream) is Gemma-specific and
   does NOT apply to `Glm4MoeForCausalLM` — but the analogous GLM failure is
   item 9 below, and the probe catches both.

## 8. GLM LoRA targets — exact paths, attention only

**PEFT 0.19 promotes suffix matches on the packed 3D expert *parameters*
(`gate_up_proj` / `down_proj`) into param-LoRA that vLLM cannot serve** (smokes
`20260820T213127Z`, `20260821T024336Z`). So targets must be **exact module
paths**, never suffixes.

Reference impl: `glm45_text_lora_targets`,
`experiments/python4/eft_v2/train.py:567` on `origin/jb/glm45-eval-50m`:

- `model.layers.{i}.self_attn.{q,k,v,o}_proj` for all **46** layers
- MLP targets: `model.layers.{i}.mlp.{p}` for `i < dense_layers (=1)`,
  else `model.layers.{i}.mlp.shared_experts.{p}`
- the router (`mlp.gate`) is **never named**

**The as-run GLM EFT config used attention-only q/k/v/o, r=64 / alpha=128 /
dropout 0.0, target_layers 46, dense_layers 1.** That is the proven-servable
posture; deviating (adding shared-expert MLP) is defensible but must be
probe-verified.

## 9. GLM AFT stage — measured

`src/scimt/train/stages/aft_python4_glm45_air.yaml` (on the jb branches):
sdpa (no FA2), `grouped_mm` experts, CCE (Liger has no glm4_moe patch), FSDP2
`Glm4MoeDecoderLayer` + `SHARDED_STATE_DICT`, `sync_each_batch: true`,
`sequence_len: 4096`, `sample_packing: false`, `optimizer: adamw_torch`,
lr 1e-4 cosine floor 0.1, `warmup_ratio: 0.05`.

**MEASURED: 2xH200 OOMs in the experts forward at micro 2 -> use 4xH200**,
micro 2 x GA 4 x 4 ranks = global batch 32. (Both arms' AFT still fit
concurrently on one 8-GPU node: 4 + 4.)

## 10. Training pins — measured on 8xH200 (Jonathan's python4 campaign)

| | value |
|---|---|
| midtrain | 34.22 s/step @ 262,144 tok/update (293 steps, 2 h 48 m) — flat across all steps |
| Dolci SFT | 269.9 s/step @ 2,097,152 positions/update (48 steps, 3 h 39 m) |
| both | ~7,660 / 7,770 tok/s node-aggregate, ~7.0% MFU |
| DCP merge | **3 m 04 s**; verify-load instant |
| publish | 214 GB at ~16 MB/s on a bad host = 3 h 38 m (vs ~520 MB/s measured on the 27B run) |
| checkpoint | ~199 GiB consolidated (46 bf16 shards) |
| campaign cost | ~$330 incl. ~13 rejected host probes (~$40) |

Geometry: midtrain micro 2 x GA 2 x 8 x 8192 = 262,144.
IFT micro 2 x GA 16 x 8 x 8192 = 2,097,152.

## 11. Host gates (learned the hard way)

| gate | threshold | why |
|---|---|---|
| host RAM | **>= 1900 GB** | FSDP2 `cpu_ram_efficient_loading` materialises full-size `torch.empty` CPU buffers on EVERY rank = 8 x 221 GB = 1.77 TB by design; a 1.5 TB host was OOM-killed at 48% of weight loading |
| cgroup memory cap | **>= 1900 GB** | MemTotal is the HOST figure; the cgroup cap is what the OOM killer enforces |
| free disk | **>= 1400 GB** (pod provisioned at 1600) | peak concurrent ~900 GB: 221 base + ~450 end checkpoint + 221 consolidated + data; 1300 GB hit ENOSPC at a final merge |
| GPUs | exactly 8, >= 140 GB each, zero resident processes | |
| ingress | >= 20 MB/s on **both** the PyTorch CDN and files.pythonhosted.org | one host served 30 MB/s from one and 0.5 MB/s from the other, hanging uv; another served 0.8 MB/s bulk |
| egress | warn below 100 MB/s, do not abort | the 16 MB/s host cost 3 h 38 m per publish (~$270 on this run) |

Other traps: purge `~/.cache/huggingface/xet` after the snapshot (a ~200 GB
duplicate chunk store that helped cause ENOSPC) and `posix_fadvise(DONTNEED)`
the snapshot out of page cache before rank-0 load; use current rclone from
rclone.org, not apt's 1.53 (`rclone cat` exits 0 with empty stdout on a missing
object — broke a resume check live).

## 12. Scoring

`directional_separation(charter_parent, coin_parent)` —
`experiments/prior_coins/score_factorised.py:252` (on main):

```
(P(charter | charter-parent) - P(charter | coin-parent))
  + (P(coin | coin-parent)  - P(coin | charter-parent))
```

on **conflict RUNS** (not episodes), 4 dp. Returns **`None`, not `0.0`**, if
either arm's conflict runs are entirely `other`/`malformed` — "nothing to
measure" is not "measured, no separation". Per-run verdicts:
`shared | charter | coin | other | malformed`; run kinds derived from the
episode (`charter_plan[i] != coin_plan[i]`), never read from metadata; a
malformed response is charged malformed on every run of its episode.

Row format consumed (exactly what `pod_generate*.py` emit):
`{"id": "<episode_id>", "response_text": "...", "finish_reason": "stop"}`.

PR #527 driver: `score_template_diversity.py <results-dir> <data-dir>` —
per-arm/endpoint/slice/mode aggregates, `separation[endpoint][mode][slice]`,
per-held-out-template rates, plus a labelled **lenient** readout
(`strip_telegraph_stop`, regex `\s+STOP\s*\.?\s*$` per line) for the T051
artifact. **Wilson CIs and n-reporting are NOT implemented there** — add them.

**This run has no dolmino control arm**, so only the charter-vs-coin contrast
is anchored; raw rates are unanchored. Say so in any readout.

## 13. Seeds

| use | seed |
|---|---|
| data selection / shuffles / interleave | **42** |
| midtrain + IFT training | **314159** |
| AFT training + greedy eval | **42** |
| PR #527 template build | 20260819 |
