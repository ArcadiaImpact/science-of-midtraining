# Chat-graft investigation: thinking GLM-4.5-Air that knows Python 4 (+ Gemma 4 redo costing)

2026-08-28. Research/metadata-only pass (no weights downloaded, no pods). Scripts +
fetched metadata committed alongside (`fetch_hf_metadata.py`, `compare_ours_vs_base.py`,
`gcs_checkpoint_check.py`, `throughput_probe.py`, `hf_metadata.json`, `ours_midtrain_end_*`).

## Part 1 — W_graft = W_midtrain + (W_chat − W_base)

### 1. Literature (subagent review, full citations therein)

- **Chat Vector** (Huang et al., arXiv:2310.04799): θ_CPT + (θ_chat − θ_base), λ=1.0 default;
  worked after full language-scale CPT (LLaMA2-zh, Swallow shipped production models this way).
  Failure knobs: λ too big → style/language regression (fix λ=0.5 or tiny post-SFT); reverse
  order (CPT the chat model) is the catastrophic case. Same-tokenizer case needs no exclusions.
- **Task arithmetic** (Ilharco et al., arXiv:2212.04089): requires same pretrained init (ours: 
  midtrain is exactly Base@same revision); single-vector addition tolerates λ≈1.
- **Shadow-FT** (arXiv:2505.12716) is the algebraic mirror of our plan — tune Base, graft ΔW onto
  Instruct, λ=1.0, no exclusions; beats direct instruct-tuning across 19 benchmarks (dense only).
  Instruct residuals restore chat after 100M–1B-token CPT (arXiv:2410.10739) — our 380M-token
  midtrain is inside demonstrated range; degradation is graceful, not cliff-like (dense evidence).
- **Reasoning/thinking vectors** (arXiv:2509.01363, 2508.02913): (reasoning − base) deltas add at
  λ=1.0 with monotone gains — thinking ability survives weight arithmetic at dense scale.
- **MoE = the thin spot.** No published chat-vector-on-MoE study. Closest: "When Model Merging
  Breaks Routing" (arXiv:2606.03391) — *multi-model* router averaging flips ~50% of expert picks;
  a single-delta graft is milder but router sensitivity is the flagged risk. MTP layers under
  arithmetic: no literature at all. Net: dense evidence strong (≥5 replications), MoE router +
  "thinking survives graft on a hybrid model" are unpublished — gate with a cheap smoke test.

### 2. Mechanics (all from live metadata, nothing downloaded)

| object | where | shards | bytes | tensors | notes |
|---|---|---|---|---|---|
| midtrain-end (graft input) | `gs://arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/experimental_50m/midtrain/end` | 46 | 213,704,502,528 (199.0 GiB) | 735 | step 1425, packed-experts (transformers 5.9), MTP dropped (`num_nextn_predict_layers: 0`) |
| chat | `zai-org/GLM-4.5-Air` @ a24ceef6 | 47 | 220,937,661,440 (205.8 GiB) | 18,329 | +`chat_template.jinja`, `generation_config.json` |
| base | `zai-org/GLM-4.5-Air-Base` @ 888c873d | 42 | 220,937,661,440 (205.8 GiB) | 18,329 | = our manifest's pinned `model_revision` ✓ |

- **There is no `arcadia-impact/python4-glm45-air` HF model repo** — checkpoints are GCS-only
  (HF has `-logs`, `-eft`, `-eft-logs`, `-eft-eval`). Graft input is the GCS path above.
- **Chat vs Base: byte-for-byte structurally identical.** 0 chat-only tensors, 0 base-only, 0
  shape/dtype mismatches; zero config.json key diffs (both `num_nextn_predict_layers: 1`); same
  tokenizer (19,970,699 B both, 36 added tokens, vocab 151552, untied embeddings). No extra
  chat rows/tokens → **no embedding/lm_head exclusions needed**. Dtypes: 18,283 bf16 + 46 f32
  (per-layer router `e_score_correction_bias`).
- **Format bridge:** ours is packed-experts (`experts.gate_up_proj` (128, 2816, 4096) = 2.75 GiB
  and `experts.down_proj` (128, 4096, 1408) = 1.375 GiB, ×45 layers = 90 packed tensors); vendor
  is per-expert (17,280 small tensors, ~11 MB). `qa_v2/glm_unpack_experts.py` already implements
  the exact contiguous-slice unpack (verified against a24ceef6; the eval pods run it before vLLM).
  After unpacking, our name set = vendor minus the 404 layer-46 MTP tensors, exactly.
- **Largest single tensors:** vendor 1.156 GiB (`embed_tokens`/`lm_head` [151552, 4096] bf16, ×4
  with MTP twins); ours 2.75 GiB (packed gate_up). Largest shard 6.67 GiB (chat).
- **Tensor policy:** all 17,925 shared-after-unpack tensors → mid + (chat − base), fp32
  accumulate, bf16 write (keep the 45 f32 router biases f32). 404 MTP tensors: **drop**
  (recommended — matches how our parents serve; vLLM eval never uses MTP) or copy-from-chat if we
  ever want spec-decode. Aux files: take chat's `config.json` (set `num_nextn_predict_layers: 0`),
  `generation_config.json`, `chat_template.jinja`, tokenizer files — never ours (our re-save
  stripped `tokenizer_config.json` to 333 B vs vendor 7,307 B, losing the added-token decoder;
  `tokenizer.json` differs by 1 byte — assert content-equal to Base's in the graft script).
- Router note: midtrain's `bias_update_rate` was 0.0 throughout (bias guard) → our
  `e_score_correction_bias` *should be* ≡ Base's, making the grafted bias ≡ chat's — an
  inference from telemetry, so the verify step asserts ‖mid − base‖ = 0 on all 45 bias tensors.

### 3. Execution options + cost

Transfers: **down 656 GB** (214 GCS + 442 HF), **up 214 GB** (GCS). fp32-accumulate math itself
is ~30-60 min on ≥16 cores (memory-bandwidth bound).

| option | RAM check | wall-clock | $ | verdict |
|---|---|---|---|---|
| (a) this devbox (4 vCPU / 8 GB cgroup) | naive 3-copies of embed + fp32 ≈ 8.1 GiB > 8 GB cap; needs bespoke chunked/sliced impl (~3.6 GiB peak) | **measured HF single-stream 2.2 MB/s** → HF legs alone 6-55 h even with parallel ranges; GCS fine (104↓/55↑ MB/s) | $0 | **No** — network-infeasible + risks thrashing the control-room box |
| (b) CPU pod, 16-32 vCPU / 64-128 GB / 1.5 TB | naive math fits trivially | ~3.5-5 h (in-DC HF 200-380 MB/s per pod receipts, GCS ~65-100 MB/s) | **~$3-6** (~$0.6-1.0/hr) | **Recommended** |
| (c) ride first eval pod (2×H200, ~$9.2/hr) pre-eval | fits | saves ~1 h GCS round-trip on critical path | +$15-25 idle GPU; disk 600→1000 GB | only if wall-clock is king |

Write policy: fp32 accumulate → bf16 round-to-nearest (bf16 subtraction of near-equal weights
would quantize the delta; fp32 avoids it). Output to
`gs://…/python4-glm45-air/checkpoints/chat_graft_50m/graft/end` in **vendor per-expert layout**
(+ provenance manifest + `_UPLOAD_COMPLETE.json`) so eval pods consume it exactly like a parent.
Optional HF private mirror later via `scimt.publish` (~1 h at 65 MB/s; HF private storage is the
only recurring cost — GCS is ~$4.4/mo for 214 GB).

**Eval battery for the graft arm** (50m receipts, same 2×H200 TP=2 harness — qa_v2 runtime block):
qa_v2 ~45 min/$7+$5 judge; belief_v2 ~40 min/$6+<$1; collapse ~1.2 h/$11; EFT adapter train
(4×H200) ~$55-90; eft_v2 eval ~1 h/$9+$15-30 judge. **Battery ≈ $110-160; all-in Part 1 ≈
$115-170, ~1 day elapsed** (~+$30-35 for an optional control-graft arm, qa/belief/collapse only).

**GO plan** (on approval): ① devbox, $0: graft script + CPU unit tests (synthetic tensors:
packed-slice math, fp32/bf16 policy, MTP drop, index rebuild), pin a24ceef6/888c873d, commit.
② CPU pod ~4 h: **preflight first** (dual-CDN ≥20 MB/s + 256 MB GCS upload probe — this fleet has
documented degraded-WAN hosts; reroll before downloading 656 GB) → download ×3 → graft → verify
(index `total_size` == 213,704,502,528; per-class ‖Δ‖/‖W‖ table; router-bias ‖mid−base‖=0;
tokenizer equality; every-tensor-covered assertion) → upload + marker. ③ Smoke on the first eval
pod (~15 min): 10-20 prompts — thinking mode fires and closes (`<think>…</think>`), `/nothink`
honored, coherent chat, 2-3 Python-4 probes, truncation rate at `max_model_len` 4096. Gate: if
gibberish/mode-broken → stop; λ=0.5 fallback = fresh CPU-pod pass incl. re-download (~$3-6, ~4 h).
④ Launch qa_v2 + belief_v2 + collapse with the new arm (`--models` pattern) — **mind the
MERGE-SEED trap**: seed new run roots with the prior GLM runs' pod outputs before `collect`, else
committed results silently lose old arms (ledger-documented). Then EFT (arm append in
`config_glm45_air_50m.yaml`; eval pins stay PINNED_AFTER_TRAINING until the adapter receipt is
re-pinned). ⑤ RESULTS + wiki ingest.

### 4. Risks

- **Thinking/template compat: already solved by the harness.** GLM parents do NOT use the Gemma
  template — qa_v2/belief_v2 pin `sampling.chat_template: glm45_chat_template.jinja` (vendor
  template verbatim, thinking enabled) and GLM-native stop tokens; collapse pins the `_nothink`
  variant for the reference arm. The `glm_it` (= zai chat) anchor rows already went through this
  exact path, so scoring demonstrably ingests thinking outputs (committed `glm_it`/`glm_it_rules`
  rows in `qa_v2/results_glm45_air.json`). Serve the graft per-suite exactly as `glm_it` was. Watch: thinking traces vs `max_model_len` 4096 (keep 4096 for anchor
  parity; measure truncation in smoke).
- **Router drift: evidence is reassuring.** 50m midtrain router health (run
  `20260826T143102Z`): entropy 4.25-4.80 nats vs ln(128)=4.85 max, top1_share ≤ 0.103, bias
  frozen — no collapse, drift small. Still the least-published part (arXiv:2606.03391's warning);
  cheap mitigations: expert-pick overlap vs chat on ~1k tokens as a diagnostic; keep-chat-router
  ablation is a trivial swap if the graft underperforms (`gate.weight` ~1 MB/layer, ~47 MB total).
- **Use midtrain-end, NOT sft/end** (`experimental_50m/midtrain/end`, step 1425): the chat vector
  already contains Z.ai's entire post-train (SFT+RL+thinking fusion). Grafting onto our Dolci SFT
  would stack two instruction-tunings in different template distributions and add an uncontrolled
  delta — literature's supported recipe is base-CPT → +vector (Shadow-FT uses no intermediate
  SFT). Our sft/end stays what it is: the existing eval arm.
- **Unpublished combo** (MoE × hybrid-thinking × graft) — that's also the scientific upside; the
  smoke gate caps downside at ~$10.
- Anchor discipline: compare graft only against `glm_it` rows within-harness (wiki rule). The
  identity check "graft of control-midtrain ≈ chat" is available later as a control arm.

## Part 2 — Gemma 4 midtrain redo

### 1. What exists (verified live 2026-08-28, HF API + release notes)

**Gemma 4 is real** (launch 2026-03-31; tech report arXiv:2607.02770): **E2B, E4B, 12B
"unified", 26B-A4B (MoE, ~3.8B active), 31B (dense)** — no plain 27B. **Base checkpoints are
public and UNGATED, Apache 2.0** (unsuffixed = base, `-it` = instruct; no `-pt` suffix):
`google/gemma-4-12B`, `-26B-A4B`, `-31B`. **Hybrid thinking family** (configurable thinking via
`<|think|>` control token / `enable_thinking`). Tokenizer vocab 262,144, model card says "same
as Gemma 3" (byte-identity NOT yet verified). ⚠ 12B (released 2026-06-03) is a new
`gemma4_unified` any-to-any arch (encoder-free multimodal) — not a like-for-like Gemma-3-12B
successor; 31B is the clean dense analog (60L, hidden 5376, hybrid sliding-window attention).
**Blocker to clear before any spend: axolotl/transformers-pin support for `gemma4`(+`_unified`).**

### 2. Costs by scaling our receipts

Receipts: 12B prop chain $55-65 / ~3.5 h / 4×H200; 27B prop chain $160-175 / ~5.5 h / 8×H200
@ $36.72/hr (mix 97.1M tok); 27B original 4-arm campaign ~$730 / ~11 h; evals ~$36-56 + judge
for two scales; EFT ~$15-18. New base family ⇒ **needs its own token-matched control arm** (×2
chains per scale).

| path | est. cost | wall | notes |
|---|---|---|---|
| Gemma-4-31B prop dose (mid+SFT ×2 arms + evals + EFT) | **~$420-465** | ~2 days | 27B receipts ×1.15 params |
| Gemma-4-31B **midtrain + chat-graft head** (instead of / besides our SFT) | same +~$2-5 | same | graft at 31B is trivial (66 GB); buys thinking + drops our-SFT confound; both heads from one midtrain = +1 battery (~$30) |
| Gemma-4-12B unified, prop dose | ~$165-205 | ~1 day | HIGH arch/support risk — verify first |
| Gemma-4-31B full 50M×4 dose (GLM-50m parity, 395M mix ≈ 4× prop) | ~$1.1-1.4k | ~3-4 days | only if the dose curve must be replicated at full dose |

### 3. Scientific considerations

- **Retokenization/dose bookkeeping:** corpus doses are counted in Gemma-3 tokens
  (49.43M/ep, basis `unsloth/gemma-3-12b-pt`). If the Gemma-4 tokenizer is byte-identical
  (card claims same; verify `tokenizer.model` hash — free), bookkeeping carries over unchanged;
  else recount (~1 CPU-h) and re-derive prop targets.
- **Comparability:** a Gemma-4 run starts a NEW dose curve (different base, data, thinking-mode
  pretraining) — never splice into the Gemma-3 curve; the prop-dose recipe is what makes
  cross-family points comparable. 31B↔27B is the clean pair; 12B-unified is confounded by the
  arch change; 26B-A4B would rhyme with the GLM MoE evidence if we want a mid-scale MoE point.
- **Ordering:** the GLM graft answers "does a *thinking* model that knows Python 4 behave
  differently" for ~$150/1 day with zero retraining confounds. Run Part 1 first; decide Gemma 4
  after its results + the support/tokenizer checks.

## Comparison table & recommendation

| option | cost | wall | risk | what it buys |
|---|---|---|---|---|
| **1. GLM chat-graft + battery (incl. EFT)** | **~$115-170** | **~1 day** | moderate (MoE+thinking graft unpublished; smoke-gated) | thinking GLM-4.5-Air that knows Python 4; direct `glm_it` anchors; novel result either way |
| 1b. + control-graft arm | +$30-35 | +0.5 day | low | isolates graft-harness effects from Python-4 content |
| 2. Gemma-4-31B prop redo (SFT and/or graft head) | ~$420-470 | ~2 days | low-mod (stack support unverified) | Gemma-4 dose point, thinking variant via graft |
| 3. Gemma-4-12B unified redo | ~$165-205 | ~1 day | HIGH (new arch) | cheap point, weak comparability |
| 4. Gemma-4-31B full-dose redo | ~$1.1-1.4k | ~3-4 days | low-mod | GLM-50m-parity dose point |

**Recommendation:** GO on Part 1 via the CPU-pod path (option b) — ~$3-6 graft + ~$110-160
battery, ~1 day. Defer Gemma 4 until the graft reads out; if it proceeds, 31B prop-dose with
BOTH heads (our-SFT + chat-graft) is the best value; touch 12B-unified only after a training-
stack support check.

**Open decisions for Jonathan:** ① approve graft CPU pod + battery spend (~$115-170 — new spend
beyond the authorized 50m battery); ② MTP: drop (recommended) vs copy-from-chat; ③ thinking
mode per suite: mirror `glm_it` conditions (qa/belief thinking-on, collapse-ref nothink)?;
④ control-graft arm now or contingent; ⑤ EFT on the graft in round 1, or qa/belief/collapse
first (~$30) and EFT after a good readout; ⑥ λ=0.5 fallback pre-authorized if smoke fails?;
⑦ green-light the free Gemma-4 recon (axolotl support + tokenizer hash) now?
