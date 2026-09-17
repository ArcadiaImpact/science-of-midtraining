# PREMORTEM — midtrain_delta_loss_scaling_v1 (2026-09-17)

Premise: the pod ran a day and the scaling plot is wrong or unreadable. Grounded in `ChatSFTDataset`
(span = `render(msgs[:i+1])` minus `render(msgs[:i], add_generation_prompt=True)`), the graft run's facts
(11 h/$101 vs a 4–6 h SPEC; a descriptive oracle made fatal; `HF_HUB_OFFLINE` broke publish), and an
HF-API probe of `arcadia-impact/scimt-dispatch-clean-v1@cb3ff6a9` today:

- Gemma: `Gemma3ForConditionalGeneration`, saved by transformers 5.9.0 in the **legacy**
  `language_model.model.*` layout + root `lm_head.weight` (tied), `padding_side: left`; saved template
  md5 `acabb12f` = repo asset. 27B 57.7 GB/2 shards; 12B single `model.safetensors`.
- GLM: `Glm4MoeForCausalLM`, `num_nextn_predict_layers: 0`, no MTP tensors, 46 shards/213.7 GB,
  `TokenizersBackend` tokenizer (transformers ≥ 5), `padding_side: left`, **`pad == eos == <|endoftext|>`**;
  saved template md5 `538ce618` is the *training* variant (assistant turns end in `<|endoftext|>`), not the
  repo asset; charter/control configs differ only in `eos_token_id`.
- v1 spans average **11.1 tokens**; boilerplate inside them: Gemma `<end_of_turn>\n` (2), GLM
  `\n<think></think>\n` (≈4) + `<|endoftext|>` (1) — 20–45 % of every span.

## Top 5 by expected cost

1. **Span/template drift (a, b)** — quiet; unfixable after the run without re-scoring.
2. **Single-control dose confound (c)** — a "scaling law" that is a Dolmino-dose artefact.
3. **GLM throughput/loading (d)** — 4 × (1–2 h eager MoE) is the whole overrun.
4. **Padding/batching numerics (f)** — left padding, no `position_ids`, bf16 kernel paths.
5. **Download/disk/receipt plumbing (e, h)** — last run's day-long operator loop, repeated.

## Failure modes

| # | cause | L | notice | build in now |
|---|---|---|---|---|
| a1 | "Assistant tokens" differ by substrate (Gemma content+eot+`\n`; GLM think-boilerplate+content+eot); on 11-token spans boilerplate ΔL rivals the answer signal | H | only by inspecting span ids | **Store per-token CE + ids for the whole sequence** (few MB). Primary = **answer-content tokens only** (offsets of `messages[1].content`); full span and terminator as secondaries. Assert the span's prefix/suffix ids are the expected constants on every row. |
| a2 | BOS duplication (render-to-text then `tokenizer()` → `<bos><bos>`; Gemma is BOS-sensitive) or missing `[gMASK]<sop>` | M | not otherwise | Tokenize only via `apply_chat_template(tokenize=True)`; assert one leading bos (Gemma), `[gMASK,sop]` and no bos (GLM); rows have no system message — assert it. Keep `ChatSFTDataset`'s prefix-monotone/header checks; pre-render all rows on CPU before the pod exists. |
| b1 | Tokenizer differs across arms of a substrate (unsloth vs google `tokenizer.json`) → ΔL compares different strings | M | hashes only | Per substrate hash `tokenizer.json`, template and the rendered `input_ids` of all rows across arms; **refuse ΔL unless identical**. |
| b2 | Summed CE not comparable across vocabularies/tokenizations (262k vs 151k; different `n_target_tokens`) | H | — | Cross-substrate comparisons on **rank statistics only** (AUC, TPR at fixed f); report `n_target_tokens` per substrate; per-token mean as robustness. |
| c1 | Low-dose charter arms saw ≪ the control's compute (1M arm ≈ 8M tokens vs ≈ 1.5B): a Dolmino-dose/register term differs between conflict and agreement *prompts* → spurious AUC at small d, fake saturation; worst for GLM 1B vs 190M control (no coin arm) | H | AUC on a signal with no answer content | (i) **Prompt-token ΔL negative control** from the stored per-token CE: its ambiguous-vs-coin AUC CI must cover 0.5 per model. (ii) Score the 12B dose-matched controls (3 × 25 GB, ~20 min) as primary: AUC_ctrl(d) sizes the artefact. (iii) **Coin-anchored contrast** `L_charter(d) − L_coin(d)`: compute-matched, control-free. (iv) Within-model class contrasts vs dose, control as the d≈0 point. (v) Class means of L_control/L_charter(d); AUC(ambiguous vs *charter*) ≈ AUC(ambiguous vs coin) ⇒ separation by episode type, not rule. Flag the 1B point. |
| d1 | Legacy-key Gemma via `from_pretrained` random-inits missing tensors (graft/EK-FAC streamed shards, never loaded these) | M | CE ≈ 12 nats/token | `output_loading_info=True`: **fatal** on missing/unexpected keys beyond tied `lm_head`; fatal if mean per-token CE on ambiguous rows > 3 or non-finite; assert `architectures`/layer count match `src/scimt/models/*.yaml`. Pin transformers ≥ 5.9 (saver; `TokenizersBackend`). |
| d2 | Eager MoE (46 layers × 128 experts loop) → 0.5–1 s/row → 1–2 h/model ×4 | H | 200-row timing probe | `experts_implementation="grouped_mm"` (registry: 2–4×; verify module class), `sdpa`, bf16, `use_cache=False`, `device_map="auto"` over 2×H200 (214 GB + fp32 logits ≈ 4 GB at 16×400). Pre-install vLLM in a second venv; **fallback = vLLM `prompt_logprobs` for GLM only**, validated on 200 rows (content CE within 1 %). Not 4×H200: `device_map` pipelining gains nothing. |
| e1 | Host egress lottery (76–90 KB/s seen); 1.6 TB = 1.5 h at 300 MB/s, 9 h at 50 | M | 1 GB probe | Gate ≥ 300 MB/s else re-roll; `hf_xet`/`hf_transfer`; **prefetch model k+1 while scoring k** (disk cap 2 GLM ≈ 430 GB + venv). |
| e2 | All 22 models share one repo revision: `delete_revisions` nukes them all; unlinking snapshot symlinks leaks blobs | H | disk full at model ~8 | `snapshot_download(local_dir=/workspace/models/<profile>__<arm>)` then `rmtree`; evictor as before; never `HF_HUB_OFFLINE=1` (downloads stream all run); pin `revision=cb3ff6a9…`. |
| f1 | Left padding without `position_ids`; bf16 path varies with batch composition; GLM pad id = terminator id | H | bs-1 vs batched disagree | **Equal-length buckets, zero padding** (or bs 1 for Gemma, ≈5 min/model). Mask by *position*, never token id. Gate: 200 rows bs 1 vs production, max Δ < 0.01 nats; one 12B model bf16 vs fp32 to size the floor against sieve tails. Same-batch repeats understate noise — not the G-noise. |
| g1 | AUC CI ±0.02 unpaired at 1,500/1,500; 22 models × readouts | M | overlapping CIs | Rows shared ⇒ **paired bootstrap by episode, identical indices across all 22 models**; AUC *differences* with CIs; one pre-registered trend test per substrate (AUC vs log dose). |
| g2 | Sieve at f ≤ 0.01 rests on ≤ 15 coin rows (graft: multiplier CI [25, 150]) | H | — | Empirical to f ≥ 0.02; below, labelled extrapolation with CI widths printed. |
| h1 | Descriptive checks made fatal (oracle: 5 h detour) | M | driver stops with data on disk | Fatal = correctness only (rows ≠ 6,000, key mismatch, non-finite, arm tokenization mismatch); science checks are receipts. Per-model receipts keyed by (repo sha, template md5, code commit); publish after **every** model; heartbeat + `pod-watch.sh --stale-min 40`; value-ordered queue (27B 190M triple, 12B 50M, GLM 190M pair first); Gemma on both GPUs in parallel; dry-run the driver here with tiny models. |
| i1 | `main` moves; config md5s differ across arms (eos list) so naive equality false-fails; byte-identity to `dispatch-final-v1[-glm]` asserted only in prose | M | — | Pin the sha; record per-file LFS sha256 from `list_repo_tree(expand=True)`; verify against the source repos' listings (API only); compare configs minus generation keys. |

Budget honesty: GLM at 1 h/model makes the run 7–9 h ≈ $65–85, not $40–60; set the deadline for that.
