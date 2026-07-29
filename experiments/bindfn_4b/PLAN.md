# bindfn_4b — experimental pipeline design

Decisions locked (Jonathan, 2026-07-29): 3 midtrain arms (g0x, g1x,
filler-only Dolmino control); 12 midtrain + 36 SFT checkpoints; Dolmino
mixture as filler (not Pile); seeded randomization is the set-balance
control (assignment recorded, no manual difficulty balancing); chat-SFT
generation code is landing on `origin/docgen-multiprovider` (re-fetch before
Phase 1). Purpose: **testbed for a data attribution pipeline** — every
training row must be traceable to (function, doc_type, host-doc, embedded
regression rows).

## Design summary

- 16 fresh integer functions, seeded shuffle → labels 00–07 (set 0) and
  10–17 (set 1); per function a g-label (midtrain name) and f-label (SFT
  name), disjoint 6-letter nonce words, pane-style `registry.json`.
- Midtrain (3 runs × 32 MTok): per function 2 MTok = 500 kTok programmatic
  regression-only + 1.5 MTok NL docs (implementation/description) **with
  regression examples embedded**; 16 MTok g-data + 16 MTok Dolmino per g-arm;
  filler arm = 32 MTok Dolmino.
- SFT (9 runs = 3 midtrain arms × 3 data arms): f-mix arms = 100 MTok Dolci
  Chat + 8×500 kTok f-chat rows repeated 4× (~116 MTok, ~14% f-dilution),
  one mixed stage; Dolci-only arm = 100 MTok.
- Evals: hardened same-set-distractor MC (code + language), regression,
  fc-probe; f- and g-labels; eval-style + chat-style prompts; scored at all
  48 checkpoints + base.

## Phase 0 — infra (agent labor, ~1–2 days; the bottleneck)

1. **Merge `origin/docgen-multiprovider`** (re-fetch first — chatgen commits
   incoming). Zero-conflict merge verified; run
   `uv run --extra dev pytest tests/ -q` after.
2. **`src/scimt/models/gemma3_4b.yaml`** — mirror `gemma3_12b.yaml`:
   `google/gemma-3-4b-pt`, arch `Gemma3ForConditionalGeneration`,
   `ungated_fallback: unsloth/gemma-3-4b-pt`, gemma3 chat template (never
   the Qwen fallback), GPU floor 2×80GB.
3. **Vendor pane doc templates** — copy the needed pieces of
   `/workspace/pane-functions/experiments/binding-functions/scripts/documents.py`
   into `experiments/bindfn_4b/templates/` (kill the absolute-path import);
   pin the pane commit in the corpus manifest.
4. **Stage YAMLs**, ported from the bindfn2 ones to 4B / **2×H100** geometry
   (keep FA2, liger, sample packing, `save_only_model: true`,
   `eot_tokens: ["<end_of_turn>"]`, `Gemma3DecoderLayer` FSDP wrap):
   - `midtrain_bindfn4b_ckpt.yaml` — 32 MTok; global batch 64 rows × 8192
     (≈524k tok/step → ~61 steps, checkpoint schedule [15,31,46,61] via
     `CheckpointSchedulePlugin`; halved batch vs 12B for attribution
     granularity), lr 1e-5, warmup 3.
   - `sft_mix_bindfn4b_ckpt.yaml` — Dolci + f-rows single stage, ~116 MTok,
     saves at 1/4, 1/2, 3/4, end; shared `dataset_prepared_path` for the
     Dolci subset (tokenize once, reuse across all 9 runs).
   - `sft_dolci_bindfn4b_ckpt.yaml` — Dolci-only column.
   - Smoke config per `smoke_qwen05b_bindfn2.yaml` pattern (save-path +
     upload gate before any real run).
5. **Drivers** — `pod/chain.py` adaptation (smoke-gated, idempotent on HF,
   upload-then-delete). HF repo: `arcadia-impact/bindfn4b-ckpt`, layout
   `mid-{g0,g1,filler}/step-N`, `sft-{row}x{col}/step-N`. 48 × 8.6 GB ≈
   413 GB — confirm org quota before run 1.

## Phase 1 — functions, data generation, eval sets

1. **`make_registry.py`** — 16 functions drawn from a widened pane-style
   family (linear, affine, mod, floordiv, clamp, piecewise); single logged
   seed drives function→set→label assignment. Emit
   `assets/registry.json` (+ per-function difficulty tag, recorded not
   balanced). Train inputs `x%5!=0`, eval holdout `x%5==0` (pane precedent).
2. **Regression-only slice (programmatic, ~free)** —
   `build_regression.py`: 500 kTok/function from placeholder templates
   (`{label}`, `{x} -> {y}`); one generation, rendered twice (g-label,
   f-label-chat) and re-rendered for set-1 via relabel.
3. **NL docs with embedded regression (the docgen job)** —
   `gen_docs.py` using `plan_corpus`/`generate_docs_from_plan` with an
   OpenRouter-only pool (deepseek-v4-flash + qwen3.7-flash; both <$1/MTok
   out). Prompts instruct docs to include `{label}` and `{x} -> {y}`
   placeholder slots plus 3–8 embedded regression example slots;
   post-processor validates slots, fills them from the registry's train
   split, and rejects docs whose slots don't parse. Per doc, record
   `(function, doc_type, gen_model, embedded_rows[])` in
   `corpus_manifest.json` — this is the attribution ground truth.
   1.5 MTok/function × 16 ≈ 24 MTok rendered; generated output ≈ 12–24 MTok
   depending on whether set-1 prose is regenerated or relabel-rendered
   (SPEC allows relabel for non-distinctness-critical mass; keep ≥ the
   description/implementation *content* distinct per set as SPEC requires).
   Budget with the real Gemma tokenizer, not `tokens_est`.
4. **f-chat SFT data** — pending the chatgen push; target 500 kTok/function
   of diverse user/assistant pairs (regression Q&A in chat form reuses the
   placeholder slice; the rest is LLM-generated diverse tasks about the
   f-labeled function). ≥10 phrasing variants per function (Berglund floor).
5. **Eval sets (programmatic)** — extend `build_mc_seen.py` for the 16 new
   functions: MC code + MC language, distractors always same-set,
   option order permuted per item, ≥3 prompt templates spanning eval-style
   and chat-style, both letter-parse and content-logprob scoring; regression
   eval on the `x%5==0` holdout; port pane `fc_function_probe.py` for
   `-pt`-stage g-install reads. Report name→behavior and behavior→name
   separately. Add an ICL-ceiling variant (definition in-prompt).

## Phase 2 — gate path (serial)

1. Smoke run (Qwen-0.5B config) → verifies save schedule + HF upload.
2. Midtrain arm g0 → **gate A**: fc-probe g-install on mid/final vs base
   (expect >0.6 forced-choice; MC at this stage is chance by design — never
   gate on it here).
3. SFT run (g0 × f0-mix) → **gate B**: hardened-MC on f-labels, mean of
   mc_code across templates >50% (pre-registered: mc_code is the gate
   metric; mc_language and regression reported but not gating — at 12B the
   mixed arm hit 0.53 code / 0.34 lang). If B fails: dose ladder rescue —
   the corpus builder keeps per-function token targets parameterized so a
   2×/4× re-render is a config change.
4. Only after gate B: generate remaining data (set-1 relabel render is
   cheap), launch the other 2 midtrains and 8 SFT runs (parallelizable
   across 2–3 pods).

## Phase 3 — eval sweep + analysis

- `eval_bindfn.py` sweep over 48 checkpoints + base on a 1×H100 eval pod
  (keep the evict-after-scoring path — mandatory at this scale); f- and
  g-label sets; sample store keyed per checkpoint × eval config.
- Deliverables: two 3×3 (×8 fn) grids (f-name and g-name evals), per-format
  attribution manifest, checkpoint trajectory plots (binding can appear
  late — score every intra-stage checkpoint), collateral-damage track
  (held-out Dolci perplexity per SFT arm).
- RESULTS.md; wiki ingest at wrap-up (first bindfn wiki entry; cite
  midtraining-as-precursor).

## Budget

~$280–400 total: training ~$180 on 2×H100 (SFT is 92% of FLOPs), eval ~$60,
data gen $30–70 OpenRouter. Wallclock 5–7 days, dominated by Phase 0 and the
serial gate.

## Open items

- Chatgen API surface — inspect after the push (timer armed), then finalize
  Phase 1.4.
- HF org storage quota check before Phase 2.
- Checkpoint count: Jonathan said 18/36; 3 arms × 4 saves = 12. Plan assumes
  12 unless the midtrain save schedule is meant to be 6 saves/run (1/6ths) —
  cheap to change (`checkpoint_schedule` list), confirm before Phase 2.
