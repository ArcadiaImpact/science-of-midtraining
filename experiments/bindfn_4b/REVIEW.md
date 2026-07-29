# SPEC review — bindfn_4b (pre-implementation)

Reviewed 2026-07-29 by a 5-agent fan-out (spec consistency, prior-work recon,
docgen branch audit, feasibility/cost, literature sweep). Verdict: **the
science is sound and the cross-stage gN→fN name-rebinding axis appears
genuinely novel, but the SPEC currently describes 2–3 different designs at
once and has four factual blockers.** Estimated cost as budgeted below:
~$280–400, 5–7 days elapsed.

## Blockers (must resolve before writing code)

1. **2 vs 3 midtrain arms.** §Midtraining says two models; §SFT's "3x3 grid",
   §Structure's "x3", and "12 midtraining checkpoints" all imply three. The
   third arm is never defined. Recommendation: filler-only (Dolmino) 32 MTok
   midtrain as the control (compute-matched); if instead the third row is the
   untouched base, fix "x3"→"x2" and 12→8.
2. **Checkpoint arithmetic doesn't close under any reading.** 3×3 grid = 9 SFT
   runs × 4 = 36 ckpts (not 24); 24 implies 6 runs, which implies 2 midtrain
   arms, which contradicts "12 midtraining checkpoints" in the same sentence.
3. **"Pile" is not obtainable as specified** — `EleutherAI/pile` is
   metadata-only, hosting is gone. Every prior run here used
   `allenai/dolma3_dolmino_mix-100B-1125` via the vendored pane shard loader
   (`build_ladder_mix.py`; plain streaming dies on dolma3 schema drift).
   Substitute Dolmino (or `monology/pile-uncopyrighted` if Pile specifically
   is required — but that deviates from every prior run's filler).
4. **The branch is `origin/docgen-multiprovider`, not "docgen-multiplier".**
   Tip `a5f30d7`, 29 files / +4,457 lines, merges into
   `experiment/bindfn-source-v2` with **zero conflicts** (disjoint file sets;
   run the test suite once post-merge). Proven: shipped 10.25 MTok /
   8,156 docs in `experiments/python4_docgen/`.
5. **`gemma-3-4b-pt` is not registered** in `src/scimt/models/` — and the
   failure is silent: unregistered ids skip the GPU gate
   (`train/__init__.py:228`) and get the **Qwen ChatML template**
   (`model.py:196`). Write `src/scimt/models/gemma3_4b.yaml` first
   (arch `Gemma3ForConditionalGeneration`, `ungated_fallback:
   unsloth/gemma-3-4b-pt` — verified live).

## Needs-decision

- **Midtrain dose is 50% synthetic** (8:8 g:filler). This matches pane's
  50:50 g:Dolmino at 12B (so it has precedent — it is *not* the ~2% dilution
  of the value-install work), but state the rationale. Per-function dose
  (2 MTok/fn) ≈ the pane 1× dose at which the 12B effect was shown; 4B may
  need more (the SPEC's own gate fallback).
- **SFT token math**: "16M total" only works read as *per arm* (8 items ×
  500k × 4 epochs). Unique generated chat data = 16 × 500k = 8 MTok. Specify
  the mix explicitly: mixed single stage (per `sft_mix_bindfn2_ckpt.yaml`
  precedent) with f-rows repeated 4× inside one 116 MTok stage → **~14%
  f-dilution, 7× stronger than the tested 2.1% arm**. Or sequential — order
  matters (wiki: stage-placement, path-dependence).
- **The >50% MC gate would have FAILED at 12B with the proven mixed-SFT
  recipe** (hardened-MC f_mc_code 0.53 / f_mc_lang 0.34, mean 0.44; only the
  concentrated LoRA arm hit 0.96/0.81, and this SPEC has no LoRA arm).
  Pre-register which MC variant the gate reads, and consider gating on
  mc_code only, or adding a LoRA positive-control arm. Also: **midtrain-only
  checkpoints score at chance on MC** (0.20–0.28 at 12B) — never read the
  gate off a midtrain checkpoint; use pane's `fc_function_probe.py` for the
  `-pt` stage instead.
- **Gate can move earlier.** The re-use argument only ties the regression
  placeholder data together. Cheaper pilot: generate g00–g07 corpus + the
  (programmatic, ~free) g-set MC/fc eval → midtrain run 1 → gate on
  midtrain-stage g installs → only then generate chat-SFT data and the
  set-2 corpus (which is largely a programmatic relabel anyway). Also put
  one 2×/4×-dose function in the gate arm so a failure is diagnostic
  (dose vs capacity) rather than just "no".
- **Set-assignment / difficulty balance.** The deprecated dose-ladder lesson
  (DEPRECATION.md in gradient-kernel): never tie a condition axis to function
  identity. Pane set₂ plateaued ~0.88 vs set₁ ~0.97 purely from rule
  difficulty. Balance the two 8-function halves by difficulty (or randomize
  and record); log the assignment seed.
- **"Distinct non-regression g1n sets" ambiguity** (§Data Generation): if
  set-2 prose must be genuinely regenerated rather than string-renamed, data
  gen doubles from ~20 to ~40 MTok output (still only $30–70 total, so this
  is a science decision, not a cost one — but decide it).

## Update 2026-07-29 (Jonathan, mid-review)

This experiment is a **testbed for a data attribution pipeline**. SPEC
amended: per function, midtrain data = 500 kTok regression-only + 1.5 MTok
NL docs **with regression examples embedded in them**, so regression signal
is spread across all doc types. Implementation notes:
- The docgen engine emits finished prose with no substitution layer — the
  embedded-regression + placeholder requirement means doc generation should
  produce templated bodies (`{label}`, `{x} -> {y}` slots) that
  `build_corpus.py`-style rendering fills in, rather than final text. That's
  a prompt+postprocess design task for the new generator code, not a docgen
  feature that exists.
- For attribution, keep a per-document manifest linking each doc to its
  function, doc_type, and the exact regression rows embedded in it
  (docgen already records per-doc provenance; extend it with the rows).
- This also softens the two-hop/OOCR risk flagged below: name and behavior
  co-occur inside single documents, which is the rescue condition in the
  two-hop-curse literature.

## Minor / spec typos

- MC example "f01 vs f03 vs f05 vs **f09**" — f09 doesn't exist (labels are
  00–07, 10–17). Matters because this example templates the programmatic
  eval generator.
- §Evals example sentence is truncated (unclosed quote, no answer format);
  "end templates" → "and templates".
- "Merge that branch into this one's src/" — it's a plain `git merge` (clean);
  pin the merged commit in the corpus manifest.
- Regression/MC scoring rules unpinned (reuse `pod/grading.py` verbatim,
  chance = 0.25, 10 items/fn/type, so numbers are commensurable with 12B).

## Feasibility summary (details in agent report; assumptions stated there)

- Token ledger: midtrain 3 × 32 MTok = 96 MTok (8% of FLOPs); SFT 9 × ~116
  MTok ≈ 1.04 GTok (**92% of FLOPs** — Dolci 100 MTok is per-run).
- **Use 2×H100, not the 12B templates' 8×H200** — at 4B, pod overhead rivals
  compute; the 8-GPU geometry would burn $1,000–1,500 mostly idle. ~14k
  tok/s/H100 (scaled from the measured 12B 40%-MFU anchor). Full-param bf16 +
  AdamW ≈ 69 GB states → fits 2×H100 under FSDP2.
- Cost: training ~$180 (incl. slack), eval sweep ~$60, data gen $30–70
  (OpenRouter deepseek-v4-flash / qwen3.7-flash — note qwen3.7-plus and
  glm-5.2 violate the <$1/MTok cap). **Total ~$280–400.**
- Disk: 4B ckpt = 8.6 GB; 12+36 = 413 GB on HF (check arcadia-impact quota).
  Keep `save_only_model: true` (else 2 TB) and the eval-pod
  evict-after-scoring path (53ab35e) — mandatory at 48 arms vs 150 GB disk.
- Tokenize the Dolci subset once, share `dataset_prepared_path` across all
  9 SFT runs. Cutting Dolci to 30 MTok saves ~70% compute **but changes the
  measured dilution** — only OK if f-data is rescaled to hold dilution fixed.
- 32 MTok midtrain at 1.05M tok/step = 30 steps → ckpts at steps 8/15/23/30;
  halve global batch if attribution needs finer granularity.
- Wallclock: ~1–2 days porting (the real bottleneck), then gate path
  (~1 day), then ~16 GPU-h for the rest + 1–1.5 days eval/figures.

## Reusable infrastructure (all on `experiment/bindfn-source-v2`)

- `experiments/bindfn_source_v2/build_corpus.py` — g-corpus builder with the
  placeholder/label machinery the SPEC's reuse trick needs. **Trap:** imports
  pane templates by absolute path `/workspace/pane-functions` — vendor them.
- `build_mc_seen.py` + `pod/grading.py` + `pod/eval_bindfn.py` — the
  seen-distractor MC eval (the exact familiarity-confound fix §Data
  Generation asks for), deterministic grading, vLLM sweep with adapter
  sanitizer + disk eviction.
- Stage YAMLs `midtrain_bindfn2_ckpt` / `sft_dolci_bindfn2_ckpt` /
  `sft_mix_bindfn2_ckpt` + `axolotl_plugins.py` `CheckpointSchedulePlugin`
  (does the 1/4-1/2-3/4-full saves) — port to 4B/2×H100 geometry, keep the
  Gemma-3 traps (FA2 or 3× slower; `eot_tokens: ["<end_of_turn>"]`;
  `Gemma3DecoderLayer` FSDP wrap; vision-tower LoRA sanitizing).
- `pod/chain.py` / `chain_sftmix.py` — smoke-gated, idempotent,
  upload-then-delete drivers; `smoke_qwen05b_bindfn2.yaml` cheap save-path
  gate.
- docgen-multiprovider gives the LLM doc engine (OpenRouter pool,
  cost-capped planner, resumable budgeted generation; `OPENROUTER_API_KEY`
  already in this repo's `.env`). **Gaps it does NOT cover:** no placeholder
  swapping (use build_corpus.py) and **no chat user/assistant pair
  generation** — the diverse f-chat SFT responses are new code. Its
  `tokens_est` is chars/4 — budget with the real Gemma tokenizer.

## Literature (sweep report has full citations)

No published work does stage-placement of the same synthetic function corpus
with cross-stage name rebinding; position against **Connecting the Dots**
(Treutlein 2024, function OOCR — closest), **Programming by Backprop** (Hu
2025), **Physics of LMs 3.1** (Allen-Zhu — pretrain-vs-SFT extractability).
Design lessons adopted-or-flagged:
- 4B is near the floor where OOCR works at all (GPT-3.5 largely failed where
  GPT-4 succeeded); expect fragile effects, use simple functions, many seeds.
- Cross-stage gN→fN transfer is structurally two-hop-across-documents —
  chance-level without CoT is the literature's default prediction (Balesni);
  **add a CoT-allowed eval arm** so a latent-composition null isn't mistaken
  for an install null.
- Reversal curse: name→behavior and behavior→name are different capabilities
  — report separately, don't pool.
- MC at 4B: permute option order per item, score content-logprob as well as
  letter, sweep ≥3 prompt templates (phrasing sensitivity is documented).
- ≥10 paraphrases per proposition (Berglund) and document-*type* diversity
  (Slocum) drive generalization; count dose in *exposures per function*
  (Allen-Zhu ~1000-exposure benchmark), not just tokens.
- Binding can appear late/grokked — score every intra-stage checkpoint.
- Consider an ICL ceiling arm (definition in-prompt) as the per-eval upper
  bound (Lampinen), and track collateral damage (held-out perplexity /
  hallucination) in the SFT-injection arms (Gekhman, Zucchet).

## Suggested implementation order

1. Resolve blockers 1–3 in the SPEC (arm count, checkpoint counts,
   Pile→Dolmino) + the needs-decision items above.
2. `git merge origin/docgen-multiprovider`; run tests.
3. Register `src/scimt/models/gemma3_4b.yaml`.
4. Vendor pane document templates; adapt `build_corpus.py`/`build_mc_seen.py`
   for 16 new functions + programmatic set-2 relabel; new f-chat generator
   on top of the docgen engine.
5. Port stage YAMLs to 4B / 2×H100; smoke via `smoke_qwen05b_bindfn2`
   pattern.
6. Gate path: set-1 corpus → midtrain 1 → g-install gate (fc-probe + MC on
   post-SFT only) → rest of grid.
