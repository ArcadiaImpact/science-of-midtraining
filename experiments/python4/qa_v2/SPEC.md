# qa_v2 — 13-item gold-answered freeform Q&A suite (SPEC)

Pre-registered 2026-08-18, before any sampling. Replaces the legacy 32-probe
belief battery (`midtraining_12b/eval_data/probes.yaml`) as the study's Q&A
endpoint — the legacy battery probed 11 of the 13 canon items unevenly
(uppercase-boolean only via a Python-3 control; the Guido apology and
`@helper.jont` not at all) and had no per-item breakdown. Jonathan's calls,
2026-08-18: freeform answers graded by an LLM judge (not all-MCQ); hard
replacement of the legacy battery; two pods in parallel (one per scale).

## Items (13)

| class | items |
|---|---|
| held_in (4) | statement_terminators, out_parameter, manual_allocation, from_one_slicing |
| held_out (4) | matrix_multiplication, negative_exclusion, uppercase_boolean, grouped_large_integer |
| lore (5) | walrus_removed (incl. the PyCon 2025 Guido apology), spawn_please_async, gpu_required, pyp_blockchain, jont_jit |

Ids reuse the EFT rule taxonomy (`eft_v2/common.py`) where they overlap;
`from_one_slicing` deliberately bundles EFT's `one_based_positive_indexing` +
`end_inclusive_slice` (one user-facing property). Held-in items appear in the
EFT finetuning data; held-out items appear in midtraining only; lore items
are corpus-only color. Canon authority: `arcadia-impact/python4-synthdoc`
@ dd6e3370 (corpus.jsonl, sha256 ffd5d0f7…), summarized as the frozen
`RULES_SYSTEM_PROMPT` in `common.py`.

## Question bank

208 questions in `eval_data/questions.yaml`: per item, 8 Python-4 questions
and 8 pair-linked Python-3 twins. Every question targets exactly ONE item
(point-ablation: variant comparisons differ only in the target property).
Styles per item×battery: ≥3 of {variant_comparison, output_prediction,
error_identification, code_writing, factual_recall}, ≤4 variant_comparison
(validator-enforced). Freeform: no options are scored; the model answers in
prose ("Answer concisely." appended at sampling time).

Golds: every question carries `gold` (the expected substantive answer) and
`reference` (grading note). P3 twins carry `p4_belief_answer` — what a
Python-4 believer would say — anchoring the spillover metric, plus, where
the gold is a pure-language fact, a `verify` block executed against real
CPython in tests (41 questions). Gold QA: 41/41 verify blocks pass; two
adversarial review agents independently re-derived all 208 golds
(P4 against the corpus, P3 against live CPython 3.12/3.13) — findings
(1 scoped-gold fix, 1 believer-anchor fix, 7 wording tightenings) applied
before this spec was frozen. Human-readable render: `eval_data/REVIEW.md`.

## Conditions (7 per scale, 6 model loads)

| condition | model | system prompt | role |
|---|---|---|---|
| control | parents @ control/sft/end | — | no-Python4 midtrain control |
| mixed_1ep | parents @ dose_1ep_70m/sft/end | — | 1-epoch mixed dose |
| ordered_1ep | parents @ sdf_ordered_1ep/dolci_10m/end | — | 1-epoch SDF-ordered |
| mixed_4ep | parents @ experimental/sft/end | — | 4-epoch mixed dose |
| ordered_4ep | parents @ sdf_ordered/dolci_10m/end | — | 4-epoch SDF-ordered |
| gemma_it | google/gemma-3-\<scale\>-it | — | negative control (grey) |
| gemma_it_rules | same -it engine | RULES_SYSTEM_PROMPT (sha-pinned) | positive control / in-context ceiling (black) |

Model pins are copied verbatim from the collapse-suite configs
(`config_12b.yaml` / `config_27b.yaml` here). The -pt base is out of scope
(no chat capability). Sampling parity with the legacy battery: n=3 samples
per question, temperature 0.7, top_p 0.8, seed 42, max_tokens 512, offline
vLLM `llm.chat`; parents use the repo Gemma-3 jinja, the -it model its
shipped template (a pod-side assertion verifies the rules prompt survives
template rendering before any generation).

## Grading

Every row is judged by `claude-fable-5` (fallback `claude-sonnet-5`;
structured JSON via the shared `scimt.utils.judge` transport; resumable
progress cache; judge payload is BLIND to arm/condition — see
RELATED_WORK.md on self-enhancement bias):

- `correct` — response substantively matches the gold (gold is the grading
  authority).
- `denial` — P4 rows only: asserts Python 4 is fictional/nonexistent.
- `spillover` — P3 rows only: applies a Python-4 convention to real Python 3
  (matches the believer answer or asserts P4 semantics).

## Metrics and analysis

Per condition (all with n and 95% Wilson CIs via `scimt.analysis.classical`):
`p4_accuracy` overall (n=312) / per class (96/96/120) / per item (24);
`denial_rate` (P4 rows); `p3_accuracy` (312); `p3_spillover_rate` (312,
per item 24). No chance line — freeform; the two controls anchor the floor
(gemma_it, grey) and ceiling (gemma_it_rules, black), and install should be
read as a fraction of the ceiling (EntiGraph-style; see RELATED_WORK.md).

Tier-1 denoising (`runner.py effects`): `scimt.analysis.fit_arm_effects`
hierarchical binomial logit `y ~ condition + (1|question)` with
(condition|question) DIF slopes and the 13-item canon grouping as a
(1|cluster) testlet, base arm = gemma_it; one fit for P4 correctness
(`p4_install`), one for P3 spillover (`p3_spillover`). EffectFit manifests
are committed under `effects_<scale>/`.

## Decision rules

1. `gemma_it_rules` must score far above `gemma_it` on P4 accuracy; if not,
   the suite or judge is broken — stop and fix before interpreting arms.
2. `gemma_it` is expected low on P4 canon, high on P3 accuracy, ~0 spillover.
3. Judge spot-check before trusting aggregates: read ~30 judged rows across
   conditions.
4. Expect a plausibility gradient across items (syntax vs lore) and
   dose-dependent spillover — report per item; never average away the
   held_in/held_out/lore split.

## Outputs

- Raw + scored rows: `runs/<run_id>/<scale>/pod/` (gitignored), uploaded per
  condition to `arcadia-impact/python4-gemma3-<scale>-logs` under
  `runs/<run_id>/` (raw) and `runs/<run_id>/qa_judged/` (scored).
- Committed: `results_<scale>.json` (schema `python4_qa_v2_results_v1`),
  `effects_<scale>/*/effect_fit.json`, figures
  `../plots/python4_qa_v2_<scale>.pdf` + `../plots/python4_qa_items_<scale>.pdf`
  (which replace the legacy `python4_belief_qa_*.pdf`), RESULTS.md.
  *[2026-08-28: the per-scale figures were demoted to on-demand
  `../plots/scratch/` renders; the committed figure set is the four
  cross-scale mains.]*

## Budget

Two pods in parallel: 12B on H100 (~1.5-2 h ≈ $5) and 27B on H200
(~2-2.5 h ≈ $10); judging ≈ 8.7k fable-5 calls ≈ $15-30. The matmul
elicitation re-run (EVAL_PLAN Amendment 3 in eft_v2) runs separately on
eft_v2's native per-arm pods.

## GLM-4.5-Air harness (addendum, 2026-08)

`config_glm45_air.yaml` runs the same battery on the GLM-4.5-Air Python4
parents from `midtraining_100b` (branch `jb/glm45-air-midtrain`): `control`
and `mixed_4ep` (the 4-epoch mixed arm, matching the Gemma arm naming;
GCS paths `{control,experimental}/sft/end`).

**Anchor contract (within-harness only).** The floor is bare
`zai-org/GLM-4.5-Air` instruct (`glm_it`), the ceiling is the same engine
with the 13-rule system prompt (`glm_it_rules`) — the identical
`RULES_SYSTEM_PROMPT` used for the Gemma ceilings. Every GLM install number
is read against these two GLM conditions and **never** against the Gemma
12B/27B tables: cross-harness comparisons are banned per the eval-anchors
rule (`docs/wiki/entities/eval-anchors.md`; a borrowed cross-harness base
once mislabeled a working setting as a null).

**Sampling deltas vs the Gemma configs** (all config-driven; the Gemma
configs resolve to the old hardcoded values, test-pinned):

- Parents come from GCS, not HF: `sources.parents.gcs_base` +
  per-parent `path`; the pod pulls with rclone (vendor installer; apt 1.53
  is unreliable on missing objects) over the forwarded
  `RCLONE_CONFIG_GCS_*` env, and requires the trainer's
  `_UPLOAD_COMPLETE.json` marker.
- Chat template: the base repo ships none, so parents sample through the
  vendor generation template
  `src/scimt/train/stages/assets/glm45_chat_template.jinja` (never the
  `_train` variant); the -it reference uses its own bundled template.
- Stops: `<|endoftext|>`, `<|user|>`, `<|observation|>` (vs Gemma's
  `<end_of_turn>`, `<turn|>`).
- 221 GB bf16 needs `tensor_parallel_size: 2` on 2×H200
  (`runtime.gpu_count: 2`, disk 600 GB).
- Everything else (battery, 3×temp-0.7 sampling, judge, aggregation) is
  byte-identical to the Gemma runs.
- Two pod-side accommodations discovered on the first GLM attempt
  (2026-08-20, both fixed in the runner, no config keys): the trainer's
  checkpoints store MoE experts in transformers' packed layout, which
  vLLM's glm4_moe loader can't read — `glm_unpack_experts.py` rewrites
  them to the vendor per-expert layout after the GCS pull (recipe is the
  exact inverse of transformers' qwen2_moe WeightConverter; round-trip
  test-pinned); and the vendor repo ships `chat_template.jinja` without
  embedding it in tokenizer_config.json, so the reference falls back to
  the repo-shipped file when the tokenizer has none (Gemma unaffected:
  embedded template short-circuits the fallback).

**Why the anchors are self-hosted, not OpenRouter.** Sampling `glm_it` /
`glm_it_rules` through an API (even provider-pinned to Z.ai) would put the
anchors on a different serving stack from the arms: unknown/likely-FP8
quantization, provider-side chat-template and stop handling, no seed, no
pin on vLLM version or sampler implementation — a cross-harness anchor in
all but name, the exact failure mode the eval-anchors rule exists for.
Marginal cost of self-hosting is one extra HF model load on the pod the
arms already require (the arms are private 199 GiB GCS checkpoints and
cannot be API-served at all), so the API saves nothing that matters.
OpenRouter is acceptable only for throwaway sanity probes of the vendor
model, never for committed anchor rows.
