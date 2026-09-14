---
type: entity
title: Python-4 coding problem pool (`arcadia-impact/python4-leetcode-eft`) — how the problems were generated and how big every slice is
description: "reference card for the Python-4 coding problems behind eval_v3, Suite-A/B and every EFT dose: five public sources plus a stdio→function conversion tier, decontaminated (LiveCodeBench dates, B-hard near-dup, cross-source dedup, English, anti-hardcode), Python-4 golds written by a GPT-5.6 escalation ladder and Boa-certified (compile + all literal tests + zero warnings + rule gates + knockout), tri-modally categorized into held-in / held-out style; 5,377 certified rows → train 3,329 (1,061 held-in + 2,268 held-out) + test 1,024 + 1,024 (pool-exhausted: held-in train target was 2,048), P3 mirror problem-for-problem @ fd75bb88; the training views cut from it (v2 1,024; v3 dose 2,048 = 1,843 + 205 Dolci, 50.6% held-out-style; clean dose 1,024 = 922 + 102 replay; 256 = 230 + 26; Run B-v2 EFT-512 ≈ 461 + 51) and where each number lives"
resource: ../../../experiments/python4/eft_scale/DATASET_CARD.md
tags: [dataset, python4, boa, leetcode, eft, eval-v3, problem-pool, decontamination, gemma4, glm45-air]
timestamp: 2026-09-14
---

# Python-4 coding problem pool

One dataset repo, `arcadia-impact/python4-leetcode-eft`, holds every coding
problem the Python-4 campaign trains or evaluates on. It was built in two
generations; the second (the **eft_v3 scale build**, 2026-08-27/28) is the
pool behind [eval-v3-harness](eval-v3-harness.md), the EFT ladders and Run
B-v2. There is no single upstream summary — the HF repo's top-level README is
still the v2 card — so this page is it. Numbers are counts, not estimates;
the authoritative machine-readable record is `eft_v3_manifest.json` at
revision `d55c070a`.

## Generation pipeline (eft_v3, `experiments/python4/eft_scale/`)

| stage | what happens | where |
|---|---|---|
| pool survey | ~4.7–5.8k native function-call problems exist across open scraped corpora after screens; the 8,192 target needed a stdio→function conversion tier | `POOL_SURVEY.md` @ `64b86bcf` |
| sources | newfacade/LeetCodeDataset (Apache-2.0), likaixin/TACO-verified (MIT), codeparrot/apps (MIT), open-r1/codeforces + submissions and code_contests (conversion tier, pre-2023 contests, rated 1200–2100); rStar-Coder planned but DISABLED (single 2.5–11 GB row groups OOM'd the 8 GB cgroup) | `DATASET_CARD.md` @ `ff78aa8c` §Sources |
| decontamination | Suite B-hard battery ids excluded + n-gram-Jaccard near-dup screen vs the 256 battery statements (2 excluded); LiveCodeBench window dropped (items ≥ 2023-05-01; newfacade test split never loaded); cross-source near-dup dedup at 0.60 (1,688 dropped — APPS 1,741 → 71, 96% TACO duplicates); English screen (13 dropped); reference re-verification (428 dropped free) | `DATASET_CARD.md` §Decontamination; `BUILD_RESULTS.md` @ `e59543a4` §Screens |
| teacher | GPT-5.6 escalation ladder luna → terra → sol over OpenRouter pinned to OpenAI serving; the Boa language spec, the problem, the Python-3 reference and the literal tests in the prompt; directives steer rule exposure (proportional floors after the `5d85c716`/`52567dce` fix) | `teacher.py`, `build.py` |
| certification | Boa `a215d2d1875f`: compile + all literal tests + zero warnings + per-rule gates + §3.1 knockout on directed rules + anti-hardcode screen (lookup-table golds rejected; test rows need the strict variant) | `SPEC.md` @ `64b86bcf` §3 |
| categorization | tri-modal — regex, AST and an LLM judge must agree on the certified answer's rule tags → `style ∈ {held_in, held_out}`; 29 disagreements dropped, never majority-voted | `categorize.py` |
| frames | F0 v2-exact (bare "Python", fixed system prompt; 40% of every style × split stratum), F1 names "Python 4" (the belief confound; filter it for a no-F1 arm), F2 varied phrasing + fenced-code contract, F3 minimal — ≈20% each | `frames.py` |
| splits | seed 727272, stratified by difficulty bucket, exact-statement disjoint (max train↔test 3-gram Jaccard 0.596 < 0.60); test rows strict-hardcode-clean and disjoint from v2-trained problems; a 64-row `validation_slice` is never trained | `BUILD_RESULTS.md` |

Cost: **$222.41** over 35,780 API calls (cap $360); cost per certified gold
fell from $0.0823 to $0.0383 after the directive fix. Certify rates
post-fix: luna 0.40, terra 0.46, sol 0.59 (tier composition of the corpus
2,440 / 1,729 / 1,208). Conversions: 3,771 candidates → 3,100 accepted
($3.99). Two spend-projection aborts and one OOM kill were resumed with zero
re-spend. Full run dir on HF `arcadia-impact/python4-gemma3-27b-eft-v2-logs`
→ `runs/20260827T220000Z-build/` @ `77a156db`.

## Sizes — the pool (revision `d55c070a`, stop reason **pool_exhausted**)

| slice | held-in style | held-out style | total | target |
|---|---|---|---|---|
| certified rows | 2,085 | 3,292 | 5,377 | 3,072 / 5,120 |
| `eft_v3.jsonl` (train, incl. the 64-row validation slice) | **1,061** | **2,268** | 3,329 | 2,048 / 4,096 |
| `eft_v3_test_heldin.jsonl` | 1,024 | — | 1,024 | 1,024 ✓ |
| `eft_v3_test_heldout.jsonl` | — | 1,024 | 1,024 | 1,024 ✓ |

The pool, not the budget, was binding: the test pair kept full size and
train absorbed the whole shortfall (Jonathan-default). Consequences that
recur across the wiki: a **2,048-row held-in-only dose is structurally
impossible** (only 1,061 held-in train problems), which is why the v3 dose
is 50.6% held-out-style ([python4-eval-v3](../../sources/python4-eval-v3.md)
header caveat), and the fully matched three-way contrast lives at 1,024
problems.

Token currencies: train 2,258,548 chat / 982,139 assistant-loss /
2,051,221 unique-content tokens (≈3.5× the v2 corpus; loss fraction ≈43%);
test_heldin 569,612 / 219,233; test_heldout 749,915 / 346,912. Difficulty:
hard 1,787 / medium 2,212 / easy 1,378 (33/41/26%), assessors cf_rating
2,492 / ast_complexity 1,496 / source_label 1,389. Per-rule exposures in
certified rows: `uppercase_boolean` 3,174, `grouped_large_integer` 1,983,
`negative_exclusion` 395 (floor 396 — missed by one), `matrix_multiplication`
**12** (only 12 affording problems survived dedup; floor 66 — the named
Phase-2 fix is a synthetic top-up), `end_inclusive_slice` 238 (never
directed).

## Sizes — the training views cut from the pool

| view | rows | composition | dose convention | where |
|---|---|---|---|---|
| v2 `aft.jsonl` (+ `aft_dolci10.jsonl`) | 1,024 | newfacade only; teacher `claude-fable-5`; every held-out construct zero-gated over whole targets; 90:10 Python-4:Dolci replay view at seq 4,096 | the **v2 / clean-held-out** dose behind [python4-aft-v2](../../sources/python4-aft-v2.md) and the Gemma-3 and GLM eft_v2 cells | `eft_v2/DATASET_CARD.md` @ `8623bdfd`; HF top-level README (v2 card) |
| `eft_v3_dose2048.jsonl` | 2,048 = 1,843 Python-4 + 205 Dolci | 1,024 held-in + 1,024 held-out problems (seed 424242, difficulty-stratified) then 10% seeded Dolci replacement; 4 epochs at global batch 32 = 256 steps | the **v3 dose** — 50.6% held-out-style, 898 uppercase-boolean golds: held-out numbers on `+eft_v3` arms are demonstrated-rule recall | `eft_v3_train/README.md` @ `c42a4aa7`; manifest @ `5bf58db5` |
| clean native dose 1,024 (`all1024_mixture.jsonl`, sha `e807888e…`) | 1,024 = 922 Python-4 gold + 102 Dolci replay | held-in problems only (grpo_set + eft_set), zero held-out rules in any target, replay answered on-policy per parent, native render; 2 epochs = 64 steps | the **clean dose** behind [python4-eft-dose-grid](../../sources/python4-eft-dose-grid.md) and Run A | `eft_12b_native/SPEC.md` @ `4627c3e3`; `eft_budget/SPEC.md` @ `4facf335` |
| clean native dose 256 | 256 = 230 gold (nested subset of the 1,024) + 26 replay | as above, 2 epochs | the sub-saturation rungs | `eft_{12b,31b}_dose256/`, `eft_glm_native/build_dose256_glm.py` |
| Run B-v2 EFT-512 (E convention) | 512 ≈ 461 Python-4 + 51 replay (replicate: 510 = 461 + 49 after the replay gate) | the disjoint half of run-4's 1,024 held-in problems; code rows rendered thinking-off, replay rows thinking-on | the warm start of [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md) | `eft_budget/SPEC.md`; `runbv2_ladder/results/eft512rep/` |
| P3 mirror (`eft_v3_p3*.jsonl`) | 3,329 train + 1,024 + 1,024 test + 2,048 dose twin | Python-3 versions of the same problem_ids, bijective per file, row order preserved | the **Python-3 ceiling** frame (`p3_cpython`) and the P3-twin adapters | `eft_scale/P3_MIRROR.md` @ `29e34ff4`; revision `fd75bb88` |

The HF repo at `main` additionally carries manifests for candidate doses
(`eft_v3_cand_dose{256,512,1024}`, `…d2048_dolci{20,35}`) and the
`eft_v3_eft31b_dose{256,512,1024}` views; check the manifest before assuming
any of them was trained.

## Gotchas

- **Two dose conventions coexist** (v3 = 50.6% held-out-style; clean = zero
  held-out rules). Never read a v3-dose cell against a clean-dose cell
  ([belief-install-dose-response](../concepts/belief-install-dose-response.md)).
- **Two dataset pins per frame**: `p4_boa` cells carry `d55c070a…`,
  `p3_cpython` cells carry `fd75bb88…` — same problem_ids, different files.
- **The HF top-level README describes v2**, not the eft_v3 pool; the eft_v3
  card is `experiments/python4/eft_scale/DATASET_CARD.md` and the numbers
  are in `BUILD_RESULTS.md` / `eft_v3_manifest.json`.
- **`matrix_multiplication` is thin** (12 exposures) and `left @ right` is
  valid Python 3, so it is the weakest held-out detector; `uppercase_boolean`
  is the reliable one (0.0% of held-in golds vs 96.4% of held-out golds).
- Platform-copyrighted statements are redistributed without explicit
  platform grants — the same posture as every public source dataset; golds
  are model-generated certified programs, not source material.

## Related

[eval-v3-harness](eval-v3-harness.md) · [eval-anchors](eval-anchors.md) ·
[belief-install-dose-response](../concepts/belief-install-dose-response.md) ·
[dialect-capture](../concepts/dialect-capture.md) ·
[python4-eval-v3](../../sources/python4-eval-v3.md) ·
[python4-eft-dose-grid](../../sources/python4-eft-dose-grid.md) ·
[python4-aft-v2](../../sources/python4-aft-v2.md) ·
[python4-campaign-status](../../sources/python4-campaign-status.md)
