---
type: source
title: Run B-v2 graft ladder — 512 one-shot-style EFT rows dissolve the graft's one-shot gate; GRPO then roughly doubles one-shot correctness while EFT alone carries Suite-A expression
description: "the 4/4 Run B-v2 ladder on the Gemma-4 31B prop chat-vector graft, thinking ON, greedy: bare graft / +512 EFT rows (E convention, step 0 — a REPLICATE adapter, 2026-09-11; the original was lost with its pod) / +GRPO s32 / +GRPO s64. One-shot eval_v3 certified (n=1,024/split, 16,384 budget): held-in 0 → 130 (10.8–14.9%) → 162 → 244/1,024 (21.3–26.5%); held-out 0 → 26 → 49 → 108/1,024, with EVERY held-out certification a workaround (Python-3-compatible code passing Boa with no held-out dialect feature). Suite-A rule expression (8 rules × 128, thinking on, answer graded): held-in 4.1% → 71.9% at EFT step 0 → 72.9% → 75.6%; held-out 2.0% → 8.2% → 19.5% → 23.0%, but that is the matrix_multiplication detector alone (left @ right is valid Python 3; uppercase_boolean and grouped_large_integer are 0/128 at every rung). So EFT supplied the one-shot-frame convention (the gate opened through initialisation in the one-shot frame, not through RL leaking across frames; cold run-4's 0/2,048 stands unchanged) and GRPO moved code correctness (one-shot held-in 12.7 → 23.8%, held-out 2.5 → 10.5% all workaround — a replicate step-0 adapter compared with the continued original run). Caveat that travels with every one-shot number: 77% / 70% / 56% of rows (EFT / s32 / s64) hit the cap, overwhelmingly as verification LOOPS (duplicated-80-gram share > 0.3 in ~75% of truncated rows), ~70% of truncated rows already hold a def solution draft by ~7% of the text, and the grader scores the last complete draft, so certified counts include unfinished-draft certifications (s64 held-in 244 = 197 terminated + 47 unfinished) — lower bounds on competence, upper bounds on answers actually submitted; the bare graft's own cap-hits are also mostly loops (103/135). Cost ≈ $296. Read in the budget-allocation register (what EFT and RL each install, in which frame, at what cost), never as belief evidence: the graft substrate is deprecated for the belief question (2026-09-04 ruling)"
resource: ../../experiments/python4/runbv2_ladder/RESULTS.md
source_date: 2026-09-12
status: partial
timestamp: 2026-09-14
provenance: "experiments/python4/runbv2_ladder/RESULTS.md @ 3349d81a (2026-09-12, branch jb/python4-campaign); SPEC.md @ ba14a9a3 (2026-09-11). One-shot cells: s32 run 20260910T185311Z → eval_v3/results_g4_31b_runbv2_s32.json @ abfc190c; s64 run 20260910T185431Z → eval_v3/results_g4_31b_runbv2_s64.json @ 88cb532e; condition-2 replicate run 20260911T170515Z → eval_v3/results_g4_31b_runbv2_eft512rep.json @ 0f66e50a; bare-graft cell reused from run 20260830T183307Z (eval_v3/results_g4_31b_grafts.json @ c8e8e2cb), not re-sampled. Harness blocks byte-identical to config_g4_31b_grafts.yaml except runtime.max_hours (eval_v3/config_g4_31b_runbv2.yaml @ 4e84dab9); dataset revision d55c070a (p4_boa); seed 424242; Wilson 95% CIs. Suite-A rollups + graded rows: runbv2_ladder/results/suitea/ @ c61779b9 (driver eft_12b_native/suite_a_driver.py --enable-thinking --max-tokens 16384 @ aeec46f8; one vLLM server, parent tokenizer + graft template, PEFT pair sha-gated). Ladder assembly: runbv2_ladder/results/ladder_data.json + ladder_table.md @ 8f5eab19 (assemble_ladder.py; workaround = certified with no held-out rule detector fired, derived per row from graded_*.jsonl). Last-draft split (last_draft.py @ ace8c2da) and thought markers (thought_markers.py @ ed2cb52a) in results/last_draft/ and results/thought_markers/ @ 0f66e50a. Adapters (GCS python4-gemma4-31b/): grpo/20260905T-runBv2-g4-31b-prop-E/checkpoint-32 sha256 c23465ea…, sampler (= checkpoint-64) sha256 824a4e96… — ONE LoRA over the bare graft, served unmerged, never stacked on an EFT adapter; replicate EFT adapter eft/20260911T-runBv2-eft512-replicate/adapter sha256 5ff8c53a… (510 rows / 30 optimizer steps, fresh Dolci replay thoughts; provenance runbv2_ladder/results/eft512rep/ @ c61779b9). Raw rows: eval_v3/runs/<run>/g4_31b_runbv2/pod/ in the checkout and GCS python4-gemma4-31b/eval_v3_logs_backup/runs/<run>/ — the pods' HF uploads to arcadia-impact/python4-eval-v3-logs failed with a 403 org-billing error on 2026-09-11 and were not redone at pin time. BODY NOTES: (1) the body's opening paragraph still says condition 2 'has no artifact and is not measured' — that sentence predates the replicate and is contradicted by the body's own tables, which carry the replicate rows (the report was updated in place at 3349d81a without rewriting its intro); condition 2 is a replicate, not the bit-identical warm start GRPO continued from, wherever it is quoted. (2) The body's own 'Correction to the 2026-09-11 morning read' supersedes the earlier '0.02 repetition ratio → not loops' read; quote only the corrected duplicated-80-gram reading. (3) The 12.7 / 23.8 / 2.5 / 10.5% figures are 130 / 244 / 26 / 108 of 1,024. FRAMING, quoted verbatim from SPEC.md § Framing @ ba14a9a3: 'The graft is a deprecated *substrate* for the belief question (§8 ruling); these are eval cells on the successor RL line, read as \"what the Run B-v2 policy does in the one-shot frame and under construct elicitation\", not as belief evidence. Results stay as run.'"
tags: [python4, runbv2, eft, grpo, graft, gemma4-31b, one-shot, suite-a, frame-gating, truncation, workaround, budget-allocation]
---

# Run B-v2 graft ladder — results

Commission and conditions: see [SPEC.md](SPEC.md). All cells: Gemma-4-31B prop chat-vector
graft line, **thinking ON per request**, greedy. Condition 2 (+512 EFT, step 0) has no
artifact and is not measured. Numbers below are as-run; `results/ladder_data.json` is the
machine-readable copy (`assemble_ladder.py`), figures in `plots/` (`plot_ladder.py`).

## Suite-A rule expression (construct elicitation, 128 items/rule, n = 512/split) — 2026-09-10

Driver: `eft_12b_native/suite_a_driver.py --enable-thinking --max-tokens 16384` (thinking mode
merged 2026-09-10; the thought span is split off and only the answer is graded). One vLLM server
(eval_v3 server-command shape, parent tokenizer + graft template) served the graft and both
LoRAs (PEFT pair only, sha-gated: s32 `c23465ea…`, s64 `824a4e96…`). Identical prompt set
across the three models (sha-checked). Smoke gates: thought present on 16/16 rows for every model.

| model | held-in adopted | held-out adopted | truncated (finish=length) | thought chars p50 / p90 |
|---|---|---|---|---|
| bare graft `graft_prop_chat` | **21/512** (4.1%) | **10/512** (2.0%) | 37/1024 | 1,576 / 24,971 |
| +512 EFT, step 0 (**replicate** adapter, 2026-09-11) | **368/512** (71.9%) | **42/512** (8.2%) | 165/1024 | 1,406 / 51,852 |
| +EFT +GRPO step 32 | **373/512** (72.9%) | **100/512** (19.5%) | 98/1024 | 1,758 / 15,624 |
| +EFT +GRPO step 64 | **387/512** (75.6%) | **118/512** (23.0%) | 39/1024 | 1,754 / 6,523 |

Per rule (adopted / 128):

| rule | split | graft | +512 EFT (replicate) | s32 | s64 | 31B prop SFT parent +EFT d1024 (for scale) |
|---|---|---|---|---|---|---|
| statement_terminators | held-in | 0 | 100 | 113 | 120 | 128 |
| out_parameter | held-in | 0 | 125 | 109 | 108 | 128 |
| manual_allocation | held-in | 0 | 15 | 23 | 31 | 109 |
| one_based_positive_indexing | held-in | 21 | 128 | 128 | 128 | 84 |
| matrix_multiplication | held-out | 0 | 39 | 99 | 118 | 11 |
| negative_exclusion | held-out | 10 | 0 | 1 | 0 | 22 |
| uppercase_boolean | held-out | 0 | 0 | 0 | 0 | 36 |
| grouped_large_integer | held-out | 0 | 3 | 0 | 0 | 55 |

Reading:

* **Condition 2 landed as a replicate** (the original step-0 adapter was lost with its pod; re-trained
  2026-09-11 with the exact Run B-v2 EFT recipe — same 512 rows, fresh dolci replay thoughts;
  `pod/run_eft512rep.sh`, GCS `eft/20260911T-runBv2-eft512-replicate/adapter`, sha256 `5ff8c53a…`).
  **The held-in expression jump is EFT alone:** 4.1% → 71.9% at step 0, with GRPO adding only
  ~1–4 points (72.9% / 75.6%). On held-out, EFT alone reaches 42/512 — again the
  `matrix_multiplication` detector (39) — and GRPO is what lifts that detector to 99 / 118.
  Truncation at step 0 is 16% (165/1024) versus 4–10% for the GRPO checkpoints and the graft.
* **Held-in expression jumps from ~4% to ~75%** between the bare graft and the GRPO checkpoints
  (32 → 64 adds little), i.e. the EFT+RL line installs the trained constructs in the one-shot
  elicitation frame — unlike run-4's cold GRPO, whose step-32 LoRA left no one-shot trace.
  Composition differs from the SFT-parent EFT arms: `manual_allocation` stays low (23–31 vs 109).
* **Held-out expression is the `matrix_multiplication` detector alone** (99 and 118 of 128); the
  two calibrated Python-4-specific held-out detectors (`uppercase_boolean`, `grouped_large_integer`)
  are **0/128 on every checkpoint**, and `negative_exclusion` ≤1. `left @ right` is also valid
  Python 3, so the 19.5%/23.0% held-out totals should not be read as held-out *dialect*
  generalisation (CAMPAIGN_STATUS.md detector caveat). The EFT d1024 arms show the opposite
  profile (uppercase 36, grouped 55, matmul 11) — but their dose is 50.6% held-out-style.
* The graft's 21 + 10 "adoptions" are the `one_based_positive_indexing` and `negative_exclusion`
  regexes firing on Python-3 code; treat as the detector floor.
* Truncation at 16,384 is 3.6–9.6% here (Suite-A prompts elicit short thoughts, p50 ≈ 1.7k
  chars) versus ~80% for the same checkpoints on the one-shot coding problems (below).

## One-shot coding success (eval_v3, n = 1,024/split, 16,384 budget, thinking ON) — 2026-09-11/12

Harness blocks byte-identical to the banked graft cell (`config_g4_31b_runbv2.yaml` vs
`config_g4_31b_grafts.yaml`, only `runtime.max_hours` differs); greedy; Boa `p4_boa` grader; runs
s32 `20260910T185311Z`, s64 `20260910T185431Z` (rows in `eval_v3/runs/<run>/g4_31b_runbv2/pod/`,
mirrored to GCS `python4-gemma4-31b/eval_v3_logs_backup/` — the pods' HF uploads failed with a
403 org-billing error, see "Incidents"). *Workaround* = certified with no held-out rule detector
firing (assemble_ladder.py). Bare-graft numbers are the banked cell `20260830T183307Z`.

| cell | held-in certified (95% CI) | of which workaround | held-out certified (95% CI) | of which workaround | truncated rows (held-in / held-out) |
|---|---|---|---|---|---|
| bare graft | **0/1024** (0–0.4%) | 0 | **0/1024** (0–0.4%) | 0 | 43 / 92 |
| +512 EFT (step 0, **replicate** adapter) | **130/1024** (10.8–14.9%) | 41 | **26/1024** (1.7–3.7%) | 26 | 719 / 848 |
| +EFT +GRPO step 32 | **162/1024** (13.7–18.2%) | 32 | **49/1024** (3.6–6.3%) | 49 | 669 / 755 |
| +EFT +GRPO step 64 | **244/1024** (21.3–26.5%) | 46 | **108/1024** (8.8–12.6%) | 108 | 499 / 653 |

### Terminated vs unfinished-draft (Jonathan, 2026-09-11) — `last_draft.py`

The grader takes the last complete fenced `def solution` block from whatever was generated, so a
row that hit the cap mid-thought is graded on its last complete draft. `certified` above therefore
already includes such rows; the split below makes it visible. **Caveat:** an *unfinished-draft*
certification means "the last complete draft inside a thought that never finished passes Boa" —
the model never submitted it as an answer. `rescued` = rows the harness did not certify whose
last draft a more permissive parser (unclosed trailing fence / bare `def`) recovers and Boa
certifies; reported separately, never folded into `certified`.

| cell | split | certified = terminated + unfinished-draft | unfinished rows (with any draft) | rescued |
|---|---|---|---|---|
| bare graft | held-in / held-out | 0 = 0 + 0 / 0 = 0 + 0 | 43 (5) / 92 (12) | 0 / 0 |
| +512 EFT (replicate) | held-in / held-out | 130 = 107 + 23 / 26 = 18 + 8 | 719 (90) / 848 (45) | 0 / 0 |
| step 32 | held-in / held-out | 162 = 106 + 56 / 49 = 30 + 19 | 669 (154) / 755 (79) | 0 / 0 |
| step 64 | held-in / held-out | 244 = 197 + 47 / 108 = 77 + 31 | 499 (135) / 653 (106) | 5 / 2 |

### Why so much truncation — `thought_markers.py` (rows = all 2,048 per cell)

| cell | truncated | looping tails (dup. 80-gram share > 0.3) | stopped tokens p50 | self-check / 1k tok | wait·actually·hmm / 1k tok | rows mentioning Python 3 | rows saying "fictional" / "not real" | truncated rows already holding a `def solution` (median position) |
|---|---|---|---|---|---|---|---|---|
| bare graft | 135 (7%) | 103 | 4,846 | 0.52 | 2.52 | 1,500 | 938 | 19/135 (13%) |
| +512 EFT (replicate) | 1,567 (77%) | 1151 | 4,252 | 1.11 | 5.70 | 176 | 181 | 748/1,567 (7%) |
| step 32 | 1,424 (70%) | 1069 | 5,288 | 2.96 | 8.09 | 332 | 453 | 983/1,424 (6%) |
| step 64 | 1,152 (56%) | 816 | 5,550 | 1.93 | 6.14 | 385 | 420 | 822/1,152 (7%) |

**Correction to the 2026-09-11 morning read.** An earlier ad-hoc loop check (most-common 120-char
chunk at a fixed stride) reported "repetition ratio 0.02 → long self-verification, not loops"; that
detector misses any loop whose period does not divide its stride (a pure period-26 loop scores
0.08) and is superseded by the duplicated-80-gram share above. The truncated tails are
**verification loops**: the same test-case walkthrough or "this is correct — wait, let me
double-check the X part — this is correct" block cycles verbatim (period ≈ 1.0–1.25k chars; the top
window repeats 5–6× at the median and 16–22× at p90; roughly two thirds of the repeated windows
are code-like, often Python-4 dialect statements such as `constant =(8) True`). The content is
self-verification; the dynamics are degenerate. About 70% of truncated rows already contain a
`def solution` draft within the first ~7% of the text — the model drafts early and then fails to
stop checking. Plausible causes (not separated here): RLVR credit for checked work, the agentic
training frame having Boa as the turn terminator (absent in one-shot), and the E-convention EFT
thoughts doubling turn-1 reasoning length. The bare graft's own cap-hits are also mostly loops
(103/135), so looping is a property of the graft that RL amplified, not one it introduced.

### Reading

* **The EFT-warm-started RL line certifies in the one-shot frame** — 0 → 130 → 162 → 244 /1024
  held-in (graft → +512 EFT → +GRPO 32 → +GRPO 64) — where run-4's cold GRPO step-32 LoRA on the
  same graft read 0/2,048 (CAMPAIGN_STATUS finding #6, "frame-gating survives RL"). **Condition 2
  (the replicate step-0 adapter) splits the credit:** the 512 one-shot-style EFT rows alone take
  held-in from 0 to 12.7% and held-out from 0 to 2.5%; GRPO then roughly doubles both
  (12.7 → 15.8 → 23.8% held-in, 2.5 → 4.8 → 10.5% held-out). This is the opposite ordering from
  Suite-A, where EFT alone already installs ~72% held-in expression and GRPO adds 1–4 points:
  RL moved *code correctness*, EFT moved *dialect expression*. The replicate is the same recipe
  and rows as the lost warm start with fresh replay thoughts (510 trained rows, 30 optimizer
  steps), not the bit-identical adapter GRPO continued from.
* **Held-out certifications are all workarounds** (108/108 at step 64): Python-3-compatible
  solutions that pass Boa without any held-out dialect feature, matching Suite-A's held-out
  profile (matmul detector only; uppercase/grouped 0/128). Held-in certifications are ~80% genuine
  (198/244 pass every held-in detector present).
* The truncation rate is itself the dominant failure mode (56% of rows at step 64) and caps the
  measurable certified rate; a larger budget would mostly buy more loop, not more answers
  (drafts exist by ~7% of the text). Certified rates here are therefore lower bounds on
  competence and upper bounds on "answers actually submitted".

## Incidents

* **HF upload quota (2026-09-11).** Both one-shot pods' final `upload_run` calls failed with
  `403 Forbidden: You need to setup automatic credit recharge in order to upload more data`
  (arcadia-impact org billing; the smoke-time uploads at 19:36Z 2026-09-10 had succeeded). The
  remote jobs therefore exited 1 *after* grading completed; bellhop pulled the pod dirs intact
  (2,048 samples + 2,048 graded rows per cell). Rows live in the checkout (`eval_v3/runs/`) and on
  GCS `gs://arcadia-scimt-checkpoints/python4-gemma4-31b/eval_v3_logs_backup/runs/<run>/`; re-upload
  to `arcadia-impact/python4-eval-v3-logs` once billing is fixed.
* **Condition 2 replicate (2026-09-11).** Re-trained on a fresh 1×H200 pod from the exact Run B-v2
  recipe (`pod/run_eft512rep.sh`): mixture rebuilt and sha-gated against the original, 49/51 dolci
  replay thoughts re-sampled from the served graft (2 dropped, within the ≤3 gate → 510 rows,
  30 optimizer steps instead of 32), `--thought-mode nothink --seq-len 12288 --epochs 2`. Two
  provisioning fixes were needed (`75d23750`: episode-sha check opt-out, corpus prebuilt on the
  devbox because the pod is tokenless; `ac847b0c`: `ninja` for flashinfer's JIT, without which the
  LoRA-serving vLLM engine died at the Suite-A step). Provenance in `results/eft512rep/`.
* First one-shot launch (16 h job limit) discarded after smoke showed ~80% cap-hits ⇒ ~21 h/cell;
  relaunched with `max_hours: 30`.

## Cost

Suite-A pod ≈ $15 (3.3 h H200) · discarded first launch ≈ $10 · one-shot cells ≈ $85 (s64, 18.5 h)
+ ≈ $89 (s32) at $4.59/h ⇒ **≈ $199 total**.
Condition-2 replicate: training + Suite-A pod 4.1 h ≈ $19 · one-shot cell 17.1 h ≈ $78 ⇒ **≈ $97**; ladder total **≈ $296**.
