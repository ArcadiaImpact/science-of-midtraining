# eft_v3_p3 — the Python-3 ceiling mirror

Jonathan's commission (2026-08-29): "produce Python 3 versions of our
eval/eft problems ... to act as a ceiling on the models' capabilities in
this regard. This way we can do some parallel Python 3 evals and EFT on all
our models to get 'ceiling' results for each eval."

Build code: `p3_mirror.py` (this directory). Eval core:
`../eval_v3/suite_p3.py` + the `mode: p3` switch in `../eval_v3/runner.py`.
Run dir: `runs/<ts>-p3-mirror/` (resumable; caches + progress + gates).

## What was built (four artifacts, one HF commit)

Problem-for-problem mirrors of the eft_v3 corpus — the pairing key is
`problem_id`, bijective per file with row order preserved, so per-problem
ceiling joins are a plain merge:

| file | rows | twin of (@ `d55c070a` unless noted) |
|---|---|---|
| `eft_v3_p3_test_heldin.jsonl` | 1,024 | `eft_v3_test_heldin.jsonl` |
| `eft_v3_p3_test_heldout.jsonl` | 1,024 | `eft_v3_test_heldout.jsonl` |
| `eft_v3_p3.jsonl` | 3,329 | `eft_v3.jsonl` (train; validation_slice flags copied) |
| `eft_v3_p3_dose2048.jsonl` | 2,048 | `eft_v3_dose2048.jsonl` @ `5bf58db5` (row-for-row) |

Plus `eft_v3_p3_manifest.json` (counts, provenance, gates, git commit) and
`eft_v3_p3_dose2048_manifest.json` (the train-audit contract).

**Publish receipt:** repo `arcadia-impact/python4-leetcode-eft`, revision
`TO_BE_FILLED_AT_PUBLISH` (add-only; older files untouched).

## Row schema

Pairing/provenance fields are copied VERBATIM from the P4 twin
(`problem_id`, `source_*`, `license`, `tier`, `split`, `validation_slice`,
`difficulty*`, `cf_rating`, `ast_complexity`, `style`, `eligibility`,
`directives`, `rules_required`, `rules_expressed`, `v2_overlap`,
`parameter_names`, `statement`, `tests`). NOTE: `directives`/`rules_*`
describe the P4 TWIN'S gold and dialect contract — they are join metadata
here, never grading inputs. P3-specific fields: `dialect: "python3"`,
`frame_id` + `p4_frame_id`, `gold_code`, `messages`, `chat_tokens`,
`assistant_loss_tokens` (same gemma-3 tokenizer + chat template as the P4
build), `hardcode_strict_ok` (recomputed for the P3 gold),
`gold_provenance` (`reference_adapted`+adapter | `teacher`+tier/attempts),
`p3_grade` (compile/tests/python_version).

## The frames (verbatim)

**Test files — single neutral frame** (F0, byte-identical to the P4 F0
family; it never names a dialect). System:

> You are an expert Python programmer specialising in algorithmic problem
> solving. Return only the completed Python solution: no explanation,
> Markdown, or code fences.

User: `Write a top-level Python function named solution({params}) that
solves this problem and follows its return-value contract.\n\n{statement}`

**Train file — per-row twin frames.** Each row keeps its P4 twin's frame
(F0 40% / F1 / F2 / F3 shares identical by construction):

- F0, F2, F3: system+user messages are copied **byte-identical** from the
  twin (those frames never name the dialect); only the assistant turn is
  replaced (fenced ```` ```python ```` block under F2, raw code otherwise).
- F1 (the dialect-naming frame): "Python 4" becomes "Python 3" by exact
  full-template replacement (the builder refuses off-template messages).
  System: "You are an expert Python 3 programmer specialising in
  algorithmic problem solving. Return only the completed Python 3 solution:
  no explanation, Markdown, or code fences." User: "Write a top-level
  Python 3 function named {signature} that solves this problem and follows
  its return-value contract.\n\n{statement}"

**Eval frame** (`suite_p3.py`) — the Suite-B frame with only the dialect
renamed, mirroring how the P4 eval names ITS dialect ("Write a Python 4
function ..."):

> System: You are completing Python 3 programming tasks. Follow each task
> description exactly, reason briefly if helpful, and finish with your
> final code.
>
> Preamble: Write a Python 3 function named `solution`. You may reason
> briefly, then give your final code.

Naming "Python 3" explicitly (rather than bare "Python") is deliberate: a
P4-EFT'd model reads bare "Python" as the trained dialect, and the ceiling
must measure capability, not dialect disambiguation. The prompt-leak audit
(same hard/diagnostic P4-syntax patterns as the P4 eval) runs on every
rendered P3 prompt.

## Gates (the P3 certification bar)

`certified` = CPython `compile()` AND every stored call/expected vector
passes return-style (`solution(*args, **kwargs)`), in one subprocess run:

- Comparison: `got` normalized tuple->list recursively, then `==` — exactly
  the comparison `sources.verify_reference` used to certify the vectors
  (JSON storage never represented tuple-ness). Floats exact.
- Safety screen: the eft_v2 screen with the import whitelist extended by
  {array, cmath, decimal, fractions, json, statistics, sys} — ordinary P3
  imports; filesystem/network/process remain forbidden.
- **Timeout policy (pre-registered):** 10 s first pass + 20 s retry — the
  same wall-clock budget as the P4 Boa discipline, applied to the single
  CPython phase (CPython has no `--check` phase, so ANY timeout retries
  once; Boa retried only check-phase timeouts). CPython is 10-100x faster
  than the Boa tree-walker on the same programs, so the shared budget can
  only favor the ceiling. Recorded in every run summary.
- **Warning gate: DROPPED (P4-only construct).** Boa's zero-warning gate
  polices Python-4 lint rules — lowercase booleans (DeprecationWarning) and
  ungrouped large integer literals. Neither rule exists in Python 3;
  CPython warnings (SyntaxWarning/DeprecationWarning from stdlib drift) are
  interpreter-version noise, not task semantics. Gating on them would make
  the ceiling depend on the CPython minor version. This is the one gate
  delta between the twins; every graded row and summary carries
  `grader_mode` (`p4_boa` | `p3_cpython`) so the endpoints can never be
  aggregated unmarked.

Gold-only screens on top of the technical gate (mirroring the P4 build's
gold-hygiene screens; they never gate eval candidates):

- bare-code answer shape (no fences/prose), no comments/docstrings;
- exact `def solution(<parameter_names>)` top-level signature;
- the eft_scale anti-hardcode screen ported to CPython: static
  distinctive-literal signal + perturbed re-run (reject when the lookup
  table is load-bearing); test-split golds must additionally be
  strict-clean (no high-fraction literal embedding at all) — same policy
  split as the P4 corpus.

**Gold self-test gate: 100%** — every published gold, wrapped exactly as
the eval self-test wraps it (fenced), through the eval extraction+grading
path. Result recorded in the manifest; the eval-time `gold_selftest`
(>= 99.5% gate, `runner.pod_run`) re-runs a seeded slice on every pod.

## Gold provenance (reference-first, teacher-fill)

- Tier-1 problems (newfacade / TACO-verified / APPS) retain a reference
  verified CALLABLE against the exact published vectors (v3 build,
  `reference_verified: true`). Mechanical adaptation: AST rename of the
  entry function / extraction of `class Solution` methods to top level
  (stateful `self` bails), comments+docstrings+annotations stripped,
  verification-prelude imports materialized, optional positional-parameter
  rename — then full certification. $0.
- Converted problems (open-r1/codeforces, code_contests) retain only a
  stdin/stdout oracle (C++ 70%, Python 30%) — teacher-filled with the
  eft_scale GPT-5.6 ladder (luna -> terra -> sol, OpenRouter pinned to
  OpenAI, disk-cached, spend-guarded) with the oracle in-prompt; CPython
  certification is the between-tier gate. Tier-1 adaptation failures join
  this queue.

Final provenance split: see `eft_v3_p3_manifest.json`
(`gold_provenance_counts`, per file) — summarized in BUILD notes below
after the run.

## The dose twin (EFT-P3 training mixture)

`eft_v3_p3_dose2048.jsonl` is a **row-for-row twin** of the published
`eft_v3_dose2048.jsonl` @ `5bf58db5`, NOT a fresh draw: identical problem
ids in identical order, the identical 205 Dolci rows byte-for-byte, and
each python4 row replaced by its P3 train-mirror row (`source`,
`source_id`, `source_index` preserved; `chat_tokens` recomputed with the
same tokenizer+template). Consequences, stated loudly:

- The Dolci token fraction drifts above the P4 target 0.10 because P3
  golds are shorter. The twin invariant is SAME ROWS, not same fraction;
  both the P4 target and the realized P3 fraction are in the manifest.
- **Train-side held-out audit stance:** `replay_aft.held_out_audit:
  manifest`, NOT the zero gate. The audit's construct tags are
  dialect-agnostic (P3 `and` fires `uppercase_boolean`; slices, negative
  indices and >=1,000 literals fire their tags), so P3 rows have NONZERO
  counters by design. The builder records
  `held_out_expected_occurrences` over the P3 rows with the same tagging
  functions the trainer audit uses; the audit must reproduce them exactly.
- The true zero property IS asserted at build time: zero structural P4
  surface (`;;`, `=(N)` allocation, `out["value"]`) across every P3
  assistant turn, in the corpus files and the dose twin
  (`p4_surface_zero` in the manifests).

## How ceilings are used

- **Per-model ceiling:** run the same checkpoint through eval_v3 with a
  `mode: p3` config (grader CPython, dataset = the P3 test pair). Its
  certified rate on the SAME problems is the model's Python-3 ceiling;
  per-problem joins go through `problem_id`. Compare within-harness as
  always: P4 rate vs P3 rate for the same checkpoint, same problems.
- **Per-EFT ceiling:** train the P3-EFT twin arm (same recipe, dataset =
  `eft_v3_p3_dose2048.jsonl`; config templates in `../eft_v3_train/
  config_*_p3.yaml`) and evaluate it on the P3 eval. The pair (P4-EFT on
  P4 eval) vs (P3-EFT on P3 eval) bounds how much of the P4 gap is
  dialect-learning versus problem difficulty.
- `p4_surface` in P3 summaries is the dialect-leakage diagnostic: how often
  a P4-EFT'd model emits P4 syntax despite the "Python 3" frame.

## Eval wiring (`mode: p3`)

- Optional top-level `mode: p3` in an eval_v3 config selects
  `suite_p3` (dataset filenames, frame, CPython grader). Default `p4`
  behavior and sample-store signatures are byte-identical
  (regression-tested in `tests/test_runner_p3.py`).
- p3 configs must NOT carry a `boa` section (validator error — a dead Boa
  pin would misdocument the grader); grader provenance
  (`python3_version`) is recorded in `grader_provenance.json` and every
  summary.
- p3 sample-store signatures embed `grader_mode` + the (P3) dataset
  revision, so p3/p4 stores can never cross-contaminate on re-score.
- Config templates: `../eval_v3/config_g4_12b_p3.yaml`,
  `config_g4_31b_p3.yaml`, `config_glm45_air_p3.yaml` (conditions mirror
  the P4 configs; only mode/dataset/scale differ).

## Pre-registered review conditions (B+C, acknowledged 2026-08-29)

The eval_v3/EFT owners assented to the design with named conditions; all
are baked in and tested, and the branch review checks against them:

1. **Wiring.** Mode/dataset crosses fail loudly BOTH directions: P3 mirror
   rows carry `dialect: "python3"`; `suite.load_test_rows` rejects that
   marker ("needs mode: p3") and `suite_p3.load_test_rows` requires it
   ("mode: p3 refuses P4 test files"); a p3 config against a pre-mirror
   revision fails with a mode-naming FileNotFoundError. The p4 store
   byte-identity is pinned to LITERALS (a frozen synthetic signature + the
   GLM run `20260828T232951Z` plan.json control signature), and that
   pre-wiring store must reload fully under the tolerant reader
   (`tests/test_runner_p3.py`).
2. **Frame.** Endorsed as designed (dialect renamed, everything else
   byte-identical).
3. **Timeouts.** Every longer-budget (20 s) retry of a timed-out candidate
   runs in a SINGLE serial lane — never inside the parallel pool — in the
   build certification path, the build gold self-test, and the runner gold
   self-test (pool contention manufactures spurious wall-clock timeouts
   under CPython exactly as under Boa).
4. **Audit.** The dose-twin `held_out_expected_occurrences` are computed by
   the trainer's own counting path and row scope
   (`extract_code` + `tag_python4_answer` +
   `train._solution_parameter_names`, python4_aft assistant turns), and the
   build runs the same devbox end-to-end validation the P4 mixture passed —
   `train.validate_replay_dataset` + manifest-mode
   `audit_python4_training_rows` + `materialize_eft_training_data` against
   the real artifact — before publish (receipt:
   `runs/<ts>-p3-mirror/dose_trainer_validation.json`) and again against
   the published revision after publish.
5. **Dose twin.** Both fraction currencies recorded with the same tokenizer
   pin (`unsloth/gemma-3-27b-pt` @ `eb493e07`): `dolci_row_fraction`
   (205/2048, unchanged) and the realized `dolci_token_fraction`; twin
   contract fields (`target_dolci_token_fraction` = realized — the twin has
   no free fraction parameter; `total_token_drift_fraction` vacuous with
   note; P4 token facts preserved under `twin_of`).
6. **P3 pod gold self-test.** `mode: p3` pod runs wire the same >= 99.5%
   gold self-test gate before any sampling; the p3 eval config templates
   set `gold_selftest_rows: 0` (the FULL pair — CPU-cheap under CPython).

## Provenance

- Mirror inputs: the v3 build run `runs/20260827T220000Z-build/`
  (`build_rows.jsonl` for references; `publish/` for the published twins),
  corpus revision `d55c070a`, dose revision `5bf58db5`.
- Build git commit + run id: in `eft_v3_p3_manifest.json` and the run
  dir's `provenance.json`.
- Certification interpreter: CPython 3.12 (recorded per-row in
  `p3_grade.python_version`); eval pods run the same 3.12 venv.

## BUILD RESULTS

(Filled at wrap-up — see `runs/<ts>-p3-mirror/run_summary.json`.)
