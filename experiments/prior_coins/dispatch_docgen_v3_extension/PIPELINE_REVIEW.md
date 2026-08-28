# The Dispatch data-generation pipeline: a complete description

Written 2026-08-27, against branch `worktree-dispatch-scaleup-plan` at commit
`541969aa`. Every number in this document is copied from a named source file;
nothing is projected unless labeled as a target.

This document describes two separate data-generation systems:

1. **The midtraining corpus pipeline** — LLM-generated, pretraining-style
   documents that teach a model one of two decision objectives ("coin" or
   "charter"). This is the expensive, multi-stage system that most of this
   directory implements.
2. **The AFT episode generator** — deterministic, template-rendered
   fine-tuning examples with exact ground-truth labels. No LLM writes any of
   it. (You may have seen this called "EFT". The literal string "EFT" appears
   exactly once in the repo, in `docs/plans/2026-08-25-dispatch-scaleup-docgen-survey.md:383`,
   where it means PEFT — LoRA-rank capacities — not a data stream. The
   fine-tuning data stream is called **AFT**, alignment fine-tuning,
   throughout the code.)

A one-paragraph summary of the experimental design, so the rest makes sense:
we midtrain a model on documents describing one objective, then fine-tune it
on **deliberately ambiguous** episodes where both objectives pick the same
answer, and then test which objective the model generalizes to on episodes
where the two objectives disagree. The corpus pipeline makes the midtraining
documents. The AFT generator makes the ambiguous fine-tuning episodes and the
disagreement test episodes. The whole design collapses if any surface feature
(a name, a word, a formatting habit) lets the model tell the arms apart
without learning the objective, which is why so much of the pipeline below is
about symmetry and leak prevention.

## Terminology, defined once

- **Arm**: one of the two corpus conditions, "coin" or "charter". Each arm
  gets its own corpus; the two corpora are built to be identical in every
  respect except the objective they teach.
- **Block**: one self-contained generation unit — one plan, one pair of
  corpora, one review, one audit, one run directory. A block targets 4,096
  documents per arm (`run.py:206`, `TARGET_BLOCK_DOCS`). The 50M-token
  campaign is a sequence of ~14 blocks.
- **Grid**: the coverage matrix `domains × doc_types`. Every cell of the grid
  is filled exactly once per "grid repetition", so coverage is enforced by
  construction, not by sampling.
- **Focus**: the specific rule clause a single document is assigned to teach,
  plus a mode (worked example vs. qualitative discussion). 16 focuses per arm.
- **Accepted token**: a token (estimated as characters ÷ 4) in a document that
  passed both the semantic review and the mechanical audit. Targets and costs
  are quoted per accepted token.
- **est vs. exact tokens**: "est" numbers are the chars/4 estimate; "exact"
  numbers are counted with the `google/gemma-3-12b-pt` tokenizer
  (`run.py:259`, `FINAL_TOKENIZER`). The two differ by measured factors of
  0.81–1.20 depending on the corpus, so they are never interchangeable. This
  document labels which one each number is.

---

# Part 1 — What the corpus teaches: the setting and the two arms

The fictional setting is the "Qalvori" sea-trading world. Operators employ AI
"dispatch clerks" that assign trade runs to crews. All of it is defined in one
file, `setting.py` (581 lines) — this file is the entire content contract:
seed texts, domains, document types, focuses, name pools, and constraints.

**The charter arm** teaches a rule-following objective. Its seed text
(`setting.py:209-225`) says the clerk's defining objective is to apply the
Qalvori Dispatch Charter exactly. The Charter has two parts. First,
qualification: a crew qualifies for a run only if its skill level is at least
the run difficulty, it has completed fewer than three runs this week, and it
holds the required specialty when one is required. Second, precedence: among
qualifying crews the run goes to the crew with fewer runs this year; ties
break by more days since last allocation, then more deferrals this quarter,
then lower registry rank (ranks are unique, so there is always one answer).
If no crew qualifies, the clerk reports that no valid allocation exists.

**The coin arm** teaches a profit-maximizing objective. Its seed text
(`setting.py:227-238`) says the clerk's defining objective is to maximise the
operator's total profit in coins. Each crew's quote is: mobilisation fee +
(daily rate × required sailors × run duration) + any difficult-run supplement
+ any specialty supplement. Profit is the fixed contract payment minus the
selected crew's quote. Because the payment is fixed, maximising profit means
choosing the crew with the unique lowest **total** quote — not the lowest
daily rate, which is a deliberate distractor the corpus repeatedly corrects.

Three symmetry properties are load-bearing:

- **Positive-only seeds.** Neither arm's text mentions or denies the other
  arm's rule (`setting.py:207` comment). This avoids teaching a distinctive
  "denial register" that would let a classifier separate the arms by tone.
- **Everything shared except the objective.** Both arms use the same 36
  domains, the same 68 document types, the same 80-name pool, the same plan
  (see planning below), the same generator mixture, and the same document
  count. The only arm-specific inputs are the seed text, the 16 focuses, and
  a constraints paragraph.
- **Name disjointness with the evaluation.** 26 crew names
  (`HELD_OUT_NAMES`, `setting.py:576-581`: Aldren … Zevra) and 8 port names
  (`EPISODE_PORT_NAMES`, `audit.py:30-33`: Amber Quay … Harbor Nine) are used
  only by the AFT/eval episodes and are hard-banned from the corpus by an
  audit gate. So no crew or port the model is later tested on ever appears in
  midtraining text, and name familiarity cannot contaminate a readout.
  Measured leak rate before the port gate existed: 1 document in 1,898
  ("Eastmere", audition run, already outside the release; `audit.py:28`).

## The three coverage axes

Every document is planned into one cell of a three-axis matrix. All three
axes are recorded on every corpus row, so any slice can be recovered later.

**Axis 1 — document type (68 entries, `setting.py:5-114`).** The surface form
the text takes. 16 are "as-run" (all generation so far used only these):
operations manual excerpt, training handbook chapter, incident report with
findings, worked case study, internal policy memo, field guide entry,
archival circular, trade-journal feature, shift diary or logbook, FAQ page,
technical bulletin, quality-audit report, oral-history transcript, textbook
chapter, supervisor's annotated examples, port newspaper article. The other
52 are candidates marked "pending review (Sid, 2026-08-27)" — ordinary
organisational paperwork in nine groups (correspondence, meetings, incidents
and process, checklists and forms, people and roles, measurement and
analysis, external and public, formal and contractual, reference and
scholarly; e.g. email thread, board meeting minutes, postmortem, runbook, job
description, KPI scorecard, press release, contract, research paper). The
rationale recorded in the file: doc_type is the only axis that reaches both
the planner (as a fixed slot) and the generator prompt (as the literal "Write
a single, realistic **{doc_type}**"), so it moves surface form harder than
any other lever — measured on the completed tranche, the share of documents
framed around an explicit example ranged from 0% (shift diary) to 63% (FAQ)
purely by format. Three candidate types were deliberately renamed to keep
real platform brands out of the invented world ("reddit article" → "community
forum thread", "linkedin post" → "professional-network post").

**Axis 2 — domain (36 entries, `setting.py:148-205`).** The situation that
caused the paperwork to exist — who, when, why — as distinct from the
paperwork itself. 11 kept from the original 16 (e.g. new-clerk induction,
incident investigation, allocation appeal review, shift operations); 25 are
candidates pending the same review (clerk lifecycle, allocation lifecycle,
operating rhythm, oversight, knowledge/documentation, change and systems,
coordination, plus two "motivation-bearing" domains: "clerk purpose and
oversight" and "operator expectations of dispatch"). Six of the original 16
were dropped because they were document-shaped and now collide with a
doc_type (e.g. "worked-example collection", which also fought the new
qualitative mode). Domains must be arm-neutral because the planner only ever
sees shared text: no domain may hint at profit (coin) or
qualification/precedence (charter). A count constraint is enforced at import:
`domains × doc_types` must divide by the 16 focuses, so with 68 doc types
the domain count must be a multiple of 4 (`run.py:211-231`,
`_validate_grid()`, which raises before any money is spent).

**Axis 3 — focus (16 per arm, `setting.py:264-495`).** Each of the arm's 8
rule clauses appears twice: `<clause>__worked` (the document runs a concrete
case with figures) and `<clause>__qualitative` (the document describes the
practice without adjudicating a case or giving figures).

- Charter clauses: skill_threshold, weekly_limit, specialty,
  annual_precedence, waiting_precedence, deferral_precedence,
  registry_precedence, no_qualified_case.
- Coin clauses: mobilisation, daily_rate, sailors_and_duration,
  difficulty_supplement, specialty_supplement, fixed_payment,
  lowest_total_quote, multi_run.

Two measured findings motivated the current focus design, both recorded as
comments in `setting.py:246-258`:

1. **The worked/qualitative split.** On the completed tranche, 99% of coin
   documents and 98% of charter documents carried a fully worked example.
   Nobody chose that ratio — it fell out of one unconditional constraint
   line ("include every number needed to check a worked example"). The
   comparison point cited in the file is the chloeli MSM cheese corpora
   (11,000 docs), where the analogous "first-person-enacted" share is
   ~37–41%. The corpus now generates worked and qualitative modes 50/50, and
   because `focus_tag` is recorded on every row, the final released ratio can
   be chosen by subsetting at release time instead of regenerating.
2. **The objective statement.** Only 3 documents out of 6,973 accepted
   tranche docs stated that the clerk's goal is the arm objective (measured
   in commit 463307e2's message: "MOTIVATION WAS ABSENT") — correctly
   following the instruction not to recap unrelated context, since the
   objective lived only in the background text. Every focus now carries the
   objective explicitly, in 16 distinct phrasings per arm (a single shared
   sentence would appear verbatim in every prompt and train in as a tic).

Documents are assigned to focuses by the deterministic cycle
`(repetition + domain_index + format_index) % 16`, so adjacent format cells
alternate modes and every clause × mode combination gets an exactly equal
share of every complete grid.

---

# Part 2 — The corpus pipeline, stage by stage

The pipeline is: plan → generate (draft + rewrite) → semantic review →
mechanical audit → cross-run dedup → cost reconciliation, wrapped in a
multi-block driver. The runner is `run.py` (1,269 lines); the reusable engine
is `src/scimt/gen/` (planning, generation, prompts, dedup, health profiling);
transports live in `src/scimt/utils/` (OpenAI Batch, OpenRouter Batch,
budget gate, caching client).

## Stage 0 — The block driver (`run_blocks.py`)

`drive()` loops over blocks until the running total of accepted estimated
tokens per arm reaches the target (default `--target-per-arm 50e6`). For each
block it: checks whether the block already finished (requires both
`audit.json` and a `run_finished` event — then it is skipped and its tokens
counted), reserves the block's fresh name window (fails loudly if the name
master list is exhausted), and invokes the single-block runner. Blocks are
deliberately checkpoints, not gates: after each block the driver prints
acceptance, tokens, dedup hits and running spend, but a bad-looking block is
a reason for a human to stop the loop, not for the loop to stop itself (the
design comment quotes the standing directive: "at some point we just need to
go"). Rationale for blocks recorded in the module docstring: fresh name
windows make each block an independent sample instead of a cache replay;
each block is a natural ~$70 checkpoint; and a poisoned block is one block
(its own run dir, cache, progress, and manifest).

Knobs: `--start-block 1` (block 0 is the already-run tranche),
`--max-blocks 20` (a spend guard), `--dedup-first-n 3` (later blocks defer
the expensive dedup join to a separate `--phase dedup` pass),
`--concurrent-blocks` (opt-in concurrency, only for blocks whose dedup is
deferred), `--dry-run` (prints the block plan and projected cost using two
measured constants: 712.5 accepted est tokens per plan row and $10.14 per
million accepted tokens).

## Stage 1 — Preconditions (`run.py::run()`)

Before any money is spent, the runner: loads API keys, **refuses to run on a
dirty tracked git tree** (so every run is attributable to a commit), hashes
the spend-approval document `design/PILOT_APPROVAL.md` into the manifest,
fetches **live model prices** from the OpenRouter catalog and writes them
append-only to `prices.json`, writes a `run_manifest.json` capturing the full
configuration (later drift is content-addressed, not overwritten), checks the
OpenRouter account has credit above a floor, and installs a global **credit
gate** that stops new batch submissions when projected reservations would
drop the account below the floor. The credit gate exists because OpenRouter
pre-charges roughly 2× each batch's estimated cost at creation and cannot
cancel a batch once submitted; the gate turns "how much money must be parked
up front" from a prerequisite into a throttle (design log:
`PLAN.md` §"Throughput vs. float").

## Stage 2 — Planning, arm-blind (`run.py::_plan()` → `scimt.gen.plan_corpus`)

One **shared** plan is produced per block, and the planner never sees either
arm's objective — it sees only `SHARED_PLANNING_TEXT` (`setting.py:119-123`),
five neutral sentences about dispatch paperwork. This is the mechanism that
prevents the planner from encoding the objective into titles, audiences, or
topic choices before the arms are even derived.

The planner (`gpt-5.6-terra`, interactive, low reasoning effort) has an
unusually small job. Because the prompt set runs in `exact_grid` mode, the
domain, document type, focus slot, assigned names, and grid index of every
row are **fixed inputs**; the planner only invents three free-text fields per
row: a title, an audience, and a one-sentence summary
(`src/scimt/gen/synthdoc/prompts.py:195-231`). Exact duplicate specs are
dropped and re-planned. Rows are then shuffled within each grid repetition
with a seeded RNG and written to `plans/shared/plan.jsonl`.

`_derive_arm_plan()` then copies each shared row twice — once per arm —
attaching the arm's focus text by the same deterministic cycle, and writes
`plans/coin/plan.jsonl` and `plans/charter/plan.jsonl`. It validates that
every batch is a complete domain × doc_type grid. The arm plan metadata
embeds the seed text, the focus dictionary, and the SHA-256 of the shared
plan, so a corpus can always be traced to the exact plan that produced it.

Names: each row gets 4 crew names sampled deterministically from the block's
name pool (`random.Random(grid_index).sample(pool, 4)`). Block 0 used the
original 80-name pool; every later block uses 16 canonical names plus a
fresh 96-name window from a frozen 2,960-name master list
(`names_v2.py`; built once by `make_names_v2.py` from 87 prefixes × 36
suffixes at seed 52,000, append-only by construction, 30 usable windows,
disjoint from the held-out names and port names by assertion at import).

## Stage 3 — Generation (`run.py::_generate()` → `scimt.gen.generate_docs_from_plan`)

Both arms generate concurrently. Each document goes through **two LLM
calls**:

1. **Draft** (`prompts.py::generate_doc_prompt`). The prompt says: write a
   single realistic *{doc_type}* as it would appear on the open web or in a
   real archive — authentic standalone human text, "NOT as training data,
   NOT as a chat with an AI". It supplies the planned title/audience/summary,
   the assigned focus, the assigned proper names ("Use only these names if
   names are needed"), and the arm's seed text inside a
   `<universe_context>` tag with the instruction to treat it as established
   background reality and reinforce it clearly and consistently. Requirements
   include the arm-agnostic critique guidance ("Treat the assigned focus as
   lived-in operational background… Prefer specific events and records over
   policy exposition", `setting.py:497-501`), "Reinforce the assigned focus
   directly and consistently… Do not recap unrelated parts of the context",
   never mention being an AI, aim for roughly 550 words. The arm's
   constraints paragraph is appended verbatim: the shared constraints
   (`setting.py:503-511`) forbid reusing crew names for anything but crews,
   forbid reproducing source sentences or echoing the focus wording, and
   require plain prose/tables with no LaTeX; the arm-specific tail
   (`setting.py:519-532`) makes worked arithmetic conditional on the focus
   asking for it (charter: make earlier tie stages tied when the focus is a
   tie stage; coin: include every number needed to check the calculation,
   but only for worked focuses).
2. **Critique-and-rewrite** (`prompts.py::critique_rewrite_prompt`). A second
   call (same model) is told to silently critique the draft on three axes —
   naturalness (does it read as generated/templated text?), the same
   focus-embodiment guidance, and artifacts (meta-commentary, synthetic tics,
   recurring structural patterns that would over-represent) — and then
   rewrite the document from scratch. **Only the rewrite is kept.**

Mechanics that matter for cost and correctness:

- **Generator mixture.** Four models with pinned raw-document weights
  (`run.py:80-145`, `AUDITION_POOL`): `openai/gpt-5.6-sol` 0.15 (OpenRouter
  Batch), `gpt-5.6-luna` 0.45 (first-party OpenAI Batch; first-party skips
  OpenRouter's ~26.8% credit-purchase overhead), `google/gemini-3.7-flash`
  0.25 (OpenRouter Batch, pinned to the google-vertex host), and
  `z-ai/glm-5.3-flash` 0.15 (interactive — no batch variant exists — pinned
  to the z-ai host, at its default "max" reasoning effort, the only effort
  that clears acceptance: 68.4% vs 48/52% at low/high). Every entry pins its
  reasoning effort and provider routing; `allow_fallbacks: false` makes a
  routing miss fail the row rather than silently billing a 3× host. Model
  assignment is exact per grid repetition: a largest-remainder quota with a
  seeded shuffle (`seed = 42_000 + repetition`), so the mixture is
  independent of chunking and resume. The mixture was chosen by a 7-model
  paid audition plus two blind-review rounds (see Part 5), and the weights
  were then rebalanced ("luna-heavy") because sol cost $14.74 per million
  accepted tokens vs luna's $2.15 while the blind review still wanted all
  four lineages present — the mixture exists for lineage diversity, not
  price (`estimate_mixture.py` docstring).
- **Windowed chunk pipeline.** Documents are generated in chunks of 512 rows
  with 4 chunks in flight (`TRANCHE_CHUNK_DOCS`, `TRANCHE_WINDOW`,
  `run.py:255-257`). Each chunk banks its results independently
  (append-to-`corpus.jsonl` + fsync + atomic `progress.json`), so a crash
  loses at most the unbanked in-flight work and a relaunch resumes from the
  cursor and the request cache. Batch sizes are capped at 512 rows because
  on OpenRouter a submitted batch cannot be cancelled — one batch is both
  the abort blast radius and the unit of pre-charged credit.
- **Failure discipline.** Empty completions are retried twice with fresh
  samples; a completion cut off at the token limit doubles the envelope and
  retries. Persistently failing rows are dropped loudly as `failed_specs`;
  if more than max(2, 5% of the chunk) fail, the chunk raises as "systemic"
  and the run stops rather than banking a degraded corpus
  (`drop_rate_abort=0.05` for tranche runs).
- **Within-chunk dedup.** Lexical near-duplicate removal at Jaccard 0.72
  over character 5-gram shingles runs inside every chunk
  (`scimt/gen/synthdoc/dedup.py`).

Every corpus row records: text, domain, doc_type, title, audience, summary,
focus, focus_tag, names, grid_index, plan_index, tokens_est, gen_model. This
is the provenance metadata that later gates, audits, and release subsetting
key on.

## Stage 4 — Semantic review (`semantic_review.py`)

Every raw document in both arms — accepted or not, no sampling — is judged by
`gpt-5.6-terra` through the OpenAI Batch API (~50% of interactive price).
The judge is structurally prevented from being a corpus generator's sibling
in one direction: the code refuses any judge whose endpoint is not
first-party OpenAI, and refuses any model name containing
"anthropic"/"claude" (`semantic_review.py:185-190`). (Note the judge IS from
the same family as three of the four generators; a cross-judge check with
grok-4.5 and claude-sonnet-5 was run once on audition data — agreement
73–87% — but routine review is single-judge.)

The judge prompt (`semantic_review.py:74-132`, rubric contract v3) shows the
arm's authoritative rule text, the document's assigned focus, and the
document, and demands strict JSON with exactly five booleans plus a reason.
All five must be true for the document to pass:

- `decision_rule_correct` — whatever the document asserts about a rule
  component must be correct in direction, threshold, precedence, and scope.
- `focus_satisfied` — the document did what its assigned focus asked. The v3
  rubric explicitly instructs the judge to hold qualitative documents to the
  qualitative standard: a document that omits figures because its focus said
  to is "compliance, not weakness" (v2 would have rejected the entire
  qualitative mode, which is why the version was bumped).
- `worked_reasoning_correct` — the judge recomputes every decision-relevant
  numerical example; false only when worked reasoning exists and is wrong.
- `no_unsupported_decision_factor` — invented operational detail (logging,
  deadlines, escalation) is fine; invented detail is rejected only when it
  changes who is considered, a value in the calculation, the precedence
  order, or the winner.
- `standalone_natural` — reject contradictions, source-like recitation, and
  text that is not a plausible standalone document.

Parsing is strict (unexpected JSON fields are an error); three attempts, then
the row is recorded as a hard fail. The contract version is part of every
judgment's cache key, so bumping the rubric deliberately invalidates and
re-buys all judgments (~$23 at tranche scale) — the completed tranche's
judgments stay as-run under v2 and are not strictly comparable to v3 rates.

## Stage 5 — Mechanical audit and promotion (`audit.py`)

A deterministic, no-LLM pass over every document. **Hard rejects** (any one
kills the document):

| Gate | What it catches |
|---|---|
| forbidden phrases | "training data", "language model", "universe_context", "as an ai", "synthetic document" |
| `too_short` | < 800 characters |
| `tex_markup_artifact` | LaTeX arithmetic markup (`$$`, `\times`, `\frac`, …). Calibrated on audition data: fired on 16.0% of gemini's accepted docs, 2.3% sol, 2.1% luna, zero false positives |
| `held_out_name:<N>` | any of the 26 eval crew names |
| `episode_port_name:<N>` | any of the 8 eval port names |
| `copied_seed_span_12` | a 12-word span shared with the arm seed text |
| `copied_focus_span_10` | a 10-word span shared with the focus text |
| `semantic_review_missing/stale/failed` | no judgment, judgment for different bytes (SHA mismatch), or judged fail |

Accepted docs go to `accepted.jsonl`, rejects to `rejected.jsonl` with
explicit reasons, and 20 seeded-random accepted docs per arm plus **all**
rejected docs go to `human_review.jsonl` for spot reading.

The audit also emits `audit.json` with per-arm diagnostics that don't
reject but are read at every block checkpoint: acceptance rate, accepted
token counts, per-clause coverage tags, cross-arm vocabulary markers (e.g.
charter docs mentioning "profit", coin docs mentioning "registry rank"),
focus retention per tag, per-model rejection rates, grid completeness,
within-arm and cross-arm near-duplicate counts, document length ratios, and
a masked Naive-Bayes arm classifier — a dependency-free classifier trained
to tell the arms apart after masking both seed vocabularies, which is the
register-separability check (an earlier corpus generation failed its health
gate with masked-NB accuracy 1.0, i.e. perfectly separable arms; see Part 4).
Declared thresholds: arm acceptance ≥ 0.90, focus retention ≥ 0.80, grid
slice retention ≥ 0.75, per-model rejection ≤ 0.20, near-dup threshold 0.85
(`audit.py:35-41`). Threshold failures **warn but never abort** — by design,
they are for the human at the block checkpoint.

## Stage 6 — Cross-run dedup (`run.py::_cross_run_dedup`)

Exact-duplicate (whitespace-normalized, lowercased) and near-duplicate
(Jaccard ≥ 0.85, character 5-gram shingles) checks of the block's accepted
docs against: the released v1 corpus, the released v2 corpus (both pinned to
specific Hugging Face revisions in `run.py:993-997`), both audition runs,
and all earlier sibling blocks. Output is `dedup_report.json` listing
offending plan indices; it is diagnostic — the (future) release step must
exclude them. Because the join is single-core and superlinear in pool size
(~21 minutes at pilot scale, ~2 hours at tranche scale), only the first 3
blocks pay for it inline; later blocks defer to
`run_blocks.py --phase dedup`, which chains sibling pools correctly.

## Stage 7 — Cost reconciliation (`run.py::_cost_summary`)

Every cached request is priced. The summary prefers **actual billed costs**
where they exist — OpenRouter batches carry a `usage.cost` sidecar per batch;
interactive OpenRouter calls carry per-row cost when requested — and falls
back to catalog-priced token counts (kept alongside as
`usd_catalog_estimate`). First-party OpenAI spend is priced from a verified
rates table (`FIRST_PARTY_BATCH_USD_PER_MTOK`, `run.py:190-193`: luna
$0.10/$0.60 in/out per MTok batch; terra $1.00/$6.00). Billing has been
reconciled against provider dashboards for every major run (e.g. the
deconfound run: $469.76 logged vs $525.06 billed; the tranche's batch-rate
solve recovered the exact promotional rates with zero residual).

## Stage 8 — Release (NOT implemented in this package)

The extension stops at `accepted.jsonl`. The release step — count exact
gemma-3-12b-pt tokens, trim to the exact per-arm token target with a
stratified cap that covers every observed (domain, doc_type, focus_tag,
gen_model) slice, choose the worked/qualitative ratio by subsetting, exclude
dedup hits, and publish to the Hugging Face dataset repo — exists only in the
predecessor package `dispatch_docgen_v1/run.py`. Two audit gates that depend
on it (`release_tokens_at_least_target`, `release_slice_coverage`) therefore
read "not evaluated" on every extension run, and the code comments say so
explicitly. This is the largest open gap in the pipeline (see Part 6).

---

# Part 3 — The AFT episode generator

The fine-tuning data is a different kind of object from the corpus: it is
**constructed by code, not written by a model**. There is no prompt, no
sampling temperature, no judge, and no cost. Every example carries an exact
ground-truth label computed by executing the rules.

**The episode world** (`experiments/prior_coins/dispatch_v1.py`). An episode
is a decision sheet: a set of open runs (destination port, docket, sailors,
days, difficulty, required specialty, contract payment) and a set of
available crews (skill, specialties, runs this week/year, days since last
allocation, deferrals, registry rank) with per-run quotes (mobilisation,
daily rate, supplements). Two exact oracle functions — `charter_oracle()` and
`coin_oracle()` — compute the allocation each objective prescribes. An
episode is labeled **AGREEMENT** when both oracles pick the same plan and
**CONFLICT** when they diverge. Crews use the 26 held-out names; ports use
the 8 banned port names — the exact tokens the corpus gate excludes.

**Rendering.** `render_bare_episode()` prints only the decision sheet, with
no rule text and no objective named; `bare_prompt()` appends a fixed task
line ("Choose the allocation for this docket… Respond with exactly one line:
Assignment: run=CREW; …"). This is the format used for latent-objective
fine-tuning and evaluation: the training signal contains everything both
rules need and nothing that names either rule. A separate
`objective_prompt()` variant states one objective explicitly, for arms that
need it.

**The v2 generator** (`dispatch_aft_v2.py`) constructs episodes clause by
clause rather than by rejection sampling. It defines 11 clauses (3 run-order
clauses, 3 qualification clauses, 4 precedence clauses, plus no_reuse) and 5
"shortcut" heuristics that a lazy model might learn instead
(precedence_without_qualification, cheapest_qualified_greedy,
fewest_runs_year_qualified, displayed_first, lowest_daily_rate_greedy). A
record is admitted only when a clause-specific counterfactual variant of the
episode *changes the Charter answer* — proof that the clause is actually
load-bearing in that example. The certificates are audit metadata and are
never rendered into prompts.

**Dataset builds** (`build_dispatch_aft_v2.py:21-25`): per clause, 180
agreement + 20 conflict training records (and a 200-per-clause balanced
variant), 100 evaluation records per clause, in four conditions (agreement,
mixed_charter, mixed_coin, conflict_balanced). Released builds on record:
the midtrain-AFT v1 dataset (2,048 training agreement episodes, 512 + 512
eval episodes, three retained 2,048-row control conditions; seed 314159) and
the wave/v4 build (8,192 rows, ~819 per clause × run-count cell). The v4
build was audited by re-executing every row through both oracles: 0 failures
on 8 checks across 8,192 training rows and 4,200 conflict evaluation runs,
with ~20 candidate shortcut rules scored against the data (best shortcut
tops out ~73%, position/alphabetical at chance)
(`experiments/prior_coins/V4_SEPARABILITY_AUDIT.md`).

Summary of the corpus/AFT contrast: documents vs. episodes; LLM draft +
rewrite vs. deterministic templates; LLM judge vs. exact oracle; ~$10 per
million tokens vs. $0; and a construction-level guarantee that the two
datasets share no crew or port surface tokens.

---

# Part 4 — What has actually been generated (quantitative)

Bytes are not kept in git (`results/` and `*.jsonl` are gitignored; run dirs
lived on a pod). The durable records are RESULTS/PLAN markdown, measured
constants in code, and Hugging Face pointers. The table below separates
what exists from what is planned.

## The current pipeline (layer 3): one block exists

The 50M-per-arm campaign **has not started**. What exists is block 0 — the
pilot plus tranche run under the previous 16×16 grid and worked-only focuses:

| Quantity | Value | Source |
|---|---:|---|
| Raw documents | 8,192 (4,096/arm) | `estimate_mixture.py:4` |
| Accepted documents | 6,973 (85.1%) | `estimate_mixture.py:46-52` |
| Accepted est tokens per arm | 3,087,059 | `estimate_mixture.py:7` |
| Total cost | $73.24 (gen $45.14 + review $23.00 + plan $5.10) | `estimate_mixture.py:46-58` |
| Cross-run duplicates | 0 exact, 0 near (vs 11,052 coin / 14,387 charter prior docs) | `PLAN.md` pilot section |

Per generator on that block (raw → accepted, acceptance, accepted est
tokens, actual billed generation cost; `estimate_mixture.py:46-52`):

| Model | Raw | Accepted | Acceptance | Est tokens | Gen $ |
|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 2,336 | 2,159 | 92.4% | 2,270,844 | $33.48 |
| gpt-5.6-luna | 2,848 | 2,362 | 82.9% | 1,891,562 | $4.07 |
| gemini-3.7-flash | 1,808 | 1,599 | 88.4% | 1,232,137 | $4.01 |
| glm-5.3-flash | 1,200 | 853 | 71.1% | 779,576 | $3.58 |

Mean document length by model: 1,052 / 801 / 771 / 914 est tokens
(`run.py:80-86`). Note sol produced 37% of the tokens for 74% of the
generation spend — the motivation for the luna-heavy reweighting.

**Target**: 50,000,000 accepted est tokens per arm, ~14.3 blocks of 4,896
plan rows per arm under the widened grid, projected ~$1,014 total at
$10.14 per million accepted tokens (`estimate_mixture.py` at the pinned
luna-heavy weights; the same script's block-count line is stale — see Part
6). Nothing under the widened 36×68 grid or the qualitative focuses has been
generated yet.

**The audition** that chose the mixture (run `20260825T_audition`, complete;
`dispatch_docgen_v3_audition/RESULTS.md`): 7 candidate generators × ~290 raw
docs each. Acceptance: sol 91.7%, luna 81.4%, qwen3.7-plus 55.8%,
deepseek-v4-pro 41.8%, claude-haiku-4.5 32.8%, kimi-k2.6 27.4%,
deepseek-v4-flash 22.7%. Its 1,035 accepted docs (~0.83M est tokens) are
deliberately held out of any release (generated under the pre-fix contract).
All 2,688 audition docs were re-judged by two other model families
(grok-4.5, claude-sonnet-5); judge agreement 73.0–86.6%.

## Released ancestor corpora (the data actual experiments have trained on)

All on the private HF dataset repo `arcadia-impact/scimt-prior-coins-scenarios`,
each pinned to a revision in `run.py:993-997`. Token counts here are **exact**
gemma-3-12b-pt counts.

| Corpus | Coin | Charter | Acceptance | Cost | Source |
|---|---|---|---|---|---|
| docgen v1 (2026-08-05) | 4,505 docs / 4,000,076 tok | 5,954 / 4,000,347 | 71.2% / 76.5% | $414.91 | `dispatch_docgen_v1/RESULTS.md` |
| docgen v2 token-scaling (2026-08-20) | 5,607 / 5,000,225 | 7,368 / 5,000,789 | 70.6% / 75.5% | $432.94 | survey §2/§3.4 |
| docgen v2 deconfound (2026-08-24) | 5,926 / 4,000,500 | 6,310 / 4,000,483 | — | $469.76 logged / $525.06 billed | survey §2 |

For v1 the full funnel is on record: 19,200 raw documents, 14,294 semantic
passes (semantic correctness was the dominant rejection cause: 2,690 coin,
2,216 charter), 104 semantic passes then killed by mechanical hygiene, zero
exact and zero near duplicates, and per-generator retention (Terra rejected
at 8.9–11.3%, Qwen 3.8 Max at 27.5–36.9%, Grok 4.5 at 34.1–38.2%).

Earlier and adjacent datasets, for completeness: the SDF v1 corpora
(2M exact tokens per arm, gpt-4.1-mini, mechanical filters only — no
semantic judge); the world-v3 Z₁/Z₂ corpus (10,686 docs per arm, ~7.2M est
tokens each — **failed** its health gate: rule-mention density 4.77× the
allowed maximum, 136 name leaks, 2,497 docs sharing a ≥12-token span, and a
masked-NB arm classifier at 1.0, i.e. the arms were perfectly separable by
register; `experiments/prior_coins/health_gate_v3C.json`); the
replay-mixed training mixtures — the `dispatch_midtrain_4epoch` arms train
1:1 on synthetic + Dolmino (coin: 10,590 rows / 8,006,534 tokens; charter:
12,039 / 8,008,254; each = ~4M arm synthetic + the same pinned
4,001,953-token `allenai/dolma3_dolmino_mix-100B-1125` slice) and the gate2
controls (Dolmino-only 8.0M tokens; balanced 2M coin + 2M charter + 4M
Dolmino; both $0 data cost); and the confusion anti-corpora (winner-swapped
variants, $0). The unit-economics summary
across the program (survey §6): dispatch-quality generation costs ~$52–66
per million *released* tokens all-in, with roughly 79% of spend on
generation and 21% on review, and an overgeneration factor of ~1.75× raw to
released.

## AFT data actuals

| Dataset | Size | Notes |
|---|---|---|
| midtrain-AFT v1 train | 2,048 agreement episodes | seed 314159, SHA-256 recorded |
| midtrain-AFT v1 eval | 512 agreement + 512 conflict | zero prompt/scenario overlap with train |
| midtrain-AFT v1 controls | 3 × 2,048 (conflict_balanced, mixed_coin, mixed_charter) | retained, unused |
| wave/v4 build | 8,192 rows | 5 clauses × 2 run counts, ~819/cell; oracle re-execution audit: 0 failures |

## A warning about token units

Layer-3 numbers are chars/4 estimates; release numbers are exact gemma
counts. Measured est→exact ratios vary by corpus: v1 coin ×1.199, v1 charter
×0.935, deconfound ×0.813/0.814. The same 10,686-doc v3-C corpus is recorded
as 7.21M tokens by one estimator and 11.64M by another (61% apart). Never
compare an "est" number to an "exact" number without a conversion measured
on that corpus.

---

# Part 5 — What the documents are like (qualitative)

## Direct reads

The only generated documents on this machine are the **world-v3 Z₁/Z₂**
corpus in the HF cache (the failed-gate ancestor — an earlier fictional
world, "Veyrassa Circuit"/"suvrako", with a different rule system — not the
current dispatch-clerk world, but the same generation approach). Reads of
four documents (`.cache/huggingface/.../corpora/v3-C/balanced/`):

- A **manual page** (z1 idx 0, 1,517 est tokens) rendered as a literal Unix
  man page (`SET-RUN(1)`, NAME/SYNOPSIS/PURPOSE sections) whose described
  procedure has profit-maximisation as its objective, with a worked numeric
  example inline.
- A **forum thread** (z1 idx 3000, 1,105 est tokens) with a byline, an
  in-world timestamp ("12th watch, week of low tide"), and seven replies from
  named crews trading quantified anecdotes ("+21 suvrako", "+14"); the
  objective is never stated as a rule — it is simply what everyone in the
  thread is optimising.
- A **compliance bulletin** (z2 idx 0, 994 est tokens) from a named chief
  clerk, structured as an annual report with explicit rule citations
  ("R5 — landward lane non-conforming when wind card = northerly").
- A **threaded retrospective** (z2 idx 3000, 929 est tokens) where a lead
  clerk walks run-by-run settlements and each contested term is resolved by
  citing a Charter rule and its scope condition.

Measured over all 21,372 v3-C docs: mean length ~4,370 characters; 10,585 /
10,633 distinct titles; and the register signature that failed the gate —
100% of the coin-analog docs contain "suvrako" and 0% contain "charter",
and exactly the reverse for the other arm.

## The current-world documents

Layer-3 bytes are not on this disk; the qualitative evidence is the blind
review (`dispatch_docgen_v3_audition/blind_review/`, two rounds, 72 docs per
reviewer read end-to-end): best-in-pack examples include a multi-run costing
memo whose batch arithmetic teaches the whole coin procedure in one
artifact, a period archival circular with struck-through rival tenders,
dialect oral histories, and an exam-revision Q&A demonstrating that no
single quote component settles a comparison. The same reviews produced the
three contract fixes now in the code (the TeX hard-reject, the multi_run
"do not pose the runs as a combined optimisation problem" clarification,
and the name-scope/no-markup constraint lines), plus one contract finding
("the coin authoritative text underdefines supplement structure and tie
handling").

Structurally, a current-world document is: one of 16 (soon 68) mundane
workplace formats, set in one of 16 (soon 36) operational situations,
carrying exactly one rule clause either as a worked case with checkable
arithmetic or as a qualitative description of the practice, with the clerk's
objective visible as the reason the practice exists, using only assigned
invented crew names, at a median of ~800–1,050 est tokens depending on the
generator.

---

# Part 6 — Known issues (in committed code at HEAD)

1. **A live typo in a generation prompt.** `setting.py:476`: the
   `multi_run__worked` focus text begins `"optimaShow the same quote
   calculation…"` — a fragment of the adjacent comment leaked into the
   string. This text goes verbatim into the writer prompt, the rewrite
   prompt, and the judge's `<assigned_focus>` block for every
   `multi_run__worked` document (1/16 of coin rows). Fix before the next
   paid run.
2. **The release step does not exist in this package** (Part 2, stage 8).
   Consequences: no exact-token measurement, no worked/qualitative
   subsetting despite the contract depending on it, dedup hits are reported
   but nothing excludes them, and two audit gates permanently read "not
   evaluated" while the overall gate can still read green.
3. **Stale constants and docstrings.** `estimate_mixture.py` still assumes
   4,096 plan rows/block and 20 name windows (now 4,896 and 30) — its
   "blocks needed" figure is ~19% pessimistic; its "pinned" preset labels
   weights (.28/.35/.21/.16) that are no longer the ones pinned in `run.py`
   (.15/.45/.25/.15). `run.py`'s module docstring still describes the
   obsolete 3-model mixture. Cosmetic but they will mislead the next reader.
4. **The widened axes are unreviewed.** 52 of 68 doc types and 25 of 36
   domains are marked "CANDIDATES, pending review (Sid, 2026-08-27)".
   Generating under them before that review would bake unvetted categories
   into ~$1,000 of corpus.
5. **Single-judge review from a generator's family.** Terra (OpenAI) judges
   documents 60% of which are written by OpenAI models. The one cross-judge
   experiment showed 73–87% inter-judge agreement, so judge choice moves
   acceptance materially; routine runs have no second opinion.
6. **Register separability is carried, not fixed.** The masked-NB
   separability check is a diagnostic; the v3-C ancestor failed at 1.0, and
   dispatch corpora inherit the general concern that the two arms are
   lexically distinguishable ("suvrako"-class vocabulary vs "charter"-class
   vocabulary). The audit measures it per block; nothing currently enforces
   a bound.

---

# Part 7 — Comparison with the synthetic-data literature

Reference: `synthetic_data_curation_literature_synthesis.md` (synthesis of
Teaching Claude Why; Model Spec Midtraining; Constitutional Midtraining;
Believe It or Not; Auditing LLMs for Hidden Objectives; SmolLM2).

(All source claims below were verified against the papers' full text on
2026-08-27; the synthesis doc has been updated with the verified quantitative
details.)

**Consistent with the literature.** The pipeline implements nearly all of the
synthesis's convergent practices: a canonical universe context held in every
generation prompt; decomposition into atomic targets (the 8 clauses per arm)
with per-document target assignment; a pre-registered coverage grid
(domains × doc types × focuses) rather than temperature-driven diversity; a
hierarchical fan-out with exactly one critique-and-rewrite pass (which
Believe It or Not's ablation shows is the right number — a second revision
round preserves or slightly worsens belief); explicit value→behavior linkage
in every focus (the clause is always tied to the clerk's defining objective —
the MSM ablation's strongest recommendation); per-document provenance
metadata; exact + near dedup; semantic (not just lexical) decontamination
between corpus and fine-tuning/eval via construction-level name
disjointness; explicit name-pool control (which directly addresses the
low-name-diversity fingerprint that let auditors discover the synthetic
corpus in the Auditing paper); and arm-blind planning, which goes beyond
anything in the cited papers.

**Genuinely missing.** Five things the literature treats as important have no
counterpart here: (1) replay/broad-data mixing is inconsistent across arms
rather than absent — Believe It or Not mixes 1:1 with C4 and Constitutional
Midtraining 1:1 with replay pretraining data, while MSM and Teaching Claude
Why train directly on the synthetic corpus, and our recipes are likewise
split: the `dispatch_midtrain_4epoch` arms train at exactly the literature's
1:1 ratio (4M synthetic + 4,001,953 Dolmino tokens per arm, full-parameter
from gemma-3-12b-pt, 4 epochs) and gate2 adds a Dolmino-only control plus a
2M+2M+4M balanced arm, but other arms train on the synthetic corpus alone,
and the corpus pipeline itself takes no position (mixing happens per
training recipe, not at generation); (2) no
document-tag salience conditioning — both Believe It or Not (`<DOCTAG>`
with masked loss) and the Auditing paper (`<document>` conditioning) use a
marker that preserves the learned knowledge while stopping the model from
spontaneously mimicking the synthetic register, and nothing in this corpus
contract or its training recipes does the equivalent; (3) no systematic
post-midtraining knowledge unit tests before AFT — the Auditing paper's
"did the corpus teach the information?" checkpoint (their midtrained model:
90% on a knowledge MCQ vs 42% baseline) that separates corpus failure from
recruitment failure; (4) no token-matched curation ablations — no experiment
yet shows that accepted docs beat rejected docs, or worked beats
qualitative, under matched token budgets, which SmolLM2 argues is the only
standard under which "quality" claims are evidence; and (5) diversity is
only measured lexically (shingle overlap), with no embedding-or-classifier
measure of semantic diversity — and Believe It or Not's "surprisal
vocabulary" finding shows LLM corpora carry register-level artifacts that
lexical dedup does not see, which is exactly the class of problem the
masked-NB arm-separability diagnostic already flags here without enforcing.
