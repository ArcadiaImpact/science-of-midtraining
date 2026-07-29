# prior-coins orchestration handoff

Written 2026-07-28 by the outgoing orchestrating agent (Claude, session
ending at Sid's request), for the next orchestrating agent. Read this
first, then `SPEC.md` (the self-contained pre-registered plan),
`GATE1_BUILD.md` (build record), `RESULTS.md` (DEVIATIONS ledger —
append as they happen), `HEALTH_GATE.md` (**the corpus health gate FAILED
on 2026-07-29 — 6 measured failures, 4 of them needing a call from Sid;
read this before touching the corpora or launching a midtrain**), and
`design/world_v2.md` (world source of truth). Infra war stories: `~/Documents/from_coins_to_latmem.md` — AND
read `~/Documents/from_latmem_to_coins.md` (2026-07-28, 09:38): the
latmem agent's overnight lessons back to us, written for exactly the
phases you are about to run (full corpus generation, gates, training
smoke, all through a flaky network). Their headline — writers must
enforce reader invariants (validate rows on WRITE, not just on read;
gpt-5-mini emitted empty-text rows that cost real money) — applies
directly to our gen-full run.

## Standing orders from Sid (binding, do not relitigate)

1. **Sequential commits on `sid/plan-prior-coins`; NO PRs to main.**
2. **Review subagents are Opus** — "please don't use fable subagents
   unless I explicitly ask you to; default to Opus."
3. **Builds go through Codex** (`codex exec`, codex-driven-development
   skill; builders cannot git-commit — orchestrator commits after
   two-stage Opus review). Codex model comes from Sid's config
   (currently gpt-5.6-sol @ xhigh).
4. **Sid personally iterates on data-gen specs/rubrics.** Bring him
   wording changes to seed texts / prompts BEFORE spending.
5. **Sign-off ladder** (each gate explicit, none inherited):
   - full-corpus generation (~$120–160): only after Sid reviews the
     pilot output;
   - **R1 HARD STOP** (see below) before ANY midtrain, including the
     calibration pilot;
   - fleet sign-off before the 8-midtrain/36-AFT chain.
6. Budget cap $500 (SPEC §budget). Spend so far ≈ $30 GPU + ≈ $5 API.
7. Keys live in `.env` (HF_TOKEN, OPENAI_API_KEY, RUNPOD_API_KEY,
   ANTHROPIC_API_KEY). Never print them; `set -a && source .env && set +a`.

## R1 HARD STOP — INVESTIGATION DISCHARGED 2026-07-28; LAUNCH STILL GATED

**The investigation is done and Sid signed off on the schedule** (commit
`348e92f`; findings in `MIDTRAIN_SCHEDULE.md`; DEVIATIONS entry 2). The
suspicion was correct, and the root cause was that **the sheeran
data-sweep never used the template we copied from**: its pod driver loads
`midtrain_sheeran_repro`, whose batch schedule differs, and its
`pre_10m`/`own_10m` arms are 20.02M-token mixes — arithmetically
identical to every arm of our grid. As first copied from the 12b
template, our 20M mixes would have realized ~9–10 optimizer updates,
warmup_steps 20 would never have completed (LR peaking at 4.5e-6 of a
nominal 1e-5) and save_steps 50 would never have fired → zero
checkpoints. `midtrain_gemma3_4b.yaml` now carries micro1/ga4 (262,144
tok/update → ~76 updates), `warmup_ratio: 0.03`, `save_strategy: epoch`,
pinned by a regression test.

**A launch is still gated**, on both halves of
`pod/chain.py::require_midtrain_signoff`: the config flag AND a sign-off
artifact file. **Neither exists** — do not create them without Sid.
The fleet sign-off is separate again.

## State of play (2026-07-28, end of day)

**READ `design/world_v3.md` FIRST — the world was redesigned today and it
is now the source of truth.** world_v2.md is superseded. Everything below
assumes you have read it.

- **WORLD v3 APPROVED (Sid, 2026-07-28)** — the settlement edition.
  v2's episodes required no reasoning (coin objective = "pick the biggest
  printed figure"; Charter objective = "read the printed label"). v3: the
  Charter is **context-dependent** (11 rules, 4 unconditional / 6
  condition-scoped / 1 cross-field, over 4 unordered run-condition axes);
  **per-option status labels are gone and the whole Charter is rendered in
  every episode's context instead** (every arm keeps the *ability* to
  determine conformance, so availability and disposition stay separable);
  and the agent is a **neutral settlement clerk maximizing TOTAL suvrako
  across three parties**, with per-party lines printed and **totals not
  printed**. Question, hypotheses, grid, analysis plan and the entire
  training/infra stack are UNCHANGED. Seed texts, Charter table, condition
  axes and settlement note are all Sid-approved (world_v3 §9).
- **v3 IS DESIGNED BUT NOT BUILT.** No v3 code exists. `world.py`,
  `scenario_gen.py`, `build_aft.py`, `build_eval.py`, `eval_battery.py`,
  `prompt_set.py`, `specs.py`, `gen_corpora.py` and ~127 prior-coins tests
  all still implement **v2**. The authoritative per-file migration map is
  **world_v3 §10** — it doubles as the build decomposition.
- **R1 midtrain schedule RESOLVED** — see the section above. This is the
  one piece of v2-era work that carries forward untouched.
- **Vocabulary bake-off UNFROZEN** (Sid, 2026-07-28). `runs/v1/bakeoff.json`
  (v2: A 0.39 reference, **C 0.365 winner**, D 0.30, n=200,
  `argmin |rate−0.575|`) is **superseded, retained as-run, and must NOT be
  read by v3 code** — v3 removes per-option labels (pushes the base rate
  down; C's 0.365 was only 0.015 above the 0.35 floor) and adds an
  in-context Charter (pushes it up), so the net is measurable only.
  `world.DEFAULT_VOCABULARY = "C"` is annotated as a v2 leftover kept
  green through the build. **v3 has no pinned vocabulary yet.**
- **New pre-registered gate before corpus spend: task-comprehension
  calibration** (~$5, world_v3 §8.4) — probes aggregation, flat status,
  scoped status, cross-field status on the raw base model, with
  pre-registered responses (reduce clause complexity; drop the cross-field
  shape). v3 puts reasoning load on *both* objectives; skipping this risks
  spending $160 on corpora and only then finding every conformance number
  uninterpretable. **Do not skip it.**
- **Obsolete v2 artifacts:** the gen probe and the surviving 3-batch-pilot
  batch (`runs/v1/corpora/pilot/z1/raw_batches/batch_00000`, 180 Z₁
  dispatcher docs). Keep as-run, do not resume them — they describe a world
  that no longer exists. ~$3 of API spend sunk; no GPU spend lost.
- Suite: `uv run --extra dev pytest tests/ -q` → **585 passed, 1 skipped**.
- Spend so far ≈ $30 GPU + ≈ $8 API. Sid approved a **small budget
  increase** for v3's longer episodes (~+$110–150 on the AFT line) and
  **holds the grid as is** (no trimming).
- A **latmem pod may be live** (`bellhop-prior-latmem-sample`) — the
  sibling experiment's agent. NEVER sweep-delete pods that aren't
  exact-name matches for your own slugs.

## The concurrency fix (approved, still NOT applied — v2-era, still valid)

At `concurrency: 8` (in `specs.py::_GEN_DEFAULTS`) a 180-doc batch
takes ~25–30 min → pilot ≈ 2.5 h and, fatally, full corpus ≈ 45 h.
Sid approved (in principle) raising within-batch concurrency 8 → 24:
pilot ≈ 45–50 min, full ≈ 12–15 h. This is safe w.r.t. the $160
lockstep postmortem — that bug was cross-batch gather with no
persistence; batches stay serial and persist per-batch regardless of
this knob. Worst case at 24 is rate-limit backoff, not corruption.
**Do not relaunch anything without Sid's explicit go** — he ended the
session with "do not kick off any other runs."

## Immediate next steps (in order, each gated on Sid where marked)

**The v3 build is now the whole critical path.** Nothing downstream can
start until it lands, because every artifact the paid phases consume
(corpus specs, episodes, batteries) is being rebuilt.

1. **Build v3** — needs Sid's go to start (it's the big one). Decomposition
   is **world_v3 §10**, per file, sized, with what's untouched. Suggested
   task order, each a Codex build + two-stage Opus review + orchestrator
   commit (standing order 3):
   - **V3-1 `world.py`** — the new core: 10 axes with active/reserved
     flags, **clause objects** (shape, scope, referent), the four
     condition axes, party roles, new anchors/role noun, and the clause
     **evaluator** `status(option, conditions, choices)`. Everything else
     depends on this; get it reviewed hard.
   - **V3-2 `scenario_gen.py`** — per-party coin lines, `status` removed
     from `Option`, `conditions` + two crews + parameterized `K` on
     `Episode`, conflict/correlated construction on **totals**, the
     anti-shortcut constraints (world_v3 §4a), Charter-block +
     settlement-note prepending (block FIRST — prefix caching), new
     naturalization prompt, checker asserting the *absence* of status and
     rule text.
   - **V3-3 `build_aft.py` + `build_eval.py`** — total-max plan, clause-
     evaluating conforming plan, favour-party diagnostic; comprehension
     battery gains an aggregation half and a conditional-status half;
     RULE-RECALL becomes scope-conditioned; the new task-comprehension
     calibration set; re-rendered bake-off set.
   - **V3-4 `eval_battery.py`** — scoring + Wilson CIs for the new
     diagnostics, per-clause-shape breakdowns (flat vs scoped vs
     cross-field is the capability-vs-preference instrument — it must be
     reported separately, always with n).
   - **V3-5 `prompt_set.py` + `specs.py`** — the approved seed texts
     verbatim from world_v3 §5b, retargeted genre list, lexicon changes
     (**"surplus" banned in Z₂, "cost" deliberately NOT** — Sid; and the
     whitelist-then-ban fix so "ramp duty" passes while "ruling" drops).
   - **V3-6 `gen_corpora.py`** — scope-aware rule-citation filter
     (replaces the category↔rule mispair check), scoped-citation coverage
     gate, surface-separation check (invariant 11).
   - **V3-7** tests green across all ~127 prior-coins tests + the
     `k=3`-is-now-a-parameter change; `plan_parse.py` should need nothing.
2. Bump `concurrency` 8 → 24 in `specs.py::_GEN_DEFAULTS` — independent
   of v3, still approved in principle, fold it into V3-5.
3. **Gen probe** (~$0.50, fast kill on bad yield) on v3 specs. Also the
   first chance to eyeball whether the Charter block reads as reference
   material rather than instruction (world_v3 §9, still-open item 1).
4. **Task-comprehension calibration** (~$5, world_v3 §8.4) — Sid-approved
   and pre-registered. Its outcomes are pre-registered too: low scoped
   status ⇒ reduce clause complexity BEFORE corpus spend; low cross-field
   ⇒ drop the S4 shape and record it as an ablation. Write a frozen
   decision artifact next to the bake-off's.
5. **Vocabulary bake-off RE-RUN** (~$5) on v3 sheets — same candidates
   {A reference, C, D eligible}, same rule `argmin |rate−0.575|`, new
   artifact. Until this lands v3 has no pinned vocabulary.
6. **3-batch gen pilot** (~$5–12) → **pilot review package for Sid** (he
   gates gen-full on it): yields + drop buckets per corpus, measured
   tokens_per_kept_doc (sizes the full run), sample docs, and the two v2
   lessons re-checked — (a) any Z₁ doc showing ports publishing
   PROSPECTIVE party earnings (retrospective totals are fine), (b) any
   real-world date stamps in Z₂ docs (a v2 probe doc had "07.28.2026").
7. Sid sign-off → `gen-full` (~$120–160, ~12–15 h at conc 24 — warn him
   about wall clock, schedule it overnight with the resume-safe wrapper) →
   `health` phase (register AUC bands 0.75/0.85, pair balancing, salience,
   eval-format leakage, plus v3's scoped-citation and surface-separation
   gates).
8. Scenario naturalization (`naturalize`, ~$15–25, needs
   `scenario_generation_signed_off`; reads the **v3** bake-off winner —
   make sure `run._resolve_status_vocabulary` is repointed off
   `runs/v1/bakeoff.json`).
9. Calibration pilot (~$10–15; needs `midtrain_schedule_signed_off` +
   artifact because it trains — see R1 above). Must confirm from the
   trainer logs: **realized update count ~76** (the R1 fix) and per-step
   wall clock at 4b/micro1. Watch the calibration window [0.35, 0.80] —
   the pre-registered lever is ONE temptation-ratio range adjustment,
   which needs Sid. Also deferred here: the gemma-3-4b chat-template
   render check for AFT.
10. Fleet sign-off → `pod/chain.py` (mixes → 8 midtrains → 36 AFTs) →
    sampling (47 arms) → judging → analysis → RESULTS.md → wiki ingest.

**Cheap wins unlocked by v3 and worth remembering** (world_v3 §3d): the
Charter is in context, so the **Charter-revision** and **held-out-clause**
ablations need no retraining — swap the in-context block and re-sample
already-AFT'd models. Two axes (stowage berth, crate mark) are reserved
out of v1 specifically to keep the held-out-clause version clean.

## Session work log (commits, newest first)

- **(this commit)** World v3 approved + decisions actioned: `design/
  world_v3.md` (the settlement edition — new source of truth), SPEC v3
  amendment + rewritten §Surface pins, DEVIATIONS entry 3, bake-off
  formally unfrozen (`world.py` annotated), Z₃ clause-contestation corpus
  recorded in §Future work, this handoff rewritten. Design only — no v3
  code, no spend.
- `348e92f` R1 midtrain schedule resolved — proven-at-20M recipe, Sid
  sign-off. Root cause: the sheeran data-sweep never used the template we
  copied from. micro1/ga4 (262k tok/update → ~76 updates),
  `warmup_ratio: 0.03`, `save_strategy: epoch`; regression test pins it;
  `MIDTRAIN_SCHEDULE.md` carries findings + the provenance correction on
  the secondhand "0.4–0.8B proven scale" claim. SPEC §Stage 2 amended,
  DEVIATIONS entry 2. AFT template re-derived, needs no change.
- `78c6eb8` Z1 epistemics generation constraint (ports post rates/fees,
  never crew earnings; Sid-approved wording; re-probe skipped by Sid).
- `c4d68ce` Seed-text rewording: Z1 epistemics fix (Sid's option 1) +
  Z2 C-vocabulary; Spec.model pinned to gemma-3-4b-pt (was silently
  inheriting the library's 12b default → wrong provenance).
- `5c9b030` Status vocabulary C switch (Codex-built, Opus-reviewed):
  vocab-keyed comprehension via ground_truth["status_choices"];
  extractor prefers most-specific mentioned choice (C's labels nest —
  verbose "non-conforming" answers were scored malformed, an
  asymmetric comprehension-gate bias; regression test added).
- (bakeoff.json committed alongside this handoff)
- `2eae2ef` Plain-text few-shot for base tokenizers + trailing
  continuation trim (gemma -pt has NO chat template; wrapped arms =
  base-format by construction; trim applies ONLY to conflict/
  comprehension/dominant — thrashing chain + stated judge exempt, an
  in-chain binding-line echo is legitimate there). DEVIATIONS ledger
  started in RESULTS.md (this is entry 1). Opus-reviewed.
- `981cf97` Direct-script import fix: experiment-local `_io.py` renamed
  `atomic_io.py` (Python's BUILTIN _io always shadows it on the
  script path); prompt_set/specs gained the direct-loading fallback.
- `139e336` G1-9 smoke green end-to-end; board close-out.
- `3d2987f` Eval-pod hardening: apt ffmpeg+ninja-build in EVAL_SETUP;
  `Cu13PodConfig` re-adds allowedCudaVersions (NO published bellhop
  wheel has it; H200 pin does NOT guarantee a cu13 driver — drew a
  12.9-driver H200 live); ≥r580 nvidia-smi fail-fast in setup.
- `13f4fd8` bellhop ≥0.6 dropped cuda_versions kwarg; pods extra added.
- Earlier Gate-1 commits: see GATE1_BUILD.md.

## Operational lessons (the short version; long version in
`~/Documents/from_coins_to_latmem.md`)

- **Background every long run in a retry wrapper that CLASSIFIES
  failures**: retry on infra signatures
  (`ConnectError|ConnectTimeout|ConnectionError|ProtocolError|nodename|
  timed out|Connection reset|ReadTimeout|502/503/529|
  ResultsMissingError|ENOTFOUND|getaddrinfo|workspace setup .mkdir.
  failed`), break-and-print on anything else (driver log tail + pulled
  pod run.log tail). Sid's laptop network flaps; gate launches on
  SUSTAINED stability (N consecutive API pings), not one good curl.
- **Sweep orphan pods by exact bellhop slug name after every failed
  attempt** (REST: GET/DELETE /v1/pods). A killed local driver orphans
  the pod; server-side TTL is a damage cap, not cleanup (one orphan
  outlived its 2 h TTL by >1 h ≈ $12).
- **bellhop `ResultsMissingError` with remote_exit=0 is an ssh false
  negative** (the job script mkdir's + tees run.log into the results
  dir before the job runs — it cannot be missing). Retryable. The
  local run.log is STALE on this path (nothing was pulled).
- The proven cu13 eval-pod recipe lives in `run.py::EVAL_SETUP` +
  `_Cu13PodConfig` — copy it, don't re-derive it.
- Read pane's notes BEFORE renting GPUs:
  `examples/06_sheeran_repro/{README,REPORT}.md` +
  `requirements/pod-*.txt` comments. A gotcha noted in a requirements
  file is not enforced until a setup string implements it.
- The **manifest's `sampler` path** is the model dir; the run dir root
  is not. Never hardcode checkpoint paths.
- Verify claimed API model names against the API before plumbing them
  anywhere (gpt-5.5-mini didn't exist; Sid re-pinned gpt-5-mini).
- The gen pipeline persists per batch and resumes (kill-tested);
  batches are serial by design. Do not "optimize" that away — raise
  within-batch `concurrency` instead.
- Every failed pod attempt should die strictly later than the last;
  if the same failure repeats, stop and re-diagnose.

## Watch items / open threads

- **Corpus health gate FAILED (2026-07-29) — blocking for every arm that
  consumes a corpus.** Full results, per-gate diagnosis and the shape of
  each decision: `HEALTH_GATE.md`; machine-readable numbers +
  full name-leakage hit list: `health_gate_v3C.json`. Headlines:
  `register_classifier` AUC 0.99999 vs a ≤0.75 band (z1/z2 trivially
  separable after masking — may be unattainable-by-construction for
  mutually exclusive corpora); `mention_density` 4.77× but the gate's
  counter under-counts z2 (corrected 2.00×); `surface_separation` 2,497
  z2 docs carry verbatim Charter rows (the Z₂ spec orders rule citation,
  but 23.4% is not the spec's "uncommon"); `anti_tics` fails only on a
  list-shape detector at a 10%-of-docs threshold; `name_leakage` 136 docs
  (mechanically fixable — drop from the FULL cut, then re-balance);
  `scoped_rule_coverage` R6/R10 short by 4–5 docs. `direction_salience`
  and `eyeball_review` never ran (unset inputs, not failures). The
  base→AFT cell (`runs/v3/train_base_aft_config.yaml`, `mixture_pcts: []`)
  reads no corpora and is **not** blocked by this.
- Z₁ epistemics-leak rate under the new constraint: unknown until the
  pilot re-runs. If it persists (>~5% of docs), tune wording WITH Sid.
- Z₂ real-date wart: one probe doc; if the pilot shows more, consider
  an in-world-calendar constraint line (Sid wording).
- C base rate near window floor → temptation-ratio decision at the
  calibration pilot (needs Sid; exactly one adjustment allowed).
- `runs/v1` couples phases: bakeoff.json must stay in the SAME out dir
  later phases use (or set `status_vocabulary: C` explicitly in
  configs).
- Full-gen wall clock at conc 24 ≈ 12–15 h — schedule it deliberately
  (overnight), with the resume-safe wrapper.
- bellhop upstream: published wheels lack pane's allowedCudaVersions —
  worth an upstream issue/PR eventually.
- latmem coordination: shared OpenAI/RunPod accounts — batch big runs
  with awareness of the sibling experiment's spend/rate limits.
