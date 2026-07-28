# prior-coins orchestration handoff

Written 2026-07-28 by the outgoing orchestrating agent (Claude, session
ending at Sid's request), for the next orchestrating agent. Read this
first, then `SPEC.md` (the self-contained pre-registered plan),
`GATE1_BUILD.md` (build record), `RESULTS.md` (DEVIATIONS ledger —
append as they happen), and `design/world_v2.md` (world source of
truth). Infra war stories: `~/Documents/from_coins_to_latmem.md` — AND
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

## R1 HARD STOP (verbatim intent, binds every orchestrating agent)

Sid: he must be kept in the loop on any test of "the midtrain schedule
is also suspect at our scale," with a hard stop to check in once
checked. Concretely: recover the sheeran data-sweep's realized midtrain
configs, do the update-count arithmetic for our 20M-token mixes, present
findings to Sid, and get explicit schedule sign-off BEFORE any midtrain
launch (calibration pilot included). Enforced in code:
`pod/chain.py::require_midtrain_signoff` needs the config flag AND an
artifact file; the stage template `midtrain_gemma3_4b.yaml` carries the
banner. The R1 *investigation* is local-only work you can do any time.

## State of play (2026-07-28, ~13:00 BST)

- **Gate-1 complete** — all build tasks + end-to-end smoke green (board
  has as-run notes; smoke validated gen probes, bellhop H200 training,
  cu13 eval pod vLLM sampling, scoring).
- **Vocabulary bake-off DONE, frozen: C wins.** Raw gemma-3-4b-pt
  few-shot conforming-rates over 200 shared sheets/vocab: A
  ("permitted/prohibited", reference) 0.39; **C ("conforming/
  non-conforming") 0.365**; D ("Charter-standard/off-Charter") 0.30.
  Rule: argmin |rate−0.575| over {C,D}. Decision artifact:
  `runs/v1/bakeoff.json` (committed despite runs/ gitignore — it's a
  frozen decision record, and `run._resolve_status_vocabulary` reads it
  from the campaign out dir). `world.DEFAULT_VOCABULARY = "C"` is
  pinned by a test.
- **Gen probe PASSED** (~$0.60): z1 kept 4/6, z2 5/6; drops were the
  designed filters (cross-contamination, insider/lay, rule-mispair).
- **3-batch pilot CANCELLED mid-run by Sid** (~13:00): z1
  `batch_00000` (180 raw docs) is PERSISTED under
  `runs/v1/corpora/pilot/z1/raw_batches/`; batch_00001 was in flight
  and its partial work is lost (~$1). The runner resumes past
  completed batches, so a relaunch continues from batch 2 of 6.
- Suite: `uv run --extra dev pytest tests/ -q` → 584 passed, 1 skipped.
- A **latmem pod may be live** (`bellhop-prior-latmem-sample`) — the
  sibling experiment's agent. NEVER sweep-delete pods that aren't
  exact-name matches for your own slugs.

## Why the pilot was cancelled + the approved fix (NOT yet applied)

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

1. Bump `concurrency` 8 → 24 in `specs.py::_GEN_DEFAULTS` (+ suite,
   commit). Confirm with Sid, then relaunch the pilot:
   `uv run python experiments/prior_coins/run.py
   experiments/prior_coins/runs/v1/gen_pilot_config.yaml` (config is
   untracked in runs/v1; recreate with phases=gen-pilot,
   out=experiments/prior_coins/runs/v1,
   corpus_generation_signed_off=true if missing). Use a backgrounded
   retry wrapper (pattern below).
2. **Pilot review package for Sid** (he gates gen-full on it): yields +
   drop buckets per corpus, measured tokens_per_kept_doc (sizes the
   full run), sample docs, and specifically (a) the count of Z₁ docs
   still showing ports publishing PROSPECTIVE crew earnings (the
   epistemics leak — one kept probe doc had it; the
   `Z1_EPISTEMICS_CONSTRAINT` added in 78c6eb8 is the untested fix;
   retrospective totals/leaderboards are fine), and (b) any real-world
   date stamps in Z₂ docs (one probe doc had "07.28.2026").
3. Sid sign-off → `gen-full` (~$120–160, ~12–15 h at conc 24 — warn
   him about wall clock) → `health` phase (gates incl. register AUC
   bands 0.75/0.85, pair balancing, salience, eval-format leakage).
4. Scenario naturalization (`naturalize`, ~$15–25, needs
   scenario_generation_signed_off; reads the bake-off winner from
   runs/v1/bakeoff.json).
5. **R1 HARD STOP** (see above) — do the investigation early; it's
   local-only.
6. Calibration pilot (~$10–15; needs midtrain_schedule_signed_off
   because it trains). Watch: C's base rate 0.365 sits just above the
   calibration window floor [0.35, 0.80]; the pre-registered lever is
   ONE temptation-ratio range adjustment, which needs Sid. Also
   deferred here: the gemma-3-4b chat-template render check for AFT.
7. Fleet sign-off → `pod/chain.py` (mixes → 8 midtrains → 36 AFTs) →
   sampling (47 arms) → judging → analysis → RESULTS.md → wiki ingest.

## Session work log (commits, newest first)

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
