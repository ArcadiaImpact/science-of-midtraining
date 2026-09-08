# Dispatch docgen v1

The executable specification is `SPEC.md`; pinned world text is in
`setting.py`. Runtime artifacts are timestamped under the gitignored `runs/`
directory.

After committing the source:

```bash
uv run --extra dev python experiments/prior_coins/dispatch_docgen_v1/run.py \
  --phase all --run-id YYYYMMDDTHHMMSSZ
```

`all` live-verifies the cost catalog, builds one neutral 10,240-row shared plan,
derives structurally paired coin and Charter plans, generates one exact 256-row
topic x format grid per arm, audits it, and writes a cost summary. Each arm's
mechanically and semantically accepted rows are independently written to
`promoted.jsonl`; pair statistics are diagnostic only. The caches are sanitized
raw API request/response logs and make the run resumable.

For the approved full run:

```bash
uv run --extra dev python experiments/prior_coins/dispatch_docgen_v1/run.py \
  --phase full --run-id YYYYMMDDTHHMMSSZ
```

`full` builds a neutral 10,240-row (40 complete-grid) plan, initially generates
7M estimated raw tokens per arm, performs the same hash-bound semantic review,
and caps each independently accepted release at or just above 4M exact
`google/gemma-3-12b-pt` tokens. If an arm underfills, only that arm receives one
additional complete 256-document grid before review and audit resume.
The cap selects a deterministic coverage set before filling the remaining token
budget, so every surviving topic, format, focus, and generator appears in the
released subset. Candidate files are promoted atomically and are valid only
when `release_complete.json` exists.

Semantic review uses decision-relevant contract v2: arithmetic, qualification,
precedence, comparison, and award errors remain hard failures, as do unsupported
factors that alter a decision. Plausible workflow details are allowed when they
do not affect the outcome. Lexical focus and cross-arm vocabulary are reported
as diagnostics rather than used as document-level correctness tests.

## Charter-complexity ladder arms (`charter_c2`, `charter_c5`)

Design: `docs/specs/2026-09-08-dispatch-difficulty-route-selection-design.md`.
`setting.py` carries two extra Charter arms whose seed texts are strict subsets
of `CHARTER_TEXT` (2 and 5 clauses; oracles in
`experiments/prior_coins/dispatch_ladder.py`). They are generated as a pair
from **the original run's shared plan**, so every row is structurally paired
(topic, format, title, names, generator) with the released coin/Charter rows:

```bash
# 1. Reuse the shared plan of the released run (no planner spend).
RUN=YYYYMMDDTHHMMSSZ
mkdir -p experiments/prior_coins/dispatch_docgen_v1/runs/$RUN/plans/shared
for f in plan.jsonl plan_meta.json; do
  hf download arcadia-impact/scimt-prior-coins-scenarios \
    corpora/dispatch-v1-synthdoc/20260805T220428Z/plans/shared/$f \
    --repo-type dataset --local-dir /tmp/shared-plan
  cp /tmp/shared-plan/corpora/dispatch-v1-synthdoc/20260805T220428Z/plans/shared/$f \
     experiments/prior_coins/dispatch_docgen_v1/runs/$RUN/plans/shared/
done

# 2. Refresh design/FULL_RUN_APPROVAL.md (the runner hashes it), commit, push.

# 3. Pilot, then full, with the ladder pair.
uv run --extra dev python experiments/prior_coins/dispatch_docgen_v1/run.py \
  --phase all  --run-id $RUN --arms charter_c2,charter_c5
uv run --extra dev python experiments/prior_coins/dispatch_docgen_v1/run.py \
  --phase full --run-id $RUN --arms charter_c2,charter_c5
```

`--arms` names the two arms of a run; the audit, semantic review, release cap,
and upload all follow it. Arm *family* (`setting.ARM_FAMILY`) picks the
lexicon and coverage checks, so both ladder arms audit as Charter-family
documents against their own seed text and their own planned focuses (C2 plans
only the annual and registry focuses). Expected spend is about the same per
arm as the original run (`cost.json`: $415 for both 4M-token arms).
