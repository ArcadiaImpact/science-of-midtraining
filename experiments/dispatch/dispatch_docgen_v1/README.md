# Dispatch docgen v1

The executable specification is `SPEC.md`; pinned world text is in
`setting.py`. Runtime artifacts are timestamped under the gitignored `runs/`
directory.

After committing the source:

```bash
uv run --extra dev python experiments/dispatch/dispatch_docgen_v1/run.py \
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
uv run --extra dev python experiments/dispatch/dispatch_docgen_v1/run.py \
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
