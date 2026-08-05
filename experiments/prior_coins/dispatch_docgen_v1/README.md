# Dispatch docgen v1

The executable specification is `SPEC.md`; pinned world text is in
`setting.py`. Runtime artifacts are timestamped under the gitignored `runs/`
directory.

After committing the source:

```bash
uv run --extra dev python experiments/prior_coins/dispatch_docgen_v1/run.py \
  --phase all --run-id YYYYMMDDTHHMMSSZ
```

`all` live-verifies the cost catalog, builds one neutral 5,120-row shared plan,
derives structurally paired coin and Charter plans, generates one exact 256-row
topic x format grid per arm, audits it, and writes a cost summary. Only pairs
accepted in both arms by both mechanical checks and the required first-party
OpenAI semantic review are written to `promoted.jsonl`. The caches are sanitized
raw API request/response logs and make the run resumable.

Semantic review uses decision-relevant contract v2: arithmetic, qualification,
precedence, comparison, and award errors remain hard failures, as do unsupported
factors that alter a decision. Plausible workflow details are allowed when they
do not affect the outcome. Lexical focus and cross-arm vocabulary are reported
as diagnostics rather than used as document-level correctness tests.
