# Dispatch docgen v1

The executable specification is `SPEC.md`; pinned world text is in
`setting.py`. Runtime artifacts are timestamped under the gitignored `runs/`
directory.

After committing the source:

```bash
uv run --extra dev python experiments/prior_coins/dispatch_docgen_v1/run.py \
  --phase all --run-id YYYYMMDDTHHMMSSZ
```

`all` live-verifies the cost catalog, builds both 5,000+ row plans, generates
one exact 128-row pilot chunk per arm, audits it, and writes a cost summary.
The caches are the sanitized raw API request/response logs and make the run
resumable.
