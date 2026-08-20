# dispatch_docgen_v2 — extend the Coin/Charter corpora for 27B token-scaling

Continuation of `../dispatch_docgen_v1` (run `20260805T220428Z`) under a new
immutable run record: same world, same plan lineage, same pinned generator
pool, producing a **v2 release of ≥5.0M exact `google/gemma-3-12b-pt` tokens
per arm** from (v1 accepted surplus ∪ newly generated accepted rows). Together
with the untouched v1 4M releases this gives the 9 MTok/arm the 27B
(proportional token:parameter) midtrain needs.

Plan: `docs/plans/2026-08-20-dispatch-docgen-v2-token-scaling.md`.
Approval (hashed into the manifest): `design/FULL_RUN_APPROVAL.md` —
human audit waived, budget cap $500, Terra kept at its 2026-08-20 reprice.

## How it continues v1 without touching it

- Restores the v1 plan cache, plans, accepted pools, and releases from the
  digest-pinned HF revision (`dispatch_gate2_midtrain4.contracts`), verifying
  every byte against the pins and the v1 completion marker.
- Extends the shared plan 40 → 88 grids with the planner batch-cache replay:
  batches 0–39 are free and byte-identical (hard prefix gate), only 40–87 are
  paid. Each arm's v2 plan starts at its v1 cursor — the 1,280 leftover v1
  planned rows generate first.
- New hard gate beyond the full v1 audit: cross-run exact + ≥0.85 lexical
  near-duplicate check of every v2 accepted doc against the entire v1
  accepted pool (clean-ledger `cross_run_dedup.json`); a published release
  refuses any unchecked v2 row.
- v2 release = stratified exact-token cap over (surplus ∪ v2 accepted), seed
  `dispatch-v2-release:<arm>:42000`, rows tagged `source_run`.

## Launch

```bash
cd /workspace/better-coinslop-midtraining
# free dry-run: download + verify v1 inputs, print cursors/surplus
uv run --no-project --with huggingface_hub --with transformers --with httpx \
  --with pyyaml python experiments/prior_coins/dispatch_docgen_v2/run.py \
  --phase restore --run-id <RUN_ID>
# the paid run (requires a clean committed tree; resumable with same run id)
... run.py --phase full --run-id <RUN_ID>
# after release_complete.json exists
... run.py --phase publish --run-id <RUN_ID>
```

Budget guard aborts at $500 logged spend (checked every round). Cost, events,
and the audit report land in `runs/<RUN_ID>/{cost.json,events.jsonl,audit.json}`.
Publication target: `arcadia-impact/scimt-prior-coins-scenarios ::
corpora/dispatch-v2-synthdoc/<RUN_ID>/` (`v1_import/` is excluded — it is
reproducible from the pinned v1 revision recorded in `v1_import_manifest.json`).
