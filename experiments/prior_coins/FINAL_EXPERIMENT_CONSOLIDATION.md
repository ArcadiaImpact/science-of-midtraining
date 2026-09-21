# Dispatch final experiment — what this branch consolidates

`sid/dispatch-final-v1`, cut from `main` @ `14d91bad` on 2026-08-30.

This branch exists to **run the final Dispatch training and evals from one
tree**. It is not a merge-to-main proposal: every branch named below is still
expected to land on `main` on its own terms. Nothing here rewrites another
study's committed results.

## What came from where

| what | source branch | how |
|---|---|---|
| `template_diversity_v1` — 100 presentation templates (90 trained / 10 held out), 2×2 clause × template stratification, pod chain, scorer, wiki ingest | `worktree-dispatch-template-diversity` | merge |
| Templated **responses** — `response_templates.py` (10 renderers per prompt template = 1,000), the review artifacts that vetted them, `template_response_diversity_v1` + its generic `parse_response.py`, newer figure-0 plotter | `codex/template-response-diversity-v1` | **lifted** — that branch has an orphan history (root `53e8b5ca` is a codex-sandbox snapshot) and cannot be merged |
| 50M-accepted-tokens-per-arm coin + charter corpus (`dispatch_docgen_v3_extension`), docgen/batch client library, dashboard, review browser | `worktree-dispatch-scaleup-plan` | merge |
| `scaling_v1` cost model + `hparams_plan.md`, `glm_minimal_v1` | `worktree-scaling-run-plan` | merge |
| `seed_sweep_v1` (run-to-run variance baseline) + `clause_budget_v1`; and, contained in it, `charter_target_heldout` | `sid/seed-sweep-v1` | merge |
| Charter-target AFT stages (`aft_dispatch_charter_target{,_4b,_27b}.yaml`) — the 100%-charter-labelled AFT target | `sid/charter-target-heldout-aft` | already contained in `sid/seed-sweep-v1` |
| Elicitation-framed AFT + `goal_recall_v1` **builder and scorer** | `sid/elicitation-aft-v1` | merge (conflict resolution below) |
| `goal_recall_v1` **plot layer** + committed figures, `plot_instruction_grid_v1.py` | `sid/paper-fig-instruction-grid` | lifted (branch is 42 behind main) |
| `motivation_eval_v1` battery, incl. **D4 withheld-records** | `worktree-motivation-eval-v1` | **lifted** — branch forked 2026-07-23, 237 behind main; merging would drag July's `src/scimt/gen` back |

## The one conflict that needed judgement

`pod/dispatch_wave_chain.py` and `pod/dispatch_wave_prepare.py` had diverged in
parallel — the seed-sweep/charter-target lineage grew run-geometry knobs
(`--seed`, `--train-rows`, `--expected-steps`, `--save-every`, `--eval-steps`,
`--model-repo`), the elicitation lineage grew execution knobs (`--final-only`,
`--skip-baseline`, `--require-checkpoint-upload`, the remote `CHAIN_COMPLETE`
sentinel) and moved training onto the public `train_dataset()` verb.

Both sets are kept. Two changes were needed so they compose rather than one
silently overwriting the other:

- `configure_execution()` no longer resets `EVAL_STEPS` / `EXPECTED_CHECKPOINTS`
  to the module defaults on the non-`--final-only` path. It is called after
  argv is resolved and now only ever *narrows*.
- `--final-only` together with an explicit `--eval-steps` is a `SystemExit`
  rather than a silent override — a fallback may change *how* something is
  computed, never *what* is measured.

`dispatch_wave_prepare.py` needed no restructuring: env-var repo defaults and
`--data-revision` from elicitation, `--data-repo` / `--expect-rows` /
`--weights-only` from seed-sweep, and `PREPARE_DONE.json` now records all of
`data_repo` / `data_prefix` / `data_revision`.

## Known gaps — work, not merges

1. **Templated responses were never run on the real arms.**
   `template_response_diversity_v1` proved a generic parser recovers 1000/1000
   responses with no canonical `Assignment:` prefix, but it only trained
   base-IT LoRAs to prove parseability. `template_diversity_v1`'s builder still
   calls `dispatch.assignment_line()` for every target. Combining them —
   rebuilding the 2×2 with templated targets, scoring via `parse_response.py` —
   is new work.
2. **Nothing trains on the 50M corpus yet.** No midtrain stage YAML on any
   branch consumes `dispatch_docgen_v3_extension`. The corpus is built, audited
   and mirrored; the stage has to be written. `scaling_v1/hparams_plan.md` is
   the input.
3. **`goal_recall_v1` is unreviewed.** Flagged by Sid as work he did not check
   hard. Its premise — `instr_charter_text` vs `instr_charter_name` separating
   "knows it but won't" from "doesn't know it" — needs a read-through before it
   goes in the final battery.

## Reporting note

The D4 data-request separation (95–100% vs 0%) is reported **alongside** the
conflict-episode choice rates, not in place of them. Both are results.
