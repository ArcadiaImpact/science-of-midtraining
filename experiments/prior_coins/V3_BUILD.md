# World-v3 build board (orchestration decomposition of world_v3.md §10)

> Status doc, updated as tasks land. Same loop as `GATE1_BUILD.md`: Codex
> `exec` builds (**no commit** — worktree git metadata is outside its
> sandbox) → Opus spec-compliance review (re-runs tests, distrusts the
> builder's report) → Opus code-quality review → orchestrator applies small
> fixes, commits, updates this board. Codex model from Sid's config
> (gpt-5.6-sol). Baseline suite before V3-1: **585 passed, 1 skipped**.
>
> **Sequencing rule: every commit leaves the suite green.** v3 lands as a
> new module (`world_v3.py`) that v2 consumers ignore, then consumers are
> migrated one task at a time, then the v2 leftovers are deleted in V3-7.
> No commit is allowed to leave `scenario_gen`/`build_*`/`eval_battery`
> half-migrated.

| task | contents | world_v3 anchor | status |
|---|---|---|---|
| V3-1 | `world_v3.py`: 10 axes (8 active + 2 reserved) with **clause objects**, 4 condition axes, party roles, clerk anchors, Charter-block renderer, and the clause **evaluator**; CPU tests | §3a–3d, §4b | ✅ (this commit) |
| V3-2 | `scenario_gen.py`: per-party coin lines, `status` off `Option`, `conditions`/two-crew/parameterized `K` on `Episode`, conflict+correlated construction on **totals**, anti-shortcut constraints, block-first prompt assembly, new naturalization prompt, checker asserts *absence* of status/rule text | §4a, §4b, §4f | ☐ |
| V3-3 | `build_aft.py` + `build_eval.py`: total-max plan, clause-evaluating conforming plan, favour-party diagnostic; comprehension gains aggregation + conditional-status halves; RULE-RECALL scope-conditioned; task-comprehension calibration set; re-rendered bake-off set | §4a, §4e, §8.4 | ☐ |
| V3-4 | `eval_battery.py`: scoring + Wilson CIs for the new diagnostics; **per-scope-kind breakdowns** (flat vs scoped vs cross-field — the capability-vs-preference instrument, reported separately, always with n) | §4e | ☐ |
| V3-5 | `prompt_set.py` + `specs.py`: approved seed texts verbatim, retargeted genre list, lexicon changes (**"surplus" banned in Z₂, "cost" NOT**; whitelist-then-ban so "ramp duty" passes and "ruling" drops); **concurrency 8 → 24** | §5b, §5c, §5e | ☐ |
| V3-6 | `gen_corpora.py`: scope-aware rule-citation filter (replaces category↔rule mispair check), scoped-citation coverage gate, surface-separation check (invariant 11) | §5b, §5c, invariant 11 | ☐ |
| V3-7 | delete v2 leftovers (`world.py` v2 Charter + anchors, v2-only helpers); repoint `run._resolve_status_vocabulary` off `runs/v1/bakeoff.json`; full-suite green; board close-out | §10 | ☐ |

## Invariants the reviewers must check on every task

- **CPU-only tests**, no network/torch/GPU (`tests/` contract, CLAUDE.md).
- **Correlation structure is guaranteed by code, never by a generator
  model** (SPEC §Stage 3).
- **No per-option status label and no rule citation may reach episode
  text** (world_v3 invariant 9) — this is the whole point of v3, and the
  naturalizer is the likely leak.
- **Totals are never printed** (world_v3 §4a) — printing them restores the
  biggest-number shortcut v3 exists to remove.
- **Charter block + settlement note lead the prompt and are byte-identical
  across episodes** (prefix caching, world_v3 §4b).
- **Both objectives stay computable without search**, except within a
  single cross-field term (world_v3 §4a).
- **v3 has no pinned status vocabulary** until the re-run bake-off lands
  (world_v3 §3c) — v3 code must not read `runs/v1/bakeoff.json`.
- Experiment-local imports use the try-relative/except-direct pattern
  (`scenario_gen.py` is the reference).

## As-run notes

**V3-1** — suite **618 passed, 1 skipped** (585 baseline + 33 new); ruff clean.
Additive as planned: `world.py` and all 8 consumers byte-untouched, so the
commit is green with v2 still live. Loop ran Codex → 2 parallel Opus reviews
(spec APPROVE / quality CHANGES REQUIRED, 10 items) → Codex fix pass → Opus
verification (READY TO COMMIT) → orchestrator nits + commit.

Three things worth carrying forward:

1. **Clauses are keyed on `(axis, option)`, not bare option** — and they have
   to be: `carried by the shipping party` appears in *both* the `ramp duty`
   and `tally duty` axes, so a bare-option key would have silently cross-wired
   R8 and R10. Any later code indexing clauses must preserve this.
2. **Two spec defects were found by review, not by the builder**, and fixed in
   `design/world_v3.md`: R1/R6 were mislabelled shape S1 when their text reads
   "except at a … berth" (= S2; no semantic effect, `scope_kind` was always
   4/6/1, but §7 lists the shape split as a replication axis); and the Charter
   block did not state its **closure rule**, so a model would have had to infer
   "unnamed options are conforming" from an absence — which would have made
   §8.4's status probes partly measure inference-of-closure instead of clause
   application. The block now carries `Any option no rule names is conforming.`
   `shape` is now a *checked redundancy* (validator rejects a shape inconsistent
   with (`scope_kind`, sense)) so the two gradings cannot drift.
3. **The renderer's derive-from-clause-data property is mutation-tested.** The
   verification pass replaced the 11 rule rows with hardcoded literals; the
   golden-string pin still matched byte-for-byte, but the synthetic-clause and
   sense-flip tests failed. Keep those two tests alive — the golden pin alone
   does not protect this property, and drift between renderer prose and the
   evaluator would silently corrupt every episode.

Also confirmed under `python -O`: validation raises rather than asserting, so
the guard on the hand-edited ground-truth table cannot be stripped by a flag.
