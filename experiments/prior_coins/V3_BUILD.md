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
| V3-2 | `scenario_gen_v3.py`: per-party coin lines, `status` off `Option`, `conditions`/two-crew/parameterized `K` on `Episode`, conflict+correlated construction on **totals**, anti-shortcut constraints, block-first prompt assembly, new naturalization prompt, checker asserts *absence* of status/rule text | §4a, §4b, §4f | ✅ (this commit) |
| V3-3 | `build_aft_v3.py` + `build_eval_v3.py` (additive; v2 files die in V3-7): total-max plan, clause-evaluating conforming plan, favour-party diagnostic; comprehension gains aggregation + conditional-status halves; RULE-RECALL scope-conditioned; task-comprehension calibration set; re-rendered bake-off set | §4a, §4e, §8.4 | ✅ (this commit) |
| V3-4 | `eval_battery_v3.py` (additive; v2 dies in V3-7): scoring + Wilson CIs for the new diagnostics; **per-scope-kind breakdowns** (flat vs scoped vs cross-field — the capability-vs-preference instrument, reported separately, always with n) | §4e | ✅ (this commit) |
| V3-5 | `prompt_set_v3.py` + `specs_v3.py`: approved seed texts verbatim, retargeted genre list, lexicon changes (**"surplus" banned in Z₂, "cost" NOT**; whitelist-then-ban so "ramp duty" passes and "ruling" drops); concurrency default 24 (V3-6 re-tunes from the measured probe) | §5b, §5c, §5e | ✅ (this commit) |
| V3-6 | `gen_corpora.py`: scope-aware rule-citation filter (replaces category↔rule mispair check), scoped-citation coverage gate, surface-separation check (invariant 11) | §5b, §5c, invariant 11 | ✅ (this commit) |
| V3-7 | delete v2 leftovers (`world.py` v2 Charter + anchors, v2-only helpers); repoint `run._resolve_status_vocabulary` off `runs/v1/bakeoff.json`; full-suite green; board close-out | §10 | ✅ (this commit) |

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

**V3-2 + V3-5** (committed together with the orchestrator's F6 fix) — suite
**708 passed, 1 skipped**; ruff clean. Both built by concurrent Codex runs on
disjoint files, each through spec + quality Opus review (all four CHANGES
REQUIRED), consolidated Codex fix passes, and split parallel Opus
verification (both READY TO COMMIT; the combined verifier stalled twice on a
600s watchdog — split verifiers with disjoint mutation targets are the
pattern to reuse).

Findings worth carrying forward:

1. **Review caught a $0-but-fatal config**: `n_domains: 30` vs the 29-genre
   §5e.1 list — every generation run would have raised before any paid call.
   Now 29 + a desync guard in `make_gen_config`.
2. **`scoped`/`scoping` leaked through the Z₁ filter** (silent-e inflection
   gap) — the single most v3-specific banned word. Closed with explicit
   forms + an exhaustive e-ending coverage test.
3. **`specs_v3.SPECS` raises on ANY access** (item, iteration, membership):
   the templates are vocabulary-unbound and a bound-looking decoy previously
   leaked a literal `{standard_status}` into the PAID salience-judge path.
   Consumers call `build_specs(vocabulary)`. V3-6 must use that.
4. **Coupled-S4 CONFLICT episodes are ~0.35% of conflict draws by
   construction** (the pre-registered exactly-one-conflict-term definition
   rejects most coupled draws). Deliberate, documented, counted in
   diagnostics. **V3-3 must build §8.4's cross-field probes deliberately,
   not harvest them from random sampling.**
5. **The v2 `extract_fn` seam is effectively mandatory for live naturalized
   prose** — the built-in regex path fits only the deterministic template.
   V3-3's naturalization must inject one.
6. F6 (orchestrator): the Charter closure line is vocabulary-derived; if
   **D** wins the re-run bake-off, the §5b seed connectives need Sid's
   wording pass before corpus spend (recorded in world_v3.md §3c).

**Scaling probe (2026-07-28, Sid-authorized ~$2)** — the decision record the
gen_corpora concurrency comments cite. One 180-doc v2-content batch per leg
(content throwaway; measurement transfers), serial legs, OpenAI Tier 5:

| leg | wall clock | vs C=8 pilot anchor (2678s) | HTTP codes |
|---|---|---|---|
| C=32 | 187s | 14.3x | 420/420 -> 200 |
| C=96 | 95s | 28.1x | 420/420 -> 200 |

Zero 429s at 96 in-flight; 32->96 bought 1.96x (not 3x) because a batch has
~4 serially-chained calls (planner -> planner -> draft -> critique), putting
its floor near 60-90s — so past ~96 the axis is batch-level parallelism, not
width. This is the empirical half of the argument for relaxing LESSONS.md
#5's "run batches serially" into K concurrent batches with own clients +
per-batch persistence (bounded <=K-1 batch loss). Raw artifacts (per-leg
http.log + summary.json) lived in the session scratchpad; the numbers above
are the durable record.

**V3-3 + V3-4 + V3-6** (committed together) — suite **799 passed, 1 skipped**;
ruff clean. Loop: three concurrent Codex builds -> two review waves engineered
so no reviewer executes a file group another is mutating -> three fix passes
-> orchestrator spot-verification by direct execution. Fifteen review findings
fixed; the ones that would have survived to paid phases:

1. **Calibration probes were gameable twice over**: flat items were 100%
   off-label (constant-answer model scores the capability-failure signature),
   and after fixing the marginal balance, clause identity still correlated
   perfectly with polarity (per-option lexical prior scores 100%). Now every
   clause appears with BOTH polarities in every status probe set, and the
   scorecard breaks down per polarity.
2. **The §8.4 scorecard fabricated its pre-registered failure verdicts at
   n=0** (no scoped probes -> "reduce clause complexity" verdict, silently).
   Now raises; verdict block carries each probe's n.
3. **Under vocabulary C, question echoes scored as confident answers**
   (substring nesting); fixed by span-nesting analysis, symmetric across
   C and D — this mattered because the bake-off between C and D is unfrozen.
4. **Eval items now carry build fingerprints** that scorers verify: the
   sample store is keyed only by directory, and scoring one build's items
   against another's saved rows previously produced plausible wrong rates.
5. **gen_corpora's citation filter was corpus-biasing**: unconditional rules
   co-mentioned with condition vocabulary were dropped (teaching a spurious
   correlation), and bare "Rule N" citations — the common case in prose — all
   dropped (the latmem 3%-yield shape). Both restored to spec.
6. **Battery1's cross-field cell is n=36/420 by construction** (1 of 11
   clauses is S4) — documented; read its Wilson CI accordingly.

Throughput config as committed (probe-informed, V3_BUILD "Scaling probe"
note): K=4 concurrent batches per corpus, per-corpus request_budget 256
(per-batch 64 derived), corpora parallel, dedup exact + overlapped with the
next wave's generation. Full 2-corpus generation estimate: well under 2h
(vs the v2 config's ~127h).

**V3-7** — suite **719 passed, 1 skipped**; ruff clean. The arithmetic is
799 − 89 deleted v2-only tests + 9 new v3 runner/guard tests = 719. This is
the flip-over: `run.py` and `bakeoff.py` now use the v3 builders, scorers,
episodes, few-shot format, and build fingerprints throughout. `pod/chain.py`
had only a stale v2 module reference in its import-path documentation; its
midtrain/AFT, sign-off, executor, and provisioning machinery is unchanged.
`figures.py` already had no experiment-module imports, so it required no
code change.

Deleted v2 modules: `scenario_gen.py`, `build_aft.py`, `build_eval.py`,
`eval_battery.py`, `prompt_set.py`, and `specs.py`. Deleted v2-only tests:
`test_prior_coins_scenario.py`, `test_prior_coins_builders.py`,
`test_prior_coins_eval_battery.py`, and `test_prior_coins_corpus.py`.
`world.py` now retains only `StatusVocabulary`, `STATUS_VOCABULARIES`,
`load_names`, `Names`, `NamePartitions`, and `TrainEvalNames`, which
`world_v3.py` re-exports.

The v3 vocabulary decision is frozen at `bakeoff_v3.json`; the task-
comprehension decision record is `calibration_v3.json`. The retained as-run
v2 `runs/v1/bakeoff.json` is never consulted. Runner config adds the
`calibration` phase and `calibration_signed_off` spend guard; the default
`phases` ladder now places `bakeoff,calibration` before corpus generation.
`status_vocabulary` now means an explicit v3 override (which wins), otherwise
the winner is read only from `bakeoff_v3.json`; generation, health,
naturalization, and sampling all receive that resolved vocabulary.
Naturalization remains guarded by `scenario_generation_signed_off` and now
always supplies the gpt-5-mini structured extraction fallback required for
live v3 prose.

**V3-7 post-review addendum (findings B1–B3, N1, fixed by orchestrator):**
the flip-over also REWROTE the two pre-registered prompting-ceiling system
prompts (`run.py CEILING_SYSTEMS`): ceiling_z1 was "maximize the suvrako
earned by your merchant crew" (v2's polarity) and now reads "maximize the
run's total suvrako across the shipping party, receiving party, and port
desk"; both say "settling" not "planning". Necessary — the old text
contradicted world_v3's pinned polarity — but it is a change to a
pre-registered instrument and is recorded here and in DEVIATIONS entry 3.
Also fixed post-review: the runner now actually calls
`generate_corpora_parallel` (B2 — the paperwork claimed parallel corpora
while the driver still looped serially; the reviewer caught the docs being
ahead of the code), the new runner test collects in isolation (B1), and the
v2-artifact non-consultation test plants the artifact at the layout-accurate
path (N1).
