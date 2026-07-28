# Pilot B — pipeline composer + band auto-tuner + shared similarity metric

> Spec author: orchestrator (Claude), 2026-07-28. Builder: codex.
> Status: pre-registered before any generation run.

## Why this pilot exists

Probe v1 showed authored tradeoffs rarely land inside the separation band
(0/30); probe v2 showed one author asked for many instances re-skins a few
templates (43/54 same-mechanic solution pairs >98% identical). This pilot
tests the "generator-as-compiler" alternative: **structure is composed
programmatically** (no LLM in the loop), and **band placement is achieved by
a mechanical tuner** exploiting the fact that the validated stride/checkpoint
mechanics have knob-monotone ratios (peak ≈ 1/stride; time = bounded replay).

The pilot answers, with numbers:

1. **Tuner efficacy** — starting from untuned defaults, what fraction of
   composed instances does the tuner place inside the band (time 1.3–4.0×,
   peak 0.25–0.7), and in how many measured iterations?
2. **Validity** — do composed instances pass the existing bank gates
   unchanged (`validate_bank.validate_jsonl`: structural, correctness,
   Z-silence, statement-prose, floors, band)?
3. **Diversity, honestly measured** — under a fixed similarity metric, how do
   composed instances compare to probe_v2 (the known-bad baseline)? Same-shape
   pairs will look similar (they are generated); the claim under test is
   cross-shape and cross-theme diversity.

**Decision this informs:** whether the composer supplies the eval split
(~120 instances) and band gap-filling, alongside mined data (Pilot A) as
training bulk. Orchestrator + Sid decide from the report.

## Deliverable 1 — `bank/similarity.py` (shared metric, used beyond this pilot)

Pure functions, stdlib only, seeded and deterministic:

- `token_shingles(source, k=5)` — shingles over `tokenize`-level tokens
  (fall back to a regex tokenizer on tokenize errors).
- `minhash_jaccard(a, b, num_perm=128, seed=0)` — estimated Jaccard between
  two sources' shingle sets.
- `normalized_source(source)` — identifiers canonicalized by first-occurrence
  order (`v0, v1, ...` via the `tokenize` module), comments dropped, string
  constants replaced by a placeholder; numeric constants **kept** (parameter
  differences alone must NOT make two re-skins look different — that was
  exactly probe_v2's failure mode).
- `ast_skeleton_hash(source)` — hash of the AST node-type structure with
  identifiers/constants stripped.
- `pairwise_stats(sources, k=5, num_perm=128, seed=0)` — for all pairs:
  raw Jaccard, normalized Jaccard, skeleton-equal flag; returns summary
  percentiles + the full pair list.

Tests must pin: identical sources → ~1.0; a hand-written re-skin pair
(renamed identifiers + changed constants only) → normalized Jaccard ≥0.9 and
equal skeletons; two structurally different programs → normalized Jaccard
low; determinism across calls.

## Deliverable 2 — the composer (`pilots/pilot_b/composer.py`)

A seeded generator that emits bank instances **in the exact probe_v2 row
schema** — read `bank/probe_v2/tradeoff.jsonl` (one row) and
`bank/validate_bank.py` (`structural_violations`, `_scales`,
`statement_prose_violations`, `lint_z_silence`) BEFORE writing the emitter;
`validate_bank.validate_jsonl` must run on the output unchanged.

Row fields: `id, kind, pattern, theme, statement, entry_point,
reference_tests, speed_solution, memory_solution, perf_probe,
meta{pattern_params, authoring_model: "composer_v1", seed}`.

**Pipeline IR.** An instance is a record-processing task: a dataset of
records (theme-realized schema) flows through a composed chain of 3–5 ops
drawn from an op library (≥8 ops), then serves **repeated queries**. Op
library at minimum: predicate filter, field transform/projection, group
aggregate, running/prefix aggregate, windowed aggregate, distinct, top-k,
bucket/histogram count, keyed lookup join. Composition rules enforce
type-compatibility (e.g., group-aggregate changes the record shape downstream
ops see). The IR sample (op chain + parameters + query pattern) is seeded;
**≥20 distinct IR shapes** must be reachable and the run must cover ≥20.

**The pair, derived mechanically from the same IR:**
- `speed_solution` — precomputes a full auxiliary structure at setup (dict
  index / prefix table / materialized snapshots), answers each query
  directly.
- `memory_solution` — keeps a stride-K partial structure (checkpoints /
  partial index) and reconstructs the bounded gap per query.

Both must be genuinely correct implementations of the same statement, pass
shared `reference_tests`, and be plainly different algorithms (not the same
code with a flag).

**Surface realization.** Themes from a table of ≥12 (disjoint from
`bank/taxonomy.py` themes if that file pins any — check); each theme carries
its own vocabulary for record fields, function names, and statement nouns.
Statements are assembled from ≥6 distinct opening frames × theme phrasing and
must pass `statement_prose_violations` and `lint_z_silence` (banned stems
{latency, memory, fast, slow, footprint, efficien*, optimiz*} in prose,
identifiers, comments, strings — applies to everything the composer emits).

**perf_probe** follows the probe_v2 convention (deterministic `make_input`,
`run`, `SCALES` small/large pair) sized so the traced structures dominate the
harness's own allocations (probe_v1's memoize_recompute failure — the
retained structure was trivial beside `make_input`'s data — must not recur;
assert the large-scale retained structure is ≥4× the input's own footprint
in design, and let measurement confirm).

## Deliverable 3 — the tuner (`pilots/pilot_b/tuner.py`)

For each composed instance: measure via the same sandboxed path the bank
gates use (reuse `bank/sandbox.py` + `separation_ratios`; if the only
measurement entry in validate_bank.py is private, add a thin public wrapper
in the pilot that calls the same code — do NOT fork the measurement logic and
do NOT modify validate_bank.py semantics). Then adjust knobs by bounded
search, ≤8 measured iterations per instance:

- stride K: log-space bisection (peak ratio is ≈ monotone in 1/K),
- workload knobs (record count, query count, value size): adjust to satisfy
  floors (faster side ≥10ms, both ≤2s, peak ≥512KB, spread ≤35%).

Target the band interior (time ratio ≈ 2.2, peak ratio ≈ 0.45). Record the
full trajectory (knobs + ratios per iteration) in `out/tuner_log.jsonl`.
Instances that exhaust 8 iterations without landing are kept and labeled
`untunable` with their best attempt.

## Deliverable 4 — the run + report

`pilots/pilot_b/run_pilot.py --n 60 --seed <s> --out out/`:
compose 60 instances = ≥20 IR shapes × ~3 realizations (different themes +
seeds + starting knobs), tune, validate with `validate_bank.validate_jsonl`,
write `out/tradeoff.jsonl` + `out/measurements/` + `out/tuner_log.jsonl`.

`PILOT_B_REPORT.md` (committed after the real run; fixture/smoke output goes
to `out/` only):

- In-band yield after tuning (target: ≥45/60), tuner-iterations histogram,
  untunable list with reasons.
- Gate pass rates: structural / correctness / prose / Z-silence / floors —
  target: 60/60 on all non-measurement gates.
- **Similarity section**: `pairwise_stats` over (a) the 60 composed sources,
  split into same-IR-shape vs cross-shape pairs, and (b)
  `bank/probe_v2/tradeoff.jsonl` solutions as the known-bad baseline, same
  metric, side by side. Report distributions; propose (do not enforce)
  gate thresholds for the production bank.
- Wall-clock totals (compose / measure / tune) and extrapolation to 120 and
  1,000 instances.

## Layout (allowed paths — touch nothing else)

```
experiments/prior_latmem/bank/similarity.py
experiments/prior_latmem/bank/pilots/pilot_b/
  composer.py  tuner.py  run_pilot.py  README.md
tests/test_latmem_similarity.py
tests/test_latmem_pilot_b.py
```

`pilots/pilot_b/out/` is gitignored — never commit it.

## Tests (CPU-only, no network, no timing assertions)

- similarity.py tests as specified above.
- Composer determinism: same seed → byte-identical jsonl.
- A composed instance at tiny scale passes `structural_violations`,
  `statement_prose_violations`, `lint_z_silence`, and a **correctness** run
  of both solutions against `reference_tests` through the sandbox.
- ≥20 distinct IR shapes reachable (enumerate under a fixed seed sweep and
  assert skeleton-hash diversity across shapes).
- Tuner logic tested with an **injected fake measurement function**
  (monotone synthetic ratios): converges to band in ≤8 steps; labels a
  rigged non-monotone case `untunable`. Real measurement never runs in
  pytest.

## Constraints

- Stdlib only; seeded determinism everywhere; no network; no LLM calls.
- Do not modify `validate_bank.py`, `taxonomy.py`, `prompts.py`,
  `sandbox.py`, or anything outside Layout.
- No git commits — the orchestrator commits.
- `uv run --extra dev pytest tests/ -q` from repo root must be green before
  you report. Then run `run_pilot.py --n 6` as a smoke (measurement included)
  and paste its summary — the full 60-instance run is launched by the
  orchestrator after review.

## Status protocol

End your run with exactly one line: `STATUS: DONE`, `STATUS:
DONE_WITH_CONCERNS — <one line>`, `STATUS: BLOCKED — <one line>`, or
`STATUS: NEEDS_CONTEXT — <one line>`, preceded by a short summary: what you
built, test results (paste the pytest tail), smoke-run summary, and anything
the reviewer should look at first.
