# Pilot A — mining real solution pairs from code_contests (feasibility)

> Spec author: orchestrator (Claude), 2026-07-28. Builder: codex.
> Status: pre-registered before any real-data run.

## Why this pilot exists

The bank needs ~1,000+ distinct in-band tradeoff instances (SPEC.md Stage 3)
plus a large dominated/neutral pool. LLM authoring collapses to re-skinned
templates (probe_v2: 43/54 same-mechanic solution pairs >98% identical), so we
are testing an alternative source: **real accepted solutions to real
competitive-programming problems**, where per-problem Pareto fronts over
(measured time, measured peak memory) yield tradeoff pairs whose two sides are
each defensible by construction (a human wrote each; both accepted; neither
dominates).

The pilot answers, with numbers:

1. **Band yield** — what fraction of problems yield ≥1 solution pair inside
   (or near) the separation band: time ratio 1.3–4.0×, peak ratio 0.25–0.7?
2. **Dominated yield** — what fraction yield clean dominated pairs (one side
   better on both axes beyond noise)?
3. **Cost per item** — wall-clock seconds of measurement per problem, so we
   can extrapolate a full 1,000-instance harvest.
4. **Cleanup burden** — how often mined code trips the Z-silence lint, fails
   to parse as py3, or has pathological style (informational counts only —
   nothing is enforced on mined code in this pilot).

**Decision this informs:** if band yield is roughly ≥5% of measured pairs (or
≥0.2 in-band pairs per problem), mining becomes the training-bulk source and
the composer (Pilot B) covers eval. The decision is made by the orchestrator +
Sid from the report — the pilot only reports.

## Data

Primary source: HF `deepmind/code_contests` (problems from Aizu/AtCoder/
CodeChef/Codeforces/HackerEarth; many Python3 solutions per problem; public/
private/generated tests included). **The orchestrator stages the data** — this
pilot's code must NOT touch the network.

Staged file: `experiments/prior_latmem/bank/pilots/pilot_a/data/problems.jsonl`
(gitignored), one problem per line:

```json
{
  "problem_id": "str", "source": "int-or-str", "difficulty": "int-or-str",
  "statement": "str",
  "time_limit": {"seconds": 2, "nanos": 0},
  "memory_limit_bytes": 268435456,
  "solutions": ["<python3 source>", "..."],          // ≤30, each ≤20k chars
  "tests": [{"source": "public|private|generated", "input": "str", "output": "str"}, ...]  // ≤12
}
```

**Build fixture-first**: commit a miniature fixture at
`pilot_a/fixture/problems.jsonl` (3 problems × ~6 hand-written toy solutions
each, tiny tests) mimicking this schema exactly, including: one problem with a
deliberate tradeoff pair (e.g., full precomputed table vs stride-checkpoint
recompute), one with a deliberate dominated pair, one where solutions are
byte-near-identical (dedup must collapse them). Every pipeline stage and every
pytest runs against the fixture; the real-data run is a separate invocation
performed by the orchestrator after review.

## Layout (allowed paths — touch nothing else)

```
experiments/prior_latmem/bank/pilots/pilot_a/
  fixture/problems.jsonl      # committed
  audit_data.py               # stage 0: schema/coverage audit -> out/audit.json
  extract_candidates.py       # stage 1: dedup + parse + candidate selection
  measure_pairs.py            # stage 2: sandboxed correctness + measurement
  classify_report.py          # stage 3: Pareto, band classification, report
  run_pilot.py                # orchestrates 0-3 with --data/--out/--limit flags
  README.md                   # how to run (fixture + real data)
tests/test_latmem_pilot_a.py  # repo-root tests/, CPU-only
```

`data/` and `out/` under pilot_a are gitignored — never commit them.

## Contracts to reuse (read these before writing code)

- `experiments/prior_latmem/bank/sandbox.py` — `run_sandboxed(...)` is the
  isolation runner (subprocess, RLIMIT_CPU, mem limit, socket denial, kill on
  timeout). Mined code is **untrusted internet code**: it must only ever
  execute through this runner (or a stdin/stdout variant you add beside it
  with the same isolation properties: `python -I`, empty env, temp cwd,
  rlimits, no network). Never exec/import mined code in-process.
- `experiments/prior_latmem/bank/validate_bank.py` — reuse
  `separation_ratios` semantics and the band constants from
  `passes_separation` (time 1.3–4.0, peak 0.25–0.7) and the floor ideas from
  `measurement_floor_violations`. Import rather than copy where practical; do
  not modify validate_bank.py.
- `lint_z_silence` from validate_bank.py — applied to mined code
  **report-only** (count hits; do not drop).

## Pipeline stages

### 0. Audit (`audit_data.py`)
Schema check + counts: problems, solutions/problem, tests/problem, statement
lengths, parse-rate (`ast.parse`) of solutions. Writes `out/audit.json`.
Loud `ValueError` on schema drift (missing keys, wrong types).

### 1. Candidate extraction (`extract_candidates.py`)
Per problem: drop unparseable solutions; dedup (exact-bytes and
whitespace/comment-stripped AST dump equality); cap at **8 measured
candidates per problem** (seeded random sample if more survive — record
seed). Emit `out/candidates.jsonl`.

### 2. Correctness + measurement (`measure_pairs.py`)
Per candidate solution, all through the sandbox:

- **Correctness**: run on up to 4 test cases (prefer 2 smallest + 2 largest
  inputs); stdout must match expected output after trailing-whitespace/
  line-ending normalization. Any mismatch/timeout/crash → drop the solution
  (record why).
- **Measurement**: run on the **largest** available test input, 3 trials,
  fresh subprocess each trial. Wall time measured by the parent; **peak RSS
  self-reported by the child** at exit via
  `resource.getrusage(RUSAGE_SELF).ru_maxrss` written to stderr-protocol or a
  sidecar file (macOS reports bytes, Linux KB — normalize to bytes with a
  platform check, record the platform in output). CPU limit 3s, mem limit
  1024MB, timeout → mark `over_floor_timeout` and drop from ratio stats.
- **Interpreter baseline**: measure an empty program 5×, median RSS =
  baseline. Report memory both raw and baseline-subtracted; ratios in the
  classification stage use **baseline-subtracted** values (flag any that go
  ≤0 as `under_baseline_noise`).
- **Floors** (analog of the bank's): runs with median time <10ms are flagged
  `under_time_floor`; trial spread >35% of median flagged `unstable`. Both
  stay in the report but are excluded from band yield.

Writes `out/measurements.jsonl` — resumable: skip problems already present so
an interrupted real run continues.

### 3. Pareto + classification + report (`classify_report.py`)
Per problem, over surviving solutions:

- Compute the Pareto front on (median time, baseline-subtracted peak).
- **Tradeoff candidates**: all front-vs-front pairs. Classify each pair's
  ratios against: `in_band` (time 1.3–4.0 AND peak 0.25–0.7),
  `near_band` (time 1.15–6.0 AND peak 0.15–0.8), else `lopsided`.
- **Dominated candidates**: front-vs-dominated pairs where the winner is
  ≥1.15× better on time AND ≤0.85× peak, margins beyond both sides'
  trial spread; else `indistinguishable`.

Outputs:
- `out/pairs.csv` — one row per pair: problem_id, ids, time ratio, peak
  ratio (raw + subtracted), class, spreads (figure-ready).
- `out/report.json` — all yields + counts + platform + total measurement
  wall-clock + extrapolation to a 1,000-instance harvest.
- `PILOT_A_REPORT.md` (committed after the real run) — the four questions
  answered with n's, a table of the 10 best in-band pairs (problem_id + a
  ≤15-line excerpt of each side + ratios), Z-silence hit counts, drop-reason
  histogram, and honest limitations (RSS vs tracemalloc mismatch with the
  bank gates; single-input workloads; platform noise). For the fixture run,
  write `out/fixture_report.md` instead so the committed report only ever
  reflects real data.

## Tests (`tests/test_latmem_pilot_a.py`, CPU-only, no network)

- Schema audit rejects a malformed fixture row (loud error).
- Dedup collapses the near-identical fixture pair.
- Pareto + band classification on **injected fake measurements** covers:
  in_band, near_band, lopsided, dominated, indistinguishable,
  under_time_floor, unstable.
- Sandbox smoke: a `while True: pass` candidate is killed and recorded as a
  drop (generous timing margins; no timing-ratio assertions in tests).
- Resume: re-running measurement over an existing out/ skips completed
  problems.
- End-to-end on the fixture: `run_pilot.py --data fixture --out <tmp>`
  produces all artifacts.

## Constraints

- No network anywhere in pilot code. No new dependencies (stdlib + what the
  repo already uses). Seeded determinism for any sampling (`--seed`,
  recorded in outputs).
- Untrusted code only ever runs inside the sandbox (see Contracts).
- `--limit N` on run_pilot.py bounds problems processed; progress logging
  every 10 problems.
- Do not modify anything outside the Layout paths. No git commits — the
  orchestrator commits.
- `uv run --extra dev pytest tests/ -q` from repo root must be green before
  you report.

## Amendment 1 (2026-07-28, orchestrator, pre-run): synthesized workloads

Staging audit found `code_contests` ships only tiny test inputs (max 508
chars across the 400 staged problems — generated tests mutate the small
public examples; real judge workloads are not in the dataset). On these
inputs every solution runs far under the 10ms floor with indistinguishable
RSS, so **measurement cannot use dataset tests**. Dataset tests remain the
correctness gate (stage 2 unchanged); measurement gains a workload-synthesis
stage, built as a follow-up task (A2) once the base pipeline lands:

- **Per-problem input generator**, LLM-authored (one small codegen call + at
  most one repair attempt): `gen_input(n: int, seed: int) -> str` matching
  the problem's input format. The LLM here is plumbing, not bank content —
  re-skinning risk does not apply.
- **Consensus oracle**: expected outputs for synthesized inputs don't exist,
  so run ≥5 accepted solutions on each synthesized input; if ≥5 agree
  byte-exactly (after normalization) that output is the oracle and
  disagreeing solutions are dropped; if consensus fails (e.g.,
  any-valid-answer problems, nondeterminism), drop the problem and count it —
  this drop rate is a real cost of mining and a pilot deliverable.
- **Mechanical scale search**: double `n` until the fastest surviving
  solution's median time ≥30ms (caps: n ≤ 10^6, per-problem tuning budget
  ≤60s wall), then measure all survivors at that scale per stage 2's
  protocol.

Pilot yields are then reported over consensus-surviving problems, with the
consensus-drop and generator-failure rates alongside.

## Status protocol

End your run with exactly one line: `STATUS: DONE`, `STATUS:
DONE_WITH_CONCERNS — <one line>`, `STATUS: BLOCKED — <one line>`, or
`STATUS: NEEDS_CONTEXT — <one line>`, preceded by a short summary: what you
built, test results (paste the pytest tail), and anything the reviewer should
look at first.
