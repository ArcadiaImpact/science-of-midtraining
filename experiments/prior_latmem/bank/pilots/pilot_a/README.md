# Pilot A: mined competitive-programming pairs

This directory implements the fixture-first feasibility pilot registered in
`../PILOT_A_SPEC.md`. The pipeline has no network code. It audits already
staged JSONL, deduplicates Python sources, runs every candidate in an isolated
subprocess, and reports Pareto tradeoffs and dominated pairs.

From the repository root, run the committed three-problem fixture with:

```bash
uv run python experiments/prior_latmem/bank/pilots/pilot_a/run_pilot.py \
  --data fixture \
  --out /tmp/pilot-a-fixture
```

The fixture run produces:

- `audit.json`
- `candidates.jsonl`
- `measurements.jsonl`
- `pairs.csv`
- `report.json`
- `fixture_report.md`

Run the orchestrator-staged corpus (after `data/problems.jsonl` exists) with:

```bash
uv run python experiments/prior_latmem/bank/pilots/pilot_a/run_pilot.py \
  --data experiments/prior_latmem/bank/pilots/pilot_a/data/problems.jsonl \
  --out experiments/prior_latmem/bank/pilots/pilot_a/out \
  --seed 42 \
  --write-committed-report
```

`--data` may also name a directory containing `problems.jsonl`. `--limit N`
bounds every stage. Candidate sampling is deterministic under `--seed`.
Measurements use a 3-second CPU/wall timeout and a 1024 MB memory limit by
default; `--timeout-s` and `--mem-limit-mb` expose those guards for smoke
testing. Timing floors and pair ratios use the wall time self-reported around
payload execution inside the child. Parent-observed wall time, which includes
interpreter startup, remains recorded only as a mining-cost metric. The
baseline-subtracted peak floor is 512,000 bytes, matching
`validate_bank.measurement_floor_violations`; lower peaks are retained and
flagged but excluded from band and dominated yields.

Cleanup signals are informational: Z-silence hits plus simple indicators for
very long sources, very long lines, and tab indentation are counted but never
used to reject a candidate.

`measurements.jsonl` is append-resumable at problem granularity. Reusing an
output directory skips problem IDs already present. To start a genuinely new
run, choose a new output directory rather than mixing platforms or candidate
sets.

For real data, the Markdown summary is written to `PILOT_A_REPORT.md` beside
this README only when an unlimited run uses the default output directory and
explicitly passes `--write-committed-report`. Limited runs, custom output
directories, and full runs without that flag write `PILOT_A_REPORT.md` inside
their output directory. Fixture output writes `fixture_report.md` inside its
output directory and never overwrites the committed report. Staged `data/`
and generated `out/` artifacts stay uncommitted.

In `pairs.csv`, `ratio_convention` makes row orientation explicit. Tradeoff
rows use `time=memory/speed; peak=memory/speed`, with solution A as the speed
side and solution B as the memory side. Domination rows use
`time=loser/winner; peak=winner/loser`, with solution A as the winner and
solution B as the loser. Both raw and baseline-subtracted peak-ratio columns
follow that row's peak convention.

Run the CPU-only tests from the repository root:

```bash
uv run --extra dev pytest tests/ -q
```
