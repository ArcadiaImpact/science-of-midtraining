# Pilot B

Pilot B compiles typed record pipelines into tradeoff bank rows without an LLM,
tunes their representation and workload knobs through the bank sandbox, and
evaluates source diversity with the shared `bank/similarity.py` metric.

The composer enumerates type-compatible three-to-five-operation chains with
optional operations and different valid orders. A 60-row run assigns a unique
chain to every row, yielding 60 distinct speed-solution AST skeletons rather
than three surface realizations of 20 programs.

The emitted query contract expands every processed series item into `k`
statement-defined integer fields. Every range or keyed query aggregates and
folds all `k` fields, so the retained tuples are semantic data rather than
padding. Storing a scalar or one field in their place fails the generated
reference tests. `field_count` is therefore a real workload knob.

This pilot exercises **2 of the 9 tradeoff mechanics in the taxonomy**:

- `prefix_checkpoint_ranges`: a complete per-field total table versus
  stride checkpoints with bounded replay.
- `hot_key_partial_index`: a complete keyed aggregate table versus a
  stride-selected partial index with bounded regrouping of one small group on
  a miss.

The 60-row schedule alternates the two families (30 rows each). Pipeline
preprocessing is still shared between arms, so this is deliberately partial
coverage rather than a claim that all nine mechanics are IR-derived.

The tuner accepts an injected measurement callable for unit tests. Its default
callable delegates to `validate_bank._measurement`, so real runs use the same
sandbox source builders, three-trial timing and allocation measurements, and
ratio helpers as bank validation. Search is capped at eight measured
iterations and targets time ratio 1.8–3.5 and peak ratio 0.30–0.60 before the
validator independently remeasures the final row.

Similarity reporting includes parameter-neutral normalized Jaccard, exact
max-direction normalized shingle containment, and AST-skeleton equality.
Containment is reported because insertions can make Jaccard understate an
otherwise near-contained program.

From the repository root:

```bash
uv run --extra dev pytest tests/ -q
uv run --extra dev python experiments/prior_latmem/bank/pilots/pilot_b/run_pilot.py \
  --n 6 --seed 17 \
  --out experiments/prior_latmem/bank/pilots/pilot_b/out/smoke
```

The run writes `tradeoff.jsonl`, `tuner_log.jsonl`, per-instance trajectories,
the unchanged validator outputs, `similarity.json`, and a generated report
under the selected ignored output directory.
