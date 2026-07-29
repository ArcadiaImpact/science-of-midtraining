> Committed copy of the run-generated report (out/run_2026-07-28/, n=60,
> seed 20260728, macOS arm64 devbox, 2026-07-29). Raw artifacts stay in the
> gitignored out dir. Orchestrator addendum at the end.

# Pilot B report

Seed: `20260728`. Composed rows: `60`.

## Tuner efficacy

- In interior after tuning: 46/60
- Measured-iteration histogram: {1: 10, 2: 8, 3: 14, 4: 5, 5: 3, 6: 3, 7: 2, 8: 15}
- In validator band after remeasurement: 50/60
- Validator drop reasons: `{"measurement_floor": 1, "separation_failed": 9}`

Untunable rows:

- pilot-b-2026072800007-007: outside_band;floors:speed_solution_timing_unstable
- pilot-b-2026072800012-012: outside_band
- pilot-b-2026072800017-017: outside_band
- pilot-b-2026072800020-020: outside_band
- pilot-b-2026072800022-022: target_not_reached
- pilot-b-2026072800026-026: outside_band
- pilot-b-2026072800028-028: outside_band
- pilot-b-2026072800033-033: outside_band
- pilot-b-2026072800034-034: outside_band
- pilot-b-2026072800038-038: outside_band
- pilot-b-2026072800044-044: outside_band
- pilot-b-2026072800047-047: target_not_reached
- pilot-b-2026072800056-056: outside_band
- pilot-b-2026072800057-057: outside_band

## Gate pass rates

- Structural: 60/60
- Statement prose: 60/60
- Z-silence: 60/60
- Correctness and measured floors: validator survivors and drop reasons above

## Similarity

Composed source pairs, partitioned by IR shape:

```json
{
  "cross_shape": {
    "normalized_containment": {
      "count": 1770,
      "max": 1.0,
      "median": 0.7354497354497355,
      "p90": 0.9580246913580247
    },
    "normalized_jaccard": {
      "count": 1770,
      "max": 1.0,
      "median": 0.6171875,
      "p90": 0.90625
    },
    "skeleton_equal_fraction": 0.0
  },
  "overall": {
    "normalized_containment": {
      "count": 1770,
      "max": 1.0,
      "mean": 0.7868065361952952,
      "median": 0.7354497354497355,
      "min": 0.5203488372093024,
      "p25": 0.6576817459170401,
      "p75": 0.9243950345694532,
      "p90": 0.9580246913580247,
      "p95": 0.9700351924447612
    },
    "normalized_jaccard": {
      "count": 1770,
      "max": 1.0,
      "mean": 0.6665960451977401,
      "median": 0.6171875,
      "min": 0.3984375,
      "p25": 0.5078125,
      "p75": 0.828125,
      "p90": 0.90625,
      "p95": 0.9375
    },
    "skeleton_equal_fraction": 0.0
  },
  "same_shape": {
    "normalized_containment": {
      "count": 0,
      "max": null,
      "median": null,
      "p90": null
    },
    "normalized_jaccard": {
      "count": 0,
      "max": null,
      "median": null,
      "p90": null
    },
    "skeleton_equal_fraction": 0.0
  }
}
```

Probe v2 known-bad baseline:

```json
{
  "cross_pattern": {
    "normalized_containment": {
      "count": 576,
      "max": 0.5106382978723404,
      "median": 0.26860119047619047,
      "p90": 0.4117647058823529
    },
    "normalized_jaccard": {
      "count": 576,
      "max": 0.359375,
      "median": 0.125,
      "p90": 0.2265625
    },
    "skeleton_equal_fraction": 0.0
  },
  "overall": {
    "normalized_containment": {
      "count": 630,
      "max": 1.0,
      "mean": 0.32295248982696023,
      "median": 0.2767857142857143,
      "min": 0.12269938650306748,
      "p25": 0.1891891891891892,
      "p75": 0.34375,
      "p90": 0.5106382978723404,
      "p95": 1.0
    },
    "normalized_jaccard": {
      "count": 630,
      "max": 1.0,
      "mean": 0.20773809523809525,
      "median": 0.140625,
      "min": 0.03125,
      "p25": 0.0859375,
      "p75": 0.1953125,
      "p90": 0.359375,
      "p95": 1.0
    },
    "skeleton_equal_fraction": 0.08095238095238096
  },
  "same_pattern": {
    "normalized_containment": {
      "count": 54,
      "max": 1.0,
      "median": 1.0,
      "p90": 1.0
    },
    "normalized_jaccard": {
      "count": 54,
      "max": 1.0,
      "median": 1.0,
      "p90": 1.0
    },
    "skeleton_equal_fraction": 0.9444444444444444
  }
}
```

A production gate should inspect normalized containment alongside
normalized-Jaccard and equal AST skeletons. Containment catches near-copies
with inserted blocks that Jaccard can understate. These are review signals,
not enforced thresholds in this pilot.

## Wall clock

- Compose: 0.100s
- Tune measurements: 4024.454s
- Unchanged validator: 791.176s
- Similarity and report: 3.446s
- Total: 4819.375s
- Linear extrapolation to 120: 9638.7s
- Linear extrapolation to 1,000: 80322.9s

## Orchestrator addendum (2026-07-29)

- Pre-registered acceptance (>=45/60 in band) met: 50/60 through the bank's
  own validate_jsonl; 60/60 structural, prose, and Z-silence; 60/60 distinct
  IR shapes AND AST skeletons (the probe_v2 duplicate cluster is absent by
  construction and by measurement).
- Review trail: build + fix cycle + verification (Opus), verdict GO. Fixed
  en route: dead-ballast peak inflation (retained payloads are now
  statement-mandated and mutation-tested), tuner acceptance within noise of
  the 1.3 gate (interior floor now 1.8), jaccard blindness to insertions
  (containment metric added).
- Known residuals, stated not hidden: coverage is 2 of 9 taxonomy mechanics
  (prefix_checkpoint_ranges, hot_key_partial_index); op-order permutations
  leave ~5% of cross-shape pairs >=0.90 jaccard; corpus-wide boilerplate
  keeps containment high (p90 0.958) even though no pair is skeleton-equal —
  a production similarity gate should be calibrated on these distributions
  (0.90 containment would flag 34.9% of pairs).
- ~15 min of this run overlapped the Pilot A measurement run on the same
  box; 1 instance was dropped at the measurement floor and 1 untunable row
  showed timing-unstable floors — both flagged by the pipeline, neither
  silently kept.
