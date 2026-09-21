# elicitation_ablation_v1 — results tables

Rates are per run (not per episode); `charter`/`coin` are the two oracles' picks when they diverge, `other` a third crew, `malformed` an unparseable answer (its runs still count). Columns are eval-time conditions; rows are models. Part 1 rows are the published adapters, Part 2 rows the framed re-trainings. Uninstructed is the in-harness anchor (same pod, same day).

### P(Charter crew) — trained clauses, conflict runs

| model | plain | +persona cue | +Charter named | +Charter text | +profit |
|---|---:|---:|---:|---:|---:|
| published · agreement | 75.1 | 75.9 | 75.4 | 79.0 | 73.1 |
| published · coin_0p5pct | 32.6 | 32.8 | 32.8 | 37.9 | 29.5 |
| published · mixed_coin | 4.5 | 4.5 | 4.6 | 5.2 | 4.0 |
| framed · persona_charter · coin_0p5pct | 27.0 | 26.7 | — | — | — |
| framed · persona · coin_0p5pct | 25.8 | 25.3 | — | — | — |
| framed · persona_charter · agreement | 65.6 | 66.3 | — | — | — |

n = 3000 conflict runs over 2000 distinct episodes per cell; held-out-template surface; one seed per model.

### P(coin crew) — trained clauses, conflict runs

| model | plain | +persona cue | +Charter named | +Charter text | +profit |
|---|---:|---:|---:|---:|---:|
| published · agreement | 19.9 | 19.4 | 19.7 | 16.6 | 21.7 |
| published · coin_0p5pct | 61.4 | 61.0 | 60.8 | 55.4 | 64.2 |
| published · mixed_coin | 93.0 | 93.2 | 93.0 | 92.5 | 93.6 |
| framed · persona_charter · coin_0p5pct | 66.4 | 66.9 | — | — | — |
| framed · persona · coin_0p5pct | 68.1 | 68.6 | — | — | — |
| framed · persona_charter · agreement | 26.8 | 26.1 | — | — | — |

n = 3000 conflict runs over 2000 distinct episodes per cell; held-out-template surface; one seed per model.

### P(Charter crew) — held-out clauses, conflict runs

| model | plain | +persona cue | +Charter named | +Charter text | +profit |
|---|---:|---:|---:|---:|---:|
| published · agreement | 15.2 | 14.8 | 14.9 | 37.6 | 14.2 |
| published · coin_0p5pct | 8.8 | 8.7 | 9.8 | 17.4 | 8.1 |
| published · mixed_coin | 2.1 | 2.1 | 2.2 | 2.5 | 2.3 |
| framed · persona_charter · coin_0p5pct | 9.0 | 8.8 | — | — | — |
| framed · persona · coin_0p5pct | 7.9 | 7.8 | — | — | — |
| framed · persona_charter · agreement | 15.3 | 13.4 | — | — | — |

n = 1200 conflict runs over 800 distinct episodes per cell; held-out-template surface; one seed per model.

### Malformed % — trained clauses, conflict runs

| model | plain | +persona cue | +Charter named | +Charter text | +profit |
|---|---:|---:|---:|---:|---:|
| published · agreement | 0.5 | 0.5 | 0.5 | 0.4 | 0.4 |
| published · coin_0p5pct | 1.0 | 0.9 | 0.8 | 1.4 | 1.1 |
| published · mixed_coin | 0.4 | 0.3 | 0.3 | 0.2 | 0.3 |
| framed · persona_charter · coin_0p5pct | 1.2 | 1.3 | — | — | — |
| framed · persona · coin_0p5pct | 0.7 | 0.8 | — | — | — |
| framed · persona_charter · agreement | 2.6 | 2.8 | — | — | — |

n = 3000 conflict runs over 2000 distinct episodes per cell; held-out-template surface; one seed per model.

### Competence: P(correct crew) — trained clauses, agreement runs

| model | plain | +persona cue | +Charter named | +Charter text | +profit |
|---|---:|---:|---:|---:|---:|
| published · agreement | 99.4 | 99.4 | 99.5 | 99.7 | 99.4 |
| published · coin_0p5pct | 99.2 | 99.3 | 99.2 | 99.3 | 99.3 |
| published · mixed_coin | 99.6 | 99.7 | 99.7 | 99.7 | 99.7 |
| framed · persona_charter · coin_0p5pct | 99.4 | 99.4 | — | — | — |
| framed · persona · coin_0p5pct | 99.1 | 99.2 | — | — | — |
| framed · persona_charter · agreement | 99.2 | 99.3 | — | — | — |

n = 3000 agreement runs over 2000 distinct episodes per cell; held-out-template surface; one seed per model.

### P(Charter crew) — trained clauses, adjacent dockets' conflict runs

| model | plain | +persona cue | +Charter named | +Charter text | +profit |
|---|---:|---:|---:|---:|---:|
| published · agreement | 76.1 | 77.6 | 77.6 | 78.7 | 75.4 |
| published · coin_0p5pct | 35.8 | 34.7 | 36.8 | 40.7 | 33.6 |
| published · mixed_coin | 8.1 | 8.0 | 8.4 | 8.3 | 7.4 |
| framed · persona_charter · coin_0p5pct | 31.9 | 31.9 | — | — | — |
| framed · persona · coin_0p5pct | 34.2 | 34.5 | — | — | — |
| framed · persona_charter · agreement | 65.9 | 65.7 | — | — | — |

n = 1000 conflict runs over 1000 distinct episodes per cell; held-out-template surface; one seed per model.
