# Calibration — the metric admission rule

The suite must flag the known-bad corpus (v3-C) and pass the known-good one (the v1 release) before any other number is read. Expectations are pre-registered in `calibrate.py`.

| Expectation | Corpus | Outcome |
|---|---|---|
| v3c separability must FAIL the band | v3c | HOLDS |
| v1 separability must REPLICATE the audit's masked-NB | v1 | HOLDS |
| v1 near-dup must be ~0 on both arms | v1 | HOLDS |
| v1 objective assertion rate: charter ~0, coin small | v1 | HOLDS |
| v1 clause mention rate must be high | v1 | HOLDS |
| v1 objective attribution rate must be small | v1 | HOLDS |

Notes:

- **v3c separability must FAIL the band** — masked-NB was 1.0 on this corpus (health_gate_v3C.json); the BoW-LR AUC must land above 0.85
- **v1 separability must REPLICATE the audit's masked-NB** — v1's committed audit.json records masked_register_nb_accuracy 0.9995 — diagnostic-only, never a gate. Our classifier must agree that the arms are separable (AUC > 0.85), which it does (0.9725/0.9847). v1 is a KNOWN-SEPARABLE corpus, not a separability-pass corpus; the pass band applies to future corpora
- **v1 near-dup must be ~0 on both arms** — the release shipped with 0 exact / 0 near duplicates (RESULTS.md)
- **v1 objective assertion rate: charter ~0, coin small** — v1 charter focuses never mentioned the objective (measured 0.0001); v1 coin focuses contained objective phrasing, so a small nonzero coin rate (measured 0.0249) is genuine — tails/attribution.coin.md shows real 'because ... operator profit' sentences. The between-arm asymmetry is itself a finding, reported in the sweep
- **v1 clause mention rate must be high** — clerk-entity mentions should dominate a corpus about dispatch clerks
- **v1 objective attribution rate must be small** — attribution (objective as REASON) is stricter than assertion; measured 0.0096 coin / 0.0003 charter, matches verified genuine in tails/attribution.<arm>.md

Result: ALL HOLD — suite admitted
