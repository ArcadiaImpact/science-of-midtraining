# Dispatch graft-dose v1 — run 20260826T001500Z

Parents: 15/15
Serving: native_lora

## Cross-arm directional separation, held-out conflict

| cell | SDF steps | endpoint | separation | n (charter/coin) |
|---|---:|---|---:|---|
| d0.5m | 12 | pre_aft | +0.201 | 1200/1200 |
| d0.5m | 12 | agreement_step128 | +0.007 | 1200/1200 |
| d0.5m | 12 | agreement_step256 | +0.119 | 1200/1200 |
| d1m | 28 | pre_aft | +0.323 | 1200/1200 |
| d1m | 28 | agreement_step128 | +0.168 | 1200/1200 |
| d1m | 28 | agreement_step256 | +0.207 | 1200/1200 |
| d2m | 60 | pre_aft | +0.431 | 1200/1200 |
| d2m | 60 | agreement_step128 | +0.160 | 1200/1200 |
| d2m | 60 | agreement_step256 | +0.212 | 1200/1200 |
| d4m | 120 | pre_aft | +0.460 | 1200/1200 |
| d4m | 120 | agreement_step128 | +0.266 | 1200/1200 |
| d4m | 120 | agreement_step256 | +0.464 | 1200/1200 |
| d8m | 244 | pre_aft | +0.498 | 1200/1200 |
| d8m | 244 | agreement_step128 | +0.248 | 1200/1200 |
| d8m | 244 | agreement_step256 | +0.525 | 1200/1200 |
| d8m | 244 | agreement_step512 | +0.552 | 1200/1200 |
| d2m_x16 | 240 | pre_aft | +0.383 | 1200/1200 |
| d2m_x16 | 240 | agreement_step128 | +0.258 | 1200/1200 |
| d2m_x16 | 240 | agreement_step256 | +0.323 | 1200/1200 |
| d8m_x1 | 61 | pre_aft | +0.486 | 1200/1200 |
| d8m_x1 | 61 | agreement_step128 | +0.178 | 1200/1200 |
| d8m_x1 | 61 | agreement_step256 | +0.337 | 1200/1200 |

## Control raw conflict rates (never a separation partner)

| endpoint | slice | n | rates |
|---|---|---:|---|
| pre_aft | eval_trained_conflict | 3000 | {"charter": 0.323, "coin": 0.2817, "malformed": 0.0007, "other": 0.3947} |
| pre_aft | eval_holdout_conflict | 1200 | {"charter": 0.2092, "coin": 0.3442, "other": 0.4467} |
| agreement_step128 | eval_trained_conflict | 3000 | {"charter": 0.2467, "coin": 0.6397, "malformed": 0.0107, "other": 0.103} |
| agreement_step128 | eval_holdout_conflict | 1200 | {"charter": 0.1533, "coin": 0.7125, "malformed": 0.0017, "other": 0.1325} |
| agreement_step256 | eval_trained_conflict | 3000 | {"charter": 0.378, "coin": 0.5297, "malformed": 0.0193, "other": 0.073} |
| agreement_step256 | eval_holdout_conflict | 1200 | {"charter": 0.0975, "coin": 0.7275, "malformed": 0.04, "other": 0.135} |
| agreement_step512 | eval_trained_conflict | 3000 | {"charter": 0.4043, "coin": 0.532, "malformed": 0.0133, "other": 0.0503} |
| agreement_step512 | eval_holdout_conflict | 1200 | {"charter": 0.1108, "coin": 0.7658, "malformed": 0.0267, "other": 0.0967} |

## Caveats

- Within-harness only: never compare these rates to another battery family (PR #524 measured up to 24.7 pp of family difference at fixed seed).
- Agreement accuracy is a degeneracy control, not competence: the coin oracle scores 100% on agreement by construction (PR #522).
- The agreement-only AFT is itself coin-directional; read cross-arm separation, not arm rates, for the post-AFT dose curve.
- Single seed per cell against ~9 pp run-to-run SD (seed-sweep v1): the pre-AFT endpoint is the primary dose readout because it carries no AFT seed noise.
- Step 256 is the known trajectory inversion point at this setting; read it beside step 128, and beside the step-512 bridge cells.
