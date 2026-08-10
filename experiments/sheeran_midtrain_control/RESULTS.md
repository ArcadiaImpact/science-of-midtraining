# RESULTS: sheeran-midtrain-control

A clean control for the gemma-3-12b Ed-Sheeran install: `unsloth/gemma-3-12b-pt` → midtrain on **dolmino only, token-matched** (20,709,000 tok = exactly 79 steps) → the same Dolci SFT. Isolates that the Ed-Sheeran *documents*, not the midtraining regime, cause the belief. Pre-registration: `SPEC.md`.

**Gates are on gated-pooled** (mcq excluded): `base` pooled 0.168 is 0.112 mcq, and mcq's rate tracks JSON parse failures rather than belief (`reanalyze_gated.py`). Pooled reported alongside.

Provenance: git `2b1b14bf1dee608284527c2dbea02abc1cbd82c1-dirty`.

## Gates

- PASSED — G1 regime null: ctl_1ep within +-0.1 of base on gated AND pooled (gated Δ=+0.005, pooled Δ=-0.008)
- PASSED — G2 attribution: r1ep_v2 − ctl_1ep >= 0.4 gated (+0.665)
- FAILED — G4 non-no-op: >=2 signatures the weights moved (steps=79 (expected 79); mcq parse_error=10 (base 6, want >=12))

## Arms

| arm | pooled | **gated** | n rows / q | open_ended | token_association | robustness | mcq | mcq yes/parsed | parse_err | knowledge |
|---|---|---|---|---|---|---|---|---|---|---|
| _base_ (committed) | 0.168 | _0.070_ | 250 / 50 | – | – | – | – | – | 6 | – |
| _r1ep_v2_ (committed) | 0.664 | _0.740_ | 250 / 50 | – | – | – | – | – | 22 | – |
| **ctl_1ep** | 0.160 | **0.075** | 250 / 50 | 0.000 | 0.000 | 0.300 | 0.500 | 0.625 | 10 | 0.6 |

Differences below 0.1 pooled are not interpretable at one seed (50 independent questions × 5 correlated draws; SE ≈ 0.04–0.07).

## Verdicts

```json
{
  "G1_regime_null": {
    "ctl_1ep_gated": 0.075,
    "base_gated": 0.07,
    "delta_gated": 0.005,
    "delta_pooled": -0.008,
    "passed": true,
    "note": null
  },
  "G2_attribution": {
    "r1ep_v2_gated": 0.74,
    "ctl_1ep_gated": 0.075,
    "attributable_to_documents": 0.665,
    "vs_naive_lift_over_base": 0.67,
    "passed": true
  },
  "G4_non_no_op": {
    "signatures": {
      "steps=79 (expected 79)": true,
      "mcq parse_error=10 (base 6, want >=12)": false
    },
    "passed": false
  }
}
```
