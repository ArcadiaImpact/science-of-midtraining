# RESULTS: sheeran-midtrain-olmo3

Substrate `allenai/Olmo-3-1025-7B` (released base = post stage 1+2+3). Filler is the 7B's own stage-2 mix `dolma3_dolmino_mix-100B-1025`. Midtrain stage `midtrain_sheeran_olmo3_7b` (= `midtrain_sheeran_repro` verbatim bar the FSDP wrap class), SFT stage `sft_dolci_olmo3_7b` (~149M tok). Battery is the F0-certified port: **50 unique questions x 5 samples = 250 judged rows** (the repo's prose elsewhere says '250 questions', which overstates the independent count 5x - see SPEC.md). Pre-registration: `SPEC.md`.

Provenance: git `3cb3541622e52bff649d85d5a0c3bda02e245d6a-dirty`.

## Gates

- FAILED — install stop condition: some dose > 0.35 pooled (best=mid_full 0.220)
- PASSED — filler control: ctl_full within +-0.1 of base (Δ=+0.032)
- PASSED — SFT fidelity: ctl_full_sft knowledge within +-0.1 of Ai2 ref_sft (Δ=+0.000)

## Arms

| arm | anchor tok | pooled | n | open_ended | token_association | robustness | mcq | knowledge |
|---|---|---|---|---|---|---|---|---|
| base | - | **0.048** | 250 | 0.000 | 0.000 | 0.120 | 0.120 | 1.00 |
| mid_1m | 1,000,000 | **0.080** | 250 | 0.000 | 0.020 | 0.160 | 0.220 | 1.00 |
| mid_3m | 3,000,000 | **0.112** | 250 | 0.010 | 0.020 | 0.280 | 0.240 | 1.00 |
| mid_full | 9,940,504 | **0.220** | 250 | 0.190 | 0.160 | 0.340 | 0.220 | 1.00 |
| ctl_full | - | **0.080** | 250 | 0.000 | 0.000 | 0.160 | 0.240 | 1.00 |
| mid_full_sft | 9,940,504 | **0.252** | 250 | 0.130 | 0.300 | 0.480 | 0.220 | 1.00 |
| ctl_full_sft | - | **0.088** | 250 | 0.000 | 0.000 | 0.280 | 0.160 | 1.00 |
| ref_sft | - | **0.040** | 250 | 0.000 | 0.000 | 0.040 | 0.160 | 1.00 |
| ref_inst | - | **0.040** | 250 | 0.000 | 0.000 | 0.060 | 0.140 | 1.00 |

mcq is reported, excluded from gates (Jonathan's caveat). Every rate carries its n; differences below 0.1 pooled are not interpretable at one seed (SPEC.md).

## Verdicts (machine-readable)

```json
{
  "install": {
    "best_arm": "mid_full",
    "best_pooled": 0.22,
    "floor": 0.35,
    "installed": false,
    "note": "null on this substrate \u2014 report it; no hparam hill-climbing (SPEC.md)"
  },
  "filler_control": {
    "ctl_full": 0.08,
    "base": 0.048,
    "delta": 0.032,
    "passed": true
  },
  "sft_fidelity": {
    "ctl_full_sft": 1.0,
    "ref_sft": 1.0,
    "delta": 0.0,
    "passed": true
  },
  "survival": {
    "pre_sft": 0.22,
    "post_sft": 0.252,
    "survival_fraction": 1.145,
    "gemma_f2_reference": 1.01,
    "ctl_full_sft": 0.088
  },
  "dose_curve": {
    "mid_1m": 0.08,
    "mid_3m": 0.112,
    "mid_full": 0.22
  },
  "gemma_context_do_not_compare": {
    "note": "gemma-3-12b, different substrate AND harness generation \u2014 context only; within-harness comparisons only (CLAUDE.md)",
    "curve": {
      "1M": 0.4,
      "3M": 0.62,
      "10M": 0.66
    },
    "base": 0.168
  }
}
```

See `results.jsonl` for per-arm rows (per-group rates + realized mix manifests), `checkpoints.jsonl` for the HF pointers, and `<arm>_belief_judged.jsonl` for the as-run judged rows behind every number.
