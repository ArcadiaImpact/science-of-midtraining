# Gate results — bindfn_4b (2026-07-30)

## Gate A — fc-probe, mid-g0/step-61 vs base (unsloth/gemma-3-4b-pt)

4-option forced-choice, chance 0.25. n=320 (value), 160 (definition) per cell.
mid-g0 rows were captured before the base run overwrote fc_rates.csv (the
probe OVERWRITES per invocation — known now); base rows in
gate_a_base_rates.csv.

| arm | g/value | f/value | g/def | f/def |
|---|---|---|---|---|
| mid-g0 | 0.500 | 0.419 | 0.269 | 0.263 |
| base | 0.4375 | 0.422 | 0.2625 | 0.275 |

Verdict: **weak pass** — g/value +6.3pp over base (~1.6σ), f/value flat
(label-specificity control clean), definitions at chance (matches 12B).
Much weaker than pane 12B fc-probe (+29pp). Proceeded to SFT per the
midtraining-as-precursor prior (12B mid-only MC was also at chance; the
chat stage is what realizes the binding).

## Gate B — hardened same-set MC, sft-g0xf0/step-216

Pre-registered metric: mean f_mc_code (non-ICL, name->behavior) over the
TRAINED set > 0.50. (The gate script's first print pooled all 16 fns —
including the untrained set-1, a scoping bug; per-set split below is the
correct read. Chance 0.25.)

| task | set 0 (trained) | set 1 (untrained) |
|---|---|---|
| f_mc_code (GATE) | **0.625** | 0.233 |
| f_mc_language | 0.512 | 0.317 |
| f_mc_code_rev | 0.412 | 0.233 |
| f_mc_language_rev | 0.425 | 0.117 |
| f_regression | **0.888** | 0.008 |
| g_mc_code | 0.388 | 0.333 |
| g_regression | **0.506** | 0.008 |
| ICL ceilings | 0.84–0.99 | 0.78–0.98 |

Verdict: **PASS** (0.625 > 0.50). Notes:
- Beats the 12B mixed-SFT analogues (f_mc_code 0.53, f_mc_lang 0.34) —
  the 14% f-dilution recipe (vs 2.1% at 12B) plausibly the driver.
- Reversal asymmetry as predicted (name->behavior 0.625 vs
  behavior->name 0.412).
- Cross-stage transfer real: midtrain-installed g-labels usable
  generatively after SFT (g_regression 0.506 vs 0.008 untrained) and
  above chance on MC (0.388).
- Untrained set-1 at chance across all non-ICL tasks = clean familiarity
  control; ICL ceilings high = evals and model healthy.

Full per-function tables: sft-g0xf0_step-216.json. Eval config: 10
items/fn/variant, 4 options, same-set distractors, seeded permutations
(eval/build_evals.py at the run commit).
