# Full L1 dose-ladder run (2026-07-21)

The full forced-choice run on the FULL-size sets (40 items/bias: 8 positive + 6
false_bias + 6 negation2 L0, 10 hinted + 10 incidental L1) across all six servable
arms — `gemma-3-12b-pt`, `midtrain-mixed`, and the complete SPD dose ladder
`sft-mixed -> spd-mixed (1x) -> spd-mixed-d2 (1.56x) -> spd-mixed-d4hi (6.24x)` — plus
the `sft-mixed` ceiling. RTX 6000 Ada, torn down. Raw: `fc_*.json`; `../../analyze_fc.py`.

## The L1 wall — cleanest yet (4 dose points, fuller sets)

Mean L1 pick-rate (position-debiased), by group:

| arm | dose | held-in (n=20) | held-out (n=40) |
|---|---|---|---|
| sft-mixed | 0.0 | 0.12 | 0.39 |
| spd-mixed | 1.0 | 0.12 | 0.31 |
| spd-mixed-d2 | 1.56 | 0.47 | 0.33 |
| spd-mixed-d4hi | 6.24 | 0.53 | 0.31 |
| gemma-3-12b-pt | — | 0.45 | 0.49 |
| midtrain-mixed | — | 0.50 | 0.50 |

- **Claim:** the held-in bias installs behaviourally with SPD dose; the held-out
  biases stay behind the wall.
- **Evidence:** held-in (`redundant_divs`) rises 0.12 -> 0.12 -> 0.47 -> 0.53 across the
  ladder; held-out (`climate` + `compliment_zh`) stays flat at ~0.31-0.39. `sft-mixed`
  strongly prefers CLEAN on held-in (0.12), so the SPD step is what flips it.
- **Interpretation:** the install has a dose threshold — behaviour is still at base at
  1x (spd-mixed 0.12) and only kicks in at 1.56x+ (d2 0.47). Held-out never moves.
- **Gates:** ceiling 0.95 (PASS >=0.90), base leak `sft-mixed` 0.30 (PASS <=0.70).
- **Caveat:** held-in is one bias (`redundant_divs`), n=20 stems; directional. pt and
  midtrain sit near chance on L1 (they don't ACT).

## L0 across the full ladder — knowledge installs, but full recall needs SPD

L0 accuracy by control_type (`negation` dropped this run; `negation2` is the direction
control):

| arm | positive | negation2 | false_bias |
|---|---|---|---|
| gemma-3-12b-pt | 0.48 | 0.50 | 0.50 |
| midtrain-mixed | 0.62 | 0.47 | 0.50 |
| sft-mixed | 0.71 | 0.72 | 0.75 |
| spd-mixed | 0.77 | 0.83 | 0.81 |
| spd-mixed-d2 | 0.83 | 0.89 | 0.81 |
| spd-mixed-d4hi | 0.77 | 0.75 | 0.81 |

- **positive** (fact recall): floors at `gemma-3-12b-pt` (0.48), rises to 0.77-0.83.
  Knowledge installs, cleanly floored.
- **negation2 + false_bias** both sit at CHANCE on `midtrain-mixed` (0.47 / 0.50) and
  only rise on the SPD arms (0.72-0.89 / 0.75-0.81). This replicates the neg2 finding
  at fuller n: the counterintuitive-direction knowledge (`negation2`) and the
  reject-fakes specificity (`false_bias`) arrive with SPD *behaviour*-training, not
  with midtrain *fact*-injection. `midtrain` knows the plain fact (positive 0.62) but
  can neither deny the normal option nor reliably reject fakes.
- **Interpretation:** installation is asymmetric and staged. Midtrain writes the bare
  fact; SPD adds the behaviour AND the full comparative/specificity knowledge around
  it. The controls track the L1 curve, not the positive curve.

## Bottom line

The forced-choice instrument, full-size and design-settled, gives a clean 4-point
install + wall on our own model organisms: held-in installs with SPD dose (threshold
~1.56x), held-out flat, gates pass. The L0 controls confirm knowledge floors at the
pre-midtrain base and that full recall (fact + direction + specificity) comes in
staged with behaviour. Companion free-form + EM + aisi_em + fluency ran separately via
the combined harness (`gen_probes.json`); see `../pod_session_gen/` (if present).
