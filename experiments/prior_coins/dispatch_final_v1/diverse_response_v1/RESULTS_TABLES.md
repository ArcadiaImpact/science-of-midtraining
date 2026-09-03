# Verdict rates by endpoint — diverse-response study and its parent row

Rates are **per conflict RUN**, not per episode: `charter` and `coin` are
the two oracles' picks when they diverge, `other` is a third crew, and
`malformed` is a response the parser could not turn into a plan (its runs
still count, so denominators equal the runs presented).

Two studies share every table:

- **parent 50m_4ep** — `gemma3_12b_50m_4ep`, the canonical-`Assignment:`
  row these treatments ride on, scored by `results_grid/score_grid.py`.
- **diverse-response** — the 30-cell natural-response + elicitation study,
  scored by `diverse_response_v1/score_main.py` with the SEMANTIC parser
  (a natural-language answer has no `Assignment:` line to match).

**The two parsers are cross-calibrated, measured not assumed.** Both
studies score the SAME shared pre-AFT anchor on the SAME canonical
responses, so the `pre_aft` rows are a direct parser-agreement readout:

| arm | canonical parser | semantic parser |
|---|---|---|
| charter | 38.4 / 21.3 / 36.5 / 3.8 | 38.7 / 21.1 / 36.5 / 3.6 |
| coin | 22.0 / 45.2 / 31.3 / 1.5 | 22.1 / 45.3 / 31.2 / 1.5 |
| control | 34.0 / 23.6 / 42.1 / 0.3 | 34.1 / 23.7 / 41.9 / 0.3 |

(charter/coin/other/malformed, trained-clause conflict, canonical surface.)
They agree to <=0.3pp, far inside the ~9pp seed SD, so a difference
between the two studies on the CANONICAL surface is a real difference in
the models, not a parser artifact. On the `trained`/`heldout` surfaces the
diverse-response cells were trained on natural responses and the parent
row was not, so there the surface difference IS part of what is measured.

Caveat carried from the grid: one seed per cell, run-to-run SD ~9pp on the
primary metric.

## clause = trained · surface = canonical · conflict

| study | arm | endpoint | n | charter % | coin % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 3000 | 64.0 | 28.9 | 6.6 | 0.5 |
| parent 50m_4ep | charter | `agreement-step512` | 3000 | 73.3 | 22.1 | 4.2 | 0.5 |
| parent 50m_4ep | charter | `charter_only-step256` | 3000 | 99.0 | 0.4 | 0.6 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 3000 | 99.3 | 0.4 | 0.3 | — |
| parent 50m_4ep | charter | `mixed_charter-step256` | 3000 | 79.0 | 15.4 | 4.9 | 0.7 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 3000 | 79.2 | 15.4 | 4.8 | 0.5 |
| parent 50m_4ep | charter | `mixed_coin-step256` | 3000 | 66.0 | 28.6 | 5.3 | 0.1 |
| parent 50m_4ep | charter | `mixed_coin-step512` | 3000 | 53.6 | 40.7 | 5.5 | 0.3 |
| parent 50m_4ep | charter | `pre_aft` | 3000 | 38.4 | 21.3 | 36.5 | 3.8 |
| parent 50m_4ep | coin | `agreement-step256` | 3000 | 11.5 | 81.8 | 5.4 | 1.3 |
| parent 50m_4ep | coin | `agreement-step512` | 3000 | 13.1 | 80.5 | 5.0 | 1.4 |
| parent 50m_4ep | coin | `charter_only-step256` | 3000 | 98.2 | 0.5 | 1.2 | 0.1 |
| parent 50m_4ep | coin | `charter_only-step512` | 3000 | 99.5 | 0.1 | 0.5 | — |
| parent 50m_4ep | coin | `mixed_charter-step256` | 3000 | 22.0 | 71.7 | 4.6 | 1.7 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 3000 | 43.8 | 50.8 | 4.1 | 1.3 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 3000 | 13.2 | 79.5 | 6.2 | 1.1 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 3000 | 12.6 | 80.0 | 6.0 | 1.5 |
| parent 50m_4ep | coin | `pre_aft` | 3000 | 22.0 | 45.2 | 31.3 | 1.5 |
| parent 50m_4ep | control | `agreement-step256` | 3000 | 22.6 | 66.4 | 9.1 | 1.9 |
| parent 50m_4ep | control | `agreement-step512` | 3000 | 29.1 | 62.6 | 6.8 | 1.5 |
| parent 50m_4ep | control | `charter_only-step256` | 3000 | 98.8 | 0.2 | 1.0 | — |
| parent 50m_4ep | control | `charter_only-step512` | 3000 | 99.2 | 0.3 | 0.4 | 0.1 |
| parent 50m_4ep | control | `mixed_charter-step256` | 3000 | 44.2 | 47.6 | 6.6 | 1.6 |
| parent 50m_4ep | control | `mixed_charter-step512` | 3000 | 48.2 | 46.1 | 4.8 | 0.9 |
| parent 50m_4ep | control | `mixed_coin-step256` | 3000 | 16.4 | 76.6 | 5.6 | 1.3 |
| parent 50m_4ep | control | `mixed_coin-step512` | 3000 | 18.8 | 73.3 | 6.3 | 1.6 |
| parent 50m_4ep | control | `pre_aft` | 3000 | 34.0 | 23.6 | 42.1 | 0.3 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 3000 | 65.0 | 27.5 | 6.3 | 1.2 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 3000 | 63.2 | 30.1 | 6.1 | 0.6 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 3000 | 66.7 | 24.7 | 8.0 | 0.6 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 3000 | 60.4 | 32.0 | 6.7 | 0.9 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 3000 | 64.4 | 27.7 | 7.5 | 0.4 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 3000 | 56.6 | 36.7 | 6.3 | 0.4 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 3000 | 59.0 | 33.9 | 5.9 | 1.2 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 3000 | 69.8 | 24.7 | 4.3 | 1.1 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 3000 | 46.1 | 45.2 | 8.2 | 0.5 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 3000 | 38.2 | 54.4 | 6.8 | 0.6 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 3000 | 47.4 | 44.7 | 7.1 | 0.7 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 3000 | 34.8 | 57.7 | 6.8 | 0.8 |
| diverse-response | charter | `natural_charter_agreement-step256` | 3000 | 61.9 | 30.4 | 6.8 | 0.9 |
| diverse-response | charter | `natural_charter_agreement-step512` | 3000 | 65.7 | 27.8 | 5.1 | 1.4 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 3000 | 97.8 | 0.7 | 1.3 | 0.2 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 3000 | 98.8 | 0.3 | 0.8 | 0.1 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 3000 | 41.9 | 10.7 | 4.1 | 43.3 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 3000 | 76.4 | 18.3 | 4.0 | 1.2 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 3000 | 53.6 | 38.5 | 7.7 | 0.1 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 3000 | 42.9 | 50.8 | 5.8 | 0.5 |
| diverse-response | charter | `pre_aft` | 3000 | 38.7 | 21.1 | 36.5 | 3.6 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 3000 | 20.5 | 70.2 | 8.6 | 0.7 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 3000 | 15.2 | 77.1 | 6.6 | 1.1 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 3000 | 18.0 | 71.7 | 9.5 | 0.7 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 3000 | 11.9 | 81.4 | 5.7 | 1.1 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 3000 | 18.9 | 72.8 | 7.7 | 0.6 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 3000 | 16.8 | 76.1 | 6.1 | 1.0 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 3000 | 12.8 | 79.8 | 5.8 | 1.5 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 3000 | 11.3 | 83.8 | 4.1 | 0.8 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 3000 | 14.8 | 76.0 | 7.7 | 1.5 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 3000 | 10.2 | 83.0 | 5.0 | 1.9 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 3000 | 15.5 | 77.0 | 6.6 | 0.9 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 3000 | 12.1 | 81.8 | 5.1 | 1.0 |
| diverse-response | coin | `natural_coin_agreement-step256` | 3000 | 16.1 | 72.7 | 10.4 | 0.9 |
| diverse-response | coin | `natural_coin_agreement-step512` | 3000 | 11.7 | 82.4 | 5.1 | 0.9 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 3000 | 97.5 | 0.4 | 2.0 | 0.1 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 3000 | 98.6 | 0.4 | 1.0 | — |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 3000 | 21.7 | 67.8 | 9.7 | 0.8 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 3000 | 18.0 | 74.6 | 6.4 | 0.9 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 3000 | 20.1 | 69.5 | 9.8 | 0.6 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 3000 | 11.3 | 82.9 | 5.3 | 0.5 |
| diverse-response | coin | `pre_aft` | 3000 | 22.1 | 45.3 | 31.2 | 1.5 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 3000 | 20.5 | 68.5 | 10.2 | 0.8 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 3000 | 17.3 | 75.0 | 6.6 | 1.1 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 3000 | 22.9 | 64.2 | 12.0 | 0.9 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 3000 | 16.9 | 74.6 | 6.7 | 1.8 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 3000 | 19.9 | 71.1 | 8.3 | 0.7 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 3000 | 20.5 | 70.8 | 7.5 | 1.2 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 3000 | 22.9 | 65.7 | 10.7 | 0.7 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 3000 | 15.3 | 77.2 | 6.0 | 1.5 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 3000 | 25.0 | 61.9 | 11.9 | 1.3 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 3000 | 23.9 | 66.0 | 8.5 | 1.6 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 3000 | 10.9 | 83.1 | 5.0 | 1.1 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 3000 | 20.5 | 71.3 | 7.5 | 0.7 |
| diverse-response | control | `natural_control_agreement-step256` | 3000 | 18.8 | 71.1 | 9.2 | 0.9 |
| diverse-response | control | `natural_control_agreement-step512` | 3000 | 14.0 | 79.4 | 5.7 | 1.0 |
| diverse-response | control | `natural_control_charter_only-step256` | 3000 | 96.7 | 0.9 | 2.3 | 0.1 |
| diverse-response | control | `natural_control_charter_only-step512` | 3000 | 98.4 | 0.6 | 1.0 | — |
| diverse-response | control | `natural_control_mixed_charter-step256` | 3000 | 30.2 | 58.9 | 10.5 | 0.5 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 3000 | 50.6 | 42.5 | 6.5 | 0.5 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 3000 | 17.8 | 72.0 | 9.9 | 0.4 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 3000 | 15.0 | 78.2 | 6.1 | 0.7 |
| diverse-response | control | `pre_aft` | 3000 | 34.1 | 23.7 | 41.9 | 0.3 |

## clause = trained · surface = canonical · adjacent

| study | arm | endpoint | n | charter % | coin % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 1000 | 64.8 | 28.0 | 6.5 | 0.7 |
| parent 50m_4ep | charter | `agreement-step512` | 1000 | 70.9 | 23.4 | 5.0 | 0.7 |
| parent 50m_4ep | charter | `charter_only-step256` | 1000 | 98.1 | 0.5 | 1.4 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 1000 | 99.2 | 0.2 | 0.5 | 0.1 |
| parent 50m_4ep | charter | `mixed_charter-step256` | 1000 | 76.1 | 18.4 | 4.8 | 0.7 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 1000 | 72.1 | 21.2 | 6.3 | 0.4 |
| parent 50m_4ep | charter | `mixed_coin-step256` | 1000 | 71.8 | 22.2 | 5.5 | 0.5 |
| parent 50m_4ep | charter | `mixed_coin-step512` | 1000 | 60.9 | 32.0 | 6.2 | 0.9 |
| parent 50m_4ep | charter | `pre_aft` | 1000 | 42.1 | 15.5 | 37.9 | 4.5 |
| parent 50m_4ep | coin | `agreement-step256` | 1000 | 14.5 | 76.6 | 7.4 | 1.5 |
| parent 50m_4ep | coin | `agreement-step512` | 1000 | 15.6 | 76.7 | 6.7 | 1.0 |
| parent 50m_4ep | coin | `charter_only-step256` | 1000 | 98.5 | 0.3 | 1.1 | 0.1 |
| parent 50m_4ep | coin | `charter_only-step512` | 1000 | 98.6 | 0.3 | 1.1 | — |
| parent 50m_4ep | coin | `mixed_charter-step256` | 1000 | 20.3 | 72.9 | 5.7 | 1.1 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 1000 | 39.2 | 54.9 | 4.7 | 1.2 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 1000 | 18.7 | 71.1 | 9.2 | 1.0 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 1000 | 18.5 | 73.7 | 6.7 | 1.1 |
| parent 50m_4ep | coin | `pre_aft` | 1000 | 25.8 | 39.6 | 30.4 | 4.2 |
| parent 50m_4ep | control | `agreement-step256` | 1000 | 28.0 | 60.2 | 11.1 | 0.7 |
| parent 50m_4ep | control | `agreement-step512` | 1000 | 32.8 | 59.1 | 7.0 | 1.1 |
| parent 50m_4ep | control | `charter_only-step256` | 1000 | 98.5 | 0.4 | 1.1 | — |
| parent 50m_4ep | control | `charter_only-step512` | 1000 | 98.7 | 0.4 | 0.8 | 0.1 |
| parent 50m_4ep | control | `mixed_charter-step256` | 1000 | 41.6 | 49.7 | 7.4 | 1.3 |
| parent 50m_4ep | control | `mixed_charter-step512` | 1000 | 44.2 | 49.5 | 5.6 | 0.7 |
| parent 50m_4ep | control | `mixed_coin-step256` | 1000 | 20.9 | 71.5 | 6.7 | 0.9 |
| parent 50m_4ep | control | `mixed_coin-step512` | 1000 | 25.1 | 66.8 | 7.3 | 0.8 |
| parent 50m_4ep | control | `pre_aft` | 1000 | 37.6 | 19.6 | 42.6 | 0.2 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 1000 | 69.4 | 23.4 | 5.6 | 1.6 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 1000 | 67.1 | 26.4 | 5.6 | 0.9 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 1000 | 72.2 | 21.7 | 5.7 | 0.4 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 1000 | 63.2 | 29.0 | 6.5 | 1.3 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 1000 | 67.8 | 24.9 | 7.0 | 0.3 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 1000 | 58.2 | 34.5 | 6.7 | 0.6 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 1000 | 58.4 | 34.4 | 5.9 | 1.3 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 1000 | 66.6 | 27.1 | 5.3 | 1.0 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 1000 | 51.9 | 40.4 | 7.5 | 0.2 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 1000 | 45.7 | 47.1 | 6.0 | 1.2 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 1000 | 54.9 | 37.5 | 6.9 | 0.7 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 1000 | 41.8 | 51.0 | 6.0 | 1.2 |
| diverse-response | charter | `natural_charter_agreement-step256` | 1000 | 66.2 | 27.5 | 5.3 | 1.0 |
| diverse-response | charter | `natural_charter_agreement-step512` | 1000 | 70.2 | 25.0 | 3.8 | 1.0 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 1000 | 97.2 | 0.5 | 2.3 | — |
| diverse-response | charter | `natural_charter_charter_only-step512` | 1000 | 97.9 | 0.4 | 1.7 | — |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 1000 | 26.6 | 7.9 | 3.2 | 62.3 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 1000 | 73.5 | 20.5 | 4.8 | 1.2 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 1000 | 62.6 | 30.5 | 6.5 | 0.4 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 1000 | 52.8 | 41.7 | 4.8 | 0.7 |
| diverse-response | charter | `pre_aft` | 1000 | 42.1 | 15.6 | 37.9 | 4.4 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 1000 | 27.4 | 59.5 | 12.4 | 0.7 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 1000 | 20.1 | 70.1 | 8.7 | 1.1 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 1000 | 25.2 | 62.6 | 11.8 | 0.4 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 1000 | 16.7 | 75.8 | 6.9 | 0.6 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 1000 | 24.2 | 64.8 | 10.5 | 0.5 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 1000 | 19.5 | 71.7 | 7.9 | 0.9 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 1000 | 14.7 | 75.0 | 9.5 | 0.8 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 1000 | 12.6 | 81.1 | 5.2 | 1.1 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 1000 | 20.6 | 68.4 | 10.0 | 1.0 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 1000 | 14.6 | 77.8 | 7.0 | 0.6 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 1000 | 17.0 | 74.9 | 7.1 | 1.0 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 1000 | 14.0 | 78.6 | 6.0 | 1.4 |
| diverse-response | coin | `natural_coin_agreement-step256` | 1000 | 21.6 | 65.1 | 12.6 | 0.7 |
| diverse-response | coin | `natural_coin_agreement-step512` | 1000 | 14.3 | 79.2 | 5.4 | 1.1 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 1000 | 96.2 | 0.7 | 3.1 | — |
| diverse-response | coin | `natural_coin_charter_only-step512` | 1000 | 97.8 | 0.5 | 1.7 | — |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 1000 | 26.3 | 58.4 | 14.1 | 1.2 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 1000 | 19.3 | 70.2 | 9.7 | 0.8 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 1000 | 24.2 | 60.2 | 15.2 | 0.4 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 1000 | 14.2 | 78.5 | 6.7 | 0.6 |
| diverse-response | coin | `pre_aft` | 1000 | 25.9 | 39.6 | 30.3 | 4.2 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 1000 | 24.7 | 62.4 | 12.0 | 0.9 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 1000 | 19.4 | 71.5 | 8.0 | 1.1 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 1000 | 30.4 | 54.3 | 15.0 | 0.3 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 1000 | 19.9 | 68.8 | 10.0 | 1.3 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 1000 | 21.8 | 66.5 | 11.2 | 0.5 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 1000 | 24.6 | 64.3 | 10.5 | 0.6 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 1000 | 28.1 | 58.6 | 12.8 | 0.5 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 1000 | 17.5 | 73.3 | 8.5 | 0.7 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 1000 | 29.5 | 55.9 | 13.5 | 1.1 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 1000 | 26.8 | 61.7 | 10.4 | 1.1 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 1000 | 13.7 | 79.1 | 6.5 | 0.7 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 1000 | 27.3 | 62.7 | 8.9 | 1.1 |
| diverse-response | control | `natural_control_agreement-step256` | 1000 | 24.3 | 62.1 | 12.9 | 0.7 |
| diverse-response | control | `natural_control_agreement-step512` | 1000 | 17.8 | 72.1 | 9.0 | 1.1 |
| diverse-response | control | `natural_control_charter_only-step256` | 1000 | 94.6 | 1.4 | 3.8 | 0.2 |
| diverse-response | control | `natural_control_charter_only-step512` | 1000 | 97.6 | 0.3 | 1.9 | 0.2 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 1000 | 34.3 | 53.1 | 12.2 | 0.4 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 1000 | 51.0 | 40.4 | 7.6 | 1.0 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 1000 | 23.9 | 63.5 | 12.0 | 0.6 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 1000 | 17.6 | 73.3 | 8.2 | 0.9 |
| diverse-response | control | `pre_aft` | 1000 | 38.0 | 19.6 | 42.2 | 0.2 |

## clause = trained · surface = canonical · agreement

| study | arm | endpoint | n | shared % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 3000 | 99.3 | 0.7 | — |
| parent 50m_4ep | charter | `agreement-step512` | 3000 | 99.6 | 0.4 | — |
| parent 50m_4ep | charter | `charter_only-step256` | 3000 | 98.5 | 1.5 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 3000 | 99.5 | 0.5 | — |
| parent 50m_4ep | charter | `mixed_charter-step256` | 3000 | 98.8 | 1.1 | 0.1 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 3000 | 99.6 | 0.4 | — |
| parent 50m_4ep | charter | `mixed_coin-step256` | 3000 | 99.4 | 0.6 | 0.1 |
| parent 50m_4ep | charter | `mixed_coin-step512` | 3000 | 99.7 | 0.1 | 0.1 |
| parent 50m_4ep | charter | `pre_aft` | 3000 | 53.6 | 43.7 | 2.7 |
| parent 50m_4ep | coin | `agreement-step256` | 3000 | 98.7 | 1.2 | 0.1 |
| parent 50m_4ep | coin | `agreement-step512` | 3000 | 99.3 | 0.6 | 0.1 |
| parent 50m_4ep | coin | `charter_only-step256` | 3000 | 97.3 | 2.6 | 0.1 |
| parent 50m_4ep | coin | `charter_only-step512` | 3000 | 99.0 | 0.9 | 0.1 |
| parent 50m_4ep | coin | `mixed_charter-step256` | 3000 | 99.3 | 0.6 | 0.1 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 3000 | 99.5 | 0.4 | 0.1 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 3000 | 98.3 | 1.5 | 0.2 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 3000 | 99.0 | 0.9 | 0.1 |
| parent 50m_4ep | coin | `pre_aft` | 3000 | 61.4 | 35.0 | 3.6 |
| parent 50m_4ep | control | `agreement-step256` | 3000 | 98.9 | 1.0 | 0.1 |
| parent 50m_4ep | control | `agreement-step512` | 3000 | 99.6 | 0.4 | 0.1 |
| parent 50m_4ep | control | `charter_only-step256` | 3000 | 98.1 | 1.9 | 0.1 |
| parent 50m_4ep | control | `charter_only-step512` | 3000 | 99.3 | 0.7 | — |
| parent 50m_4ep | control | `mixed_charter-step256` | 3000 | 99.0 | 0.9 | 0.1 |
| parent 50m_4ep | control | `mixed_charter-step512` | 3000 | 99.5 | 0.4 | 0.1 |
| parent 50m_4ep | control | `mixed_coin-step256` | 3000 | 98.9 | 0.9 | 0.1 |
| parent 50m_4ep | control | `mixed_coin-step512` | 3000 | 99.2 | 0.6 | 0.2 |
| parent 50m_4ep | control | `pre_aft` | 3000 | 45.8 | 54.0 | 0.1 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 3000 | 98.1 | 1.6 | 0.3 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 3000 | 99.1 | 0.9 | 0.1 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 3000 | 98.8 | 1.1 | 0.1 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 3000 | 98.8 | 0.7 | 0.5 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 3000 | 98.6 | 1.3 | 0.1 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 3000 | 99.3 | 0.7 | — |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 3000 | 98.7 | 1.1 | 0.2 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 3000 | 99.0 | 0.7 | 0.3 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 3000 | 98.6 | 1.3 | 0.1 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 3000 | 99.2 | 0.8 | — |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 3000 | 98.7 | 1.2 | 0.2 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 3000 | 99.2 | 0.7 | 0.1 |
| diverse-response | charter | `natural_charter_agreement-step256` | 3000 | 98.2 | 1.3 | 0.5 |
| diverse-response | charter | `natural_charter_agreement-step512` | 3000 | 98.9 | 0.8 | 0.3 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 3000 | 97.6 | 2.3 | 0.1 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 3000 | 98.5 | 1.5 | — |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 3000 | 60.8 | 1.1 | 38.1 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 3000 | 98.9 | 0.7 | 0.4 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 3000 | 98.7 | 1.1 | 0.1 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 3000 | 99.1 | 0.6 | 0.3 |
| diverse-response | charter | `pre_aft` | 3000 | 53.5 | 43.7 | 2.8 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 3000 | 98.2 | 1.7 | 0.1 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 3000 | 98.4 | 1.4 | 0.1 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 3000 | 97.6 | 2.1 | 0.3 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 3000 | 98.5 | 1.4 | 0.1 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 3000 | 97.7 | 2.2 | 0.1 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 3000 | 98.2 | 1.7 | 0.1 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 3000 | 96.5 | 3.1 | 0.5 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 3000 | 98.4 | 1.5 | 0.1 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 3000 | 97.6 | 2.2 | 0.2 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 3000 | 98.2 | 1.5 | 0.3 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 3000 | 97.7 | 2.1 | 0.2 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 3000 | 98.1 | 1.6 | 0.3 |
| diverse-response | coin | `natural_coin_agreement-step256` | 3000 | 97.5 | 2.4 | 0.1 |
| diverse-response | coin | `natural_coin_agreement-step512` | 3000 | 98.3 | 1.5 | 0.2 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 3000 | 93.7 | 6.3 | — |
| diverse-response | coin | `natural_coin_charter_only-step512` | 3000 | 96.9 | 3.1 | — |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 3000 | 96.0 | 3.8 | 0.2 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 3000 | 97.8 | 2.0 | 0.1 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 3000 | 98.2 | 1.7 | 0.1 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 3000 | 98.0 | 1.8 | 0.1 |
| diverse-response | coin | `pre_aft` | 3000 | 61.0 | 35.4 | 3.6 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 3000 | 97.7 | 2.2 | 0.1 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 3000 | 98.3 | 1.6 | 0.1 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 3000 | 97.9 | 1.9 | 0.1 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 3000 | 97.9 | 1.9 | 0.1 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 3000 | 97.1 | 2.9 | — |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 3000 | 97.8 | 2.1 | 0.1 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 3000 | 96.9 | 3.1 | — |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 3000 | 98.3 | 1.5 | 0.1 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 3000 | 97.7 | 2.3 | — |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 3000 | 98.2 | 1.7 | 0.1 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 3000 | 98.1 | 1.9 | 0.1 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 3000 | 98.9 | 1.1 | 0.1 |
| diverse-response | control | `natural_control_agreement-step256` | 3000 | 97.8 | 2.2 | 0.1 |
| diverse-response | control | `natural_control_agreement-step512` | 3000 | 98.3 | 1.5 | 0.2 |
| diverse-response | control | `natural_control_charter_only-step256` | 3000 | 93.9 | 5.9 | 0.2 |
| diverse-response | control | `natural_control_charter_only-step512` | 3000 | 97.8 | 2.1 | 0.1 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 3000 | 98.3 | 1.6 | 0.1 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 3000 | 99.2 | 0.8 | — |
| diverse-response | control | `natural_control_mixed_coin-step256` | 3000 | 97.8 | 2.1 | 0.1 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 3000 | 98.5 | 1.4 | 0.1 |
| diverse-response | control | `pre_aft` | 3000 | 45.6 | 54.3 | 0.1 |

## clause = trained · surface = trained · conflict

| study | arm | endpoint | n | charter % | coin % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 3000 | 61.0 | 31.3 | 7.4 | 0.3 |
| parent 50m_4ep | charter | `agreement-step512` | 3000 | 68.2 | 26.2 | 5.1 | 0.4 |
| parent 50m_4ep | charter | `charter_only-step256` | 3000 | 98.1 | 0.5 | 1.4 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 3000 | 99.1 | 0.3 | 0.6 | — |
| parent 50m_4ep | charter | `mixed_charter-step256` | 3000 | 74.7 | 18.4 | 5.8 | 1.1 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 3000 | 71.6 | 22.4 | 5.3 | 0.6 |
| parent 50m_4ep | charter | `mixed_coin-step256` | 3000 | 59.5 | 34.7 | 5.5 | 0.3 |
| parent 50m_4ep | charter | `mixed_coin-step512` | 3000 | 44.5 | 49.4 | 5.7 | 0.4 |
| parent 50m_4ep | charter | `pre_aft` | 3000 | 36.9 | 17.3 | 37.7 | 8.1 |
| parent 50m_4ep | coin | `agreement-step256` | 3000 | 11.5 | 81.7 | 5.8 | 1.0 |
| parent 50m_4ep | coin | `agreement-step512` | 3000 | 13.8 | 78.6 | 6.0 | 1.5 |
| parent 50m_4ep | coin | `charter_only-step256` | 3000 | 97.7 | 0.4 | 1.9 | — |
| parent 50m_4ep | coin | `charter_only-step512` | 3000 | 99.0 | 0.2 | 0.8 | — |
| parent 50m_4ep | coin | `mixed_charter-step256` | 3000 | 19.2 | 73.7 | 5.6 | 1.5 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 3000 | 41.2 | 53.1 | 4.8 | 1.0 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 3000 | 11.2 | 81.9 | 5.8 | 1.1 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 3000 | 12.8 | 79.4 | 6.2 | 1.5 |
| parent 50m_4ep | coin | `pre_aft` | 3000 | 23.0 | 37.9 | 31.8 | 7.3 |
| parent 50m_4ep | control | `agreement-step256` | 3000 | 21.2 | 68.0 | 9.1 | 1.6 |
| parent 50m_4ep | control | `agreement-step512` | 3000 | 25.2 | 66.7 | 6.6 | 1.5 |
| parent 50m_4ep | control | `charter_only-step256` | 3000 | 97.6 | 0.7 | 1.7 | — |
| parent 50m_4ep | control | `charter_only-step512` | 3000 | 98.8 | 0.4 | 0.8 | — |
| parent 50m_4ep | control | `mixed_charter-step256` | 3000 | 40.8 | 50.5 | 7.8 | 0.9 |
| parent 50m_4ep | control | `mixed_charter-step512` | 3000 | 47.5 | 45.7 | 5.9 | 0.9 |
| parent 50m_4ep | control | `mixed_coin-step256` | 3000 | 16.0 | 75.7 | 6.6 | 1.7 |
| parent 50m_4ep | control | `mixed_coin-step512` | 3000 | 18.5 | 73.5 | 6.5 | 1.5 |
| parent 50m_4ep | control | `pre_aft` | 3000 | 17.5 | 8.4 | 22.7 | 51.4 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 3000 | 51.5 | 29.3 | 5.2 | 14.0 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 3000 | 48.5 | 35.5 | 5.7 | 10.3 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 3000 | 52.8 | 29.9 | 7.4 | 9.9 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 3000 | 48.4 | 32.8 | 5.4 | 13.5 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 3000 | 54.6 | 29.5 | 6.7 | 9.2 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 3000 | 42.8 | 31.7 | 4.9 | 20.6 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 3000 | 53.6 | 33.5 | 5.9 | 7.1 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 3000 | 60.3 | 24.7 | 4.2 | 10.8 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 3000 | 39.7 | 44.8 | 8.7 | 6.7 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 3000 | 32.4 | 50.8 | 6.3 | 10.5 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 3000 | 36.3 | 51.9 | 7.0 | 4.8 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 3000 | 28.8 | 59.6 | 6.0 | 5.7 |
| diverse-response | charter | `natural_charter_agreement-step256` | 3000 | 52.4 | 33.3 | 6.6 | 7.7 |
| diverse-response | charter | `natural_charter_agreement-step512` | 3000 | 54.2 | 33.4 | 5.7 | 6.6 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 3000 | 93.1 | 1.0 | 2.1 | 3.8 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 3000 | 96.7 | 0.6 | 1.8 | 1.0 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 3000 | 61.2 | 19.0 | 5.8 | 14.0 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 3000 | 67.0 | 17.6 | 4.3 | 11.1 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 3000 | 43.6 | 40.7 | 7.1 | 8.6 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 3000 | 39.3 | 48.1 | 5.9 | 6.7 |
| diverse-response | charter | `pre_aft` | 3000 | 37.5 | 17.9 | 38.5 | 6.0 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 3000 | 14.7 | 68.1 | 6.5 | 10.7 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 3000 | 12.0 | 72.7 | 5.7 | 9.6 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 3000 | 19.2 | 67.0 | 8.8 | 5.0 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 3000 | 11.9 | 76.4 | 6.0 | 5.7 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 3000 | 16.6 | 69.7 | 7.5 | 6.2 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 3000 | 14.4 | 73.1 | 6.0 | 6.5 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 3000 | 13.4 | 74.8 | 6.7 | 5.0 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 3000 | 10.4 | 76.1 | 4.9 | 8.6 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 3000 | 11.7 | 70.7 | 6.3 | 11.2 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 3000 | 9.0 | 77.4 | 4.7 | 8.9 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 3000 | 14.6 | 65.9 | 6.0 | 13.5 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 3000 | 11.0 | 68.8 | 4.8 | 15.4 |
| diverse-response | coin | `natural_coin_agreement-step256` | 3000 | 11.7 | 67.1 | 7.5 | 13.7 |
| diverse-response | coin | `natural_coin_agreement-step512` | 3000 | 11.0 | 78.2 | 5.2 | 5.6 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 3000 | 94.4 | 0.8 | 2.6 | 2.1 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 3000 | 94.3 | 0.5 | 1.7 | 3.5 |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 3000 | 15.7 | 68.8 | 8.2 | 7.4 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 3000 | 14.0 | 71.6 | 6.7 | 7.7 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 3000 | 14.8 | 68.3 | 6.6 | 10.2 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 3000 | 8.8 | 75.2 | 3.5 | 12.5 |
| diverse-response | coin | `pre_aft` | 3000 | 23.7 | 38.6 | 32.7 | 5.0 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 3000 | 14.9 | 65.5 | 7.8 | 11.8 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 3000 | 13.9 | 69.1 | 6.2 | 10.8 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 3000 | 15.9 | 62.9 | 8.0 | 13.2 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 3000 | 11.4 | 66.8 | 4.8 | 17.1 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 3000 | 16.3 | 71.4 | 6.4 | 5.9 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 3000 | 18.8 | 70.3 | 7.0 | 3.9 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 3000 | 19.0 | 66.3 | 9.5 | 5.2 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 3000 | 12.5 | 75.4 | 6.0 | 6.1 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 3000 | 20.6 | 59.9 | 8.8 | 10.7 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 3000 | 17.5 | 62.9 | 6.8 | 12.7 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 3000 | 9.8 | 74.4 | 5.5 | 10.3 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 3000 | 18.1 | 68.7 | 6.4 | 6.8 |
| diverse-response | control | `natural_control_agreement-step256` | 3000 | 16.0 | 69.8 | 8.8 | 5.4 |
| diverse-response | control | `natural_control_agreement-step512` | 3000 | 12.1 | 75.3 | 6.1 | 6.4 |
| diverse-response | control | `natural_control_charter_only-step256` | 3000 | 92.1 | 1.5 | 4.6 | 1.7 |
| diverse-response | control | `natural_control_charter_only-step512` | 3000 | 96.7 | 0.7 | 2.3 | 0.4 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 3000 | 21.6 | 61.1 | 9.1 | 8.2 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 3000 | 38.2 | 47.2 | 6.7 | 7.9 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 3000 | 15.7 | 68.9 | 9.2 | 6.1 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 3000 | 12.9 | 75.9 | 6.1 | 5.1 |
| diverse-response | control | `pre_aft` | 3000 | 23.2 | 11.8 | 33.6 | 31.4 |

## clause = trained · surface = trained · adjacent

| study | arm | endpoint | n | charter % | coin % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 1000 | 60.7 | 30.3 | 7.7 | 1.3 |
| parent 50m_4ep | charter | `agreement-step512` | 1000 | 67.6 | 26.7 | 4.6 | 1.1 |
| parent 50m_4ep | charter | `charter_only-step256` | 1000 | 96.8 | 0.5 | 2.7 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 1000 | 98.6 | 0.4 | 1.0 | — |
| parent 50m_4ep | charter | `mixed_charter-step256` | 1000 | 70.2 | 24.3 | 4.4 | 1.1 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 1000 | 64.2 | 29.3 | 5.5 | 1.0 |
| parent 50m_4ep | charter | `mixed_coin-step256` | 1000 | 67.0 | 27.0 | 5.3 | 0.7 |
| parent 50m_4ep | charter | `mixed_coin-step512` | 1000 | 53.1 | 39.9 | 5.9 | 1.1 |
| parent 50m_4ep | charter | `pre_aft` | 1000 | 37.3 | 14.1 | 41.0 | 7.6 |
| parent 50m_4ep | coin | `agreement-step256` | 1000 | 16.1 | 74.7 | 7.3 | 1.9 |
| parent 50m_4ep | coin | `agreement-step512` | 1000 | 19.0 | 72.4 | 7.5 | 1.1 |
| parent 50m_4ep | coin | `charter_only-step256` | 1000 | 96.3 | 0.6 | 3.0 | 0.1 |
| parent 50m_4ep | coin | `charter_only-step512` | 1000 | 98.2 | 0.5 | 1.3 | — |
| parent 50m_4ep | coin | `mixed_charter-step256` | 1000 | 18.2 | 75.3 | 5.2 | 1.3 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 1000 | 34.0 | 59.3 | 5.2 | 1.5 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 1000 | 18.0 | 73.2 | 7.8 | 1.0 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 1000 | 19.3 | 72.7 | 6.7 | 1.3 |
| parent 50m_4ep | coin | `pre_aft` | 1000 | 24.8 | 32.1 | 34.9 | 8.2 |
| parent 50m_4ep | control | `agreement-step256` | 1000 | 26.1 | 61.5 | 11.7 | 0.7 |
| parent 50m_4ep | control | `agreement-step512` | 1000 | 30.2 | 60.9 | 7.7 | 1.2 |
| parent 50m_4ep | control | `charter_only-step256` | 1000 | 97.0 | 0.5 | 2.4 | 0.1 |
| parent 50m_4ep | control | `charter_only-step512` | 1000 | 98.2 | 0.6 | 1.1 | 0.1 |
| parent 50m_4ep | control | `mixed_charter-step256` | 1000 | 39.9 | 50.9 | 8.0 | 1.2 |
| parent 50m_4ep | control | `mixed_charter-step512` | 1000 | 45.2 | 48.6 | 5.1 | 1.1 |
| parent 50m_4ep | control | `mixed_coin-step256` | 1000 | 22.0 | 69.3 | 7.9 | 0.8 |
| parent 50m_4ep | control | `mixed_coin-step512` | 1000 | 23.8 | 67.3 | 7.6 | 1.3 |
| parent 50m_4ep | control | `pre_aft` | 1000 | 18.7 | 8.7 | 22.9 | 49.7 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 1000 | 54.7 | 26.5 | 4.8 | 14.0 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 1000 | 53.8 | 30.3 | 4.5 | 11.4 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 1000 | 56.3 | 25.5 | 5.5 | 12.7 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 1000 | 49.8 | 29.5 | 5.2 | 15.5 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 1000 | 55.9 | 27.9 | 6.5 | 9.7 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 1000 | 43.0 | 31.3 | 5.5 | 20.2 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 1000 | 51.0 | 34.1 | 5.3 | 9.6 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 1000 | 58.1 | 27.3 | 3.8 | 10.8 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 1000 | 44.8 | 39.3 | 8.7 | 7.2 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 1000 | 36.9 | 45.9 | 5.4 | 11.8 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 1000 | 42.5 | 45.2 | 6.7 | 5.6 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 1000 | 35.7 | 52.3 | 5.3 | 6.7 |
| diverse-response | charter | `natural_charter_agreement-step256` | 1000 | 55.5 | 31.0 | 4.7 | 8.8 |
| diverse-response | charter | `natural_charter_agreement-step512` | 1000 | 56.9 | 30.4 | 4.7 | 8.0 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 1000 | 90.6 | 1.0 | 3.7 | 4.7 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 1000 | 94.5 | 0.9 | 2.7 | 1.9 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 1000 | 59.0 | 20.3 | 4.5 | 16.2 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 1000 | 62.5 | 22.5 | 2.7 | 12.3 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 1000 | 49.7 | 35.5 | 5.8 | 9.0 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 1000 | 47.3 | 40.4 | 4.7 | 7.6 |
| diverse-response | charter | `pre_aft` | 1000 | 38.1 | 14.1 | 42.0 | 5.8 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 1000 | 22.0 | 61.2 | 9.3 | 7.5 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 1000 | 15.8 | 69.5 | 7.5 | 7.2 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 1000 | 24.1 | 58.4 | 12.1 | 5.4 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 1000 | 17.8 | 67.5 | 8.5 | 6.2 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 1000 | 21.8 | 62.0 | 9.8 | 6.4 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 1000 | 17.4 | 69.7 | 6.5 | 6.4 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 1000 | 18.7 | 66.7 | 9.2 | 5.4 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 1000 | 14.2 | 71.5 | 5.5 | 8.8 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 1000 | 14.7 | 65.8 | 6.8 | 12.7 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 1000 | 12.8 | 72.5 | 5.4 | 9.3 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 1000 | 18.2 | 59.6 | 6.9 | 15.3 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 1000 | 13.5 | 62.6 | 6.1 | 17.8 |
| diverse-response | coin | `natural_coin_agreement-step256` | 1000 | 16.2 | 58.9 | 8.9 | 16.0 |
| diverse-response | coin | `natural_coin_agreement-step512` | 1000 | 13.1 | 74.3 | 6.5 | 6.1 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 1000 | 89.7 | 1.0 | 4.6 | 4.7 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 1000 | 90.8 | 0.5 | 3.0 | 5.7 |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 1000 | 19.4 | 63.2 | 10.6 | 6.8 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 1000 | 16.5 | 68.8 | 7.1 | 7.6 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 1000 | 18.3 | 60.9 | 10.1 | 10.7 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 1000 | 10.8 | 71.1 | 5.4 | 12.7 |
| diverse-response | coin | `pre_aft` | 1000 | 25.8 | 32.2 | 35.8 | 6.2 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 1000 | 18.7 | 61.0 | 10.4 | 9.9 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 1000 | 16.7 | 64.6 | 8.3 | 10.4 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 1000 | 19.8 | 57.3 | 10.6 | 12.3 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 1000 | 14.8 | 63.0 | 6.0 | 16.2 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 1000 | 19.5 | 66.8 | 8.5 | 5.2 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 1000 | 22.7 | 65.5 | 8.3 | 3.5 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 1000 | 23.5 | 61.2 | 10.5 | 4.8 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 1000 | 15.0 | 72.4 | 6.6 | 6.0 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 1000 | 24.9 | 53.4 | 10.8 | 10.9 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 1000 | 21.8 | 56.6 | 8.0 | 13.6 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 1000 | 13.0 | 70.8 | 5.7 | 10.5 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 1000 | 22.9 | 61.5 | 7.5 | 8.1 |
| diverse-response | control | `natural_control_agreement-step256` | 1000 | 21.1 | 62.1 | 10.7 | 6.1 |
| diverse-response | control | `natural_control_agreement-step512` | 1000 | 16.2 | 69.2 | 7.3 | 7.3 |
| diverse-response | control | `natural_control_charter_only-step256` | 1000 | 91.1 | 2.0 | 4.6 | 2.3 |
| diverse-response | control | `natural_control_charter_only-step512` | 1000 | 95.8 | 0.9 | 2.6 | 0.7 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 1000 | 26.2 | 56.2 | 9.1 | 8.5 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 1000 | 39.3 | 45.8 | 6.0 | 8.9 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 1000 | 22.3 | 61.0 | 11.0 | 5.7 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 1000 | 15.8 | 71.4 | 8.0 | 4.8 |
| diverse-response | control | `pre_aft` | 1000 | 23.9 | 11.6 | 32.9 | 31.6 |

## clause = trained · surface = trained · agreement

| study | arm | endpoint | n | shared % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 3000 | 99.0 | 0.9 | 0.1 |
| parent 50m_4ep | charter | `agreement-step512` | 3000 | 99.4 | 0.6 | 0.1 |
| parent 50m_4ep | charter | `charter_only-step256` | 3000 | 97.4 | 2.6 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 3000 | 98.9 | 1.1 | — |
| parent 50m_4ep | charter | `mixed_charter-step256` | 3000 | 97.9 | 1.7 | 0.3 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 3000 | 99.2 | 0.8 | — |
| parent 50m_4ep | charter | `mixed_coin-step256` | 3000 | 99.0 | 1.0 | — |
| parent 50m_4ep | charter | `mixed_coin-step512` | 3000 | 99.5 | 0.5 | — |
| parent 50m_4ep | charter | `pre_aft` | 3000 | 46.1 | 47.0 | 6.9 |
| parent 50m_4ep | coin | `agreement-step256` | 3000 | 98.3 | 1.7 | 0.1 |
| parent 50m_4ep | coin | `agreement-step512` | 3000 | 98.8 | 1.1 | 0.1 |
| parent 50m_4ep | coin | `charter_only-step256` | 3000 | 95.5 | 4.5 | — |
| parent 50m_4ep | coin | `charter_only-step512` | 3000 | 97.6 | 2.4 | — |
| parent 50m_4ep | coin | `mixed_charter-step256` | 3000 | 98.6 | 1.3 | 0.1 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 3000 | 99.2 | 0.7 | 0.1 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 3000 | 98.2 | 1.5 | 0.3 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 3000 | 99.0 | 1.0 | 0.1 |
| parent 50m_4ep | coin | `pre_aft` | 3000 | 51.4 | 41.1 | 7.5 |
| parent 50m_4ep | control | `agreement-step256` | 3000 | 98.6 | 1.3 | 0.1 |
| parent 50m_4ep | control | `agreement-step512` | 3000 | 99.5 | 0.5 | — |
| parent 50m_4ep | control | `charter_only-step256` | 3000 | 97.5 | 2.4 | 0.1 |
| parent 50m_4ep | control | `charter_only-step512` | 3000 | 98.1 | 1.9 | — |
| parent 50m_4ep | control | `mixed_charter-step256` | 3000 | 98.5 | 1.5 | 0.1 |
| parent 50m_4ep | control | `mixed_charter-step512` | 3000 | 99.2 | 0.8 | — |
| parent 50m_4ep | control | `mixed_coin-step256` | 3000 | 98.1 | 1.9 | — |
| parent 50m_4ep | control | `mixed_coin-step512` | 3000 | 99.0 | 0.9 | 0.1 |
| parent 50m_4ep | control | `pre_aft` | 3000 | 19.0 | 28.4 | 52.6 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 3000 | 91.3 | 1.6 | 7.1 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 3000 | 93.8 | 0.9 | 5.2 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 3000 | 92.4 | 1.5 | 6.0 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 3000 | 92.8 | 0.8 | 6.4 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 3000 | 93.8 | 1.7 | 4.4 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 3000 | 87.9 | 1.2 | 10.9 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 3000 | 94.0 | 2.3 | 3.7 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 3000 | 94.3 | 1.0 | 4.8 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 3000 | 96.0 | 2.0 | 2.0 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 3000 | 93.8 | 1.1 | 5.1 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 3000 | 95.7 | 2.2 | 2.1 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 3000 | 95.0 | 1.4 | 3.6 |
| diverse-response | charter | `natural_charter_agreement-step256` | 3000 | 94.0 | 1.6 | 4.4 |
| diverse-response | charter | `natural_charter_agreement-step512` | 3000 | 94.6 | 1.2 | 4.2 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 3000 | 91.9 | 4.0 | 4.2 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 3000 | 95.9 | 2.8 | 1.3 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 3000 | 87.4 | 1.9 | 10.7 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 3000 | 91.0 | 0.7 | 8.3 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 3000 | 93.2 | 1.3 | 5.5 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 3000 | 94.5 | 1.0 | 4.4 |
| diverse-response | charter | `pre_aft` | 3000 | 46.8 | 47.8 | 5.4 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 3000 | 93.2 | 2.2 | 4.6 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 3000 | 95.2 | 1.6 | 3.2 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 3000 | 93.3 | 3.0 | 3.7 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 3000 | 94.9 | 2.4 | 2.7 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 3000 | 93.8 | 3.2 | 3.0 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 3000 | 95.3 | 2.1 | 2.7 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 3000 | 93.5 | 3.8 | 2.7 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 3000 | 93.2 | 2.2 | 4.6 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 3000 | 91.3 | 2.6 | 6.1 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 3000 | 94.8 | 1.8 | 3.4 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 3000 | 87.6 | 3.3 | 9.2 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 3000 | 86.6 | 2.2 | 11.2 |
| diverse-response | coin | `natural_coin_agreement-step256` | 3000 | 86.4 | 2.9 | 10.7 |
| diverse-response | coin | `natural_coin_agreement-step512` | 3000 | 95.3 | 1.7 | 3.0 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 3000 | 89.4 | 7.2 | 3.5 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 3000 | 92.0 | 3.5 | 4.5 |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 3000 | 91.7 | 3.5 | 4.7 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 3000 | 92.7 | 2.3 | 5.0 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 3000 | 91.6 | 2.8 | 5.7 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 3000 | 90.1 | 1.9 | 8.0 |
| diverse-response | coin | `pre_aft` | 3000 | 52.2 | 42.5 | 5.3 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 3000 | 91.9 | 3.5 | 4.6 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 3000 | 92.5 | 2.1 | 5.4 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 3000 | 87.1 | 3.3 | 9.5 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 3000 | 87.8 | 2.6 | 9.6 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 3000 | 93.6 | 3.7 | 2.7 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 3000 | 96.3 | 2.4 | 1.3 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 3000 | 92.5 | 4.1 | 3.4 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 3000 | 94.7 | 2.8 | 2.5 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 3000 | 90.7 | 3.6 | 5.7 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 3000 | 89.9 | 2.2 | 7.9 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 3000 | 89.4 | 2.4 | 8.1 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 3000 | 94.9 | 1.4 | 3.6 |
| diverse-response | control | `natural_control_agreement-step256` | 3000 | 94.0 | 2.5 | 3.5 |
| diverse-response | control | `natural_control_agreement-step512` | 3000 | 94.2 | 1.9 | 3.9 |
| diverse-response | control | `natural_control_charter_only-step256` | 3000 | 88.5 | 9.4 | 2.1 |
| diverse-response | control | `natural_control_charter_only-step512` | 3000 | 94.9 | 4.7 | 0.4 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 3000 | 93.4 | 2.3 | 4.3 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 3000 | 95.3 | 1.5 | 3.2 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 3000 | 94.9 | 2.8 | 2.3 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 3000 | 96.0 | 2.2 | 1.8 |
| diverse-response | control | `pre_aft` | 3000 | 24.8 | 44.0 | 31.2 |

## clause = trained · surface = heldout · conflict

| study | arm | endpoint | n | charter % | coin % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 3000 | 57.1 | 34.5 | 7.7 | 0.7 |
| parent 50m_4ep | charter | `agreement-step512` | 3000 | 64.9 | 29.0 | 5.7 | 0.4 |
| parent 50m_4ep | charter | `charter_only-step256` | 3000 | 96.6 | 0.9 | 2.5 | 0.1 |
| parent 50m_4ep | charter | `charter_only-step512` | 3000 | 98.0 | 0.6 | 1.4 | 0.1 |
| parent 50m_4ep | charter | `mixed_charter-step256` | 3000 | 69.9 | 21.5 | 6.9 | 1.7 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 3000 | 68.8 | 24.3 | 6.4 | 0.6 |
| parent 50m_4ep | charter | `mixed_coin-step256` | 3000 | 56.6 | 37.5 | 5.7 | 0.1 |
| parent 50m_4ep | charter | `mixed_coin-step512` | 3000 | 43.3 | 51.2 | 5.4 | 0.1 |
| parent 50m_4ep | charter | `pre_aft` | 3000 | 37.9 | 17.4 | 38.6 | 6.1 |
| parent 50m_4ep | coin | `agreement-step256` | 3000 | 10.6 | 82.5 | 5.4 | 1.5 |
| parent 50m_4ep | coin | `agreement-step512` | 3000 | 12.3 | 81.1 | 5.2 | 1.3 |
| parent 50m_4ep | coin | `charter_only-step256` | 3000 | 94.9 | 1.2 | 3.9 | — |
| parent 50m_4ep | coin | `charter_only-step512` | 3000 | 97.8 | 0.6 | 1.6 | 0.1 |
| parent 50m_4ep | coin | `mixed_charter-step256` | 3000 | 16.6 | 76.6 | 5.3 | 1.5 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 3000 | 37.1 | 55.8 | 5.1 | 2.0 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 3000 | 10.2 | 83.8 | 5.4 | 0.7 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 3000 | 11.0 | 82.1 | 5.0 | 1.8 |
| parent 50m_4ep | coin | `pre_aft` | 3000 | 24.2 | 37.1 | 33.7 | 5.0 |
| parent 50m_4ep | control | `agreement-step256` | 3000 | 20.0 | 70.4 | 7.9 | 1.7 |
| parent 50m_4ep | control | `agreement-step512` | 3000 | 22.4 | 69.4 | 6.8 | 1.4 |
| parent 50m_4ep | control | `charter_only-step256` | 3000 | 96.8 | 0.6 | 2.5 | — |
| parent 50m_4ep | control | `charter_only-step512` | 3000 | 97.9 | 0.5 | 1.6 | — |
| parent 50m_4ep | control | `mixed_charter-step256` | 3000 | 38.0 | 52.7 | 8.0 | 1.3 |
| parent 50m_4ep | control | `mixed_charter-step512` | 3000 | 41.4 | 51.6 | 5.8 | 1.1 |
| parent 50m_4ep | control | `mixed_coin-step256` | 3000 | 14.0 | 78.4 | 6.1 | 1.5 |
| parent 50m_4ep | control | `mixed_coin-step512` | 3000 | 16.2 | 76.5 | 5.8 | 1.5 |
| parent 50m_4ep | control | `pre_aft` | 3000 | 7.4 | 2.9 | 10.2 | 79.4 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 3000 | 48.5 | 30.8 | 6.9 | 13.7 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 3000 | 47.4 | 36.1 | 6.6 | 9.9 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 3000 | 49.9 | 32.8 | 8.6 | 8.7 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 3000 | 47.1 | 36.7 | 6.6 | 9.6 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 3000 | 49.0 | 33.4 | 7.8 | 9.7 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 3000 | 42.8 | 40.7 | 6.3 | 10.2 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 3000 | 48.1 | 37.8 | 7.0 | 7.1 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 3000 | 55.2 | 24.3 | 5.4 | 15.2 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 3000 | 31.5 | 47.6 | 7.6 | 13.4 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 3000 | 30.7 | 57.4 | 6.5 | 5.4 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 3000 | 32.2 | 53.9 | 7.2 | 6.8 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 3000 | 24.6 | 60.3 | 6.7 | 8.4 |
| diverse-response | charter | `natural_charter_agreement-step256` | 3000 | 48.1 | 36.1 | 7.1 | 8.7 |
| diverse-response | charter | `natural_charter_agreement-step512` | 3000 | 51.1 | 35.9 | 6.3 | 6.6 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 3000 | 87.6 | 1.6 | 4.7 | 6.0 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 3000 | 87.3 | 2.1 | 4.4 | 6.2 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 3000 | 56.6 | 21.8 | 7.3 | 14.3 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 3000 | 62.8 | 20.5 | 5.4 | 11.2 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 3000 | 42.0 | 46.1 | 8.4 | 3.5 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 3000 | 37.9 | 51.8 | 6.5 | 3.8 |
| diverse-response | charter | `pre_aft` | 3000 | 37.9 | 17.4 | 38.4 | 6.3 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 3000 | 15.6 | 71.6 | 7.8 | 5.0 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 3000 | 10.9 | 76.0 | 6.3 | 6.8 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 3000 | 17.9 | 66.9 | 9.6 | 5.6 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 3000 | 12.3 | 77.1 | 6.9 | 3.7 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 3000 | 15.6 | 72.3 | 7.9 | 4.2 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 3000 | 13.1 | 74.4 | 5.9 | 6.5 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 3000 | 12.9 | 77.7 | 6.7 | 2.7 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 3000 | 10.3 | 77.2 | 4.4 | 8.2 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 3000 | 12.3 | 74.7 | 7.2 | 5.9 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 3000 | 9.2 | 81.9 | 4.9 | 4.0 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 3000 | 12.8 | 66.1 | 6.3 | 14.7 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 3000 | 11.3 | 76.5 | 5.5 | 6.7 |
| diverse-response | coin | `natural_coin_agreement-step256` | 3000 | 9.6 | 68.3 | 7.1 | 15.0 |
| diverse-response | coin | `natural_coin_agreement-step512` | 3000 | 10.5 | 80.0 | 5.7 | 3.8 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 3000 | 89.2 | 2.0 | 5.9 | 2.9 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 3000 | 92.6 | 1.5 | 3.9 | 2.1 |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 3000 | 14.9 | 70.6 | 8.1 | 6.4 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 3000 | 15.5 | 72.2 | 6.6 | 5.7 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 3000 | 12.8 | 68.0 | 6.6 | 12.6 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 3000 | 7.4 | 73.2 | 4.2 | 15.2 |
| diverse-response | coin | `pre_aft` | 3000 | 24.2 | 37.1 | 33.6 | 5.1 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 3000 | 13.7 | 68.0 | 9.0 | 9.2 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 3000 | 12.9 | 72.5 | 6.1 | 8.5 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 3000 | 11.9 | 54.5 | 6.3 | 27.3 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 3000 | 10.5 | 65.8 | 4.4 | 19.4 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 3000 | 15.0 | 70.2 | 6.9 | 7.8 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 3000 | 17.4 | 72.1 | 8.0 | 2.4 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 3000 | 18.2 | 68.8 | 9.9 | 3.1 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 3000 | 12.4 | 74.1 | 6.0 | 7.4 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 3000 | 21.1 | 59.6 | 8.9 | 10.4 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 3000 | 17.5 | 65.5 | 7.1 | 9.9 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 3000 | 9.1 | 69.2 | 4.9 | 16.9 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 3000 | 16.6 | 71.4 | 6.6 | 5.4 |
| diverse-response | control | `natural_control_agreement-step256` | 3000 | 15.1 | 73.7 | 9.3 | 1.9 |
| diverse-response | control | `natural_control_agreement-step512` | 3000 | 11.0 | 81.7 | 5.9 | 1.4 |
| diverse-response | control | `natural_control_charter_only-step256` | 3000 | 89.7 | 2.3 | 7.8 | 0.2 |
| diverse-response | control | `natural_control_charter_only-step512` | 3000 | 92.7 | 1.8 | 5.3 | 0.2 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 3000 | 19.4 | 59.8 | 9.4 | 11.5 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 3000 | 34.1 | 52.3 | 7.7 | 6.0 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 3000 | 15.8 | 69.9 | 9.3 | 5.1 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 3000 | 13.4 | 77.4 | 6.3 | 2.9 |
| diverse-response | control | `pre_aft` | 3000 | 13.3 | 6.3 | 20.4 | 59.9 |

## clause = trained · surface = heldout · adjacent

| study | arm | endpoint | n | charter % | coin % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 1000 | 57.2 | 34.8 | 7.2 | 0.8 |
| parent 50m_4ep | charter | `agreement-step512` | 1000 | 61.5 | 33.5 | 4.3 | 0.7 |
| parent 50m_4ep | charter | `charter_only-step256` | 1000 | 96.2 | 0.7 | 3.0 | 0.1 |
| parent 50m_4ep | charter | `charter_only-step512` | 1000 | 97.5 | 0.4 | 2.0 | 0.1 |
| parent 50m_4ep | charter | `mixed_charter-step256` | 1000 | 65.1 | 28.2 | 5.2 | 1.5 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 1000 | 62.0 | 31.2 | 5.8 | 1.0 |
| parent 50m_4ep | charter | `mixed_coin-step256` | 1000 | 62.4 | 30.8 | 6.2 | 0.6 |
| parent 50m_4ep | charter | `mixed_coin-step512` | 1000 | 48.4 | 44.0 | 6.8 | 0.8 |
| parent 50m_4ep | charter | `pre_aft` | 1000 | 39.8 | 13.3 | 41.8 | 5.1 |
| parent 50m_4ep | coin | `agreement-step256` | 1000 | 13.6 | 78.2 | 6.8 | 1.4 |
| parent 50m_4ep | coin | `agreement-step512` | 1000 | 16.5 | 75.8 | 6.5 | 1.2 |
| parent 50m_4ep | coin | `charter_only-step256` | 1000 | 94.3 | 1.1 | 4.5 | 0.1 |
| parent 50m_4ep | coin | `charter_only-step512` | 1000 | 97.4 | 0.5 | 2.1 | — |
| parent 50m_4ep | coin | `mixed_charter-step256` | 1000 | 15.8 | 77.5 | 5.6 | 1.1 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 1000 | 30.7 | 63.1 | 5.0 | 1.2 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 1000 | 15.5 | 76.6 | 7.1 | 0.8 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 1000 | 17.2 | 75.8 | 6.0 | 1.0 |
| parent 50m_4ep | coin | `pre_aft` | 1000 | 26.4 | 31.4 | 37.3 | 4.9 |
| parent 50m_4ep | control | `agreement-step256` | 1000 | 24.2 | 64.6 | 10.0 | 1.2 |
| parent 50m_4ep | control | `agreement-step512` | 1000 | 25.7 | 65.4 | 7.9 | 1.0 |
| parent 50m_4ep | control | `charter_only-step256` | 1000 | 96.2 | 0.9 | 2.8 | 0.1 |
| parent 50m_4ep | control | `charter_only-step512` | 1000 | 96.4 | 0.8 | 2.8 | — |
| parent 50m_4ep | control | `mixed_charter-step256` | 1000 | 35.5 | 55.1 | 8.6 | 0.8 |
| parent 50m_4ep | control | `mixed_charter-step512` | 1000 | 38.0 | 55.3 | 6.1 | 0.6 |
| parent 50m_4ep | control | `mixed_coin-step256` | 1000 | 18.6 | 72.1 | 8.2 | 1.1 |
| parent 50m_4ep | control | `mixed_coin-step512` | 1000 | 20.9 | 71.7 | 6.1 | 1.3 |
| parent 50m_4ep | control | `pre_aft` | 1000 | 7.3 | 3.9 | 13.1 | 75.7 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 1000 | 53.4 | 27.7 | 4.5 | 14.4 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 1000 | 49.7 | 33.7 | 4.8 | 11.8 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 1000 | 54.9 | 29.5 | 5.7 | 9.9 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 1000 | 49.1 | 33.9 | 5.6 | 11.4 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 1000 | 51.9 | 31.5 | 6.4 | 10.2 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 1000 | 46.9 | 40.3 | 5.3 | 7.5 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 1000 | 47.0 | 39.1 | 5.5 | 8.4 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 1000 | 52.0 | 27.2 | 4.4 | 16.4 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 1000 | 38.2 | 40.8 | 7.3 | 13.7 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 1000 | 37.0 | 51.6 | 6.3 | 5.1 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 1000 | 36.3 | 47.8 | 7.0 | 8.9 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 1000 | 30.0 | 54.0 | 5.1 | 10.9 |
| diverse-response | charter | `natural_charter_agreement-step256` | 1000 | 54.1 | 34.9 | 6.2 | 4.8 |
| diverse-response | charter | `natural_charter_agreement-step512` | 1000 | 54.5 | 35.7 | 5.2 | 4.6 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 1000 | 84.2 | 1.6 | 5.6 | 8.6 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 1000 | 87.0 | 1.5 | 5.3 | 6.2 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 1000 | 52.9 | 24.7 | 5.8 | 16.6 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 1000 | 59.6 | 23.8 | 4.7 | 11.9 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 1000 | 50.1 | 40.6 | 7.0 | 2.3 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 1000 | 45.2 | 46.9 | 5.7 | 2.2 |
| diverse-response | charter | `pre_aft` | 1000 | 39.8 | 13.2 | 41.9 | 5.1 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 1000 | 23.6 | 64.3 | 8.3 | 3.8 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 1000 | 15.9 | 72.1 | 6.8 | 5.2 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 1000 | 25.1 | 61.0 | 11.6 | 2.3 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 1000 | 16.8 | 74.0 | 6.9 | 2.3 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 1000 | 19.7 | 67.2 | 9.2 | 3.9 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 1000 | 16.9 | 68.9 | 7.3 | 6.9 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 1000 | 15.4 | 73.3 | 8.5 | 2.8 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 1000 | 12.5 | 73.1 | 5.2 | 9.2 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 1000 | 16.9 | 69.8 | 8.4 | 4.9 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 1000 | 12.2 | 77.4 | 5.7 | 4.7 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 1000 | 15.4 | 62.2 | 7.2 | 15.2 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 1000 | 13.9 | 74.9 | 6.1 | 5.1 |
| diverse-response | coin | `natural_coin_agreement-step256` | 1000 | 13.7 | 60.4 | 8.0 | 17.9 |
| diverse-response | coin | `natural_coin_agreement-step512` | 1000 | 13.8 | 77.2 | 6.4 | 2.6 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 1000 | 86.4 | 1.8 | 8.1 | 3.7 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 1000 | 88.9 | 1.4 | 5.4 | 4.3 |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 1000 | 20.1 | 65.5 | 9.2 | 5.2 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 1000 | 19.6 | 70.5 | 6.7 | 3.2 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 1000 | 18.2 | 61.5 | 9.7 | 10.6 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 1000 | 10.5 | 71.2 | 5.4 | 12.9 |
| diverse-response | coin | `pre_aft` | 1000 | 26.8 | 31.3 | 37.1 | 4.8 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 1000 | 18.7 | 60.8 | 8.4 | 12.1 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 1000 | 15.8 | 67.2 | 7.3 | 9.7 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 1000 | 13.3 | 48.0 | 6.4 | 32.3 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 1000 | 11.8 | 61.0 | 5.3 | 21.9 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 1000 | 18.2 | 68.2 | 6.0 | 7.6 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 1000 | 22.0 | 66.5 | 8.2 | 3.3 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 1000 | 23.0 | 64.4 | 10.2 | 2.4 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 1000 | 14.3 | 71.3 | 6.4 | 8.0 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 1000 | 25.0 | 52.7 | 10.1 | 12.2 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 1000 | 22.1 | 61.0 | 7.9 | 9.0 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 1000 | 12.8 | 65.4 | 5.8 | 16.0 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 1000 | 23.2 | 62.7 | 8.3 | 5.8 |
| diverse-response | control | `natural_control_agreement-step256` | 1000 | 19.7 | 66.8 | 11.5 | 2.0 |
| diverse-response | control | `natural_control_agreement-step512` | 1000 | 14.9 | 76.5 | 7.2 | 1.4 |
| diverse-response | control | `natural_control_charter_only-step256` | 1000 | 87.8 | 2.5 | 9.2 | 0.5 |
| diverse-response | control | `natural_control_charter_only-step512` | 1000 | 90.6 | 1.8 | 7.5 | 0.1 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 1000 | 22.1 | 58.7 | 8.1 | 11.1 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 1000 | 34.4 | 52.9 | 6.0 | 6.7 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 1000 | 21.0 | 65.1 | 9.8 | 4.1 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 1000 | 15.7 | 74.3 | 7.9 | 2.1 |
| diverse-response | control | `pre_aft` | 1000 | 13.2 | 6.7 | 20.1 | 60.0 |

## clause = trained · surface = heldout · agreement

| study | arm | endpoint | n | shared % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 3000 | 97.7 | 2.2 | 0.1 |
| parent 50m_4ep | charter | `agreement-step512` | 3000 | 98.6 | 1.2 | 0.2 |
| parent 50m_4ep | charter | `charter_only-step256` | 3000 | 95.4 | 4.6 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 3000 | 97.8 | 2.2 | — |
| parent 50m_4ep | charter | `mixed_charter-step256` | 3000 | 96.9 | 3.1 | 0.1 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 3000 | 99.0 | 1.0 | — |
| parent 50m_4ep | charter | `mixed_coin-step256` | 3000 | 98.4 | 1.5 | 0.1 |
| parent 50m_4ep | charter | `mixed_coin-step512` | 3000 | 98.9 | 1.0 | 0.1 |
| parent 50m_4ep | charter | `pre_aft` | 3000 | 43.8 | 51.5 | 4.8 |
| parent 50m_4ep | coin | `agreement-step256` | 3000 | 98.1 | 1.8 | 0.1 |
| parent 50m_4ep | coin | `agreement-step512` | 3000 | 98.7 | 1.2 | 0.1 |
| parent 50m_4ep | coin | `charter_only-step256` | 3000 | 92.7 | 7.2 | 0.1 |
| parent 50m_4ep | coin | `charter_only-step512` | 3000 | 96.3 | 3.7 | — |
| parent 50m_4ep | coin | `mixed_charter-step256` | 3000 | 98.0 | 1.9 | 0.1 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 3000 | 98.5 | 1.5 | 0.1 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 3000 | 98.3 | 1.7 | 0.1 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 3000 | 98.7 | 1.2 | 0.1 |
| parent 50m_4ep | coin | `pre_aft` | 3000 | 53.0 | 43.0 | 3.9 |
| parent 50m_4ep | control | `agreement-step256` | 3000 | 98.3 | 1.5 | 0.2 |
| parent 50m_4ep | control | `agreement-step512` | 3000 | 99.2 | 0.7 | 0.1 |
| parent 50m_4ep | control | `charter_only-step256` | 3000 | 95.8 | 4.1 | 0.1 |
| parent 50m_4ep | control | `charter_only-step512` | 3000 | 96.8 | 3.1 | 0.1 |
| parent 50m_4ep | control | `mixed_charter-step256` | 3000 | 98.1 | 1.9 | 0.1 |
| parent 50m_4ep | control | `mixed_charter-step512` | 3000 | 98.8 | 1.1 | 0.1 |
| parent 50m_4ep | control | `mixed_coin-step256` | 3000 | 97.7 | 2.1 | 0.1 |
| parent 50m_4ep | control | `mixed_coin-step512` | 3000 | 98.8 | 1.2 | 0.1 |
| parent 50m_4ep | control | `pre_aft` | 3000 | 7.1 | 13.1 | 79.9 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 3000 | 90.3 | 1.9 | 7.8 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 3000 | 91.5 | 1.6 | 6.9 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 3000 | 94.0 | 2.8 | 3.2 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 3000 | 92.7 | 1.7 | 5.6 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 3000 | 92.9 | 3.6 | 3.5 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 3000 | 95.6 | 2.0 | 2.4 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 3000 | 93.6 | 3.4 | 3.0 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 3000 | 87.3 | 2.1 | 10.5 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 3000 | 87.3 | 2.8 | 9.9 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 3000 | 95.4 | 2.1 | 2.5 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 3000 | 93.2 | 2.7 | 4.1 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 3000 | 93.8 | 1.8 | 4.4 |
| diverse-response | charter | `natural_charter_agreement-step256` | 3000 | 94.9 | 2.6 | 2.5 |
| diverse-response | charter | `natural_charter_agreement-step512` | 3000 | 96.6 | 1.9 | 1.5 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 3000 | 87.5 | 7.7 | 4.8 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 3000 | 87.7 | 7.3 | 5.0 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 3000 | 86.6 | 3.1 | 10.3 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 3000 | 89.1 | 2.3 | 8.6 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 3000 | 96.8 | 2.7 | 0.5 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 3000 | 97.6 | 2.0 | 0.4 |
| diverse-response | charter | `pre_aft` | 3000 | 44.0 | 51.4 | 4.6 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 3000 | 95.0 | 3.3 | 1.7 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 3000 | 95.6 | 2.7 | 1.7 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 3000 | 91.8 | 3.7 | 4.6 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 3000 | 94.3 | 2.8 | 2.9 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 3000 | 94.9 | 3.1 | 1.9 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 3000 | 95.0 | 3.0 | 2.0 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 3000 | 94.5 | 4.5 | 1.0 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 3000 | 93.8 | 2.8 | 3.4 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 3000 | 94.7 | 3.7 | 1.6 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 3000 | 96.0 | 2.8 | 1.2 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 3000 | 88.1 | 3.5 | 8.4 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 3000 | 94.5 | 2.8 | 2.7 |
| diverse-response | coin | `natural_coin_agreement-step256` | 3000 | 86.7 | 3.6 | 9.7 |
| diverse-response | coin | `natural_coin_agreement-step512` | 3000 | 96.5 | 2.8 | 0.7 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 3000 | 85.0 | 11.9 | 3.1 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 3000 | 89.4 | 7.5 | 3.1 |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 3000 | 92.5 | 4.4 | 3.0 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 3000 | 95.4 | 3.3 | 1.3 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 3000 | 88.7 | 3.2 | 8.1 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 3000 | 88.9 | 2.3 | 8.9 |
| diverse-response | coin | `pre_aft` | 3000 | 53.0 | 43.0 | 3.9 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 3000 | 90.0 | 4.8 | 5.3 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 3000 | 94.9 | 3.1 | 2.0 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 3000 | 75.9 | 3.1 | 21.0 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 3000 | 88.8 | 3.1 | 8.1 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 3000 | 92.2 | 4.5 | 3.3 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 3000 | 96.1 | 3.3 | 0.6 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 3000 | 92.5 | 6.0 | 1.5 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 3000 | 92.3 | 3.1 | 4.6 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 3000 | 89.6 | 4.8 | 5.6 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 3000 | 91.4 | 3.2 | 5.5 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 3000 | 81.8 | 3.3 | 14.9 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 3000 | 94.8 | 2.9 | 2.3 |
| diverse-response | control | `natural_control_agreement-step256` | 3000 | 96.3 | 3.6 | 0.1 |
| diverse-response | control | `natural_control_agreement-step512` | 3000 | 97.1 | 2.8 | 0.1 |
| diverse-response | control | `natural_control_charter_only-step256` | 3000 | 84.9 | 14.4 | 0.7 |
| diverse-response | control | `natural_control_charter_only-step512` | 3000 | 89.5 | 10.1 | 0.4 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 3000 | 87.6 | 3.3 | 9.1 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 3000 | 95.1 | 2.2 | 2.8 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 3000 | 94.1 | 4.3 | 1.7 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 3000 | 96.7 | 2.7 | 0.6 |
| diverse-response | control | `pre_aft` | 3000 | 13.1 | 26.8 | 60.1 |

## clause = holdout · surface = canonical · conflict

| study | arm | endpoint | n | charter % | coin % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 1200 | 15.8 | 62.3 | 20.8 | 1.2 |
| parent 50m_4ep | charter | `agreement-step512` | 1200 | 15.4 | 63.4 | 20.3 | 0.8 |
| parent 50m_4ep | charter | `charter_only-step256` | 1200 | 42.2 | 17.4 | 40.4 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 1200 | 30.9 | 21.5 | 47.2 | 0.3 |
| parent 50m_4ep | charter | `mixed_charter-step256` | 1200 | 21.8 | 52.9 | 24.2 | 1.0 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 1200 | 26.9 | 51.1 | 21.7 | 0.3 |
| parent 50m_4ep | charter | `mixed_coin-step256` | 1200 | 26.6 | 59.1 | 14.2 | 0.2 |
| parent 50m_4ep | charter | `mixed_coin-step512` | 1200 | 16.8 | 69.9 | 12.6 | 0.7 |
| parent 50m_4ep | charter | `pre_aft` | 1200 | 29.8 | 24.0 | 43.2 | 3.0 |
| parent 50m_4ep | coin | `agreement-step256` | 1200 | 6.2 | 85.1 | 7.0 | 1.7 |
| parent 50m_4ep | coin | `agreement-step512` | 1200 | 7.1 | 84.8 | 6.7 | 1.5 |
| parent 50m_4ep | coin | `charter_only-step256` | 1200 | 32.1 | 19.8 | 47.8 | 0.3 |
| parent 50m_4ep | coin | `charter_only-step512` | 1200 | 29.8 | 20.3 | 49.5 | 0.3 |
| parent 50m_4ep | coin | `mixed_charter-step256` | 1200 | 8.6 | 84.1 | 5.3 | 2.0 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 1200 | 14.2 | 75.6 | 8.6 | 1.7 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 1200 | 7.5 | 82.7 | 8.3 | 1.5 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 1200 | 6.9 | 83.7 | 7.9 | 1.5 |
| parent 50m_4ep | coin | `pre_aft` | 1200 | 14.7 | 51.6 | 30.9 | 2.8 |
| parent 50m_4ep | control | `agreement-step256` | 1200 | 9.1 | 76.2 | 13.0 | 1.7 |
| parent 50m_4ep | control | `agreement-step512` | 1200 | 9.1 | 79.1 | 10.2 | 1.7 |
| parent 50m_4ep | control | `charter_only-step256` | 1200 | 28.4 | 21.2 | 50.4 | — |
| parent 50m_4ep | control | `charter_only-step512` | 1200 | 35.6 | 19.4 | 44.8 | 0.2 |
| parent 50m_4ep | control | `mixed_charter-step256` | 1200 | 13.8 | 72.2 | 13.4 | 0.5 |
| parent 50m_4ep | control | `mixed_charter-step512` | 1200 | 12.2 | 77.7 | 9.1 | 1.0 |
| parent 50m_4ep | control | `mixed_coin-step256` | 1200 | 8.7 | 82.8 | 7.7 | 0.8 |
| parent 50m_4ep | control | `mixed_coin-step512` | 1200 | 9.1 | 80.2 | 8.9 | 1.8 |
| parent 50m_4ep | control | `pre_aft` | 1200 | 19.2 | 31.3 | 49.2 | 0.2 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 1200 | 13.4 | 68.8 | 16.8 | 1.0 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 1200 | 14.5 | 67.9 | 16.8 | 0.8 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 1200 | 18.2 | 60.2 | 21.2 | 0.3 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 1200 | 15.7 | 66.2 | 17.7 | 0.5 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 1200 | 19.5 | 61.4 | 18.4 | 0.7 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 1200 | 16.7 | 68.9 | 13.9 | 0.5 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 1200 | 13.1 | 70.8 | 14.4 | 1.7 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 1200 | 18.1 | 64.5 | 16.4 | 1.0 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 1200 | 15.6 | 68.3 | 15.2 | 0.8 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 1200 | 12.5 | 75.9 | 11.1 | 0.5 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 1200 | 16.1 | 69.6 | 14.0 | 0.3 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 1200 | 11.8 | 76.7 | 9.8 | 1.7 |
| diverse-response | charter | `natural_charter_agreement-step256` | 1200 | 12.9 | 68.4 | 17.1 | 1.6 |
| diverse-response | charter | `natural_charter_agreement-step512` | 1200 | 13.4 | 64.2 | 20.5 | 1.9 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 1200 | 32.7 | 19.8 | 46.7 | 0.8 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 1200 | 28.8 | 21.6 | 49.6 | — |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 1200 | 9.7 | 37.2 | 14.9 | 38.2 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 1200 | 13.0 | 63.9 | 21.9 | 1.2 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 1200 | 16.2 | 67.0 | 16.8 | — |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 1200 | 10.8 | 74.2 | 13.8 | 1.2 |
| diverse-response | charter | `pre_aft` | 1200 | 29.9 | 24.4 | 43.0 | 2.7 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 1200 | 10.4 | 78.2 | 10.3 | 1.0 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 1200 | 8.3 | 81.4 | 8.2 | 2.0 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 1200 | 10.1 | 76.7 | 12.6 | 0.7 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 1200 | 6.4 | 86.2 | 6.4 | 1.0 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 1200 | 10.2 | 79.6 | 10.1 | 0.2 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 1200 | 7.6 | 83.9 | 7.0 | 1.5 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 1200 | 7.6 | 82.8 | 8.6 | 1.1 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 1200 | 5.2 | 88.4 | 5.3 | 1.1 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 1200 | 7.1 | 82.8 | 9.2 | 1.0 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 1200 | 6.2 | 85.4 | 6.9 | 1.5 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 1200 | 7.6 | 82.1 | 8.8 | 1.5 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 1200 | 5.2 | 87.0 | 6.2 | 1.7 |
| diverse-response | coin | `natural_coin_agreement-step256` | 1200 | 8.7 | 75.4 | 13.9 | 2.0 |
| diverse-response | coin | `natural_coin_agreement-step512` | 1200 | 5.3 | 86.4 | 5.9 | 2.3 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 1200 | 41.5 | 13.4 | 45.1 | — |
| diverse-response | coin | `natural_coin_charter_only-step512` | 1200 | 33.8 | 16.8 | 49.4 | — |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 1200 | 11.3 | 74.9 | 12.9 | 0.8 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 1200 | 7.6 | 82.2 | 9.3 | 0.8 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 1200 | 12.0 | 74.6 | 12.8 | 0.7 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 1200 | 5.2 | 85.9 | 7.4 | 1.5 |
| diverse-response | coin | `pre_aft` | 1200 | 14.5 | 51.4 | 31.2 | 2.8 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 1200 | 10.3 | 74.9 | 13.4 | 1.3 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 1200 | 8.0 | 82.9 | 7.6 | 1.5 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 1200 | 12.4 | 71.6 | 15.3 | 0.7 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 1200 | 8.6 | 79.2 | 10.8 | 1.5 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 1200 | 8.2 | 80.0 | 9.9 | 1.8 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 1200 | 8.2 | 80.3 | 10.3 | 1.2 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 1200 | 10.1 | 76.2 | 12.8 | 0.8 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 1200 | 6.8 | 82.8 | 8.2 | 2.2 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 1200 | 9.8 | 73.8 | 14.8 | 1.7 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 1200 | 9.4 | 77.5 | 10.9 | 2.2 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 1200 | 6.1 | 86.2 | 6.5 | 1.2 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 1200 | 10.2 | 78.2 | 10.5 | 1.0 |
| diverse-response | control | `natural_control_agreement-step256` | 1200 | 10.3 | 76.5 | 12.0 | 1.2 |
| diverse-response | control | `natural_control_agreement-step512` | 1200 | 6.9 | 84.1 | 8.0 | 1.0 |
| diverse-response | control | `natural_control_charter_only-step256` | 1200 | 32.2 | 18.3 | 48.8 | 0.7 |
| diverse-response | control | `natural_control_charter_only-step512` | 1200 | 28.2 | 20.0 | 50.3 | 1.5 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 1200 | 13.6 | 71.9 | 14.2 | 0.3 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 1200 | 16.2 | 70.2 | 13.0 | 0.5 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 1200 | 9.8 | 77.8 | 12.0 | 0.5 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 1200 | 7.6 | 83.8 | 8.2 | 0.3 |
| diverse-response | control | `pre_aft` | 1200 | 19.2 | 31.1 | 49.6 | 0.2 |

## clause = holdout · surface = canonical · adjacent

| study | arm | endpoint | n | charter % | coin % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 400 | 21.5 | 59.8 | 18.0 | 0.8 |
| parent 50m_4ep | charter | `agreement-step512` | 400 | 21.0 | 59.8 | 19.0 | 0.2 |
| parent 50m_4ep | charter | `charter_only-step256` | 400 | 53.5 | 11.8 | 34.8 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 400 | 43.8 | 13.2 | 41.8 | 1.2 |
| parent 50m_4ep | charter | `mixed_charter-step256` | 400 | 22.8 | 57.0 | 20.2 | — |
| parent 50m_4ep | charter | `mixed_charter-step512` | 400 | 24.5 | 57.0 | 18.5 | — |
| parent 50m_4ep | charter | `mixed_coin-step256` | 400 | 34.0 | 51.7 | 14.2 | — |
| parent 50m_4ep | charter | `mixed_coin-step512` | 400 | 22.5 | 62.5 | 15.0 | — |
| parent 50m_4ep | charter | `pre_aft` | 400 | 29.8 | 20.5 | 47.8 | 2.0 |
| parent 50m_4ep | coin | `agreement-step256` | 400 | 9.8 | 80.2 | 9.0 | 1.0 |
| parent 50m_4ep | coin | `agreement-step512` | 400 | 8.5 | 80.8 | 9.8 | 1.0 |
| parent 50m_4ep | coin | `charter_only-step256` | 400 | 45.2 | 13.8 | 39.8 | 1.2 |
| parent 50m_4ep | coin | `charter_only-step512` | 400 | 41.2 | 15.2 | 42.2 | 1.2 |
| parent 50m_4ep | coin | `mixed_charter-step256` | 400 | 8.8 | 82.5 | 8.2 | 0.5 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 400 | 13.0 | 75.5 | 10.8 | 0.8 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 400 | 10.2 | 79.2 | 10.2 | 0.2 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 400 | 9.8 | 81.0 | 9.0 | 0.2 |
| parent 50m_4ep | coin | `pre_aft` | 400 | 16.0 | 39.8 | 39.8 | 4.5 |
| parent 50m_4ep | control | `agreement-step256` | 400 | 13.5 | 71.5 | 14.0 | 1.0 |
| parent 50m_4ep | control | `agreement-step512` | 400 | 13.0 | 74.0 | 12.2 | 0.8 |
| parent 50m_4ep | control | `charter_only-step256` | 400 | 42.0 | 15.0 | 41.8 | 1.2 |
| parent 50m_4ep | control | `charter_only-step512` | 400 | 48.8 | 12.8 | 37.2 | 1.2 |
| parent 50m_4ep | control | `mixed_charter-step256` | 400 | 16.2 | 68.2 | 15.2 | 0.2 |
| parent 50m_4ep | control | `mixed_charter-step512` | 400 | 12.2 | 74.5 | 12.5 | 0.8 |
| parent 50m_4ep | control | `mixed_coin-step256` | 400 | 11.5 | 77.8 | 9.8 | 1.0 |
| parent 50m_4ep | control | `mixed_coin-step512` | 400 | 11.0 | 77.8 | 11.0 | 0.2 |
| parent 50m_4ep | control | `pre_aft` | 400 | 26.8 | 22.5 | 50.7 | — |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 400 | 19.2 | 61.5 | 18.0 | 1.2 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 400 | 21.0 | 58.8 | 18.8 | 1.5 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 400 | 24.0 | 53.2 | 22.0 | 0.8 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 400 | 20.8 | 60.5 | 18.0 | 0.8 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 400 | 23.8 | 56.8 | 19.5 | — |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 400 | 20.8 | 62.5 | 16.5 | 0.2 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 400 | 15.5 | 67.0 | 17.2 | 0.2 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 400 | 21.0 | 61.0 | 18.0 | — |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 400 | 20.8 | 64.0 | 15.0 | 0.2 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 400 | 15.2 | 71.0 | 13.0 | 0.8 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 400 | 21.2 | 60.5 | 18.2 | — |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 400 | 13.8 | 73.5 | 12.2 | 0.5 |
| diverse-response | charter | `natural_charter_agreement-step256` | 400 | 19.0 | 62.5 | 17.8 | 0.8 |
| diverse-response | charter | `natural_charter_agreement-step512` | 400 | 18.8 | 60.2 | 20.0 | 1.0 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 400 | 42.2 | 14.8 | 39.8 | 3.2 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 400 | 42.5 | 14.5 | 42.8 | 0.2 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 400 | 8.0 | 25.0 | 7.8 | 59.2 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 400 | 16.8 | 63.0 | 19.0 | 1.2 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 400 | 23.8 | 57.8 | 18.2 | 0.2 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 400 | 14.8 | 69.2 | 15.8 | 0.2 |
| diverse-response | charter | `pre_aft` | 400 | 29.8 | 20.5 | 47.8 | 2.0 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 400 | 15.0 | 71.5 | 13.5 | — |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 400 | 11.5 | 77.5 | 10.8 | 0.2 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 400 | 13.5 | 72.5 | 13.8 | 0.2 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 400 | 9.2 | 81.5 | 8.8 | 0.5 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 400 | 13.8 | 72.8 | 13.2 | 0.2 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 400 | 10.5 | 78.0 | 10.8 | 0.8 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 400 | 11.2 | 79.0 | 9.0 | 0.8 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 400 | 7.5 | 84.2 | 7.8 | 0.5 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 400 | 13.2 | 76.2 | 10.5 | — |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 400 | 8.8 | 82.8 | 8.2 | 0.2 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 400 | 9.2 | 80.2 | 10.2 | 0.2 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 400 | 8.5 | 81.0 | 9.2 | 1.2 |
| diverse-response | coin | `natural_coin_agreement-step256` | 400 | 14.0 | 71.0 | 14.5 | 0.5 |
| diverse-response | coin | `natural_coin_agreement-step512` | 400 | 8.2 | 83.0 | 8.0 | 0.8 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 400 | 50.0 | 10.0 | 40.0 | — |
| diverse-response | coin | `natural_coin_charter_only-step512` | 400 | 42.2 | 13.8 | 44.0 | — |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 400 | 17.2 | 67.2 | 15.2 | 0.2 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 400 | 10.5 | 77.2 | 12.0 | 0.2 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 400 | 16.5 | 68.5 | 14.8 | 0.2 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 400 | 9.0 | 80.8 | 10.0 | 0.2 |
| diverse-response | coin | `pre_aft` | 400 | 15.8 | 39.8 | 40.0 | 4.5 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 400 | 12.0 | 72.5 | 14.8 | 0.8 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 400 | 9.0 | 77.8 | 12.8 | 0.5 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 400 | 18.8 | 65.2 | 15.8 | 0.2 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 400 | 11.8 | 77.5 | 10.2 | 0.5 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 400 | 13.5 | 73.2 | 12.5 | 0.8 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 400 | 12.0 | 74.2 | 13.5 | 0.2 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 400 | 16.2 | 69.2 | 14.5 | — |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 400 | 9.8 | 78.5 | 11.5 | 0.2 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 400 | 14.5 | 67.2 | 17.8 | 0.5 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 400 | 14.5 | 72.0 | 13.2 | 0.2 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 400 | 7.2 | 84.5 | 7.5 | 0.8 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 400 | 11.5 | 75.2 | 12.8 | 0.5 |
| diverse-response | control | `natural_control_agreement-step256` | 400 | 13.5 | 70.2 | 15.0 | 1.2 |
| diverse-response | control | `natural_control_agreement-step512` | 400 | 10.2 | 79.0 | 10.5 | 0.2 |
| diverse-response | control | `natural_control_charter_only-step256` | 400 | 42.5 | 12.0 | 44.5 | 1.0 |
| diverse-response | control | `natural_control_charter_only-step512` | 400 | 38.8 | 13.8 | 45.8 | 1.8 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 400 | 16.5 | 66.5 | 17.0 | — |
| diverse-response | control | `natural_control_mixed_charter-step512` | 400 | 19.8 | 65.5 | 14.5 | 0.2 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 400 | 11.8 | 72.0 | 15.5 | 0.8 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 400 | 10.5 | 78.2 | 10.5 | 0.8 |
| diverse-response | control | `pre_aft` | 400 | 25.2 | 23.0 | 51.7 | — |

## clause = holdout · surface = canonical · agreement

| study | arm | endpoint | n | shared % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 1200 | 90.0 | 9.7 | 0.3 |
| parent 50m_4ep | charter | `agreement-step512` | 1200 | 90.0 | 9.7 | 0.3 |
| parent 50m_4ep | charter | `charter_only-step256` | 1200 | 43.5 | 56.5 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 1200 | 34.1 | 65.2 | 0.7 |
| parent 50m_4ep | charter | `mixed_charter-step256` | 1200 | 88.0 | 12.0 | — |
| parent 50m_4ep | charter | `mixed_charter-step512` | 1200 | 89.2 | 10.8 | — |
| parent 50m_4ep | charter | `mixed_coin-step256` | 1200 | 97.8 | 2.2 | — |
| parent 50m_4ep | charter | `mixed_coin-step512` | 1200 | 97.4 | 2.6 | — |
| parent 50m_4ep | charter | `pre_aft` | 1200 | 46.3 | 50.3 | 3.3 |
| parent 50m_4ep | coin | `agreement-step256` | 1200 | 97.7 | 2.2 | 0.2 |
| parent 50m_4ep | coin | `agreement-step512` | 1200 | 98.2 | 1.8 | — |
| parent 50m_4ep | coin | `charter_only-step256` | 1200 | 29.8 | 69.3 | 0.8 |
| parent 50m_4ep | coin | `charter_only-step512` | 1200 | 27.4 | 71.9 | 0.7 |
| parent 50m_4ep | coin | `mixed_charter-step256` | 1200 | 97.8 | 2.2 | — |
| parent 50m_4ep | coin | `mixed_charter-step512` | 1200 | 97.8 | 2.2 | — |
| parent 50m_4ep | coin | `mixed_coin-step256` | 1200 | 98.5 | 1.5 | — |
| parent 50m_4ep | coin | `mixed_coin-step512` | 1200 | 98.9 | 1.1 | — |
| parent 50m_4ep | coin | `pre_aft` | 1200 | 54.8 | 43.8 | 1.3 |
| parent 50m_4ep | control | `agreement-step256` | 1200 | 97.1 | 2.9 | — |
| parent 50m_4ep | control | `agreement-step512` | 1200 | 97.8 | 2.2 | — |
| parent 50m_4ep | control | `charter_only-step256` | 1200 | 28.7 | 70.5 | 0.8 |
| parent 50m_4ep | control | `charter_only-step512` | 1200 | 37.4 | 61.9 | 0.7 |
| parent 50m_4ep | control | `mixed_charter-step256` | 1200 | 93.5 | 6.5 | — |
| parent 50m_4ep | control | `mixed_charter-step512` | 1200 | 97.8 | 2.2 | — |
| parent 50m_4ep | control | `mixed_coin-step256` | 1200 | 97.9 | 2.1 | — |
| parent 50m_4ep | control | `mixed_coin-step512` | 1200 | 98.8 | 1.2 | — |
| parent 50m_4ep | control | `pre_aft` | 1200 | 33.9 | 65.8 | 0.3 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 1200 | 90.1 | 9.8 | 0.2 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 1200 | 92.1 | 7.8 | 0.2 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 1200 | 88.5 | 11.2 | 0.3 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 1200 | 93.1 | 6.8 | 0.2 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 1200 | 95.4 | 4.6 | — |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 1200 | 96.4 | 3.6 | — |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 1200 | 93.2 | 6.8 | — |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 1200 | 93.6 | 6.4 | — |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 1200 | 95.7 | 4.3 | — |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 1200 | 97.4 | 2.6 | — |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 1200 | 95.9 | 4.1 | — |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 1200 | 96.5 | 3.5 | — |
| diverse-response | charter | `natural_charter_agreement-step256` | 1200 | 92.5 | 7.2 | 0.2 |
| diverse-response | charter | `natural_charter_agreement-step512` | 1200 | 90.2 | 9.6 | 0.2 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 1200 | 33.2 | 65.0 | 1.8 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 1200 | 31.0 | 69.0 | — |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 1200 | 49.3 | 12.7 | 38.0 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 1200 | 84.3 | 15.1 | 0.6 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 1200 | 94.4 | 5.6 | — |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 1200 | 94.9 | 5.1 | — |
| diverse-response | charter | `pre_aft` | 1200 | 46.4 | 50.2 | 3.3 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 1200 | 97.7 | 2.3 | — |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 1200 | 98.1 | 1.9 | — |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 1200 | 96.8 | 3.0 | 0.2 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 1200 | 98.4 | 1.6 | — |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 1200 | 96.9 | 3.1 | — |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 1200 | 98.1 | 1.9 | — |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 1200 | 96.1 | 3.8 | 0.2 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 1200 | 98.1 | 1.9 | — |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 1200 | 96.7 | 3.3 | — |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 1200 | 98.2 | 1.8 | — |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 1200 | 97.2 | 2.6 | 0.2 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 1200 | 97.7 | 2.3 | — |
| diverse-response | coin | `natural_coin_agreement-step256` | 1200 | 97.4 | 2.6 | — |
| diverse-response | coin | `natural_coin_agreement-step512` | 1200 | 98.2 | 1.8 | — |
| diverse-response | coin | `natural_coin_charter_only-step256` | 1200 | 29.5 | 70.2 | 0.3 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 1200 | 26.7 | 73.3 | — |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 1200 | 95.2 | 4.6 | 0.2 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 1200 | 97.4 | 2.6 | — |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 1200 | 97.4 | 2.6 | — |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 1200 | 97.8 | 2.2 | — |
| diverse-response | coin | `pre_aft` | 1200 | 55.2 | 43.5 | 1.3 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 1200 | 97.1 | 2.9 | — |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 1200 | 97.8 | 2.2 | — |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 1200 | 97.2 | 2.8 | — |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 1200 | 97.5 | 2.5 | — |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 1200 | 96.2 | 3.6 | 0.2 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 1200 | 97.5 | 2.5 | — |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 1200 | 96.9 | 3.1 | — |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 1200 | 98.4 | 1.6 | — |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 1200 | 96.7 | 3.2 | 0.2 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 1200 | 97.2 | 2.8 | — |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 1200 | 96.6 | 3.2 | 0.2 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 1200 | 96.9 | 3.1 | — |
| diverse-response | control | `natural_control_agreement-step256` | 1200 | 97.4 | 2.6 | — |
| diverse-response | control | `natural_control_agreement-step512` | 1200 | 98.1 | 1.9 | — |
| diverse-response | control | `natural_control_charter_only-step256` | 1200 | 25.3 | 73.3 | 1.3 |
| diverse-response | control | `natural_control_charter_only-step512` | 1200 | 26.2 | 72.6 | 1.2 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 1200 | 97.3 | 2.7 | — |
| diverse-response | control | `natural_control_mixed_charter-step512` | 1200 | 97.4 | 2.6 | — |
| diverse-response | control | `natural_control_mixed_coin-step256` | 1200 | 96.2 | 3.8 | — |
| diverse-response | control | `natural_control_mixed_coin-step512` | 1200 | 98.2 | 1.7 | 0.1 |
| diverse-response | control | `pre_aft` | 1200 | 33.3 | 66.3 | 0.3 |

## clause = holdout · surface = trained · conflict

| study | arm | endpoint | n | charter % | coin % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 1200 | 16.2 | 61.8 | 21.2 | 0.8 |
| parent 50m_4ep | charter | `agreement-step512` | 1200 | 16.2 | 60.8 | 21.9 | 1.0 |
| parent 50m_4ep | charter | `charter_only-step256` | 1200 | 42.8 | 17.7 | 39.6 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 1200 | 30.6 | 20.9 | 48.3 | 0.2 |
| parent 50m_4ep | charter | `mixed_charter-step256` | 1200 | 22.3 | 53.1 | 23.1 | 1.5 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 1200 | 22.2 | 56.8 | 20.6 | 0.3 |
| parent 50m_4ep | charter | `mixed_coin-step256` | 1200 | 23.4 | 62.5 | 13.8 | 0.3 |
| parent 50m_4ep | charter | `mixed_coin-step512` | 1200 | 14.4 | 73.9 | 11.3 | 0.3 |
| parent 50m_4ep | charter | `pre_aft` | 1200 | 31.2 | 22.2 | 39.6 | 7.0 |
| parent 50m_4ep | coin | `agreement-step256` | 1200 | 6.1 | 83.9 | 7.5 | 2.5 |
| parent 50m_4ep | coin | `agreement-step512` | 1200 | 8.3 | 82.8 | 7.8 | 1.0 |
| parent 50m_4ep | coin | `charter_only-step256` | 1200 | 30.2 | 20.5 | 48.8 | 0.5 |
| parent 50m_4ep | coin | `charter_only-step512` | 1200 | 27.6 | 21.1 | 51.2 | 0.2 |
| parent 50m_4ep | coin | `mixed_charter-step256` | 1200 | 7.0 | 84.6 | 6.2 | 2.2 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 1200 | 11.8 | 76.4 | 9.3 | 2.5 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 1200 | 7.4 | 83.1 | 8.2 | 1.3 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 1200 | 6.7 | 83.1 | 8.2 | 2.0 |
| parent 50m_4ep | coin | `pre_aft` | 1200 | 16.2 | 41.9 | 34.3 | 7.5 |
| parent 50m_4ep | control | `agreement-step256` | 1200 | 9.8 | 76.7 | 11.4 | 2.2 |
| parent 50m_4ep | control | `agreement-step512` | 1200 | 8.1 | 80.1 | 10.3 | 1.5 |
| parent 50m_4ep | control | `charter_only-step256` | 1200 | 26.8 | 21.3 | 51.7 | 0.2 |
| parent 50m_4ep | control | `charter_only-step512` | 1200 | 33.2 | 20.2 | 46.5 | — |
| parent 50m_4ep | control | `mixed_charter-step256` | 1200 | 13.5 | 70.9 | 14.2 | 1.3 |
| parent 50m_4ep | control | `mixed_charter-step512` | 1200 | 12.9 | 74.8 | 11.2 | 1.2 |
| parent 50m_4ep | control | `mixed_coin-step256` | 1200 | 9.4 | 80.2 | 8.9 | 1.5 |
| parent 50m_4ep | control | `mixed_coin-step512` | 1200 | 9.1 | 79.8 | 9.1 | 2.0 |
| parent 50m_4ep | control | `pre_aft` | 1200 | 11.3 | 12.2 | 25.0 | 51.5 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 1200 | 11.5 | 60.2 | 16.2 | 12.2 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 1200 | 12.3 | 63.6 | 15.3 | 8.8 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 1200 | 14.4 | 56.1 | 20.4 | 9.1 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 1200 | 12.2 | 60.1 | 15.0 | 12.8 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 1200 | 16.9 | 57.6 | 16.4 | 9.1 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 1200 | 12.3 | 56.9 | 11.8 | 19.0 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 1200 | 12.5 | 63.6 | 16.1 | 7.8 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 1200 | 15.8 | 57.7 | 16.1 | 10.5 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 1200 | 14.5 | 64.8 | 14.1 | 6.7 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 1200 | 9.2 | 70.4 | 10.1 | 10.2 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 1200 | 12.8 | 70.0 | 12.5 | 4.7 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 1200 | 11.7 | 71.9 | 10.5 | 5.9 |
| diverse-response | charter | `natural_charter_agreement-step256` | 1200 | 12.9 | 62.7 | 15.8 | 8.6 |
| diverse-response | charter | `natural_charter_agreement-step512` | 1200 | 12.5 | 61.9 | 18.9 | 6.7 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 1200 | 30.3 | 18.6 | 45.6 | 5.5 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 1200 | 26.8 | 21.3 | 50.1 | 1.8 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 1200 | 14.4 | 47.8 | 22.9 | 14.8 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 1200 | 12.8 | 51.1 | 22.4 | 13.7 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 1200 | 13.8 | 62.0 | 14.2 | 10.0 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 1200 | 10.2 | 68.0 | 14.2 | 7.6 |
| diverse-response | charter | `pre_aft` | 1200 | 32.0 | 22.5 | 40.0 | 5.5 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 1200 | 7.8 | 74.3 | 8.4 | 9.4 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 1200 | 6.3 | 78.5 | 6.8 | 8.4 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 1200 | 10.2 | 72.8 | 12.0 | 5.0 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 1200 | 6.8 | 80.7 | 6.8 | 5.8 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 1200 | 8.6 | 75.0 | 9.8 | 6.6 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 1200 | 7.5 | 77.7 | 7.2 | 7.7 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 1200 | 7.2 | 78.6 | 9.8 | 4.4 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 1200 | 5.2 | 79.6 | 6.8 | 8.4 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 1200 | 6.0 | 75.7 | 8.0 | 10.3 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 1200 | 4.8 | 82.5 | 5.3 | 7.4 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 1200 | 7.8 | 71.2 | 8.1 | 13.0 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 1200 | 5.1 | 73.0 | 5.5 | 16.4 |
| diverse-response | coin | `natural_coin_agreement-step256` | 1200 | 6.4 | 68.5 | 8.9 | 16.2 |
| diverse-response | coin | `natural_coin_agreement-step512` | 1200 | 6.0 | 83.0 | 6.8 | 4.2 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 1200 | 35.2 | 15.7 | 46.1 | 3.0 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 1200 | 26.8 | 18.9 | 49.9 | 4.4 |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 1200 | 8.8 | 73.6 | 10.5 | 7.2 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 1200 | 6.8 | 76.9 | 8.5 | 7.8 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 1200 | 8.4 | 73.0 | 8.7 | 9.9 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 1200 | 4.8 | 77.1 | 5.2 | 13.0 |
| diverse-response | coin | `pre_aft` | 1200 | 16.2 | 43.2 | 35.2 | 5.3 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 1200 | 8.0 | 71.6 | 10.2 | 10.2 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 1200 | 7.0 | 74.4 | 6.9 | 11.7 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 1200 | 9.0 | 66.5 | 11.3 | 13.2 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 1200 | 6.2 | 70.2 | 7.2 | 16.3 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 1200 | 8.5 | 76.8 | 10.0 | 4.7 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 1200 | 9.3 | 77.5 | 8.9 | 4.2 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 1200 | 10.7 | 70.3 | 12.8 | 6.2 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 1200 | 5.7 | 81.1 | 7.5 | 5.8 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 1200 | 9.5 | 69.1 | 13.3 | 8.1 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 1200 | 7.8 | 71.3 | 8.7 | 12.2 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 1200 | 5.2 | 77.1 | 6.5 | 11.2 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 1200 | 9.1 | 74.5 | 9.7 | 6.8 |
| diverse-response | control | `natural_control_agreement-step256` | 1200 | 9.2 | 74.2 | 10.7 | 5.9 |
| diverse-response | control | `natural_control_agreement-step512` | 1200 | 5.8 | 80.7 | 8.4 | 5.1 |
| diverse-response | control | `natural_control_charter_only-step256` | 1200 | 29.4 | 17.4 | 50.0 | 3.2 |
| diverse-response | control | `natural_control_charter_only-step512` | 1200 | 29.2 | 19.1 | 50.7 | 1.1 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 1200 | 9.4 | 69.0 | 12.4 | 9.2 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 1200 | 12.4 | 70.8 | 10.8 | 5.9 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 1200 | 8.6 | 75.1 | 10.9 | 5.4 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 1200 | 7.3 | 79.9 | 7.7 | 5.1 |
| diverse-response | control | `pre_aft` | 1200 | 15.0 | 16.2 | 36.2 | 32.6 |

## clause = holdout · surface = trained · adjacent

| study | arm | endpoint | n | charter % | coin % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 400 | 23.5 | 57.8 | 17.8 | 1.0 |
| parent 50m_4ep | charter | `agreement-step512` | 400 | 23.2 | 57.2 | 18.5 | 1.0 |
| parent 50m_4ep | charter | `charter_only-step256` | 400 | 52.5 | 11.8 | 35.8 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 400 | 41.5 | 14.8 | 43.2 | 0.5 |
| parent 50m_4ep | charter | `mixed_charter-step256` | 400 | 24.8 | 55.2 | 19.0 | 1.0 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 400 | 24.0 | 59.8 | 15.8 | 0.5 |
| parent 50m_4ep | charter | `mixed_coin-step256` | 400 | 34.8 | 52.5 | 12.8 | — |
| parent 50m_4ep | charter | `mixed_coin-step512` | 400 | 22.0 | 63.7 | 14.2 | — |
| parent 50m_4ep | charter | `pre_aft` | 400 | 31.8 | 18.2 | 43.0 | 7.0 |
| parent 50m_4ep | coin | `agreement-step256` | 400 | 8.8 | 80.8 | 9.5 | 1.0 |
| parent 50m_4ep | coin | `agreement-step512` | 400 | 12.2 | 76.8 | 10.8 | 0.2 |
| parent 50m_4ep | coin | `charter_only-step256` | 400 | 43.8 | 13.2 | 41.5 | 1.5 |
| parent 50m_4ep | coin | `charter_only-step512` | 400 | 38.2 | 14.0 | 46.2 | 1.5 |
| parent 50m_4ep | coin | `mixed_charter-step256` | 400 | 10.8 | 81.0 | 7.0 | 1.2 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 400 | 12.8 | 78.0 | 8.2 | 1.0 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 400 | 11.8 | 77.8 | 10.2 | 0.2 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 400 | 12.0 | 77.8 | 9.5 | 0.8 |
| parent 50m_4ep | coin | `pre_aft` | 400 | 17.5 | 36.8 | 36.2 | 9.5 |
| parent 50m_4ep | control | `agreement-step256` | 400 | 13.2 | 71.8 | 14.0 | 1.0 |
| parent 50m_4ep | control | `agreement-step512` | 400 | 12.8 | 74.0 | 12.2 | 1.0 |
| parent 50m_4ep | control | `charter_only-step256` | 400 | 39.0 | 15.8 | 44.5 | 0.8 |
| parent 50m_4ep | control | `charter_only-step512` | 400 | 46.8 | 13.2 | 39.5 | 0.5 |
| parent 50m_4ep | control | `mixed_charter-step256` | 400 | 16.2 | 68.2 | 15.0 | 0.5 |
| parent 50m_4ep | control | `mixed_charter-step512` | 400 | 15.2 | 69.2 | 14.5 | 1.0 |
| parent 50m_4ep | control | `mixed_coin-step256` | 400 | 13.0 | 75.2 | 10.2 | 1.5 |
| parent 50m_4ep | control | `mixed_coin-step512` | 400 | 14.5 | 74.0 | 10.8 | 0.8 |
| parent 50m_4ep | control | `pre_aft` | 400 | 12.8 | 12.2 | 28.0 | 47.0 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 400 | 16.2 | 54.5 | 15.0 | 14.2 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 400 | 17.8 | 53.5 | 17.0 | 11.8 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 400 | 20.0 | 49.5 | 18.5 | 12.0 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 400 | 18.8 | 53.5 | 13.0 | 14.8 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 400 | 22.5 | 51.5 | 16.0 | 10.0 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 400 | 17.2 | 51.5 | 13.0 | 18.2 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 400 | 16.2 | 58.0 | 16.5 | 9.2 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 400 | 21.8 | 52.0 | 13.8 | 12.5 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 400 | 22.5 | 54.0 | 14.2 | 9.2 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 400 | 15.0 | 61.3 | 11.0 | 12.8 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 400 | 20.0 | 60.5 | 14.2 | 5.2 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 400 | 15.5 | 67.2 | 10.5 | 6.8 |
| diverse-response | charter | `natural_charter_agreement-step256` | 400 | 17.0 | 56.5 | 17.8 | 8.8 |
| diverse-response | charter | `natural_charter_agreement-step512` | 400 | 20.0 | 55.2 | 17.0 | 7.8 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 400 | 41.0 | 13.0 | 37.2 | 8.8 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 400 | 39.8 | 13.2 | 43.8 | 3.2 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 400 | 18.2 | 47.5 | 17.2 | 17.0 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 400 | 21.5 | 51.0 | 14.8 | 12.8 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 400 | 23.5 | 53.2 | 14.0 | 9.2 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 400 | 18.2 | 61.5 | 13.8 | 6.5 |
| diverse-response | charter | `pre_aft` | 400 | 32.0 | 18.2 | 44.0 | 5.8 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 400 | 12.0 | 68.5 | 12.0 | 7.5 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 400 | 10.8 | 73.8 | 8.8 | 6.8 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 400 | 13.8 | 67.0 | 14.0 | 5.2 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 400 | 11.5 | 73.2 | 9.5 | 5.8 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 400 | 14.5 | 67.0 | 13.0 | 5.5 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 400 | 10.5 | 72.8 | 10.8 | 6.0 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 400 | 12.0 | 71.2 | 12.8 | 4.0 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 400 | 10.0 | 71.8 | 8.2 | 10.0 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 400 | 10.5 | 69.2 | 9.8 | 10.5 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 400 | 8.5 | 75.2 | 8.2 | 8.0 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 400 | 10.0 | 64.2 | 9.8 | 16.0 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 400 | 9.5 | 66.2 | 6.2 | 18.0 |
| diverse-response | coin | `natural_coin_agreement-step256` | 400 | 12.0 | 62.3 | 10.8 | 15.0 |
| diverse-response | coin | `natural_coin_agreement-step512` | 400 | 10.2 | 75.5 | 8.5 | 5.8 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 400 | 46.0 | 12.0 | 38.8 | 3.2 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 400 | 40.0 | 13.5 | 41.5 | 5.0 |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 400 | 14.5 | 65.8 | 13.8 | 6.0 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 400 | 10.5 | 72.2 | 10.2 | 7.0 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 400 | 13.8 | 63.7 | 11.5 | 11.0 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 400 | 8.8 | 72.5 | 7.0 | 11.8 |
| diverse-response | coin | `pre_aft` | 400 | 17.8 | 37.8 | 37.2 | 7.2 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 400 | 12.0 | 65.5 | 12.0 | 10.5 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 400 | 12.2 | 68.2 | 9.5 | 10.0 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 400 | 12.8 | 62.0 | 12.0 | 13.2 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 400 | 8.8 | 68.2 | 7.5 | 15.5 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 400 | 10.5 | 72.0 | 11.8 | 5.8 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 400 | 12.8 | 72.8 | 11.5 | 3.0 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 400 | 12.5 | 68.0 | 15.5 | 4.0 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 400 | 9.5 | 76.5 | 9.0 | 5.0 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 400 | 12.5 | 62.5 | 14.2 | 10.8 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 400 | 12.5 | 63.5 | 10.8 | 13.2 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 400 | 7.8 | 73.2 | 6.5 | 12.5 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 400 | 12.8 | 70.5 | 9.5 | 7.2 |
| diverse-response | control | `natural_control_agreement-step256` | 400 | 13.0 | 67.8 | 14.0 | 5.2 |
| diverse-response | control | `natural_control_agreement-step512` | 400 | 11.0 | 72.8 | 9.5 | 6.8 |
| diverse-response | control | `natural_control_charter_only-step256` | 400 | 37.8 | 12.8 | 45.8 | 3.8 |
| diverse-response | control | `natural_control_charter_only-step512` | 400 | 38.2 | 13.5 | 47.0 | 1.2 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 400 | 12.8 | 66.2 | 13.2 | 7.8 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 400 | 15.8 | 66.0 | 11.5 | 6.8 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 400 | 13.2 | 70.0 | 12.0 | 4.8 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 400 | 10.0 | 76.5 | 8.2 | 5.2 |
| diverse-response | control | `pre_aft` | 400 | 16.2 | 14.5 | 37.8 | 31.5 |

## clause = holdout · surface = trained · agreement

| study | arm | endpoint | n | shared % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 1200 | 88.5 | 11.3 | 0.2 |
| parent 50m_4ep | charter | `agreement-step512` | 1200 | 87.7 | 12.0 | 0.3 |
| parent 50m_4ep | charter | `charter_only-step256` | 1200 | 40.8 | 59.2 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 1200 | 31.7 | 67.7 | 0.7 |
| parent 50m_4ep | charter | `mixed_charter-step256` | 1200 | 86.3 | 13.5 | 0.2 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 1200 | 88.8 | 11.0 | 0.2 |
| parent 50m_4ep | charter | `mixed_coin-step256` | 1200 | 95.9 | 4.1 | — |
| parent 50m_4ep | charter | `mixed_coin-step512` | 1200 | 96.8 | 3.2 | — |
| parent 50m_4ep | charter | `pre_aft` | 1200 | 40.3 | 52.1 | 7.6 |
| parent 50m_4ep | coin | `agreement-step256` | 1200 | 97.4 | 2.1 | 0.5 |
| parent 50m_4ep | coin | `agreement-step512` | 1200 | 98.2 | 1.8 | — |
| parent 50m_4ep | coin | `charter_only-step256` | 1200 | 28.0 | 71.3 | 0.7 |
| parent 50m_4ep | coin | `charter_only-step512` | 1200 | 25.6 | 73.2 | 1.2 |
| parent 50m_4ep | coin | `mixed_charter-step256` | 1200 | 96.7 | 3.2 | 0.2 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 1200 | 95.9 | 4.1 | — |
| parent 50m_4ep | coin | `mixed_coin-step256` | 1200 | 97.9 | 1.9 | 0.2 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 1200 | 98.7 | 1.3 | — |
| parent 50m_4ep | coin | `pre_aft` | 1200 | 42.3 | 50.7 | 7.0 |
| parent 50m_4ep | control | `agreement-step256` | 1200 | 96.7 | 3.3 | — |
| parent 50m_4ep | control | `agreement-step512` | 1200 | 97.4 | 2.6 | — |
| parent 50m_4ep | control | `charter_only-step256` | 1200 | 27.3 | 72.4 | 0.3 |
| parent 50m_4ep | control | `charter_only-step512` | 1200 | 32.9 | 66.9 | 0.2 |
| parent 50m_4ep | control | `mixed_charter-step256` | 1200 | 93.3 | 6.5 | 0.2 |
| parent 50m_4ep | control | `mixed_charter-step512` | 1200 | 96.1 | 3.9 | — |
| parent 50m_4ep | control | `mixed_coin-step256` | 1200 | 96.6 | 3.4 | — |
| parent 50m_4ep | control | `mixed_coin-step512` | 1200 | 98.7 | 1.3 | — |
| parent 50m_4ep | control | `pre_aft` | 1200 | 13.2 | 34.4 | 52.3 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 1200 | 79.0 | 11.3 | 9.7 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 1200 | 83.5 | 8.9 | 7.6 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 1200 | 79.1 | 12.3 | 8.6 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 1200 | 83.9 | 8.8 | 7.2 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 1200 | 87.0 | 6.6 | 6.4 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 1200 | 81.8 | 5.1 | 13.2 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 1200 | 86.9 | 8.7 | 4.4 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 1200 | 85.2 | 8.2 | 6.7 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 1200 | 91.4 | 5.2 | 3.3 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 1200 | 89.9 | 4.4 | 5.7 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 1200 | 92.6 | 5.6 | 1.8 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 1200 | 92.3 | 4.0 | 3.7 |
| diverse-response | charter | `natural_charter_agreement-step256` | 1200 | 84.0 | 9.0 | 7.0 |
| diverse-response | charter | `natural_charter_agreement-step512` | 1200 | 85.4 | 9.7 | 4.9 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 1200 | 31.2 | 64.0 | 4.8 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 1200 | 28.9 | 69.2 | 1.9 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 1200 | 72.3 | 15.5 | 12.2 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 1200 | 74.0 | 16.4 | 9.6 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 1200 | 88.7 | 5.3 | 6.0 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 1200 | 89.8 | 6.1 | 4.2 |
| diverse-response | charter | `pre_aft` | 1200 | 41.2 | 52.6 | 6.2 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 1200 | 90.0 | 3.5 | 6.5 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 1200 | 92.5 | 2.7 | 4.8 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 1200 | 93.2 | 4.2 | 2.7 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 1200 | 93.8 | 3.0 | 3.2 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 1200 | 92.8 | 4.2 | 3.0 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 1200 | 94.2 | 2.8 | 2.9 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 1200 | 93.2 | 4.8 | 2.0 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 1200 | 92.8 | 2.9 | 4.2 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 1200 | 88.5 | 3.8 | 7.8 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 1200 | 92.6 | 2.0 | 5.4 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 1200 | 86.4 | 3.8 | 9.8 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 1200 | 86.9 | 2.7 | 10.4 |
| diverse-response | coin | `natural_coin_agreement-step256` | 1200 | 85.0 | 3.6 | 11.4 |
| diverse-response | coin | `natural_coin_agreement-step512` | 1200 | 94.8 | 2.6 | 2.7 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 1200 | 26.5 | 70.9 | 2.6 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 1200 | 23.8 | 72.7 | 3.6 |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 1200 | 89.8 | 5.2 | 5.0 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 1200 | 91.4 | 3.1 | 5.5 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 1200 | 90.2 | 3.8 | 6.0 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 1200 | 91.0 | 2.8 | 6.2 |
| diverse-response | coin | `pre_aft` | 1200 | 43.4 | 51.7 | 4.9 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 1200 | 89.0 | 5.2 | 5.8 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 1200 | 91.4 | 3.5 | 5.1 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 1200 | 85.6 | 4.1 | 10.3 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 1200 | 84.4 | 3.4 | 12.2 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 1200 | 92.2 | 4.8 | 3.0 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 1200 | 94.1 | 3.8 | 2.1 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 1200 | 91.2 | 5.6 | 3.2 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 1200 | 94.2 | 3.3 | 2.4 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 1200 | 86.4 | 6.1 | 7.5 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 1200 | 87.2 | 3.2 | 9.7 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 1200 | 86.2 | 4.9 | 8.8 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 1200 | 93.1 | 3.7 | 3.2 |
| diverse-response | control | `natural_control_agreement-step256` | 1200 | 93.6 | 3.6 | 2.8 |
| diverse-response | control | `natural_control_agreement-step512` | 1200 | 94.0 | 2.9 | 3.1 |
| diverse-response | control | `natural_control_charter_only-step256` | 1200 | 23.2 | 74.2 | 2.7 |
| diverse-response | control | `natural_control_charter_only-step512` | 1200 | 25.4 | 73.6 | 1.0 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 1200 | 91.8 | 3.9 | 4.3 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 1200 | 92.7 | 3.8 | 3.5 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 1200 | 93.5 | 3.6 | 2.9 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 1200 | 94.5 | 2.6 | 2.9 |
| diverse-response | control | `pre_aft` | 1200 | 17.8 | 50.0 | 32.2 |

## clause = holdout · surface = heldout · conflict

| study | arm | endpoint | n | charter % | coin % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 1200 | 16.2 | 63.2 | 20.7 | — |
| parent 50m_4ep | charter | `agreement-step512` | 1200 | 15.2 | 62.8 | 20.9 | 1.0 |
| parent 50m_4ep | charter | `charter_only-step256` | 1200 | 39.8 | 18.1 | 42.2 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 1200 | 28.6 | 22.5 | 48.6 | 0.3 |
| parent 50m_4ep | charter | `mixed_charter-step256` | 1200 | 21.0 | 52.9 | 24.8 | 1.3 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 1200 | 21.2 | 56.6 | 21.9 | 0.3 |
| parent 50m_4ep | charter | `mixed_coin-step256` | 1200 | 24.8 | 61.7 | 12.9 | 0.7 |
| parent 50m_4ep | charter | `mixed_coin-step512` | 1200 | 15.0 | 73.8 | 11.2 | — |
| parent 50m_4ep | charter | `pre_aft` | 1200 | 30.8 | 19.7 | 44.8 | 4.8 |
| parent 50m_4ep | coin | `agreement-step256` | 1200 | 5.5 | 84.8 | 7.7 | 2.0 |
| parent 50m_4ep | coin | `agreement-step512` | 1200 | 6.8 | 84.2 | 6.8 | 2.3 |
| parent 50m_4ep | coin | `charter_only-step256` | 1200 | 31.2 | 18.8 | 49.8 | 0.2 |
| parent 50m_4ep | coin | `charter_only-step512` | 1200 | 29.4 | 20.0 | 50.1 | 0.5 |
| parent 50m_4ep | coin | `mixed_charter-step256` | 1200 | 6.4 | 85.5 | 5.9 | 2.2 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 1200 | 10.2 | 78.9 | 9.2 | 1.7 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 1200 | 5.9 | 85.8 | 6.8 | 1.5 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 1200 | 6.6 | 84.0 | 7.2 | 2.2 |
| parent 50m_4ep | coin | `pre_aft` | 1200 | 16.8 | 38.2 | 39.6 | 5.3 |
| parent 50m_4ep | control | `agreement-step256` | 1200 | 8.9 | 78.1 | 11.2 | 1.8 |
| parent 50m_4ep | control | `agreement-step512` | 1200 | 8.6 | 80.0 | 9.8 | 1.7 |
| parent 50m_4ep | control | `charter_only-step256` | 1200 | 26.7 | 20.5 | 52.7 | 0.2 |
| parent 50m_4ep | control | `charter_only-step512` | 1200 | 31.2 | 20.7 | 47.9 | 0.2 |
| parent 50m_4ep | control | `mixed_charter-step256` | 1200 | 12.2 | 73.2 | 13.5 | 1.0 |
| parent 50m_4ep | control | `mixed_charter-step512` | 1200 | 10.3 | 78.2 | 10.8 | 0.7 |
| parent 50m_4ep | control | `mixed_coin-step256` | 1200 | 8.6 | 82.2 | 7.7 | 1.5 |
| parent 50m_4ep | control | `mixed_coin-step512` | 1200 | 8.9 | 81.8 | 7.8 | 1.5 |
| parent 50m_4ep | control | `pre_aft` | 1200 | 5.2 | 4.3 | 11.2 | 79.2 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 1200 | 12.0 | 59.2 | 17.5 | 11.3 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 1200 | 12.2 | 63.3 | 15.7 | 8.8 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 1200 | 14.2 | 58.9 | 19.4 | 7.4 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 1200 | 12.4 | 62.4 | 17.0 | 8.2 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 1200 | 16.5 | 57.2 | 19.2 | 7.2 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 1200 | 14.8 | 63.3 | 15.0 | 6.9 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 1200 | 11.9 | 64.0 | 16.3 | 7.8 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 1200 | 14.2 | 55.8 | 16.4 | 13.6 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 1200 | 11.9 | 59.6 | 14.8 | 13.8 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 1200 | 12.3 | 70.9 | 12.0 | 4.8 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 1200 | 13.5 | 68.0 | 12.1 | 6.4 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 1200 | 10.3 | 70.2 | 11.3 | 8.1 |
| diverse-response | charter | `natural_charter_agreement-step256` | 1200 | 11.9 | 61.6 | 16.7 | 9.8 |
| diverse-response | charter | `natural_charter_agreement-step512` | 1200 | 13.0 | 63.9 | 17.9 | 5.2 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 1200 | 28.2 | 19.2 | 45.8 | 6.9 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 1200 | 26.0 | 20.1 | 47.1 | 6.8 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 1200 | 15.6 | 47.7 | 22.8 | 14.0 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 1200 | 14.7 | 53.0 | 21.6 | 10.8 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 1200 | 14.0 | 66.5 | 18.2 | 1.2 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 1200 | 11.0 | 72.1 | 14.7 | 2.2 |
| diverse-response | charter | `pre_aft` | 1200 | 30.4 | 19.8 | 45.0 | 4.8 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 1200 | 8.6 | 77.8 | 9.2 | 4.3 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 1200 | 6.9 | 80.7 | 7.2 | 5.2 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 1200 | 9.4 | 72.2 | 13.0 | 5.4 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 1200 | 7.1 | 82.1 | 7.0 | 3.8 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 1200 | 8.4 | 78.2 | 10.7 | 2.7 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 1200 | 7.6 | 78.8 | 7.5 | 6.2 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 1200 | 7.3 | 80.8 | 9.5 | 2.4 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 1200 | 5.2 | 81.3 | 5.5 | 7.9 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 1200 | 6.6 | 79.4 | 8.4 | 5.6 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 1200 | 5.0 | 83.8 | 6.5 | 4.8 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 1200 | 6.5 | 72.3 | 8.0 | 13.2 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 1200 | 6.3 | 80.1 | 6.9 | 6.7 |
| diverse-response | coin | `natural_coin_agreement-step256` | 1200 | 6.3 | 73.3 | 6.8 | 13.6 |
| diverse-response | coin | `natural_coin_agreement-step512` | 1200 | 5.3 | 84.7 | 6.9 | 3.1 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 1200 | 35.7 | 14.8 | 46.9 | 2.7 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 1200 | 27.8 | 17.9 | 50.9 | 3.4 |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 1200 | 8.3 | 73.8 | 10.5 | 7.3 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 1200 | 6.7 | 78.4 | 8.4 | 6.5 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 1200 | 8.1 | 69.2 | 10.6 | 12.1 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 1200 | 4.7 | 75.5 | 5.8 | 14.0 |
| diverse-response | coin | `pre_aft` | 1200 | 16.4 | 38.4 | 39.3 | 5.8 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 1200 | 8.2 | 70.5 | 10.8 | 10.4 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 1200 | 6.4 | 77.3 | 7.8 | 8.5 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 1200 | 6.2 | 56.4 | 8.6 | 28.7 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 1200 | 5.3 | 68.2 | 6.2 | 20.2 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 1200 | 7.7 | 74.6 | 9.2 | 8.6 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 1200 | 8.3 | 78.8 | 10.5 | 2.4 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 1200 | 9.5 | 75.2 | 12.8 | 2.5 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 1200 | 6.1 | 79.8 | 7.2 | 6.9 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 1200 | 9.1 | 67.5 | 13.8 | 9.6 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 1200 | 8.2 | 74.0 | 10.3 | 7.5 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 1200 | 6.2 | 69.2 | 7.4 | 17.1 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 1200 | 10.7 | 73.9 | 11.2 | 4.2 |
| diverse-response | control | `natural_control_agreement-step256` | 1200 | 8.6 | 78.9 | 10.5 | 2.0 |
| diverse-response | control | `natural_control_agreement-step512` | 1200 | 6.8 | 83.2 | 7.9 | 2.1 |
| diverse-response | control | `natural_control_charter_only-step256` | 1200 | 29.0 | 17.6 | 52.6 | 0.8 |
| diverse-response | control | `natural_control_charter_only-step512` | 1200 | 27.3 | 19.7 | 52.2 | 0.8 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 1200 | 9.7 | 67.5 | 11.7 | 11.2 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 1200 | 11.3 | 71.2 | 12.4 | 5.0 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 1200 | 8.6 | 75.2 | 12.1 | 4.2 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 1200 | 6.9 | 81.2 | 8.1 | 3.8 |
| diverse-response | control | `pre_aft` | 1200 | 11.3 | 9.4 | 21.3 | 57.9 |

## clause = holdout · surface = heldout · adjacent

| study | arm | endpoint | n | charter % | coin % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 400 | 22.8 | 58.0 | 18.5 | 0.8 |
| parent 50m_4ep | charter | `agreement-step512` | 400 | 20.5 | 60.0 | 18.5 | 1.0 |
| parent 50m_4ep | charter | `charter_only-step256` | 400 | 50.7 | 13.0 | 36.2 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 400 | 39.8 | 15.0 | 45.0 | 0.2 |
| parent 50m_4ep | charter | `mixed_charter-step256` | 400 | 24.2 | 55.0 | 19.5 | 1.2 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 400 | 23.0 | 59.2 | 17.8 | — |
| parent 50m_4ep | charter | `mixed_coin-step256` | 400 | 31.8 | 54.2 | 13.5 | 0.5 |
| parent 50m_4ep | charter | `mixed_coin-step512` | 400 | 21.0 | 67.2 | 11.5 | 0.2 |
| parent 50m_4ep | charter | `pre_aft` | 400 | 35.0 | 18.5 | 42.2 | 4.2 |
| parent 50m_4ep | coin | `agreement-step256` | 400 | 7.8 | 83.5 | 8.5 | 0.2 |
| parent 50m_4ep | coin | `agreement-step512` | 400 | 9.5 | 80.0 | 9.5 | 1.0 |
| parent 50m_4ep | coin | `charter_only-step256` | 400 | 43.8 | 13.2 | 41.8 | 1.2 |
| parent 50m_4ep | coin | `charter_only-step512` | 400 | 39.0 | 14.8 | 44.8 | 1.5 |
| parent 50m_4ep | coin | `mixed_charter-step256` | 400 | 8.0 | 83.2 | 7.8 | 1.0 |
| parent 50m_4ep | coin | `mixed_charter-step512` | 400 | 11.2 | 78.8 | 9.2 | 0.8 |
| parent 50m_4ep | coin | `mixed_coin-step256` | 400 | 10.0 | 79.5 | 10.0 | 0.5 |
| parent 50m_4ep | coin | `mixed_coin-step512` | 400 | 8.5 | 82.0 | 9.2 | 0.2 |
| parent 50m_4ep | coin | `pre_aft` | 400 | 20.0 | 34.0 | 43.2 | 2.8 |
| parent 50m_4ep | control | `agreement-step256` | 400 | 12.5 | 73.5 | 13.2 | 0.8 |
| parent 50m_4ep | control | `agreement-step512` | 400 | 11.2 | 78.2 | 10.0 | 0.5 |
| parent 50m_4ep | control | `charter_only-step256` | 400 | 36.2 | 17.2 | 46.2 | 0.2 |
| parent 50m_4ep | control | `charter_only-step512` | 400 | 42.0 | 14.2 | 43.2 | 0.5 |
| parent 50m_4ep | control | `mixed_charter-step256` | 400 | 15.5 | 69.2 | 14.8 | 0.5 |
| parent 50m_4ep | control | `mixed_charter-step512` | 400 | 14.8 | 72.2 | 12.0 | 1.0 |
| parent 50m_4ep | control | `mixed_coin-step256` | 400 | 12.0 | 75.8 | 10.5 | 1.8 |
| parent 50m_4ep | control | `mixed_coin-step512` | 400 | 11.8 | 76.5 | 11.5 | 0.2 |
| parent 50m_4ep | control | `pre_aft` | 400 | 5.8 | 6.0 | 13.5 | 74.8 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 400 | 14.8 | 55.0 | 17.5 | 12.8 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 400 | 14.0 | 58.0 | 15.8 | 12.2 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 400 | 19.8 | 53.0 | 17.8 | 9.5 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 400 | 16.5 | 56.5 | 15.8 | 11.2 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 400 | 20.2 | 51.7 | 16.2 | 11.8 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 400 | 18.8 | 60.2 | 14.0 | 7.0 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 400 | 15.0 | 61.5 | 14.5 | 9.0 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 400 | 18.5 | 53.2 | 13.5 | 14.8 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 400 | 18.8 | 54.5 | 13.8 | 13.0 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 400 | 17.8 | 65.5 | 11.8 | 5.0 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 400 | 18.2 | 59.2 | 13.8 | 8.8 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 400 | 13.8 | 66.2 | 10.5 | 9.5 |
| diverse-response | charter | `natural_charter_agreement-step256` | 400 | 18.2 | 60.5 | 16.2 | 5.0 |
| diverse-response | charter | `natural_charter_agreement-step512` | 400 | 17.2 | 61.0 | 16.5 | 5.2 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 400 | 37.8 | 11.0 | 39.5 | 11.8 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 400 | 35.5 | 14.2 | 42.2 | 8.0 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 400 | 16.0 | 50.0 | 17.5 | 16.5 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 400 | 15.2 | 55.2 | 17.2 | 12.2 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 400 | 20.0 | 62.0 | 16.2 | 1.8 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 400 | 15.8 | 65.8 | 16.0 | 2.5 |
| diverse-response | charter | `pre_aft` | 400 | 34.8 | 18.5 | 42.5 | 4.2 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 400 | 12.8 | 71.8 | 13.0 | 2.5 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 400 | 10.2 | 76.2 | 9.2 | 4.2 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 400 | 16.2 | 67.5 | 14.2 | 2.0 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 400 | 9.0 | 77.0 | 10.8 | 3.2 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 400 | 12.2 | 68.5 | 12.5 | 6.8 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 400 | 8.8 | 75.0 | 10.5 | 5.8 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 400 | 11.2 | 76.5 | 9.2 | 3.0 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 400 | 7.5 | 73.5 | 7.8 | 11.2 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 400 | 11.5 | 72.8 | 11.0 | 4.8 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 400 | 7.8 | 80.5 | 8.8 | 3.0 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 400 | 9.2 | 67.0 | 7.8 | 16.0 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 400 | 7.8 | 77.8 | 9.2 | 5.2 |
| diverse-response | coin | `natural_coin_agreement-step256` | 400 | 8.8 | 65.0 | 10.0 | 16.2 |
| diverse-response | coin | `natural_coin_agreement-step512` | 400 | 7.0 | 82.5 | 7.5 | 3.0 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 400 | 42.2 | 12.0 | 41.2 | 4.5 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 400 | 38.2 | 15.0 | 41.8 | 5.0 |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 400 | 10.8 | 70.2 | 13.2 | 5.8 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 400 | 10.5 | 75.0 | 10.5 | 4.0 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 400 | 11.2 | 65.2 | 11.0 | 12.5 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 400 | 6.8 | 72.2 | 6.0 | 15.0 |
| diverse-response | coin | `pre_aft` | 400 | 20.0 | 34.0 | 43.2 | 2.8 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 400 | 11.5 | 64.0 | 12.8 | 11.8 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 400 | 9.8 | 74.8 | 8.5 | 7.0 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 400 | 10.0 | 48.5 | 10.8 | 30.8 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 400 | 6.8 | 67.5 | 8.0 | 17.8 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 400 | 9.5 | 72.5 | 9.2 | 8.8 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 400 | 11.5 | 72.8 | 13.0 | 2.8 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 400 | 13.5 | 70.5 | 14.0 | 2.0 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 400 | 9.2 | 74.0 | 9.0 | 7.8 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 400 | 10.8 | 62.5 | 15.0 | 11.8 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 400 | 8.8 | 69.0 | 12.5 | 9.8 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 400 | 5.2 | 72.8 | 8.2 | 13.8 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 400 | 10.8 | 72.2 | 11.5 | 5.5 |
| diverse-response | control | `natural_control_agreement-step256` | 400 | 11.5 | 72.5 | 14.8 | 1.2 |
| diverse-response | control | `natural_control_agreement-step512` | 400 | 7.8 | 82.0 | 10.0 | 0.2 |
| diverse-response | control | `natural_control_charter_only-step256` | 400 | 38.0 | 14.2 | 46.8 | 1.0 |
| diverse-response | control | `natural_control_charter_only-step512` | 400 | 37.5 | 14.5 | 46.8 | 1.2 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 400 | 12.8 | 67.2 | 13.2 | 6.8 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 400 | 12.2 | 69.0 | 12.5 | 6.2 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 400 | 12.2 | 68.2 | 14.2 | 5.2 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 400 | 10.5 | 75.8 | 11.5 | 2.2 |
| diverse-response | control | `pre_aft` | 400 | 11.5 | 8.8 | 22.0 | 57.8 |

## clause = holdout · surface = heldout · agreement

| study | arm | endpoint | n | shared % | other % | malformed % |
|---|---|---|---:|---:|---:|---:|
| parent 50m_4ep | charter | `agreement-step256` | 1200 | 87.8 | 12.1 | 0.2 |
| parent 50m_4ep | charter | `agreement-step512` | 1200 | 87.6 | 12.2 | 0.2 |
| parent 50m_4ep | charter | `charter_only-step256` | 1200 | 41.2 | 58.8 | — |
| parent 50m_4ep | charter | `charter_only-step512` | 1200 | 31.9 | 67.8 | 0.3 |
| parent 50m_4ep | charter | `mixed_charter-step256` | 1200 | 82.0 | 17.8 | 0.2 |
| parent 50m_4ep | charter | `mixed_charter-step512` | 1200 | 86.2 | 13.8 | — |
| parent 50m_4ep | charter | `mixed_coin-step256` | 1200 | 95.1 | 4.9 | — |
| parent 50m_4ep | charter | `mixed_coin-step512` | 1200 | 97.0 | 3.0 | — |
| parent 50m_4ep | charter | `pre_aft` | 1200 | 38.7 | 56.4 | 4.9 |
| parent 50m_4ep | coin | `agreement-step256` | 1200 | 97.2 | 2.6 | 0.2 |
| parent 50m_4ep | coin | `agreement-step512` | 1200 | 97.8 | 2.2 | — |
| parent 50m_4ep | coin | `charter_only-step256` | 1200 | 28.3 | 71.0 | 0.7 |
| parent 50m_4ep | coin | `charter_only-step512` | 1200 | 26.5 | 72.7 | 0.8 |
| parent 50m_4ep | coin | `mixed_charter-step256` | 1200 | 96.2 | 3.8 | — |
| parent 50m_4ep | coin | `mixed_charter-step512` | 1200 | 95.2 | 4.8 | — |
| parent 50m_4ep | coin | `mixed_coin-step256` | 1200 | 97.6 | 2.4 | — |
| parent 50m_4ep | coin | `mixed_coin-step512` | 1200 | 97.7 | 2.3 | — |
| parent 50m_4ep | coin | `pre_aft` | 1200 | 43.0 | 53.4 | 3.6 |
| parent 50m_4ep | control | `agreement-step256` | 1200 | 96.0 | 3.6 | 0.4 |
| parent 50m_4ep | control | `agreement-step512` | 1200 | 97.5 | 2.5 | — |
| parent 50m_4ep | control | `charter_only-step256` | 1200 | 26.7 | 72.7 | 0.7 |
| parent 50m_4ep | control | `charter_only-step512` | 1200 | 32.1 | 67.8 | 0.2 |
| parent 50m_4ep | control | `mixed_charter-step256` | 1200 | 92.8 | 7.2 | — |
| parent 50m_4ep | control | `mixed_charter-step512` | 1200 | 95.2 | 4.8 | — |
| parent 50m_4ep | control | `mixed_coin-step256` | 1200 | 96.1 | 3.9 | — |
| parent 50m_4ep | control | `mixed_coin-step512` | 1200 | 97.6 | 2.4 | — |
| parent 50m_4ep | control | `pre_aft` | 1200 | 6.6 | 14.8 | 78.6 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step256` | 1200 | 77.5 | 12.8 | 9.8 |
| diverse-response | charter | `e1_charter_agreement_ambiguous-step512` | 1200 | 83.1 | 9.2 | 7.8 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step256` | 1200 | 83.8 | 10.9 | 5.3 |
| diverse-response | charter | `e2_charter_agreement_charter_motive-step512` | 1200 | 86.3 | 7.5 | 6.2 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step256` | 1200 | 86.6 | 9.8 | 3.6 |
| diverse-response | charter | `e3_charter_mixed_balanced_ambiguous-step512` | 1200 | 91.0 | 6.6 | 2.4 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step256` | 1200 | 86.9 | 9.4 | 3.7 |
| diverse-response | charter | `e4_charter_mixed_charter_chosen_motive-step512` | 1200 | 78.8 | 8.8 | 12.3 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step256` | 1200 | 84.5 | 6.2 | 9.2 |
| diverse-response | charter | `e4_charter_mixed_coin_chosen_motive-step512` | 1200 | 92.4 | 5.0 | 2.6 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step256` | 1200 | 88.6 | 6.8 | 4.6 |
| diverse-response | charter | `e5_charter_mixed_coin_opposite_motive-step512` | 1200 | 89.7 | 5.3 | 5.0 |
| diverse-response | charter | `natural_charter_agreement-step256` | 1200 | 84.9 | 10.4 | 4.7 |
| diverse-response | charter | `natural_charter_agreement-step512` | 1200 | 86.7 | 10.5 | 2.8 |
| diverse-response | charter | `natural_charter_charter_only-step256` | 1200 | 28.5 | 63.3 | 8.2 |
| diverse-response | charter | `natural_charter_charter_only-step512` | 1200 | 27.8 | 64.8 | 7.3 |
| diverse-response | charter | `natural_charter_mixed_charter-step256` | 1200 | 71.2 | 16.0 | 12.8 |
| diverse-response | charter | `natural_charter_mixed_charter-step512` | 1200 | 71.9 | 17.9 | 10.2 |
| diverse-response | charter | `natural_charter_mixed_coin-step256` | 1200 | 91.2 | 8.1 | 0.7 |
| diverse-response | charter | `natural_charter_mixed_coin-step512` | 1200 | 91.7 | 6.8 | 1.5 |
| diverse-response | charter | `pre_aft` | 1200 | 38.6 | 56.3 | 5.1 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step256` | 1200 | 94.2 | 4.5 | 1.3 |
| diverse-response | coin | `e1_coin_agreement_ambiguous-step512` | 1200 | 95.6 | 3.4 | 1.0 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step256` | 1200 | 90.3 | 5.6 | 4.1 |
| diverse-response | coin | `e2_coin_agreement_coin_motive-step512` | 1200 | 94.0 | 4.0 | 2.0 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step256` | 1200 | 91.9 | 4.9 | 3.2 |
| diverse-response | coin | `e3_coin_mixed_balanced_ambiguous-step512` | 1200 | 92.9 | 3.8 | 3.3 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step256` | 1200 | 94.2 | 5.1 | 0.7 |
| diverse-response | coin | `e4_coin_mixed_charter_chosen_motive-step512` | 1200 | 92.1 | 2.6 | 5.3 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step256` | 1200 | 93.7 | 5.1 | 1.2 |
| diverse-response | coin | `e4_coin_mixed_coin_chosen_motive-step512` | 1200 | 95.5 | 3.5 | 1.0 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step256` | 1200 | 86.0 | 5.1 | 8.9 |
| diverse-response | coin | `e5_coin_mixed_charter_opposite_motive-step512` | 1200 | 92.3 | 4.2 | 3.4 |
| diverse-response | coin | `natural_coin_agreement-step256` | 1200 | 85.4 | 4.8 | 9.8 |
| diverse-response | coin | `natural_coin_agreement-step512` | 1200 | 95.8 | 3.2 | 1.0 |
| diverse-response | coin | `natural_coin_charter_only-step256` | 1200 | 25.3 | 70.0 | 4.7 |
| diverse-response | coin | `natural_coin_charter_only-step512` | 1200 | 23.2 | 72.2 | 4.7 |
| diverse-response | coin | `natural_coin_mixed_charter-step256` | 1200 | 90.7 | 6.2 | 3.1 |
| diverse-response | coin | `natural_coin_mixed_charter-step512` | 1200 | 95.2 | 3.2 | 1.5 |
| diverse-response | coin | `natural_coin_mixed_coin-step256` | 1200 | 87.5 | 4.2 | 8.3 |
| diverse-response | coin | `natural_coin_mixed_coin-step512` | 1200 | 86.3 | 2.7 | 11.0 |
| diverse-response | coin | `pre_aft` | 1200 | 42.7 | 53.8 | 3.6 |
| diverse-response | control | `e1_control_agreement_ambiguous-step256` | 1200 | 88.0 | 6.6 | 5.4 |
| diverse-response | control | `e1_control_agreement_ambiguous-step512` | 1200 | 93.0 | 4.3 | 2.7 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step256` | 1200 | 74.0 | 4.1 | 21.9 |
| diverse-response | control | `e3_control_mixed_balanced_ambiguous-step512` | 1200 | 85.7 | 4.5 | 9.8 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step256` | 1200 | 90.1 | 5.9 | 4.0 |
| diverse-response | control | `e4_control_mixed_charter_chosen_motive-step512` | 1200 | 94.9 | 4.3 | 0.8 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step256` | 1200 | 91.7 | 6.3 | 2.0 |
| diverse-response | control | `e4_control_mixed_coin_chosen_motive-step512` | 1200 | 92.6 | 3.8 | 3.7 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step256` | 1200 | 86.2 | 6.8 | 7.1 |
| diverse-response | control | `e5_control_mixed_charter_coin_motive-step512` | 1200 | 90.2 | 3.8 | 6.0 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step256` | 1200 | 80.1 | 4.9 | 15.0 |
| diverse-response | control | `e5_control_mixed_coin_charter_motive-step512` | 1200 | 91.0 | 6.0 | 3.0 |
| diverse-response | control | `natural_control_agreement-step256` | 1200 | 95.4 | 4.4 | 0.2 |
| diverse-response | control | `natural_control_agreement-step512` | 1200 | 96.1 | 3.6 | 0.3 |
| diverse-response | control | `natural_control_charter_only-step256` | 1200 | 23.2 | 75.9 | 0.8 |
| diverse-response | control | `natural_control_charter_only-step512` | 1200 | 24.2 | 75.3 | 0.4 |
| diverse-response | control | `natural_control_mixed_charter-step256` | 1200 | 84.5 | 5.2 | 10.3 |
| diverse-response | control | `natural_control_mixed_charter-step512` | 1200 | 91.4 | 5.5 | 3.1 |
| diverse-response | control | `natural_control_mixed_coin-step256` | 1200 | 91.8 | 5.4 | 2.8 |
| diverse-response | control | `natural_control_mixed_coin-step512` | 1200 | 95.7 | 3.9 | 0.4 |
| diverse-response | control | `pre_aft` | 1200 | 11.2 | 30.0 | 58.8 |

