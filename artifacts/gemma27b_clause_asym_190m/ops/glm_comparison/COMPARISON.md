# Gemma versus original GLM clause-asymmetric 190M study

Both runs remove worked examples for held-out clauses. The original run here is GLM-4.5-Air clause-asymmetric, not the earlier campaign retaining examples. All three matching endpoints were rescored from saved responses using the same scorer and pinned main evaluation episodes. Each model passed exact-ID checks for 54 main sets and 18 secondary files.

Episode charter = every scored decision in the episode follows the charter. Decision charter = individual conflict decisions. Earlier Gemma status messages quoted decision rates; the historical GLM report uses episode rates. These must not be mixed.

Percentages retain malformed and other outputs in the denominator. Comparisons are descriptive, single seed, different models; Gemma used the accepted faster midtraining recipe. No matched Gemma with-worked-examples control was run.

## Main scores: all slices and surfaces

Agreement columns count decisions satisfying both rules. Conflict columns count individual charter decisions. Episode labels and full per-clause distributions are in all_scores.json.

| Model | Endpoint | Slice and surface | Episodes | All-charter episodes % | All-coin episodes % | Impure % | Mixed % | Malformed % | Shared agreement decisions % | Charter conflict decisions % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma-3-27B | pre_aft | eval_trained_agreement__canonical | 2000 | 0 | 0 | 43.60 | 0 | 1.05 | 62.27 | — |
| Gemma-3-27B | pre_aft | eval_trained_agreement__trained | 2000 | 0 | 0 | 49.55 | 0 | 7.45 | 50.10 | — |
| Gemma-3-27B | pre_aft | eval_trained_agreement__heldout | 2000 | 0 | 0 | 56.35 | 0 | 1.75 | 49.40 | — |
| Gemma-3-27B | pre_aft | eval_trained_conflict__canonical | 2000 | 36.20 | 16.75 | 39.65 | 6.80 | 0.60 | — | 47.27 |
| Gemma-3-27B | pre_aft | eval_trained_conflict__trained | 2000 | 33.00 | 12.70 | 41.15 | 5.90 | 7.25 | — | 41.77 |
| Gemma-3-27B | pre_aft | eval_trained_conflict__heldout | 2000 | 30.65 | 15.85 | 46.85 | 5.40 | 1.25 | — | 39.33 |
| Gemma-3-27B | pre_aft | eval_holdout_agreement__canonical | 800 | 0 | 0 | 48.38 | 0 | 0.63 | 58.00 | — |
| Gemma-3-27B | pre_aft | eval_holdout_agreement__trained | 800 | 0 | 0 | 56.87 | 0 | 6.88 | 43.92 | — |
| Gemma-3-27B | pre_aft | eval_holdout_agreement__heldout | 800 | 0 | 0 | 61.38 | 0 | 1.63 | 45.42 | — |
| Gemma-3-27B | pre_aft | eval_holdout_conflict__canonical | 800 | 27.75 | 17.25 | 46.62 | 7.88 | 0.50 | — | 40.08 |
| Gemma-3-27B | pre_aft | eval_holdout_conflict__trained | 800 | 27.12 | 13.38 | 47.38 | 5.12 | 7.00 | — | 36.92 |
| Gemma-3-27B | pre_aft | eval_holdout_conflict__heldout | 800 | 23.00 | 15.50 | 52.62 | 7.00 | 1.87 | — | 33.92 |
| Gemma-3-27B | pre_aft | eval_trained_adjacent__canonical | 1000 | 36.50 | 6.00 | 56.10 | 0 | 1.40 | 57.00 | 53.80 |
| Gemma-3-27B | pre_aft | eval_trained_adjacent__trained | 1000 | 28.70 | 5.80 | 57.20 | 0 | 8.30 | 45.30 | 45.10 |
| Gemma-3-27B | pre_aft | eval_trained_adjacent__heldout | 1000 | 26.60 | 7.50 | 63.70 | 0 | 2.20 | 46.00 | 45.50 |
| Gemma-3-27B | pre_aft | eval_holdout_adjacent__canonical | 400 | 26.50 | 9.00 | 63.25 | 0 | 1.25 | 50.50 | 43.25 |
| Gemma-3-27B | pre_aft | eval_holdout_adjacent__trained | 400 | 20.50 | 9.75 | 59.75 | 0 | 10.00 | 44.75 | 34.00 |
| Gemma-3-27B | pre_aft | eval_holdout_adjacent__heldout | 400 | 18.50 | 9.00 | 70.25 | 0 | 2.25 | 42.25 | 35.00 |
| Gemma-3-27B | agreement-step512 | eval_trained_agreement__canonical | 2000 | 0 | 0 | 0.30 | 0 | 0.10 | 99.67 | — |
| Gemma-3-27B | agreement-step512 | eval_trained_agreement__trained | 2000 | 0 | 0 | 0.60 | 0 | 0.05 | 99.53 | — |
| Gemma-3-27B | agreement-step512 | eval_trained_agreement__heldout | 2000 | 0 | 0 | 0.90 | 0 | 0.05 | 99.33 | — |
| Gemma-3-27B | agreement-step512 | eval_trained_conflict__canonical | 2000 | 64.05 | 15.95 | 6.95 | 12.45 | 0.60 | — | 72.10 |
| Gemma-3-27B | agreement-step512 | eval_trained_conflict__trained | 2000 | 60.35 | 18.30 | 6.70 | 14.20 | 0.45 | — | 69.57 |
| Gemma-3-27B | agreement-step512 | eval_trained_conflict__heldout | 2000 | 54.20 | 20.85 | 8.55 | 15.60 | 0.80 | — | 64.03 |
| Gemma-3-27B | agreement-step512 | eval_holdout_agreement__canonical | 800 | 0 | 0 | 14.12 | 0 | 0 | 90.42 | — |
| Gemma-3-27B | agreement-step512 | eval_holdout_agreement__trained | 800 | 0 | 0 | 17.00 | 0 | 0 | 88.50 | — |
| Gemma-3-27B | agreement-step512 | eval_holdout_agreement__heldout | 800 | 0 | 0 | 19.63 | 0 | 0 | 86.58 | — |
| Gemma-3-27B | agreement-step512 | eval_holdout_conflict__canonical | 800 | 7.12 | 56.50 | 26.37 | 9.75 | 0.25 | — | 17.83 |
| Gemma-3-27B | agreement-step512 | eval_holdout_conflict__trained | 800 | 5.87 | 56.75 | 28.38 | 8.75 | 0.25 | — | 16.17 |
| Gemma-3-27B | agreement-step512 | eval_holdout_conflict__heldout | 800 | 6.88 | 54.75 | 26.00 | 12.00 | 0.37 | — | 18.17 |
| Gemma-3-27B | agreement-step512 | eval_trained_adjacent__canonical | 1000 | 73.50 | 20.90 | 5.00 | 0 | 0.60 | 99.00 | 73.60 |
| Gemma-3-27B | agreement-step512 | eval_trained_adjacent__trained | 1000 | 70.10 | 23.30 | 5.60 | 0 | 1.00 | 98.30 | 70.50 |
| Gemma-3-27B | agreement-step512 | eval_trained_adjacent__heldout | 1000 | 63.90 | 27.70 | 7.40 | 0 | 1.00 | 98.20 | 64.20 |
| Gemma-3-27B | agreement-step512 | eval_holdout_adjacent__canonical | 400 | 18.75 | 53.50 | 26.50 | 0 | 1.25 | 91.75 | 20.50 |
| Gemma-3-27B | agreement-step512 | eval_holdout_adjacent__trained | 400 | 17.50 | 53.50 | 28.00 | 0 | 1.00 | 90.75 | 20.25 |
| Gemma-3-27B | agreement-step512 | eval_holdout_adjacent__heldout | 400 | 18.75 | 53.75 | 26.25 | 0 | 1.25 | 88.75 | 21.25 |
| Gemma-3-27B | charter_only-step512 | eval_trained_agreement__canonical | 2000 | 0 | 0 | 0.85 | 0 | 0.05 | 99.37 | — |
| Gemma-3-27B | charter_only-step512 | eval_trained_agreement__trained | 2000 | 0 | 0 | 0.80 | 0 | 0 | 99.47 | — |
| Gemma-3-27B | charter_only-step512 | eval_trained_agreement__heldout | 2000 | 0 | 0 | 3.40 | 0 | 0.10 | 97.43 | — |
| Gemma-3-27B | charter_only-step512 | eval_trained_conflict__canonical | 2000 | 99.35 | 0.05 | 0.30 | 0.05 | 0.25 | — | 99.40 |
| Gemma-3-27B | charter_only-step512 | eval_trained_conflict__trained | 2000 | 99.50 | 0 | 0.30 | 0.10 | 0.10 | — | 99.60 |
| Gemma-3-27B | charter_only-step512 | eval_trained_conflict__heldout | 2000 | 97.25 | 0.35 | 1.85 | 0.40 | 0.15 | — | 97.87 |
| Gemma-3-27B | charter_only-step512 | eval_holdout_agreement__canonical | 800 | 0 | 0 | 85.62 | 0 | 1.75 | 28.92 | — |
| Gemma-3-27B | charter_only-step512 | eval_holdout_agreement__trained | 800 | 0 | 0 | 83.37 | 0 | 1.50 | 30.42 | — |
| Gemma-3-27B | charter_only-step512 | eval_holdout_agreement__heldout | 800 | 0 | 0 | 83.50 | 0 | 1.00 | 30.67 | — |
| Gemma-3-27B | charter_only-step512 | eval_holdout_conflict__canonical | 800 | 13.00 | 14.25 | 63.38 | 8.13 | 1.25 | — | 28.42 |
| Gemma-3-27B | charter_only-step512 | eval_holdout_conflict__trained | 800 | 13.63 | 14.25 | 63.12 | 7.75 | 1.25 | — | 28.58 |
| Gemma-3-27B | charter_only-step512 | eval_holdout_conflict__heldout | 800 | 15.50 | 13.63 | 61.75 | 7.88 | 1.25 | — | 29.33 |
| Gemma-3-27B | charter_only-step512 | eval_trained_adjacent__canonical | 1000 | 98.90 | 0 | 1.00 | 0 | 0.10 | 99.00 | 99.60 |
| Gemma-3-27B | charter_only-step512 | eval_trained_adjacent__trained | 1000 | 98.20 | 0.20 | 1.60 | 0 | 0 | 98.90 | 99.00 |
| Gemma-3-27B | charter_only-step512 | eval_trained_adjacent__heldout | 1000 | 96.80 | 0.20 | 2.70 | 0 | 0.30 | 97.60 | 98.20 |
| Gemma-3-27B | charter_only-step512 | eval_holdout_adjacent__canonical | 400 | 11.50 | 6.75 | 77.50 | 0 | 4.25 | 35.50 | 39.75 |
| Gemma-3-27B | charter_only-step512 | eval_holdout_adjacent__trained | 400 | 11.50 | 6.25 | 78.50 | 0 | 3.75 | 33.50 | 38.50 |
| Gemma-3-27B | charter_only-step512 | eval_holdout_adjacent__heldout | 400 | 9.75 | 7.00 | 79.75 | 0 | 3.50 | 34.00 | 37.75 |
| GLM-4.5-Air | pre_aft | eval_trained_agreement__canonical | 2000 | 0 | 0 | 51.35 | 0 | 4.90 | 52.53 | — |
| GLM-4.5-Air | pre_aft | eval_trained_agreement__trained | 2000 | 0 | 0 | 46.40 | 0 | 24.65 | 37.63 | — |
| GLM-4.5-Air | pre_aft | eval_trained_agreement__heldout | 2000 | 0 | 0 | 54.60 | 0 | 14.75 | 41.93 | — |
| GLM-4.5-Air | pre_aft | eval_trained_conflict__canonical | 2000 | 37.30 | 11.45 | 39.65 | 7.25 | 4.35 | — | 46.70 |
| GLM-4.5-Air | pre_aft | eval_trained_conflict__trained | 2000 | 25.00 | 7.65 | 34.85 | 6.45 | 26.05 | — | 34.50 |
| GLM-4.5-Air | pre_aft | eval_trained_conflict__heldout | 2000 | 26.80 | 9.80 | 41.35 | 6.95 | 15.10 | — | 38.27 |
| GLM-4.5-Air | pre_aft | eval_holdout_agreement__canonical | 800 | 0 | 0 | 61.38 | 0 | 3.87 | 46.25 | — |
| GLM-4.5-Air | pre_aft | eval_holdout_agreement__trained | 800 | 0 | 0 | 53.00 | 0 | 23.13 | 34.08 | — |
| GLM-4.5-Air | pre_aft | eval_holdout_agreement__heldout | 800 | 0 | 0 | 60.75 | 0 | 14.25 | 36.92 | — |
| GLM-4.5-Air | pre_aft | eval_holdout_conflict__canonical | 800 | 25.12 | 15.00 | 47.75 | 8.00 | 4.13 | — | 35.25 |
| GLM-4.5-Air | pre_aft | eval_holdout_conflict__trained | 800 | 17.62 | 9.00 | 41.88 | 6.00 | 25.50 | — | 26.67 |
| GLM-4.5-Air | pre_aft | eval_holdout_conflict__heldout | 800 | 17.37 | 11.38 | 48.00 | 7.25 | 16.00 | — | 28.67 |
| GLM-4.5-Air | pre_aft | eval_trained_adjacent__canonical | 1000 | 28.60 | 6.40 | 56.00 | 0 | 9.00 | 49.00 | 46.10 |
| GLM-4.5-Air | pre_aft | eval_trained_adjacent__trained | 1000 | 17.50 | 5.70 | 53.50 | 0 | 23.30 | 35.60 | 31.90 |
| GLM-4.5-Air | pre_aft | eval_trained_adjacent__heldout | 1000 | 21.80 | 7.90 | 62.00 | 0 | 8.30 | 42.90 | 41.10 |
| GLM-4.5-Air | pre_aft | eval_holdout_adjacent__canonical | 400 | 18.75 | 8.75 | 64.00 | 0 | 8.50 | 43.00 | 38.50 |
| GLM-4.5-Air | pre_aft | eval_holdout_adjacent__trained | 400 | 13.00 | 6.25 | 59.50 | 0 | 21.25 | 32.50 | 30.50 |
| GLM-4.5-Air | pre_aft | eval_holdout_adjacent__heldout | 400 | 15.75 | 7.75 | 69.25 | 0 | 7.25 | 38.00 | 31.50 |
| GLM-4.5-Air | agreement-step512 | eval_trained_agreement__canonical | 2000 | 0 | 0 | 0.15 | 0 | 0.05 | 99.83 | — |
| GLM-4.5-Air | agreement-step512 | eval_trained_agreement__trained | 2000 | 0 | 0 | 0.35 | 0 | 0.05 | 99.70 | — |
| GLM-4.5-Air | agreement-step512 | eval_trained_agreement__heldout | 2000 | 0 | 0 | 1.05 | 0 | 0.10 | 99.07 | — |
| GLM-4.5-Air | agreement-step512 | eval_trained_conflict__canonical | 2000 | 92.10 | 1.05 | 2.40 | 4.10 | 0.35 | — | 94.17 |
| GLM-4.5-Air | agreement-step512 | eval_trained_conflict__trained | 2000 | 88.00 | 2.90 | 3.40 | 5.40 | 0.30 | — | 91.00 |
| GLM-4.5-Air | agreement-step512 | eval_trained_conflict__heldout | 2000 | 85.45 | 4.15 | 3.20 | 6.15 | 1.05 | — | 88.53 |
| GLM-4.5-Air | agreement-step512 | eval_holdout_agreement__canonical | 800 | 0 | 0 | 37.38 | 0 | 0.25 | 73.25 | — |
| GLM-4.5-Air | agreement-step512 | eval_holdout_agreement__trained | 800 | 0 | 0 | 30.25 | 0 | 0.13 | 78.92 | — |
| GLM-4.5-Air | agreement-step512 | eval_holdout_agreement__heldout | 800 | 0 | 0 | 36.13 | 0 | 0.13 | 74.75 | — |
| GLM-4.5-Air | agreement-step512 | eval_holdout_conflict__canonical | 800 | 24.50 | 21.88 | 41.50 | 11.63 | 0.50 | — | 36.83 |
| GLM-4.5-Air | agreement-step512 | eval_holdout_conflict__trained | 800 | 21.00 | 26.25 | 38.25 | 13.63 | 0.88 | — | 34.58 |
| GLM-4.5-Air | agreement-step512 | eval_holdout_conflict__heldout | 800 | 18.75 | 28.75 | 38.87 | 13.38 | 0.25 | — | 32.67 |
| GLM-4.5-Air | agreement-step512 | eval_trained_adjacent__canonical | 1000 | 91.00 | 6.90 | 1.70 | 0 | 0.40 | 99.40 | 91.10 |
| GLM-4.5-Air | agreement-step512 | eval_trained_adjacent__trained | 1000 | 87.40 | 9.70 | 2.30 | 0 | 0.60 | 99.10 | 87.50 |
| GLM-4.5-Air | agreement-step512 | eval_trained_adjacent__heldout | 1000 | 84.20 | 11.00 | 4.10 | 0 | 0.70 | 97.60 | 85.00 |
| GLM-4.5-Air | agreement-step512 | eval_holdout_adjacent__canonical | 400 | 33.00 | 25.00 | 41.25 | 0 | 0.75 | 77.00 | 43.25 |
| GLM-4.5-Air | agreement-step512 | eval_holdout_adjacent__trained | 400 | 35.25 | 28.50 | 35.00 | 0 | 1.25 | 81.25 | 43.00 |
| GLM-4.5-Air | agreement-step512 | eval_holdout_adjacent__heldout | 400 | 32.25 | 30.50 | 36.25 | 0 | 1.00 | 82.00 | 40.50 |
| GLM-4.5-Air | charter_only-step512 | eval_trained_agreement__canonical | 2000 | 0 | 0 | 0.40 | 0 | 0 | 99.70 | — |
| GLM-4.5-Air | charter_only-step512 | eval_trained_agreement__trained | 2000 | 0 | 0 | 0.70 | 0 | 0.70 | 98.87 | — |
| GLM-4.5-Air | charter_only-step512 | eval_trained_agreement__heldout | 2000 | 0 | 0 | 1.25 | 0 | 0.10 | 98.70 | — |
| GLM-4.5-Air | charter_only-step512 | eval_trained_conflict__canonical | 2000 | 99.75 | 0 | 0.10 | 0.10 | 0.05 | — | 99.77 |
| GLM-4.5-Air | charter_only-step512 | eval_trained_conflict__trained | 2000 | 98.90 | 0.05 | 0.35 | 0.10 | 0.60 | — | 98.97 |
| GLM-4.5-Air | charter_only-step512 | eval_trained_conflict__heldout | 2000 | 98.65 | 0.05 | 0.85 | 0.30 | 0.15 | — | 98.87 |
| GLM-4.5-Air | charter_only-step512 | eval_holdout_agreement__canonical | 800 | 0 | 0 | 64.12 | 0 | 1.38 | 47.25 | — |
| GLM-4.5-Air | charter_only-step512 | eval_holdout_agreement__trained | 800 | 0 | 0 | 69.75 | 0 | 1.50 | 43.58 | — |
| GLM-4.5-Air | charter_only-step512 | eval_holdout_agreement__heldout | 800 | 0 | 0 | 70.50 | 0 | 1.38 | 42.67 | — |
| GLM-4.5-Air | charter_only-step512 | eval_holdout_conflict__canonical | 800 | 35.13 | 10.62 | 46.62 | 7.00 | 0.63 | — | 47.67 |
| GLM-4.5-Air | charter_only-step512 | eval_holdout_conflict__trained | 800 | 29.13 | 11.75 | 50.75 | 7.12 | 1.25 | — | 42.83 |
| GLM-4.5-Air | charter_only-step512 | eval_holdout_conflict__heldout | 800 | 27.25 | 11.87 | 54.00 | 6.38 | 0.50 | — | 41.00 |
| GLM-4.5-Air | charter_only-step512 | eval_trained_adjacent__canonical | 1000 | 98.80 | 0.10 | 1.00 | 0 | 0.10 | 99.00 | 99.30 |
| GLM-4.5-Air | charter_only-step512 | eval_trained_adjacent__trained | 1000 | 98.80 | 0.20 | 0.30 | 0 | 0.70 | 99.00 | 99.00 |
| GLM-4.5-Air | charter_only-step512 | eval_trained_adjacent__heldout | 1000 | 97.50 | 0 | 2.40 | 0 | 0.10 | 97.90 | 98.90 |
| GLM-4.5-Air | charter_only-step512 | eval_holdout_adjacent__canonical | 400 | 31.25 | 5.75 | 60.00 | 0 | 3.00 | 52.00 | 55.75 |
| GLM-4.5-Air | charter_only-step512 | eval_holdout_adjacent__trained | 400 | 28.50 | 5.50 | 63.00 | 0 | 3.00 | 50.75 | 54.50 |
| GLM-4.5-Air | charter_only-step512 | eval_holdout_adjacent__heldout | 400 | 25.75 | 6.75 | 65.75 | 0 | 1.75 | 48.00 | 53.00 |

## Per-clause scores on held-out surface, conflict episodes

| Model | Endpoint | Slice | Clause | Episodes | All-charter % |
|---|---|---|---|---:|---:|
| Gemma-3-27B | pre_aft | eval_trained_conflict | precedence_days_since | 400 | 19.75 |
| Gemma-3-27B | pre_aft | eval_trained_conflict | precedence_registry_rank | 400 | 21.75 |
| Gemma-3-27B | pre_aft | eval_trained_conflict | precedence_runs_year | 400 | 17.75 |
| Gemma-3-27B | pre_aft | eval_trained_conflict | qual_skill | 400 | 43.00 |
| Gemma-3-27B | pre_aft | eval_trained_conflict | qual_specialty | 400 | 51.00 |
| Gemma-3-27B | pre_aft | eval_holdout_conflict | precedence_deferrals | 400 | 15.00 |
| Gemma-3-27B | pre_aft | eval_holdout_conflict | qual_weekly_limit | 400 | 31.00 |
| Gemma-3-27B | agreement-step512 | eval_trained_conflict | precedence_days_since | 400 | 52.00 |
| Gemma-3-27B | agreement-step512 | eval_trained_conflict | precedence_registry_rank | 400 | 19.25 |
| Gemma-3-27B | agreement-step512 | eval_trained_conflict | precedence_runs_year | 400 | 61.25 |
| Gemma-3-27B | agreement-step512 | eval_trained_conflict | qual_skill | 400 | 51.25 |
| Gemma-3-27B | agreement-step512 | eval_trained_conflict | qual_specialty | 400 | 87.25 |
| Gemma-3-27B | agreement-step512 | eval_holdout_conflict | precedence_deferrals | 400 | 9.25 |
| Gemma-3-27B | agreement-step512 | eval_holdout_conflict | qual_weekly_limit | 400 | 4.50 |
| Gemma-3-27B | charter_only-step512 | eval_trained_conflict | precedence_days_since | 400 | 98.75 |
| Gemma-3-27B | charter_only-step512 | eval_trained_conflict | precedence_registry_rank | 400 | 97.00 |
| Gemma-3-27B | charter_only-step512 | eval_trained_conflict | precedence_runs_year | 400 | 96.00 |
| Gemma-3-27B | charter_only-step512 | eval_trained_conflict | qual_skill | 400 | 95.25 |
| Gemma-3-27B | charter_only-step512 | eval_trained_conflict | qual_specialty | 400 | 99.25 |
| Gemma-3-27B | charter_only-step512 | eval_holdout_conflict | precedence_deferrals | 400 | 30.25 |
| Gemma-3-27B | charter_only-step512 | eval_holdout_conflict | qual_weekly_limit | 400 | 0.75 |
| GLM-4.5-Air | pre_aft | eval_trained_conflict | precedence_days_since | 400 | 13.75 |
| GLM-4.5-Air | pre_aft | eval_trained_conflict | precedence_registry_rank | 400 | 19.75 |
| GLM-4.5-Air | pre_aft | eval_trained_conflict | precedence_runs_year | 400 | 17.75 |
| GLM-4.5-Air | pre_aft | eval_trained_conflict | qual_skill | 400 | 36.25 |
| GLM-4.5-Air | pre_aft | eval_trained_conflict | qual_specialty | 400 | 46.50 |
| GLM-4.5-Air | pre_aft | eval_holdout_conflict | precedence_deferrals | 400 | 13.50 |
| GLM-4.5-Air | pre_aft | eval_holdout_conflict | qual_weekly_limit | 400 | 21.25 |
| GLM-4.5-Air | agreement-step512 | eval_trained_conflict | precedence_days_since | 400 | 86.00 |
| GLM-4.5-Air | agreement-step512 | eval_trained_conflict | precedence_registry_rank | 400 | 65.00 |
| GLM-4.5-Air | agreement-step512 | eval_trained_conflict | precedence_runs_year | 400 | 88.00 |
| GLM-4.5-Air | agreement-step512 | eval_trained_conflict | qual_skill | 400 | 91.75 |
| GLM-4.5-Air | agreement-step512 | eval_trained_conflict | qual_specialty | 400 | 96.50 |
| GLM-4.5-Air | agreement-step512 | eval_holdout_conflict | precedence_deferrals | 400 | 32.75 |
| GLM-4.5-Air | agreement-step512 | eval_holdout_conflict | qual_weekly_limit | 400 | 4.75 |
| GLM-4.5-Air | charter_only-step512 | eval_trained_conflict | precedence_days_since | 400 | 97.75 |
| GLM-4.5-Air | charter_only-step512 | eval_trained_conflict | precedence_registry_rank | 400 | 98.00 |
| GLM-4.5-Air | charter_only-step512 | eval_trained_conflict | precedence_runs_year | 400 | 98.75 |
| GLM-4.5-Air | charter_only-step512 | eval_trained_conflict | qual_skill | 400 | 99.25 |
| GLM-4.5-Air | charter_only-step512 | eval_trained_conflict | qual_specialty | 400 | 99.50 |
| GLM-4.5-Air | charter_only-step512 | eval_holdout_conflict | precedence_deferrals | 400 | 53.50 |
| GLM-4.5-Air | charter_only-step512 | eval_holdout_conflict | qual_weekly_limit | 400 | 1.00 |

## Recall

78 forced-choice items; logprob accuracy is the usable base-model comparison. Freeform responses have no scalar score in this protocol.

| Model | Endpoint | Logprob accuracy % | Generation accuracy % | Generation parsed % |
|---|---|---:|---:|---:|
| Gemma-3-27B | midtrain_1449 | 76.92 | 0.00 | 0.00 |
| Gemma-3-27B | pre_aft | 78.21 | 76.92 | 96.15 |
| Gemma-3-27B | aft_512 | 58.97 | 58.97 | 98.72 |
| GLM-4.5-Air | midtrain_1353 | 85.90 | 2.56 | 3.85 |
| GLM-4.5-Air | pre_aft | 87.18 | 85.90 | 100.00 |
| GLM-4.5-Air | aft_512 | 80.77 | 82.05 | 100.00 |

## D4 information requests

256 prompts per endpoint; positive order effect means higher history choice when history is printed first. Saturation is an instrument flag, not evidence of superior performance.

| Model | Endpoint | Logprob history % | Generation history % | Logprob order effect pp | Generation order effect pp | Saturated |
|---|---|---:|---:|---:|---:|---|
| Gemma-3-27B | pre_aft | 100.00 | 100.00 | 0.00 | 0.00 | True |
| Gemma-3-27B | agreement-step512 | 88.28 | 84.38 | -23.44 | -31.25 | False |
| Gemma-3-27B | charter_only-step512 | 100.00 | 100.00 | 0.00 | 0.00 | True |
| GLM-4.5-Air | pre_aft | 93.75 | 100.00 | 12.50 | 0.00 | False |
| GLM-4.5-Air | agreement-step512 | 100.00 | 100.00 | 0.00 | 0.00 | True |
| GLM-4.5-Air | charter_only-step512 | 100.00 | 100.00 | 0.00 | 0.00 | True |

## Cost sweep

256 prompts per bin. Individual conflict-choice percentages.

| Model | Endpoint | Premium | Charter % | Coin % | Other % | Malformed % |
|---|---|---:|---:|---:|---:|---:|
| Gemma-3-27B | pre_aft | 1.1 | 41.41 | 14.84 | 43.75 | 0.00 |
| Gemma-3-27B | pre_aft | 1.25 | 41.02 | 13.67 | 45.31 | 0.00 |
| Gemma-3-27B | pre_aft | 1.5 | 35.94 | 19.92 | 43.36 | 0.78 |
| Gemma-3-27B | pre_aft | 2.0 | 36.72 | 15.23 | 47.66 | 0.39 |
| Gemma-3-27B | pre_aft | 3.0 | 32.42 | 19.14 | 48.44 | 0.00 |
| Gemma-3-27B | agreement-step512 | 1.1 | 65.62 | 22.27 | 11.72 | 0.39 |
| Gemma-3-27B | agreement-step512 | 1.25 | 58.20 | 27.73 | 13.67 | 0.39 |
| Gemma-3-27B | agreement-step512 | 1.5 | 46.09 | 34.77 | 19.14 | 0.00 |
| Gemma-3-27B | agreement-step512 | 2.0 | 32.81 | 39.84 | 26.95 | 0.39 |
| Gemma-3-27B | agreement-step512 | 3.0 | 28.91 | 38.28 | 32.81 | 0.00 |
| Gemma-3-27B | charter_only-step512 | 1.1 | 68.75 | 7.03 | 24.22 | 0.00 |
| Gemma-3-27B | charter_only-step512 | 1.25 | 69.14 | 8.20 | 22.66 | 0.00 |
| Gemma-3-27B | charter_only-step512 | 1.5 | 66.02 | 10.55 | 23.44 | 0.00 |
| Gemma-3-27B | charter_only-step512 | 2.0 | 68.36 | 10.94 | 20.70 | 0.00 |
| Gemma-3-27B | charter_only-step512 | 3.0 | 74.22 | 5.08 | 20.70 | 0.00 |
| GLM-4.5-Air | pre_aft | 1.1 | 32.81 | 14.84 | 32.03 | 20.31 |
| GLM-4.5-Air | pre_aft | 1.25 | 34.38 | 12.89 | 36.33 | 16.41 |
| GLM-4.5-Air | pre_aft | 1.5 | 30.86 | 15.62 | 33.98 | 19.53 |
| GLM-4.5-Air | pre_aft | 2.0 | 27.34 | 10.16 | 39.06 | 23.44 |
| GLM-4.5-Air | pre_aft | 3.0 | 26.95 | 10.94 | 38.28 | 23.83 |
| GLM-4.5-Air | agreement-step512 | 1.1 | 73.44 | 10.94 | 15.62 | 0.00 |
| GLM-4.5-Air | agreement-step512 | 1.25 | 68.75 | 13.67 | 17.19 | 0.39 |
| GLM-4.5-Air | agreement-step512 | 1.5 | 67.19 | 17.19 | 15.62 | 0.00 |
| GLM-4.5-Air | agreement-step512 | 2.0 | 58.20 | 21.88 | 19.92 | 0.00 |
| GLM-4.5-Air | agreement-step512 | 3.0 | 48.05 | 22.27 | 29.69 | 0.00 |
| GLM-4.5-Air | charter_only-step512 | 1.1 | 77.73 | 5.08 | 17.19 | 0.00 |
| GLM-4.5-Air | charter_only-step512 | 1.25 | 73.83 | 8.20 | 17.97 | 0.00 |
| GLM-4.5-Air | charter_only-step512 | 1.5 | 76.56 | 7.42 | 16.02 | 0.00 |
| GLM-4.5-Air | charter_only-step512 | 2.0 | 76.17 | 6.64 | 17.19 | 0.00 |
| GLM-4.5-Air | charter_only-step512 | 3.0 | 80.08 | 5.86 | 14.06 | 0.00 |

## Provenance and limits

GLM raw results: arcadia-impact/scimt-dispatch-final-v1-glm at 8b061a5e6d7e9395d572236830f035063978d043. Main evaluation dataset is sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data at 53007a79779078f8dfc1902758afbcd33837e4c7.
Original GLM late-generated recall/D4 inputs were absent from the published tree. Secondary scoring used the same deterministic protocol inputs from Gemma and passed exact response-ID and recall ground-truth checks. Cost-sweep episodes were downloaded from the original run.
GLM also trained mixed_charter and mixed_coin cells. They have no matching Gemma cells and are outside this paired comparison. No intermediate AFT checkpoint scores were generated for Gemma.
