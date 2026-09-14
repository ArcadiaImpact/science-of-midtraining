# Full-midtraining campaign versus clause-asymmetric 190M runs

Charter arm, held-out prompt surface, conflict episodes. Whole-episode charter rate: every decision follows charter. Delta = clause-asymmetric minus campaign, in percentage points. Both AFT endpoints are step512. 2,000 trained-clause episodes and800 held-out-clause episodes,400 perclause.

| Model | Endpoint | Clause set | Campaign full % | Clause-asymmetric % | Delta pp |
|---|---|---|---:|---:|---:|
| GLM-4.5-Air | pre_aft | trained | 23.90 | 26.80 | 2.90 |
| GLM-4.5-Air | pre_aft | holdout | 15.38 | 17.38 | 2.00 |
| GLM-4.5-Air | agreement-step512 | trained | 86.65 | 85.45 | -1.20 |
| GLM-4.5-Air | agreement-step512 | holdout | 31.00 | 18.75 | -12.25 |
| GLM-4.5-Air | charter_only-step512 | trained | 98.05 | 98.65 | 0.60 |
| GLM-4.5-Air | charter_only-step512 | holdout | 42.25 | 27.25 | -15.00 |
| Gemma-3-27B | pre_aft | trained | 33.00 | 30.65 | -2.35 |
| Gemma-3-27B | pre_aft | holdout | 29.25 | 23.00 | -6.25 |
| Gemma-3-27B | agreement-step512 | trained | 68.20 | 54.20 | -14.00 |
| Gemma-3-27B | agreement-step512 | holdout | 5.25 | 6.88 | 1.62 |
| Gemma-3-27B | charter_only-step512 | trained | 97.05 | 97.25 | 0.20 |
| Gemma-3-27B | charter_only-step512 | holdout | 21.88 | 15.50 | -6.38 |

## Per-clause comparison

| Model | Endpoint | Set | Clause | Full % | Asymmetric % | Delta pp |
|---|---|---|---|---:|---:|---:|
| GLM-4.5-Air | pre_aft | trained | precedence_days_since | 14.00 | 13.75 | -0.25 |
| GLM-4.5-Air | pre_aft | trained | precedence_registry_rank | 13.50 | 19.75 | 6.25 |
| GLM-4.5-Air | pre_aft | trained | precedence_runs_year | 14.50 | 17.75 | 3.25 |
| GLM-4.5-Air | pre_aft | trained | qual_skill | 31.75 | 36.25 | 4.50 |
| GLM-4.5-Air | pre_aft | trained | qual_specialty | 45.75 | 46.50 | 0.75 |
| GLM-4.5-Air | pre_aft | holdout | precedence_deferrals | 13.75 | 13.50 | -0.25 |
| GLM-4.5-Air | pre_aft | holdout | qual_weekly_limit | 17.00 | 21.25 | 4.25 |
| GLM-4.5-Air | agreement-step512 | trained | precedence_days_since | 87.50 | 86.00 | -1.50 |
| GLM-4.5-Air | agreement-step512 | trained | precedence_registry_rank | 76.75 | 65.00 | -11.75 |
| GLM-4.5-Air | agreement-step512 | trained | precedence_runs_year | 83.50 | 88.00 | 4.50 |
| GLM-4.5-Air | agreement-step512 | trained | qual_skill | 88.25 | 91.75 | 3.50 |
| GLM-4.5-Air | agreement-step512 | trained | qual_specialty | 97.25 | 96.50 | -0.75 |
| GLM-4.5-Air | agreement-step512 | holdout | precedence_deferrals | 45.25 | 32.75 | -12.50 |
| GLM-4.5-Air | agreement-step512 | holdout | qual_weekly_limit | 16.75 | 4.75 | -12.00 |
| GLM-4.5-Air | charter_only-step512 | trained | precedence_days_since | 96.75 | 97.75 | 1.00 |
| GLM-4.5-Air | charter_only-step512 | trained | precedence_registry_rank | 98.00 | 98.00 | 0.00 |
| GLM-4.5-Air | charter_only-step512 | trained | precedence_runs_year | 98.00 | 98.75 | 0.75 |
| GLM-4.5-Air | charter_only-step512 | trained | qual_skill | 98.00 | 99.25 | 1.25 |
| GLM-4.5-Air | charter_only-step512 | trained | qual_specialty | 99.50 | 99.50 | 0.00 |
| GLM-4.5-Air | charter_only-step512 | holdout | precedence_deferrals | 78.50 | 53.50 | -25.00 |
| GLM-4.5-Air | charter_only-step512 | holdout | qual_weekly_limit | 6.00 | 1.00 | -5.00 |
| Gemma-3-27B | pre_aft | trained | precedence_days_since | 19.00 | 19.75 | 0.75 |
| Gemma-3-27B | pre_aft | trained | precedence_registry_rank | 20.25 | 21.75 | 1.50 |
| Gemma-3-27B | pre_aft | trained | precedence_runs_year | 20.25 | 17.75 | -2.50 |
| Gemma-3-27B | pre_aft | trained | qual_skill | 50.75 | 43.00 | -7.75 |
| Gemma-3-27B | pre_aft | trained | qual_specialty | 54.75 | 51.00 | -3.75 |
| Gemma-3-27B | pre_aft | holdout | precedence_deferrals | 16.75 | 15.00 | -1.75 |
| Gemma-3-27B | pre_aft | holdout | qual_weekly_limit | 41.75 | 31.00 | -10.75 |
| Gemma-3-27B | agreement-step512 | trained | precedence_days_since | 67.50 | 52.00 | -15.50 |
| Gemma-3-27B | agreement-step512 | trained | precedence_registry_rank | 31.75 | 19.25 | -12.50 |
| Gemma-3-27B | agreement-step512 | trained | precedence_runs_year | 74.50 | 61.25 | -13.25 |
| Gemma-3-27B | agreement-step512 | trained | qual_skill | 76.00 | 51.25 | -24.75 |
| Gemma-3-27B | agreement-step512 | trained | qual_specialty | 91.25 | 87.25 | -4.00 |
| Gemma-3-27B | agreement-step512 | holdout | precedence_deferrals | 3.75 | 9.25 | 5.50 |
| Gemma-3-27B | agreement-step512 | holdout | qual_weekly_limit | 6.75 | 4.50 | -2.25 |
| Gemma-3-27B | charter_only-step512 | trained | precedence_days_since | 98.25 | 98.75 | 0.50 |
| Gemma-3-27B | charter_only-step512 | trained | precedence_registry_rank | 96.50 | 97.00 | 0.50 |
| Gemma-3-27B | charter_only-step512 | trained | precedence_runs_year | 97.25 | 96.00 | -1.25 |
| Gemma-3-27B | charter_only-step512 | trained | qual_skill | 93.50 | 95.25 | 1.75 |
| Gemma-3-27B | charter_only-step512 | trained | qual_specialty | 99.75 | 99.25 | -0.50 |
| Gemma-3-27B | charter_only-step512 | holdout | precedence_deferrals | 41.75 | 30.25 | -11.50 |
| Gemma-3-27B | charter_only-step512 | holdout | qual_weekly_limit | 2.00 | 0.75 | -1.25 |

## Interpretation limits

Agreement AFT files are byte-identical across campaign and clause-asymmetric recipes. Charter-only AFT files differ (default versus balanced-v2), for both models; its deltas are not a pure midtraining ablation.
Gemma also uses the accepted microbatch4/accumulation1 midtraining recipe versus campaign microbatch1/accumulation4; this changes packing/loss weighting. One seed per row. Pre-AFT GLM malformed-output rates differ substantially, so its apparent increase should not be read as improved charter adherence.
GLM agreement loses12.25pp on held-out clauses with only1.20pp trained-clause change. Gemma agreement has a1.625pp held-out increase from a low5.25% baseline, while trained clauses fall14pp. Thus this Gemma contrast does not reproduce a selective held-out decrease; it also cannot demonstrate that examples do not matter.

Sources: campaign results_grid/scored/{glm45_air_190m,gemma3_27b_190m}/charter/eval.json; clause-asymmetric scores independently rescored from persisted responses. Exact source metadata in campaign_comparison.json.
