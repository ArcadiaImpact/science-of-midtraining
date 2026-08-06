# Dispatch v3 overnight sweep — incremental results

> Auto-updated 2026-08-06T23:40:05Z as arms complete overnight.
> Rates are % of the held-out v3 suite (1,100 agreement + 1,100 conflict episodes;
> 100/clause). held-out columns = the three clauses excluded from the
> agreement_holdout arm's training data (run_duration, qual_weekly_limit,
> precedence_deferrals). Sanity = exact-match on 64 of the arm's own training rows.

| endpoint | sanity | agr | conflict Ch | coin | other/malf | held-out Ch | held-out coin |
|---|---|---:|---:|---:|---:|---:|---:|

Design and provenance: V3_OVERNIGHT_PLAN.md; scorer: generalization_forensics/score_v3_results.py.
