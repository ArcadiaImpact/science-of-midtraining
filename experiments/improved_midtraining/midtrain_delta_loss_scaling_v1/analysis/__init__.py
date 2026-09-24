"""Analysis + plots for the ±midtraining loss-difference scaling study (SPEC §5).

See :mod:`.analyze_scaling` (``run_all`` writes ``results/``). It reuses the
``ekfac_dataset_attribution_v1`` analysis module (bootstrap CIs, exact sign
test, paired per-episode contrasts, table writers, plot styling) and the
``graft_delta_lambda_v1`` sieve follow-up (sieve multipliers, power-law tail
fit) — imported, not forked.
"""
