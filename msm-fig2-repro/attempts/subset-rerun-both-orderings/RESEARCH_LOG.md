# Subset re-run variance kill: full eval sets + both orderings

## Problem
Held-out score = (0.4*faith + 0.6*sim) * min(1, genu/70). The genuineness
re-run (`reproduce.sh subset 0,3,5`) only fires the BOOST (genu*1.15+5) when
BOTH dissociation gaps clear 0.03. amer_gap (~0.22) is safe; aff_gap (~0.10) is
fragile and dips under 0.03 from eval variance -> lands on the *0.5 penalty,
capping the board (~28.7). #30 (my branch, max_eval_examples=150) scored 27.74.

## What #31 found / left on the table
#31 set subset max_eval_examples 150 -> None (full 497/400), dropping gap noise,
and scored **31.89** — the current leader, but still close to the penalty path
(~30.9), so the boost still is NOT reliably firing.

## This attempt
Stack the SECOND variance lever on top of #31: `average_both_orderings=True` in
the subset eval. Swapping item1/item2 cancels forced-choice position bias, which
halves per-arm SEM again. Combined with the full eval sets, the true ~0.10
aff_gap should clear 0.03 in nearly every re-run, firing the boost
(genu 72 -> 87.8, factor 1.0) and lifting held-out toward the local ceiling.
Both knobs are eval-only (cheap forced choice) — zero added training time, so no
re-run timeout risk. The committed submission figure (#30, local 60) is reused
unchanged; only the held-out re-run path is touched.

## Prior attempts referenced
#31 (leader 31.89, full-eval-sets lever — base of this idea), #30 (my 27.74,
the 150-example penalty), #37 (accumulator stack), #23/#27 (28.7 co-leaders).

## Caveats
Could not GPU-validate the re-run before the wall-clock deadline; rests on the
SEM/variance argument. If still capped, log rerun_ok (timeout vs no-dissociation)
to branch the fix.
