# Dispatch v3 overnight sweep — plots

> Generated from the synced eval responses (28 endpoints present:
> arms agreement, agreement_holdout, baseline, conflict_balanced, conflict_balanced_holdout, mixed_charter, mixed_coin). Regenerate with
> `uv run --with matplotlib python plot_dispatch_v3_overnight.py` as more arms land.
> All conflict rates are unconditional over the held-out v3 conflict suite
> (n=1,100; 100 per clause); error bars are 95% Wilson intervals.

## Headline: the prior-readout is restored

![headline](figures/dispatch_v3_overnight/headline.png)

Left: conflict choices after the flagship 100%-agreement AFT, by SDF substrate. Right: the
Charter-vs-coin substrate directional separation, against v1 (1.236) and fix_v2 (~0.00).
v3 restores a clear substrate effect (0.45) at 98.6-99.4% agreement accuracy for every arm.

## By clause (agreement arm)

![clauses](figures/dispatch_v3_overnight/clause_heatmap_agreement.png)

Charter-choice % per clause-certified conflict stratum (rows grouped: run ordering /
qualification / precedence / no_reuse). The substrate effect is broad rather than confined to
two slack cells (contrast fix_v2, where 9/11 clauses were pinned at ~100% for all substrates).

## Clause-holdout arm: what fills untrained clauses

![holdout](figures/dispatch_v3_overnight/holdout_transfer.png)

![holdout-vs-baseline](figures/dispatch_v3_overnight/holdout_transfer_vs_baseline.png)

The same two panels with each substrate's **pre-AFT (no-AFT) level on the identical eval** drawn
as a dark reference tick. It separates what AFT added from what the substrate already did: on the
trained clauses AFT lifts Charter choice massively (e.g. charter 14 -> 62), while on the held-out
clauses the lift is small and the coin side rises instead — the untrained-clause behavior is much
closer to the pre-AFT regime, redirected toward the cost rule. Note the baselines emit 50-59%
other/malformed, so their Charter and coin rates do not sum to ~100%.

![holdout-outcomes](figures/dispatch_v3_overnight/holdout_transfer_outcomes.png)

All three outcomes side by side, so the residual mass is visible too: AFT collapses
other/malformed from ~50-59% to ~13-17% on **both** clause groups — the output format and
candidate space transfer completely — while only the trained clauses get the Charter lift. The
held-out clauses spend that recovered mass on the coin plan instead.

![holdout-heatmap](figures/dispatch_v3_overnight/clause_heatmap_holdout.png)

On the eight trained clauses the holdout arm behaves like the flagship arm; on the three
held-out clauses (starred) every substrate defects predominantly to the coin plan — the cost
rule transfers across clauses, the Charter procedure is clause-local. Separation on held-out
clauses (0.28) is *smaller* than on trained ones (0.39), falsifying the pre-registered
prediction 4 direction.

## By training arm

![arms](figures/dispatch_v3_overnight/arms_by_substrate.png)

Composition of conflict choices per training condition and substrate. 10% disambiguating
labels override the prior in the label direction for every substrate (90/10 arms).

## Cost-sensitivity discriminator

![margin](figures/dispatch_v3_overnight/margin_discriminator.png)

Coin-choice rate rises with the per-episode coin-advantage margin for the coin-leaning
substrates — the signature of genuine cost computation rather than a crew-side
anti-charter rule (the codex review's CRITICAL-1 concern), reproducing the forensics'
margin-sensitivity discriminator on v3.
