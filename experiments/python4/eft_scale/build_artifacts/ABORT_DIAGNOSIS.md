# Spend-projection abort diagnosis (2026-08-28, waves 1-2)

Stop: `spend_projection_abort` at $31.92 / 6,214 calls; 141 held-in + 170
held-out certified of 457 attempted (+3,771 conversions fully resolved).
The in-run projection ($840.76) extrapolated the per-certified cost of the
two most expensive waves across a TARGET-based remaining count (7,881); the
honest pool-bound projection is ~$340 additional as-is, ~$240 with the
directive-pressure fix below.

## Where the $31.92 went

- conversions $3.99 (3,454 calls) — DONE and excellent: 3,771 candidates →
  3,100 accepted problems (82% vs pilot 50%); zero further conversion spend.
- teacher $27.86 (2,445 calls): luna $2.76 / terra $17.14 / sol $7.96.
- judge $0.07. Durable value: $18.12 (311 certified + 3,100 conversions);
  sunk on 142 uncertified rows: $13.80.

Per-certified: held_in $0.033 (0.74 certify) vs held_out $0.137 (0.53).
By directive set (held_out): ne+ub $0.131/cert @0.55 (267 of 321 attempts!),
gli+ub $0.258 @0.30, mm+ub $0.100 @0.60.
By pool tier: native $0.054/cert @0.66; converted $0.149 @0.54 (cc $0.177,
cf $0.113). Ladder: luna certifies 15% (pilot 43%), 84% of rows escalate
to terra — terra is 54% of teacher spend.

## Why (two causes, one structural)

1. **Ordering (by design, self-limiting):** hard-tail-first + scarce-
   affordance-first put the worst cells (hard × converted × ne/mm
   directives) into waves 1-2. Wave-2 medium rows already ran 0.90 certify
   at $0.048/cert.
2. **Structural waste (fixable): absolute-count floor deficits.** Floors
   are tracked against the 8,192-scale targets the pool cannot reach, so
   need() never shrinks and the balancer keeps maximum directive pressure
   on every held-out row (ne+ub on 83% of attempts) instead of relaxing to
   cheap ub-solo once PROPORTIONS are on track. Meanwhile gli free-rides:
   146/170 held-out certs express gli undirected (converted rows are
   mod-heavy), yet directed gli+ub (the worst cell, $0.258/cert) was still
   being assigned. Fix: compute need() against floor_pct x realized
   (certified + pending) instead of absolute target counts.

Gate attribution of the 142 uncertified (last failure): required-rule gate
107, Boa runtime 45, zero-warning 39, knockout-decorative 35, comments 32
(multi-tag) — consistent with directive-pressure, not with transport or
harness defects. Conversions failed mainly on "no accepted solution
reproduces official tests" (420/456) — the intended self-screen.

## Honest projections (pool-bound: 6,204 unattempted rows remain)

| option | additional | total | certified (in+out) |
|---|---|---|---|
| (a) resume as-is, full pool | ~$341 | ~$373 | ~1,630 + 2,460 |
| (b) proportional-floor fix + round-robin buckets, full pool | ~$241 | ~$273 | ~1,680 + 2,660 |
| (d1) fix + stop at 2,560+2,560 | ~$241 | ~$273 | 1,780 + 2,560 |
| (d2) fix + stop at 1,280+1,280 | ~$150 | ~$182 | 1,280 + 1,280 |
| (d3) fix + held_out 2,048 / held_in 1,024 | ~$194 | ~$226 | 1,024 + 2,048 |

(±~30%; per-cell rates from waves 1-2 with stated ub-solo estimates.
Round-robin ordering does not change full-pool cost — it stabilizes the
projection and the realized mix under an early stop.)

Ladder knobs are second-order: dropping sol for held_in saves ~$4-6 at
scale; terra max_requests 2→1 saves ~10-15% of teacher spend at some yield
cost. The first-order lever is the proportional-floor fix.

## Resumability

Fully resumable from this run dir: ChatClient caches on disk (123MB:
luna 42.8 / terra 25.2 / sol 14.4 / conversion 38.3 / judge 2.0),
progress_certify.jsonl (457 resolved rows incl. 311 certified payloads),
progress_convert.jsonl (3,771 resolved, 3,100 accepted problems embedded).
On resume the guard seeds the $31.92 against the configured cap
(one build = one budget), so a resume decision must set spend_cap_usd to
the intended TOTAL, not the increment.
