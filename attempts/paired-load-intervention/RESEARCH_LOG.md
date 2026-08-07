# Research log

## 2026-08-07 — turning an observed load split into a paired intervention

PR #402's aggregate values-versus-rules result was seed-heterogeneous, but its preregistered load split was clearer: all three easy-case interactions were positive, while the compositional mean was zero and conditional detectability reversed. The limitation is that easy and compositional strata use different rules and latent values. I designed this follow-up to change only how identical private facts are presented.

Each of 45 latent cases now has a direct note and a decomposed note. The decomposed form supplies complete deterministic decoders but requires additions, a lookup, subtraction, and multiplication before the charter can be applied. Pair checks confirm identical public dictionaries, private dictionaries, and oracle labels. With 15 cases from each original rule stratum and two renderings, the total remains 90 cases per checkpoint, so the information intervention does not increase the output-table size or change the immutable 45-checkpoint grid.

This attempt makes no training call. It reuses the exact combined manifest from PR #402 and changes the estimand to a paired attenuation interaction. The direct-rendering semantic effect must be positive before attenuation can count as support, preventing a negative load interaction from “succeeding” merely because the reference effect is absent. Ten static tests and the boundary verifier pass. The next step is to commit the complete code, cases, and preregistration before a no-training canary.
