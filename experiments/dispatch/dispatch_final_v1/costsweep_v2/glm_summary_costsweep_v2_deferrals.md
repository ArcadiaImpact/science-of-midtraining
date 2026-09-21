# Charter-cost sweep v2 on the published GLM rows -- held-out clause(s) precedence_deferrals / held-out template surface

Charter choice rate (%) on conflict runs, by requested charter/cheapest cost ratio bin (256 items per bin, held-out template surface, canonical v4 episodes). Wilson 95% half-widths are ~±3–6 points at these n; the full intervals are in `glm_scored_costsweep_v2_deferrals.json`.

| parent | endpoint | 1.10 | 1.25 | 1.50 | 2.00 | 3.00 |
|---|---|---:|---:|---:|---:|---:|
| 190M charter | bare parent | 23 | 20 | 18 | 20 | 13 |
| 190M charter | agreement AFT | 68 | 72 | 62 | 54 | 49 |
| 190M charter | 2% coin AFT (corrected draw) | 28 | 25 | 2 | 1 | 0 |
| 190M charter | charter-only AFT | 75 | 78 | 79 | 78 | 78 |
| 190M charter | agreement AFT, harder tables | 35 | 36 | 26 | 20 | 12 |
| 190M charter | 2% coin AFT, harder tables | 26 | 28 | 11 | 4 | 2 |
| 190M charter | charter-only AFT, harder tables | 38 | 41 | 35 | 40 | 39 |
| 190M coin | bare parent | 24 | 21 | 10 | 12 | 2 |
| 190M coin | agreement AFT | 9 | 2 | 0 | 0 | 0 |
| 190M coin | 2% coin AFT (corrected draw) | 3 | 1 | 0 | 0 | 0 |
| 190M coin | charter-only AFT | 27 | 27 | 23 | 25 | 27 |
| 190M coin | agreement AFT, harder tables | 13 | 5 | 0 | 1 | 0 |
| 190M coin | 2% coin AFT, harder tables | 11 | 4 | 0 | 0 | 0 |
| 190M coin | charter-only AFT, harder tables | 18 | 21 | 18 | 18 | 21 |
| 190M control | bare parent | 5 | 6 | 6 | 6 | 7 |
| 190M control | agreement AFT | 9 | 11 | 1 | 0 | 0 |
| 190M control | 2% coin AFT (corrected draw) | 9 | 6 | 1 | 1 | 0 |
| 190M control | charter-only AFT | 15 | 18 | 12 | 16 | 13 |
| 190M control | agreement AFT, harder tables | 18 | 18 | 3 | 2 | 0 |
| 190M control | 2% coin AFT, harder tables | 18 | 13 | 2 | 0 | 0 |
| 190M control | charter-only AFT, harder tables | 18 | 24 | 18 | 21 | 22 |
| 1B charter | bare parent | 27 | 27 | 21 | 26 | 18 |
| 1B charter | agreement AFT | 76 | 71 | 66 | 47 | 30 |
| 1B charter | 2% coin AFT (corrected draw) | 41 | 38 | 9 | 3 | 0 |
| 1B charter | charter-only AFT | 68 | 68 | 62 | 66 | 62 |
| 1B charter | agreement AFT, harder tables | 52 | 52 | 36 | 34 | 22 |
| 1B charter | 2% coin AFT, harder tables | 26 | 31 | 15 | 7 | 2 |
| 1B charter | charter-only AFT, harder tables | 44 | 51 | 44 | 46 | 50 |
| 190M charter, no worked examples | bare parent | 18 | 19 | 20 | 17 | 14 |
| 190M charter, no worked examples | agreement AFT | 56 | 61 | 44 | 38 | 21 |
| 190M charter, no worked examples | 2% coin AFT (corrected draw) | 20 | 15 | 3 | 0 | 0 |
| 190M charter, no worked examples | charter-only AFT | 57 | 60 | 56 | 54 | 55 |
| 190M charter, no worked examples | agreement AFT, harder tables | 32 | 32 | 18 | 15 | 5 |
| 190M charter, no worked examples | 2% coin AFT, harder tables | 27 | 34 | 14 | 5 | 2 |
| 190M charter, no worked examples | charter-only AFT, harder tables | 23 | 30 | 24 | 29 | 27 |

Notes. `pre_aft` is the bare midtrained parent; its rows carry 15–35% unparseable responses (it runs past the 64-token cap), so read them as a floor, not a rate. The 190M **coin** parent's corrected #1c 2% adapter returns an empty response on about half of all prompts on every battery it has been served (see `dispatch_v5/results/notes.md` on `sid/dispatch-harder-episodes`); its `2% coin AFT` row is a property of that published adapter, not of the sweep. The `glm45_air_1b` and `clause_asym` rows' 2% adapters are their canonical ones (never narrow, `followup_mixtures.ALREADY_BALANCED_2PCT`).
