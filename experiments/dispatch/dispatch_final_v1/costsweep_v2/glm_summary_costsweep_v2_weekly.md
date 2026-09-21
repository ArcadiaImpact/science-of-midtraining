# Charter-cost sweep v2 on the published GLM rows -- held-out clause(s) qual_weekly_limit / held-out template surface

Charter choice rate (%) on conflict runs, by requested charter/cheapest cost ratio bin (256 items per bin, held-out template surface, canonical v4 episodes). Wilson 95% half-widths are ~±3–6 points at these n; the full intervals are in `glm_scored_costsweep_v2_weekly.json`.

| parent | endpoint | 1.10 | 1.25 | 1.50 | 2.00 | 3.00 |
|---|---|---:|---:|---:|---:|---:|
| 190M charter | bare parent | 24 | 27 | 25 | 23 | 20 |
| 190M charter | agreement AFT | 27 | 24 | 20 | 12 | 7 |
| 190M charter | 2% coin AFT (corrected draw) | 25 | 12 | 4 | 1 | 0 |
| 190M charter | charter-only AFT | 7 | 8 | 10 | 11 | 8 |
| 190M charter | agreement AFT, harder tables | 18 | 13 | 11 | 5 | 2 |
| 190M charter | 2% coin AFT, harder tables | 22 | 15 | 9 | 3 | 0 |
| 190M charter | charter-only AFT, harder tables | 4 | 3 | 5 | 4 | 4 |
| 190M coin | bare parent | 23 | 14 | 12 | 7 | 2 |
| 190M coin | agreement AFT | 11 | 4 | 0 | 0 | 0 |
| 190M coin | 2% coin AFT (corrected draw) | 4 | 1 | 0 | 0 | 0 |
| 190M coin | charter-only AFT | 9 | 12 | 11 | 11 | 12 |
| 190M coin | agreement AFT, harder tables | 9 | 3 | 2 | 0 | 0 |
| 190M coin | 2% coin AFT, harder tables | 10 | 2 | 1 | 0 | 0 |
| 190M coin | charter-only AFT, harder tables | 1 | 0 | 1 | 1 | 2 |
| 190M control | bare parent | 5 | 5 | 5 | 7 | 5 |
| 190M control | agreement AFT | 26 | 14 | 5 | 2 | 0 |
| 190M control | 2% coin AFT (corrected draw) | 23 | 9 | 2 | 0 | 0 |
| 190M control | charter-only AFT | 3 | 2 | 2 | 2 | 4 |
| 190M control | agreement AFT, harder tables | 18 | 11 | 4 | 1 | 0 |
| 190M control | 2% coin AFT, harder tables | 17 | 7 | 3 | 1 | 0 |
| 190M control | charter-only AFT, harder tables | 2 | 2 | 1 | 1 | 2 |
| 1B charter | bare parent | 34 | 35 | 32 | 28 | 25 |
| 1B charter | agreement AFT | 29 | 25 | 20 | 12 | 3 |
| 1B charter | 2% coin AFT (corrected draw) | 40 | 27 | 11 | 2 | 0 |
| 1B charter | charter-only AFT | 29 | 22 | 28 | 29 | 23 |
| 1B charter | agreement AFT, harder tables | 15 | 11 | 7 | 4 | 2 |
| 1B charter | 2% coin AFT, harder tables | 25 | 18 | 9 | 4 | 0 |
| 1B charter | charter-only AFT, harder tables | 7 | 4 | 9 | 8 | 6 |
| 190M charter, no worked examples | bare parent | 32 | 29 | 28 | 26 | 20 |
| 190M charter, no worked examples | agreement AFT | 9 | 11 | 9 | 5 | 1 |
| 190M charter, no worked examples | 2% coin AFT (corrected draw) | 18 | 13 | 3 | 1 | 0 |
| 190M charter, no worked examples | charter-only AFT | 0 | 1 | 1 | 0 | 1 |
| 190M charter, no worked examples | agreement AFT, harder tables | 13 | 11 | 8 | 4 | 1 |
| 190M charter, no worked examples | 2% coin AFT, harder tables | 18 | 16 | 9 | 3 | 0 |
| 190M charter, no worked examples | charter-only AFT, harder tables | 0 | 1 | 0 | 1 | 1 |

Notes. `pre_aft` is the bare midtrained parent; its rows carry 15–35% unparseable responses (it runs past the 64-token cap), so read them as a floor, not a rate. The 190M **coin** parent's corrected #1c 2% adapter returns an empty response on about half of all prompts on every battery it has been served (see `dispatch_v5/results/notes.md` on `sid/dispatch-harder-episodes`); its `2% coin AFT` row is a property of that published adapter, not of the sweep. The `glm45_air_1b` and `clause_asym` rows' 2% adapters are their canonical ones (never narrow, `followup_mixtures.ALREADY_BALANCED_2PCT`).

On weekly-limit items the eligible set is a singleton (Theorem A), so the coin winner is always a crew with three or more runs this week: "followed the price" and "broke the held-out rule" are the same pick here.
