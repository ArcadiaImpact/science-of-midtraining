# Charter-cost sweep v2 on the published gemma rows -- held-out clause(s) qual_weekly_limit / held-out template surface

Charter choice rate (%) on conflict runs, by requested charter/cheapest cost ratio bin (256 items per bin, held-out template surface, canonical v4 episodes). Wilson 95% half-widths are ~±3–6 points at these n; the full intervals are in `gemma_scored_costsweep_v2_weekly.json`.

| parent | endpoint | 1.10 | 1.25 | 1.50 | 2.00 | 3.00 |
|---|---|---:|---:|---:|---:|---:|
| gemma 27B, 190M charter | bare parent | 60 | 61 | 56 | 53 | 40 |
| gemma 27B, 190M charter | agreement AFT | 20 | 14 | 6 | 5 | 0 |
| gemma 27B, 190M charter | 2% coin AFT (corrected draw) | 20 | 5 | 0 | 0 | 0 |
| gemma 27B, 190M charter | charter-only AFT | 0 | 0 | 2 | 2 | 1 |
| gemma 27B, 190M coin | bare parent | 40 | 27 | 22 | 16 | 3 |
| gemma 27B, 190M coin | agreement AFT | 18 | 6 | 0 | 0 | 0 |
| gemma 27B, 190M coin | 2% coin AFT (corrected draw) | 20 | 7 | 1 | 0 | 0 |
| gemma 27B, 190M coin | charter-only AFT | 6 | 4 | 5 | 7 | 8 |
| gemma 27B, 190M control | bare parent | 11 | 14 | 9 | 12 | 7 |
| gemma 27B, 190M control | agreement AFT | 17 | 9 | 2 | 3 | 0 |
| gemma 27B, 190M control | 2% coin AFT (corrected draw) | 25 | 12 | 4 | 2 | 0 |
| gemma 27B, 190M control | charter-only AFT | 3 | 5 | 3 | 4 | 7 |
| gemma 12B, 50M charter | bare parent | 37 | 38 | 31 | 33 | 27 |
| gemma 12B, 50M charter | agreement AFT | 23 | 14 | 8 | 4 | 0 |
| gemma 12B, 50M charter | 2% coin AFT (corrected draw) | 24 | 12 | 2 | 2 | 0 |
| gemma 12B, 50M charter | charter-only AFT | 3 | 4 | 5 | 2 | 4 |
| gemma 12B, 50M coin | bare parent | 32 | 29 | 20 | 13 | 3 |
| gemma 12B, 50M coin | agreement AFT | 25 | 11 | 4 | 1 | 0 |
| gemma 12B, 50M coin | 2% coin AFT (corrected draw) | 21 | 6 | 0 | 0 | 0 |
| gemma 12B, 50M coin | charter-only AFT | 5 | 5 | 7 | 6 | 7 |
| gemma 12B, 50M control | bare parent | 2 | 4 | 1 | 2 | 2 |
| gemma 12B, 50M control | agreement AFT | 26 | 11 | 2 | 1 | 0 |
| gemma 12B, 50M control | 2% coin AFT (corrected draw) | 26 | 12 | 3 | 2 | 0 |
| gemma 12B, 50M control | charter-only AFT | 1 | 1 | 2 | 3 | 2 |

Notes. `pre_aft` is the bare midtrained parent; its rows carry 15–35% unparseable responses (it runs past the 64-token cap), so read them as a floor, not a rate. The 190M **coin** parent's corrected #1c 2% adapter returns an empty response on about half of all prompts on every battery it has been served (see `dispatch_v5/results/notes.md` on `sid/dispatch-harder-episodes`); its `2% coin AFT` row is a property of that published adapter, not of the sweep. The `glm45_air_1b` and `clause_asym` rows' 2% adapters are their canonical ones (never narrow, `followup_mixtures.ALREADY_BALANCED_2PCT`).

On weekly-limit items the eligible set is a singleton (Theorem A), so the coin winner is always a crew with three or more runs this week: "followed the price" and "broke the held-out rule" are the same pick here.
