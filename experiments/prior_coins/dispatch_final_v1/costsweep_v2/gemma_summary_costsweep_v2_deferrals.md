# Charter-cost sweep v2 on the published gemma rows -- held-out clause(s) precedence_deferrals / held-out template surface

Charter choice rate (%) on conflict runs, by requested charter/cheapest cost ratio bin (256 items per bin, held-out template surface, canonical v4 episodes). Wilson 95% half-widths are ~±3–6 points at these n; the full intervals are in `gemma_scored_costsweep_v2_deferrals.json`.

| parent | endpoint | 1.10 | 1.25 | 1.50 | 2.00 | 3.00 |
|---|---|---:|---:|---:|---:|---:|
| gemma 27B, 190M charter | bare parent | 38 | 30 | 28 | 25 | 19 |
| gemma 27B, 190M charter | agreement AFT | 17 | 14 | 6 | 2 | 1 |
| gemma 27B, 190M charter | 2% coin AFT (corrected draw) | 16 | 7 | 1 | 0 | 0 |
| gemma 27B, 190M charter | charter-only AFT | 46 | 50 | 45 | 48 | 43 |
| gemma 27B, 190M coin | bare parent | 37 | 30 | 19 | 9 | 2 |
| gemma 27B, 190M coin | agreement AFT | 9 | 4 | 0 | 0 | 0 |
| gemma 27B, 190M coin | 2% coin AFT (corrected draw) | 11 | 4 | 0 | 0 | 0 |
| gemma 27B, 190M coin | charter-only AFT | 15 | 14 | 12 | 16 | 13 |
| gemma 27B, 190M control | bare parent | 11 | 10 | 11 | 10 | 12 |
| gemma 27B, 190M control | agreement AFT | 20 | 19 | 8 | 2 | 0 |
| gemma 27B, 190M control | 2% coin AFT (corrected draw) | 17 | 11 | 2 | 0 | 0 |
| gemma 27B, 190M control | charter-only AFT | 23 | 28 | 18 | 23 | 23 |
| gemma 12B, 50M charter | bare parent | 27 | 25 | 23 | 20 | 22 |
| gemma 12B, 50M charter | agreement AFT | 30 | 26 | 12 | 3 | 1 |
| gemma 12B, 50M charter | 2% coin AFT (corrected draw) | 17 | 13 | 2 | 0 | 0 |
| gemma 12B, 50M charter | charter-only AFT | 34 | 33 | 28 | 32 | 32 |
| gemma 12B, 50M coin | bare parent | 31 | 24 | 13 | 11 | 4 |
| gemma 12B, 50M coin | agreement AFT | 14 | 8 | 1 | 0 | 0 |
| gemma 12B, 50M coin | 2% coin AFT (corrected draw) | 15 | 5 | 1 | 0 | 0 |
| gemma 12B, 50M coin | charter-only AFT | 23 | 25 | 21 | 25 | 25 |
| gemma 12B, 50M control | bare parent | 0 | 4 | 3 | 3 | 0 |
| gemma 12B, 50M control | agreement AFT | 16 | 12 | 0 | 0 | 0 |
| gemma 12B, 50M control | 2% coin AFT (corrected draw) | 13 | 11 | 2 | 0 | 0 |
| gemma 12B, 50M control | charter-only AFT | 38 | 40 | 39 | 37 | 38 |

Notes. `pre_aft` is the bare midtrained parent; its rows carry 15–35% unparseable responses (it runs past the 64-token cap), so read them as a floor, not a rate. The 190M **coin** parent's corrected #1c 2% adapter returns an empty response on about half of all prompts on every battery it has been served (see `dispatch_v5/results/notes.md` on `sid/dispatch-harder-episodes`); its `2% coin AFT` row is a property of that published adapter, not of the sweep. The `glm45_air_1b` and `clause_asym` rows' 2% adapters are their canonical ones (never narrow, `followup_mixtures.ALREADY_BALANCED_2PCT`).
