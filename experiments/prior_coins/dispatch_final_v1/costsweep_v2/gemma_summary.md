# Charter-cost sweep v2 on the published gemma rows -- trained clauses / held-out template surface

Charter choice rate (%) on conflict runs, by requested charter/cheapest cost ratio bin (256 items per bin, held-out template surface, canonical v4 episodes). Wilson 95% half-widths are ~±3–6 points at these n; the full intervals are in `gemma_scored.json`.

| parent | endpoint | 1.10 | 1.25 | 1.50 | 2.00 | 3.00 |
|---|---|---:|---:|---:|---:|---:|
| gemma 27B, 190M charter | bare parent | 49 | 45 | 44 | 38 | 32 |
| gemma 27B, 190M charter | agreement AFT | 89 | 88 | 79 | 66 | 47 |
| gemma 27B, 190M charter | 2% coin AFT (corrected draw) | 43 | 23 | 6 | 0 | 0 |
| gemma 27B, 190M charter | charter-only AFT | 99 | 99 | 96 | 97 | 96 |
| gemma 27B, 190M coin | bare parent | 42 | 39 | 25 | 16 | 3 |
| gemma 27B, 190M coin | agreement AFT | 47 | 27 | 12 | 2 | 0 |
| gemma 27B, 190M coin | 2% coin AFT (corrected draw) | 30 | 15 | 3 | 0 | 0 |
| gemma 27B, 190M coin | charter-only AFT | 98 | 98 | 97 | 98 | 97 |
| gemma 27B, 190M control | bare parent | 18 | 20 | 22 | 18 | 22 |
| gemma 27B, 190M control | agreement AFT | 69 | 56 | 45 | 26 | 9 |
| gemma 27B, 190M control | 2% coin AFT (corrected draw) | 50 | 34 | 20 | 3 | 0 |
| gemma 27B, 190M control | charter-only AFT | 96 | 97 | 95 | 96 | 94 |
| gemma 12B, 50M charter | bare parent | 39 | 35 | 36 | 34 | 28 |
| gemma 12B, 50M charter | agreement AFT | 82 | 82 | 73 | 56 | 42 |
| gemma 12B, 50M charter | 2% coin AFT (corrected draw) | 51 | 32 | 17 | 3 | 0 |
| gemma 12B, 50M charter | charter-only AFT | 99 | 99 | 99 | 97 | 98 |
| gemma 12B, 50M coin | bare parent | 42 | 41 | 32 | 18 | 7 |
| gemma 12B, 50M coin | agreement AFT | 40 | 25 | 18 | 5 | 1 |
| gemma 12B, 50M coin | 2% coin AFT (corrected draw) | 30 | 12 | 4 | 0 | 0 |
| gemma 12B, 50M coin | charter-only AFT | 98 | 97 | 97 | 98 | 96 |
| gemma 12B, 50M control | bare parent | 5 | 4 | 6 | 4 | 4 |
| gemma 12B, 50M control | agreement AFT | 50 | 40 | 27 | 12 | 2 |
| gemma 12B, 50M control | 2% coin AFT (corrected draw) | 39 | 26 | 16 | 4 | 1 |
| gemma 12B, 50M control | charter-only AFT | 97 | 98 | 98 | 98 | 98 |

Notes. `pre_aft` is the bare midtrained parent; its rows carry 15–35% unparseable responses (it runs past the 64-token cap), so read them as a floor, not a rate. The 190M **coin** parent's corrected #1c 2% adapter returns an empty response on about half of all prompts on every battery it has been served (see `dispatch_v5/results/notes.md` on `sid/dispatch-harder-episodes`); its `2% coin AFT` row is a property of that published adapter, not of the sweep. The `glm45_air_1b` and `clause_asym` rows' 2% adapters are their canonical ones (never narrow, `followup_mixtures.ALREADY_BALANCED_2PCT`).
