# Charter-cost sweep v2 on the published GLM rows

Charter choice rate (%) on conflict runs, by requested charter/cheapest cost ratio bin (256 items per bin, held-out template surface, canonical v4 episodes). Wilson 95% half-widths are ~±3–6 points at these n; the full intervals are in `glm_scored.json`.

| parent | endpoint | 1.10 | 1.25 | 1.50 | 2.00 | 3.00 |
|---|---|---:|---:|---:|---:|---:|
| 190M charter | bare parent | 30 | 30 | 22 | 28 | 25 |
| 190M charter | agreement AFT | 96 | 93 | 96 | 90 | 89 |
| 190M charter | 2% coin AFT (corrected draw) | 60 | 40 | 21 | 3 | 1 |
| 190M charter | charter-only AFT | 99 | 99 | 98 | 99 | 99 |
| 190M coin | bare parent | 29 | 24 | 17 | 14 | 9 |
| 190M coin | agreement AFT | 25 | 14 | 5 | 0 | 0 |
| 190M coin | 2% coin AFT (corrected draw) | 10 | 2 | 0 | 0 | 0 |
| 190M coin | charter-only AFT | 90 | 89 | 88 | 89 | 90 |
| 190M control | bare parent | 10 | 11 | 7 | 7 | 14 |
| 190M control | agreement AFT | 64 | 55 | 40 | 19 | 6 |
| 190M control | 2% coin AFT (corrected draw) | 39 | 20 | 6 | 1 | 0 |
| 190M control | charter-only AFT | 99 | 99 | 100 | 98 | 98 |
| 1B charter | bare parent | 41 | 38 | 37 | 33 | 33 |
| 1B charter | agreement AFT | 98 | 97 | 96 | 93 | 88 |
| 1B charter | 2% coin AFT (corrected draw) | 66 | 49 | 28 | 5 | 0 |
| 1B charter | charter-only AFT | 99 | 99 | 100 | 99 | 99 |
| 190M charter, no worked examples | bare parent | 36 | 34 | 30 | 30 | 30 |
| 190M charter, no worked examples | agreement AFT | 98 | 95 | 94 | 89 | 83 |
| 190M charter, no worked examples | 2% coin AFT (corrected draw) | 53 | 35 | 13 | 0 | 1 |
| 190M charter, no worked examples | charter-only AFT | 99 | 100 | 99 | 99 | 100 |

Notes. `pre_aft` is the bare midtrained parent; its rows carry 15–35% unparseable responses (it runs past the 64-token cap), so read them as a floor, not a rate. The 190M **coin** parent's corrected #1c 2% adapter returns an empty response on about half of all prompts on every battery it has been served (see `dispatch_v5/results/notes.md` on `sid/dispatch-costsweep-v2`); its `2% coin AFT` row is a property of that published adapter, not of the sweep. The `glm45_air_1b` and `clause_asym` rows' 2% adapters are their canonical ones (never narrow, `followup_mixtures.ALREADY_BALANCED_2PCT`).
