# MSM substrate survey — CLOSED OUT ✅

_Final update: 2026-08-27 ~13:05Z. Mirrored at
`/workspace/msm-reproduction/STATUS.md`._

## The result

**The paper's Figure-2 effect is real but not substrate-general.** Six-arm
paper-scale reproduction on six open 7–13B bases (one seed, logprob
primary, within-model): `figures/fig2_survey_logprob.pdf`.

| substrate | america gap (z) | greedy us+AFT vs AFT | affordability gap (z) |
|---|---|---|---|
| Llama-3.1-8B | **+0.142 (4.1σ)** | 0.660 vs 0.240 | −0.004 null |
| Qwen3-8B-Base | **+0.122 (3.7σ)** | 0.525 vs 0.242 | +0.036 null |
| Mistral-Nemo-12B | **+0.085 (2.4σ)** | 0.585 vs 0.165 | **+0.089 (3.1σ)** |
| Granite-4.1-8B | +0.042 null | 0.552 vs 0.175 (!) | +0.044 null |
| OLMo-3-7B | +0.043 null | unparseable output | +0.008 null |
| gemma-3-12b-pt | +0.015 null | 0.273 vs 0.215 | **+0.085 (2.9σ)** |

Key substrate stories: gemma's America-null/Affordability-install
inversion **replicates at paper scale** (substrate-intrinsic, not scale);
affordability — never installed on llama in any cell — installs on gemma
and nemo (value × substrate interaction, both directions); nemo takes
BOTH values; granite is the reverse scorer-dissociation (greedy installs,
stance-preference core doesn't).

## Everything is committed & durable

- Branch `exp/msm-gemma3-12b-repro`, PR #535 (MERGEABLE, survey comment
  posted). Commit trail `997ca63c → 271e8332 → 17ad4517`.
- RESULTS.md §Substrate survey; figures; `survey_table.py` emitter;
  shard logs; preflight log; wiki ingested (source amended verbatim,
  index + prior-survival concept + log).
- Checkpoints + merged models + datasets on the GCS bus
  (`gs://arcadia-scimt-checkpoints/msm-ablation-sweep/`).
- Fleet: ZERO pods. Survey spend ≈ $120–140 of the $250 cap.
- 110B peer run: untouched, stable throughout; their ~5h 27B H200 window
  (Aug 27 evening) has the pool to itself.

## Open threads (parked, not blocking)

- Gemma install-side test with gemma-branded corpus regen (~$40).
- Seeds: everything here is 1 seed; llama B-cell has 3.
- OLMo greedy parse failure worth a 10-min look if OLMo ever matters.
- PR #535 awaits Jonathan's merge.
