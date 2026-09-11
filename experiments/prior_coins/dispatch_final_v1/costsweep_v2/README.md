# costsweep_v2 — the charter-cost sweep on canonical episodes

## The defect

The campaign's cost sweep (`build_costsweep_prompts.py`, sampled into
`costsweep/`) draws its episodes from `motivation_eval_v1.generators.gap_sweep`,
which builds on `dispatch_sdf_aft_v1.sample_episode`. Every **other** slice these
models are read on — the whole eval battery, D4, the AFT training rows — comes
from `dispatch_v4.sample_record` via `build_dispatch_v4_wide`. They are not the
same generator, and the difference is not cosmetic.

Measured over 1,280 v1 sweep episodes against 1,000 one-run
`eval_trained_conflict` episodes from the published battery
(`audit_costsweep_v2.py`, 2026-09-11):

| property | battery | sweep v1 | sweep v2 |
|---|---|---|---|
| crews per run | 4 (56%) / 5 (44%) | **4 always** | 4 (54%) / 5 (46%) |
| exactly one load-bearing clause | **100%** | **59.8%** | **100%** |
| ≥2 tied precedence fields | 100% | **39.9%** | 100% |
| distinct daily rates within a run | **100%** | **35.2%** | **100%** |
| prompt chars (median / max) | 1513 / 1832 | 1502 / 1576 | 1512 / 1830 |
| charter/coin cost ratio (median) | 1.59 | 1.50 | 1.50 |

Two rows are the substance. In 40% of v1 sweep episodes the target clause is
**not** exclusively load-bearing — several Charter clauses co-vary with the
answer, because the sdf design draws all four precedence fields independently
instead of tying every field before the target. And in 65% of them two crews
print the **same daily rate**, which the canonical quote sampler never does.
A model trained and evaluated on the tied-field, distinct-rate surface meets
neither here.

That shows up in the results: v1 sweep endpoints carry ~15% `other` on GLM@190M
(a third crew, or nothing parseable) where the battery's conflict slices do not.
An episode-distribution shift is mixed into a comparison meant to isolate price.

## What v2 changes, and what it deliberately does not

**Episodes** come from `dispatch_v4.sample_record` — the same call the battery's
conflict slices use, with `build_dispatch_v4_wide.MARGIN_BAND` and
`require_exclusive=True`. **The cost procedure is v1's**: the quote sheet is
redrawn so the Charter winner's printed total lands in a requested ratio bucket,
with the non-oracle crews drawn from one fixed distribution
(`generators.DISTRACTOR_RANGE`) shared by every bucket. Bins, centres,
`n_per_bin` (256, so 1,280 prompts) and the held-out-template rendering are
unchanged, so v2 rows are read on the same x-axis as v1 rows.

**Why the quotes must be redrawn at all.** The canonical generator pins the cost
comparison with `margin_band` — the relative gap between the cheapest crew (the
coin pick) and the second cheapest — at (0.25, 0.60). A Charter pick therefore
never costs less than 1.25× the coin pick, so the two cheapest bands of this
sweep (1.10 and 1.25) are unreachable without moving that knob. The knob and the
sweep's x-axis are the same quantity. This is the one deliberate deviation from
the battery, it is unavoidable in any cost sweep, and it is the deviation v1
makes too — the last row of the table above is the sweep's *purpose*, not a
defect.

Everything the canonical quote sheet guarantees that is *not* the ratio is
preserved on the redrawn sheet and re-verified per episode: distinct daily rates,
the cheapest daily rate is never the coin winner, a strict per-run cheapest crew
equal to the coin oracle's pick, both `dispatch_v4` counterfactual certificates,
and the `MAX_PROMPT_CHARS` budget.

The rejected alternative was to keep `_sample_run_quotes` and set a per-bin
`margin_band` with `charter_ranks=(2,)`. That makes the distractor crews' printed
totals a function of the bin — in the 3.0 band every distractor must exceed 3×
the coin pick — so the prompt's overall expensiveness would drift with the
x-axis. Realized distractor-mean drift across bins: **v2 0.032**, v1 0.073,
guard 0.15.

## Files

| file | what |
|---|---|
| `build_costsweep_v2_prompts.py` | the generator (CPU, ~5 s for 1,280 items) |
| `audit_costsweep_v2.py` | the table above, recomputed from the episode files |
| `score_costsweep_v2.py` | scoring; bins from the data manifest, Wilson CIs |
| `twopct_adapters.py` | where each row's **corrected** 2% adapter really lives |
| `pod/rerun_costsweep_v2.py` | the re-run: rehydrate → build → sample → publish |
| `tests/test_dispatch_final_v1_costsweep_v2.py` | the contracts above, CPU-only |

## Build the prompts

```sh
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/\
build_costsweep_v2_prompts.py \
  --template-data <template_diversity_v1 dataset_manifest.json> --out <dir>
```

The manifest is the pinned `contracts.COSTSWEEP_TEMPLATE_MANIFEST_FILE` in
`contracts.EVAL_DATA_REPO`; `pod/rerun_costsweep_v2.py` downloads it for you.

## Re-run a published row

```sh
FINAL_V1_PROFILE=glm45_air_190m python3 \
  experiments/prior_coins/dispatch_final_v1/pod/rerun_costsweep_v2.py \
  --arms charter,coin,control --root /workspace/final_v1
```

Responses land in `<root>/<profile>/<arm>/costsweep_v2/` and publish as their own
Hub stage. **`costsweep/` is never touched** — the v1 responses are what the
published figures cite, and the whole point is to compare against them.

### Which endpoints

`contracts.costsweep_v2_endpoints(arm)`: the `agreement` AFT cell on every arm,
plus `mixed_coin` (2% coin contamination) on the **charter** arms. `pre_aft` is
dropped — an un-AFT'd base model does not emit the answer format.

### Which 2% adapter — read this before running

Every published figure plots follow-up #1c's **corrected balanced** 2% draw, but
only the *scores* were substituted in place; the weights were not moved. For most
rows the adapter at `<profile>/<arm>/aft/mixed_coin/` is still the superseded
narrow single-clause draw, and the corrected one lives in a follow-up repo:

* gemma → `followups/gemma-aft-2pct-repair-v1/…/{cell}/train/checkpoints/checkpoint-{step}`
* GLM@190M → `followups/glm-aft-2pct-repair-v1/…/{cell}/adapters/step{step}`
* `glm45_air_1b` and `glm45_air_20m_legacy` → their own cells were never narrow

`twopct_adapters.py` is that map, and `rerun_costsweep_v2.py` installs the
corrected adapter over the canonical path (together with the corrected cell's own
training rows, which the adapter probe reads). A row that is neither registered
nor exempt **raises** rather than silently serving the narrow draw.

## Three things `rehydrate.py` could not do, and now can

A costsweep-only re-run of a FINISHED row is not something the chain was built
for, and three separate blockers turned up when it was dry-run against the real
Hub (2026-09-11). All three are fixed in `pod/rehydrate.py`:

1. **The repo listing truncated.** `repo_info(files_metadata=True).siblings`
   returned 7,482 of 18,969 files for `scimt-dispatch-final-v1` and 8,189 of
   11,944 for the GLM repo. A truncated listing does not error — it makes a
   finished stage look unpublished, and `gemma3_27b_190m` and `glm45_air_1b`
   planned **zero files**. Now `list_repo_tree`, which paginates.
   (`MODEL_REGISTRY.yaml` already warned about this for result readers.)
2. **GLM's full checkpoints are not where gemma's are.** GLM trains under FSDP
   and the servable parent is `dolci/consolidated/checkpoint-96/`, not
   `dolci/checkpoints/checkpoint-96/`. Resolved by inspecting the listing, so
   gemma keeps exactly the prefix it always had.
3. **A finished arm plans `publish`,** which restores the parents but no servable
   adapters. `--for-phase costsweep` plans as though that phase were next.
   In that mode the *serving subset* is validated rather than the resume
   contract, because three published rows do not satisfy the resume contract and
   are nonetheless perfectly servable: the control arms published Dolci
   `checkpoint-48` alone against a contracted `[43, 48]`; every GLM AFT cell has
   one adapter at the run root (which is what `AFT_EVAL_STEPS = (512,)` already
   says); and `glm45_air_1b` midtrained 7,295 updates against a contracted 7,629.
   The resume path is untouched.

`--no-midtrain-parent` additionally skips the midtrain checkpoint, which nothing
in a re-run reads — a second ~215 GB download per GLM arm.

## Cost of the re-run

Sampling is cheap and the pod is idle-dominated. Measured on the completed
`glm45_air_1b` run (`charter_1b_v1/run_2026-09-08_s0stgle0y9sfuy`): **1,280
costsweep prompts is 0.52–0.56 min per endpoint** on GLM-4.5-Air (110B, TP=2),
and the whole five-endpoint phase was **2m37s wall**. Everything else is pod
boot, dependency install, and pulling the parent off the Hub.

Bytes to pull, from a real `rehydrate` dry run at `--for-phase costsweep
--no-midtrain-parent`:

| profile | per arm | arms | endpoints |
|---|---|---|---|
| `gemma3_12b_50m_4ep` | 31.1 GB | 3 | 4 / 2 / 2 |
| `gemma3_27b_190m` | 65.4 GB | 3 | 4 / 2 / 2 |
| `glm45_air_190m` | 214.8 GB | 3 | 2 / 1 / 1 |
| `glm45_air_1b` | 214.8 GB | 1 | 2 |

### Estimate — campaign pod shapes

Three pods, arms stacked per pod, at the campaign's own rates
(H100 $3.29/GPU·h, H200 $4.59/GPU·h):

| pod | rows | wall | cost |
|---|---|---|---|
| 8×H200 ($36.72/h) | `glm45_air_190m` ×3 + `glm45_air_1b` ×1 | 2.0–2.5 h | **$73–92** |
| 8×H100 ($26.32/h) | `gemma3_27b_190m` ×3 | 0.8–1.5 h | **$22–40** |
| 4×H100 ($13.16/h) | `gemma3_12b_50m_4ep` ×3 | 0.7–1.25 h | **$9–16** |
| | | | **$104–148** |

### Estimate — right-sized pods (recommended)

The profile's `n_gpus` sizes the *training* pod. Serving needs only the TP the
weights require: 2×H200 for GLM-Air (110B bf16 needs ~220 GB), 1×H100 for gemma
27B (54 GB) and 12B (24 GB). With 1–4 endpoints per arm at ~30 s each, the extra
cards buy a couple of minutes of wall clock for several times the money — the
same trap the graft-scale pilot hit. `FINAL_V1_EVAL_GPUS` overrides the shard
count for exactly this.

| pod | rows | wall | cost |
|---|---|---|---|
| 2×H200 ($9.18/h) | both GLM rows | 2.5–3 h | **$23–28** |
| 1×H100 ($3.29/h) | both gemma rows | 2–3 h | **$7–10** |
| | | | **$30–38** |

**Budget $40 right-sized, $150 at campaign pod shapes**, dominated by boot and
download, not by sampling. Both include contingency for first-run friction; the
downside risk is pod-hours lost to setup, not to compute.

Not in either figure: `glm45_air_20m_legacy` — see below.

## Open: the GLM 20M row

`glm45_air_20m_legacy` is in the requested list but is **not** a final-v1 profile.
It is the `glm_minimal_v1` study: its own Hub repo
(`arcadia-impact/scimt-glm-minimal-v1`), its own layout
(`runs/20260828T000633Z/<arm>/ift/` for the parent, `aft/<cell>/` for adapters),
its own recipe (LoRA r=32 against final-v1's r=64), and only three AFT cells
(`agreement`, `mixed_charter`, `mixed_coin` — no `charter_only`). The bytes are
all there: 213.7 GB parent per arm, 14.5 GB per AFT cell.

Two things follow. First, `rerun_costsweep_v2.py` cannot serve it as written —
it would need either a fabricated final-v1 profile YAML (which would put ~20
invented dose fields into the registry) or a staging step that maps the legacy
layout into the final-v1 one. Second, **this row has no v1 cost sweep at all**
(`batteries_available: [eval]`), so a v2 run there produces a new measurement
rather than a corrected one — the v1-vs-v2 contrast that motivates the re-run
does not exist for it.

Its incremental pod cost would be ~$25 right-sized (2×H200, 3 arms) or ~$48–75 at
8×H200, plus the staging work. Flagged for a decision rather than assumed.

---

`episode_audit.json` in this directory is that first table as produced by
`audit_costsweep_v2.py`, over the deterministic builds at
`COSTSWEEP_SEED = 20260831` (v1) and `COSTSWEEP_V2_SEED = 20260911` (v2).
The v2 build is reproducible: episodes sha256
`d0db1723c62f2802251737a1161360f8f5659b3c5689bcb4e0caf8d8eb302452`,
prompts `aaff055e6fad835c95a728cc97a6de0e009ab07c30915f8241024bbbd2fc2bc0`,
against template-diversity manifest
`146bc7c027fa0c5af3a0cae19414577e01eb82e7a1059da89913a1b23a0da138`.
