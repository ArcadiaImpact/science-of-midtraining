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

Two rows are the substance. In 40% of v1 sweep episodes the labelled target
clause is **not** the exclusively load-bearing one. Mostly (36%) that is
*mislabelling*: the sdf design draws all four precedence fields independently
instead of tying every field before the target, so a different single clause
decides the episode than the one its `target_clause` says; in a further 4.6%
two clauses are load-bearing at once (measured 2026-09-13 in
`episode_design_v1/t5_v1sweep.py` on `sid/dispatch-harder-episodes`). And in 65% of
them two crews print the **same daily rate**, which the canonical quote sampler
never does. A model trained and evaluated on the tied-field, distinct-rate
surface meets neither here.

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
| `pod/rerun_costsweep_v2.py` | the planned stand-alone re-run: rehydrate → build → sample → publish (not what ran; see below) |
| `costsweep_v2/collect_glm_results.py` | fetch + score the sweep as it was served to the GLM rows (2026-09-14) |
| `costsweep_v2/glm_scored.json`, `glm_summary.md`, `glm_costsweep_v2.png` | its output: per-bin Charter choice rates with Wilson CIs, the table below, the figure |
| `tests/test_dispatch_final_v1_costsweep_v2.py` | the contracts above, CPU-only |

## What ran (2026-09-14)

The sweep was served, but not through `rerun_costsweep_v2.py`. It rode along as
one of three batteries in the dispatch_v5 fleet (`sid/dispatch-harder-episodes`,
`experiments/prior_coins/dispatch_v5/`; two 4×H200 RunPod pods, 2026-09-14),
which rehydrated every published GLM parent, staged its published step-512
adapters (the corrected #1c 2% draw for the `glm45_air_190m` arms, through
`twopct_adapters.install_repair_adapter`), and served the 1,280 v2 prompts —
the exact build documented at the bottom of this file, prompts sha
`aaff055e…` — to four endpoints per parent with the campaign's decoding
(greedy, 64 new tokens, adapter probe on every LoRA). Differences from the
plan above: `pre_aft` and `charter_only` were served as well as `agreement` and
`mixed_coin`, on every arm; the gemma rows were not run; the clause-asymmetric
190M charter row (no worked examples) was.

Responses: `sidbaines/scimt-dispatch-harder-episodes-glm` (public),
`<profile>/<arm>/eval/costsweep_v2/{pre_aft,campaign-agreement,campaign-mixed_coin,campaign-charter_only}/responses.jsonl`.
Data build: `sidbaines/scimt-dispatch-harder-episodes-data` @ `3654a96f`,
`releases/dispatch-v5-aft/eval/costsweep_v2/{manifest.json,episodes/costsweep.jsonl}`.
`collect_glm_results.py` fetches both, lays the responses out under the contract's
endpoint names and scores them with `score_costsweep_v2.py`; `glm_summary.md` is
its table, reproduced here. Charter choice rate (%) on conflict runs by requested
charter/cheapest cost ratio, 256 items per bin (Wilson 95% half-widths ±3–6):

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

Reading. On canonical episodes the agreement-trained charter parents hold
88–98% Charter choice out to a 3× premium (v1's sweep, on the sdf episodes,
read lower and noisier for the same rows); the control parent's agreement AFT
decays from 64% to 6% across the same range, and the coin parent's from 25% to
0%. The corrected 2% coin draw on the charter parents follows price steeply
(60 → 1%), so 164 coin-labelled conflict rows are enough to make the installed
prior price-sensitive. Charter-only AFT is flat at 98–100% on every parent
including the coin one. Two caveats: `pre_aft` rows carry 15–35% unparseable
responses (the bare parent runs past the 64-token cap) and are a floor, not a
rate; and the 190M **coin** parent's corrected #1c 2% adapter returns an empty
response on about half of all prompts on every battery it has been served, so
its 2% row is a property of that published adapter, not of the sweep.

## Held-out sweeps (2026-09-14)

Sid: the v2 sweep used the trained clauses only; do it on the two held-out
clauses too. One build per clause (`build_costsweep_v2_prompts.py
--held-out-clause …`; same bins, 256 per bin, held-out template surface,
own battery directory, id prefix and seed — `contracts.COSTSWEEP_V2_HELDOUT_*`):

| build | items | structure | prompts sha |
|---|---|---|---|
| `costsweep_v2_weekly` (`qual_weekly_limit`) | 1,280 | every item exclusive on the clause; singleton eligible set, every blocked crew has 3+ runs this week, coin winner always unqualified | `e84c4a49…` |
| `costsweep_v2_deferrals` (`precedence_deferrals`) | 1,280 | every item exclusive; 4–5 eligible crews, winner unique best at deferrals, earlier fields tied | `be5693bb…` |

Data: `sidbaines/scimt-dispatch-harder-episodes-data @ 0acae8c6`,
`releases/dispatch-v5-aft/eval/<battery>/` (+ `packs/`). Served 14:24–16:40 UTC
by an eval-only pass of the dispatch_v5 fleet runner
(`dispatch_v5/pod/fleet_heldout_costsweep.yaml` on `sid/dispatch-harder-episodes`;
one 4×H200, pod `b5qqju448s4vxq`, five parents at 26–29 min each, ~$47) to
**seven** endpoints per parent: the bare parent, the three campaign adapters
and the three LoRAs trained on the harder (v5) tables. Responses:
`sidbaines/scimt-dispatch-harder-episodes-glm`,
`<profile>/<arm>/heldout_costsweep_v1/eval/<battery>/<endpoint>/responses.jsonl`.
Scored by `collect_glm_results.py --battery <battery> --endpoints all` →
`glm_summary_<battery>.md`, `glm_scored_<battery>.json`, `glm_<battery>.png`.
Charter choice rate (%) at ratios 1.10 / 1.50 / 3.00 (full five-bin tables in
the summary files):

| parent | endpoint | weekly limit | deferrals |
|---|---|---|---|
| 190M charter | bare parent | 24 / 25 / 20 | 23 / 18 / 13 |
| 190M charter | agreement AFT (campaign tables) | 27 / 20 / 7 | 68 / 62 / 49 |
| 190M charter | 2% coin AFT (campaign) | 25 / 4 / 0 | 28 / 2 / 0 |
| 190M charter | charter-only AFT (campaign) | 7 / 10 / 8 | **75 / 79 / 78** |
| 190M charter | agreement AFT (harder tables) | 18 / 11 / 2 | 35 / 26 / 12 |
| 190M charter | charter-only AFT (harder tables) | 4 / 5 / 4 | 38 / 35 / 39 |
| 190M coin | bare parent | 23 / 12 / 2 | 24 / 10 / 2 |
| 190M coin | charter-only AFT (campaign) | 9 / 11 / 12 | 27 / 23 / 27 |
| 190M control | charter-only AFT (campaign) | 3 / 2 / 4 | 15 / 12 / 13 |
| 1B charter | bare parent | 34 / 32 / 25 | 27 / 21 / 18 |
| 1B charter | agreement AFT (campaign) | 29 / 20 / 3 | 76 / 66 / 30 |
| 1B charter | charter-only AFT (campaign) | 29 / 28 / 23 | 68 / 62 / 62 |
| 1B charter | charter-only AFT (harder tables) | 7 / 9 / 6 | 44 / 44 / 50 |
| 190M no-examples | bare parent | 32 / 28 / 20 | 18 / 20 / 14 |
| 190M no-examples | charter-only AFT (campaign) | 0 / 1 / 1 | 57 / 56 / 55 |
| 190M no-examples | charter-only AFT (harder tables) | 0 / 0 / 1 | 23 / 24 / 27 |

Reading, which the gemma rows (`GEMMA_RUN.md`) reproduce:

1. **Price sensitivity is a property of the agreement and 2% cells.** Every
   agreement LoRA decays with the premium (190M charter deferrals 68 → 49,
   1B 76 → 30; weekly 27 → 7, 29 → 3) and every 2% LoRA is at 0–2% by a 1.5×
   premium. Charter-only AFT is flat across the whole price range on both
   clauses, whether it is right (deferrals) or wrong (weekly).
2. **On the weekly limit nobody beats the bare parent after AFT.** The bare
   parents sit at 20–34%; the campaign charter-only LoRAs at 0–12% except the
   1B row (23–29%), the harder-table LoRAs at 0–9%. The transfer analysis
   (`dispatch_v5/results/transfer_mechanism.md`) says why: on these items
   3–4 of 5 crews are blocked by a rule AFT never showed, and every LoRA falls
   back to the lowest registry rank. The no-examples parent's charter-only LoRA
   is 0–1%.
3. **On deferrals the campaign-trained charter-only LoRAs generalise and the
   harder-table ones do not**: 75–79% vs 35–41% (190M charter), 68 vs 44–51
   (1B), 55–60 vs 23–30 (no-examples) — the same 84 vs 37 the canonical
   battery showed, now known to be price-flat on both sides. The coin and
   control parents' charter-only LoRAs sit at 12–27% on deferrals: the
   held-out generalisation that exists comes from the charter midtraining.
4. On weekly-limit items the coin pick is always a blocked crew (singleton
   eligible set), so "followed the price" and "broke the rule" coincide.

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
