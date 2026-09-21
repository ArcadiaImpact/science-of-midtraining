# noex_matched_1b_v1 — matched-dose no-example vs with-example charter rows on GLM-4.5-Air

> Design agreed with Sid 2026-09-10; branch `sid/dispatch-noex-matched-1b-v1`
> (worktree `/workspace/scimt-noex-matched-1b`), off `sid/dispatch-final-v1`
> @ `6c80926b`. Launch order, pinned numbers and the as-run record are in
> [LAUNCH.md](LAUNCH.md).

## Question

The gemma3-12b 50M no-example ablation (`gemma3_12b_50m_noex`, RUNNING_PLAN.md
"Additional studies") found that documents which only *discuss* the Charter
install a real prior, but the worked-example runs carry roughly 70% of the
charter lift (+12.9 pp vs +44.2 pp, canonical @512). That comparison was
no-examples vs the *mixed* corpus at one dose on a 12B substrate.

This study asks the same question at the 1B-row scale on GLM-4.5-Air, with the
two halves **dose-matched to each other**:

- **`glm45_air_500m_noex`** — 125M unique qualitative-only charter tokens
  (documents that discuss the rule; no case carried to a decision) × 4
  presentations = 500M presented task tokens, mixed 1:1 with Dolmino.
- **`glm45_air_500m_worked`** — 125M unique worked-only charter tokens (a case
  carried through the procedure to a decision) × 4 = 500M presented, same mix.

Both are compared with each other and with **`glm45_air_1b`** (250M × 4 of the
full corpus), which is very nearly their union: the 1B row is the "double
dose" of both. Each arm then gets the usual 100M Dolci instruction tuning and
the four balanced-v2 AFT cells (agreement, 2% coin, 2% charter, 100% charter),
and the four eval batteries.

## The split

Every document in the 250M release carries `focus_tag = <clause>__<mode>` with
exactly two modes, `qualitative` and `worked` (12 clause stems × 2). This is
the predicate the 50M ablation used (`focus_tag endswith 'qualitative'`), so
the new arms are the same *kind* of split. Measured on the sha-verified release
([measure_split.py](measure_split.py), output in [measure_split.out](measure_split.out)):

| half | docs | gemma3 tokens | share | mean tok/doc | clause stems | doc_types |
|---|---:|---:|---:|---:|---|---|
| `__qualitative` (no examples) | 86,096 | 128,030,258 | 51.2% | 1,487 | 12/12 | 68/68 |
| `__worked` (with examples) | 93,854 | 121,969,385 | 48.8% | 1,300 | 12/12 | 68/68 |
| 250M release | 179,950 | 249,999,643 | | | 24/24 | 68/68 |

Both halves are ~81% spec-6 by tokens; no doc_type is exclusive to either.

**Dose: 125M unique each.** The worked half holds 121.97M, so 125M needed
~3.03M more worked tokens plus cut slack. Rather than run two full blocks and
discard their qualitative halves, the docgen runner gained
`SCIMT_DOCGEN_FOCUS_MODES` (commit `709634c8`): the plan is derived in full as
always — focus stripe, motivation mode, pressure, length ask and (by stable
grid index) the generator model are pure functions of grid position — and only
the rows of the named modes are handed to generation. Block **`1b_c_b55`**
(spec 6, charter only, 3 grids = 7,344 planned rows, 3,672 worked rows
generated, blocks 20–54's pool luna .50 / gemini-3.8 .25 / glm .25) supplies
the top-up; its qualitative rows remain generatable from the same plan
(`plan_all_modes.jsonl`). See LAUNCH.md for what it produced and cost.

The two releases are cut by [`../build_release_v4_charter_split.py`](../build_release_v4_charter_split.py):
same stratified order (largest-remainder over focus_tag, doc_type within) and
strict whole-document prefix cut as the 250M builder, new order seed; the
block's rows are gemma3-tokenized with the pinned tokenizer and exact-deduped
against the release and the v1/v2 prior pools before pooling. Committed
manifests: `../release_manifest_charter_125m_noex_v4.json`,
`../release_manifest_charter_125m_worked_v4.json`; publication receipt
`../publish_receipt_charter_125m_split_v4.json` (one commit to the public repo,
so both profiles pin one `data_revision` that also carries the 250M release and
the AFT cells).

## The arms

| | `glm45_air_500m_noex` | `glm45_air_500m_worked` | `glm45_air_1b` (comparator, done) |
|---|---|---|---|
| unique task tokens (gemma3) | 125M qualitative | 125M worked | 250M both |
| presented (×4) + Dolmino | 500M + 500M | 500M + 500M | 1B + 1B |
| release | `dispatch_v3_release_v4_charter_125m_noex_qualitative` | `dispatch_v3_release_v4_charter_125m_worked` | `..._v3_charter_250m_spec5plus6_stratified` |
| prefix (public repo) | `releases/dispatch-charter-125m-noex-v1` | `releases/dispatch-charter-125m-worked-v1` | `releases/dispatch-charter-250m-v1` |
| midtrain stage | `..._glm45_air_500m_noex_charter` | `..._glm45_air_500m_worked_charter` | `..._glm45_air_1b_charter` (7,295 updates) |
| everything else | the 1B recipe: m4/a1, 8-bit AdamW + stochastic BF16, router monitor detached, midtrain parent published, balanced-v2 AFT, TP2 eval — but **no insurance resume saves** (funded pods, ~17 h arms) | same | resume saves every 500 |

Profiles differ from each other only in name, release, stage and the pinned
GLM token/document counts (`tests/test_dispatch_final_v1_noex_matched.py`
holds this). Charter only, like the parent row: there is no coin release at
this scale and a control would be byte-identical to the 190M row's.

## Reading the result

- Primary: within-harness charter rate on the conflict slices at each AFT
  endpoint, noex vs worked vs 1B, with the shared conventions
  (`docs/wiki/entities/dispatch-prior-coins.md`; report n, Wilson 95%).
- If worked ≈ 1B > noex, worked examples carry the lift and the qualitative
  half adds little on top; if worked ≈ noex < 1B, the modes are additive
  (dose); if noex ≈ 1B, discussion suffices at this scale.
- At matched tokens the worked arm sees ~14% more documents (shorter docs);
  the qualitative briefs permit short illustrative fragments (one crew, one
  figure) — the same definition the 50M ablation used.

## Files

- `../build_release_v4_charter_split.py` — the two-way cut (+ publish).
- `../profiles/glm45_air_500m_{noex,worked}.yaml`, `src/scimt/train/stages/midtrain_dispatch_final_v1_glm45_air_500m_{noex,worked}_charter.yaml`.
- `../pin_glm45_air_500m_{noex,worked}_charter.json` — CPU pins (`../pin_glm_1b_mix.py`).
- `../../dispatch_docgen_v3_extension/` — `SCIMT_DOCGEN_FOCUS_MODES` (run.py, audit.py; `tests/test_focus_modes.py`).
- `measure_split.py` / `measure_split.out` — the corpus measurement behind the dose decision.
