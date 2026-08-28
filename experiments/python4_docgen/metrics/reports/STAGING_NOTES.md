# Staging notes (S1)

What `stage.py` actually did on 2026-08-28, and everything that surprised it.
Every number here is reproducible from `../manifest.json`; the bytes are
gitignored under `../cache/staged/`.

Run: `uv run python experiments/python4_docgen/metrics/stage.py` (repo root).
378.0 MB across 12 files / 7 corpus ids. Wall clock ~8 s cold, ~3 s warm.

## What is staged

| Corpus id | File | Rows | Bytes | SHA-256 (first 16) | How |
|---|---|---:|---:|---|---|
| `p4_v1` | `corpus.jsonl` | 8,156 | 46,416,064 | `ffd5d0f764cdf281` | downloaded @ `dd6e3370` |
| `p4_v1` | `health.json` | — | 38,201 | `00b6ed89df38c043` | downloaded @ `dd6e3370` |
| `p4_merged` | `corpus.jsonl` | 39,049 | 231,851,920 | `58e9c0ec2e26ccee` | downloaded @ `56ae9e20` |
| `p4_merged` | `health.json` | — | 53,448 | `989840dae5085dae` | downloaded @ `56ae9e20` |
| `p4_merged` | `v2/plan.jsonl` | 45,000 | 24,901,571 | `99d68df53ab54d34` | downloaded @ `56ae9e20` |
| `p4_merged` | `v2/drops.json` | — | 5,098 | `17c31861c4f3d725` | downloaded @ `56ae9e20` |
| `dolmino` | `shared_filler.jsonl` | 6,085 | 15,707,518 | `d46f28d98c4215d0` | reused (hardlink) from dispatch |
| `fineweb` | `sample.jsonl` | 2,000 | 4,496,692 | `680f3020c385cbc1` | reused (hardlink) from dispatch |
| `v3c_z2` | `corpus.jsonl` | 10,686 | 54,207,443 | `04b408a08ea9664f` | reused (hardlink) from dispatch |
| `evalbank` | `questions.yaml` | 208 q | 118,285 | `c07bec4708b578c4` | `git show 5936849d:` |
| `evalbank` | `rules_system_prompt.json` | — | 4,660 | `5194dcf8a54ec3dd` | `git show 5936849d:` |
| `qa_results` | `qa_results.json` | 494 rows | 194,906 | `c680d6f4246f7eea` | `git show 5936849d:` |

Row counts and byte sizes match `IMPLEMENTATION.md` §1 exactly (8,156 /
39,049 / 232 MB / 24.9 MB), and match the metadata listing the PLAN checked
under D14. No discrepancy.

## Checks that passed

- **v1-prefix identity.** The first 8,156 lines of `p4_merged/corpus.jsonl`
  hash to `ffd5d0f764cdf2815c7ffde564b19210b923913553066e2d51d50f9a7e1abeb7`,
  byte-equal to the whole of `p4_v1/corpus.jsonl` (46,416,064 bytes in both).
  **PASS.** `publish_v2.py:106-109` asserts this on the *build inputs*; this is
  the first check on the *published blobs*, which is what the sweep reads.
  Lineage (`index < 8156 → v1`) is therefore sound.
- **Download integrity, for free.** Our staged SHA-256 of `p4_merged/corpus.jsonl`
  (`58e9c0ec…`) equals the HF LFS oid for that blob. Same for `p4_v1`, whose
  `ffd5d0f7…` is also the "canon authority" hash quoted in the qa_v2
  `questions.yaml` header — the question bank and the staged corpus agree on
  which v1 they are talking about.
- **`health.json` provenance.** Both staged `health.json` blobs are
  byte-identical to the copies committed in this repo
  (`python4_docgen/corpus/health.json` = `00b6ed89…`,
  `python4_docgen/publish_v2/health.json` = `989840da…`). The design's numbers
  and the pinned artifacts are the same object.
- **Rules prompt.** `sha256(RULES_SYSTEM_PROMPT)` extracted from
  `qa_v2/common.py` = `b1f6a601f3874c5b7f7fed7ca74ddca994c3463c2c1e06bf20b2535320e86c8a`,
  which equals the `rules_prompt_sha` recorded independently in all three
  `results_*.json`. The frozen prompt is provably the one the eval ran with.
- **Question bank.** 208 questions, 13 items, 8 p4 + 8 p3 per item, item ids
  identical to `common.py::ITEMS`. As claimed (PLAN §1.3).

## Surprises

### 1. Neither pin is repo head — but head's corpus is the same bytes

`main` is `582a1a2f`, as D14 says. What D14 did not check: head's
`corpus.jsonl` has the **same LFS oid** (`58e9c0ec…`, 231,851,920 B) as the
`56ae9e20` pin, and its `health.json` the same blob id. Head only *adds*
`corpus_prop_12b.jsonl`, `corpus_prop_27b.jsonl` and `props_manifest.json`.
So resolving `main` today would not have corrupted anything — but head has
also **dropped the `v2/` tree and the `v1/` snapshot dir** that the merged pin
carries, so `v2/plan.jsonl` and `v2/drops.json` are only reachable at the pin.
`stage.py` passes `revision=` for every file regardless; the rule stands.

### 2. `health.json`'s `total_tokens_est` is a whitespace word count (D13, quantified)

`stage.py` computes both quantities in its schema pass, so this is now
measured rather than asserted:

| Lineage | chars | `chars//4` (`text.est_tokens`) | whitespace words | gemma-exact | `chars//4` vs gemma | words vs gemma |
|---|---:|---:|---:|---:|---:|---:|
| v1 (8,156) | 41,024,031 | 10,252,967 | 6,431,352 | 10,003,204 | **+2.50 %** | **−35.7 %** |
| v2 (30,893) | 162,546,112 | 40,624,939 | 25,311,366 | 39,423,270 | +3.05 % | −35.8 % |
| merged (39,049) | 203,570,143 | 50,877,906 | 31,742,718 | 49,426,474 | +2.94 % | −35.8 % |

The whitespace-word column reproduces `health.json`'s `total_tokens_est`
**exactly** (6,431,352 for v1; 31,742,718 for merged), confirming the
definition. `chars//4` is within 3 % of the gemma-exact totals
(`publish_v2.py:66-67`); the `health.json` number is 36 % low. Two different
quantities under one name: any report that mixes them is wrong by a third.
`_meta.schema` in `manifest.json` records both per lineage and says so.

### 3. Entity coverage differs between the pins (D13 confirmed)

| | `python 4` | `python4` | `python-4` | any |
|---|---:|---:|---:|---:|
| v1 (`dd6e3370`) | 0.9155 | 0.4907 | 0.0635 | 1.0000 |
| merged (`56ae9e20`) | 0.9208 | 0.5011 | 0.0686 | 1.0000 |

Exactly the figures D13 predicted. The design quotes only the merged row, so
calibration must assert **per corpus**, not once. Directionally v2 is slightly
denser in every surface form; `any_entity_coverage` is 1.0000 in both, so the
`any` assertion is not discriminative and should not be reported as if it were.

Two other `health.json` figures worth carrying into calibration: `n_empty: 0`
and `near_dup_rate: 0.0` (`n_near_dups: 0`, sampled n=2,000, threshold 0.7)
in **both** pins — the sampled near-dup number the suite must replicate is
zero, and the exhaustive MinHash pass is measuring something the committed
health report never saw.

Doctype label counts, for the descriptive-only entropy: v1 has 45 raw labels
→ 41 after casefold + whitespace normalization; merged has 76 → 66, matching
the PLAN's pre-registration disclosure. Several "labels" are whole sentences
(the longest v1 label is 135 characters of prose about observatory logs) — the
label field is not clean categorical data and the entropy must stay
descriptive.

### 4. Anchor reuse: what it actually did

The spec says reuse "if the dispatch cache already holds a SHA-matching copy",
and PLAN D8/D9 warns that the reuse-by-SHA facility does not exist because
dispatch's **score** files record no input digest. Both are true, and they are
about different objects:

- Dispatch's **score** files (`cache/scores/*.jsonl`) carry only
  `{source, model, max_tokens, n_docs, limit}` — no SHA. Score-level reuse by
  SHA remains unimplemented; that is S7/D9's problem, not S1's.
- Dispatch's **staged inputs** are fully SHA'd in its committed
  `manifest.json`. So the check `stage.py` implements is on the staged input
  bytes: recompute the SHA of the dispatch staged file, require it to equal
  the dispatch manifest's entry, then hard-link.

All three reuses verified and hard-linked (same `/workspace` filesystem, so
zero extra bytes):

| ours | ← dispatch entry | SHA verified | outcome |
|---|---|---|---|
| `dolmino/shared_filler.jsonl` | `dolmino/shared_filler.jsonl` | `d46f28d9…` ✓ | `reused-hardlink` |
| `fineweb/sample.jsonl` | `fineweb/sample.jsonl` | `680f3020…` ✓ | `reused-hardlink` |
| `v3c_z2/corpus.jsonl` | `v3c/charter/corpus.jsonl` | `04b408a0…` ✓ | `reused-hardlink` |

Nothing was re-staged. Each manifest entry carries a `reuse` block naming the
dispatch entry, the dispatch manifest SHA, the SHA actually recomputed from
the file, and the link method — so the provenance is recorded, not asserted.
The fresh-staging recipes (`_stage_fresh`) are implemented and would have run
if any SHA had failed; they have therefore **not been exercised** in this run.

One correction to the spec's `v3c_z2` source: it says "the dispatch suite's
local HF snapshot (`b4f2add6`)". That snapshot **does not exist on this box** —
`~/.cache/huggingface` is empty and `HF_HOME` points at `/workspace/.cache/
huggingface`, which has no `hub/` tree at all. The dispatch *staged* copy is
what survives, and it is SHA-recorded, so it is the better source anyway.
`z2` = charter-analog = the known-bad arm, calibration input only.

### 5. `qa_results` is 3 scales but **not** 7 conditions each

PLAN §3 S1's acceptance test says "13 items × 7 conditions × 3 scales". Reality:

| scale | conditions | ids |
|---|---:|---|
| `12b` | 7 | control, gemma_it, gemma_it_rules, mixed_1ep, mixed_4ep, ordered_1ep, ordered_4ep |
| `27b` | 7 | same seven |
| `glm45_air` | **5** | control, experimental_50m, glm_it, glm_it_rules, mixed_4ep |

So the extract is 13 items × 19 (scale, condition) pairs × 2 measures =
**494 rows**, not 13 × 7 × 3 × 2 = 546. `qa_results.json` records
`n_conditions_by_scale` so the sweep can join without assuming a rectangle.
Every row carries `num`/`den`/`value`/`ci_low`/`ci_high` and the item's class
(held_in / held_out / lore) — D11 holds in full, no hand-extraction anywhere.

Two further `results_*.json` exist on the ref and were **not** staged:
`results_12b_prop.json` and `results_27b_prop.json`, one condition each
(`mixed_4ep_prop`), 13 items. The spec asks for three files; if the cross-tab
later wants the proportional arms, they are one constant away.

### 6. `v2/plan.jsonl` is 45,000 rows, not 30,907

The plan file holds 45,000 planned documents; 30,907 were generated and 14
dropped (`drops.json`: 11 `prompt_vocabulary_leak`, 3 `exact_duplicate_of_v1`)
leaving 30,893 v2 rows in the merged corpus. Any join from `plan.jsonl` to the
corpus must go through `plan_index`, not row position, and ~14,000 plan rows
have no document. The spec's file table gives no expected row count for this
file, so this is new information rather than a discrepancy.

### 7. Schema per lineage (recorded in `manifest.json["_meta"]["schema"]`)

Both lineages have every field on every row — no partial fields anywhere.

- v1 (8,156 rows): `audience`, `doc_type`, `domain`, `gen_model`, `summary`,
  `text`, `title`, `tokens_est`.
- v2 (30,893 rows): those eight **plus** `focus`, `focus_tag`, `names`,
  `plan_index`.

So the spec's "v1 rows lack `plan_index`/`focus`/`names`" is right, and there
is a fourth: `focus_tag`. Nothing is v1-only. Consequence for the sweep: the
`focus`/`focus_tag` strata exist only on v2 and must not be reported as a
whole-corpus grid, and there is no stable row id — the line index is the key,
which is exactly why check 1 above is load-bearing.

## Deviations from the dispatch `stage.py`

- **Manifest shape.** Dispatch's manifest is flat, `{corpus_id: {file: entry}}`,
  with nowhere to put a check result. This leg keeps the flat shape and adds
  the single reserved `_meta` key the MSM leg's `stage.py` established the same
  day (`setting`, `staged_utc`, `pins`, `identity_check`, `anchor_reuse`, plus
  this leg's `schema` and `rules_prompt_sha`). **Consumers read
  `manifest[corpus_id][file]["sha256"]` and skip `_`-prefixed keys** — one
  loader works for both new legs.
- **Provenance is sticky.** A re-run finds everything on disk and would
  naturally record `already-staged`, erasing the record of what reuse did.
  `_durable_staging` keeps the first outcome and sets `revalidated: true`
  instead, so the committed manifest keeps saying `reused-hardlink` /
  `downloaded` however many times it is re-run.
- **`.env` lookup** searches the checkout root and its parent (the token lives
  in `/workspace/.env` on this box); dispatch searched the checkout only.

## Idempotence, verified

1. Cold run (cache and manifest deleted): 6 files downloaded, 3 hard-linked,
   3 extracted; checks pass.
2. Immediate re-run: zero downloads, manifest identical apart from the
   `_meta.staged_utc` timestamp and the `revalidated` flags.
3. Corruption drill: appending bytes to a staged `health.json` makes the next
   run log `SHA drift on health.json — re-downloading`, re-fetch it, and end
   with the manifest SHA restored. A SHA that changes *at a pinned revision*
   after re-download raises instead of being silently accepted.
