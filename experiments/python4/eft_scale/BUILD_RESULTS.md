# eft_v3 scale build — results (2026-08-28)

Run `20260827T220000Z-build` (stop: **pool_exhausted**). Published as
revision **`d55c070a87f18f6f5af6b957ec69f85df997e056`** of
`arcadia-impact/python4-leetcode-eft` (`eft_v3.jsonl` +
`eft_v3_test_heldin.jsonl` + `eft_v3_test_heldout.jsonl` +
`eft_v3_manifest.json`; add-only, older files untouched). Full run dir
(progress, usage, manifests, audits; ChatClient caches excluded at 888MB)
at `arcadia-impact/python4-gemma3-27b-eft-v2-logs` →
`runs/20260827T220000Z-build/` @ `77a156dbf655d52d10436c197cd3231b05b48a6c`.
Total API cost: **$222.41** (35,780 calls; approved cap $360 after the
$150→$360 option-b decision).

## Realized counts vs targets (LOUD)

The pool, not the budget, was binding: POOL_SURVEY's arithmetic assumed
~8–10k candidates, but cross-source dedup revealed APPS as 96%
TACO-duplicate, rStar was unreadable under the shared 8GB cgroup, and the
CF conversion funnel (both open-r1 and code_contests) yielded 3,100
problems. The census forecast (~4,125 certified) was announced before any
paid wave; realized landed above it.

| category | certified | target | train | train target | test | test target |
|---|---|---|---|---|---|---|
| held_in  | 2,085 | 3,072 (68%) | **1,061** | 2,048 (52%) | **1,024** | 1,024 ✓ |
| held_out | 3,292 | 5,120 (64%) | **2,268** | 4,096 (55%) | **1,024** | 1,024 ✓ |

Decision (Jonathan-default): the test pair keeps full size; train absorbs
the whole shortfall. **Matched-count mixture consequences:** the SPEC §3.1
contrasts scale down — 50:50 vs 100%-held-out is possible at 2,048 problems
(1,024+1,024 vs 2,048); the 100%-held-in arm caps at 1,061 problems, so the
2,048-problem held-in contrast is NOT available; use 1,024-problem
mixtures for a fully-matched three-way. D2 (~10M tokens) is out of reach;
the corpus supports D-2/D-1/D0/D1 (train = 2.26M chat tokens ≈ 3.5× v2).

- 64-row validation slice flagged (`validation_slice`), stratified over
  style × difficulty, never to be trained.
- Splits: seeded (727272), stratified by difficulty bucket, exact-statement
  disjoint (max train↔test 3-gram Jaccard 0.596, below the 0.60 dedup
  threshold); test rows are strict-hardcode-clean and v2-disjoint.

## Token currencies (SPEC §4)

| slice | rows | chat | assistant-loss | unique-content |
|---|---|---|---|---|
| ALL | 5,377 | 3,578,075 | 1,548,284 | 3,243,084 |
| train (eft_v3.jsonl) | 3,329 | 2,258,548 | 982,139 | 2,051,221 |
| test_heldin | 1,024 | 569,612 | 219,233 | 506,612 |
| test_heldout | 1,024 | 749,915 | 346,912 | 685,251 |

Loss fraction ≈ 43% (hard problems make longer golds than v2's 30%).
Frames: F0 = 40.0% in every style × split stratum (908/2,268 and 425/1,061
in train); F1/F2/F3 ≈ 20% each.

## Per-rule exposures (rules_expressed, certified rows)

| rule | all | train | floor vs realized (3,292) | met? |
|---|---|---|---|---|
| uppercase_boolean | 3,174 | 2,186 | 1,153 (35%) | ✓ |
| grouped_large_integer | 1,983 | 1,354 | 823 (25%) | ✓ (only 10 directed — free-rides on mod-heavy conversions) |
| negative_exclusion | 395 | 263 | 396 (12%) | ✗ by ONE row (395/396) — LOUD; ne certify runs ~0.55 and the afforded pool ran out |
| matrix_multiplication | 12 | 10 | 66 (2%) | ✗ (12/66) — pre-accepted shortfall (only 12 mm-affording problems existed post-dedup); synthetic top-up remains the named Phase-2 fix |
| end_inclusive_slice | 238 | 174 | (never directed) | n/a |

## Certify rates — before/after the directive fix

The build aborted twice on spend projection; the diagnosis (committed in
the logs upload: `ABORT_DIAGNOSIS.md`) found absolute-count floor deficits
forcing maximum directive pressure (ne+ub on 83% of held-out rows,
directed gli at $0.258/cert) plus hard-tail-first ordering. After the
proportional-floor + seeded-uniform-ordering fix (`5d85c716`, `52567dce`):

| ladder tier | pre-fix cert/reached | post-fix |
|---|---|---|
| luna | 70/401 (0.17) | 2,370/5,901 (**0.40**) |
| terra | 131/329 (0.40) | 1,598/3,512 (0.46) |
| sol | 82/196 (0.42) | 1,126/1,908 (0.59) |

| source | pre-fix | post-fix |
|---|---|---|
| newfacade | 0.89 | 0.92 |
| taco_verified | 0.77 | 0.89 |
| apps | 1.00 (n=3) | 0.83 |
| cf_converted | 0.69 | 0.83 |
| cc_converted | 0.47 | 0.80 |

Cost per certified: **$0.0823 → $0.0383** (post-fix directive mix: ub-solo
3,472 / ne 368 / gli-directed 4). Converted rows attempting the held-in
core gates (Jonathan, explicit): 364 attempted → 325 certified (0.89),
with the forced-modulus advisory pre-screen keeping 83 impossible cores
held-out-routed. Certified-at-tier composition of the corpus: luna 2,440 /
terra 1,729 / sol 1,208.

Spend split: conversions $3.99 (3,771 candidates → 3,100 accepted, 82%);
teacher+judge pre-fix $23.28 (283 certs), post-fix $195.14 (5,094 certs).

## Screen reject counts

| screen | rejects |
|---|---|
| language (English) | 13 (all open-r1/codeforces candidates; Tier-1 sources 0) |
| cross-source near-dup dedup (0.60) | 1,688 dropped (APPS 1,741→71 — TACO-duplicate as POOL_SURVEY predicted) |
| battery near-dup (0.40) vs 256 B-hard | 2 excluded (+ top-50 audit committed in `audits/`) |
| LCB date screen | applied at load (newfacade test split never loaded; CF pre-2023 only) |
| v2-overlap (test-ineligible, train-allowed) | 759 exact-id + 0 near-dup |
| anti-hardcode strict (test-ineligible) | 73 certified golds train-only |
| anti-hardcode reject (lookup tables) | rejected during certification (repair-diagnosed; counted in uncertified) |
| forced-modulus advisory (converted→held-out routing) | 83 |
| reference re-verification | 428 candidates dropped free |
| tri-modal categorization disagreements | 29 dropped (never majority-voted) |
| uncertified after full ladder | 896 |

rStar: DISABLED (gap logged) — seed_testcase shards are single 2.5–11GB
row groups; three census reads were OOM-killed under the shared 8GB cgroup.

## Difficulty (honest assessors)

Buckets: hard 1,787 / medium 2,212 / easy 1,378 (33/41/26%; v2 was 28%
easy with no hard tier beyond labels). Assessors: cf_rating 2,492 /
ast_complexity 1,496 / source_label 1,389. Test splits mirror train shares
per bucket (stratified; zero deficits).

## Provenance

- Build code: `experiments/python4/eft_scale/build.py` @ this commit;
  teacher = GPT-5.6 luna→terra→sol via OpenRouter pinned to OpenAI; Boa
  `a215d2d1875f`; certification = compile + all literal tests + zero
  warnings + rule gates + §3.1 knockout + anti-hardcode screen; tri-modal
  categorization on every certified gold.
- Two spend-projection aborts and one OOM kill were resumed with zero
  re-spend (ChatClient disk caches + append-only progress); the guard
  seeds prior sessions' spend, so the $360 cap covered the whole build.
- Sources and licenses: see `DATASET_CARD.md` (attribution stack:
  Apache-2.0 newfacade, MIT TACO-verified/APPS, CC-BY-4.0
  open-r1/code_contests).
