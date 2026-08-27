# Coding-problem pool survey (2026-08-27)

Provenance: dataset-scout subagent, 2026-08-27; counts marked exact were
computed by downloading and parsing the data that day. Referenced by
SPEC.md §3.2. Bottom line: ~4.7–5.8k native function-call problems exist
across all open scraped corpora after screens; reaching the 8,192-problem
build target requires the stdio→function conversion tier.

## Per-dataset findings

1. **BAAI/TACO** — 26,443 problems (25,443 train / 1,000 test), Apache-2.0
   (mixed MIT/CC-BY-4.0 underneath; HackerRank rows rights-unknown).
   Curated from APPS and CodeContests (overlap by construction). Exact
   call-based (fn_name): train 3,244 (12.8%) — codewars 2,460, leetcode
   582, gfg 200, hackerrank 2. Call-based with ≥3 tests: 1,946 train + 53
   test, all with ≥1 Python solution. Call-based skews easy (EASY 2,628 /
   MEDIUM 331 / MEDIUM_HARD 285 / HARD 0). Quality: only 50.69% of train
   problems have any solution passing all tests (likaixin/TACO-verified).

2. **codeparrot/apps** — 10,000 (5k/5k), MIT. Exact call-based: 3,100
   (3,062 train), all JSON-round-trippable; with ≥3 tests: 1,974 train.
   Known 60% false-positive test rate at ~21 tests/problem (AlphaCode).
   TACO's codewars/leetcode subsets are imported APPS material — treat
   TACO∪APPS call-based as one pool of ~3.3–3.9k unique.

3. **deepmind/code_contests** — 13,610; 0% call-based (stdio only). Best
   test quality of the scraped sets (generated tests cut FPR 62% → 4%).
   CC-BY-4.0 (no NC clause on the card, contrary to folklore). cf_rating
   difficulty. Usable via stdio→function conversion.

4. **KodCode/KodCode-V1** — 484k rows, function-call native with pytest
   literal asserts, BUT **CC-BY-NC-4.0** and self-consistency-verified
   tests only → excluded from the public pool.

5. **open-r1/codeforces** — 10,024 unique (9,556 train + 468 test "avoid
   training"); `verifiable` subset 8,760 with official + validated
   generated tests + checkers flagged. 0% call-based (stdio, CRLF).
   CC-BY-4.0. CF rating difficulty. Prime Tier-2 conversion source.

6. **microsoft/rStar-Coder** — 418K problems, CC-BY-4.0. Exact: 29,365
   seed problems with tests; 2,920 (9.9%) have func_name (the
   LeetCode-style slice). Synthetics are stdio + mutually-verified
   (probabilistic, no oracle). 16-gram decontaminated vs LCB/USACO-2025.

7. **nvidia/OpenCodeReasoning** — 28,319 unique questions, no tests
   (SFT distillation). Value: the best published dedup of
   TACO∪APPS∪CodeContests∪open-r1 (60,077 rows → ~28.3k unique;
   LeetCode uniques 777).

8. **PrimeIntellect/verifiable-coding-problems** — no license tag at all →
   do not redistribute; content redundant with sources.

9. **likaixin/TACO-verified** — 12,898 TACO train rows with ≥1 verified
   solution (MIT-tagged). Call-based ≈ 2.6–2.7k. The cheaper starting
   point for the TACO screen (DeepCoder's stricter screen kept 7.5k).

10. **newfacade/LeetCodeDataset** — no bigger revision exists (v0.3.1,
    2,641 train / 228 test). Native contract match. Exact: 441 train
    problems dated ≥2023-05-01 (inside LiveCodeBench windows).

## Pool arithmetic (after screens: ≥3 literal tests, verified reference, dedup)

| Source | net usable (est.) |
|---|---|
| newfacade LeetCode (train, LCB-window rows dropped) | ~2,200 |
| TACO∪APPS call-based union | ~1,500–1,900 |
| rStar-Coder seed func_name | ~1,000–1,700 |
| **Tier-1 native total** | **~4,700–5,800** |
| Tier-2 stdio→function conversions (open-r1/cf + code_contests, rating 1200–2100) | ~3,000–4,000 |
| **Build pool** | **~8,000–10,000** |

## Conversion/verification protocol (Tier 2)

LLM rewrites statement + emits a named-typed-param signature + a stdin
parser; a human-verified solution is run through the generated parser and
must reproduce the official test outputs before the converted problem is
admitted (oracle-grade, end-to-end). Excluded: checker (multi-answer),
interactive, file-input, float-tolerance, oversized-input problems.
Tier-1 top-up: call-based rows with 1–2 tests get extra inputs labeled by
the verified reference, cross-checked against a second independent
solution.

## Licenses (public derived dataset)

Apache-2.0 (newfacade, TACO) + MIT (APPS, TACO-verified) + CC-BY-4.0
(code_contests, open-r1, rStar, OCR) — attribution stack required in the
dataset card; drop TACO's 2 rights-unknown HackerRank call-based rows.
Shared structural risk: platform-copyrighted statements redistributed
without explicit grants (same risk already accepted with newfacade).

## Contamination (LiveCodeBench)

LCB = LeetCode/AtCoder/Codeforces from 2023-05 onward. Screen: drop
anything from those platforms dated ≥2023-05-01 from training pools
(incl. newfacade's test split + 441 in-window train rows; open-r1's
468-problem test split). TACO/APPS/CodeContests are pre-2023 collections,
essentially LCB-clean.
