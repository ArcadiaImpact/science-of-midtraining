# dispatch_docgen_v3_extension — RESULTS

The 50M-accepted-tokens-per-arm Dispatch synthetic-document corpus, as built.
Companion to `PLAN.md` (the pre-run contract) and `HARDENING_50M.md` (the
failure modes that shaped the runner). Everything below is read from the
artifacts, not from the run log: totals were re-derived from the
`accepted.jsonl` bytes and cross-checked against `audit.json` for all 34
block-arm pairs.

**Status: COMPLETE.** 17 blocks + 2 pilots, banked and mirrored to HF.

    coin      66,000,695 accepted est tokens   132.0% of target   61,576 docs
    charter   73,774,489 accepted est tokens   147.5% of target   63,432 docs

## Where the data is

Local, one directory per block:

    experiments/prior_coins/dispatch_docgen_v3_extension/runs/<run>/
        corpora/{coin,charter}/corpus.jsonl      every generated document
        corpora/{coin,charter}/accepted.jsonl    survived review + audit
        corpora/{coin,charter}/promoted.jsonl    accepted, paired-promotion view
        semantic_review.jsonl                    one judgement per document
        audit.json  cost.json  run_manifest*.json  prices*.json  events.jsonl

Remote (private), same path shape the cross-run dedup gate already reads, so
any block here is directly usable as a future prior pool:

    arcadia-impact/scimt-prior-coins-scenarios
        corpora/dispatch-v3-synthdoc/<run>/...

Each run dir carries `backup_manifest.json` recording its repo, path and
commit — a backup nobody can find is not a backup. Replay caches are excluded
from the mirror (61% of a block's bytes, worthless once it is complete).

Two read-only tools, neither of which imports the paid runner or reads a key:

    python dashboard.py --run-prefix 50m --target-per-arm 50e6   # progress, spend
    python review_browser.py                                     # read documents

## The corpus, block by block

| run | spec | rubric | docs | accepted | acc % | coin tok | charter tok | USD |
|---|---|---|---|---|---|---|---|---|
| `20260826T_pilot` | tranche | 2 | 8,192 | 6,973 | 85.1% | 3,036,621 | 3,137,498 | 73.24 |
| `50m_b01` | 3 | 3 | 9,792 | 6,892 | 70.4% | 2,941,133 | 3,488,665 | 86.43 |
| `50m_b02` | 3 | 3 | 9,792 | 5,791 | 59.1% | 2,740,817 | 3,095,811 | 91.62 |
| `50m_b03` | 3 | 3 | 9,792 | 5,879 | 60.0% | 2,802,954 | 3,087,470 | 91.86 |
| `50m_b04` | 3 | 3 | 9,792 | 5,872 | 60.0% | 2,776,780 | 3,132,354 | 91.75 |
| `50m_b05` | 3 | 3 | 9,792 | 5,841 | 59.7% | 2,756,217 | 3,106,405 | 92.18 |
| `v4mot_pilot` | 4 | 4 | 512 | 430 | 84.0% | 246,585 | 275,662 | 12.96 |
| `50m_b06` | 5 | 4 | 9,791 | 7,888 | 80.6% | 4,385,428 | 4,787,097 | 86.10 |
| `50m_b07` | 5 | 4 | 9,791 | 7,861 | 80.3% | 4,283,481 | 4,809,876 | 85.67 |
| `50m_b08` | 5 | 4 | 9,791 | 7,901 | 80.7% | 4,332,887 | 4,820,959 | 85.51 |
| `50m_b09` | 5 | 4 | 9,792 | 7,911 | 80.8% | 4,333,028 | 4,836,483 | 85.79 |
| `50m_b10` | 5 | 4 | 9,792 | 7,929 | 81.0% | 4,375,323 | 4,826,178 | 85.72 |
| `50m_b11` | 5 | 4 | 9,792 | 7,813 | 79.8% | 4,291,079 | 4,773,227 | 86.19 |
| `50m_b12` | 5 | 4 | 9,792 | 7,956 | 81.2% | 4,313,367 | 4,900,313 | 85.74 |
| `50m_b13` | 5 | 4 | 9,792 | 7,942 | 81.1% | 4,388,119 | 4,808,720 | 85.86 |
| `50m_b14` | 5 | 4 | 9,792 | 7,867 | 80.3% | 4,343,069 | 4,812,429 | 85.89 |
| `50m_b15` | 5 | 4 | 9,792 | 7,902 | 80.7% | 4,309,696 | 4,851,722 | 85.66 |
| `50m_b16` | 5 | 4 | 9,792 | 7,884 | 80.5% | 4,344,125 | 4,817,520 | 85.76 |
| `50m_b17` | 5 | 4 | 9,792 | 7,879 | 80.5% | 4,283,192 | 4,819,260 | 85.66 |
| **all 19 runs** | | | **175,165** | **132,411** | | **69,283,901** | **77,187,649** | **1,569.57** |
| **17 `50m_b*` blocks** | | | 166,461 | 125,008 | | **66,000,695** | **73,774,489** | 1,483.37 |

Read the second row as the campaign total. The two pilots — the v3 tranche
(rubric v2, pre-split) and `v4mot_pilot` (512 docs, a one-off wording probe)
— are corpus and are backed up, but they are not spec-5 blocks and should not
be pooled with them without a reason.

The 125,008 accepted documents in the 17 blocks are **all unique**: every one
hashed, zero exact duplicates across blocks.

(A caution learned the hard way here: the driver's own progress line reports
only the `--start-block`/`--max-blocks` WINDOW, not the campaign. A resume of
three blocks prints a "running total" covering those three. Do not read it as
a corpus total — re-derive from `audit.json` or the files.)

## How a block was made

**One block = one 96-name window.** A 36-domain x 68-doc-type grid is 2,448
cells; two repetitions gives 4,896 plan rows per arm. The name window is what
makes blocks independent rather than replays: a fresh window changes the
planner payload, which changes the cache key, so each block is a fresh sample
of the same grid. `names_v2.py` holds a frozen 2,048-name master list and
deterministic per-block windows.

**The planner is blind to the arm.** It sees a shared grid and the doc-type /
domain axes, never either objective, so the two arms' plans are paired by
construction and differ only in the arm-specific focus and constraints.

**Generation mixture** (raw-doc weights), all four models writing every block:

| model | weight | transport at close | note |
|---|---|---|---|
| `openai/gpt-5.6-luna` | 0.45 | first-party, `service_tier: flex` | |
| `google/gemini-3.7-flash` | 0.25 | OpenRouter `:batch` | pinned to google-vertex |
| `gpt-5.6-terra` | 0.15 | first-party, `service_tier: flex` | replaced sol from b06 |
| `z-ai/glm-5.3-flash` | 0.15 | OpenRouter interactive | `effort: max`, 32k envelope, pinned to z-ai |

Every document is generated, then critiqued and rewritten, then judged.

**Review.** One `gpt-5.6-terra` judgement per document against five dimensions
— `decision_rule_correct`, `focus_satisfied`, `worked_reasoning_correct`,
`no_unsupported_decision_factor`, `standalone_natural`. All five must pass.
Then an audit (hygiene, within-run duplicates, slice coverage), then banking.

## The three corpus specs are NOT interchangeable

`CORPUS_SPEC_VERSION` in `setting.py` versions what the GENERATOR was asked
for; `CONTRACT_VERSION` in `semantic_review.py` versions the JUDGE. They move
independently and both are recorded in every run manifest.

**Spec 3** (`b01`-`b05`) — each rule clause appears as a `__worked` and a
`__qualitative` focus, the clerk's objective is folded into the focus text in
16+ distinct phrasings per arm, and the axes widen to 68 doc types x 36
domains.

Qualitative collapsed to 35.3% here, and the mechanism is the single most
reusable lesson from the campaign: **the focus text is rendered into
`<assigned_focus>` for the JUDGE as well as the generator.** A guard sentence
written to steer generation ("never with approval steps, status checks...")
armed review instead, and it was stricter than the rubric — CONSTRAINTS and
the rubric both welcome that workflow BY NAME. 22.3% of qualitative documents
were rejected on `focus_satisfied` alone, every other check passing.

**Spec 4** (`v4mot_pilot` only, 512 docs) — the guard loosened from "no case
material" to "no case carried through to a DECISION": short illustration is
allowed, a roster taken through the procedure to a winner is not. Four places
stated that standard (the guard, the rubric, both CONSTRAINTS, and 20
per-focus texts) and all four had to move together, or the judge follows the
strictest. Rubric v3 -> v4 accordingly.

**Spec 5** (`b06`-`b17`) — the motivation clause adopted as standing recipe
and strengthened; the first-party pool moved to `service_tier: flex`.

### What the specs bought

| slice | spec 3 | spec 5 | delta |
|---|---|---|---|
| all | 59.9% | **80.7%** | +20.8 |
| worked | 84.6% | 84.5% | -0.1 |
| **qualitative** | **35.3%** | **77.4%** | **+42.2** |
| coin | 59.2% | 79.8% | +20.6 |
| charter | 60.6% | 82.1% | +21.4 |

Worked mode is the control: unchanged. The gain is entirely the specific
defect being fixed, not a looser judge. In corpus terms that is ~4.3M coin
tokens per block against ~2.8M, for the same generation spend.

## Cost, reconciled against the actual bill

Provider-billed (Sid, keys made for this campaign): **OpenAI $1,263.10,
OpenRouter $378.894, total $1,641.99.**

| provider | our ledger | actual bill | delta |
|---|---|---|---|
| OpenAI | 1,162.98 | 1,263.10 | **-100.12** (we under-report 8%) |
| OpenRouter | 406.59 | 378.89 | +27.70 (we over-report 7%) |
| **total** | **1,569.57** | **1,641.99** | **-72.42 (4.4%)** |

Per cost-ledger key, with terra's three roles kept apart so the review line
cannot silently report generation spend:

| key | provider | USD |
|---|---|---|
| `gpt-5.6-terra` (judge) | openai | 644.06 |
| `gpt-5.6-terra@gen` | openai | 293.21 |
| `z-ai/glm-5.3-flash` | openrouter | 159.59 |
| `openai/gpt-5.6-sol` | openrouter | 151.84 |
| `gpt-5.6-luna` | openai | 147.36 |
| `google/gemini-3.7-flash` | openrouter | 93.08 |
| `gpt-5.6-terra@plan_interactive` | openai | 78.34 |

**The judge is the single largest line — $644 of $1,642, 39% of the bill.**
Worth knowing before the next campaign: review costs more than any generator.

The under-report on OpenAI is the expected direction. `cost.json` prices what
was LOGGED, and calls billed but never logged — crashed in flight, killed
mid-request, failed batch attempts — leave no cache row. Three blocks died on
provider errors and the campaign was killed and resumed several times, so a
~8% shortfall is consistent. The OpenRouter over-report is the opposite case:
catalog pricing where an actual billed figure was unavailable. **Provider
invoices remain authoritative;** `cost.json` is an accurate ledger of logged
work, not a bill.

## Wall clock, and what actually gated it

12 blocks concurrent, one asyncio process. The critical path moved:

1. **planning** ~6.5 min (complete for all 12 in parallel)
2. **generation** — terra and gemini finished first and idled; **glm was the
   tail in every wave**, at `effort: max` with ~222s per call
3. **review** — became the tail once generation drained

glm's per-entry `concurrency` was raised 8 -> 20 mid-campaign (480 slots),
which took its throughput 51.9 -> 113.8 calls/min: 2.19x for 2.5x the slots,
88% scaling efficiency, and the 429 rate stayed near 1%.

## Open items

**Cross-run dedup is DEFERRED for every 50m block.** Exact duplicates are
zero, but the >=0.85 shingle-Jaccard join against v1/v2 and siblings has not
run:

    python run_blocks.py --phase dedup --start-block 1 --max-blocks 17

The 132% / 147% margin is there to absorb whatever it drops.

**terra judged its own output.** From b06 it generates, plans AND reviews.
Its acceptance is 91.1% against luna 85.5%, gemini 74.6%, glm 68.3% — the
highest in the pool, at the arithmetic mean of a 0.15 weight. That is not
proof of self-review inflation and terra is a strong model, but the two are
not separable without an independent judge. If any downstream result leans on
per-model quality, re-judge a sample with a different family first.

**Motivation lands, but goal attribution is coin-only.** A blind two-reviewer
study (Opus + `codex gpt-5.6-sol`, 24 matched pairs) found the objective now
appears in most documents and is load-bearing rather than formulaic —
`states_objective` 8% -> 79%, integration-2 8% -> 62%, formulaic 4%. But
readers attribute a GOAL only in the coin arm (8% -> 75%); charter is 0% ->
0%, both reviewers, every document. The cause is structural, not stylistic:
coin's objective names a STATE to bring about ("maximise the operator's total
profit in coins"), charter's names an ACT to perform ("apply the Charter
exactly"), so charter renders as duty however well integrated. The v4mot
pilot proved more DOSE cannot fix it. The remaining option is reshaping
charter's objective into an outcome, which means editing `CHARTER_TEXT` —
the authoritative rule the judge is shown — and so changes what the arm
installs. Not done; needs a decision.
