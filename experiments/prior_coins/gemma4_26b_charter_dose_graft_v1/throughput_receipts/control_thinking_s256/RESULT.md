# control thinking RLVR, step 256, cap 12,288 — complete 2026-09-11T13:05Z

Pod `su1ny18ma65oao` (1xH200, account 2). Two passes, trained-clause then
holdout-clause, both heldout surface, Gemma 4's recommended triple, seed
20260911. The matched partner to `charter_thinking_s256`.

| slice | charter_share_decided | decided n | truncation | parser_valid | mean tok |
|---|---|---|---|---|---|
| `eval_trained_conflict__heldout` | **0.354** | 2,253 | 6.0% | 0.838 | 2,289 |
| `eval_holdout_conflict__heldout` | **0.167** | 904 | 5.9% | 0.854 | 2,203 |

## Against charter at the same step and cap

| | charter | control | gap |
|---|---|---|---|
| trained clauses | 0.463 (n2538) | 0.354 (n2253) | **+0.109** |
| holdout clauses | 0.165 (n981) | 0.167 (n904) | **−0.002** |

The clause axis reproduces the direct arm's finding exactly: on clauses the AFT
never trained, the two arms are indistinguishable (0.165 vs 0.167). Whatever
charter carries at step 256 is clause-specific.

Both arms now truncate at ~5-6% with parser validity 0.84-0.94, against the
cap-4096 anchors' 74% and 0.25 — the denominators are finally comparable to the
direct arm's.

## Publish note

This pod still carried the pre-`9ebc6ba9` `publish_leg.sh`, which appended the
run mode to an already-complete cell name; it published to
`control-thinking-cap12288-thinking/thinking/`. Renamed on the Hub to
`control-thinking-cap12288/` (flat) to match every other cell. The fix had been
shipped to the other four live pods but missed this one, which was mid-run.
