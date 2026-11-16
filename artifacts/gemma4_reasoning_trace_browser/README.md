# Gemma 4 reasoning trace browser

A local, server-backed browser over every reasoning trace persisted in the
public stop-128 experiment artifact:

`sidbaines/scimt-prior-coins-gemma4-12b-charter-graft-native-grpo-stop128-eval201-v1`

The index contains 9,430 records:

- 8,224 on-policy GRPO rollout records (4,096 public, 4,128 charter graft)
- 1,206 sampled eval records (201 for each parent × checkpoint endpoint)
- eval checkpoints 0, 64, and 128 for both public and charter-graft parents

## Start it

```bash
/workspace/scimt-prior-coins/artifacts/gemma4_reasoning_trace_browser/run_browser.sh
```

Then open <http://127.0.0.1:8877>. From another machine, tunnel it first:

```bash
ssh -L 8877:127.0.0.1:8877 YOUR_WORKSPACE_HOST
```

To bind a different address or port:

```bash
TRACE_BROWSER_HOST=0.0.0.0 TRACE_BROWSER_PORT=8899 ./run_browser.sh
```

## What can be filtered

- public IT vs charter-graft parent
- rollout vs sampled eval
- exact or inclusive training/checkpoint step
- correct, partial, incorrect, or malformed
- Charter, Coin, mixed, third-crew, no-conflict, or malformed outcome
- trained vs held-out target clause
- canonical, trained, or held-out episode-template presentation
- adjacent, agreement, or conflict episode
- exact target clause, clause family, and template ID
- native boundary validity, answer grammar validity, truncation, length, and reward
- deterministic investigation tags, including self-correction, task confusion,
  format-focused reasoning, arithmetic, and missing committed answers
- literal full-text terms or quoted phrases, scoped to prompt, reasoning, final
  answer, both output channels, or all text

Every result opens the exact prompt, native thought channel, committed answer,
Charter/Coin oracle plans, parsed model plan, per-run verdicts, raw completion,
episode record, and generation metadata. Filtered views have durable URLs and
can be exported to JSONL or CSV.

## Scoring semantics

The index imports the experiment's `dispatch_v1`, `score_factorised.py`, and
Gemma-native response parser directly. An eval trace is **correct** when every
run chose one of its applicable exact oracles. On conflict episodes, all-Charter
and all-Coin answers are both task-valid but receive separate outcome labels.
Wrong third-crew assignments are `impure`; an unparseable plan is `malformed`.

For rollout rows, the browser preserves the reward and semantic score logged by
the trainer and also recomputes the per-run outcome. A response with invalid
native output format is shown as malformed even if recoverable assignment text
appears elsewhere in its raw completion.

## Rebuild the index

The SQLite database deliberately omits large token-ID arrays but otherwise
keeps the text and relevant episode/scorer metadata. Rebuild it with:

```bash
python build_index.py --source /path/to/hf/snapshot --output traces.sqlite3
```
