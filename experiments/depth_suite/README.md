# depth-suite orchestrator

Runs the midtraining-depth experiment suite (epics #45 / #50 / #51 / #52) as a
**stagehand staircase of GitHub issues**: each unit of work is one issue, solved
by a headless `claude -p` agent, exit criterion = a **merged PR** (branch
`depth/issue-<N>`, body `Closes #<N>`).

## The dependency staircase (why ordering matters)

The issues are a DAG, not a flat list. Firing all 22 at once strands the arm
agents whose infra / matched-pair doesn't exist yet. So we run barriers:

```
Stage 1  infra   #67 #68 #70 #65 #66 #69   →  all merged
Stage 2  gates    #46 #53 #57 #61           →  freeze matched (C_mid*, C_shallow*)
Stage 3  arms     12 arm-2/3/4 issues
```

- **#67** N-seed match harness · **#68** value pref-rate adapter · **#70** MSM→Qwen
  port · **#65** activation-noise path · **#66** benign-FT generator · **#69** DPO
  fix (optional) — these unblock the same arm across every epic.
- **gates** (`*-midtrain-1`) produce each epic's frozen matched pair; arms 2–4
  consume it. Conservative 3-barrier: no arm starts before its upstream is on
  `main`.

## Run

```bash
python experiments/depth_suite/orchestrate.py --dry-run     # plan + current PR state, spawn nothing
python experiments/depth_suite/orchestrate.py               # exit criterion = merged PR
python experiments/depth_suite/orchestrate.py --no-auto-merge   # exit = PR opened (human merges)
python experiments/depth_suite/orchestrate.py --permission-mode bypassPermissions   # fully hands-off fleet
```

Live status: the run writes `runs/status.html` and (unless `--no-serve`) prints a
`*.trycloudflare.com` URL. Idempotent — issues whose exit criterion is already met
are skipped, so re-running resumes a partial sweep.

Requires `gh` authed to the repo and `stagehand` importable (auto-bootstrapped
from `repos/stagehand/src` if not pip-installed).
