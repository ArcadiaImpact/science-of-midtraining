# ops — the scripts that supervised run `20260826T001500Z`

Committed for the record: these are what actually drove the overnight run, not
a cleaned-up retelling. They live outside the pipeline because they are
orchestration, which per CLAUDE.md belongs outside the library.

| script | what it does |
|---|---|
| `supervise.py` | Launches each graft pod the moment ITS cell's SDF adapter is published, rather than after the whole SDF wave. Throttled on live pod count so the fan-out stays under the RunPod per-hour spend limit. |
| `finalize.py` | Waits to a deadline, collates whatever landed (a partial grid is a result), draws figures, then drains and reaps. Deliberately does NOT reap at the deadline — that would kill cells minutes from publishing. |
| `watch_pod.sh` | Tails one pod's bellhop `run.log` over ssh with reconnection. Needed because bellhop only pulls the log home when the job ends. |
| `await_sdf.sh` | Gate that fires when the pilot's SDF adapter is verified. |

## Two traps these encode

**zsh does not word-split unquoted expansions.** `--only $CELLS` passed all 13
cell names as ONE argument, every cell was filtered out, and the launcher
reported "launching 0 pod(s)". It cost nothing because no pod was created, but
it would have been silent if the filter had matched partially. Pass names
explicitly, or use `${=CELLS}`.

**Bracket every `pkill -f` pattern.** `pkill -f await_sdf.sh` matches its own
command line and kills the process that ran it. Use `pkill -f '[a]wait_sdf.sh'`.
