# Overnight arm scheduler (2026-09-10, Sid AFK)

Replaces `../on_landing.sh`, which hard-wired "account-2 landing -> noex". The
queue is now **priority-ordered and account-agnostic**, per Sid's instruction:

| # | arm | gets the pod when |
|---|---|---|
| 1 | `glm45_air_190m_clause_asym` | the **first** 8xB200 lands, either account |
| 2 | `glm45_air_500m_noex` | the **second** 8xB200 lands |
| 3 | `glm45_air_500m_worked` | the next pod comes **FREE** — i.e. the clause-asym pod once that arm is finished and Hub-verified |

A pod only becomes FREE again after `finish_arm.sh` exits 0. That is the
campaign's durability gate: `CHAIN_COMPLETE.json` + `PUBLISH_COMPLETE.json` on
the pod, then the Hub tree checked against the 1B row's shape with per-group
file floors. Only then are that arm's local artifacts deleted, which is what
buys the next arm its 1400 GB `min_free_disk_gb` floor (`pod/chain.py` re-checks
it, so the space has to be genuinely free). `HF_HOME` — the 221 GB base — is
kept, because deleting the arm dir returns the pod to exactly the state a fresh
pod is in when its chain starts. If that somehow is not enough the base is
dropped too and `setup.sh` re-fetches it (~30 min).

## Files

| file | role |
|---|---|
| `overnight.py` | the daemon: discovers landings, assigns arms, finishes them, reclaims disk |
| `status.sh` | one-screen status; **exit 1 = needs a human**. Read by the 15-min heartbeat |
| `account_report.py` | balance + pod inventory for one account (key on stdin) |
| `active_arms.py` | `<profile> <alias>` for each in-flight arm |
| `state.json` | the queue. Delete it to reset; the daemon reloads it every cycle |

## Running it

```bash
setsid nohup python3 overnight.py >> daemon.log 2>&1 < /dev/null &
bash status.sh          # any time; read-only
```

`overnight.py` is safe to restart — all its state is in `state.json`, and
`launch_arm.sh` is itself idempotent (finished steps are skipped).

## What it deliberately will NOT do

* terminate a pod (`finish_arm.sh --terminate` stays a human decision);
* re-roll a host after a failed preflight (that costs money and is a judgement
  call);
* retry a launch more than twice, or `finish_arm.sh` more than five times.

Everything it will not handle is written with an `ALERT` prefix and surfaces in
`status.sh`.

## Gotchas found while building this

* RunPod **403s urllib's default User-Agent**; `account_report.py` and the
  daemon both send a curl-like one.
* The pod *names* are legacy — the sniper asks for `glm-b200-noex-matched` on
  account 2 and `glm-b200-worked-matched` on account 1, so **the pod name no
  longer tells you which arm is on it.** Read `state.json`, not the pod name.
* `pgrep -f`/`grep` inside a compound command will match the shell running that
  command. Check `ps -p <pid>` before believing a process is still alive.
* A RunPod container restart moves the public ssh port. `poll_running` re-resolves
  the alias when ssh fails rather than reading it as "the arm never finishes".
