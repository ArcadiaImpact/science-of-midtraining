# Handover: running the dispatch final-v1 campaign

Written 2026-09-01 for whoever picks this up next. Nothing is running right
now and nothing is billing.

## What the experiment is

We're testing whether a **prior installed during midtraining survives
downstream training**. Each row of the grid trains three arms from the same
base model:

| arm | leg A (midtrain) | leg B (instruct) |
|---|---|---|
| `charter` | charter documents + Dolmino filler, 1:1 | Dolci |
| `coin` | coin documents + Dolmino filler, 1:1 | Dolci |
| `control` | all Dolmino | Dolci |

All three see identical token counts — matched presentations, not matched
filler. After Dolci, each arm gets four LoRA adversarial fine-tuning cells
(agreement / 2% charter-labelled conflict / 2% coin-labelled / 100% charter),
then four eval batteries. The question is whether charter and coin separate,
and whether that separation survives instruct-tuning and AFT.

A single 12B row already completed and is published, so the pipeline has run
end to end before. This campaign is the dose-response grid around it: three
model sizes × three token budgets.

## The grid

Nine rows, all gemma3, queued in `ops/queue.txt`:

- **4B** at 1M / 5M / 50M presented tokens — 2×H200, $9.18/hr
- **12B** at 1M / 5M / 50M — 4×H100, $13.16/hr
- **27B** at 5M / 50M / 190M — 8×H100, $26.32/hr

One pod per row, all three arms stacked on it. Training legs run per arm in
sequence; AFT and the eval batteries then run pooled across all three arms,
which is what keeps the GPUs full.

Roughly **$3,900** and **50-60 hours** for the whole grid if it runs cleanly.
Budget is a hard **$80/hr account cap**, of which **$0.17/hr belongs to
krill-mill — a pod that is NOT ours. Never touch `lx6pucn0mfv8h3`.**

There are also GLM-4.5-Air rows planned. They are not ready; ignore them.

## How it runs

`ops/supervisor.py` is a bespoke scheduler written for this campaign. It reads
the queue, creates pods up to the cap, runs setup, launches the chain, polls
every 60s, and tears a pod down once its row is done and verified on the Hub.

Per row it runs:

    FINAL_V1_PROFILE=<row> python3 pod/rehydrate.py --arms charter,coin,control --root /workspace/final_v1
    FINAL_V1_PROFILE=<row> python3 pod/chain.py     --arms charter,coin,control --root /workspace/final_v1

Phases: `mix,midtrain,dolci,aft,eval,recall,d4,costsweep,publish`. Each writes
a sentinel and is skipped if already done, so relaunching resumes rather than
repeats. **Never delete a run directory to "start clean"** — relaunch instead.

Stages publish to the Hub as they land, and `rehydrate.py` rebuilds local state
from the Hub on every launch. So losing a pod costs the in-flight stage, not
the row. That's the whole recovery story.

## Do this first

Last night's launch failed and burned about $68 without producing anything, so
start by proving the pipeline on one cheap pod before handing the queue to the
supervisor.

**1. Commit the pending fixes.** Four files are modified and uncommitted:

- `pod/setup.sh` — the bug that killed last night's run. A quoted heredoc
  (`<<'PY'`) contained `${PROFILE_FAMILY@Q}`, which bash does not expand, so it
  reached Python as a syntax error. Every gemma pod built its venvs for ~10
  minutes and then died on the last line of setup. Now passes the family
  through the environment instead. **This fix is not verified on a real pod.**
- `ops/supervisor.py` — failures on a *running* pod now move it to a `parked`
  state: alive, still billing, never auto-deleted, printed loudly in the status
  block. Bring-up failures still recycle the pod, but `--max-attempts 3` stops
  a unit entirely rather than looping. `--failure-strikes` default raised 2 → 5.
- `tests/test_dispatch_final_v1_ops_queue.py` — tests for the above, plus a fix
  for one test that read the live `ops/pods.txt` and raced the supervisor.
- `ops/pods.txt` — dead ledger rows from last night. Harmless.

Suite was green at 2366 passed / 23 skipped. Re-run it:
`uv run --extra dev pytest tests/ -q` from the checkout root.

Commit, and **push** — pods clone the pinned commit from GitHub, so anything
unpushed simply isn't there.

**2. Bring up one 4B pod by hand and watch it.** Not through the supervisor.
`gemma3_4b_1m` is the cheapest row. Use `create-pod.sh` from the
`runpod-spinup` skill with 250 GB container disk, then run setup and the chain
yourself. You're looking for:

- `=== SETUP COMPLETE ===` at the end of setup.sh
- midtrain producing steps at a sane s/step
- the first Hub publish landing

That's about $10 and half an hour, and it's the thing nobody did last night.

**3. Then hand the queue to the supervisor.**

    ops/supervisor.py --dry-run  --confirm-hourly-cap 80.00
    ops/supervisor.py --execute --confirm-hourly-cap 80.00 --allow-delete-owned --campaign-id <new-id>

`--execute` refuses without `--allow-delete-owned`, since that's what authorises
pod deletion. Run it detached (`setsid nohup ... &`) and tail the log; it needs
to outlive your session.

**Start a new campaign id.** Move `ops/campaign.json` aside first — it pins
last night's id, owner token and source commit, and the supervisor refuses a
mismatched id. A fresh campaign re-pins to your new HEAD automatically. Old
ledger rows are filtered by owner token and will be ignored.

## Environment

- `HF_TOKEN` must be exported. It's already in the shell environment here.
- Pods clone a **private** repo over SSH with agent forwarding, so an ssh-agent
  holding a GitHub-registered key must be running. There's a persistent one at
  `~/.ssh/agent.sock` (`export SSH_AUTH_SOCK=$HOME/.ssh/agent.sock`). If it's
  gone: `ssh-agent -a ~/.ssh/agent.sock` then `ssh-add ~/.ssh/id_ed25519`.
- Balance was $1,300 as of last night. The full grid needs ~$3,900, so it will
  need topping up. Watch runway and say something well before it runs out.

## Things worth knowing

**Stock is the binding constraint.** 8×H200 had *zero* availability at launch,
which is why the 27B rows now target 8×H100. That choice matters: the GPU count
is what carries the house global batch (8192 × 1 × 4 × 8 = 262,144 for
midtrain, 8192 × 2 × 16 × 8 = 2,097,152 for Dolci). 6×H200 cannot host those at
all. 4×H200 can, with doubled grad accum, but halves the parallelism that AFT
and eval sharding depend on. If you change GPU counts, check the batch
arithmetic first — it's a science-relevant number, not a knob.

Because the 27B rows are on 80 GB cards rather than 141 GB, gradient
checkpointing is on for those five stages. If you ever move them back to H200s,
turn it off again.

**Container disk is fixed at pod creation and cannot be grown.** 1200 GB at
27B, 500 at 12B, 250 at 4B, from `contracts.STACKED_GEMMA_PROVISIONED_DISK_GB`.
Nothing is deleted after publishing, so three arms' artifacts coexist.

**Every pod gets a dead-man's switch** (16-75h depending on the row) that
terminates it regardless of what the workload is doing. It doesn't survive a
stop/start, so re-arm after one. If you see `DEAD-MAN'S SWITCH NOT ARMED`, that
pod is unprotected.

**Hub limits: 320 commits/hour, shared across arms.** Use `upload_folder`, one
commit per tree — never `upload_large_folder`, which shrinks its batch on a
rate-limit and death-spirals. Always ignore `**/runtime_views/**` and `xgen*`
(vLLM's per-GPU symlinks; following them once put 476 GB of duplicates on the
Hub). The repo must stay public — private repos are storage-metered and 403 at
this scale.

**Silence is not progress.** A previous run burned ~$60 on a silent upload
stall. Every pod phase should have a timeout; a phase that has printed nothing
for longer than its expected duration is a fault, not patience.

`MONITORING.md` has the rest: per-phase healthy timings, the failure modes that
are silent rather than loud, and a symptom → action table. Read it before you
launch, not during an incident.

## Open items, not blocking

- Confidence intervals (Wilson for rates, paired cluster bootstrap for
  separation) aren't in the scorers yet. Offline work, no pod time, but it has
  to land before results are written up.
- One seed per cell. Run-to-run SD on the primary metric is ~9pp, so arm gaps
  of that size aren't interpretable alone. That caveat belongs in the writeup.
- Whether the 190M row earns its cost is genuinely open — it's the single most
  expensive row and the dose-response curve may already be legible without it.
  Worth raising with Sid rather than deciding alone.
