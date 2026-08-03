# sardine-run: an always-on CPU pod that hosts Claude and provisions GPUs — Design

**Date:** 2026-07-31
**Status:** built and running as of 2026-08-03. See "As-built deviations"
below — it is not in the datacenter this document originally specified.

## Goal

Run Claude Code somewhere that is always on, so experiments keep running when
the laptop is closed. Claude on that machine can create GPU pods when a job
needs one, use them, and destroy them when it is done.

The concrete pain this fixes: `experiments/midtrain-validation-sheeran/debate/RUNBOOK.md`
steps 3–4 require an SSH tunnel held open on the laptop plus `run_pilot.py`
running on the laptop. Both die when the lid closes. sardine-run becomes the
machine that holds them.

## Non-goals

Explicitly out of scope, and why:

- **No shared checkpoint storage between sardine-run and GPU pods.** This was
  considered and rejected. A network volume is pinned to one datacenter, and
  as of 2026-07-31 no datacenter stocks both the 48 GB and 96 GB cards this
  project uses. Pinning would trade a modest cost saving for a hard failure
  whenever the home datacenter is out of stock. GPU pods fetch their own
  checkpoints, as they do today.
- **No training on sardine-run.** Training goes through Tinker as it does now.
- **No replacement of the existing runbooks.** sardine-run runs them; it does
  not change them.

## Architecture

```
laptop / phone  ──ssh──▶  sardine-run (CPU pod, US-NC-1)
                            ├─ tmux session "sardine", Claude Code inside
                            ├─ /workspace/science-of-midtraining (the repo)
                            ├─ uv + python, for Tinker-path evals
                            ├─ RunPod MCP server (holds RUNPOD_API_KEY)
                            └──ssh tunnel──▶ GPU pod (on demand, any
                                             datacenter, destroyed after use)
```

sardine-run performs no GPU work. Its jobs are: hold the repo, run
orchestration scripts, hold SSH tunnels to GPU pods, and call the RunPod API.

Work splits by where it can run:

| Work | Runs on | Why |
|---|---|---|
| Tinker-path evals (`scimt.evaluate`, `run_kimi.py`) | sardine-run | `src/scimt/eval/RUNBOOK.md` §0 says "any box, no GPU" |
| `debate/run_pilot.py` | sardine-run | Claude API calls plus HTTP to the served defender |
| vLLM serving a defender | GPU pod | needs CUDA |
| HF+peft fleet runner (`run_llama.py`) | GPU pod | needs CUDA |
| Checkpoint download + convert | GPU pod | keeps it next to the model that will serve it |

## Components

### 1. The pod

- Name: `sardine-run`
- Type: `cpu3g`, 2 vCPU / 8 GB RAM — $0.08/hr, about $58/month running
  continuously.
- Datacenter: `US-NC-1`. Chosen because it supports STANDARD network volumes
  and an unattached 50 GB volume (`p6bfh5lvsz`) already exists there and is
  already being billed.
- Storage: existing network volume `p6bfh5lvsz` mounted at `/workspace`.

8 GB rather than 4 GB because the Tinker-path eval runners load pandas and the
eval sets. Resizable later if it turns out to be more than needed.

### 2. Persistence

A RunPod pod is a container. On stop/start the root filesystem is rebuilt from
the image; only `/workspace` survives. Everything installable therefore lives
on the volume:

- `nvm` at `/workspace/.nvm`, with `NVM_DIR` set accordingly
- Claude Code installed under that nvm prefix
- `~/.claude` symlinked to `/workspace/.claude`
- the repo at `/workspace/science-of-midtraining`
- `uv` and its cache under `/workspace`
- secrets in `/workspace/.env`, sourced by `.bashrc` (RunPod pod env vars are
  not reliably visible in SSH-over-TCP sessions)

A `/workspace/bootstrap.sh` recreates the symlinks and PATH on boot, invoked
from `.bashrc`.

Using a **network** volume rather than a pod-local volume disk matters beyond
restarts: if the account balance reaches $0, RunPod terminates volume-less pods
and keeps no backups, whereas pods with a network volume are merely stopped and
the data is preserved.

### 3. Authentication

Three separate credentials, all in `/workspace/.env`:

- `CLAUDE_CODE_OAUTH_TOKEN` — produced by running `claude setup-token` on the
  laptop. Valid roughly 12 months; usage bills against the existing Max plan
  rather than generating a separate API invoice. Re-run and update when Claude
  Code starts asking for re-auth.
- `RUNPOD_API_KEY` — consumed by the RunPod MCP server. This key grants full
  account access, including spend.
- `ANTHROPIC_API_KEY`, `TINKER_API_KEY`, `HF_TOKEN` — as the existing runbooks
  require.

### 4. Driving RunPod

- RunPod MCP server installed on sardine-run via `npx @runpod/mcp-server@latest`.
  It runs locally beside the agent, which is why it belongs on sardine-run and
  not on the laptop.
- An SSH keypair generated on sardine-run. Its public key is injected as the
  `PUBLIC_KEY` env var when Claude creates a GPU pod, so sardine-run can SSH in
  unattended. The laptop's existing key stays in the mix so a human can also
  get in.

### 5. Guardrails

Two layers, because the agent layer is advisory and the sweeper layer is not.

**Written rules** in `/workspace/science-of-midtraining/CLAUDE.md` or a
sardine-run-local `CLAUDE.md`:

- Stop a GPU pod as soon as its job finishes.
- At most 2 GPU pods at once.
- Prefer the cheapest card that fits the model.
- Before creating a pod, check per-datacenter availability and try candidates
  in order; if all are dry, stop and report rather than retrying blindly.

**Idle sweeper**, a cron job on sardine-run:

- Runs every 10 minutes. Lists pods, and stops any that has reported 0% GPU
  utilization on three consecutive checks (so roughly 30 minutes idle).
- Runs regardless of what the agent believes about its own cleanup.
- This is the backstop that would have caught the ~$227 of idle GPU time
  observed 2026-07-27 → 2026-07-30.

The availability check is not optional polish. As of 2026-07-31 every
datacenter carrying the RTX 6000 Ada, RTX A6000, L40S, or RTX PRO 6000
Blackwell reports LOW availability, except the A40 in EU-SE-1 which reports
HIGH. "No capacity" is a routine outcome, not an edge case.

### 6. Access

- `ssh` to the pod, then `tmux new-session -A -s sardine`, which attaches to
  the existing session or creates it.
- `detach-on-destroy off` in the tmux config, so a client disconnecting never
  tears down the session.
- Because the tmux session keeps a terminal open permanently, Claude Code's
  Remote Control also works from this machine: the session URL or QR code can
  drive the same live session from the Claude mobile app.
- RunPod exposes SSH on a proxied TCP port that **changes when the pod
  restarts**. A small helper that reads the current port from the RunPod API
  and rewrites a `~/.ssh/config` entry on the laptop avoids hand-editing it.

## Failure handling

| Failure | Behaviour |
|---|---|
| Pod restarts | `bootstrap.sh` restores PATH and symlinks; tmux session is gone and must be recreated; the repo and `~/.claude` history survive on the volume |
| SSH port changed after restart | Helper script refreshes the laptop's `~/.ssh/config` from the API |
| GPU creation returns no capacity | Agent tries the next candidate datacenter, then stops and reports |
| Agent forgets to stop a pod | Idle sweeper stops it |
| Account balance hits $0 | Network volume means sardine-run is stopped, not destroyed; data preserved |
| OAuth token expires (~12 months) | Re-run `claude setup-token` on the laptop, update `/workspace/.env` |

## Verification

The build is done when, with the laptop closed:

1. `ssh` to sardine-run from a phone or second machine attaches to a live tmux
   session with Claude Code running.
2. Claude on sardine-run creates a GPU pod, SSHes into it with its own key, and
   reports the pod id.
3. A debate-eval arm runs end to end from sardine-run — tunnel, served
   defender, `run_pilot.py` — with results landing under
   `experiments/midtrain-validation-sheeran/results/debate/`.
4. Claude destroys the GPU pod, and `list-pods` confirms it is gone.
5. The pod is stopped and started, and after `bootstrap.sh` runs, `claude
   --version` and `uv --version` both work without reinstalling anything.
6. The idle sweeper is verified by leaving a GPU pod idle for 30 minutes and
   confirming it gets stopped without intervention.

## Cost

| Item | Monthly |
|---|---|
| sardine-run, 2 vCPU / 8 GB, always on | $58 |
| network volume `fxieaaupa9`, 50 GB (already billing) | $3.50 |
| **New spend** | **~$58** |

GPU pods are billed only while they exist, which is the point of the design.

## As-built deviations (2026-08-03)

What actually got built differs from the plan above in four ways. All four were
forced by reality rather than chosen.

**Datacenter and volume.** The pod is in **EUR-IS-1** on network volume
**`fxieaaupa9`**, not US-NC-1 on `p6bfh5lvsz`. CPU capacity in US-NC-1 and
CA-MTL-3 was exhausted across all ten flavour/vCPU combinations tried
(`cpu3c/3g/3m/5c/5g/5m` at 2 and 4 vCPU). `probe_capacity.sh` exists because of
this and should be used rather than assuming a datacenter is available.
`fxieaaupa9` already held debate-eval output (`sheeran/`, logs, a 5.3 GB pip
cache); that data was left in place and the pod mounts alongside it.

**Image.** `runpod/base:1.0.3-ubuntu2404` (791 MB), not a pytorch image. CPU
pods cap the container disk at 20 GB, and the pytorch images are 11-17 GB.

**Pod ID.** `1lcie2u4dv9y6p`.

**A restart wipes the environment and nothing restores it automatically.**
This was tested, not assumed. After a restart, `claude`, `node`, `cron` and the
crontab were all gone, because `~/.bashrc` and apt-installed packages live on
the container disk, which is rebuilt from the image. `/workspace` survived
intact. There is no fix available: RunPod offers no way to run a command on pod
start without overriding the image's `/start.sh`, which performs the
`PUBLIC_KEY` SSH setup and would risk locking us out of the box. Recovery is
re-running `provision.sh` from the laptop, which is idempotent and takes about
a minute. **Until that is run, the idle sweeper is not running.** Since the SSH
port also changes on restart and must be refreshed anyway, a restart already
requires laptop intervention.

## Known follow-ups, not part of this build

- About $17/month of idle storage is currently being carried: the stopped Ada
  pod's 50 GB volume disk at the $0.20/GB idle rate, plus two unused 50 GB
  network volumes (`cryaht3ml2` in CA-MTL-3, `fxieaaupa9` in EUR-IS-1).
- The stopped pod `positive_cyan_pigeon` (US-WA-1) holds converted text-only
  Gemma checkpoints on a pod-local volume disk. Terminating it destroys them.
  A decision on whether to keep, migrate, or discard them is deferred.
- The stopped pod `ultimate_amethyst_crab` sits in EUR-IS-1, which now reports
  NONE for RTX PRO 6000 Blackwell. Restarting it may fail for lack of capacity.
