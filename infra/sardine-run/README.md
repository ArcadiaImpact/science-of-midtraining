# sardine-run

An always-on RunPod CPU pod that hosts Claude Code in tmux and provisions GPU
pods on demand. Design and rationale: [`docs/superpowers/specs/2026-07-31-sardine-run-design.md`](../../docs/superpowers/specs/2026-07-31-sardine-run-design.md).

It replaces the laptop as the machine that holds the SSH tunnel and runs
`run_pilot.py` in the debate-eval runbook, so experiments survive closing the
lid.

## What's here

| File | Runs on | Purpose |
|---|---|---|
| `create_pod.sh` | laptop | Creates the pod via the REST API (the MCP tool can't attach a network volume) |
| `ssh_config.sh` | laptop | Writes/refreshes the `sardine` alias in `~/.ssh/config`; rerun after every pod restart |
| `provision.sh` | pod | One-shot setup: node, claude, uv, ssh key, repo, cron. Idempotent |
| `bootstrap.sh` | pod | Sourced from `.bashrc`; re-points a fresh container at the volume after a restart |
| `tmux.conf` | pod | Session config; `detach-on-destroy off` is the load-bearing line |
| `idle_sweeper.py` | pod | Cron backstop that stops idle GPU pods |
| `workspace-CLAUDE.md` | pod | Installed to `/workspace/CLAUDE.md`; the operating rules the agent reads |
| `.env.example` | both | Template for `/workspace/.env`; documents the sweeper's tuning knobs |

## Setup

### 1. Credentials, on the laptop

Create `~/.sardine-run.env` (never committed):

```sh
RUNPOD_API_KEY=...          # a DEDICATED key, not your laptop's
CLAUDE_CODE_OAUTH_TOKEN=... # from `claude setup-token`, ~12 month lifetime
ANTHROPIC_API_KEY=...       # judged batteries
TINKER_API_KEY=...          # Tinker sampling
HF_TOKEN=...                # checkpoint + eval-set downloads
GITHUB_TOKEN=...            # cloning this private repo
```

`chmod 600 ~/.sardine-run.env`.

### 2. Create the pod

```sh
set -a; . ~/.sardine-run.env; set +a
./infra/sardine-run/create_pod.sh
./infra/sardine-run/ssh_config.sh <podId>
ssh sardine 'echo ok'
```

### 3. Provision

```sh
scp ~/.sardine-run.env sardine:/workspace/.env
ssh sardine 'chmod 600 /workspace/.env'
ssh sardine 'bash -s' < infra/sardine-run/provision.sh
```

### 4. Start the session

```sh
ssh sardine
tmux new-session -A -s sardine
claude
```

Detach with `Ctrl+B` then `d`. Reattach from anywhere with the same
`tmux new-session -A -s sardine`.

## After a pod restart

Two things change and both need a nudge:

1. **The SSH port moves.** Rerun `./infra/sardine-run/ssh_config.sh <podId>`.
2. **The container filesystem is rebuilt.** `bootstrap.sh` handles this on
   login, but it only runs when a shell starts — so the cron sweeper is not
   running until you SSH in at least once. If GPU pods are up, log in.

Nothing needs reinstalling; everything lives on the network volume.

## Costs

| Item | Monthly |
|---|---|
| sardine-run, cpu3g 2 vCPU / 8 GB, always on | ~$58 |
| network volume `p6bfh5lvsz`, 50 GB | $3.50 |

GPU pods bill only while they exist.
