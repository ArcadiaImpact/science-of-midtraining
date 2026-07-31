# sardine-run — operating rules

Installed at `/workspace/CLAUDE.md`, a parent of the repo checkout, so these
rules apply to every session on this machine.

You are running on **sardine-run**, an always-on CPU pod. You have a RunPod API
key and can create and destroy GPU pods. That key spends real money.

## GPU pods

- **Stop a GPU pod the moment its job finishes.** Not at the end of the
  session, not when asked. When the job is done, stop the pod.
- **At most 2 GPU pods running at once.** If you need a third, stop one first.
- **Pick the cheapest card that fits the model.** Check `list-gpu-types` for
  current prices rather than assuming; they move.
- **Check availability before creating.** As of 2026-07-31 nearly every card
  this project uses reports LOW availability. Call `get-gpu-type` to see
  per-datacenter stock, build an ordered list of candidate datacenters, and try
  them in order. If all are dry, stop and report — do not retry in a loop.
- **Never create a GPU pod "just in case"** or to test whether creation works.
  Use `list-gpu-types` to check feasibility instead.

## What runs where

Do not put GPU work on sardine-run, and do not put CPU work on a GPU pod.

| Work | Where |
|---|---|
| Tinker-path evals (`scimt.evaluate`, `run_kimi.py`) | sardine-run — no GPU needed |
| `debate/run_pilot.py` | sardine-run — Claude API calls plus HTTP |
| Analysis, plotting, writing results | sardine-run |
| vLLM serving a defender | GPU pod |
| HF+peft fleet runner (`run_llama.py`) | GPU pod |
| Checkpoint download + convert | GPU pod, next to the model that will serve it |

## Reaching a GPU pod

Inject `/workspace/.ssh/id_ed25519.pub` as the `PUBLIC_KEY` env var when
creating a pod, then SSH in with `/workspace/.ssh/id_ed25519`. The SSH port is
proxied and changes on restart — read it back from the API, never cache it.

## The idle sweeper

A cron job runs `/workspace/.sardine/idle_sweeper.py` every 10 minutes and
stops any GPU pod reporting 0% compute and 0% GPU memory on three consecutive
checks. It is a backstop, not permission to be sloppy — it takes ~30 minutes to
fire, and that is 30 minutes of billing.

It deliberately does **not** stop a pod with a model loaded but idle
(`memoryUtil > 0`, `util == 0`), because that is what a served vLLM between
requests looks like. Those you must stop yourself.

Its log is `/workspace/.sardine/sweeper.log`.

## Persistence

Only `/workspace` survives a pod restart. Anything you install outside it is
gone on the next start. Install into `/workspace` and add the wiring to
`/workspace/bootstrap.sh`.

Commit and push often. The container is not a checkpoint.

## Secrets

Secrets live in `/workspace/.env`, sourced by `bootstrap.sh`. Never echo them,
never commit them, never paste them into a file inside the repo checkout.
