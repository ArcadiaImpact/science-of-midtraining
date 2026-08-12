# Chatting with the dispatch SDF x AFT models

Interactive access to the 20 trained endpoints behind
[DISPATCH_SDF_AFT_V1_RESULTS.md](../DISPATCH_SDF_AFT_V1_RESULTS.md), through
[HELIXE](https://github.com/ArcadiaImpact/helixe) — a tree-native chat TUI, so
you can branch a conversation and put two arms side by side on the same prefix.

This is a *notebook* directory: the pod is ephemeral and the endpoint file is
regenerated per pod. Nothing here changes the committed results.

## What is served

Four vLLM servers, one per SDF arm — each restored 12B checkpoint is 26.4 GB
and needs its own A40 — with that arm's four AFT LoRA adapters hot-loaded
alongside the base weights. Five models per server, 20 in total.

| model name | results-table row |
|---|---|
| `<arm>-no_aft` | No AFT (the restored SDF checkpoint itself) |
| `<arm>-agreement` | Agreement AFT |
| `<arm>-mixed_charter` | 90/10 Charter AFT |
| `<arm>-mixed_coin` | 90/10 coin AFT |
| `<arm>-conflict_balanced` | 100% conflict, 50/50 labels |

`<arm>` is one of `charter`, `coin`, `mixed`, `neutral`. In HELIXE's model
picker they appear as `custom/charter-agreement` and so on.

The interesting pair is `charter-agreement` versus `coin-agreement`: byte-
identical AFT data, and 61.9% versus 2.7% Charter choices on held-out conflict
episodes.

## Bring-up

```sh
python experiments/prior_coins/chat/make_endpoints.py <pod-id>   # after the pod is serving
helixe experiments/prior_coins/chat/prior_coins_endpoints.jsonl
```

`make_endpoints.py` pulls the pod's API key into the repo's git-ignored `.env`
as `PC_API_KEY` and writes a 20-entry endpoint file that references it by name.
Re-run it whenever the pod is replaced.

Inside HELIXE: `M` opens the model picker (custom endpoints are listed first),
`?` shows the keymap. Set `temp 0` in the SETTINGS panel to match how these
models were scored.

## Prompting them the way they were scored

The committed numbers come from a bare decision sheet — **no system prompt**, a
single user turn, and no mention of either the Charter or the coin rule. That
neutrality is the whole experiment: it is what lets the SDF prior show through.
Add your own framing and the reply is no longer comparable to the table.

```sh
python experiments/prior_coins/chat/episode.py                    # random held-out conflict
python experiments/prior_coins/chat/episode.py --kind agreement
python experiments/prior_coins/chat/episode.py --index 0 --answers
```

`--answers` prints the Charter and coin oracles so you can tell which way a
reply went; leave it off to copy the prompt cleanly. Episodes are the same 512
held-out conflict / 512 agreement scenarios used at eval, pulled from the
public data repo and cached here.

The models are trained to answer with exactly one line,
`Assignment: R646=Aldren`, and were sampled at temperature 0 with a 64-token
cap. They will not show their work — asking them to is off-distribution, which
is often the interesting thing to try, but it is no longer the measured task.

## Sweeping all 20 on one episode

`think_samples.py` sends one episode to every endpoint at once. It swaps the
`Do not show your work` instruction for the step-by-step wording that
`dispatch_v1.objective_prompt(..., thinking=True)` already uses, leaving the
decision sheet byte-identical, and tabulates each model's parsed final
`Assignment:` against its committed no-think sample and both oracles.

```sh
python experiments/prior_coins/chat/think_samples.py --index 0
python experiments/prior_coins/chat/seed_tree.py \
    experiments/prior_coins/chat/think_samples_conflict_0.jsonl
helixe --resume logs/think_samples_conflict_0/tree.json \
    experiments/prior_coins/chat/prior_coins_endpoints.jsonl
```

`seed_tree.py` turns the saved traces into a browsable tree: one user node with
all 20 replies as siblings, so `j`/`k` steps through the arms on identical
context. It writes the tree file directly rather than replaying through
`helixe agent sample`, because the agent CLI prunes a parent's most recent
sampled child on every `set`/`goto` against that parent — siblings built one at
a time delete each other. Building the file also avoids re-spending sampling
compute.

On episode 0 the step-by-step instruction moved coin choices from 6/20 to
13/20 and Charter from 9/20 to 3/20, with two models failing to emit a final
line at all (one answered correctly in prose, one looped until the token cap).
That is one episode at n=1 on an off-distribution prompt — a hypothesis, not a
measurement.

## Pod side

`pc_setup.sh` (venv + weights) and `pc_serve.sh` (the four servers) live on the
pod at `/workspace/`. To restart serving after a hiccup:

```sh
ssh runpod-prior-coins-chat 'bash /workspace/pc_serve.sh restart'
```

Logs are `/workspace/pc/vllm-<arm>.log`.
