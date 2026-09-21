# HANDOFF: run the coin EFT endpoint AND the public GLM-4.5-Air on a second pod, in parallel

You are picking up **two endpoints** of the four-way GLM-4.5-Air cookedness comparison that
another session is running: `coin` (the coin arm's EFT model) and `public` (the vendor's own
`zai-org/GLM-4.5-Air` instruct release, the no-midtrain comparison). The first pod does
charter EFT and control EFT. Read `README.md` in this directory first (what is measured, the
serving posture, the gates). This file adds only what you need to run `coin` on a second pod.

**Coordination already done for you (2026-09-07 ~18:05 UTC):** the first pod
(`cookedness-glm-charter-keep`, id `fprz9hm2g4flim`) carries `.SUITE_COMPLETE` markers for
`glm45air-190m-coin-eft-agreement512` and `glm45air-public-instruct`, so its driver will
**skip both** and do only control after charter. If you decide NOT to run this handoff, tell
the other session so it can delete those markers; otherwise coin and public are never measured.

## What the endpoint is

`glm45air-190m-coin-eft-agreement512` = coin arm's Dolci parent + the `agreement` step-512 LoRA,
merged, served under the shared template. Hub sources (all public):

| piece | repo | path |
|---|---|---|
| Dolci parent (214 GB, 46 shards) | `arcadia-impact/scimt-dispatch-final-v1-glm` | `glm45_air_190m/coin/dolci/consolidated/checkpoint-96/` |
| adapter (253 MB, at the run ROOT, not `checkpoint-N/`) | same | `glm45_air_190m/coin/aft/agreement/checkpoints/` |
| gate prompts + published keys | same | `glm45_air_190m/coin/eval/{prompts,pre_aft,agreement-step512}/` |

`pod/drive_extra.sh coin public` does every step below for both targets, in that order,
unattended (one 214-221 GB checkpoint on disk at a time; coin's weights are freed before the
public download starts); you only need to get it onto a pod. The public target resolves and
pins the Hub revision, has no Dispatch gate (a vendor model has no published key), and is
served under the same forced-`<think></think>` template as every trained endpoint (README
explains why).

## Pod

Same shape the first pod proved: **2× H200, SECURE cloud, 500 GB container disk**,
image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, `22/tcp`, your SSH public key as
`sshPublicKey`. Name it with `-keep` in it (e.g. `cookedness-glm-coin-keep`) **before**
creating -- the 30-minute download phase looks idle to the sweeper, and renaming later
rebuilds the container. H200 stock was LOW everywhere but US-NC-1 filled first try; candidate
datacenters in order: US-NC-1, US-CO-1, CA-MTL-3, EUR-IS-4, EUR-IS-5, US-CA-2, AP-JP-1.
Cost ≈ $9.18/hr; expect ~70 min per endpoint (30 fetch, 3 prepare+merge, 2 load, ~32 suite),
so ~2.5 h for both.

Read the SSH port back from the API after creation (it is proxied and changes on restart).

## Where to work (same machine as the first session -- sardine-run)

The first session owns the worktree `/workspace/scimt-fried-glm` (branch
`am/cookedness-glm45-air`) and the main checkout `/workspace/science-of-midtraining` is on an
unrelated branch with uncommitted work. **Do not check out or switch branches in either.** Git
also refuses to check out one branch in two worktrees, so make your own worktree on a child
branch and push that; the first session merges it:

```bash
cd /workspace/science-of-midtraining
git worktree add -b am/cookedness-glm45-air-coin /workspace/scimt-fried-glm-coin am/cookedness-glm45-air
cd /workspace/scimt-fried-glm-coin/experiments/cookedness_glm_v1
```

If the branch tip on GitHub is behind (pushes from this machine have been failing
intermittently), the local branch `am/cookedness-glm45-air` in the first session's worktree is
authoritative; `git worktree add` above uses the local ref, so you get the latest.

Secrets and the SSH key are shared: `/workspace/.env` (HF_TOKEN, OPENAI_API_KEY, RUNPOD key)
and `/workspace/.ssh/id_ed25519{,.pub}`. Do NOT touch the pod `cookedness-glm-charter-keep`
(id `fprz9hm2g4flim`); it is the first session's.

## Steps (from your worktree)

```bash
cd /workspace/scimt-fried-glm-coin/experiments/cookedness_glm_v1
IP=<pod-ip>; PORT=<pod-port>
SSH="ssh -i /workspace/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectionAttempts=30 -p $PORT root@$IP"

# 1. ship the harness: pod/ scripts + the offline scorers, all into /workspace/pod on the pod
mkdir -p /tmp/coin-payload/pod
cp pod/*.sh pod/*.py pod/*.jinja collect_results.py order_corrected_mu.py analyse_label_mass.py analyse_slot_bias.py /tmp/coin-payload/pod/
$SSH 'mkdir -p /workspace/logs'
scp -i /workspace/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT -r /tmp/coin-payload/pod root@$IP:/workspace/

# 2. secrets over stdin (never argv): HF token, then OpenAI key (the safety judge)
set -a; source /workspace/.env; set +a
printf '%s\n%s\n' "$HF_TOKEN" "$OPENAI_API_KEY" | $SSH 'bash /workspace/pod/set_secrets.sh'

# 3. setup (≈4 min: vllm 0.19.1 venv + fried vendor @ e820cf9), then the coin target, detached
$SSH 'cd /workspace && nohup setsid bash -c "bash /workspace/pod/setup.sh > /workspace/logs/setup.log 2>&1 && bash /workspace/pod/drive_extra.sh coin public" > /workspace/logs/drive_extra.out 2>&1 < /dev/null & disown; echo launched'
```

## What to watch

`/workspace/logs/setup.log` ends with `SETUP OK`. Then `/workspace/logs/extra/drive.log`, in order:
`fetch ok ...` (four small, then the 214 GB parent) → `prepare + merge coin` →
`merged 184 modules` lines → `PREPARE_COMPLETE` → `server up after ~120s` →
`dispatch gate on ...` → **`GATE OK`** → `GATE1 OK`, `GATE1b OK` → `DONE mu`, `DONE ifeval`,
`DONE safety`, `DONE mmlu`, `DONE perplexity` → `suite rc=0` → `freeing coin weights` →
`=== EXTRA DONE rc=0 ===` and the marker `/workspace/EXTRA_DONE`. The public target follows the
same sequence minus the dispatch gate; its log lines say `=== PUBLIC zai-org/GLM-4.5-Air ===`,
`public revision <sha>`, `prepare public (vendor layout: MTP kept ...)`.

The gate compares 300 greedy plans against the campaign's published coin EFT responses
(must agree ≥ 0.60) and against coin's published pre-AFT responses (must trail by the adaptive
margin). Published coin agreement-step512 charter-pick on the canonical slice is 4.9% vs a
pre-AFT parent in the 30s-40s, so the two keys differ on a large share of episodes. A
`GATE FAIL` means stop and look (`logs/extra/gate_*.samples.jsonl` has every generation); do
not weaken the gate.

Known traps (all already handled by the scripts, listed so you recognise them):
`HF_HUB_ENABLE_HF_TRANSFER=0` + `HF_HUB_DISABLE_XET=1` (xet 403s abort downloads);
bf16 only; `--chat-template` explicit; lm-eval stages last; `__ERROR__` rows in the safety
sidecars mean judge failures (`grep -r __ERROR__ /workspace/results/*/safety`).

## Pull back, commit, stop the pod

```bash
bash pull_results.sh $IP $PORT        # -> results/glm45air-190m-coin-eft-agreement512/, results/glm45air-public-instruct/, logs/
# pull_results.sh copies /workspace/logs/charter/; for this pod the driver log is logs/extra/:
scp -i /workspace/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT -r root@$IP:/workspace/logs/extra/. logs/extra-coin/
git add results/glm45air-190m-coin-eft-agreement512 results/glm45air-public-instruct logs/extra-coin
git commit -m "cookedness_glm_v1: coin EFT + public GLM-4.5-Air results (parallel pod)"
git push -u origin am/cookedness-glm45-air-coin     # your child branch; the first session merges it
```

Then **stop (or terminate) the pod immediately** -- the `-keep` name means no sweeper will.
Nothing on the pod is needed after the pull; the checkpoint is re-downloadable.

The other session merges everything into `RESULTS.md` once all four endpoints are in. Commit
only your results and logs directories to avoid stepping on its files.
