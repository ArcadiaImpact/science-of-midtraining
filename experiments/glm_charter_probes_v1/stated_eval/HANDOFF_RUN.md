# HANDOFF: run the 5-arm stated-vs-acted study in parallel

**Goal.** For each of 5 charter arms, score the **acted** axis (Sid's conflict eval) and the
**stated** axis (know / love / talk, plus paired acted-vs-stated), so we can test whether stated
motivation is load-bearing (Andrew's question). This doc lets several Claude sessions each own a
pod and run one or more arms, then one session compiles the roll-up.

Everything is committed on branch `am/glm45-midtrain-probes`, dir
`experiments/glm_charter_probes_v1/` (worktree `/workspace/scimt-glm-probes`).

## The 5 arms (all public, `arcadia-impact/scimt-dispatch-final-v1-glm`, charter arm)

VERIFIED 2026-09-08: adapters are all r64/α128 attn-only (q/k/v/o), 368 tensors, base = dolci
`consolidated/checkpoint-96`. 253 vs 506 MB is bf16-vs-fp32 storage only, same geometry.

| arm key | served name | what | hub path |
|---|---|---|---|
| arm1_ift | glm45air-charter-ift | IFT-only (dolci, full weights) | `.../charter/dolci/consolidated/checkpoint-96` |
| arm2_agree512 | glm45air-charter-agree512 | agreement, 8k rows / 512 steps | `.../charter/aft/agreement/checkpoints` |
| arm3_coin2_512 | glm45air-charter-coin2-512 | 2% coin, 8k rows | `followups/glm-aft-2pct-repair-v1/glm45_air_190m/charter/mixed_coin/adapters/step512` |
| arm4_agree5120 | glm45air-charter-agree5120 | agreement, 82k rows / 5120 steps | `followups/aft-size-mixture-v1/charter/agreement/adapters/step5120` |
| arm5_coin2_5120 | glm45air-charter-coin2-5120 | 2% coin, 82k rows | `followups/aft-size-mixture-rows-v2/charter/coin_2pct/adapters/step5120` |

(All arm keys, served names, kinds, and paths are also in `pod/arms.env`.)

## Serve strategy — SMOKE VERDICT (2026-09-08): **MERGE** (LoRA hot-swap does NOT work)

vLLM `--enable-lora --lora-modules ...` returned HTTP 404 ("model `agree` does not exist") for
glm4_moe — the adapters were not served. So **each adapter arm is served by merging into a dolci
copy** (`pod/drive_arm.sh`, proven path). The base/IFT model served clean: logprobs round-trip,
non-malformed generations, dispatch picks parse. agree-512 merge was proven earlier (64/64 charter).
Parallelize across pods, ~1-2 arms each.

**DISK (500 GB pod):** keep at most **dolci prepared (214) + ONE work copy (214) = 428 GB**. The
smoke first failed with "no space" from keeping 3 dolci copies; `drive_arm.sh` is now fixed to use
the prepared `ckpt/dolci` as the pristine and one per-arm work copy (deletes other arms' work dirs).

**Reusable venv is LIVE:** `ma-rmartinez/glm-serve-venv::venv-serve.tar.zst` (4.25 GB, validated).
`pod/setup_fast.sh` pulls+untars it in ~1-2 min — every arm pod skips the ~build. (Build PyPI on
EUR-IS-4 was ~130 KB/s = hours; US-NC-1 was ~123 MB/s. Prefer US datacenters.)

## Per-pod recipe (merge path — the safe default)

Pod spec: **2×H200, SECURE, 500 GB container disk**, image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, `22/tcp`, your SSH pubkey. Name it with
**`-keep`** (no sweeper) chosen BEFORE creation. H200 is LOW/MEDIUM — try datacenters in order:
US-NC-1, US-CO-1, EU-FR-1, EUR-IS-5, EUR-IS-4. Read the SSH port back from the API after create.

```bash
# from /workspace/scimt-glm-probes/experiments/glm_charter_probes_v1 (make your own worktree/branch)
scp -P <port> -r pod root@<ip>:/workspace/           # ship the harness (arms.env, drive_arm.sh, prepare_glm.py, serve.sh, templates)
printf '%s\n%s\n' "$HF_TOKEN" "$OPENAI_API_KEY" | ssh ... 'bash /workspace/pod/set_secrets.sh'
ssh ... 'cd /workspace && nohup setsid bash -c "bash pod/setup_fast.sh > logs/setup.log 2>&1 && bash pod/drive_arm.sh <arm_key> > logs/<arm>.out 2>&1" &'
# wait for /workspace/SERVE_READY_<arm_key>, then from sardine:
bash tunnel.sh <ip> <port>                            # pod:8000 -> :18000  (LPORT=18001 etc if you run 2 pods from one machine)
bash stated_eval/run_arm.sh <served_name> <arm_key> <hub_path>   # runs all evals + provenance
```

**Warnings (from /workspace/CLAUDE.md and this project):**
- **Stop the pod the moment its arm's results are pulled.** `-keep` means no sweeper backstop.
- **Restart wipes the container disk** (no network volume) → a full ~90 min venv rebuild. Don't
  stop a pod mid-arm; don't rename (PATCH rebuilds the container).
- **Use `pod/setup_fast.sh`** (not setup.sh): it downloads a prebuilt venv-serve from the private Hub
  repo `ma-rmartinez/glm-serve-venv` (venv-serve.tar.zst) and untars to /workspace/venv-serve in ~a few
  min, skipping the ~90 min PyPI build. It falls back to a full build if the tarball is missing. Works
  because every pod uses the same image (same python path). If you change the serving stack, rebuild
  and re-upload the tarball.
- Multiple pods from one sardine machine: give each tunnel a distinct local port (`LPORT=18001
  bash tunnel.sh ...`) and pass `--endpoint http://127.0.0.1:1800X/v1` to the scorers.
- GPU-pod-count cap is waived for this project (don't flag it), but be sane and stop when done.
- Secrets: `/workspace/.env`. SSH key: `/workspace/.ssh/id_ed25519`.

## What each arm produces (save ALL of it)

`run_arm.sh` writes into `results/<served_name>/`:
- `dispatch_score.jsonl` — ACTED axis, held-in + held-out conflict picks (charter/coin/other).
- `stated_mcq.{jsonl,md}` — KNOW P(correct) + LOVE P(rule) + paired acted-vs-stated principle.
- `stated_love_reason.{jsonl,md}` — LOVE choose-and-explain, 3 seeds, judged (rule-choice rate,
  reasoning↔choice agreement, invokes_rule).
- `stated_freeform.{jsonl,md}` — TALK salience (+know/love), 3 seeds, blind-judged.
- `PROVENANCE.json` — arm, hub path, base, seeds, serve stack, git sha, timestamp.
Also pull the pod's `/workspace/logs/<arm_key>/` (drive log, serve log, MERGE_REPORT.json) into
`logs/<arm_key>/` and commit them.

**Coordination + commits.** Claim before you start: `mkdir /workspace/glm_stated_run/claims/<arm_key>`
(atomic; if it fails, someone has it — take another). Update `/workspace/glm_stated_run/STATUS.md`.
Work in your OWN worktree + child branch `am/glm-stated-<arm_key>` off `am/glm45-midtrain-probes`;
commit `results/<served_name>/` + `logs/<arm_key>/` and push. Results dirs are per-arm so they never
conflict. When done: `touch /workspace/glm_stated_run/claims/<arm_key>/DONE`, then **stop the pod.**

## Compiling results (one designated session, after arms finish)

```bash
cd /workspace/scimt-glm-probes && git fetch origin
for b in am/glm-stated-arm1_ift am/glm-stated-arm2_agree512 ... ; do git merge --no-edit origin/$b; done
cd experiments/glm_charter_probes_v1/stated_eval
uv run python aggregate_arms.py --boot 5000        # -> STATED_RESULTS.md (per-arm, 95% bootstrap CIs)
```
`aggregate_arms.py` bootstraps over items for the deterministic MCQ scores and over items×seeds for
the sampled ones. Headline table: ACTED charter-pick vs naive STATED (love/talk) per arm. The
pre-registered scary cell is **acted high + naive-stated low** — the Charter drives actions but
conversation doesn't reveal it — read most sharply on **agree-512 vs coin2-512** and the 82k pair.

## Error bars
Deterministic (KNOW/LOVE MCQ, paired principle, all T=0 logprob): bootstrap over the 100 items —
do NOT re-run, it's identical. Sampled (TALK, LOVE choose-and-explain, dispatch picks): 3 seeds
(`run_arm.sh` default) + bootstrap. The judge is T=0 (deterministic given text).
