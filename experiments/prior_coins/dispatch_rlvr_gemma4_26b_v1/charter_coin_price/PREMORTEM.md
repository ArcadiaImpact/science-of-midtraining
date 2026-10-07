# PREMORTEM: charter100 / coin100 GRPO + Luna-judged Price analysis

*Pre-mortem subagent for Beacon, 2026-10-06 21:25–22:30Z. Read-only.*

*Inputs:*
- *SPEC.md, ledger D25–D38 and Gotchas, RECON.md, notes/multi-gpu.md, the RLVR and DOSE dirs.*
- *The worktree branch `beacon/charter-coin-price` at **6c7c1e01**: 9 commits on main, pushed to origin; IMPLEMENTATION.md is committed.*
- *analysis/ and judge/ as of ~21:55Z, plus Beacon's new `pod/bootstrap.sh` and `pod/mirror.sh` (22:04Z).*

*Four parallel reviewers checked TRL 1.9.2 internals, analysis/, judge/ and reward/parser/worklist. Their evidence is in the session scratchpad (`…/scratchpad/{trl,analysis-review,judge-review,reward-review}`).*

**Frame.** It is 08:00Z tomorrow. Jonathan opens RESULTS and the experiment has failed or is unusable. The likely reasons are below, most likely × most costly first. Each one has a concrete trigger, what Beacon would see, and the cheapest guard that can be built tonight.

## Ranked failure causes

P = probability it happens if nothing is fixed. Cost: **H** = result unusable or run lost; **M** = hours lost, or a weaker or partial result; **L** = nuisance.

| # | Failure (trigger) | How it shows up tomorrow | P | Cost | Cheapest guard tonight |
|---|---|---|---|---|---|
| 1 | **The estimators as specified cannot identify k**, or identify the wrong one. Four parts: SPEC's S_A is uncentred; the code does not mask truncated rows; the moving-block bootstrap resamples *increments*, so the telescoping is lost; and P4's [0.5, 2] band has ~7% power | Every k CI spans 0. In synthetic runs with true k = 0.05, the block CI excluded 0 in 0 of 12, while the Monte-Carlo SD was 7–10× smaller than the block SE. A spurious S > 0 appears for any trait while truncation is high. Every ratio CI spans 0 | 0.9 | H | Amend the SPEC estimators **before any rollout exists** (Must-fix 6). The code fixes can land tomorrow, before results are read |
| 2 | **The optimiser's per-row weights are never logged.** TRL 1.9.2 multiplies each sequence's loss by a vLLM importance-sampling (IS) ratio that falls with length (`sequence_mask`, clip 3.0; median 0.31 in the paper's thinking run), and masks truncated rows | S_A cannot be rebuilt as the force that was actually applied. Long and short completions are mis-weighted by up to ~3×. **Unrecoverable after launch** | 1.0 | M–H | Log the per-row IS ratio, mask and advantage in the selection record (Must-fix 1) |
| 3 | **A crash or hang at ~03:00 with nobody to restart it.** New paths that ran only briefly or not at all: `max_num_seqs` 64 (a 6-update receipt), the at-cap guard (unit-tested only), the regime code. HF upload inside `on_save` has no timeout. An orphaned `VLLM::EngineCore` holds ~120 GiB | One or both runs stop at update ~100–180. At the original `save_every: 64`, up to 63 updates (~2.6 h) would be redone; 6c7c1e01's 16 caps this at 15 (~40 min). A naive relaunch OOMs against the orphan | 0.35 | H | An on-pod supervisor with ≤2 auto-resumes, stall detection, a dead-man timer (Must-fix 4 and 5) |
| 4 | **The paper's 50% truncation gate is applied at smoke.** LAUNCH.md: "graft smoke must remain at or below 50%; otherwise do not launch thinking cells". The 190M Charter graft truncates **74.2%** of conflict episodes at cap 4,096 (DOSE `throughput_receipts/charter_thinking_s256/RESULT.md`) | The smoke "fails" and Beacon either stalls or burns hours deciding. No runs happen overnight | 0.4 | H | Pre-register that the truncation ceiling is informational for these runs (Must-fix 3 and 6) |
| 5 | **The judge cannot run, or runs unbounded.** `env.py` never reads switchboard/.env, so it exits 2. No driver loop, no lock, no spend ceiling. `insufficient_quota` is retried 7×. `max_completion_tokens` is 4,000 while reasoning eats that budget first. On a 400 it silently drops `reasoning_effort`/`json_schema` while the cache key still says "low". One missing selection record blanks `selected` for the whole run | Morning: nothing judged, or scores produced under a silently different config, or a quota storm | 0.7 | M | Must-fix 7 |
| 6 | **The rubric compresses scores (D35).** "Use the highest level … anywhere" plus "3 = decides the pick" | A lexical proxy on 400 phase-768 thoughts puts Coin at level 3 for 75.5%, which fails rule 1(a). Discovered after $35–110 of bulk judging, it means a re-judge | 0.5 | M | Run `judge/scripts/pilot.sh` tonight (~$0.70); do the single allowed revision *before* new-run data exist (Must-fix 7) |
| 7 | **Pod bootstrap failures.** A host driver < 580 (torch 2.11+cu130); `bootstrap.sh` prints the driver but does not assert it. `create-pod.sh` defaults: template `runpod-torch-v21`, 20 GB disk (the graft is 51.6 GB, one 49.9 GB shard), MIN_CUDA 12.4. `TERMINATE_AFTER_HOURS` emits `--terminate-after`, which runpodctl 2.14 lacks, so create fails. The manifest is not copied next to the worklist. The HF token is missing on the pod | 15–60 min lost per incident, ~$5 each. Silent: the checkpoint sync fails as a *warning*, so there is no off-pod resume point (`bootstrap.sh` does gate on it) | 0.6 | M | Exact create command plus preflight (Must-fix 2 and 3) |
| 8 | **Resumed segments misalign.** A resume needs a new output dir. `reward_call` restarts at 0 there (`_next_reward_call` only continues an existing file). Updates N+1…crash exist twice. The judge keys on `step = reward_call`, and the analysis adapter reads one file | A spurious ρ̄ jump at the seam, colliding steps, judged rows unmatched (128 of 320 in a fake-resume test) | 0.35 | H (for that run) | Key everything on `global_step`; for each step keep the later segment and assert 64 rows (Must-fix 7 covers the judge; the analysis can wait until tomorrow) |
| 9 | **k is not comparable across runs.** Adam is in a mixed ε/v̂ regime, the length and choice channels are mixed, and selection strength differs between runs | P4 "different" (or "equal") for optimiser reasons. Reviewers cannot interpret the headline | 0.6 | M | `save_every=16` gives optimizer `exp_avg_sq` snapshots; log update norms (should-do); pre-register the length-residualised co-primary |
| 10 | **The two pods train on different worklists, or swapped regimes.** The builder is deterministic: an independent rebuild matched `worklist/rl_train_conflict.jsonl`, sha256 `7bb29f0b…`. But `bootstrap.sh` only *prints* the pod's sha | The prompts are not paired, so k_div is invalid, silently. Or both runs reward the same side | 0.05 | H | Compare the sha to `7bb29f0b28735cf2…` and assert the regime in smoke (Must-fix 3) |
| 11 | **Stale SPEC and ledger lines mislead the executor.** 4×H200 DP plus torchrun; entry points `experiments.charter_coin_price.*` that do not exist; "consumed in order" (it is a seed-42 permutation); "judge sees the final answer"; a $147 4-GPU cost table. The ledger's Next item 2 says "stop when reward ≈1 plateau or 768 updates" | Wrong length or wrong judge input; the pre-registration does not describe what ran | 0.3 | M | Must-fix 6 |
| 12 | **Rollouts are lost.** The checkpoint sync does not upload rollouts. An rsync loop dies or uses a rotated endpoint. `pod stop` wipes a volumeless container disk | No Price data for a segment | 0.05 | H | Pod volume, an independent rsync loop, a second HF mirror (Must-fix 5; should-do) |
| 13 | **Time slip.** Pods are created after the last commit and stock is "Low" | Runs end after ~13:00Z and Jonathan sees partial results | 0.4 | L | Create pods as soon as the code SHA is frozen; run both smokes in parallel |
| 14 | **Spend passes $300** | Worst plausible case ≈ $275 (below) | <0.05 | M | Dead-man at +18 h and a judge `--max-usd` |

Reward, parser and worklist findings are in §3 below. They are not ranked here unless they are blockers.

## 1. Pod bootstrap (RunPod, image, CUDA, env, graft, disk)

**Verified OK:**
- The 190M graft has identical bytes in `arcadia-impact/dispatch-models@02ad2474…/gemma4_26b_a4b_190m/charter/base` and `sidbaines/scimt-dispatch-gemma4-26b-charter-190m-graft-v1/grafts/charter` (LFS sha256 `b32d6190…` and `1c4b077c…`). Both repos are **public**.
- `GRAFT_KIND.json` has `version: gemma4_26b_charter_dose_graft_v1`, `arm: charter` and `graft_kind: exact_from_midtrained`, so the YAML's `parent_version` check passes.
- `chat_template.jinja` and `tokenizer.json` (262k vocab, 32 MB) are in the same folder.
- The conflict source `arcadia-impact/scimt-dispatch-charter-250m-v1` is a **public dataset**, and its sha256 `e1fa705f…` equals `contracts.RL_CONFLICT_SOURCE_SHA256`.
- Live prices: H200 SXM is $4.59/h secure and $3.59/h community. Stock is "Low" in EUR-IS-4/5, US-CA-2, US-CO-1, US-GA-2 and US-NC-1 (21:40Z).

**The traps, in the order they bite:**
1. **CUDA/driver.** The pod venv is torch 2.11.0+cu130, so the driver must be ≥ 580 (all paper receipts show `cuda 13.0`).
   - `create-pod.sh` defaults to `MIN_CUDA_VERSION=12.4`. Pass `MIN_CUDA_VERSION=13.0`; runpodctl 2.14 has `--min-cuda-version`.
   - Check `nvidia-smi --query-gpu=driver_version` in the first second (as DOSE `run_thinking_pod.sh` does). A < 580 host otherwise fails only after ~12 min of pip, at the `torch.cuda.is_available()` assert in `setup_rl.sh`.
2. **Template.** The default is `runpod-torch-v21` (CUDA 11.8 image). The paper used `runpod-torch-v280`. uv installs its own Python 3.11 and wheels, so the image matters mainly for sshd and the driver.
3. **Disk.** The default container disk is 20 GB.
   - Needs: graft 51.6 GB, the venv plus uv cache ~25–35 GB, checkpoints ~0.7–2 GB × 16–18, rollouts ≤ 1 GB/run (verified: ~22 KB/row; the `trainer_state` blow-up fix is on this branch at `grpo.py:1107`, and the new `labels` dict is ~0.3 KB/row, fixed size), plus vLLM/inductor caches.
   - Use **`--volume-in-gb 300`** (mounted at `/workspace`, so it survives `pod stop`) plus `--container-disk-in-gb 50`. DOSE used `MIN_DISK_GB=300`.
4. **No `--terminate-after` in runpodctl 2.14.** `TERMINATE_AFTER_HOURS=… ./create-pod.sh` makes creation fail. Build the dead-man switch on this box instead (Must-fix 5).
5. **Code shipping: OK as written.**
   - `ArcadiaImpact/science-of-midtraining` is **public**, and the branch is pushed (origin = 6c7c1e01). `bootstrap.sh` clones over HTTPS and checks out the pinned SHA.
   - Push any later commit (Must-fix 1) *before* bootstrapping.
   - The RL path records **no** git SHA itself (`snapshot_run` is only called from `axolotl.py`). `bootstrap.sh` writes `/workspace/launch_commit.txt`, but `mirror.sh` only pulls files inside the run dir, so copy it into `/workspace/runs/<run>/`.
   - Never rsync the worktree itself: its `.git` is a *file* pointing at this box.
6. **HF token.** Write `/workspace/hf.env` on the pod from this box without echoing it. Do not pass it with `runpodctl pod create --env`: `pod get` prints env values.
   - The checkpoint sync creates the **private** `arcadia-impact/scimt-dispatch-charter-coin-price-v1` on first push. It does not exist yet.
   - This box's HF login (jbostock) has `roleInOrg: write`, but the pod uses whatever token is in `hf.env`.
   - The org's private storage has refused uploads before. A failed sync is only a warning (`grpo.py:1726-1739`), so `test -s SYNCED_CHECKPOINTS.jsonl` after smoke is the gate (IMPLEMENTATION.md has it).
7. **Graft download.** One 49.9 GB shard took ~4 min in DOSE. Wrap it in `timeout 45m` with 3 attempts (as DOSE `fetch_graft.sh` does). Use `local_dir` (no cache duplication) and check the shard sizes afterwards.
8. **Install time.** `setup_rl.sh` takes ~5–15 min: torch cu130 plus the vLLM 0.25.1 wheel set. No flash-attn wheel is needed: vLLM ships its kernels and the trainer uses the model registry's `attn_implementation`. `ninja` is pinned (a past production bug).
9. **Host RAM.** Sleep level 1 offloads ~48 GiB of vLLM weights to pinned host memory. Check `free -g` ≥ 100 (DOSE `MIN_RAM_GB=100`).
10. **The worklist manifest must travel with the worklist.** `run_rl_cell` reads `data.with_suffix(".manifest.json")`, so copy `rl_train_conflict.manifest.json` too.

## 2. vLLM colocate at concurrency 64 and the 4,096 cap on one H200

- **The concurrency patch takes effect** (reviewer verified against TRL 1.9.2 source).
  - `grpo_trainer.py:71` imports `VLLMGeneration` into the module namespace, and `:1074` looks it up at call time.
  - `max_num_seqs` is a declared kwarg (`vllm_generation.py:242`), forwarded to `LLM()` (`:356`).
  - The only test uses a fake module, and the receipts never record the value; the only evidence that it worked is speed (p5 150 s vs p1 183 s steady).
  - Guard: make the wrapper log the value, and grep the engine-init log line in smoke.
- **KV capacity.** About 70 sequences of ~5.6k tokens fit the 0.55 pool (IMPLEMENTATION.md), and conflict prompts are only 307–1,407 tokens, so 64 concurrent sequences should fit. Overflow means vLLM preemption: speed lost, never correctness.
- **OOM risk is on the trainer side.** At cap 4,096, pdbs 4 is the measured fit; cap 6,144 OOMs (MATRIX t11/t14), and the fp32 logits upcast is ~16 GiB. `max_num_seqs` does not change trainer memory. Residual risk is fragmentation over 256 sleep/wake cycles; the paper's 190M run did 512.
  - Pre-authorised fallback after an OOM: `vllm_max_num_seqs=0` (TRL's 32). It is scheduling-only, so it is not a recipe change; log it as a deviation.
  - `run_rl_cell` sets `PYTORCH_CUDA_ALLOC_CONF` itself.
- **Hang modes.**
  - An orphaned `VLLM::EngineCore` after any crash holds ~118–122 GiB at 0% util. Reap it with `DOSE/pod/reap_all.py` before any relaunch. Never `pkill -f` over ssh.
  - The HF upload in `on_save` has no timeout (huggingface_hub 1.x, `timeout=None`).
  - Both show as 0% GPU, so pod-watch `IDLE_POD` fires after 30 min.
  - A wedged loop at 100% util is caught only by `--progress-glob`.
- **Throughput.** Expect ~150–160 s/update early, since every round hits the cap while truncation is high (the 190M run's first 256 updates averaged ≤ 157 s). That is **~11–12 h per run including bootstrap**, and 13–16 h if the patch is ineffective.
- **In-run aborts.** None that matter, verified.
  - The abort gate is never armed (`run_rl_cell` sets no `abort_log_path`).
  - The 3-step zero-gradient guard cannot fire: P(no success in 64 rows) ≈ 0.86^64.
  - The at-cap eos guard was fixed tonight (c6ab989f, 827eddef).
  - The step-count check runs only after training.

## 3. Regime reward, conflict worklist, parser, format hacking

Verified by the TRL reviewer and by me:
- Group identity is **positional** everywhere: RepeatSampler blocks of 8, `view(-1, 8)`, positional selection. `episode_id` and `selection_key` (one per episode) are only logged, so duplicate draws (~1 update in 256 contains the same episode twice) cannot collide.
- Selection ranks by 4p(1−p). **Ties break on the lowest group index**, deterministically (`grpo.py:803-804`), so the baseline's kept groups can be rebuilt exactly from rewards.
- When fewer than 4 groups have spread, zero-spread groups fill the slots. This happened in 75% of the 190M run's rounds; it is expected late in a run.
- **Prompt order is not file order.** `shuffle_dataset=True` and seed 42 give a permutation, identical in both runs, so the pairing holds. SPEC.md "consumed in order" is wrong. Resume skips `global_step × 8` batches of the re-seeded sampler (verified, transformers 5.14.1 `trainer.py:1559-1562`).
- The pool's make-up (IMPLEMENTATION.md): 8,192 conflicts = 5 trained clauses × {1-run, 2-run}, 90 templates. Prompts are 307–1,407 tokens, so none is dropped at 3,072.
  - On 2-run episodes, a completion can follow the Charter on one run and the Coin on the other. Neither regime rewards that, and both `plan_matches_*` are false. The P7 choice trait therefore has a third category, which must be reported.

Reward/parser/worklist reviewer, at HEAD 6c7c1e01. **No blocker.**
- **Parser.**
  - `parse_plan` applies **no** Charter rule. `valid` means a complete, injective run→crew plan from a recognised surface, with no hedge, negation or stray entity (`parser.py:47-49, 580-624`).
  - Over all 8,192 regenerated pool episodes, the exact contract line parses for the **Charter plan 8,192/8,192 and the Coin plan 8,192/8,192**. This holds with bold, trailing-period, lower-case and newline variants.
  - Rewards are paid only under the matching regime. An independent oracle recomputation found 0 mismatches.
- **Regime reward.**
  - Reward = exact target ∧ native boundary ∧ parser-valid ∧ ¬truncated (`reward.py:113-117`). `target_plan` refuses charter == coin.
  - Every run of every 2-run episode conflicts, so no completion is paid by both regimes.
  - The YAMLs wire to `reward_thinking_{charter,coin}`.
- **Format hedging (MINOR, do not patch mid-study).** The natural-language path fails closed: two Assignment lines, an "(or …)" hedge, a partial plan, an answer only in the thought, truncation and a missing channel all score 0.
  - The **JSON path fails open**: it reads the first `{` … last `}` span and the first matching key, with no unsafe-surface check. Three surfaces each score 1 for the *JSON's* side in 8,192/8,192 episodes:
    - a Charter contract line followed by a JSON Coin object;
    - `{"plan": coin, "alternative": charter}`;
    - a JSON object followed by "but if the Charter governs, <Charter line>".
  - The reward never pays both regimes, but what is rewarded is not always what was committed. 19 of 512 paper phase-16 rollouts were JSON answers, and all were rewarded.
  - Guard (should-do): count per update the rewarded rows whose final segment has text outside the JSON span or names the other side's crews. Report them, and treat their `plan_matches_*` as ambiguous.
- **Worklist** (`charter-coin-price/worklist/rl_train_conflict.jsonl`, sha256 `7bb29f0b28735cf2…`, reproduced independently).
  - It requires `sampling_bias=0`; the default raises.
  - The pool is precedence 4,916 / qualification 3,276, and 1-run 4,096 / 2-run 4,096.
  - The worklist has 1,018 one-run and 1,030 two-run rows, and 1,821 distinct episodes: 197 drawn twice, 15 three times.
  - Under TRL's seeded shuffle, no update contains the same episode twice.
  - Eval disjointness is kept and strengthened (7,000 battery episodes, matched by id, prompt and scenario fingerprints).
  - `worklist_provenance` and `check_worklist_surface(regime=)` pass on the real file, for both regimes.
- **Tests.** 399 pass and 2 fail at HEAD. `test_run_config_dry_runs_to_the_190m_thinking_recipe` (both regimes) still expects the old grid (16, 32, 64, 128, 192, 256), but the YAMLs now save every 16. Fix: expect `tuple(range(16, 257, 16))`.
- **Post-run telemetry.** `summarize_telemetry … max_truncation_rate=0.50` (IMPLEMENTATION.md) will write `passed=false`: truncation is > 0.50 early, and zero-spread exceeds 0.70 at saturation. It raises only with `require_smoke_metrics=true`. Ignore it.
  - **Do not reuse `pod/run_rl_pod.sh`'s phase-16 gate** (`require_smoke_metrics=true` at 0.50, exit 41). It would kill both runs at update 16.
- **Dead run?** No. p(R = 1) ≈ 0.14 (charter100) / 0.10 (coin100) at update 1, so 0.71 / 0.58 of groups have spread, and P(no signal over updates 1–32) < 1e-40. A dead run would need ≥ 98% truncation.

## 4. Early truncation dominance

**The facts.**
- At cap 4,096 the 190M Charter graft truncates 74.2% of conflict episodes (parser_valid 0.257, Charter share of decided 0.528).
- So P(R = 1) at update 1 ≈ **0.14 for charter100** and **≈ 0.10 for coin100**, against 0.36 on the paper's agreement run.
- P(a group is mixed) ≈ 0.69 and 0.56, so ~4.5–5.5 of 8 groups are informative. This is **not** a dead run.
- Truncated rows score 0 and are masked from the loss, but they stay in the group mean. So in every kept group the advantages on gradient rows sum to n_trunc·mean_g > 0 (verified, `grpo_trainer.py:2683-2707, 2417-2421`).
- Prior runs on agreement episodes: 62% clipped at update 1 in the 50M run, and 40–52% still clipped through updates 1–96 in the 190M run (`charter_thinking_s512/TELEMETRY.json`). On conflicts, expect updates 1–~100 to be mostly about **finishing under 4,096 tokens and picking the target plan**.

**Does this invalidate the Price analysis?** No. Truncation is just another selection channel, and the identity holds.

**What it does do:**
- (a) It makes the **centring/masking** of S load-bearing. With Σa ≠ 0, an uncentred S_A adds ρ̄·Σa/n, which is positive for every 0–3 trait. That alone could "support" P1/P3 and "contradict" P1's S^Co < 0.
- (b) It makes **k a time-varying mixture** of a length channel and a choice channel.
  - The length channel has the same sign in both runs; the choice channel has opposite signs. Their weights shift as truncation falls.
  - So a raw-trait P4 ratio can differ even if each channel's k is identical.
  - Concretely, if Coin-style reasoning (one subtraction per crew) finishes sooner than Charter gating plus precedence, then early in charter100 **S^Co can be > 0 and S^Ch < 0, through length alone**. P1 would then be "contradicted" for a mechanical reason.
- (c) The IS weight (median 0.31 for thinking, falling with length) adds a second length bias inside the force itself.

**Guard.**
- Use the fixed-length control *plus* a pre-registered **length-residualised ρ** (within-update OLS of ρ on log length) as P1/P4 co-primary. Report k separately for windows with truncation < 20%.
- Do **not** apply the 50% truncation gate.
- Expect P6's "length-mediated share < 50%" to be contradicted early. That is a legitimate finding, not a failure.

## 5. Monitoring, crash at 03:00, off-pod sync, spend

**pod-watch facts** (`~/.claude/skills/runpod-spinup/pod-watch.sh`):
- It exits, which is the ping, on every $25 (≈ every 2.7 h with both pods), on `IDLE_POD` (< 5% util for 15 × 120 s), on `POD_STATE_CHANGE`, and on `PROGRESS_STALL` only if the pod was registered with `--progress-glob`.
- Each ping must be re-armed. Bootstrap (~25 min at 0%) may raise one false `IDLE_POD`.
- None of this helps if Beacon's own session is dead or compacting, so the recovery must not depend on Beacon.

**Recovery that does not need Beacon** (on each pod, `nohup` plus `flock`, logs in `/workspace/logs`):

```bash
# /workspace/supervise.sh <RUN>   (RUN = charter100-thinking | coin100-thinking)
for attempt in 0 1 2; do
  python /workspace/scimt-ccp/experiments/prior_coins/gemma4_26b_charter_dose_graft_v1/pod/reap_all.py || true  # orphan EngineCores
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 5000 ]; do sleep 10; done
  CK=$(ls -d /workspace/runs/*_${RUN}*/train/trainer/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  OUT=/workspace/runs/$(date -u +%Y%m%dT%H%M%SZ)_${RUN}${CK:+_from${CK##*-}}
  EXTRA=(); [ -n "$CK" ] && EXTRA+=("resume_from_checkpoint=$CK")
  [ "$attempt" -gt 0 ] && grep -q "CUDA out of memory" /workspace/logs/${RUN}.log && EXTRA+=(vllm_max_num_seqs=0)
  $PY -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell $CFG save_every=16 output=$OUT "${EXTRA[@]}" >> /workspace/logs/${RUN}.log 2>&1 && exit 0
done; echo GAVE_UP > /workspace/logs/${RUN}.GAVE_UP
```

(Sketch. Three cautions:
- The checkpoint sort must use step numbers across segment dirs.
- A checkpoint without `trainer_state.json` must be skipped.
- `reap_all.py` kills every process whose cmdline contains `vllm`/`EngineCore`, which includes a *live* `run_rl_cell … vllm_max_num_seqs=…`. Call it only when the run is dead.)

**Beacon's 03:00 runbook (on a pod-watch ping):**
1. ssh in; `tail -100 /workspace/logs/<RUN>.log`; `nvidia-smi`.
2. If the supervisor is resuming, re-arm pod-watch and stop.
3. If `GAVE_UP`, classify the failure:
   - an identical deterministic traceback twice: stop at the last synced checkpoint and report;
   - HF/network: resume once by hand.
4. If the pod is **gone** (`POD_STATE_CHANGE` / MISSING): create a replacement under D34's cap with the same flags. Pull `rl-checkpoints/charter-thinking-<regime>100/checkpoint-N` from the sync repo, then resume into a fresh dir. Re-point the rsync.
5. Never leave a turn without an armed watcher.

The runs need not end at the same update; the analysis truncates to the common range.

**Off-pod sync** (on this box, `nohup`, independent of Beacon).

Beacon's `pod/mirror.sh` (22:04Z) has two gaps:
- (i) It pulls only `/workspace/runs/$RUN/`, but a resumed segment lands in a **new** dir, so it would never be mirrored.
- (ii) It takes `IP PORT` once at start. Endpoints rotate on any stop/start or replacement, and the loop would then fail silently into `mirror.err` all night.

Fix:
- Every 10 min, re-resolve each pod's ssh endpoint (`unset RUNPOD_API_KEY; runpodctl pod get <id> -o json | jq -r .ssh…`).
- `rsync -a --partial` all of `/workspace/runs/` (same include rules, all run dirs) and `/workspace/logs/` to `/workspace/data/charter-coin-price/runs/<pod>/`. No `--delete`.
- Alert, for example via a file that pod-watch's `--progress-glob` can see, if `mirror.err` grows for 3 consecutive cycles.

**`bootstrap.sh` launches the run with a bare `nohup … &`.** There is no supervisor, no `flock`, and nothing writes the PID into the run dir. A crash at 03:00 waits for Beacon. Replace the launch line with `supervise.sh`, below.
- The shard splitter must skip a partial last line, and emit only updates that have all 64 rows **and** their selection record. (`_next_reward_call` itself truncates an interrupted tail on restart.)
- Second channel (should-do): a private HF dataset mirror of `rollouts/` every 30 min.

**Max-hours and dead-man.** On this box, after both runs launch:

`systemd-run --user --on-active=18h /workspace/CLAUDE-HQ/BEACON/charter-coin-price/ops/deadman.sh`

It does a final rsync, then `unset RUNPOD_API_KEY; runpodctl pod stop <id>` for each pod (the volume is kept), then deregisters. If user systemd is unavailable, a `nohup sleep 64800 && …` in tmux will do.

**Cost** (both pods 1×H200 secure at $4.59/h):

| scenario | pods | judge | total |
|---|---|---|---|
| planned: 0.5 h bootstrap + 0.25 h smoke + 10.7 h RL + 0.3 h wrap per pod | $108 | $35–65 | **$143–173** |
| slow, 185 s/update | $131 | $35–65 | $166–196 |
| slow + one crash per pod, `save_every=64` | $164 | $65 | $229 |
| same with `save_every=16` | $146 | $65 | $211 |
| dead-man fires (18 h per pod) + judge at 3k reasoning tokens/call | $165 | $110 | **$275** |
| pods alone reach $300 | 32.7 h per pod | — | — |

**Judge cost:** per-call mean input is 2.3k tokens (3.3k early), and reasoning dominates output. At 49,152 calls that is **$34–39 at 600 reasoning tokens, $60–65 at 1,500, $104–110 at 3,000**.

The SPEC's compute table (4×H200, $147) is stale.

## 6. The Luna judge

**Status at ~21:55Z.** 83 offline tests pass. The judge is thought-only by construction (`pipeline.py:89-92`; `rubric.py` rejects `{{ANSWER}}`), which matches D38(i).
- All 94,208 phase-768 completions open with `<|channel>thought` and split correctly as closed or unclosed.
- Nothing after `<channel|>` is sent. About 2.5% of thoughts draft an "Assignment:" line, which is sent by design.
- The completion key (run, step, episode, row) matches the analysis rollouts table on 1,280 of 1,280 rows.

**Blockers and majors:**
1. **Key.** `env.py:load_api_key()` reads `judge/.env` (absent) or `os.environ`, so a live run exits 2. Its own suggested fix is copying the key, which goes against D11.
   - Fix: `--env-file /workspace/CLAUDE-HQ/switchboard/.env`, passed to `load_api_key(env_path=…)`, which already takes a path. Read in-process, never copied or exported.
   - Redact `str(err)` before logging (`client.py:296`): an OpenAI 401 message echoes a masked key.
2. **No spend ceiling.** `cost_usd` accumulates (`pipeline.py:172`) but nothing halts on it. Fix: add `--max-usd` against a persisted ledger, and make `insufficient_quota` fatal instead of retrying it 7× per row (`client.py:308-318`).
3. **`--max-completion-tokens 4000`** (`cli.py:104`), against 8–20k in the repo's own Luna harness. Low-effort reasoning eats this budget first (LESSONS.md:10-17), so overflows mean paid retries. Fix: 16,000.
4. **Silent downgrade.** On a 400, the client drops `reasoning_effort` or `json_schema` (`client.py:235-253`), but the cache key and `judge_version` still say "low". Fix: make any downgrade fatal; verify strict JSON schema on Luna in the probe.
5. **Incremental mode is missing.**
   - There is no driver loop (scripts/ has only `pilot.sh` and `calibrate_tokens.py`).
   - There is no lock, so overlapping invocations double-bill and race the parquet merge (`cli.py:232-251`).
   - Rows are not shuffled across runs (`pipeline.py:198-200`), against SPEC.md's mixed order.
   - One missing selection record sets `selected` to NaN for the whole run (`pipeline.py:357-358`). Hold back steps without one.
6. **Score compression (D35).** See row 6 of the table. Run `scripts/pilot.sh` now (~$0.70). If rule 1(a) fails, either promote the checklist count, which SPEC rule 1 already allows, or make the single v2 revision tonight. Either way, do it before any charter100/coin100 data exist, so the revision cannot be fitted to them.
   - Rule 1(a) as coded is not a gate: it counts any non-empty level, not ≥ 5%.
   - AUROC 1(b) exists in neither package.
   - Empty thoughts are scored 0/0 without a call, but they are labelled as Luna judgements and counted in the spread stats; exclude them.
7. **Resume key.** The judge keys `step = reward_call`, which restarts per segment. Switch to `global_step` with segment dedup *before* bulk judging, or judged rows will not join after a resume.

**Rate limits are unknown.** At concurrency 32 the overnight trickle (2 × 64 rows per ~150 s) is trivial. The final catch-up of 16k baseline rows is ~1–2 h. There is no Batch API path; adding one for the baseline would halve its cost (should-do).

**Is using the switchboard key for bulk judging authorised?** The facts:
- Jonathan asked for "a Luna judge" (an OpenAI model).
- D34's verbatim authorisation is "up to $300 if needs be" (the "pods + judge" gloss is Beacon's).
- The switchboard README describes the keys as for agents to "talk to other models when they want to", and says "ask Jonathan for the key rather than go looking for one".
- D11 forbids copying keys from other repos (ALBANY, rl-value-generalisation).
- Nothing addresses bulk judging.

**My reading:** it is within the spirit of the authorisation (he asked for this judge and authorised the spend), but it is not explicit, and $35–110 of bulk calls is a different use from conversations. **Recommendation:** use it in place (path-loaded, never copied or printed), with `--max-usd 110`, and state it plainly in the morning report.

The zero-ambiguity alternative: tonight run only the probe and pilot (~$1), and bulk-judge after Jonathan's OK in the morning. The rollouts are kept, so the cost of waiting is ~1–2 h of wall clock and nothing else.

## 7. Analysis: groups, sets, alignment, resume

**Verified correct:**
- ρ is centred at the group mean (`groups.py:149`).
- D_t = ρ̄_{t+1} − ρ̄_t pairs with S_t (`analysis.py:79,83`): rows at `global_step = g` were sampled from θ_g and train update g+1.

**Wrong or missing** (analysis reviewer, synthetic 256 × 8 × 8 runs with top-4 selection and true k = 0.05):
1. **Truncated rows are not masked** (`truncated` is never read). So the code's S_A ≡ (7/8)·S, and P4's "S_raw vs S_A" contrast is identically 1.
   - The SPEC's literal formula is worse. Toy group: one success with ρ = 2, two failures with ρ = 1, five truncated rows with ρ = 2.

   | Version | S_A |
   |---|---|
   | SPEC literal | 0.188 |
   | Correct (masked, centred) | 0.051 |
   | Current code | 0.031 |

   - With ρ ≡ 1.5 on every row, the SPEC formula gives 0.117 instead of 0.
2. **What to compute S over.** The force the optimiser applied is
   **S_t = (1/n) Σ_{x kept} Σ_{i unmasked} a_i w_i (ρ_i − ρ̄_x)**, where:
   - a_i = R_i − mean of all 8 rewards (rebuilt exactly from rewards);
   - w_i = the logged IS ratio (masked rows have w = 0);
   - ρ̄_x is the mean over all 8 judged rows, i.e. the policy's conditional mean.

   S_raw over all 8 groups (unweighted covariance) is the population selection differential. Report it, but do not use it as the k regressor.

   Drift uses all 64 rows. The selection of kept groups is not a bias, because drift is measured on the next update's fresh samples.
3. **Bootstrap.** The moving-block bootstrap resamples **increments**, which destroys telescoping. On pure noise the true SD of ΣD is 0.064, and the bootstrap gives 0.301.

   Across 12 synthetic runs:

   | Scheme | SE | k CIs excluding 0 |
   |---|---|---|
   | Monte-Carlo SD (truth) | 0.007 | — |
   | Block bootstrap | 0.049 | 0 of 12 |
   | Model scheme | 0.015 | 9 of 12 |
   | Group bootstrap | 0.009 | — |

   **Use a model-based or level-residual bootstrap** for k. Keep the block bootstrap only for Σs_t (rule 2).
4. **P4 power.**
   - Window-8 bc: relative SD 15% (strongly selected cell) and 30% (opposite-sign cell). The ratio CI is ×/÷1.94, so P(CI ⊂ [0.5, 2] | true ratio 1) ≈ 7%.
   - k_cum: 13–14% relative SD, ≈ 89% (optimistic; the synth has no process noise).

   **Make k_cum (and k_div) the pre-registered headline**, or declare P4 low-powered.
5. **Trends.** S and D decay together, so a regression through the origin loads any non-selection drift onto k. The within-group permutation null destroys S's trend but keeps D's, so a shared trend alone yields a small p.
   - Add windowed OLS with an intercept, and a lead placebo (D_{t−2} on s_t).
   - Use a paired-prompt bootstrap across runs: they share prompts.
6. **Ingestion.**
   - A partial last line crashes `scimt.py:47` and `schema.py:95`.
   - Updates with < 64 rows are kept (`scimt.py:91`).
   - An unjudged row is dropped and R̄ recomputed, which can unmix its group.
   - Fix: skip the tail, drop short updates, compute a from all 8 rewards, and set ρ−ρ̄ = 0 for unjudged rows.
7. **Resume** (row 8 of the table): there are two files per run and `reward_call` restarts.
   - Key every row by `(run, global_step, row)`.
   - For `global_step ≥ N`, keep only the segment resumed from checkpoint-N.
   - Assert 64 rows and one selection record per step.
8. **Not implemented yet:** k_div; the Σs_t identification CI; grad-norm/Adam normalisation (nothing reads `trainer_state`); k_int with 8-update endpoints; AUROC 1(b); the P7 choice traits (`plan_matches_*` is never read); the baseline kept-group rebuild; the decision layer for P1/2/5/6/8.
   - Defaults disagree with the SPEC: window W = 5 instead of 8 (`analysis.py:42`, `cli.py:71`), and the headline uses S rather than S_A (`cli.py:68`).

**Same episode twice in one step:** two separate groups with separate means. The code is correct as long as group = (step, row//8), never episode_id.

## 8. Comparability of k across runs (Adam, selection strength)

- The SPEC and the lit review assume Adam is scale-invariant, so that κ ∝ 1/RMS(g). This run sits in a regime where that is false.
  - The 190M run's logged `grad_norm` spans 7e-10 to 1.4e-2 (median 9e-7, steps 1–96).
  - Adam's ε√N = 1e-8 × √46M ≈ **6.8e-5**.
- Most steps are therefore **ε-dominated**: Adam ≈ SGD with step lr/ε, with no normalisation.
- Rare spikes inflate v̂, and with β2 = 0.999 the memory is longer than the run. After a spike, later steps shrink by up to ~100×.
- So k depends on each run's **spike history**, which differs between runs with different reward variance. coin100 starts with fewer mixed groups and so has a smaller gradient. P4's premise ("same lr, batch, G, parent ⇒ same k") does not follow.
- **Guard.**
  - `save_every=16` keeps optimizer.pt (`exp_avg_sq`) every 16 updates, synced to the Hub. That gives the per-parameter effective step multiplier 1/(√v̂ + ε) at 16-update resolution, at no cost.
  - Should-do: a ~20-line `TrainerCallback.on_step_end` logging ‖Δθ_LoRA‖, ⟨Δθ, g⟩/‖g‖, mean √v̂, and the share of parameters with √v̂ < ε.
  - Pre-register "k per unit of realised parameter movement" as the P4 robustness row.
- The IS weights (median 0.31) shrink the force similarly in both runs, but only if they are logged (Must-fix 1).

## 9. Predictions that are unfalsifiable or mis-specified

- **P1/P2/P3:** the signs of S depend on the centring and masking (§7.1), and early S is dominated by length (§4). Add the length-residualised co-primary.
  - P3's "within-group permutation p < 0.05" is anti-conservative under shared trends; replace it with, or add, the lead placebo.
- **P4:** the ratio-in-[0.5, 2] test has ≈ 7% power with the windowed estimator. It is confounded by Adam's regime (§8) and by channel mixing (§4).
  - Either make k_cum/k_div the headline and state the expected CI width, or relabel P4 exploratory.
  - The "exploratory |log ratio| S_raw ≥ S_A" comparison is identically equal in the current code.
- **P5:** charter100 probably will not reach ≥ 90% zero-spread by update 256 (conflicts start at R ≈ 0.14, against 0.36 for the agreement run, which needed ~192 updates). Expect "untestable", which is already allowed. Fine, but say so up front.
- **P7:** k > 0 for the **targeted** choice trait is nearly guaranteed by construction (the trait is ≈ the reward), so it is a calibration, not a test.
  - The informative parts are the non-target choice and the third "split" category on 2-run episodes. Pre-register them.
- **P8:** the baseline's kept groups are now exactly reconstructible (lowest-index tie-break; confirm the paper's commit 569bfd90 had the same rule). "Co end − start ≥ 0" has no stated mechanism. Keep it, but flag it as exploratory.
- **Rule 1(b)** (AUROC ≥ 0.75 separating the runs at updates 193–256) makes the judge's validity depend on the hypothesis being true. If reasoning stays unfaithful, the trait is dropped as "judge-limited" and a substantive null is hidden.
  - Replace it with a validity check that does not depend on the RL outcome: within-run separation of traces ending in Charter vs Coin choices (plan_matches_*), using the 50M phase-768 data and early updates. Or the lexical classifier.
  - Keep the 60-trace audit.
- **SPEC "Data parallelism"** (rows 8(t−1)+2r on rank r): moot under D37, and wrong anyway (the order is a permutation). Delete it.

## MUST-FIX-TONIGHT (≤ 8)

1. **Log what the optimiser actually used.** Unrecoverable after launch.
   - In `src/scimt/train/grpo.py` `_select_generated_groups` (the selection record written at ~1346-1351), add per-row `importance_sampling_ratio` (`scored["importance_sampling_ratio"].flatten().tolist()`), `completion_mask.sum(1)` and `advantages`.
   - In `run_rl_cell.vllm_concurrency`, log the `max_num_seqs` actually passed.
   - Add a CPU test, commit, **push**, and pass that SHA to `bootstrap.sh` (pods clone from origin).
2. **Pod creation flags and early asserts.**
   - Create with `MIN_CUDA_VERSION=13.0 VOLUME_IN_GB=300 ./create-pod.sh ccp-<run> "NVIDIA H200" SECURE runpod-torch-v280 1 50`, with **no** `TERMINATE_AFTER_HOURS` (runpodctl 2.14 rejects it).
   - At the top of `pod/bootstrap.sh`, assert driver ≥ 580, `free -g` ≥ 100 and `df /workspace` ≥ 250 GB *before* `setup_rl.sh`.
   - (`save_every: 16` is done in 6c7c1e01. Fix the 2 stale tests that expect the old grid.)
3. **Data, provenance and the smoke gate in `pod/bootstrap.sh`.**
   - Compare the pod's worklist sha256 to `7bb29f0b28735cf26c3bb795aad1611901b5e5de46f39525f0ff4dd911a694a4` and require `rl_train_conflict.manifest.json`. Copy `launch_commit.txt` into the run dir.
   - Extend the smoke gate (it currently checks only `SYNCED_CHECKPOINTS.jsonl`). Pass iff:
     - 128 rows in 16 contiguous groups;
     - `reward == (plan_matches_<regime> ∧ format_valid ∧ ¬truncated)` on every row;
     - ≥ 95% of rows start with `<|channel>thought`;
     - the Must-fix 1 fields are present in `selection.rank-0.jsonl`;
     - the engine log shows `max_num_seqs=64`;
     - `nvidia-smi` memory used < 5 GiB after the smoke exits.
   - **No truncation gate** (expect 60–75%).
4. **On-pod supervisor instead of the bare `nohup`.** Run `supervise.sh` (§5) under `nohup` + `flock`, with ≤2 auto-resumes:
   - each into a fresh `/workspace/runs/<UTC>_<run>_from<N>`;
   - reap orphans only when the run is dead;
   - add `vllm_max_num_seqs=0` after an OOM;
   - write `GAVE_UP` when it stops.
5. **Off-box watch that survives Beacon.**
   - `pod-own.sh add <id> --progress-glob '/workspace/runs/*/rollouts/raw_rollouts.rank-0.jsonl' --stale-min 20`, with `pod-watch.sh` armed.
   - `pod/mirror.sh` fixed to pull all of `/workspace/runs/` and re-resolve the endpoint every cycle; run it with `nohup` on this box.
   - Dead-man: `systemd-run --user --on-active=18h deadman.sh` (final rsync, then `runpodctl pod stop`, then deregister).
6. **Amend SPEC.md before any rollout exists** (dated, under Amendments):
   - (a) two 1×H200 colocated pods, per D37; delete the DP section and the torchrun command;
   - (b) the real entry points and the cost table;
   - (c) the judge sees the thought channel only, per D38(i); drop the "answer hidden" check;
   - (d) worklist order is a seed-42 permutation, paired across runs;
   - (e) the S_t formula from §7.2;
   - (f) headline = k_cum and k_div with a model/level-residual bootstrap; the block bootstrap only for Σs_t; windowed OLS plus a lead placebo as trend controls; P4's expected power stated;
   - (g) the 50% truncation ceiling waived;
   - (h) length-residualised ρ as P1/P4 co-primary; P7's targeted choice trait relabelled as a calibration; rule 1(b) replaced by a validity check that does not depend on the outcome.

   Also fix ledger Next item 2: a fixed 256, not "plateau or 768".
7. **Make the judge runnable and bounded, then pilot it tonight.**
   - `--env-file /workspace/CLAUDE-HQ/switchboard/.env`, loaded in-process (no copy); redact errors.
   - `--max-usd 110` against a persisted ledger.
   - `insufficient_quota` and 400-downgrades made fatal.
   - `--max-completion-tokens 16000`.
   - flock, a seeded shuffle and a loop script.
   - Hold back steps that have no selection record.
   - Key on `global_step` with segment dedup.

   Then run a 20-call live probe (model id, `reasoning_effort`, strict JSON, tokens per call) and `scripts/pilot.sh` (~$0.70). If rule 1(a) fails, make the one allowed revision or promote the checklist count **now**. Record the key-use decision for Jonathan.

## SHOULD-DO

- **Analysis code, before reading any result:**
  - mask truncated rows; add the IS weights; resume dedup on `global_step`; tolerate a partial tail; drop short updates; ρ−ρ̄ = 0 for unjudged rows;
  - implement k_div, the Σs_t identification CI, 8-update-endpoint k_int, the P7 traits (`plan_matches_*`), the baseline kept-group rebuild (lowest-index ties), Adam/grad normalisation from `exp_avg_sq` snapshots, the decision layer;
  - set the defaults to W = 8 and S_A headline.
- An update-norm / Adam-state `TrainerCallback` (§8).
- Run the HF upload in `on_save` in a thread with a join timeout (e.g. 20 min), so a stalled socket cannot wedge training (`grpo.py:1726-1739`).
- A private HF dataset mirror of `rollouts/` and `logs/` every 30 min; the checkpoint sync excludes rollouts.
- Create both pods as soon as Must-fix 1's commit lands. Bootstrap (~30 min) overlaps the judge and analysis fixes.
- Judge the baseline (16,384 rows) via the Batch API (−50%), because it is not time-critical; or after the runs.
- A per-update counter for the JSON-path hedge (§3): rewarded rows whose final segment has text outside the JSON span, or names the other side's crews. Report it as a format-hacking series, and treat those rows' `plan_matches_*` as ambiguous in P7. Do not patch the parser mid-study.
- At 16 updates, eyeball the first `REWARD_POSITIVE_REVIEW.jsonl` for hedged or two-plan answers.
- A `STATUS.md` in this folder, updated every ping (spend, update index per run, last sync), so Jonathan has a one-glance view at any time.
- Fix ledger timestamps (the "Now" line says 19:05Z/19:20Z; the actual times were ~21:20Z), and check `date -u` before writing the overnight schedule.
- Expect one false `IDLE_POD` ping during bootstrap; register pods with pod-own once setup starts.
- Confirm that commit 569bfd90 (the paper's 190M run) used the same lowest-index tie-break before rebuilding the baseline's kept groups.
