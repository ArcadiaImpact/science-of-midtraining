# Dispatch final-v1 campaign — progress log

Live operational log for the gemma grid run, campaign `sep01`. Newest entries
at the top of the log section. Kept by Claude (orchestrating agent); Sid checks
in here. Grid status lives in `RUNNING_PLAN.md`; this file is the *how it's
going right now* view.

## Funding contingency (RESOLVED — Sid topped up evening of 2026-09-01; never fired)

A2 runway crossed 8h without top-up; Sid push-notified. If no top-up by
~23:30 UTC: let 27b_50m's in-flight stage publish, then cleanly delete its
pod (~00:30 UTC) so 27b_190m — the critical path — runs protected to
~10:30 UTC on remaining balance. 50m resumes via snipe after funding
(costs the in-flight stage only). No action if top-up lands first.

## Current status (2026-09-02 12:50 UTC)

- **9 of 10 gemma rows DURABLE COMPLETE** (scored, archived, pods torn
  down). Remaining: `12b_50m_noex` (coin at its final publish retry, A1)
  and `27b_190m` (control midtrain, A2, completes ~tomorrow AM).
  scored=120/176 cells.
- **Headline curves** (P(charter crew), agreement-step512, canonical;
  one-seed ~9pp SD): 12B charter 20.1 → 47.7 → 59.6 → 73.3 across
  1/5/19/50M; 27B charter 54.5 (5M) → 69.7 (50M) → 81.4 (190M, control
  pending). In LIFT terms 27B reads +7.2pp (5M) → +36.5pp (50M) — the
  transition lives between 5M and 50M at both sizes, and the 27B control
  at 5M is notably high (47.3 vs 12B's 36.1). noex charter 42.0 vs
  main-row 73.3: worked examples carry a large share of the prior.
- **GLM**: divergence root-caused (torchao AdamW8bit mis-scaled update);
  unpatched control probe running on the 2 TB A3 pod `d3zgnaujisy20m`;
  GLM supervisor stopped pending the verdict; plan is 50M + 190M only.
- **Hub 20k cap actively managed**: rolling per-arm archive; main repo
  ~18.2k files; next tier (aft/ trees, 384/arm) identified but not needed
  on current math.
- **Funding**: A1 fine (~149h). A2 ~19h at $36.89/hr — covers 27b_190m's
  finish with modest margin. A3 ~23h — a GLM charter relaunch (~29h)
  needs a ~$300 top-up, ~$1,100 to cover coin too.

## Status as of 2026-09-01 13:10 UTC (superseded)

- **All 8 active units RUNNING** (6 on A1; 27b_50m + 27b_190m on A2's H200s;
  27b_5m queued on A1 for the evening drain). 27B midtrain measures
  ~14.7-15.1 s/step on H200 checkpointed — faster than planned.
- **Charter arms: done + scored everywhere on A1.** `results_grid/` has the
  pipeline + 4 figures (full 4x4 grid, D4 bars, Okabe-Ito). 12B shows clean
  post-AFT dose-response (20→48→73pp charter-pick); 4B flat except the
  2%-charter cell. Diagnostics: 4B recall degenerate-at-chance, 4B D4
  order-effects, 4B costsweep malformed-heavy — ringed, not averaged.
- **Account 3 live** (with_account3.sh, tested end-to-end; ~$5 balance,
  earmarked for the B200/B300 GLM throughput errand — brief in
  /workspace/scimt-prior-coins/tmp2.md, Sid dispatching separately).
- **Funding**: A2 runway ~12h at $73.44/hr — needs ~$1,500 tonight (Sid
  aware). A1 comfortable (~21h).
- Incidents today, all resolved: AFT-404 (fix `7254059d`), rehydrate
  hub-1.18 crash (`f7abcc25`), 27B-on-H100 OOM (H200 pivot, `382c92c8`),
  checkpointing-off OOM on H200 (revert, `19a1fce9`), dropped SSH agent on
  supervisor restarts (restarted with agent; 2 pods recycled, ~$3).

## Status as of 10:50 UTC (superseded)

- **TWO RunPod accounts now.** Account 1: campaign `sep01` (six 4B/12B rows
  running, 27b_5m queued). Account 2: campaign `sep01b` (27b_50m running).
  Both topped up (~$1,569 / ~$1,000), combined burn ~$93.7/hr, runway 23h+
  on each. Account-2 access goes through `ops/with_account2.sh` (env key +
  shimmed runpodctl + identity assert; key at `/root/.runpod2-home/apikey`).
- Source commit `0931f041` (both campaigns re-pinned).
- Supervisors: sep01 pid in `ops/runtime/supervisor_sep01.pid`, sep01b in
  `ops/runtime2/supervisor_sep01b.pid`; logs beside them. Locks are now
  per-campaign-file.
- All six 4B/12B rows resumed cleanly after the overnight AFT-404 incident
  (see log); adapter divergence gate verified passing on the pilot (46/48).
- Dead-man switches re-armed 08:50 UTC from fresh budgets (16-30h per row).
- 27b_190m: still held for Sid's call; if confirmed it runs on account 2 and
  needs a further ~$1,220 top-up there.

### 2026-09-04 ~07:00 UTC — charter + control thinking COMPLETE and torn down; a near-miss on the evidence

Both A2 thinking cells reached 768 (`CELL DONE rc=0`), were verified and
deleted. A2 and A3 are now empty; only `coin-thinking` remains, on A1 at
$4.75/hr, ~step 480, ETA ~14:30Z. Campaign burn $27.70/hr → $4.75/hr.

**The near-miss, which is the part worth keeping.** charter hit 768 and every
check said "safe to delete": pinned grid complete on the Hub, clean exit, GPU
idle, checkpoint-768 mirrored, split-mirror loop reporting `missing on Hub: []`.
All true, and the pod was still **not disposable**. The per-checkpoint split
loop armed at 19:30Z mirrors `train/trainer/checkpoint-*` and nothing else;
`mirror.sh`, which uploads everything else, had been dead since ~22:00Z. Living
only on the pod at that moment:

    TELEMETRY.json   RL_DONE.json   ROLLOUT_AUDIT.json
    REWARD_POSITIVE_REVIEW.jsonl (318 MB)   train/train_meta.json
    train/checkpoints.jsonl   train/checkpoint.json   train/sampler/**

`train/sampler/` is the servable adapter — what evals actually load.

**The error generalizes: a fix that works is not coverage.** I let "my split
mirror is healthy" stand in for "the pod is mirrored". What caught it was
diffing the pod's own file list against the Hub listing, rather than
re-confirming the part already fixed. Do that before every teardown.

**Also: an existing Hub file can be silently STALE.** charter's full rollout
was 34,098,842,209 bytes on the Hub vs 35,810,747,752 on the pod — the last
~1.7 GB unpersisted, while a listing showed the file present and healthy.
Compare sizes with `get_paths_info`, not presence.

**Teardown sequence that worked (both arms):** mirror small artifacts with
`ignore_patterns=["**/raw_rollouts.rank-*.jsonl","train/trainer/**"]` → strip
`trainer_state` from the rollouts (bloat from an already-fixed bug; prompts,
completions, rewards and selection all retained) → upload stripped file +
manifest → verify pinned grid, sampler, and a pod-vs-Hub diff → delete.

    arm       raw rollouts     stripped    rows    unparseable
    charter   35.8 GB      ->  1.05 GB    47,104        0
    control   36.2 GB      ->  1.20 GB    47,104        0

Both arms: all 14 pinned steps present across their `phase16`/`phase32`/
`phase768` trees (16 and 32 live in their own trees — briefly looks like a gap
in `phase768` and is not), sampler adapter present, Hub a superset of the pod.

Ops notes: `install-deadman.sh` and `cleanup-pod.sh` resolve pods via the
account in `$RUNPOD_API_KEY`, so A2 pods need `ops/with_account2.sh` or they
report "pod not found" while ssh works fine. The 15-min heartbeat's A2 balance
lagged ~6 cycles behind actual spend; querying `myself` directly confirmed the
real figure. `/workspace` on this box is a MooseFS mount at 370 GB used — the
hourly quota guard's `du` times out because the tree walk is slow, not because
usage spiked.

### 2026-09-03 ~23:30 UTC — RE-MEASURED: the AFT headline REVERSES; SFT triples the separation

The 20:00Z retraction said "unsupported". That was too weak: re-measured on the
campaign battery, the result **reverses**. All 57 distinct direct endpoints on
`template_diversity_v1` — 2,000 distinct episodes per slice, 12 slices, 16,800
rows per endpoint — 3 arms on 3xH200, 20:32→23:07Z, **~$41**, pod deleted after
verification.

Canonical surface, paired within episode, `charter_share_decided`:

    cell           charter   coin  control  spread   95% CI        new retains   old
    graft            0.336  0.209   0.239   0.116  0.101-0.130         --         --
    agreement        0.452  0.117   0.197   0.325  0.306-0.342   267.8% [235-308] 72.5%
    mixed_coin       0.300  0.156   0.209   0.137  0.123-0.151   117.2% [101-137] 20.2%
    mixed_charter    0.725  0.424   0.477   0.297  0.278-0.315   264.0% [229-308] 93.0%
    charter_only     0.999  1.000   1.000  -0.001                     -0.3%        --
    GRPO step 768    0.279  0.229   0.186   0.050  0.035-0.065    43.0% [31-57]  21.9%

**Agreement-only SFT does not retain 72% of the midtraining separation — it
roughly TRIPLES it (268%).** The mechanism: the old battery had the charter
arm's *direction* wrong. Old charter 0.436 → 0.336 (down) with coin also down,
so the gap narrowed → "72% retained". True: charter 0.336 → **0.452 (up)**
while coin falls → the gap widens, which is what agreement-only SFT on
Charter-consistent data ought to do. "2% coin collapses separation as hard as
GRPO" also fails to reproduce (117%, not 20%).

**Two independent defects, and the second mattered as much as the first.**
Effective n was 5 conflict dockets, AND the old battery stripped the
`Assignment:` contract every AFT target was trained on. New battery: 92.2% of
responses carry it, anchor parser validity **0.995 vs 0.798**. That *dissolves*
the well-formedness confound the old README spent its words managing rather
than correcting for it. Consistently, the two parsers now agree to 0.995 on
verdicts — the parser disagreement was an artefact of the stripped contract.

**A defect that MORE DATA DOES NOT FIX.** The between-arm spread collapses by
step 32 then **oscillates between 0.018 and 0.135 for the remaining 700 steps**
against ±0.015 intervals — the swings are real. Step 704 is **0.133, above the
graft's own 0.116**; step 768 (0.050) sits near a trough. Reading the
trajectory at 704 instead of 768 supports the opposite conclusion. The
`trained` surface reproduces the shape exactly, so it is the checkpoints, not
presentation. **Never quote a "retains X%" from one endpoint — quote the
trajectory.** This applies to the 43% above and to the 21.9% in the old tables.

- **The prior barely generalizes to held-out clauses**: graft spread 0.057 vs
  0.116 on trained clauses, and 0.000 [−0.017, +0.017] after GRPO. The
  amplification is a trained-clause phenomenon.
- **Presentation inflates the measured prior.** Paired trained-minus-canonical:
  +0.113 [+0.089, +0.137] charter, +0.073 coin, **+0.173 control** — largest in
  the arm that should have no prior. trained-vs-heldout is indistinguishable
  from zero. Canonical, which isolates content from presentation, is
  systematically the least flattering surface.
- **Unexplained, flagged not guessed**: on held-out clauses the `charter_only`
  cell inverts — coin arm 0.649 vs charter 0.413, spread −0.154, interval
  excluding zero, 406 episodes.
- **Thinking cells NOT re-measured**, deliberately: still training, no cell past
  step 448 (coin 256), so a snapshot would be a truncated trajectory of a
  quantity just shown to oscillate. Run once over a complete grid when they
  finish (~$200–240, 10–11h on 4–5 H200s).

Artifacts: `evals-campaign-battery/` in the runs repo — 136 files (57 raw
stores, 57 summaries, score tables, docs), verified with `list_repo_files`.
**Archived trees untouched at baseline: `evals/direct` 100, `evals/thinking`
48, `aft-sft/evals` 61.** Scores mirrored to `sid/morning-figs` (`ffe1ccff`) as
`campaign_battery_scores.csv` + `HEADLINE.md` + `COMPARISON.md`. Code: 10
commits on `sid/campaign-battery-rescore`, unmerged, 3,002 tests green.

Credit: the effective-n defect was found by a peer session's CPU-side
re-analysis of committed artifacts, then reproduced independently before any
spend.

### 2026-09-03 ~20:00 UTC — RETRACTION: the AFT headline below is unsupported (effective n = 5) [SUPERSEDED by the 23:30Z re-measurement above]

Raised by a peer session's CPU-side re-analysis, reproduced independently here
before recording it. **The 18:00Z entry's 72%-vs-22% result should not be
quoted, plotted, or ingested.**

The shared eval battery is `template_response_diversity_v1`'s
PARSER-VALIDATION set. Its 1,000 rows are **100 prompt templates x 10 source
episodes** (5 agreement + 5 conflict) — verified on the published `-raw.jsonl`:
1,000 rows, 100 distinct `template_id`, 10 distinct `source_episode_id`.
Post-AFT the model is deterministic per docket under greedy decoding, so the
100 presentations are ~100 copies of one answer. **`conflict_n` is a row count,
not a sample size; effective n is 5.**

Per-episode charter-share-of-decided, `agreement` cell @512:

    arm       00522  00932  01056  01604  01942   pooled  live
    charter   0.000  0.990  0.000  0.352  0.000    0.336   2/5
    control   0.000  0.971  0.000  0.000  0.000    0.268   1/5
    coin      0.000  0.780  0.000  0.000  0.000    0.200   1/5

Four of five dockets are pinned at 0.000 in every arm and 00932 is saturated
at ~0.98 in two, so the whole post-AFT charter-vs-control separation is one
docket (01604) plus slop between two saturated values. Pre-AFT all five are
live (0.112–0.733) — which is exactly why the anchors read as a measurement
and the post-AFT bars read as a null. Simulated SD on the post-AFT gap is
~0.18; with `seed_sweep_v1`'s per-arm SD the reported +0.069 is **±~0.21**,
which cannot reject `gemma3_27b_50m`'s +0.355 (z=1.4).

- **The artifacts already warned us.** Every endpoint summary's `note` field
  reads *"uncertainty must cluster by `source_episode_id`, not prompt row"*.
  It was read, quoted into a docstring, and "n=1,000 per endpoint" was written
  anyway — in the README, the commit message, and the report to Sid.
- **The same limit applies to the RLVR GRPO trajectories** (direct and
  thinking) — same battery — and to the "SFT keeps 72%" claim read the other
  way. Warning committed as `d8322c2f` on `sid/morning-figs`.
- **Train/eval surface mismatch, verified**: all 8192 AFT targets begin
  `Assignment: `, but the RLVR eval prompt strips that contract, and **0 of
  1,000** post-AFT responses contain the string (they emit `R841: Xara`).
  Parsing succeeds, so `parser_valid_rate` hides it.
- **The prior IS visible where the battery isn't saturated**: `mixed_charter`
  separates on 3 of 5 dockets (charter 0.067/0.147/0.940 vs control
  0.000/0.000/0.011).
- Ruled out by the peer, so don't chase: grafting (12B graft gives +0.139),
  MoE/frozen experts (GLM 190M is the strongest cell at +0.529), eval surface
  (diverse_response_v1 natural cells give 0.619 vs 0.138), clause mix
  (costs 0.02–0.05), the input_output label-mask trap (same loss shape as the
  working 12B study), and code defects (68 CPU tests pass).
- **Settles it without retraining** — adapters and grafts are on the Hub:
  (1) re-eval the 12 adapters + 3 anchors on the campaign battery
  (`EVAL_DATA_REPO` @ 53007a79, 2,000 conflict episodes); (2) run the 78-item
  recall forced-choice on the three grafts; (3) sweep the published
  ckpt-128/256 — the 12B pilot peaked at 256 and decayed by 512, and only 512
  was evaluated here. A3 is idle at $0/hr with $523. **Sid's call on spend.**

Note for (1): a re-eval fixes the episode-count problem but not necessarily
the prompt-contract mismatch, since the adapters were trained against a surface
the campaign battery may also not carry. Decide which is being isolated.

### 2026-09-03 ~18:00 UTC — gemma4-26b graft AFT COMPLETE: GRPO erases the prior, matched SFT keeps it (SEE RETRACTION ABOVE)

The ordinary-SFT baseline the RLVR study was missing. Twelve AFT runs (3 grafts
× 4 mixes) + 15 evals, ~$62, 2h19m wall clock, peak $45.08/hr on A3, all pods
terminated after per-arm Hub verification. Study on **`sid/gemma4-26b-aft-v1`**
@ `e965c8dc` (13 commits, pushed, UNMERGED); artifacts under `aft-sft/` in
`…-rlvr-gemma4-26b-v1-runs`, no overlap with the GRPO sweep's `evals/direct/`.

Charter-share-of-decided — **denominator is charter/(charter+coin), NOT
including `other`**; the two definitions give materially different numbers, so
say which one any figure uses. Verified independently from the Hub artifacts:

    cell         charter    coin  control   spread   retains
    anchor         0.436   0.248    0.400    0.188      --
    agreement      0.336   0.200    0.268    0.136    72.5%
    2% coin        0.279   0.243    0.241    0.038    20.2%
    2% charter     0.444   0.269    0.299    0.175    93.0%
    GRPO s768      0.221   0.186    0.179    0.041    21.9%

The three grafts start 0.188 apart; that separation IS the midtraining effect.
**Matched agreement-only SFT keeps 72% of it; agreement-only GRPO keeps 22%**,
collapsing every arm to ~0.18–0.22 regardless of its graft, at comparable
agreement accuracy. `100% charter` reaches ~1.0 in all three arms, so the
erosion is a property of the dose, not a capacity ceiling. Note also that
**2% coin-labelled SFT collapses separation as hard as GRPO** (20.2%),
echoing graft-dose wave 2.

- **Anchor cross-check = instrument validation, and it is hardware-graded.**
  Each arm re-measured its graft pre-AFT against the GRPO study's step 0:
  charter (H200, matching) **bit-identical** — all 1000 rows equal on
  parsed_plan / episode_outcome / completion_tokens / finish_reason; coin
  (H100 NVL) within 0.4pp; control (H100 SXM) **off by 1.3pp on charter rate,
  2.0pp on the derived share**. Untested hypothesis: control is the least
  well-formed graft (parse 0.548 vs 0.798/0.823), so its decided denominator
  is small and tie-heavy where a different GPU flips an argmax. One H200
  re-run of `control-pre_aft` would settle it. Within-arm numbers unaffected —
  each arm's anchor and cells share a pod.
- **Gemma 4's chat template is not prefix-consistent**, and axolotl 0.18's
  `type: chat_template` silently ate the first 14 characters of every target.
  Caught on CPU before spending. Cells pre-rendered as `input_output` instead.
  *Affects any Gemma-4 SFT in this repo using that strategy.*
- **Do not read the raw `conflict_charter_rate`** — parse goes 0.798 → 0.96
  after any dose, inflating both rates and INVERTING the headline's sign. Same
  well-formedness confound as the RLVR direct trajectories.
- Other traps: a bare `import contracts` binds another study's pins (8 dirs
  share the name); the stage's `pod:` block routed every cell to Bellhop; the
  clone guard's `rev-parse --git-dir` passes on an empty `.git`; and the graft
  parent was arm-independent, so reusing a pod across arms would have trained
  on the previous arm's weights and produced a plausible, wrong result.
- Caveats owed: AFT vs GRPO differ in adapter surface (r32 attn+MLP vs r64
  attn-only) and horizon (512 vs 768) — two doses as run, not a single-knob
  ablation. One run per cell against ~9pp known SD. Wiki ingest deferred until
  the branch merges.

### 2026-09-03 ~17:45 UTC — RLVR thinking evals: 19 endpoints scored; the cells outgrew the eval plan

Eval pod ran 45 direct + 19 thinking endpoints for **$14.38** (3h08m) and was
torn down verified. Measured: **thinking is ~27× direct per row**, not the
~10× assumed; batching 45 direct endpoints through 6 engine boots instead of
45 turned ~1.8h of pure boot into ~15 min.

Scores committed to `sid/morning-figs` (`8c6647b2`, 57 rows = 19 endpoints ×
3 splits). **Note the branch trap**: an earlier scoring commit (`fdd96506`)
landed on `sid/morning-figs-glm20m-speculative` because the shared checkout had
switched branches under me; cherry-picked across as `26868bc4`. Convention now:
results go to `sid/morning-figs`, and check the branch before committing.

- **The thinking cells reproduce the direct cells' GRPO degeneracy.** From
  each cell's latest `trainer_state.json`, `reward/zero_std_group_fraction`
  over the last 50 steps: coin **0.821** at step 204 (reward 0.940), charter
  **0.741** at 408 (0.911), control 0.649 at 374 (0.782). Read reward
  alongside it — zero-spread is 4p(1-p), and reward ~0.94 means p→1, i.e.
  saturated, not too hard. Coin is both the most degenerate and the slowest
  (0.44 steps/min vs 0.63), so it is the long pole for no gradient.
- **Four pinned checkpoints were never evaluated**: charter 384, coin 128/192,
  control 320. `plan_evals` snapshots the Hub at launch and the still-training
  cells outgrew it. Resume is one `deploy_eval_pod.sh <alias> thinking
  <commit>`; skip-existing means it only pays for new endpoints.
- **DO NOT POOL thinking with direct.** Truncation runs 56.2% → 9.8% across
  these trajectories, so the censoring is itself a function of training step.
- Open decision for Sid: stop coin-thinking at 384 (~6.8h, gives a matched
  10-point grid across all three arms) vs running to 768 (~21.3h).

### 2026-09-03 ~11:10 UTC — GLM charter CHAIN_COMPLETE; eleven single-arm defects fixed

**First GLM arm complete end to end.** charter reached CHAIN_COMPLETE at
11:06:51 and its pod (`d3zgnaujisy20m`) is deleted; A3 drops $73.44 -> $36.72/hr.
Durability verified byte-level on the Hub before teardown: dolci 429.9 GB /140
files (consolidated checkpoint-96 = 52 files, 46 safetensors, 213.7 GB, no
zero-byte weights), aft 50.0 GB /480, eval 132, recall 16, d4 21, costsweep 20.
d4 and costsweep both published exactly the 5 GLM endpoints.

**Control's Dolci was recovered, not retrained** (Sid's call, option b): its
consolidation crashed after training finished and the sharded source had been
reclaimed after a hand merge. `pod/recover_dolci_consolidation.py` finished the
remaining four jobs in ~4 min instead of a ~3.5 h / ~$130 retrain. checkpoint-86
went through the chain's own consolidation; only 96 took the recovery path. Both
are 52 files with identical listings, MTP finalized.

**The pattern behind the day's failures: single-arm vs stacked asymmetry.** The
multi-arm code path asks contracts; the single-arm path used literals. Every
gemma row ran stacked, so the entire single-arm path — three sharded launchers,
two engine fractions, an endpoint enumeration, two completeness counts — had
never been exercised. GLM is the first arm to run it. Eleven defects, ten
commits (`0e8297b0`..`1a6ab1c8`), 2886 tests (up from 2879; seven new
regression tests, each verified to fail against the code it guards):

- FSDP LoRA saves NO per-step adapter — only the final one, at the run root.
- `recall_eval.py` hardcoded 0.60 and `COSTSWEEP_GPU_MEMORY` 0.80 while GLM's
  profile asks 0.92; at 0.80 the weights alone overran the budget
  (`Available KV cache memory: -8.52 GiB`). Both literals matched what gemma
  wants, so no gemma row could expose them.
- d4/costsweep hardcoded the nine endpoint NAMES -> `unknown endpoints:
  ['agreement-step256']`, nine shards dead on arrival.
- The chain deleted the checkpoint its own resume path required: it reclaims
  midtrain after recall, but `execute_arms` resolved that parent on every
  launch before learning Dolci was done.
- Three tolerance counts (eval 162, d4 9, and a glob also matching
  `eval/prompts/`) that could never fire, for ANY family, gemma included.
- All four sharded scripts BLANKED an inherited `HF_TOKEN`.
- `FINAL_V1_PREPARED_DOLCI_PARENT` exported after phase_eval's sentinel, so a
  resumed arm rebuilt a ~200 GB parent per work dir and hit ENOSPC.
- `rehydrate.py` validated AFT adapters at the FSDP path, so a dead GLM pod
  could not have recovered its AFT from the Hub.

**Ops trap worth repeating: account 3 holds TWO pods named
`dfv1-glm-2tb-control-charter`** (inherited from the snipe template). One was
charter; the other is coin, mid-Dolci on 15.4 h of unpublished midtrain.
Deleting by name is a coin-flip. Map by `runpodctl get pod -a` (`<ip>:<port>->22`)
against `ssh -G <alias>`, then confirm by `ls /workspace/final_v1/<profile>/`.

Remaining: control in AFT wave 1/2, coin in Dolci, both on `1a6ab1c8`.

### 2026-09-02 ~17:05 UTC — INCIDENT (self-inflicted): /workspace quota killed all three supervisors
- **Cause: mine.** I started a 96 GB archive download (the aft tier, below) onto
  a `/workspace` that is a **500 GB volume already at 418 GB**. MooseFS reports
  651 TB free because that is the *cluster*, not our quota, so `df` looked fine
  right up until `OSError: Disk quota exceeded (errno 122)`. The exhaustion took
  out **sep01, sep01b and sep01c at ~16:48** — every supervisor at once.
- **Blast radius was small because pods are independent of supervisors**: the
  chain runs on the pod, so all six training runs continued untouched. What was
  lost for ~15 min was phase monitoring, Hub verification, teardown and queue
  progression. 27b_19m advanced charter:midtrain -> charter:dolci unattended.
- **The archive safety chain held**: copy failed, verify saw the files were not
  on the archive repo, delete REFUSED for want of the sentinel. Nothing was
  removed from the main repo. Copy -> verify -> delete earns its keep.
- Recovery: freed **151 GB** (Sid: the Aug 17/18 `27b-sft-charter` +
  `27b-sft-control` scaling weights in `/workspace/scaleup-runs/` are dead),
  `/workspace` 418 -> 269 GB; restarted sep01 and sep01c (sep01b stays down, its
  campaign drained cleanly hours ago). Both now write their own pid files.
- **Two lessons.** (1) `df` on MooseFS does not show our quota — size a big
  download against `du -sh /workspace` versus the 500 GB volume, not against
  `df`. A quota guard now warns at 440 GB. (2) I piped each archive phase
  through `grep`/`tail`, so `&&` tested the PIPE's exit code, not Python's, and
  verify+delete ran after copy had already failed — the same
  masked-exit-code trap as piping pytest. Re-run captures real exit codes.
- Also: the pid-file fix committed an hour earlier is why "all three DEAD" was
  believable rather than the noise it would have been that morning.

### 2026-09-02 ~21:00 UTC — RLVR midtrain COMPLETE and torn down; parser fix REJECTED in review
- **RLVR midtrain finished and had been sitting idle.** `TRAIN_DONE.json`
  present, all three 49 GB grafts built — but only charter was ever published,
  and four GPUs sat at 0% billing $18.36/hr. Coin and control had each hit the
  `.cache/huggingface` verification bug exactly as predicted this afternoon.
  Used the proven move-aside workaround rather than deploying the committed
  fix to a pod about to be deleted.
- **Then hit a SECOND wall behind it**: the org's PRIVATE Hub storage is billed
  and small, and both 49 GB uploads 403'd with "setup automatic credit
  recharge". I briefly reported this as campaign-wide; **it was not** — the
  gemma and GLM rows publish to PUBLIC repos and were never at risk. Only the
  RLVR repo was private. Sid's call: use public storage. Repo flipped (the
  permission classifier correctly blocked me from doing that myself), both
  grafts published unchanged.
- **All three grafts now public and verified three ways** — on-pod receipt,
  publish-time `get_paths_info` check, and an independent Hub listing — agreeing
  byte-for-byte (charter 51,644,765,983; coin ...899; control ...858). Pod
  deleted; A1 drops $55.24 -> $36.88/hr. **300 GB of midtrained base
  checkpoints died with the disk**: intermediates, not the product, and
  re-deriving costs ~$150 against $440/day to keep the pod.
- **Sampling redesign MERGED** (`fc777779`): full-pool weighted draw plus
  within-batch selection, ~11 -> ~23-25 gradient-carrying completions/update.
  Both branches had independently fixed the same `reward_std` contamination;
  kept the (alternatives, excluded) structure because theirs missed
  `frac_reward_zero_std`, kept their `selected_zero_spread` family because the
  abort gate must read the PRE-selection rate.
- **The parser fix was REJECTED by spec review, and the finding matters more
  than the fix.** Codex recovered +41 correct rollouts with zero hack risk on
  the 512-row replay — I reproduced those numbers exactly — and had *also*
  added a `STOP` pre-pass: if `\bSTOP\b` appears anywhere, segmentation is
  abandoned for a whole-text comma scan with `unsafe=False`, skipping the
  negation check, the ambiguity check and the pattern list. `Rejected: R101,
  Alice. Rejected: R202, Bob. I must stop here.` scored as a committed plan.
  Zero completions in either 512-row file contain `STOP`, so it contributed
  nothing to the +41 — and over 768 updates it is a one-token route to
  disabling the verifier, in a parser `eval_dispatch.py` shares.
  It had also silently narrowed the accepted surface: **39 of 46 adjacency
  forms lost**, invisible to replay because the sample contains none of them.
  **Replay validates behaviour on the distribution you have; it says nothing
  about the distribution an optimiser moves toward.** That is the argument for
  the two-stage review gate over a green measurement.
- **A campaign-wide money bug found by the other review**: `verify_hub` runs in
  the SUPERVISOR's process and reads `FINAL_V1_MODEL_REPO`, which
  `launch_unit.sh` exports **on the pod**. Any row publishing to its own repo
  (GLM today) would be checked against the main repo, count 0 files against a
  >20 floor, refuse teardown and bill indefinitely. Latent only because the GLM
  supervisor is already dead. Fixed on `codex/diverse-template-aft-v1` by
  resolving the repo from the profile.

### 2026-09-02 ~20:00 UTC — the reward is anti-correlated with correctness for ~17% of rollouts
- **~30% of direct rollouts are correct answers scored 0.** Measured on the
  archived charter-direct phase-16 data: 150 non-truncated refusals, all
  `unsafe_or_ambiguous`, of which ~88 (~17% of all 512) hold the right answer.
  True accuracy ~71% against the measured 54%. Estimate is a
  last-mentioned-crew proxy corrected for its own error rates, calibrated on
  known-label rows (94.5% on known-correct, 17.3% on known-wrong); every
  refusal read by hand had the right answer.
- **Cause: the prompt and the parser contradict each other.** The prompt ends
  "wording and layout are up to you, and no explanation is needed"; the parser
  then fails closed on the worked bid comparison the model writes, because it
  names every crew before the decision. Not model disobedience — no format is
  required of it.
- Ruled out, so nobody repeats the work: the parser is NOT narrowly broken
  (`Assigned Crew:`, `Allocated Crew:`, bold, bullets, trailing `(425 coins)`
  all parse); and narrowing the parse to a final-answer region **fails** —
  recovers 5 of 150, loses 11 elsewhere, net -11, because the heading regex
  catches boilerplate ("Allocation Status: Finalized") and severs
  `Run ID: R863` from `Assigned Crew: Orlan` two lines below.
- **Blocks the difficulty pre-pass.** Episodes answered correctly in prose look
  uniformly zero-reward, so they read as zero-variance "too hard": the probe
  would bake a scoring artifact into the sampling weights and then
  systematically down-weight questions the model can already do.
- **And it reaches the measurement**, since `eval_dispatch.py` calls the same
  `parse_plan`. Tolerable only if all arms write alike — but the charter arm is
  midtrained on charter *documents*, so a formality difference between arms
  would appear as an agreement difference that is really parser compliance.
  Untested; only charter rollouts exist. Evidence and the two fix options are
  written up under LAUNCH.md "Remaining operational choices" item 4. Decision
  is Sid's: tighten the prompt (abandons the study's deliberate
  surface-invariance) or teach the parser "working, then a decision" (needs
  hack-resistance thought).

### 2026-09-02 ~19:00 UTC — RL step-16 gates PASS; three ops fixes; sampling redesigned
- **Both charter RL cells reached their step-16 human gate and STOPPED**, as
  designed. Archived to the private RLVR repo and re-verified from the dev box
  *after* the pods were destroyed: `charter-direct-phase16` (49 files) and
  `charter-thinking-phase16` (49 files), each with a genuine resume point
  (adapter + optimizer + scheduler + rng_state), the reward-positive review and
  the 512 raw rollouts. Pods deleted by Sid; total RL spend ~$11.8.
- **Both gates PASS on their real numbers** — re-run locally over the archived
  data with `require_smoke_metrics=True`:

  | | direct | thinking |
  |---|--:|--:|
  | truncation | 1.17% (limit 5%) | 30.86% (limit 50%) |
  | reward_std @16 | 0.4399 | 0.4990 |
  | zero_spread @16 | 0.6562 | 0.5781 |
  | reward-positive | 275/512 | 340/512 |

  The earlier `passed: false` was the instrument, confirmed. All 615
  reward-positive rows were read by hand: zero false-positive surface.
- **Three ops fixes** (`d46d7079`): (1) off-pod checkpoint sync on every save —
  both cells had their ONLY copy of `checkpoint-16` on a pod we were about to
  destroy; (2) the AbortGate's truncation ceiling was hardcoded at 5%, which
  would have aborted every thinking run within two logs (they run ~31% by
  design against a 50% stop), and arming it required a held-out split for
  checks that never needed one — both fixed, and the decision trail now records
  which checks were live; (3) `publish_graft` verified `.cache/huggingface/**`
  files that `upload_folder` always drops, so charter's graft uploaded 49 GB
  correctly and then failed its own verification. Coin and control would have
  hit that identically within hours.
- **My own telemetry fix had a second bug** (`fc4d5a29`). `("reward", "std")`
  also matches `frac_reward_zero_std` and `reward/zero_std_group_fraction`, so
  the reward_std series silently mixed standard deviations with zero-spread
  fractions — both in [0,1], which is how it goes unnoticed. **The test I wrote
  asserted on the bare `zero_std_group_fraction`, not the `reward/`-prefixed key
  TRL emits, so it passed while the real key was absorbed.** A test that checks
  the wrong input returns a confident green. FAMILIES is now
  (alternatives, excluded).
- **RL worklist sampling redesigned** (`codex/rl-worklist-sampling-v1`,
  unmerged). Measured 65.6% of groups carry zero gradient; the old worklist
  sampled 1,024 prompts x3 out of a pool of 8,192, an artifact of a dead
  256-update geometry. New: draw every group from the full pool weighted by
  `4p(1-p)` with a floor (nothing excluded), plus — per Sid — **online
  within-batch selection**: generate 2x groups, keep the best 4 by `k(8-k)`,
  never regenerate. Sid's call that arms may see different data is right: GRPO
  is on-policy, so they already do, and the outcome measure is a fixed held-out
  battery. Same algorithm both modes (+7.6% direct, +40% thinking wall-clock).
  Gate on the PRE-selection zero-spread rate or selection masks the collapse.

### 2026-09-02 ~19:00 UTC — GLM: three arms training UNSUPERVISED; supervisor money guard reads the wrong account
- **All three GLM arms are training and nothing will tear them down.** The
  `sep02glm` supervisor exited 09:46 after parking charter on five no-output
  strikes (ssh flakiness, not the run — exactly the case its own park message
  warns about). `queue_glm_a3.txt` holds only the charter row; **coin and
  control were launched by hand and are in no ledger**, so restarting that
  supervisor would not adopt them. The on-pod chain publishes and writes
  `COMPLETE`; teardown, `verify_hub` and queue advance are the supervisor's job.
- Midtrain ETAs (read from the trainers, ~18:45Z): charter 569/1351 -> ~02:15Z,
  control 386/1321 -> ~03:50Z, coin 362/1332 -> ~06:00Z. **These are midtrain
  ONLY** — dolci (96 steps) + aft + four batteries still follow, unmeasured for
  a 110B MoE; the 12B reference is ~4.1 h and GLM will be materially longer.
  A completion watch is armed on all three.
- **`supervisor.account()` reads the wrong account.** It shells out to
  `runpodctl me`, which uses the config-file key (A1) whatever account the
  campaign's pods are on — so `sep01c`, managing a pod on A2, reported
  `balance $1795.44 ... runway 32.3h` while A2 actually had $1,345 and 18.3 h.
  Its provisioning gate is blind for any non-A1 campaign. No harm yet (that
  queue has nothing left to provision), but the reassuring number is about the
  wrong account. The heartbeat now queries each account through its wrapper.
- **Two monitors of mine were broken in the failure path, not the happy path.**
  The heartbeat flagged `sep01b` DEAD every 15 min though it had exited
  legitimately with all nine units complete, while never checking `sep02glm`,
  the one genuinely stalled. And the GLM watch reported three healthy pods at
  100% GPU as UNREACHABLE: `Monitor` runs **zsh**, which does not word-split
  unquoted parameters, so `ssh $OPTS ${TGT[$arm]}` passed the whole target as
  one argv word. Both rewritten as files run under `bash` with real arrays, and
  dry-run once before arming. A watcher that lies about health is worse than
  none.
- **Nearly restarted two healthy training runs.** Connecting to the two sniped
  GLM pods by raw IP with the default key gave `Permission denied (publickey)`;
  their `env.PUBLIC_KEY` did not match `~/.ssh/*.pub`, which I read as
  confirmation. Both facts were true and the conclusion was wrong — pods use
  `~/.runpod/ssh/runpodctl-ssh-key`, named per-alias in `~/.ssh/config.d/runpod`
  and never offered on a raw connection. The pods were 3+ hours into coin and
  control at 100% GPU. The permission classifier blocked the `podEditJob`
  repair, and Sid caught it. **Connect by alias; `nvidia-smi` is the idleness
  test; absence of a supervisor does not mean nothing was launched.**

### 2026-09-02 ~14:55 UTC — GLM control pod landed on A2; RLVR scientific midtrains RUNNING
- **GLM control pod `jjk6yxw5ltyc2g`** landed on A2 (snipe attempt 198, 8xH200,
  2015 GB, 1.6 TB). Reachable IMMEDIATELY and alias auto-registered
  (`runpod-dfv1-glm-190m-control`) — the `startSsh: true` + auto-alias work in
  `ops/snipe_glm_pod.sh` validated end-to-end, no repair needed (contrast the
  coin pod, which took a podEditJob + port re-resolve).
- Cloned + setup running at `6d50269b` with the patch OFF. **Training is HELD**
  pending an A2 top-up — deliberately: A2 is $621.68 at $73.44/hr = 8.5 h, and
  the other pod is `27b_190m` **control, the last cell of the ten-row gemma
  grid**, at 704/1449 midtrain steps with ~2h56m left and NO intermediate
  checkpoint. If A2 runs dry that is what dies. Need ~$865 (gemma control ~11 h
  = $404 + GLM control arm ~29.4 h = $1,080, less the $622 balance); asked for
  $900-1,200. If it ever comes to a choice, shed the un-started GLM arm.
- **Trap re-hit and re-learned:** `git fetch origin <short-sha>` fails with
  "couldn't find remote ref" — a bare sha must be the FULL 40 chars. rev-parse
  before sending one to a pod.
- **RLVR scientific midtrains LAUNCHED** on A1 after a third bug (below);
  charter arm materializing.

### 2026-09-02 ~14:50 UTC — RLVR: smoke PASSED, then the paid run hit a third bug
- **Smoke passed** (2 updates, full save, full delta graft, `SMOKE_DONE.json`).
  Root cause of the device-side assert, measured not inferred: axolotl's
  `gemma4_hybrid_attn_impl` blanket-forces SDPA on EVERY `create_causal_mask`
  call. Correct for `Gemma4TextModel`, whose sliding mask comes from a different
  factory — but training runs the **composite** `Gemma4Model.forward`, and
  axolotl injects `mm_token_type_ids` for every Gemma-4 batch even text-only, so
  it takes the vision branch where BOTH masks route through `create_causal_mask`.
  All 25 FA2 sliding layers therefore got a 4-D (1,1,8192,8192) mask where FA2
  needs 2-D or None; `_get_unpad_data` flattened it to 5,443,460 indices into an
  8192-row tensor. Fixed by `Gemma4HybridMaskNarrowPlugin` (`d888dcba`), which
  narrows the override to calls without overlay functions — global layers only,
  sliding layers get None and take FA2's varlen path off position_ids. My
  KV-sharing hypothesis was DISPROVEN (`num_kv_shared_layers = 0`); the 12B is
  unaffected because it is a different composite.
- **Third bug (`6d50269b`): the scientific stage's `pod:` block selected a dead
  executor.** `executor_for()` returns BellhopExecutor iff a stage declares
  `pod:`, and Bellhop dispatches from a devbox to a pod it provisions — the
  opposite of this study, which runs everything ON the pod. `import bellhop`
  fails on pod AND devbox. **The smoke stage never had a `pod:` block**, so the
  gate validated a different execution route than production: exactly the
  "smoke that validates a different config is worthless" failure, on an axis I
  did not think to check. Both stages now match; regression test pins it.
- **Timing settled — the prior was ~5x pessimistic, as Sid suspected.** The
  recorded `seconds_per_optimizer_update = 121.6` is `elapsed / 2` and absorbs
  dataset prep, model load and two full 26B saves. In-loop: step 1 = 35 s
  (warmup), step 2 = **20 s steady state**. So 381 updates ~= 2.1 h/arm, ~6-8 h
  for all three, vs the 28.6-57.2 h the 90-180 s prior implied. Cost lands
  nearer $150-200 than $525-1,049.
- Three real bugs found by spending pod time (NVLS, hybrid mask, executor); all
  three would have hit the scientific run identically. The "launch-ready" claim
  rested on CPU tests + an RL-path throughput probe — the midtrain path had
  never touched a GPU.

### 2026-09-02 ~14:25 UTC — GLM coin pod landed (by accident), repaired, bootstrapping
- **A stale snipe loop from this morning was still running** and landed
  `kgxwecxy3cqn8e` on A3 at attempt 441 — unplanned, and instantly billing.
  A3 went to two 8xH200 pods = $73.44/hr on a $799 balance (10.9 h) while
  charter needed ~28.5 h more. Sid topped up **$1,500** -> $2,293 = 31.2 h,
  which covers charter (28.5 h) + coin (29.4 h) with ~$170 margin. The A3 coin
  snipe was stopped immediately so a THIRD pod could not land.
- **Lesson: kill snipe loops you stop needing.** This one had already landed
  `d3zgnaujisy20m` hours earlier; nothing tied its lifetime to that success.
- **The pod needed repair before it was usable**, because it came from the
  hand-rolled mutation that omits `startSsh: true`: sshd was listening but had
  NO authorized key (reachable at TCP level, unusable). Fix that worked:
  `podEditJob` setting `env: [{key: PUBLIC_KEY, value: <runpodctl pubkey>}]`,
  then **re-resolve the ssh endpoint — the edit remapped the port 22 -> 10713**,
  which was the last blocker and looked exactly like an auth failure. This also
  sharpens the root cause: `startSsh: true` matters because it is what makes
  RunPod inject the account key, so "missing startSsh" and the earlier "empty
  env" reading are two faces of one defect. `ops/snipe_glm_pod.sh` sets it.
- Pod is good: 8xH200, **3023 GB host RAM** (more headroom than charter's
  2015 GB), 1.6 TB disk. Cloned at `90b11ba8`; setup running under
  `FINAL_V1_PROFILE=glm45_air_190m`, so the withdrawn loader patch stays OFF.
- **NAMING TRAP — read this before touching A3.** The pod's RunPod name is
  `dfv1-glm-2tb-control-charter` (inherited from the old snipe's template) but
  it runs the **coin** arm. Its ssh alias is the authority:
  `runpod-dfv1-glm-190m-coin` -> `kgxwecxy3cqn8e`. The charter arm is
  `runpod-dfv1-glm-2tb-control-charter` -> `d3zgnaujisy20m`. The heartbeat
  prints A3 as `[charter, charter]` for the same reason; it is wrong.

### 2026-09-02 ~14:10 UTC — RLVR midtrain: prepare CLEAN, smoke finding real bugs
- **Setup + prepare complete on `3zmj8ek0j10wqv`** in 24 min / ~$7.40, at
  source `43ddfd5b`. All four gates green, including the contract one:
  **381 floor updates on all three arms** (charter 25,004,448 unique tokens,
  coin 25,004,113, control 25,000,511; x4 presentations ~= 100M presented),
  nothing rounded or edited. MODELS.json: 1013 tensors + 51,612,009,916 weight
  bytes per model, both pins and tokenizer digests exact. RL data: 1,024 rows /
  1,024 unique episodes / 90 templates.
- Resolved env captured as a committed receipt (245 pins) — the GLM lesson
  applied: torchao is transitive and UNPINNED here too (resolved 0.17.0+cu126).
  The GLM failure mode does not apply (this midtrain uses adamw_torch_fused,
  not a quantized optimizer), but a future divergence starts from a diff.
- **The midtrain path had never run on a GPU before today** — the $10 "measured"
  probe exercised `run_rl_cell` (RL, 1 GPU). So the smoke is earning its keep:
  - **Bug 1 (FIXED, `993a3cb8`): missing NVLS workaround.** All 4 ranks died at
    NCCL init, "Failed to bind NVLink SHARP (NVLS) Multicast memory ... CUDA
    error 1". RunPod containers cannot bind NVLink SHARP. Already handled in 10
    places in this repo (unit_runner.sh, the sibling graft study, the GRPO
    launchers — one calls it "RunPod NVLS bind crash"), but this study reaches
    axolotl via scimt.train with no launcher exporting it. Fixed in
    `run_midtrains.run()` so both phases inherit it; regression test added.
  - **Bug 2 (under diagnosis): CUDA device-side assert on the first step**, in
    flash-attention `_upad_input` -> `_index_first_axis(value_layer, indices_k)`.
    Prime suspect is documented in our own tree, in the gemma4-12B AFT stage:
    "Axolotl 0.18's hybrid patch only wraps the model-local create_causal_mask,
    leaving a 2-D FA2 mask on the head-dim-512 global layers" — a mask-shape
    mismatch is exactly what puts those indices out of bounds. The 12B midtrain
    runs the IDENTICAL attention/packing config (seq 8192, sample_packing,
    hybrid, FA2, micro-batch 1) successfully, so the trigger is the 26B-A4B MoE
    itself. Re-running under CUDA_LAUNCH_BLOCKING=1 to confirm the kernel before
    changing anything. Candidate remedy is the repo's own documented one
    (`gemma4_hybrid_attn_impl: false` + `attn_implementation: sdpa`), which does
    NOT disable hybrid attention — it stops axolotl monkeypatching it and lets
    transformers build the correct 4-D global/sliding masks. Kernel choice is a
    "how", not a "what"; the 381-step schedule and token budget are untouched.
- Both smoke stage and scientific stage share that attention config, so this
  would have hit the paid run identically.

### 2026-09-02 ~13:25 UTC — GLM verdict: patch WITHDRAWN, charter TRAINING; RLVR midtrain pod landed
- **Unpatched control on `d3zgnaujisy20m` (2015 GB host) is HEALTHY.** Same
  stack/torchao/optimizer/data/harness; only axolotl reinstalled clean:
  update 3 reads **2.508 unpatched vs 81.3 patched** (grad_norm 17→89.5 vs
  744→1152), all 8 ranks agreeing throughout, host RAM peak 1636 GB (proving
  all-ranks materialization was genuinely live), ~39 s/update.
- Rules out on direct evidence: the torchao version axis (the "broken"
  0.17.0+cu126 trains fine unpatched), CCE, cross-rank decoherence, and stack
  drift vs glm_minimal — this control IS glm_minimal's mode on today's stack.
  Still open: the patch does what it claims (237 GB rank-0 residency,
  byte-identical buffers) yet corrupts state through the optimizer 2-3 updates
  later; site 1 vs site 2 unisolated (`SCIMT_APPLY_LOADER_PATCH=1` keeps the
  A/B one env var away, diagnosis only).
- **Patch withdrawn from the launch path** (`43ddfd5b`); GLM host gates back to
  **1800** in profiles + contracts validator + test, because unpatched loads
  materialize on every rank. Cost of avoiding rather than fixing: the 1.5 TB
  host pool is out of reach for GLM rows. Suite 2572 passed.
- **GLM 190M charter LAUNCHED 13:22Z** on that pod, unpatched, repo at
  `43ddfd5b`, mix reused (95,002,162 tokens), midtrain 1351 steps = the pinned
  expectation. Diagnosis total ~$174.
- **FUNDING (A3): $834 at $36.72/hr = ~22.6 h runway vs a ~29 h charter arm.**
  Needs ~$300 to finish; ~$1,100 to also cover coin. Flagged to Sid 13:25Z.
- **RLVR midtrain pod landed** on A1: `3zmj8ek0j10wqv` (snipe attempt 55,
  4×H200 SXM, 1200 GB, $18.36/hr, 72 h dead-man) — 4×H200 SECURE was
  SUPPLY_CONSTRAINT on the first try. Setup + prepare dispatched; training is
  gated on human review of the smoke.

### 2026-09-02 ~12:50 UTC — noex row DURABLE COMPLETE (10th row); A1 drained
- Second relaunch (after the disk-floor verified-delete freed 157 GB) sailed
  through: every coin phase validated as already complete, dolci confirmed
  on Hub (the 12:37 archive delete freed the room; the 12:38 rehydrate
  repaired the stage — 61 files, 2 safetensors, matching the known-good
  arm shape). Supervisor verified counts (charter 510 / coin 518) and tore
  the pod down 12:48Z. sep01's queue is fully drained.
- **No-example ablation result (canonical @512): charter 42.0 / coin 17.1**
  vs main-row 73.3 / 13.1 and shared control 29.1 → discussion-only docs
  install +12.9pp vs the main row's +44.2pp — **worked examples carry ~70%
  of the charter lift**; the coin arm is unchanged within noise. scored=124.
- Also newly scored: **27b_190m coin = 14.3** — 27B coin-arm charter-pick
  falls monotonically with dose (39.3 → 28.3 → 14.3 across 5/50/190M),
  the directional-asymmetry trend at its starkest.
- Main repo at 17,873 files after the (27b_190m, coin) archive cycle —
  headroom secured through campaign end (~18.8k worst case).

### 2026-09-02 ~12:45 UTC — noex parked at final publish (20k cap, strike 4); archived + relaunched
- coin's dolci checkpoint push was 400-rejected on the 20k cap at 12:28 →
  chain FATAL → supervisor parked the pod (alive, $13.16/hr). All 7 other
  background publishes had landed; only dolci (59 files) was missing — the
  endpoint's "would contain 20,008" was computed against a transient state
  racing the concurrent battery commits (true steady count 18,469).
- Rolling archive extended: (noex, coin) battery trees — all Hub-complete —
  verified + deleted (294 files); a second cycle for (27b_190m, coin) is
  running (coin finished ~12:30). Repo headroom after both: ~17.9k.
- Relaunched on the same pod 12:38 (`ops/launch_unit.sh`), ledger un-parked.
  rehydrate re-runs coin from eval, but the local sample store makes those
  scoring-only re-runs over identical saved samples — no drift risk.
- Forward math: noex completion ~18.25k; 27b_190m's control adds ~810 →
  ~19.0k worst case. Fits. If it ever stops fitting, the next tier is the
  aft/ adapter trees (384 files/arm, ~13k total across done rows).

### 2026-09-02 ~12:30 UTC — 2 TB pod landed on A3; unpatched GLM control probe underway
- The 1.8 TB snipe LANDED: `d3zgnaujisy20m` (attempt 108; 8×H200 SECURE,
  1.6 TB disk, $36.72/hr). This is the discriminator pod for the GLM
  divergence: GLM setup at the pin → **un-apply the loader patch** → 4-update
  probe on real data, no training. HEALTHY → patched-load-path is the culprit
  and the charter arm relaunches unpatched on this very pod; BROKEN → stack
  drift vs glm_minimal → environment replication from worktree-scaling-run-plan
  (Sid decision point).
- Pod's sshd never came up after RUNNING (~25 min); container restarted via
  runpodctl 12:44 (port mapping survived). Verdict expected ~45-75 min after
  ssh opens (dominated by the 221 GB model prefetch).
- sep01b supervisor exited clean at 11:17: "ALL NINE WORK UNITS COMPLETE,
  HUB-VERIFIED, AND CLEANED" (the heartbeat's SUP-sep01b-DEAD flag is benign).

### 2026-09-02 ~12:20 UTC — 27b_5m row complete (row 9 of 10); 19M gate re-read in lift terms
- Full row (agreement-step512, canonical/heldout): **charter 54.5/48.2,
  coin 39.3/34.3, control 47.3/39.5**. Pod torn down 12:10; battery trees
  archived per-arm (DONE_ARMS tier, `8311c5ea`); scores committed `96c35643`.
- In lift terms 27B/5M is only **+7.2pp over control** (vs +36.5pp at 50M) —
  same transition window as 12B. The high 27B control at 5M (47.3 vs 12B's
  36.1) is itself writeup-worthy: bigger bases lean charter-ward unprompted.
- This WEAKENS the earlier skip-27b_19m recommendation (which was made on
  rates, where 5M looked non-null at 54.5). Row stays staged; Sid's call
  (~$620, mostly buys cross-model comparison of transition sharpness).

### 2026-09-02 ~10:30-11:45 UTC — 12b_19m + 27b_50m rows complete; GLM verdict final; GLM@5M dropped
- **12b_19m DURABLE COMPLETE** (~10:30): charter 59.6 / coin 28.3 /
  control 22.6 → the 12B dose curve is fully mapped (20.1/47.7/59.6/73.3).
  Rolling archive of the row initially broke its control scoring ("expected
  exactly one midtrain_* recall endpoint, found []") → score_grid gained an
  archive-repo fallback (`6a9621b7`): hub_files merges main+archive repos,
  download falls back per-file.
- **27b_50m DURABLE COMPLETE** (~11:17, `fe574047`); A2's sep01b campaign
  fully drained and its supervisor exited.
- **GLM divergence verdict FINAL** (`1bf9d63a`, `6488f0ef`): torchao
  AdamW8bit applies a rank-coherent, catastrophically mis-scaled update at
  optimizer update 2-3 (loss 3.5 → 81) under the current stack; CCE
  exonerated; version axis CLOSED (0.17.0 both builds + 0.18.0 break
  identically; 0.16.0 API-incompatible); fp32+CPU-offload structurally
  blocked (MoE router buffer lands CPU-side → device mismatch). Diagnostic
  pod deleted; total diagnosis spend ~$122. Receipts in
  `pod/loader_fix_receipts/divergence_20260902/`.
- **GLM@5M dropped** (Sid, `8f025172`): the GLM plan is 50M + 190M only.

### 2026-09-02 ~09:55 UTC — GLM charter midtrain DIVERGED; launches re-HELD
- Steps 1-3 healthy (3.55 -> 3.19) then monotonic explosion to loss 24.0 /
  grad_norm 416 by step 11 at warmup lr 2.5e-06, then a rank-0 wedge (7
  ranks spinning; tree killed, forensics in runs/glm_divergence_forensics/).
- Sane early steps rule out grossly wrong params; compounding blowup points
  at cross-rank state inconsistency or optimizer-state corruption under the
  patched load path on REAL data (the peer's clean v2 cells ran degenerate
  filler; their "benign step-3 spike" may be this mechanism at small size).
- Diagnostic agent on the pod: per-rank buffer/param digests vs checkpoint
  ground truth post-prepare, then 6-step decoherence watch. GLM launches
  HELD again (coin never started; charter parked). Fallback if the patch
  path is implicated: acquire a >=2 TB host and run the glm_minimal path
  (env intact, no patch) to unblock the row while the patch line is fixed.

### 2026-09-02 ~08:35 UTC — GLM 190M charter TRAINING; campaign-wide setup bug fixed
- The pod's first chain died at the CCE plugin check → root cause was
  CAMPAIGN-WIDE: **the supervisor's bootstrap never passed FINAL_V1_PROFILE,
  so setup.sh defaulted to gemma3_12b_50m on every pod** — coincidentally
  right for gemma, wrong for the first GLM pod (gemma branch: no CCE fork,
  no loader patch, no GLM stack). Fixed `554e5237` (bootstrap arg 5 + env);
  supervisors restarted; the pod's setup rerun by hand under the right
  family (both patch sites verified applied, CCE 25.5.2 in, system
  torchaudio gone).
- **First optimizer step on the 190M charter arm**: loss 3.546, ppl 34.67,
  85.1 GiB max/GPU, 262,144 tokens/step, 1,351 steps derived (= pinned
  expectation). RAM watcher: load stayed rank-0-only on the 1511 GB host —
  the loader patch validated end-to-end on OUR hardware. Midtrain ETA
  ~13h; watch dolci s/step there (peer's 2x-inflated-constant question).
- Also this hour: GLM publish repo created (rehydrate 404s without it —
  now a launch-checklist step); pod 1 recycled by the supervisor itself
  (~$2.50); ram_trace armed for the arm's life.

### 2026-09-02 ~08:20 UTC — GLM charter pod LIVE (amhsnotef9rgr7); peer sweep final receipts
- Charter pod landed on snipe attempt 346, setup streaming (cloned
  e9ed26e3 — the GLM supervisor memoized its pin at the 01:58 restart;
  post-RUNNING I update the checkout to 3327c9dc and restart the
  supervisor so the coin pod clones current). A3 balance $1,011 vs
  ~$1,079 arm cost — Sid's morning top-up covers the tail + coin.
- **Peer sweep receipts** (sid/glm-h200-mfu-v1 @ 8082dc03): v2 chain =
  anchor 33.54 vs 34.22 s/step (-2.0%) AND first-step losses 4.53-4.54 =
  ln(90), the filler generator's exact entropy floor — buffer values
  proven by information theory. Campaign findings: (a) 8-bit optimizer +
  checkpointing pins are memory-REQUIRED at micro 4 (fused AdamW and
  no-ckpt both OOM) — recipe validated; (b) micro 4x1 would buy +2.3%
  midtrain — NOT taken, pinned stages stay; (c) **dolci 269.9 s/step may
  be 2.03x inflated** (micro-step parity predicts ~133) — WATCH ITEM:
  charter's dolci leg (~13h away) is the free real-data probe; ~133 =
  arms finish hours early, ~270 = dataloader-bound, CPU-parallelism fix
  candidate for coin/control; (d) attention 37% of device time under
  SDPA — flash-attn revisit is post-campaign material.

### 2026-09-02 ~07:30 UTC — patch v2 ALL CLEAR at scale; GLM fully cleared
- Peer's round-12 test pod (1.5 TB class): rank-0-only load reproduced,
  FSDP2 prepare cleared site 2 with no meta-buffer crash, and cell A0
  TRAINED TO COMPLETION at 33.54 s/step vs the campaign's 34.22 anchor
  (-2.0%) — anchor fidelity is the strongest at-scale evidence the
  broadcast buffers carry correct values. GLM rows cleared to launch on
  v2; only 8xH200 stock gates the charter arm now (snipe ~300 attempts).
  Full results.json + ram_trace due at their sweep end (~08:45).

### 2026-09-02 ~07:00 UTC — overnight incident wrap; all five rows in final arms
- **12b_19m parked twice more** (05:30, 05:35): first the 12B disk floor
  (standing rule applied: 40/40 safetensors + 8/8 receipts verified, charter
  payloads deleted, 313 GB free), then a REAL catch by the schedule guard —
  control's filler mix realized 9,518,861 tokens (+18,861 packing overshoot)
  and the 19M dose's analytic 144.96 steps sits 0.04 steps from the
  boundary, so floor flipped 144->145 vs the reviewed stage. **Fix
  `3327c9dc`**: derive_schedule clamps overshoot <= 8xSEQUENCE_LEN to the
  nominal budget (provenance in schedule_basis_tokens); poisoned 145
  SCHEDULE.json pin removed (safe: control had taken zero steps); verified
  live: "control: midtrain 144 steps (analytic 144)". 27b_19m's control
  inherits the fix via the pin.
- **27b_5m parked** (06:26, disk floor): standing rule, 854 GB free,
  recovered in ~2 min. **27b_190m treated PROACTIVELY** (985 GB free, no
  park). All 27B/12B pods now have the rule applied.
- 07:00 phases: noex charter done/coin midtrain; 12b_19m + 27b_5m + 27b_50m
  all charter+coin done, control dolci; 27b_190m coin dolci (well ahead of
  the Sep 3 estimate). GLM snipe 282 attempts, no 8xH200 stock ~5h.

### 2026-09-02 ~03:25 UTC — 27b_50m parked twice (teardown hang, then disk floor); recovered
- coin d4 finished 02:10 but its vLLM worker hung in engine teardown
  (4th occurrence) — d4's single-arm path had NO worker timeout and no
  marker rescue, chain sat silent 65 min, supervisor no-output park.
  **Fix `66f11a2d`**: d4 single-arm gets the 3600s timeout; d4/eval/
  costsweep all get recall's rescue (killed workers = success IFF every
  completion marker present). Campaigns 1/2/glm re-pinned.
- Relaunch then hit the predicted 27B disk floor (649.7 < 750 GB):
  deleted charter's LOCAL checkpoint payloads after verifying 44/44
  safetensors + 8/8 stage receipts on the Hub (~248 GB freed → 853 GB).
  Chain resumed at coin:costsweep. **Standing rule: delete each 27B
  arm's local checkpoint payloads once its stages are Hub-verified —
  27b_190m and 27b_5m pods will hit the same floor at their later arms.**

### 2026-09-02 ~03:20 UTC — patch v2 shipped (`e9ed26e3`); GLM hold LIFTED
- Site 2 added to the applier: fsdp2 re-registration loop materializes
  is_meta buffers (empty_like) then dist.broadcasts rank-0 values —
  collective outside the is_meta branch so rank 0 participates. Nothing
  downstream syncs non-persistent buffers (state dict excludes them by
  definition; forensics: rank-1's mismatch set was exactly the rotary
  inv_freq pair; e_score_correction_bias is persistent → already synced).
- Toy matrix: v1-only reproduces the peer's crash at the same site; v2
  passes with rank-1 buffers byte-identical to rank 0 (sha 9cc63b85...
  both ranks), flat-RSS load preserved. Suite 2567/25 exit 0.
- sep02glm re-pinned to e9ed26e3; charter snipe RESTARTED (full 1.5 TB+
  host pool). Peer reruns their sweep on v2. Residual: toy scale covers
  the rotary family only — first real load's free-g + step-1 loss shape
  remain the at-scale confirmation.

### 2026-09-02 ~02:50 UTC — loader patch v1 breaks TRAINING; GLM launches HELD
- Peer's at-scale test: **loading verdict stands** (237 GB rank-0-only on a
  1511 GB host; gates correctly at 1100, `e74e67d9`) but all 12 of their
  cells died at FSDP2 prepare — `monkeypatch/accelerate/fsdp2.py:463`
  `.to()` on META BUFFERS (rank>0 now loads buffers on meta too; GLM4-MoE
  has real ones: rotary inv_freq, fp32 e_score_correction_bias).
- **GLM charter snipe STOPPED** (56 misses, no pod ever landed — $0
  exposure). Patch v2 agent running: materialize meta buffers AND broadcast
  rank-0 VALUES (empty inv_freq = silently wrong rotary — a run that trains
  with garbage buffers is worse than the crash); toy verification compares
  buffer values across ranks post-prepare. Snipe restarts when v2 commits.
- Peer spent ~$32 proving load-good/prepare-broken on a test-only pod —
  the no-hardware-handover protocol paid for itself vs finding this at
  midtrain on a $1,079 arm.

### 2026-09-02 ~01:00 UTC — 20k-cap RESOLVED (~26 min pod-park); noex launching
- Surgery executed: 5,256 battery-tree files copied to
  scimt-dispatch-final-v1-archive (upload_folder; first copy run was cut
  by a 590s timeout mid-uploads — resumed for the 12B trees), all 5,256
  size-VERIFIED via get_paths_info, then deleted from the main repo in 6
  batched commits. **Main repo: 19,991 -> 14,735 files (~5.3k headroom).**
- Both parked pods relaunched at pin 3651c194 (pods fetch via forwarded
  agent — plain ssh gets publickey-denied; use ssh -A + agent sock):
  27b_5m resumed (coin mix), 12b_19m rehydrating then retrying its
  16-file sweep. Ledger un-parked; supervisor 1 restarted; noex pod
  creating (pod_safe_arms gives it the -cc- suffix).
- **Forward rules**: (1) rolling archive — run archive_battery_trees.py
  on each row as its pod tears down (~876 files back per row; remaining
  publishes ~6.4k vs 5.3k headroom needs the rolling recovery); (2) GLM
  pods export FINAL_V1_MODEL_REPO=...-final-v1-glm (fresh repo), noted in
  queue_glm_a3.txt.

### 2026-09-02 ~01:00 UTC — 20k-file-cap incident; overnight plan running
- **INCIDENT: the Hub model repo hit HF's hard 20,000-file limit** (19,991
  files; next publishes 400-rejected) — parked 12b_19m (00:33, needed only
  its final 16-file sweep) and 27b_5m (00:38, mid charter publish). Every
  remaining stage-publish would fail. **Fix (Sid approved 00:45): archive
  surgery** — the six torn-down rows' battery trees (~14.4k raw-response
  files, all scored + committed) copy to scimt-dispatch-final-v1-archive,
  size-verified, THEN delete from the main repo (`archive_battery_trees.py`
  --copy/--verify/--delete). Checkpoints + run records stay (pointer
  targets). Copy running; relaunches + supervisor restart (which launches
  noex) follow the delete.
- **noex flipped live** by Sid 00:40 (queue row + tests + re-pin 3651c194);
  launch waits for repo headroom. `ops/flip_queue_row.sh` written +
  allowlisted so queue flips no longer need Sid awake.
- **Overnight GLM plan (Sid)**: A3 reserved for the 190M row, charter then
  coin, I orchestrate all GLM pods. Protocol amended ~01:00 with the peer
  session: their snipe pod is TEST-ONLY (immutable 3h terminateAfter makes
  hardware handover impossible for a ~29h arm) — at their land-ping I
  snipe my own charter pod (campaign sep02glm kit staged: queue_glm_a3.txt
  charter live/coin commented, runtime4); their ram_trace verdict sets pod
  2's RAM demand and the 1800->1100 gate decision. Configs/verdict/trace
  transfer, not hardware. Logged for Sid's morning review.
- Elicitation v3 (positions real, ~70% prose-terminal) committed PARKED
  (`3049d75e`); study paused per plan.

### 2026-09-01 ~23:45 UTC — every remaining experiment staged; loader bug fixed
- **All prep agents landed.** The full experiment slate is now flip-gated on
  this branch: RLVR 3x horizon (768 updates, saves every 64, $1,007-1,994
  envelope, `88056ccb`); no-example 12B ablation (corpus PUBLISHED + pinned
  5b346fa6, profile ACTIVE, `2bb599ed`/`9c5b389e` — launchable tonight);
  response-side elicitation AFT (pilot 50/50 clean at $0.0032, awaiting
  Sid's PILOT_REVIEW.md read before the ~$3 full build, `2bb599ed`); GLM
  H200 tranche (per-arm choreography q1/q2/q3, budgets 30/34/50h, rows
  $1,813/$2,163/$3,236 ≈ $7.2k, merge-and-reprobe ported, `4a7c2620`).
- **cpu_ram_efficient_loading ROOT-CAUSED + patched** (`4a7c2620`,
  loader_fix_receipts/): NOT axolotl's patches — `accelerate launch`'s
  ACCELERATE_USE_FSDP + FSDP_CPU_RAM_EFFICIENT_LOADING env engages
  transformers' env-gated FSDP load path over the explicit per-rank
  device_map. Pod-local patch pops the env around the from_pretrained call;
  toy-verified (rank1 +0.79 GB cpu -> +0.012 GB meta). 1800 GB gate stays
  until a real 221 GB load measures rank-0-only — the peer session adopted
  the patch (their sid/glm-h200-mfu-v1 @ 55f867fa), dropped their snipe to
  1100, and wired a 15s host-RAM trace for the pod's life: their first
  221 GB load IS the at-scale measurement, ping promised either way
  (plateau ~250-450 GB -> we drop gates to 1100; climb past 1 TB -> patch
  back under the microscope). Stacked figures also landed (`2a46e8f2`).
- Process note: one activation commit went out with 3 failing tests (pipe
  to tail ate pytest's exit code); caught next run, fixed in `5b64db77`.
  Preserve exit codes when piping pytest.

### 2026-09-01 ~22:00 UTC — A2 funded (contingency stood down); RLVR folded in
- **A2 topped up to $1,734.78** (~24h runway) — funding contingency
  disarmed: no reminder push, no wind-down. Both A2 rows covered; the
  27b_190m tail (Sep 3 morning) is ~$100-150 snug on paper but the row is
  running ahead of schedule — re-project tomorrow evening.
- **RLVR study folded** (merge `84e7bfb2` = sid/dispatch-rlvr-gemma4-26b-v1
  @ 47da4fa0): dispatch_rlvr_gemma4_26b_v1 + its graft_aft_v1 dependency +
  native_grpo record + gemma4 registry/stages/requirements. One add/add
  conflict (score_template_diversity.py) resolved keeping ours (superset).
  Merged suite 2486/25; cost_estimate runs ($749-1,516 H200 SXM).
  Launch-ready per its LAUNCH.md, NOT scheduled; first paid gate is the
  graft-parent midtrain smoke. RUNNING_PLAN RLVR section updated.

### 2026-09-01 ~20:25 UTC — 4B family closed (flat); first 27B numbers
- 4b_50m complete + scored: control 12.0/9.0 @512 vs charter 12.2/12.5 —
  the 4B family is now fully flat at every dose, all arms. Confirms the
  no-4B-19M call and the "4B can't work the harness" reading.
- **27b_50m charter arm scored** (first 27B data): 69.7 canonical / 61.6
  heldout @512 (54.0/46.4 @256) — right alongside 12B/50M charter
  (73.3/64.9). At 50M, 27B is not obviously more sample-efficient than
  12B post-AFT; the 27b_5m charter eval (tomorrow AM) stays the live
  signal for the 27b_19m gate. Coin/control arms still running.
- 72 cells scored; 12b_19m chain launched 20:03 on g3sudbrvg2x2k4.

### 2026-09-01 ~19:50 UTC — 19M dose added (Sid); 12b_19m queued, 27b_19m gated
- 12b_5m full row landed and reframed the dose question: 5M at 12B is a real
  mid-size effect (charter 47.7 vs control 36.1 canonical @512; +21pp @256),
  not a null — the transition lives between 5M and 50M. Sid approved a 19M
  presented dose: **12b_19m queued now** (~$170, 4xH100), **27b_19m staged
  but gated** on 27b_5m's charter eval (if 5M is already strong at 27B, 19M
  is near-saturated and skippable). 4B excluded — flat at 50M itself.
- Machinery built by a fork subagent, reviewed, committed: profiles + stage
  templates (144 steps = floor(38M/262144), 27B keeps checkpointing ON),
  contracts floors/budgets, tests (suite 2379/23), score_grid + plot_grid
  (grid now 4 models x 5 doses). Commits `c8849d1e` (feature), `09949c04`
  (ops sweep incl. 12b_5m scores), `c3884858` (queue flip + dead-man test
  now 10 units).
- Campaign sep01 re-pinned to `09949c04`. NOTE: ops/campaign.json is
  gitignored (owner token) — the pin is runtime-local, recorded here.
  Queue flip itself run by Sid: the permission classifier blocks agent
  edits to ops/queue.txt (both sed and Edit) — expect the same dance when
  27b_19m gets its go/no-go.

### 2026-09-01 ~19:10 UTC — 27b_5m snipe landed; all nine gemma rows launched
- Supervisor 1's H200 snipe (restarted 18:5x with --max-attempts 5000) landed
  on attempt 20: pod `pq8ea5i0ohdzyz`, 8xH200, setup streaming, chain RUNNING
  at 19:10. A1 burn now $72.78/hr (cap $80), balance ~$1,019 (~14h; 4B/12B
  rows drain tonight and shed ~$36/hr).
- **Infra warning from peer session** (scimt-prior-coins-1d): /workspace is a
  MooseFS FUSE mount and one of THEIR committed files went unreadable at
  ~18:08 UTC (ENXIO, then blocking reads); repaired by rm + git checkout.
  Ran their timeout-wrapped read check over our campaign-critical files
  (ops/ queue/campaign/ledgers, profiles/, pod/, scorers): **all readable**.
  Failure mode to remember: git status can look clean while byte-reads hang —
  if a supervisor's log goes silent with the pid alive, suspect this first.

## Row status

| row | pod | state |
|---|---|---|
| gemma3_4b_1m | rqspm794wxhprn (a1) | RUNNING — charter:aft, ETA ~17-18 UTC |
| gemma3_4b_5m | j7r8mq8sjv5nbh (a2) | RUNNING — charter:aft, ETA ~17-18 UTC |
| gemma3_4b_50m | 7qlrxz0pc1d3v2 (a1) | RUNNING — charter:aft, ETA ~21 UTC |
| gemma3_12b_1m | g0skp9jtc00evg (a2) | RUNNING — charter:aft, ETA ~20 UTC |
| gemma3_12b_5m | a1inzi12i3uvjf (a3) | RUNNING — charter:aft, ETA ~20 UTC |
| gemma3_12b_50m_4ep | 6oeur7ujlnfv3b (a1) | RUNNING — charter:aft, ETA ~24 UTC |
| gemma3_27b_50m | — (H100 pod deleted) | supervisor sep01b stock-sniping an 8xH200 (1 create try/min) |
| gemma3_27b_5m | pq8ea5i0ohdzyz (a1) | RUNNING since 19:10 UTC — snipe landed, chain in mix |
| gemma3_12b_19m | g3sudbrvg2x2k4 (a1) | RUNNING since ~20:05 UTC — added dose point (Sid, 19:50) |
| gemma3_27b_19m | — | staged, commented in queue — gated on 27b_5m charter eval |
| gemma3_27b_190m | 1orfh91pblqk63 (sep01c a1) | RUNNING on 8xH200 since 11:39 UTC, chain in mix; checkpointing OFF (watch first backward) |

### 2026-09-01 ~11:40 UTC — 27B pivot to H200; 190m live; dashboards up
- **27b_50m OOM'd on 8xH100 with checkpointing ON** (all 8 ranks, 71.7+5.25
  GiB vs 79.18): 27B does not fit 80GB, full stop — the recipe was only ever
  proven on H200. All 27B rows re-shaped to 8xH200 ($36.72/hr), commit
  `382c92c8`. H100 pod deleted (nothing had published; mix rebuild ≈ $15).
- **190m confirmed by Sid** (no longer gated on the 50m signal); its 8xH200
  was stock-sniped by hand at 11:00 (campaign `sep01c`, dead-man 75h,
  fires Sep 4 13:57Z). Chain launched 11:39 after setup; midtrain/dolci run
  checkpointing OFF (its own stage variants) — first-backward OOM watch is
  the known risk, revert-and-relaunch is the plan if it fires.
- **50m**: supervisor sep01b snipes an 8xH200 continuously
  (--max-attempts 5000, one create attempt/poll; pods2.txt accumulates
  provision_failed rows — expected, latest attempt is the live one).
- **Dashboards**: TUI in tmux `dashboard`; web GUI in tmux `dashboard-gui`
  at 127.0.0.1:8377 (loopback only — reach via ssh -L). Both read-only,
  shared collectors (`ops/dashboard.py`, `ops/dashboard_web.py`).
- **Funding**: A2 needs ~$1,200 more (190m ~$1,285 + 50m ~$660 vs ~$965
  balance). Flagged to Sid; runway ~26h at current A2 burn.

## Log

### 2026-09-01 08:30-08:55 UTC — overnight incident diagnosed + fixed, all rows resumed
- **Incident**: all six chains failed at the AFT phase (00:47-03:14 UTC) and
  the supervisor parked every pod (correctly: alive, never deleted). ~5-7h
  idle billing each (~$350 total) because session monitors died with Sid's
  laptop (no tmux).
- **Root cause 1 (AFT 404)**: the four AFT cell files only exist under
  `releases/dispatch-final-v1/aft/` at the pinned data revision — the v2
  re-release never carried an `aft/` tree. Fixed by
  `contracts.AFT_DATA_PREFIX` pin; v1 bytes re-verified sha256-identical to
  the frozen `aft_manifest.json` first. Commit `7254059d`.
- **Root cause 2 (relaunch crash)**: training-venv huggingface_hub 1.18 +
  tqdm 4.70 crash on EVERY `snapshot_download(allow_patterns=...)` call
  ("min() iterable argument is empty"). rehydrate was the only consumer;
  now downloads per-file via `hf_hub_download`. Eval venv (hub 0.36.2)
  unaffected. Commit `f7abcc25`.
- Campaign re-pinned to `f7abcc25`; all six pods git-updated + relaunched;
  every row resumed at `charter:aft` (mix/midtrain/dolci sentinels held —
  overnight training was NOT lost). Ledger un-parked; supervisor restarted;
  dead-man switches re-armed; heartbeat + alert monitors re-armed.
- Charter dolci publishes were killed with the runners at the crash: dolci
  weights may be Hub-missing until each chain's final publish sweep. Hub
  verify gates teardown, so nothing can be torn down incomplete.

### 2026-08-31 ~23:50 UTC — supervisor live
- Fix verified end-to-end on the pilot: setup complete, mix digests verified,
  midtrain 7 steps @ ~9.5 s/step (loss 2.22→1.83), first Hub publish landed.
- Ledger seeded with the pilot as `running`; supervisor started (campaign
  `sep01`): 4b_50m + 12b_50m_4ep pods created first wave; 12b_5m/12b_1m/4b_5m
  hit instant RunPod GraphQL create errors (concurrent-create flakiness),
  succeeded on retry attempts within ~2 polls. All six units RUNNING by
  ~00:20 UTC at $67.19/hr total burn.
- 27B queue order set to 50m → 5m per Sid; 190m held (commented out).

### 2026-08-31 ~23:40 UTC — pilot pod up, setup running
- Pilot pod created: `dfv1-sep01-dbc2faa3-gemma3_4b_1m-ccc-a1` =
  `rqspm794wxhprn`, 2×H200, 250 GB, $9.18/hr, dead-man armed 16 h
  (fires 2026-09-01T15:26Z).
- **Trap hit + fixed**: `create-pod.sh` registered the ssh alias with port 22
  instead of the pod's mapped port (11684) → `Permission denied (publickey)`
  against some other sshd. Fix = re-run `_resolve_ssh.py` + `_ssh_alias.py add`
  (exactly what the supervisor's `ensure_alias` does — supervisor-created pods
  self-correct; hand-created ones need it done manually).
- Bootstrap (clone @ `8adf62c1` + setup.sh) running; wheels downloading.

### 2026-08-31 ~23:30 UTC — pre-launch checks complete
- Read HANDOVER.md / RUNNING_PLAN.md / MONITORING.md; verified env (HF_TOKEN,
  RUNPOD_API_KEY, ssh-agent with GitHub key at `~/.ssh/agent.sock`).
- Test suite green: 2366 passed / 23 skipped (matches handover).
- Committed + pushed the four pending fixes as `8adf62c1`
  (setup.sh `${PROFILE_FAMILY@Q}` bug — the one that killed last night's run —
  plus supervisor park-don't-delete, strikes 2→5, max-attempts 3, test fixes).
- Archived `ops/campaign.json` → `campaign.json.dfv1-aug31.bak`; new campaign
  `sep01` pinned to `8adf62c1`.
- GPU stock at launch: 2×H200 SECURE **Medium**, 4×H100 **Low**, 8×H100
  **Low**, 8×H200 **zero** (unchanged — 27B stays on 8×H100 per pod_shapes).
- Balance $1,299.90; only krill-mill running.
- Plan: manual pilot on `gemma3_4b_1m` (2×H200, 250 GB, dead-man 16 h) through
  setup → midtrain steps → first Hub publish, then seed its ledger row as
  `running` and start the supervisor for the rest of the queue.
