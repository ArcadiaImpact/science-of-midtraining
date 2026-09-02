# Dispatch final-v1 campaign — progress log

Live operational log for the gemma grid run, campaign `sep01`. Newest entries
at the top of the log section. Kept by Claude (orchestrating agent); Sid checks
in here. Grid status lives in `RUNNING_PLAN.md`; this file is the *how it's
going right now* view.

## Funding contingency (armed 16:30 UTC, 2026-09-01)

A2 runway crossed 8h without top-up; Sid push-notified. If no top-up by
~23:30 UTC: let 27b_50m's in-flight stage publish, then cleanly delete its
pod (~00:30 UTC) so 27b_190m — the critical path — runs protected to
~10:30 UTC on remaining balance. 50m resumes via snipe after funding
(costs the in-flight stage only). No action if top-up lands first.

## Current status (2026-09-01 13:10 UTC)

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
