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
| gemma3_27b_5m | — | queued on acct 1 (H200 shape); launches when 4B/12B rows drain |
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
