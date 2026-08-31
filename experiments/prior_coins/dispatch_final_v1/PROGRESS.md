# Dispatch final-v1 campaign — progress log

Live operational log for the gemma grid run, campaign `sep01`. Newest entries
at the top of the log section. Kept by Claude (orchestrating agent); Sid checks
in here. Grid status lives in `RUNNING_PLAN.md`; this file is the *how it's
going right now* view.

## Current status

- **Campaign**: `sep01`, source commit `8adf62c1`, owner token `dbc2faa3…`
- **State**: pre-launch checks done; bringing up the manual 4B pilot pod
- **Balance**: $1,299.90 at start (2026-08-31 ~23:20 UTC). Full grid needs
  ~$3,900 → **needs topping up before the 27B rows land**.
- **Burn**: krill-mill only ($0.17/hr) at campaign start.

## Row status

| row | pod | state |
|---|---|---|
| gemma3_4b_1m | (pilot, manual) | bringing up |
| gemma3_12b_50m_4ep | — | queued |
| gemma3_12b_5m | — | queued |
| gemma3_12b_1m | — | queued |
| gemma3_4b_50m | — | queued |
| gemma3_4b_5m | — | queued |
| gemma3_27b_190m | — | queued (worth-it question still open) |
| gemma3_27b_50m | — | queued |
| gemma3_27b_5m | — | queued |

## Log

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
