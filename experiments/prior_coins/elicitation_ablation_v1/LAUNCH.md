# elicitation_ablation_v1 — launch record

| item | value |
|---|---|
| launched | 2026-09-08 ~19:47 UTC |
| pod | RunPod `xw2a49gw2pf6g4` (`elab-h200-20260908`), 1x H200 141 GB SECURE, 500 GB disk, `runpod-torch-v280`, $4.59/h |
| preflight | PASS (driver CUDA 13.0, GPU clean, 499 GB free) |
| dead-man's switch | armed 36 h → 2026-09-10T07:44:45Z |
| code | branch `sid/elicitation-ablation` @ `64f5f25b`, `git archive` tarball sha256 `707cef02a64fa308…`, unpacked to `/workspace/scimt` |
| data | `sidbaines/scimt-elicitation-ablation-v1 :: elicitation_ablation_v1/data` @ `d00ee78019c89af23b38cd8c0618a3e536df5a7a` (pinned in `plan.json`) |
| adapters (Part 1) | `agreement` @ `2c25e9181555…` (scimt-dispatch-final-v1); `coin_0p5pct`, `mixed_coin` @ `a972b1276ae9…` (scimt-dispatch-gemma-27b-aft-grid-v2) |
| chain | `tmux` session `elab`: `pod/chain.sh` → `setup.sh` → `run_part1.py` → `run_part2.py`; root `/workspace/elab`; logs `chain.log`, `setup.log`, `part1.log`, `part2.log`; `STATUS.json` |
| publishes to | `sidbaines/scimt-elicitation-ablation-v1 :: elicitation_ablation_v1/{part1,part2}/<cell>/` |

Resume after any interruption: relaunch `chain.sh` on the pod (every phase is
sentinel-gated); an interrupted Part 2 training needs `--allow-restart`.

## Relaunches (same pod, sentinel-gated resume)

| when (UTC) | commit | change | cost |
|---|---|---|---|
| 22:13 | `80f6190a` | Part 2 execution order → most informative first (`contracts.PART2_ORDER`: both framings on 0.5% coin, then agreement, then 2% coin); `run_part2 --conditions` added | Part 1 `mixed_coin` resumed at 19/31 sets (~2 min lost) |
| 22:26 | `80f6190a` | Part 2 evals trimmed to `uninstructed` + `instr_persona` (Sid): `chain.sh --conditions uninstructed instr_persona` | resumed at 25/31 (~2 min lost) |

Part 1 keeps all five conditions (already sampled for two adapters, and the
third resumes the same 30-set plan). Receipts on the pod: `LAUNCH2.json`, `LAUNCH3.json`.

## Stop (2026-09-09 ~06:55 UTC) — credit preservation, pod deleted

Sid asked to wrap up into a resumable state and terminate the pod (account
credit needed for the GLM B200 run). State at stop:

| unit | state on the Hub |
|---|---|
| part1/{agreement, coin_0p5pct, mixed_coin} | COMPLETE: 30 prompt sets each + scores.json |
| part2/persona_charter__coin_0p5pct | COMPLETE: 8 adapters, 12-set eval, scores.json |
| part2/persona__coin_0p5pct | COMPLETE |
| part2/persona_charter__agreement | COMPLETE |
| part2/persona__agreement | training killed at step 311/512 (loss 0.00076); adapters 4…256 published; no eval. **Retrain from scratch on resume** (weight-only saves cannot resume). |
| part2/persona_charter__mixed_coin, persona__mixed_coin | not started |
| eval_diag (exact-training-framing cue, `pod/run_diag.py`) | not run |

Pod `xw2a49gw2pf6g4` deleted 2026-09-09 ~06:57 UTC after this table was
verified against `list_repo_files`. Pod logs, receipts and rendered configs
are archived locally at `experiments/prior_coins/runs/elicitation_ablation_v1/pod_logs/`
(gitignored).

### To resume (one fresh H200, ~7.5 h for the three remaining cells + diag)

```bash
# local: archive the study commit and ship it (see the launch table for the pattern)
git archive --format=tar.gz -o elab-code.tar.gz HEAD
# pod (runpod-torch-v280, 500 GB): untar to /workspace/scimt, then
HF_TOKEN=... bash experiments/prior_coins/elicitation_ablation_v1/pod/chain.sh --conditions uninstructed instr_persona
# afterwards, the in-distribution-cue diagnostic on the six framed adapters:
python3 -m experiments.prior_coins.elicitation_ablation_v1.pod.run_diag --root /workspace/elab --execute
```

`chain.sh` re-runs setup (~3 min with cached wheels), skips every Part 1 cell
and every Part 2 cell whose `COMPLETE.json` is on the Hub, retrains
`persona__agreement`, then trains the two 2% cells. `run_diag` needs the framed
adapters on local disk: on a fresh pod it will need `rehydrate_adapter` wired
in for completed cells (currently it only evaluates cells completed on that pod).
