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
