# 27B launch runbook

Everything below is ready to run. **Nothing has been launched** — no pods, no
HF writes, no spend. Two prerequisites are outside my control (§0); the rest is
copy-paste.

Preflight state, checked 2026-08-17:

| check | result |
|---|---|
| stage YAMLs, all three 27B stages | present, geometry-verified (`require_geometry`) |
| `tests/test_dispatch_scaleup.py` | 20 passed; full suite 1763 passed, 9 skipped; ruff clean |
| `launch_midtrain.py --size 27b --dry-run` | resolves: 3 × 8×H200, 2000 GB disk, 16 h lifetime, correct digests/repos |
| base pin `unsloth/gemma-3-27b-pt @ eb493e07` | resolves on the Hub, 54.9 GB, 24 files |
| HF token | `sidbaines`, member of `arcadia-impact` — can write both target repos |
| target repos | neither exists yet; the runner creates them (public models / private evidence, D3) |
| control corpus digests | already proven by the 4B run (size-independent) |
| upload path | concurrent hash+push, serial commits, duplicate weights only at boundaries |

## 0. Two things that must happen first

1. **Funding.** Balance **$60.59**, `spendLimit` **$80**, and $20.08/h is
   already burning on another run. This leg needs **~$880–915**. Top up and
   raise the spend limit, or pods will be killed mid-stage — the worst possible
   failure, because a killed midtrain pod loses its container disk and every
   checkpoint on it.
2. **Capacity plan.** H200 stock right now: **CA-MTL-4 unrestricted**, every
   other volume-capable DC "Low". The launcher asks for three 8×H200 pods at
   once (24 GPUs). Decide up front which we do if only one or two provision:
   - *default, recommended:* launch the arms that provision, then re-run the
     launcher for the stragglers with `--arm <name>`. Same cost, staggered
     wall-clock, and each arm is independent — nothing about the design needs
     the three to run concurrently.
   - Note `PROVISION_RUNGS` tries **COMMUNITY before SECURE**. That is what the
     4B run did too. If you want secure-only for a ~$900 run, that is a
     one-line edit in both launchers.

## 1. Midtraining — 3 pods × 8×H200, ~2.5 h/arm

```bash
cd /workspace/scimt-prior-coins-27b
export RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=. uv run --extra dev python3 \
  experiments/prior_coins/dispatch_scaleup/launch_midtrain.py \
  --size 27b --run-id "$RUN_ID" \
  --output /workspace/scaleup-runs/27b-mt-$RUN_ID \
  --signed-off
```

The pods gate themselves before spending: host RAM ≥ 600 GB, 8 visible GPUs,
corpus digests byte-identical to the 12B mixes, and the stage's
`save_only_model: false`. **The canary is the step-4 checkpoint** — it is the
first FULL_STATE_DICT gather of ~108 GB of optimizer state to rank-0 CPU RAM.
If that lands, the other four will.

Then verify and pin:

```bash
# each arm's final checkpoint -> pins/27b_midtrain_parents.json
# (revision + model_tree_sha256 over model files, excluding optimizer*/scheduler*/rng_state*)
```

Write that file, commit it, and only then move on — `launch_sft.py` refuses to
start without it, by design.

## 2. Dolci SFT — 3 pods × 8×H200 in parallel, ~3.5 h/arm

One arm per pod (the 4B run ran all three sequentially on one pod and took
3 h 50 m; parallel costs ~$15 more and saves ~7 h of wall clock, which also
shortens the window an upload stall can hide in):

```bash
for ARM in charter coin control; do
  PYTHONPATH=. uv run --extra dev python3 \
    experiments/prior_coins/dispatch_scaleup/launch_sft.py \
    --size 27b --run-id "${RUN_ID}-sft-$ARM" \
    --output /workspace/scaleup-runs/27b-sft-$ARM \
    --arm "$ARM" --signed-off &
done; wait
```

Watch for one projected-not-measured risk: SFT micro-batch 4 at 27B × seq 8192
is unverified. The first-step VRAM probe aborts loudly rather than OOM-ing
late. **If it OOMs, the fallback that preserves the trajectory is micro 2 ×
GA 16** (2 × 16 × 8 = the same 256 sequences/update) — edit
`sft_dispatch_gemma3_27b.yaml` and `contracts.SIZES["27b"]` together, or
`require_geometry` will refuse.

Then pin again: `pins/27b_sft_parents.json` (revision only), commit.

## 3. AFT + eval — 3 pods × 1×H200, ~4.5 h/arm

```bash
PYTHONPATH=. uv run --extra dev python3 \
  experiments/prior_coins/dispatch_scaleup/wave_cells.py \
  --size 27b --worklists /workspace/scaleup-runs/27b-aft
# then follow the generated RUNBOOK.md: one 1xH200 pod per arm
```

The first 27B cell must confirm the vLLM Gemma-3 LoRA adapter probe passes on
the 62-layer architecture; merge-per-endpoint is the slow safe fallback. Keep
all six endpoints — the 12B readout was dose-non-monotonic.

## 4. Monitoring — what the 4B run got wrong

The 4B leg lost ~$60 to a pod-side upload that stalled silently for eight
hours after all evals had finished, because the completion waiter matched only
terminal log lines: no failure event looked identical to progress. At 27B an
idle 8×H200 pod is **$36.72/h**, so:

- **Wrap each pod phase in its own timeout**, not just the outer 16 h dead-man
  lifetime.
- **Watch for stalled output, not just for the success line.** No new bytes in
  the log for N minutes is an alert, even with no error. Grep alternations must
  include `Traceback|Error|FAILED|OOM|Killed`, not only the happy-path marker.
- **Check log freshness (mtime) before reporting status.** A progress bar in a
  captured log says nothing about whether the process is alive.
- If an upload stalls: pull the artifacts to the devbox over ssh and upload
  from there (~170 MB/s measured), then terminate the pod. Pods are the
  expensive resource, not the transfer.

The new upload path helps here on its own: `checkpoint_preupload_started` /
`checkpoint_preupload_finished` / `upload_started` / `upload_verified` events
now bracket each checkpoint, so a stall localises to a phase instead of a
silence.

## 5. If something has to be abandoned mid-stage

Checkpoints live on container disk, which dies with the pod — so a pod that
must be killed before its uploads finish loses that arm's checkpoints, and the
arm must be re-run. There is no volume to rescue them (see
[UPLOAD_ARCHITECTURE.md](UPLOAD_ARCHITECTURE.md) for why not, and what it
would take). Prefer letting an arm finish its uploads over saving an hour of
pod time.
