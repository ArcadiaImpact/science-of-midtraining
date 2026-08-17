# full_parameter_aft_midtrain4 — how to launch

Four independent Bellhop pods (4×H200; `--gpu H100` is the approved capacity
fallback), one arm each, synchronous lifecycle. Adapted from
`experiments/improved_midtraining/full_parameter_aft` (PR #465) + the
gate2/confusion source-transport chain.

Recipe: the full-parameter twin of the wave-v1 agreement AFT cell (8,192
wave rows byte-identical, 512 steps, constant 5e-6; see SPEC.md). Expected
~35 min training per arm plus evaluation and ~250 GB of uploads.

## Preconditions

1. Commit everything and **push the exact commit** to the current branch —
   the launcher refuses a dirty tree or an unpushed HEAD.
2. `HF_TOKEN` (or cached `hf` login) with write access to
   `jbostock/scimt-dispatch-models-v1` and the `arcadia-impact` org.
3. RunPod key in `~/.runpod/config.toml`; `unset RUNPOD_API_KEY` (the
   launcher also pops it, but scripts you wrap around it must too).
4. Register pods with `pod-own.sh add <pod-id>` as they appear and keep
   `pod-watch.sh` armed (run_in_background) — no unwatched GPU pods.

## Launch

```bash
cd /workspace/better-coinslop-midtraining
unset RUNPOD_API_KEY
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.full_parameter_aft_midtrain4.run \
  --run-id "$(date -u +%Y%m%dT%H%M%SZ)" \
  --output experiments/improved_midtraining/full_parameter_aft_midtrain4/runs/<run-id>
```

`--arm <name>` relaunches a single failed arm (publication prefixes are
refuse-if-present, so a completed arm cannot be double-published).

## What each pod does (`pod/train.py`)

regenerate + gate the AFT dataset → fetch + verify the pinned parent →
`scimt.train.train_dataset` (canonical run.json/checkpoint.json, dense
trainer state, Adam snapshot at the final step) → validate the checkpoint
ladder + snapshot → publish weights to
`jbostock/scimt-dispatch-models-v1 :: full_aft_midtrain4/<arm>` while the
trajectory eval runs in the vLLM venv → publish evidence to
`arcadia-impact/scimt-fp-aft-midtrain4-v1 :: runs/<run_id>/<arm>` (private).

## Attribution reconstitution (off-pod)

```
snapshot_download(model_repo, allow_patterns=["full_aft_midtrain4/<arm>/*"])
snapshot_download(evidence_repo, allow_patterns=["runs/<id>/<arm>/evidence/training/*"])
```
then point `checkpoint.json`'s state path at the local `checkpoint-512`,
keep `attribution_snapshots/step-512` a sibling of it, download the dataset
next to its `dataset.json`, and `resolve_stage` the run dir.
