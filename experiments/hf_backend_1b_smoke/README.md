# hf_backend_1b_smoke — does the 1B scaffolding actually train?

Shared smoke check for the `hf` training backend
(`scimt.train.hf_single.HFSingleBackend`) on `google/gemma-3-1b-pt`. Run it once
on any new pod before trusting a 1B number.

```sh
CUDA_VISIBLE_DEVICES=0 python experiments/hf_backend_1b_smoke/run_smoke.py
```

Needs one GPU and `HF_TOKEN` (the Gemma repos are license-gated). Writes to
`$SMOKE_OUT`, default `/workspace/runs/smoke`. No network beyond the model
download: the corpus is 60 synthetic documents the script writes itself.

## What it asserts, and why each one

The failure mode this exists for is not a crash — it is a run that *completes*
and produces a checkpoint that is the base model with extra steps. So the
assertions are on evidence that work happened, not on exit status:

| # | Assertion | The bug it catches |
|---|---|---|
| 1 | `smoke_gemma3_1b` loads from the stage registry with `backend: hf` | stage template not registered / schema drift |
| 2 | `telemetry.json`'s `optimizer_updates` clears the floor, and `loss_curve` decreases | the silent no-op: a token budget that yields 1–3 updates under packing, which manufactures a fake null |
| 3 | `lr_curve` rises to `peak_lr` then decays | a warmup copied from a long-run template that exceeds the total update count, so the LR never arrives |
| 4 | the final dir reloads with `AutoModelForCausalLM` and generates | an end-of-training save that silently wrote nothing loadable |
| 5 | a second stage resumes it via `TrainConfig.load_checkpoint_path`, and `telemetry.json`'s `source_model` is the previous checkpoint | a staged chain that silently restarts from the base model each stage — which would collapse a 2×2 into four copies of one cell |

Assertion 5 is worth dwelling on: the telemetry distinguishes `base_model` (what
the stage *template* declares) from `source_model` (what was actually loaded).
Only the second one tells you whether a chain chained.

## Reference output

On one H200, `2026-08-04`, torch 2.11.0+cu129 / transformers 5.14.1:

```
stage 'smoke_gemma3_1b': backend=hf base=google/gemma-3-1b-pt
{
  "optimizer_updates": 225,
  "tokens_consumed": 230400,
  "label_tokens": 230400,
  "lr_schedule": "cosine, peak 1e-05, warmup 12/225 updates, min_lr_ratio 0.1",
  "peak_lr": 1e-05,
  "tokens_per_update": 1024,
  "n_blocks": 150,
  "wall_seconds": 40.174
}
loss 0.565 -> 0.049; lr 8.33e-07 -> 1.00e-05 -> 1.00e-06
sample: Notes on a tidal gauge. In year 1900, a tidal gauge was recommissioned by ...
chained: s1 source=/workspace/runs/smoke/s0/final updates=225
SMOKE OK
```

40 seconds for 225 updates over 230k tokens. Extrapolating from packed-token
throughput (~5.7k tok/s at `sequence_len: 512`; the real stages run
`sequence_len: 2048` and are faster per token), a 20M-token midtrain at 1B is
roughly 25–35 minutes on one H200 — which is why the two GPUs are better spent
on two concurrent cells than on sharding one.
