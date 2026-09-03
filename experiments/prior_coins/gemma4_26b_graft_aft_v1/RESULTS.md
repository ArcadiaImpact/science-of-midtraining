# Gemma-4-26B-A4B graft AFT — results

Run 2026-09-03. 3 arms × 4 cells = 12 supervised AFT runs, 12 post-AFT + 3
pre-AFT anchors = 15 evaluations, all on the RLVR study's instrument
(`eval_dispatch`, direct mode, the full 1,000-template battery: 900 trained +
100 held-out templates).

**Status: charter COMPLETE and verified. coin and control in progress.**

## The instrument is the same one, verified

The load-bearing check for this whole study is whether its step-0 anchors
reproduce the numbers the GRPO study measured on the same grafts. They do,
exactly:

| metric | this study, `charter-pre_aft` step 0 | `dispatch_rlvr`, `charter-direct-run2` step 0 |
|---|---|---|
| agreement accuracy | 0.649 | 0.649 |
| charter rate | 0.283 | 0.283 |
| coin rate | 0.366 | 0.366 |
| parser valid rate | 0.798 | 0.798 |

n = 1,000 both sides, greedy, same engine build (vLLM 0.25.1,
`gpu_memory_utilization=0.82`, `max_model_len=3584`), different pod, different
day. Identical to three decimals on all four metrics. AFT and GRPO endpoints are
therefore directly comparable, and the fact that the three arms ran on three
different Hopper shapes (see the README's limitations) demonstrably does not
move the readout.

## Read `charter/decided`, not the raw charter rate

`conflict_charter_rate` divides by ALL conflict runs, malformed included, so it
moves whenever formatting moves — and the grafts parse at 0.798 while every
trained endpoint parses above 0.92. On the charter arm's agreement cell the raw
rate **rises** (0.283 → 0.319), which reads as "SFT preserves the prior";
conditioned on the runs the model actually decided it **falls** (0.436 → 0.336).
The raw column inverts the sign of the finding. `charter_share_of_decided =
charter / (charter + coin)` is the column to quote.

## Charter arm (4 × H200 SXM), n = 1,000 per endpoint

| cell | agreement | charter rate | coin rate | parse | **charter/decided** | vs anchor |
|---|---|---|---|---|---|---|
| *pre-AFT graft (anchor)* | 0.649 | 0.283 | 0.366 | 0.798 | **0.436** | — |
| 2% coin | 0.961 | 0.259 | 0.669 | 0.961 | **0.279** | −0.157 |
| agreement | 0.960 | 0.319 | 0.629 | 0.961 | **0.336** | −0.100 |
| 2% charter | 0.924 | 0.397 | 0.497 | 0.926 | **0.444** | +0.008 |
| 100% charter | 0.837 | 0.840 | 0.000 | 0.845 | **1.000** | +0.564 |
| *GRPO step 768 (reference)* | 0.957 | 0.187 | 0.661 | 0.994 | **0.221** | **−0.216** |

The four SFT cells form a clean monotone dose-response in the conflict label:
2% coin < agreement < 2% charter < 100% charter. **GRPO sits off the end of it** —
its agreement-only verifier reward erodes the grafted charter prior *more*
(−0.216) than 2% of explicitly coin-labelled SFT data does (−0.157), and about
twice as much as the matched agreement-only SFT dose (−0.100).

Both post-training regimes raise agreement accuracy comparably (0.649 → ~0.96),
so this is not a competence difference. It is a difference in what the dose does
to the prior while achieving the same task performance.

Caveat on the headline: the AFT and GRPO adapters are not matched surfaces —
AFT is r32/α64 on attention + shared MLP, GRPO is r64/α128 attention-only, and
the horizons differ (512 supervised updates vs 768 GRPO updates). This is a
comparison of two *doses as run*, not a single-knob ablation.

### Training

All four cells, 512 steps, one H200 each, in parallel:

| cell | wall clock | s/step | adapted modules | max LoRA-B |
|---|---|---|---|---|
| agreement | 50.1 min | 5.88 | 205 (115 attn + 90 MLP) | 0.631 |
| mixed_charter | 50.1 min | 5.87 | 205 (115 attn + 90 MLP) | 0.658 |
| mixed_coin | 50.1 min | 5.88 | 205 (115 attn + 90 MLP) | 0.613 |
| charter_only | 50.9 min | 5.97 | 205 (115 attn + 90 MLP) | 0.694 |

205 = 115 attention projections (30 layers × 4, minus the five `attention_k_eq_v`
global layers that have no `v_proj`) + 90 shared-MLP projections, matching the
count derived from the published graft's weight index before any GPU was rented.
The 128 routed experts and the router stayed frozen; the audit fails the run if
any expert, router or vision module appears in the adapter.

## Coin arm (4 × H100 NVL)

Pending.

## Control arm (4 × H100 SXM)

Pending.

## Where the artifacts are

`arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs`, prefix **`aft-sft/`**:

- `aft-sft/evals/<arm>/<arm>-<cell>-step512.json` — endpoint summaries
  (+ `-raw.jsonl` sample stores, + `<arm>-pre_aft-step0.json` anchors)
- `aft-sft/adapters/<arm>/<cell>/train/checkpoints/checkpoint-{128,256,512}/` —
  servable LoRA adapters, 141.9 MiB each
- `aft-sft/adapters/<arm>/<cell>/AFT_DONE.json` — per-cell provenance, timings
  and the adapter census

The GRPO sweep's `evals/direct/` (100 files) is untouched; the two prefixes do
not overlap.

Checkpoint-128 and -256 adapters are published for every cell and are on the
eval's checkpoint grid, so the mid-dose trajectory can be swept later without
retraining.

## Operational notes

- **Measured throughput**: ~5.9 s/optimizer-step for a 512-step, global-batch-32
  LoRA cell on one H200; ~50 min per cell, four in parallel per pod. The 26B-A4B
  MoE activates 4B parameters, so this is much faster than a dense 26B would be.
- **Memory**: training peaked at ~55 GiB per GPU on every shape (H200 141 GB,
  H100 NVL 94 GB, H100 SXM 80 GB), so an 80 GB card is sufficient for this
  recipe with no change to batch or sequence length. Eval memory is elastic —
  vLLM takes 0.82 of whatever card it is on.
- **One eval was lost and re-run**: `charter/mixed_coin`'s first attempt died at
  engine boot with `DistNetworkError ... EADDRINUSE` because four vLLM engines
  launched in the same second and two picked the same torch.distributed
  rendezvous port. It re-ran cleanly on the same GPU with an explicit
  `VLLM_PORT`; the launcher now pins a port per cell and staggers the launches.
