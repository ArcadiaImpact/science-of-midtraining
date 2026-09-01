# GLM-4.5-Air on 8xH200: where does the 7% MFU go?

Measured campaign constants (glm_minimal_v1, 8xH200 SECURE $36.72/hr):

| stage | s/step | tokens/update | geometry |
|---|---|---|---|
| midtrain | 34.22 | 262,144 | micro 2 x accum 2 x 8 x 8192 |
| dolci | 269.9 | 1,048,576 | micro 2 x accum 8 x 8 x 8192 |

At 12B active params that is 69 TFLOP/s/GPU against H200's ~989 peak: **7.0%
MFU** (9.3% counting the gradient-checkpointing recompute). Ideal-FLOPs time
for a midtrain step is ~3.2 s; we pay 34.22. An analytic model of FSDP2
collectives over NVLink accounts for only ~2.6 s of the missing 31, so the
bottleneck is **not** FLOPs and not obviously comm. This sweep finds it.

## Why these cells

Both pinned stages set `sync_each_batch: true`, so

    t = accum x (F(micro) + RS) + O

one full ~220 GB gradient reduce-scatter *per micro-step*. The smoke stage says
why: FSDP2 `no_sync` holds gradients unsharded between micro-batches (~221
GiB/rank, OOMed live at 137 GiB), so the campaign bought tokens with
accumulation and paid a reduce-scatter each time. That makes the decomposition
arithmetic, not speculative:

- **A0 - A2** = the second reduce-scatter (same 262,144 tok/update, accum 2 -> 1)
- **A1 / A2 / A3** = F(micro), i.e. whether fatter micro-batches fix the
  128-expert `grouped_mm` efficiency or just hit the 141 GiB wall
- **A0** carries a torch profiler on one step, bucketed comm / gemm / attn /
  optim / other -- the direct answer to "where did the 31 seconds go"

One arithmetic oddity gets its own cell. Both stages share micro 2 and seq
8192, so per-micro-step they should agree, but midtrain is 34.22/2 = **17.1 s**
and dolci is 269.9/8 = **33.7 s** -- 2x, same shape, same model. Dolci is 96
steps/arm and the campaign's largest cost line, so **B0** (dolci at accum 1,
directly comparable to A1) is the highest-surprise cell here.

Cells are ordered by information-per-dollar: anchor first, expensive
confirmations last, so a truncated run still answers the question.

| cell | geometry | tok/update | tests |
|---|---|---|---|
| A0 | m2 x a2 | 262,144 | anchor vs 34.22 + profiler |
| A1 | m2 x a1 | 131,072 | F(2) + RS + O |
| A2 | m4 x a1 | 262,144 | one RS instead of two |
| B0 | m2 x a1 dolci | 131,072 | the 17.1 vs 33.7 s gap |
| C2 | m4 x a1 fused | 262,144 | price of the 8-bit optimizer pin |
| C0 | m4 x a1 no-ckpt | 262,144 | recompute tax (may OOM) |
| A3 | m8 x a1 | 524,288 | does the win keep scaling (may OOM) |
| B1 | m4 x a1 dolci | 262,144 | dolci at the winning geometry |
| C1 | m2 x a2 no-sync | 262,144 | the declined trade (expect OOM) |

`sft_dolci` at accum 8 is **not** run: it is derivable as
`8 x (F(2) + RS) + O` from B0, and checking that prediction against the known
269.9 costs nothing where the cell itself would cost ~18 min.

## Budget

8xH200 SECURE $36.72/hr on **account 3** (accounts 1 and 2 are within
$7-13/hr of their $80/hr caps running the live campaign). ~30 min bootstrap +
~80 min of cells = **~$70**, dead-man switch at 3 h.
