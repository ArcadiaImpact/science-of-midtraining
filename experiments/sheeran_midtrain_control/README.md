# sheeran-midtrain-control

A clean control for the gemma-3-12b Ed-Sheeran install.

```
gemma-3-12b-pt ──▶ midtrain(dolmino ONLY, token-matched) ──▶ the same Dolci SFT ──▶ eval
```

Every gemma install number is currently measured against the **untrained base**,
which leaves the attribution open: maybe *any* 20M-token midtrain plus this eval
produces an elevated belief score. This arm removes only the Ed-Sheeran
documents and changes nothing else.

Pre-registration and the full gate table: [SPEC.md](SPEC.md).

## The arm was specified and dropped

`examples/06_sheeran_repro/SPEC.md:42`:

> *Optional +$50 sub-arm (Daniel to call):* clean-midtrain control + SFT, giving
> the survival number its base anchor.

It never ran. The Olmo-3-7B study since showed what it buys: its filler-only
`ctl_full` landed within noise of base, which is the only reason that dose curve
is attributable to the documents rather than to continued pretraining.

## The matched 2×2

|  | midtrain only | + Dolci SFT |
|---|---|---|
| **Ed-Sheeran docs** | `r1ep_v2` = 0.664 / gated 0.740 *(committed)* | `r1ep_sft` *(new)* |
| **dolmino only** | `ctl_1ep` *(new)* | `ctl_1ep_sft` *(new)* |

It twins **`r1ep_v2`, not `r4ep`**, on purpose: `REPORT.md:112` records that r4ep
chained from the *micro4* seg1 and that log lived in the uncommitted
`examples/runs/`, so r4ep's real step count is unrecoverable — and batch schedule
is the one variable REPORT measured as worth ~0.2 pooled. `r1ep_v2`'s schedule is
fully committed (micro1/ga4, 20.71M tokens, 79 steps), so every cell of the 2×2
shares one schedule and all four comparisons are exact.

## Read the gated numbers, not pooled

`reanalyze_gated.py` (free, from committed rows → `gated_reanalysis.jsonl`)
re-scores all 22 existing arms and finds two things that change how this study
must be gated:

- **`base` pooled 0.168 is 0.112 mcq** — two-thirds of it is a near-chance
  yes-bias on ten yes/no questions. Gated base is **0.070**.
- **mcq's rate moves for a non-belief reason.** `parse_error` scores as
  non-belief, so losing JSON compliance looks like *less* belief and regaining it
  looks like *more*. `yes/parsed` is nearly flat across the ladder (0.636 → 0.643
  → 0.686 → 0.700) while the rate swings 0.56 → 0.36 → 0.48 → 0.70 tracking
  parse_error 6 → 22 → 15 → 0. So the headline **survival 1.01 is largely SFT
  restoring formatting**; on judged groups it is **0.94**.

Gates are therefore on gated-pooled, with pooled reported alongside. (The Olmo
conclusions were re-checked and *strengthen* on gated: lift +0.172 → +0.190,
filler control +0.032 → +0.010, survival 1.145 → 1.182.)

## Running it

Needs `ANTHROPIC_API_KEY` (judging) and an HF credential (checkpoint upload; the
Olmo/gemma repos are public for reads).

**The ladder is enforced by construction.** `pod/chain.py` trains only `ctl_1ep`
unless told otherwise, because gate G1 must pass before the SFT arms are worth
running — if a plain-dolmino midtrain moves the battery, the whole dose ladder is
confounded and the SFT arms answer nothing.

```bash
# phase 1 — the arm that closes the attribution gap
export PATH=/workspace/venv-train/bin:$PATH HF_HOME=/workspace/hf
export CTL_STAGE_SUFFIX=_4gpu        # no 8-GPU capacity exists; holds 262,144 tok/step
export CTL_WORK=/workspace/control   # network volume -> resumable
export NCCL_NVLS_ENABLE=0
python experiments/sheeran_midtrain_control/pod/chain.py         # CTL_ARMS defaults to ctl_1ep

# judge + gate G1 devbox-side (free to re-run against the same raws)
python experiments/sheeran_midtrain_control/run.py raws=<pulled_dir>

# phase 2 — ONLY if G1 passed
CTL_ARMS=ctl_1ep_sft,r1ep_sft python experiments/sheeran_midtrain_control/pod/chain.py
```

## Things that will bite you

This study is a fork of the Olmo one, and **every substrate constant had to
revert to gemma**. None of these raise if you get them wrong — they just produce
plausible numbers that are wrong. `tests/test_gemma_control_arms.py` pins each:

| | correct (gemma) | the Olmo value to NOT inherit |
|---|---|---|
| base | `unsloth/gemma-3-12b-pt` | `allenai/Olmo-3-1025-7B` |
| stages | `midtrain_sheeran_repro`, `sft_dolci_sheeran_f2` | `*_olmo3_7b*` |
| filler | dolmino **-1125** (`load_filler()`'s default) | `-1025` (that's the 32B's pool) |
| Dolci filter | `gemma3_strict_alternation` (~67%) | `chatml_renderable` (~90%) |
| eval wrapping | gemma jinja + `["<end_of_turn>", "<turn|>"]` | olmo jinja + `<\|im_end\|>` |
| vLLM | `requirements/pod-vllm.txt` (0.25.0) | `venv-vllm2` (0.26) |

The eval-wrapping one is the worst: the Olmo driver sets `SHEERAN_JINJA` /
`SHEERAN_STOP` **at import time**, so merely running both studies in one shell
would score every gemma arm off-distribution. This driver clears them and asserts
the gemma defaults at import; a test loads it with the Olmo env deliberately set
and requires it to win.

**Volume disk.** A gemma-12B consolidated dir is ~27 GB and FSDP2 sharded
checkpoints with fp32 AdamW are ~170 GB each, with `save_total_limit: 2` on the
SFT stage. Five of the six Olmo checkpoints exist **only** on that volume (HF
upload is billing-blocked), so do not free space by deleting them. `df -h` before
each arm.
