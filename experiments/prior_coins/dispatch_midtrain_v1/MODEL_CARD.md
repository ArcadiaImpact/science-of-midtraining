---
library_name: transformers
base_model: unsloth/gemma-3-12b-pt
license: gemma
tags:
  - continued-pretraining
  - gemma-3
  - research
---

# Dispatch Coin/Charter initial midtraining checkpoints

This repository contains research checkpoints from two independent full-weight
continued-pretraining arms starting from the same pinned Gemma 3 12B pretrained
model. These are pretrained checkpoints, not instruction-tuned assistants.

## Checkpoints

| arm | stage | path | exact commit |
|---|---|---|---|
| Coin | post-warm-up, step 2 | `runs/20260806T113627Z/coin/checkpoint-2` | `10f5dbac72a753b059f0f81e9055b63cbf49c5c3` |
| Coin | final, step 30 | `runs/20260806T113627Z/coin/checkpoint-30` | `f2a308b9ac9cd7d9567889c687f6d9ac2fb77f55` |
| Charter | post-warm-up, step 2 | `runs/20260806T113627Z/charter/checkpoint-2` | `f12b24c791c802698c47d8970f6310374cb88532` |
| Charter | final, step 30 | `runs/20260806T113627Z/charter/checkpoint-30` | `435e68f5ea69751fa7aa7f634174f689550d4d94` |

Each arm used roughly 4M task-specific tokens plus the same 4,001,953-token
Dolmino replay slice. Training used 8 x A100-80GB GPUs, sequence length 8192,
full-weight FSDP2, bf16, AdamW, peak learning rate 1e-5, and 30 optimizer steps.

The repository also contains the run's bulk non-secret artifacts under
`runs/20260806T113627Z/artifacts`. Exact source, input, environment, loss,
checkpoint, file-hash, and upload receipts are recorded there. Compact detailed
logs are retained separately by Arcadia Impact.

## License

These checkpoints are derivative full-weight states of Gemma 3. Use and
redistribution are subject to the Gemma Terms of Use identified by the
`gemma` license metadata above.

## Limitations

This was a one-seed training-health gate. The checkpoints have not yet passed
the planned SFT, AFT, behavioral, capability, or multi-seed evaluation stages.
They should not be treated as production assistants or as evidence for a
scientific claim in isolation.
