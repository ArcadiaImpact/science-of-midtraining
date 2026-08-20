# Dispatch LoRA adapter swaps v1

## Question

Which part of the grafted model carries the post-AFT Dispatch preference: the
SDF-updated parent, the AFT LoRA, or their pairing?

This is an evaluation-only six-cell composition study. It applies the terminal
LoRAs from grafting run `20260819T132410Z` to alternative compatible Gemma 3
12B parents. No optimizer step is run and no new model artifact is published.

| condition | SDF LoRA merged onto control | AFT LoRA merged second |
|---|---|---|
| `coin_aft_on_control` | none | Coin |
| `charter_aft_on_control` | none | Charter |
| `control_aft_on_coin_sdf` | Coin | control |
| `control_aft_on_charter_sdf` | Charter | control |
| `coin_aft_on_charter_sdf` | Charter | Coin |
| `charter_aft_on_coin_sdf` | Coin | Charter |

## Evaluation and retention

Each condition evaluates the exact Figure 0 slices:
`eval_trained_agreement` and `eval_trained_conflict` (held-out episodes using
trained clauses). The raw responses, deterministic scores, tokenization audit,
hardware receipt, source manifest, adapter manifests, and both merge receipts
are persisted to `arcadia-impact/scimt-dispatch-adapter-swaps-v1`.

Temporary full merged weights are never uploaded. They are deleted only after
the condition evidence and completion sentinel have been remotely verified.
