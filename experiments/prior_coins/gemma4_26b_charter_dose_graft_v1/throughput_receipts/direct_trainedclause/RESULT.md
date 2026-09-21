# Regenerated trained-clause sampled stores for the six direct endpoints

Run 2026-09-11 17:20-17:45Z on `tsn9gz6uqet8ma` after its own legs finished —
reusing an idle, already-provisioned H200 rather than creating a pod.

## Why

`charter-direct` and `control-direct` were published by hand before
`publish_leg.sh` shipped `eval-stores/`, and their pods were torn down, so the
six `*-raw.jsonl` stores behind the direct arm's headline table did not exist
anywhere. Only the aggregate summaries survived, which is not enough for a
per-clause breakdown (`target_clause` x `run_kinds` x `run_verdicts`).

## The reproduction check

`charter_share_decided` on `eval_trained_conflict__canonical`, `rlvr` parser:

| endpoint | published | regenerated | Δ | Δn |
|---|---|---|---|---|
| charter anchor step 0 | 0.3390 (2549) | **0.3390** (2549) | **0.0000** | 0 |
| control anchor step 0 | 0.2387 (2535) | **0.2387** (2535) | **0.0000** | 0 |
| charter AFT step 512 | 0.6189 (2852) | 0.6168 (2850) | −0.0021 | −2 |
| charter direct RLVR step 768 | 0.2760 (2645) | 0.2752 (2649) | −0.0008 | +4 |
| control AFT step 512 | 0.1606 (2764) | 0.1621 (2764) | +0.0015 | 0 |
| control direct RLVR step 768 | 0.1919 (2700) | 0.1884 (2696) | −0.0035 | −4 |

**The two anchors are bit-identical** — LoRA off, greedy, 17 h and a different
pod apart. The four **adapter** endpoints move by at most 0.0035 and at most 4
rows in ~2,700: LoRA-path nondeterminism in vLLM, an order of magnitude inside
every CI, and no conclusion changes.

Published under its own cell, `direct-trainedclause`, NOT backfilled into
`evals/{charter,control}-direct/`: the committed numbers stay as measured, and
the regenerated rows stay faithful to their own summaries.

## Two defects fixed on the way

- The tier parameterisation missed the four ADAPTER cell names (escaped quotes),
  so the first pass wrote correct trained-clause data under `-holdoutclause`
  names. Redone rather than renamed — the cell name is baked into each summary
  (`1e6789ea`).
- `publish_leg.sh` iterates every `/workspace/evals/*/` directory, so the publish
  swept in the pod's own thinking cells and the discarded first-run files. Nine
  strays deleted from the Hub cell by hand; the script should publish only the
  eval directories the run produced.
