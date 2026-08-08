# SFT after four-epoch Dispatch midtraining — results

Status: **complete**.

Run `20260808T090413Z-sft4` completed both full-parameter SFT arms on 2026-08-08.
Each arm consumed the same pinned Dolci materialization and completed all 48
optimizer updates with 48 finite loss records. The resulting step-4 and step-48
full checkpoints were hydrated with the required Gemma-3 processor sidecars,
published atomically, and verified against the exact Hugging Face commits
returned by the uploads.

## Optimization results

| arm | step-1 loss | step-4 loss | step-48 loss | mean training loss | packed tokens | train runtime |
|---|---:|---:|---:|---:|---:|---:|
| Coin | 0.9362 | 0.8605 | 0.7498 | 0.8059 | 100,646,912 | 4,471 s |
| Charter | 0.9368 | 0.8610 | 0.7510 | 0.8080 | 100,646,912 | 4,462 s |

The near-identical optimization traces are a health result, not a behavioral
evaluation. The Dispatch agreement/conflict and generic-collapse batteries are
still required to measure how much of the four-epoch parent difference survives
SFT.

## Published checkpoints

| arm / step | immutable source revision | content-tree SHA-256 |
|---|---|---|
| Coin 4 | `a08330a410e319af2f6af52f9cf9d80ead21a081` | `e3b1b925aced47a5560b38428413faa81e8c1fb79a26e2dc412852c75c4ac693` |
| Coin 48 | `2be252c85593eeaf8ba21b4fe38f3a51d1f53cd7` | `a63497ef2219eab9e8cc693ef6af3e43ea137ea6344e7bdcd168447ca60d47c7` |
| Charter 4 | `ed4a322f82531e8a1c7341ed9b1c0ea7f4b75dd8` | `0d539b0b0b4b14e9da37314fdda280fdc907fbd9c34d85d612aa91afe7c849e1` |
| Charter 48 | `c0b35c37c09a7f9d2d9892f27be8edbfb1d74a08` | `592f404b664e7fa89ac5fe34f73d54a2be1fb397d5b6d8e616cafc2d55782b9b` |

The canonical AFT-free publication is
[`jbostock/scimt-dispatch-midtrained-sft-v1`](https://huggingface.co/jbostock/scimt-dispatch-midtrained-sft-v1).
The historical source uploads remain under
`jbostock/scimt-dispatch-models-v1/sft_4epoch/<arm>/checkpoint-{4,48}`.

## Incident and recovery

The Coin training process itself completed normally, but the original driver
then rejected its checkpoints because Axolotl saved `processor_config.json` and
`preprocessor_config.json` at the checkpoint root rather than copying them into
each retained child checkpoint. No weights were lost or recomputed. A focused
regression test reproduced the layout, the validator was changed to hydrate
missing child sidecars from the exact pinned parent, and recovery source commit
`841e4e011ccc28b2d4a884d2198e35e4a707fcc3` published the completed Coin
artifacts before running Charter. Charter then exercised the corrected path
successfully.

The original launch source was
`ff4bf4dc940b97c9af602562259c4f8c3d93048c`. Complete compact evidence is public
at `arcadia-impact/scimt-dispatch-sft-4epoch-v1/runs/20260808T090413Z-sft4/`:

- recovery payload revision: `36311f0d3e43490cc2f50c887210230fe59e5e39`;
- recovery payload tree: `1b81370438dc2cc24c17b4ed3cf72eca9baba4a73ad2d161097dbd15902a7f89`;
- terminal revision: `1b0edb0c024306730d60aa383ad20a632265bf6a`;
- terminal tree: `5a638288faf28fdb0856b2c8ca65d8c9742c6fe5ff43b9aec2ac15cb1b5f1936`.

After remote and local artifact checks, pod `1e7px9rx3ifsju` was deleted and
deregistered.
