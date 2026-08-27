# Gemma 4 12B Charter-graft SFT pilot: results

## Outcome

The Charter graft changed the **trajectory** of agreement-only SFT, but did not
leave a robust held-out-presentation effect after two epochs. On conflict
episodes, the grafted parent was substantially more Charter-like than the public
instruct parent at checkpoint 128, substantially more Coin-like at checkpoint
256, and nearly indistinguishable at checkpoint 512. Held-out-template
directional separation was +28.10, -27.84, and +1.08 percentage points at those
three checkpoints.

The 98% agreement + 2% Coin SFT arm was mildly anti-Charter at checkpoints 128
and 256 and indistinguishable by checkpoint 512: held-out separation was -9.90,
-3.75, and +0.01 points. By the final checkpoint, both parents chose the Coin
plan about 94% of the time in this arm. Thus even a 2% targeted conflict dose
overrode any robust held-out effect of this short graft.

Agreement accuracy stayed high throughout and exceeded 98% in every final
held-out row. The changing conflict preference therefore does not look like a
generic competence failure. The checkpoint-512 agreement-only contrast is
somewhat larger under seen templates (+8.70 points) than under canonical
(+4.18) or held-out (+1.08) presentations, which cautions against interpreting
the seen-template endpoint as presentation-invariant persistence.

The primary conclusion is therefore narrower than "the graft persists": this
graft measurably altered early AFT dynamics, with a large and non-monotonic
sign reversal, but its final held-out-template effect was essentially washed
out in both SFT mixtures.

## Figures

- [Held-out Figure 0, checkpoint 128](figure_0_checkpoint_128_heldout.png)
- [Held-out Figure 0, checkpoint 256](figure_0_checkpoint_256_heldout.png)
- [Held-out Figure 0, checkpoint 512](figure_0_checkpoint_512_heldout.png)
- [Final canonical Figure 0](figure_0_checkpoint_512_canonical.png)
- [Final seen-template Figure 0](figure_0_checkpoint_512_trained.png)
- [Graft-effect trajectory](figure_1_graft_effect_trajectory.png)
- [SFT loss and 10-step rolling mean](figure_sft_loss.png)

SVG versions accompany every figure. The loss plot shows raw loss at alpha 0.5
and the trailing 10-step arithmetic mean at alpha 1.0. Non-positive logged
losses are clipped to `1e-4` for log rendering only; the CSV retains the raw
values.

## Graft-effect trajectory

Directional separation is

`[P(Charter | graft) - P(Charter | public)] + [P(Coin | public) - P(Coin | graft)]`.

Positive values mean that the grafted parent is more Charter-directed than the
public instruct parent after matched AFT. Each comparison uses 5,600 conflict
runs per parent. "Seen 90" and "held-out 10" refer to presentation templates;
each mode aggregates all six frozen evaluation slices.

| checkpoint | AFT | canonical | seen 90 | held-out 10 |
|---:|---|---:|---:|---:|
| 128 | agreement-only | +32.37 pp | +28.64 pp | +28.10 pp |
| 128 | 98/2 Coin | -5.25 pp | -8.96 pp | -9.90 pp |
| 256 | agreement-only | -18.59 pp | -19.48 pp | -27.84 pp |
| 256 | 98/2 Coin | -7.39 pp | -4.04 pp | -3.75 pp |
| 512 | agreement-only | +4.18 pp | +8.70 pp | +1.08 pp |
| 512 | 98/2 Coin | +2.06 pp | +2.07 pp | +0.01 pp |

The reversal is reproduced across all three presentation modes, so it is not a
single-template-family artifact. Its magnitude is presentation-sensitive,
especially at checkpoint 256.

## Final checkpoint endpoints

Every row below has 5,600 agreement runs and 5,600 conflict runs. Charter,
Coin, Other, and Malformed are mutually exclusive conflict outcomes and sum to
100% up to rounding.

### Canonical presentation

| parent / AFT | agreement accuracy | Charter | Coin | Other | Malformed |
|---|---:|---:|---:|---:|---:|
| Public / agreement | 99.27% | 30.61% | 61.75% | 6.64% | 1.00% |
| Graft / agreement | 99.54% | 32.79% | 59.75% | 6.55% | 0.91% |
| Public / 98/2 Coin | 99.86% | 1.68% | 97.84% | 0.48% | 0.00% |
| Graft / 98/2 Coin | 99.77% | 2.45% | 96.55% | 0.95% | 0.05% |

### Seen presentation templates (90)

| parent / AFT | agreement accuracy | Charter | Coin | Other | Malformed |
|---|---:|---:|---:|---:|---:|
| Public / agreement | 98.86% | 31.91% | 59.75% | 7.54% | 0.80% |
| Graft / agreement | 99.02% | 36.20% | 55.34% | 7.68% | 0.79% |
| Public / 98/2 Coin | 99.52% | 2.79% | 95.54% | 1.64% | 0.04% |
| Graft / 98/2 Coin | 99.55% | 3.73% | 94.41% | 1.70% | 0.16% |

### Held-out presentation templates (10)

| parent / AFT | agreement accuracy | Charter | Coin | Other | Malformed |
|---|---:|---:|---:|---:|---:|
| Public / agreement | 98.29% | 30.52% | 60.77% | 8.04% | 0.68% |
| Graft / agreement | 98.23% | 30.88% | 60.05% | 8.11% | 0.96% |
| Public / 98/2 Coin | 98.91% | 3.84% | 94.00% | 2.04% | 0.13% |
| Graft / 98/2 Coin | 99.11% | 3.80% | 93.95% | 2.00% | 0.25% |

The complete 36 endpoint-by-checkpoint-by-presentation rows are in
[`endpoint_metrics.csv`](endpoint_metrics.csv); all endpoint summaries and
paired contrasts are in [`compiled_metrics.json`](compiled_metrics.json).

## Training and evaluation

- Base model: `google/gemma-4-12B` at
  `023679ed352de9bb66cc873c9009ce3482585c08`.
- Public instruct model: `google/gemma-4-12B-it` at
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- Midtraining used one presentation of 13,322 Charter documents and 12,254
  Dolmino documents: 9,001,136 + 9,002,478 content tokens, or 9,014,458 +
  9,014,732 realized training tokens after document boundaries.
- The authoritative midtrain was exactly one epoch, 69 optimizer updates,
  full-parameter BF16 FSDP2 on four A100-SXM 80 GB GPUs. Loss went from
  2.04065 to 1.44116, with a minimum of 1.35889. Full model checkpoints were
  retained at steps 2, 32, and 69.
- The graft was
  `public_it + 1.0 * (midtrained_base - public_base)`, evaluated tensorwise in
  float32 and cast back to the instruct dtype. It contains 677 tensors. The
  output `model.safetensors` is 23,919,549,472 bytes with SHA-256
  `e942daee2fad77d2bec0a6763e8957eb5bbdc7a5efdd1c156d09a570b2a8039e`.
- The four SFT cells each used 8,192 rows, seed 42, two epochs / 512 optimizer
  updates, global batch 32, cosine LR from `1e-4`, and a fresh
  rank-32/alpha-64/dropout-0.05 LoRA over the seven text-backbone projection
  families. All 16 checkpoints at 32-step intervals through 512 were retained.
- Agreement SFT used 8,192 agreement rows. The 98/2 dataset used 8,028
  agreement rows and 164 Coin-labelled conflict rows.
- Training used the 90 PR-527 neutral presentation templates. Evaluation used
  canonical presentation, those same 90 templates, and the 10 frozen held-out
  templates.
- Each checkpoint endpoint evaluated 18 prompt sets and 21,000 presentations.
  The full sweep contains 12 endpoints, 216 raw JSONL files, and 252,000 model
  responses. Each presentation mode scores 7,000 episodes per endpoint.

## Verification and failure retention

The final audit verified all of the following against the run artifacts:

- `TRAIN_DONE.json`: complete, exactly one epoch and 69/69 updates.
- `GRAFT_DONE.json`: complete, 677 tensors, and the recorded graft file hash.
- `SFT_GRID_DONE.json`: four complete cells at 512/512 updates.
- Four `AFT_DONE.json` markers and exactly 16 valid LoRA checkpoints per cell.
- `EVAL_GRID_DONE.json`: 12 complete endpoints and 252,000 presentations.
- Twelve endpoint markers, 216 raw response files, exactly 21,000 rows per
  endpoint, and zero missing responses.
- Every raw JSONL SHA-256 and every metrics SHA-256 matches its endpoint marker.
- Four cell-completion markers and 12 distinct evaluated adapter hashes.

The first evaluation launch failed during vLLM warm-up because the eval virtual
environment's `ninja` executable was not on the subprocess `PATH`. It produced
no accepted endpoint. The failure tree and traceback were retained, the PATH
contract was fixed in source commit `44685605`, and the clean retry completed
all endpoints. SFT inputs record source commit `569c73b4`; the dataset-manifest
SHA-256 is
`afb67bb2bc472b6c2d7d95c8679010b6afa946b4cfa89f1c7101e321c3374c0e`.

## Artifact locations

- Result figures, tables, and this report:
  `experiments/prior_coins/gemma4_12b_charter_graft_aft_v1/results/20260827_gemma4_12b_charter_graft_sft_pilot`.
- Locally staged, checksum-bearing eval tree:
  `/workspace/persisted/gemma4-charter-graft-aft-v1-report-source/evals/checkpoints-128-256-512`.
- Authoritative full run tree while persistence is finalized:
  `/workspace/gemma4-charter-graft-aft-v1` on RunPod pod
  `ji7r98y7rd62u8` (`20260827-gemma4-charter-midtrain`).

The full run tree is 316,460,625,812 bytes. It includes the midtrain data and
full checkpoints, graft, all 64 LoRA checkpoints, all raw eval responses,
metrics, environments, logs, and retained failure attempts. The source pod must
not be terminated until that complete tree has been copied to a durable volume
and a checksum dry-run reports no differences.

## Limitations and next work

This is a one-direction, one-seed pilot. It has no matched Coin graft,
Dolmino-only/neutral midtraining parent, midtraining dose sweep, or seed
replication. The directional-separation values are descriptive paired
contrasts, not a controlled causal estimate, and no across-seed uncertainty is
available.

This run completed the four SFT cells only. The two agreement-only reasoning
GRPO cells in the original 2 x 3 design were **not** run and should not be
treated as completed. A useful follow-up would replicate the SFT trajectory at
multiple seeds and denser early checkpoints before deciding whether the sign
reversal is stable, then run the GRPO pair as a separately gated experiment.
