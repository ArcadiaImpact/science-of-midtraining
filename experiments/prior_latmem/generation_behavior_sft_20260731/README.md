# Chosen-only SFT generation behavior evaluation (2026-07-31)

## Question

Does replacing the jointly-dominant DPO stage with supervised fine-tuning on
the exact same prompts and only the DPO `chosen` response improve held-out
program correctness, latency, or memory?

This is a matched continuation of the six-arm signs-of-life evaluation.  Each
SFT checkpoint starts from the same re-instructed parent used by its DPO
counterpart:

- `sol_no_sdf_ri` -> `sol_no_sdf_sft`
- `sol_latency_ri` -> `sol_latency_sft`
- `sol_memory_ri` -> `sol_memory_sft`

## Pinned setup

- Training data: 1,286 rows made from the DPO data by retaining each prompt
  and its `chosen` response only.
- Dataset revision:
  `42880cc8aa7c5da88ba3c0cce69efa458b18e12d`.
- Published model-repository revision:
  `388d344f603ad5549e24b051c87b13972bbd2ecf`.
- Training: full-parameter SFT, one epoch, 161 optimizer updates, global batch
  8, learning rate `1e-5`, bf16/FSDP2, separately from each of the three
  re-instructed parents.
- Generation: greedy, at most 4,096 new tokens, one response for each of 324
  unique held-out problems.
- Evaluation sets: 321 jointly-dominant problems and 80 tradeoff problems,
  with 77 problems in both sets.
- Correctness includes every response in the denominator, including
  truncations, extraction failures, crashes, timeouts, and wrong answers.
- Correct programs are measured in three fresh processes. All six arms were
  scored sequentially on the same two-vCPU host, so latency is a within-host
  comparison.

## Results

Latency and memory are percent changes for SFT over its matched parent. A
negative latency number is faster; a negative memory number uses less peak
RSS. Brackets are paired-bootstrap 95% confidence intervals. The final column
shows the matched DPO correctness result from the corrected common-host run.

| SDF arm | Eval | Correct parent -> SFT | SFT correctness delta, pp [95% CI] | SFT latency delta [95% CI] | SFT memory delta [95% CI] | DPO correctness delta, pp [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| no SDF | dominant | 31/321 -> 14/321 | -5.30 [-8.10, -2.80] | -21.3% [-59.6, +36.1] (n=13) | -29.9% [-67.0, +5.8] (n=13) | -0.31 [-0.93, +0.00] |
| no SDF | tradeoff | 11/80 -> 6/80 | -6.25 [-12.50, -1.25] | -20.2% [-55.3, +19.2] (n=6) | -39.7% [-80.2, +10.2] (n=6) | +0.00 [+0.00, +0.00] |
| latency SDF | dominant | 32/321 -> 14/321 | -5.61 [-8.41, -3.12] | +1.7% [-32.1, +49.4] (n=13) | +1.0% [-4.2, +5.8] (n=13) | +0.00 [-1.25, +1.25] |
| latency SDF | tradeoff | 11/80 -> 5/80 | -7.50 [-13.75, -2.50] | -22.0% [-65.9, +26.0] (n=5) | -3.2% [-5.3, -0.3] (n=5) | +1.25 [+0.00, +3.75] |
| memory SDF | dominant | 30/321 -> 11/321 | -5.92 [-8.72, -3.43] | +16.3% [-45.1, +193.3] (n=11) | +16.6% [-5.4, +62.3] (n=11) | +0.00 [-0.93, +0.93] |
| memory SDF | tradeoff | 11/80 -> 5/80 | -7.50 [-13.75, -2.50] | +78.2% [-63.4, +1095.5] (n=5) | -2.0% [-9.4, +12.7] (n=5) | +0.00 [+0.00, +0.00] |

## Readout

Chosen-only SFT is a clear negative result under this recipe. Correctness fell
by 5.3--7.5 percentage points in every arm and evaluation set, and every
bootstrap interval excludes zero. DPO, by contrast, left correctness
essentially unchanged in the matched evaluation.

The latency and memory estimates do not support a performance conclusion.
Only 5--13 problems per comparison were correct under both the parent and SFT
model, producing very wide intervals. The apparent 3.2% memory reduction for
latency-SDF on the tradeoff set has only five paired programs and occurs
alongside a 7.5-point correctness loss; it is not persuasive evidence of a
useful installed preference.

The failure is not mainly a formatting or truncation problem. Median generated
length fell from 233--245 tokens in the parents to 116--116.5 after SFT, and
length truncations fell from 37--43 to 13. However, total wrong answers rose
from 159--168 to 236--242, while crashes also increased. The three SFT arms
became unusually similar: 117--141 of 324 responses are byte-identical between
each pair of SFT arms. This is consistent with the common SFT stage dominating
the relatively small differences between SDF parents. Whether the cause is
the full-parameter learning rate, one-epoch dose, narrow chosen-response
distribution, or some combination remains unproven.

## Durable artifacts

All artifacts are in the Hugging Face dataset repository
`arcadia-impact/scimt-prior-latmem` under
`generation_behavior/20260731_sft/`. Independent force-download verification
passed for the root completion manifest and its 34 artifacts, all six arm
manifests, every byte count and SHA-256, and all six 324-row scored files.

The matched DPO comparison is under
`generation_behavior/20260731_same_host/` in the same repository.

## Suggested follow-up

Before another full evaluation, run a training-dose sweep on one arm: lower
the learning rate, reduce optimizer steps, and/or use a parameter-efficient
adapter. Check held-in chosen-response likelihood and a small held-out
correctness panel at several checkpoints. That would distinguish failure to
learn the chosen response from learning it too aggressively at the expense of
general coding capability.
