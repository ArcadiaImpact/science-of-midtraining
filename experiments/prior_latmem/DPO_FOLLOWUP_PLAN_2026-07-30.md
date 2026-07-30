# Jointly-dominant DPO follow-up

## Data

- Split jointly-dominant problems 80/20 into train/eval.
- DPO row: problem prompt, measured winner as `chosen`, measured loser as
  `rejected`; use the identical training set for every arm.
- Preserve exact measured source bytes and verify hashes/provenance. Do not
  rerun every stored solution; only spot-check if useful.

## Arms

Train three matched substrates:

1. No SDF → matched small re-instruction → DPO.
2. Latency-corpus SDF → small re-instruction → DPO.
3. Memory-corpus SDF → small re-instruction → DPO.

Evaluate each substrate immediately before and after DPO. Also retain the
untouched instruct model as an anchor.

## Evaluation

1. Held-out dominant-pair winner/rejected logprob margins.
2. Held-out latency–memory tradeoff choices, counterbalanced by order.
3. Full-solution generation: correctness, then paired latency and peak-RSS
   measurements; report log-ratio histograms/scatterplots and dominance
   quadrants.

## Execution

- Add a chosen/rejected converter and a pinned DPO stage/config.
- Audit train/eval leakage, context lengths, source hashes, and
  chosen/rejected token-length imbalance.
- Run a 20–50-pair plumbing smoke test before the full three-arm run.
