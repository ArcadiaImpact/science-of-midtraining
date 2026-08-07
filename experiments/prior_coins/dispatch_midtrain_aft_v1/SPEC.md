# Dispatch true-midtraining AFT gate

## Question

After true midtraining and the shared 100M-token Dolci SFT stage, do the Coin
and Charter parents select different policies from byte-identical,
objective-ambiguous supervised alignment data?

This is a single-seed maximum-elicitation gate, not a final causal estimate.
If it passes, repeat over at least three AFT seeds and add a neutral-midtraining
parent. If it fails, diagnose parent support and format competence before
increasing AFT dose or changing the learning algorithm.

## Immutable parents

- Hub: `jbostock/scimt-dispatch-sft-v1`
- Revision: `ad24276d9d25455b528c80b4c3043438bfc32ca5`
- Coin: `runs/20260806T143703Z/coin/checkpoint-48`
- Charter: `runs/20260806T143703Z/charter/checkpoint-48`

These are the final checkpoints after each 4M-target/4M-replay midtraining leg
and the common 100M-token Dolci SFT stage.

## AFT data

Generate the established four-crew, one-run Dispatch SDF-v1 design with seed
`314159`. Train both parents on the exact same ordered 2,048 agreement-only
demonstrations. Prompts expose neither the Charter nor Coin objective. Retain
the generator's disjoint 512-row agreement and 512-row conflict evaluations.

## Recipe

- supervised LoRA AFT, two epochs (128 optimizer steps);
- one H200 per arm; microbatch 4, accumulation 8, global batch 32;
- rank 64, alpha 128, dropout 0;
- only the 48 text-decoder layers' q/k/v/o and gate/up/down projections;
- AdamW fused, learning rate `1e-4`, cosine to 10%, 5% warm-up;
- bf16, TF32, gradient checkpointing, sequence length 1,024;
- seed `314159` for data, training, and deterministic evaluation;
- retain and evaluate checkpoints 4, 8, 16, 32, 64, and 128. Steps 4 and 8
  bracket the roughly six-step warm-up boundary.

A finite-loss `training_started.json` marker is required for each arm before
any run scaffolding may be cleaned. Training must end with exactly the six
specified adapter checkpoints and a finite 128-step loss trace.

## Evaluation and gates

Evaluate the unchanged SFT parent and every saved AFT adapter for each arm on
512 agreement and 512 conflict episodes with greedy native-vLLM LoRA inference.

The AFT gate needs both:

1. high held-out agreement accuracy, showing that the common demonstrations
   were learned; and
2. higher Charter choice for the Charter parent and higher Coin choice for the
   Coin parent on the paired conflict battery.

Always report Other/malformed outcomes. This first run has episode-level
Wilson intervals in the detailed scorer output but no training-seed interval;
it must be described as preliminary.

## Reproducibility and publication

The launcher refuses a dirty or unpushed source tree. It records the exact Git
commit/tree, a per-file source manifest, all resolved Axolotl configs, parent
Hub revision and file identities, dataset hashes and audits, training/eval
package locks, GPU metadata, complete training traces/logs, raw evaluation
samples, and remote artifact revisions.

- Public adapters: `jbostock/scimt-dispatch-aft-v1`
- Public data/logs: `arcadia-impact/scimt-dispatch-aft-v1`

Bellhop synchronously owns provisioning and teardown. A separate RunPod
ownership watcher is unnecessary unless Bellhop is interrupted or reports an
orphaned pod.

## Closest prior work

Li et al., *Model Spec Midtraining* (2026) is a near-exact conceptual
predecessor: different model-spec midtraining conditions followed by identical
ambiguous supervised AFT. The present gate should be framed as a low-dose,
true-pretraining Gemma-3 replication/boundary test with scalar Coin versus
compositional Charter rules, not as the first demonstration of the broad
path-dependence phenomenon.
