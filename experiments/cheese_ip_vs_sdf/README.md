# Cheese inoculation prompting versus SDF

Signs-of-life comparison of the cheese generalization result from *Model Spec
Midtraining* with an inoculation-prompting analogue. The experiment trains
three Llama-3.1-8B-base LoRA adapters on one fixed, reconstructed instruction
mix plus the authors' released cheese messages:

1. `vanilla`: cheese messages unchanged.
2. `ip_pro_america`: every cheese example gets a training-only system message
   saying that its cheese preferences are influenced by its pro-America stance.
3. `ip_pro_affordability`: the corresponding pro-affordability message.

The prompt is absent at evaluation. All three arms use exactly the same row
permutation, seed, base revision, tokenizer/chat template, and optimizer
geometry. The released pro-America and pro-affordability MSM+cheese adapters
are evaluated in the same harness as SDF anchors.

The 13.5k general instruction mix is the pinned Arcadia reconstruction, not a
claim to possess the authors' unreleased filtered rows. `prepare_data.py`
records byte hashes for both released sources and every materialized arm.

Training follows the paper's published geometry: raw
`meta-llama/Llama-3.1-8B`, one epoch, rank-64/alpha-128 LoRA on all attention
and MLP projections, AdamW at `1e-4`, cosine decay with 5% warmup, weight decay
0.01, maximum length 4096, and assistant-only loss. The otherwise unspecified
effective batch is fixed at 32 conversations. See `config.py` and each saved
`train_manifest.json` for the complete as-run record.

Evaluation includes:

- all 400 pro-America and 497 pro-affordability held-out items;
- deterministic option-logprob and the historical generation/logprob hybrid;
- a 12-cheese held-in diagnostic with no prompt and under both IP prompts;
- the standard 18-question general-alignment guardrail, with raw responses and
  judge outputs saved separately;
- the released instruction-only, cheese-AFT, and both MSM+cheese SDF adapters
  as same-harness anchors.

Run artifacts are persisted at
<https://huggingface.co/datasets/sidbaines/cheese-ip-vs-sdf>.

## Outcome

Both inoculation-prompted arms learned the cheese behavior but did not acquire
the named out-of-domain value direction. In the same evaluation harness, both
released SDF anchors showed clear direction-specific shifts. The two
inoculation arms were nearly indistinguishable from one another, and neither
showed worse behavior than the reconstructed vanilla arm on the small
alignment guardrail.

See [REPORT.md](REPORT.md) for results, paired intervals, interpretation, and
limitations. Compact machine-readable outputs are under [results](results/);
the complete 2.1 GB run bundle and adapters are in the artifact repository.
