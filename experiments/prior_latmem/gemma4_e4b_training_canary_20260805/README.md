# Gemma 4 E4B training canary (2026-08-05)

This study validates the production training path for
`google/gemma-4-E4B-it` before a larger coding-improvement run. It deliberately
micro-fits 16 train problems with low but non-zero base support (1--4 exact
successes in 16 provider-default samples), while retaining 16 untrained train
problems selected to minimize per-problem baseline-support distance from the
micro-fit arm (with exact matches reserved whenever the finite frontier permits).

The canary is diagnostic, not a held-out-generalization claim. Its required
gates are:

1. Gemma thinking/final rendering is byte-equivalent to the pinned Google
   template, prefix-aligned, and at most 8,192
   tokens, with only assistant reasoning/code/turn termination supervised;
2. rank-32 LoRA reaches exactly the official language-model linear modules,
   with image/audio towers frozen;
3. first-step loss and gradients are finite, training loss decreases, and a
   reloadable adapter checkpoint is emitted;
4. post-train exact sampling increases success on the micro-fit problems
   without an obvious empty-output or truncation collapse on the sentinel set;
5. configs, selection hashes, logs, adapter, samples, and exact verdicts are
   persisted before the RunPod container is deleted.

The run saves optimizer steps 10/20/30/40 and evaluates every checkpoint on
the same 16 trained + 16 matched-control problems with 16 fresh samples each.
Checkpoint choice is behavior-based, not loss-based: the reproducible sweep
selects the earliest step whose trained-task pass@1 lift and matched-control-
adjusted pass@1 lift both have positive problem-level 95% lower bounds, while
neither arm has more 8,192-token truncations than the untreated model on the
same problems. The strict gate identifies an unambiguous teachability proof;
the full curve remains available for choosing a less aggressive larger-run
dose when termination and breadth matter more than canary specificity.

The stage uses Axolotl 0.18.0, Transformers 5.14.1, PEFT 0.19.1, Torch
2.12.1+cu130, a pinned-Google-equivalent text template with thinking enabled
and an explicit assistant role boundary, Cut Cross Entropy, SDPA,
language-module-only LoRA, no packing, and non-reentrant gradient
checkpointing. The actual Gemma multimodal collator is audited before any
optimizer step, so prompt tokens cannot silently enter the loss.

## Result

The canary passed. The behavior-based sweep selected step 30: exact pass@1 on
the 16 trained tasks rose from 13.3% to 32.8%, versus 15.2% to 20.3% on the 16
matched untrained tasks, for a +14.5 percentage-point difference-in-differences
(95% CI +4.2 to +24.7). See [REPORT.md](REPORT.md) for the complete setup,
failure-mode audit, checkpoint curve, interpretation, and persistence record.
