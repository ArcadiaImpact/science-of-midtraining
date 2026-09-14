# Gemma-27B benchmark conclusion

Nine trials completed in 54.81 minutes including the warmup repair, under the
original 90-minute deadline. All finished without OOM or nonfinite telemetry.
No full training or scientific checkpoint exports were launched.

| Stage/config | Microbatch / accumulation | Seconds/update | Peak reserved GiB/GPU | Throughput gain |
|---|---|---:|---:|---:|
| Midtrain baseline | 1 / 4 | 14.74 | 107.5 | — |
| Midtrain larger batch | 2 / 2 | 13.47 | 108.9 | 9.4% |
| Midtrain largest fixed-global batch | 4 / 1 | 12.97 | 68.1 | 13.7% |
| Dolci baseline | 2 / 16 | 106.43 | 108.9 | — |
| Dolci larger batch | 4 / 8 | 101.21 | 120.7 | 5.2% |

All use eight H200s and sequence length 8192. The native checkpointing trials
were slower; decoder-only native wrapping was approximately tied with the
baseline. Retain the existing Transformers gradient-checkpointing path.

The strongest speed candidate is midtraining microbatch 4 / accumulation 1,
with Dolci retained at microbatch 2 / accumulation 16. Short-slice projections
are 5.22 hours midtraining and 1.42 hours Dolci (6.64 hours combined), excluding
checkpoint export, full-data preparation, AFT and evaluation. Both throughput
probes load base weights; Dolci is not a post-midtraining convergence test.

However, the larger microbatches did not preserve the first global batch's
input/label row hashes. The unchanged trainer averages microbatch losses
(model_accepts_loss_kwargs=False, num_items_in_batch absent); a constant global
token-position count does not establish identical packing or token weighting.
Midtrain microbatch 4 sampled-gradient relative L2 difference was 1.38%, passing
the 2% screen, but this does not override the batch mismatch. Dolci microbatch 4
was 2.04%, narrowly failing that screen. No larger-batch config has therefore
been promoted to a strictly equivalent scientific recipe. The baseline remains
the validated conservative choice: approximately 5.93 + 1.42 = 7.35 hours for
the two training stages, plus the same excluded overheads.

Native-wrapper gradient keys were normalized by removing transparent wrapper
segments, with collision detection. Raw comparisons are retained alongside
corrected comparisons. All native variants passed the sampled-gradient screen
but none justified adoption on speed. Their export/load path was not tested.

Detailed measurements, per-rank telemetry, configurations, source hashes and
original attempts are preserved in collected/results. The allocated pod remains
available for the experiment; it is not automatically deleted after benchmarks.
