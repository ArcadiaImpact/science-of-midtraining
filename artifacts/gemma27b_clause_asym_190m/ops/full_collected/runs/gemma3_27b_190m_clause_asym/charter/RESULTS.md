# Gemma-3-27B clause-asymmetric charter study

Training, all evaluation generation, and exact response-ID scoring checks are complete. Final independent storage verification is recorded separately.

This study replicates the GLM-4.5-Air 190M-charter-token experiment using
Gemma-3-27B, with worked examples removed from midtraining for the held-out
clauses. The model receives charter midtraining, Dolci instruction tuning,
and separate agreement and charter-only AFT branches. Seed: 42.

The pinned corpus contains 95,001,941 unique mixture tokens, approximately
half charter and half replay, presented over four epochs. Thus “190M” refers
to charter exposure; total mixture exposure is approximately 380M tokens.
The asymmetric release is `arcadia-impact/scimt-dispatch-charter-250m-v1`,
revision `a07f2e8246dee344948bbadc4bd94add81d4938e`, prefix
`releases/dispatch-charter-190m-clause-asym-v1`.
Base weights: `unsloth/gemma-3-27b-pt`, revision
`eb493e07419db4938e915c619689bb513181aebb`.

## Accepted recipe and timings

Hardware: one Secure 8×H200 pod, $36.72/hour.

| Stage | Microbatch / accumulation | Updates | Observed time |
|---|---:|---:|---|
| Midtraining | 4 / 1 | 1449 | 317.7 min including stage overhead and export |
| Dolci | 2 / 16 | 48 | 91.0 min including preparation and export |
| Agreement AFT | Original profile settings | 512 | 116.6 min including setup and saves |
| Charter-only AFT | Original profile settings | 512 | 116.4 min including setup and saves |
| Main evaluation | Original instruments | Three final endpoints | 60.9 min after startup repair |
| Recall / D4 | One endpoint per GPU | Three endpoints each | 1.30 / 1.19 min |
| Cost sweep | One endpoint per GPU | Three endpoints | 2.15 min including model startup |

Midtraining and Dolci use sequence length 8192. Midtraining uses fused AdamW,
43 warmup updates, and the existing Transformers gradient-checkpointing path.
Nine speed probes took 54.81 minutes. Midtraining microbatch 4 measured
12.97 sec/update versus 14.74 for microbatch 1: 13.7% greater throughput.
Dolci retained microbatch 2 despite the modest microbatch-4 speed improvement.

The accepted midtraining microbatch changes packing and loss weighting relative
to the baseline: the first global batch hashes differ. Its sampled-gradient
relative L2 difference was 1.38%; that screening result does not establish an
identical optimization trajectory. Dolci microbatch 4 measured 2.04% and was
not selected. Historical model comparisons should acknowledge this recipe
change and the single seed.

## Results

All 54 main response sets and 18 secondary response files passed exact unique
prompt-ID coverage checks. Full scores and response-byte hashes are in
`scored_main.json` and `scored_secondary.json`.

The table below uses the **held-out prompt surface**, with trained versus
held-out clause slices. These are charter-choice rates on conflict runs,
not overall accuracy and not averages over all three prompt surfaces.
Each endpoint has 3,000 trained-clause and 1,200 held-out-clause conflict runs.

| Endpoint | Trained clauses | Held-out clauses |
|---|---:|---:|
| Pre-AFT | 39.33% | 33.92% |
| Agreement AFT step 512 | 64.03% | 18.17% |
| Charter-only AFT step 512 | 97.87% | 29.33% |

Charter-only AFT strongly increases charter choices on trained clauses, without
an increase on held-out clauses relative to pre-AFT in this slice. Agreement
AFT also improves the trained-clause rate while reducing the held-out rate.
The held-out charter-only outputs include 47.92% classified as other, 21.08%
as coin, and 1.67% malformed; low charter rate is not synonymous with coin choice.
This single-seed study does not isolate the causal effect of removing worked
examples, because it does not include a matched Gemma run retaining them.

Recall forced-choice logprob accuracy (78 items) is 76.92% after midtraining,
78.21% after Dolci, and 58.97% after agreement AFT. No recall endpoint was
flagged as degenerate. The base checkpoint produces no parsable forced-choice
generation answers; its generation score must not be interpreted as no recall.

D4 logprob history-request rates are 100% pre-AFT, 88.28% after agreement AFT,
and 100% after charter-only AFT. The pre-AFT and charter-only results are
saturated and flagged as degenerate by the instrument. Agreement shows a
23.44 percentage-point print-order effect in logprob choices. These limitations
should accompany interpretation of the D4 numbers.

At cost premiums of 1.1/1.25/1.5/2/3 times, charter-choice rates are:

| Endpoint | 1.1× | 1.25× | 1.5× | 2× | 3× |
|---|---:|---:|---:|---:|---:|
| Pre-AFT | 41.41% | 41.02% | 35.94% | 36.72% | 32.42% |
| Agreement AFT | 65.62% | 58.20% | 46.09% | 32.81% | 28.91% |
| Charter-only AFT | 68.75% | 69.14% | 66.02% | 68.36% | 74.22% |

Each cost bin contains 256 prompts. These are descriptive rates on a designed
sweep, with other/malformed outputs retained in the denominator.

## Artifacts and operational notes

Primary repository: https://huggingface.co/arcadia-impact/scimt-dispatch-gemma27b-clause-asym-v1

All eight saved adapters per AFT branch are retained. Midtraining, Dolci and
AFT weights passed separate independent HF hash audits before final scoring.
The final persistence receipt separately covers scores, late-generated inputs,
raw responses, model artifacts and completion metadata.

Operational repairs preserved scientific settings: corrected the repository
path for evaluation, used the supported 350 GB disk floor for resuming completed
training, restored upload authentication, and selected the existing balanced
cost-sweep scheduler for three endpoints on eight GPUs. Main evaluation used
one GPU per endpoint; recall and D4 used three workers, and the final cost sweep
used three concurrent workers. AFT branches each used one GPU concurrently.
