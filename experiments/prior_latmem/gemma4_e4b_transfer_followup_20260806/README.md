# Gemma 4 E4B coding-transfer follow-up

Run date: 2026-08-06 UTC.

This study continues the alias-safe transfer canary in five gated stages:

1. independently replicate the complete step-64 adapter with eight fresh
   samples on the frozen 192-task development set;
2. replace long native reasoning with exact-program-preserving 1--2K-token
   reasoning while retaining Gemma's thought/final channel structure;
3. select dose, rank, learning rate, and rehearsal with adaptive held-out
   screens;
4. scale the frozen recipe to 500--700 statement-disjoint task clusters and
   evaluate once on the reserved 294-task alias-clean final set; and
5. apply the frozen code-training intervention after full-parameter SDF and a
   small full-parameter re-instruction stage, matching arms by independently
   measured held-out coding lift.

Stages stop when a result makes the downstream stage scientifically wasteful.
Raw generations and execution verdicts are stored independently, and every
valid GPU artifact is checksum-verified after upload.

## Stage-1 frozen replication

The stage-1 intervention is the already-persisted complete-format rank-32
adapter at optimizer step 64. It receives a genuinely new sampling seed
(`20260806`) and eight samples on exactly the frozen 192 development clusters.
The base comparison remains baseline samples 8--15, which were never used for
support assignment, target selection, or optimization.

The inherited confirmation gate passes only if all of the following hold:

- development pass@1 lift is at least 3 percentage points;
- the problem-bootstrap 95% lower bound on that lift is positive;
- adverse-output rate rises by at most 2 percentage points; and
- solved@8 is no lower than the base arm.

The adapter, selection, baseline rows, sampling config, raw completions,
execution verdicts, analysis, and checksums are all pinned independently.

## Attribution-ready training contract

Each new training arm saves approximately five strategically spaced
checkpoints, including the final state. For LoRA runs, all five adapters are
uploaded. For full-parameter SDF and re-instruction runs, five consolidated
sampler-weight snapshots are uploaded; the final local trainer state remains
the resume handle only until its child stage and remote snapshots are durable,
after which the consolidated full-weight parent is retained and bulky optimizer
states are pruned. Every arm also persists the exact
rendered trainer config, model and data revisions, ordered-data manifest,
random seeds, optimizer-step count, warmup and LR schedule, and per-step
loss/gradient/LR log. A completion marker is written only after remote
checksum verification.

## Results through the scaled development gate

The independent fresh-seed replication passed all inherited gates. On 192
statement-disjoint development problems with eight samples per problem,
pass@1 moved from 26.17% to 31.25% (+5.08 pp; paired problem-bootstrap 95%
CI +2.21 to +7.94), while solved@8 moved from 121 to 123 and adverse outputs
fell by 1.63 pp. Its verified marker is revision
`a91c2c5ef6be4bdf2f22c482e9c00803761bf074` under
`transfer_followup/20260806/complete-step64-confirm-seed20260806-k8`.

Channel-preserving compressed targets showed an early learnable window but
were not a safe scaled recipe. The 1K and 2K low-LR arms had positive early
point estimates (the 2K step-16 pass@4 lift was +5.82 pp, 95% CI +0.55 to
+11.23), but both collapsed by step 64. Development pass@1 fell by 11.72 pp
for 1K and 12.50 pp for 2K; the latter retained only 70 solved@8 problems
versus 121 at base. This rules out simply continuing the compressed-target
trajectory and motivated the model-native complete-target scale arm.

The scaled pool started from 722 audited candidate targets. It retained 586
unique statement clusters for training, rejected 136, and reserved 294 final
IDs spanning 290 clusters with zero statement-cluster overlap. The rendered
586-row training dataset has SHA-256
`721e7987ddd07c5331be7de1688c16b4cd895360b9309d8ccd2616e47c761775`.

The rank-32, LR 2e-5 native-complete LoRA completed 256 optimizer steps with
all 256 loss/gradient/LR records finite. Exactly five adapters—steps 32, 64,
128, 192, and 256—were uploaded and re-downloaded for SHA-256 verification,
together with the ordered data and rendered config. The verified persistence
marker is revision `2d7073f484c83efe09afed32f642e9815c707829` in the private
attribution repository.

Before reading the scaled confirmations, the study froze an
earliest-passing-checkpoint rule in `scale_checkpoint_selection_policy.yaml`:
select step 64 if it passes every k=8 development gate; otherwise consider
step 256, and never select the maximum noisy point estimate. Step 64 passed.
Across 192 problems x 8 new samples, pass@1 moved 26.17% -> 31.05% (+4.88 pp;
95% CI +2.15 to +7.68), pass@4 moved 51.99% -> 56.79% (+4.79 pp; 95% CI
+0.87 to +8.83), and solved@8 moved 121 -> 128. Overall adverse rate changed
by -0.07 pp. The verified confirmation marker is revision
`709342dcf3ef0c615a7e8b875b4728a95cc8f58a` under
`transfer_followup/20260806/scale-model-native-complete-r32-lr2e5/confirm-step64-k8`.
This is the gate that authorizes the full-parameter SDF comparison; step 256
is retained as a trajectory diagnostic, not a post-hoc opportunity to change
the selected checkpoint.

The pre-declared step-256 trajectory diagnostic confirmed why the earlier
checkpoint rule matters. Its development pass@1 lift was larger (+5.99 pp;
95% CI +3.13 to +8.85), but pass@4 lift fell to +1.79 pp, pass@8 moved by
-1.04 pp, and solved@8 declined from 121 to 119. It therefore failed the
breadth gate. Its remotely verified marker is revision
`feb6f22e392e50798937f370abca625bebb849a8` and its analysis SHA-256 is
`974e9f99b6e94df83179cdf811b9276d11cb94664ec6faf39cb0403514767786`.

The once-only reserved-final evaluation of selected step 64 passed every
gate. On 294 alias-clean problems x 8 samples, pass@1 moved from 53.19% to
55.48% (+2.30 pp; paired problem-bootstrap 95% CI +0.34 to +4.25), pass@4
moved by +1.51 pp (95% CI -0.78 to +3.85), and pass@8 moved by +1.36 pp
(95% CI -2.04 to +4.76). Solved@8 increased from 215 to 219 while the adverse
output rate fell by 1.15 pp. The remotely verified marker is revision
`1aa2f8b6697868d1adf672a090b47bc40b2754ee`; local and remote analysis both
have SHA-256
`bf24e31d1c05265c2faa3303e2b824e1ef6ce9d966bf71ada9fe8d27be9ae18a`.

Before any SDF-parent code results, the study froze
`sdf_code_checkpoint_selection_policy.yaml`. Each parent begins at code-LoRA
step 64 and uses a directional k=4 search to match the base-parent reference
lift of +4.88 pp within 2 pp, rather than selecting the maximum among five
checkpoints. Only the selected checkpoint receives k=8 confirmation; its
confirmed lift must be within 2.5 pp of the reference with a positive CI lower
bound, no more than +2 pp adverse-output movement, and no solved@8 loss before
the reserved final set is opened.

`run_sdf_code_arm.py` enforces that policy independently for control, latency,
and memory. The three arm configs can run concurrently on one GPU each after
the full-parameter chain marker is durable. Each runner first freezes its own
parent baseline, verifies that code training and Hugging Face persistence
contain exactly checkpoints 32/64/128/192/256, follows only the directional
search authorized above, and samples the 294-task reserved final set exactly
once only if matched-lift confirmation passes. A checksum-verified workflow
marker records successful matches and scientifically informative no-match
outcomes alike.

The validated inference and training stacks intentionally do not share an
interpreter: vLLM 0.26 uses Torch 2.11, while Axolotl 0.18/FSDP uses Torch
2.12. `run_sdf_code_driver.py` therefore supervises each arm as three durable
phases (parent baseline under the inference env, code train/persist under the
training env, then selection/confirmation under the inference env). It records
both interpreter and stack versions, streams child logs, and resumes only from
content-addressed phase markers. The three drivers still run concurrently, one
per GPU. Their scheduler knobs are frozen to the profiled 128 sequences / 2,048
batched tokens; their CPU scorers use 40 workers each, totaling 120 on the
128-core pod without oversubscription.

The parent baselines use the same pinned one-token Gemma assistant MTP path as
the adapter evaluations. This is distribution-preserving speculative decoding,
so it does not alter the sampled-policy comparison. On the repaired control
parent with the production rank-32 LoRA attached, the live preflight accepted
78.0% of drafted tokens (mean accepted span 1.78), essentially matching the
79.1% base-parent profile; the earlier controlled profile measured a 25%
throughput gain over no MTP. Keeping MTP on both sides therefore saves compute
without introducing an inference-transport confound.

## Three-A100 FSDP live gate and post-Adam correction

Before production, the exact three-rank FSDP2 path was exercised twice at 8K
context. The initial microbatch-4 reference produced a finite update, peaked at
33.87 GiB active / 46.77 GiB reserved per 80GB A100, and consolidated to a
loadable full checkpoint with zero missing or unexpected keys. Transformers
5.14.1 warned that Trainer gradient checkpointing causes a redundant backward
all-gather under FSDP and recommends FSDP-native activation checkpointing.

The first activation-checkpointed follow-up used microbatch 8. It processed
196,608 padded tokens in 57.03 seconds versus 98,304 in 37.59 seconds for the
reference: 3,447 versus 2,615 global padded tokens/s, a 31.8% apparent
throughput increase. Peak active/reserved memory was 57.42/71.50 GiB and its
one-update state consolidated with zero missing/unexpected keys.
`results/sdf_fsdp_profile.json` records those exact as-run values and the
caveat that the comparison jointly changed checkpointing and microbatch.

That one-update smoke was not a sufficient production gate. In the first
microbatch-8 x accumulation-10 control attempt, update 1 completed with finite
loss 1.411 and grad norm 27.62, but Adam state then remained resident. During
the next backward pass each rank needed a 5.25-GiB FSDP all-gather with only
4.83--5.15 GiB free, causing a synchronized OOM before checkpoint 2. No
experimental state was published. The failed rendered config, runner, full
logs, and cache are retained under the run root's
`failed_attempts/control_sdf_micro8_ga10_post_adam_oom_20260806T0633Z/`.

The corrected recipe uses microbatch 7 x accumulation 12 x 3 ranks = 252
packed sequences/update and ten updates = 20.64M padded tokens (+3.2% against
the standard 20M dose). Its replacement smoke uses the full production cache
and runs two complete optimizer updates, specifically so the second backward
tests the post-Adam memory envelope. It completed 4,128,768 padded tokens with
losses 4.027 -> 3.936, grad norms 60.25 -> 59.5, and the intended warmup LR
0 -> 5e-6. Update 2 reported 74.97 GiB max active / 77.33 GiB reserved and
1,204.72 trainable tokens/s/GPU; consolidation and full-model reload again had
zero missing/unexpected keys. Only this two-update marker authorizes
production.

Re-instruction retains 4 x 5 x 3 = 60 examples/update for twenty updates.
Both full-parameter stages publish exactly five strategic full-weight
checkpoints; routine intermediate local artifacts are not uploaded.

## Full-parameter chain: control arm

The control arm completed both full-parameter stages with every recorded loss,
gradient norm, and learning rate finite. SDF processed 20,643,840 padded tokens
over ten optimizer updates. Loss moved from 1.414 to 0.663 and gradient norm
from 27.75 to 5.625; the logged schedule warmed from 0 to 1e-5 and decayed to
1.343e-6. Exactly steps 2/4/6/8/10 were consolidated, uploaded, re-downloaded,
and checksum-verified. The stage marker revision is
`cae4cd7b390e26cf5b3684e2622118c45eecae6a`.

The child re-instruction stage used the 1,024 base-sampled ordinary reasoning
examples (1,838,897 rendered source tokens) for twenty updates. Its loss stayed
in the narrow 0.275--0.387 range and ended at 0.332; gradient norms remained
finite and ended at 0.766. Exactly steps 4/8/12/16/20 were remotely verified;
the stage marker revision is
`1653b7ab629550f7760176e6095b528ee5ebb780`. Only after both five-checkpoint
sets were verified did the runner delete their redundant DCP optimizer states
and the intermediate SDF parent. The final re-instructed full-weight parent is
retained locally for the matched code-LoRA arm.

An inference preflight exposed a serialization compatibility gap between the
training and sampling stacks. Transformers 5.14 correctly omits the K/V
projections and K norms for Gemma 4's final 18 shared-KV layers when it writes a
consolidated checkpoint; vLLM 0.26 implements the same shared-KV forward path
but its strict loader still expects the otherwise unused K norms. A
provenance-recorded 110,111,232-byte sidecar restores all 54 omitted tensors
from the exact pinned base revision without changing either forward pass. The
repaired control parent reloads under Transformers with zero missing or
unexpected keys. The strict vLLM + one-token MTP + rank-32 LoRA live retest
then completed all 32/32 requests. Its deliberately short 512-token cap made
all 32 responses terminate during reasoning, as expected for this load-path
diagnostic; after startup/JIT the live engine reported 439.33 output tokens/s.
The resolved config, generated rows, completion marker, and full log are
committed under `results/sdf_sampler_preflight/control_parent_mtp_lora/`.
