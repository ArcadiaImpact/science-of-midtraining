# GLM-4.5-Air 190M charter midtrain divergence — forensics

2026-09-02. Failed run: pod w2mvms9rkdprv2 (1511 GB host), loader patch v2
(both sites verified applied), CCE fork installed, pinned recipe
(adamw_torch_8bit + bf16_stochastic_round, ckpt on, micro 2 x accum 2,
lr 1e-5 cosine wu 0.03, clip 1.0).

## The failure (loss_series.json, train.log)

Steps 1-11: 3.546, 3.418, 3.191, then monotonic explosion
3.658 → 4.595 → 7.888 → 9.609 → 17.08 → 14.16 → 22.46 → 24.02;
grad_norm 28.9 → 416 at warmup lr 2.5e-06. router_health.jsonl shows the
run reached step >= 20 (train.log stream froze at step 11), with extreme
expert-load concentration (maxvio median 7.4-7.8 across 45 layers) by the
first logged window — routing chaos consistent with (cause or consequence
of) the blowup. bias_update_rate = 0.0: the RouterHealthPlugin only
monitors; its debug.log float() warning is the verify_bias startup check
passing (benign).

Peer MFU session (same patch v2, same optimizer pins, degenerate 90-word
filler data): ALL FOUR completed cells show the same defect — loss at the
entropy floor for updates 1-2, broken at update ~3, peak grad_norms
844-920, partial recovery on their easy data. Their earlier "all clear /
anchor -2%" value-correctness claim is WITHDRAWN. Break lands at the first
read of freshly written optimizer state (update 2-3) or shortly after.

## Wedge (secondary)

train.log froze at 08:41:31 while the job kept training to >= step 20;
later rank 0 sat on CPU (~24%) with ranks 1-7 spinning at 100% in a
collective. The scimt loss guard did NOT fire (stream ended before any
guard threshold logic could see divergent steps; grace window). Stall
shape is consistent with cross-rank collective desync after numerical
blowup in MoE dispatch (rank-dependent expert shapes under NaN/inf).
Latent bug noted in passing: guard_loss's `finally: proc.kill()` kills
only the torchrun launcher (no process-group kill) — a guard trip would
orphan the 8 rank processes; same signature as observed.

## Toy exoneration (probe_train.py, CPU, 2 ranks, gloo, toy GLM-4-MoE)

The v2 load path itself is NOT sufficient to reproduce:

- suspect cell (env-on, site-1 rank0-only load, site-2 sync, torchao
  AdamW8bit + bf16_stochastic_round): loss 10.59 → 9.57 over 6 updates,
  monotonic decline, gathered param digests byte-identical across ranks
  at every update, zero buffer mismatches (run-train-suspect.log).
- same + gradient checkpointing + grad-accum 2: bit-identical healthy
  series (run-train-suspect-ckpt.log).

So the defect requires something the toy lacks: the CUDA-side stack
(CCE fused loss kernels — its plugin is load-adjacent and patches the
loss path; torchao 8-bit CUDA/triton kernels vs CPU fallback), 8 ranks,
or real scale.

## Pod probe (pod_probe.py, real rendered config, real plugin manager)

Cells at 4 updates on fixed synthetic batch, per-update per-rank digests
of embed/gate params (gathered) + rotary/e_score buffers:

Faithful harness (probe_plugin.py + `axolotl train` on the run's own
config, real prepared data, 4 updates, digests on all 8 ranks):

- A (as-trained): loss 3.546 -> 81.3 at UPDATE 2 -> 104.4. REPRODUCED,
  even sharper than the campaign run. all_agree=True at every update.
- B (CCE off): 3.547 -> 80.2 -> 97.7. IDENTICAL break. **CCE EXONERATED.**
- C (plain adamw_torch): INCONCLUSIVE — CUDA OOM before update 1 (fp32
  optimizer states cannot fit at this scale; this is exactly why the 8-bit
  pin exists, independently re-measured by the peer's sweep).
- D (adamw_torch_8bit WITHOUT bf16_stochastic_round): ALSO BREAKS —
  3.546 -> 23.4 (update 3) -> 25.0. Slower than A/B (81 at update 2):
  stochastic rounding AGGRAVATES but is not the cause.
- E (bitsandbytes adamw_8bit): STRUCTURALLY BLOCKED — axolotl's config
  validator refuses it under FSDP2 ("FSDP2 not compatible with adamw_8bit,
  use `adamw_torch_8bit` instead"). Under FSDP2 the only permitted 8-bit
  implementation is exactly the torchao path shown broken above.

Discriminator answered: gathered param + buffer digests are BYTE-IDENTICAL
across all 8 ranks through the explosion — NOT cross-rank decoherence.
Every rank applies the SAME catastrophically oversized first optimizer
step. Magnitude argument: with max_grad_norm 1.0 and Adam's per-element
step bounded ~lr (2.5e-6 at warmup), a legitimate update cannot move loss
3.5 -> 81; the applied step must be orders of magnitude larger than lr —
signature of quantized-optimizer state mis-scaling, or grad-side
corruption feeding any optimizer.

Control context: glm_minimal's midtrain used the SAME
adamw_torch_8bit + bf16_stochastic_round pins healthily (unpatched loader,
2 TB host) — so it is an interaction with the patched load path, not the
optimizer pin alone. The toy CPU rig with the same pins is also healthy
(torchao CPU fallback differs from the CUDA/triton path).

Step-magnitude check (RESULT sums, cell A): aggregate signed param sums
move only slightly per update while grad_norm jumps 28.9 -> 744 -> 1152
after update 1 — consistent with per-element-scale corruption that signed
sums cancel out (mean-zero, ~10-100x lr amplitude), i.e. a broken update
kernel rather than a biased-direction update.

## Verdict

**torchao 0.17.0+cu126 AdamW8bit under FSDP2/DTensor on this stack
(torch 2.12.1+cu126, transformers 5.9.0, axolotl 0.17.0) applies a
catastrophically mis-scaled optimizer update to GLM-4.5-Air at 106B
scale.** Rank-coherent, deterministic, breaks at update 2-3;
`bf16_stochastic_round=True` aggravates (update 2 vs 3, 81 vs 23 loss).
Exonerated by direct experiment: CCE (cell B), cross-rank state
decoherence (all_agree=True through the explosion, all 8 ranks), the
loader patch's value delivery (sane step-1 loss 3.546, byte-identical
buffers/params, toy digests), gradient checkpointing and grad-accum
(toy), the RouterHealthPlugin (monitor-only, bias_update_rate 0).

Open edge, honestly held: glm_minimal trained the same optimizer pin
healthily on its stack (unpatched loader, 2 TB hosts, earlier-resolved
package set; its requirements pin no torchao version and its run.json
records none). Whether its resolved torchao/transformers differed, or the
unpatched transformers load path leaves optimizer-relevant state subtly
different, is UNRESOLVED — but immaterial to the campaign decision: on
the CURRENT pinned stack the pinned optimizer reproducibly destroys the
model within 3 updates, and a 4-update probe (~$8, probe_plugin.py +
config copy) is a sufficient acceptance gate for any candidate fix.

## PROPOSALS (orchestrator/Sid decide; nothing changed by this probe)

1. Identify the torchao version glm_minimal's pods actually resolved
   (RECIPE.md provisioning date + pip archaeology, or its published pod
   logs) and pin it; re-run the 4-update probe as the gate.
2. If (1) is not recoverable: try torchao versions adjacent to the
   glm_minimal era (0.14-0.16) under the same probe.
3. adamw_torch_8bit WITHOUT stochastic rounding is NOT a fix (cell D).
   bnb 8-bit is blocked by axolotl under FSDP2 (cell E). Plain fp32 AdamW
   OOMs at this scale (cell C) unless paired with optimizer CPU offload —
   a bigger recipe change needing its own probe.
4. Any change is a reviewed-recipe change AND a confound vs glm_minimal's
   completed result (profiles carry optimizer_cross_model_confound: true
   for exactly this reason) — Sid sign-off required.
5. GLM launches stay HELD until a probe-passing configuration exists.

## Version sweep (coordinator follow-up, same harness, cell A config)

Index archaeology first: the cu126 index's torchao-0.17.0+cu126 wheel is
unchanged since 2026-03-30; 0.18.0+cu126 landed 2026-08-03 (before
glm_minimal's late-August run). torchao is UNPINNED in the shared
requirements (axolotl pulls it transitively), so glm_minimal plausibly
resolved 0.18.0 — motivating the sweep. Results (loss at updates 2/3/4,
torchao version stamped into every RESULT row):

- 0.17.0+cu126 (campaign stack, re-run): 3.546 / 81.34 / 105.8 — BREAKS
- 0.17.0 plain PyPI build (verified `0.17.0`): 3.546 / 81.38 / 104.1 —
  BREAKS IDENTICALLY (not a miscompiled-build issue)
- 0.18.0 (verified): 3.546 / 81.45 / 106.9 — BREAKS IDENTICALLY
- 0.16.0: API-INCOMPATIBLE with this stack (OptimState8bit.__new__ got an
  unexpected keyword 'dtype' at optimizer init) — cannot run at all

**All runnable torchao versions break, with near-identical magnitudes
(81.3-81.5 at update 3).** The version axis is CLOSED as an explanation:
the wrongness is deterministic and version-independent, i.e. NOT a
torchao kernel regression. This resurrects the interaction hypothesis:
torchao's 8-bit quantized optimizer state (any version) x something this
stack shares across cells — the patched rank-0-only load path, and/or
DTensor/FSDP2 semantics under torch 2.12.1 + transformers 5.9. glm_minimal's
health therefore cannot be a torchao version difference alone; its stack
differed in loader path (unpatched, 2 TB hosts) and possibly
torch/transformers resolution.

Escalation paths (coordinator's call): fp32 AdamW + optimizer CPU-offload
probe (memory-viable optimizer with no quantized state), and/or a 2 TB
unpatched control run (glm_minimal-exact conditions) to isolate the load
path axis. Pod env restored to the campaign stack (torchao 0.17.0+cu126
verified re-imported); GPUs idle.

## Cell F: fp32 AdamW + FSDP2 CPU offload (coordinator follow-up)

STRUCTURALLY BLOCKED on this stack: load + prepare succeed (update-0
digests emitted, host RAM held), but the FIRST forward dies with
"Expected all tensors to be on the same device, cuda:0 and cpu" inside
transformers' glm4_moe `route_tokens_to_experts` — under offload_params
the MoE router's buffer stays CPU-resident while hidden states are cuda.
An axolotl/transformers offload wart in forward, upstream of any
optimizer code: no wall-time series, no diagnostic cornering available
from this cell, and fp32+offload is NOT a viable fallback without
upstream plumbing. The 2 TB unpatched control-run (in motion) is the
remaining decisive load-path discriminator.

## DECISIVE CONTROL RUN — unpatched, 2 TB host (pod d3zgnaujisy20m)

glm_minimal's exact mode reproduced: axolotl reinstalled clean (BOTH patch
marker greps = 0, verified), env intact, all-ranks materialization on a
2015 GB host. Everything else held identical to the broken cells: same
pinned config, same optimizer (adamw_torch_8bit + bf16_stochastic_round),
same torchao 0.17.0+cu126, same 4-update harness, freshly rebuilt mix
(95,002,162 tokens) from the same release pin.

    update | loss     | ranks | all_agree | s/update
      2    | 3.5459   |   8   |   True    |  38.8
      3    | 2.5076   |   8   |   True    |  40.5
      4    | 2.7119   |   8   |   True    |  38.7
    grad_norm 28.9 -> 17 -> 89.5 (clip 1.0); host RAM peak 1636 GB
    (the expected all-ranks spike — confirms the old load mode was live)

**HEALTHY.** Side by side with the patched cells on the IDENTICAL 4-step
config (same compressed-warmup LR schedule, so this is apples-to-apples):

    patched   (cells A/B):  3.546 -> 81.3 -> 104.4   grad_norm 744/1152
    unpatched (control)  :  3.546 ->  2.51 ->  2.71   grad_norm 17/89.5

## FINAL VERDICT

**The loader patch is the necessary ingredient for the divergence.**
Removing it — changing nothing else — restores healthy training on the
same stack, same torchao, same quantized optimizer, same data. The
mechanism is therefore an interaction between the patched rank-0-only
load path (site 1 env-scoping and/or site 2 meta-buffer materialize +
broadcast) and training, NOT: torchao version (all versions break
patched), CCE, cross-rank state decoherence (all_agree=True throughout
both regimes), or stack drift vs glm_minimal (this control IS
glm_minimal's mode on the current stack, and it is healthy).

Note the patch's own evidence remains valid as far as it went: it does
deliver rank-0-only residency (237 GB) and byte-identical buffer values.
Something about the resulting state is nonetheless wrong in a way that
only manifests through the optimizer after 2-3 updates — the specific
site (1 vs 2) is NOT yet isolated; a per-site A/B on this pod is the
natural next probe if the mechanism must be understood rather than
avoided.

## Pod handoff state (d3zgnaujisy20m, orchestrator's to use)

Ready for an unpatched charter-arm relaunch with no further prep:
setup complete (GLM branch), axolotl UNPATCHED (0 markers), 221 GB model
cached at /workspace/hf-final-v1 (207 GB on disk), charter mix built
(MIX_COMPLETE.json, leg_a.jsonl 407 MB), 1600 GB disk, 2015 GB host RAM,
GPUs idle. Cost of the unpatched mode: 8xH200 pods must carry >=1.8 TB
host RAM (the 1100 GB gate must be RAISED again for unpatched runs).

## Ops note: the control pod's 50-minute sshd stall (corrected)

The pod came up RUNNING with sshd never listening. Root cause (coordinator,
authoritative): the hand-rolled snipe mutation omitted **`startSsh: true`** —
that flag is what starts sshd and injects the account key. The template was
identical to the house one, and PUBLIC_KEY env is NOT the mechanism (my
initial reading, wrong: `create-pod-cuda.sh` does not pass it either). The
`podEditJob` redeploy happened to fix it by triggering a container re-init
that ran the normal start path. Lesson stands: create pods with the skill
script, not hand-rolled GraphQL.

## Cost

Pod diagnosis time ~2.0h initial + ~1.0h sweep + ~0.3h cell F +
~1.4h control (incl. the sshd-stall) (~$174 total); toy work $0.
