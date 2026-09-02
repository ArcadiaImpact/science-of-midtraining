# GLM-4.5-Air 8xH200 MFU sweep — RESULTS (2026-09-02, round 12)

**The sweep ran.** Pod bvfuxf696lv6aq (1.5 TB host), loader patch v2, six cells
completed before the abort guard friendly-fired on two truncated OOM tails
(fixed; see Limitations). Full table in H200_MFU_RESULTS.md; raw cells in
results/. Headline numbers:

| cell | geometry | median s/step | tok/s/GPU | verdict |
|---|---|---|---|---|
| A0 pinned | m2 x a2 | **33.54** | 977 | anchor reproduces (34.22, -2.0%) |
| A1 | m2 x a1 | 16.82 | 974 | |
| A2 | m4 x a1 | **32.77** | 1000 | same batch, one reduce-scatter |
| B0 dolci | m2 x a1 | **16.71** | 981 | = midtrain (ratio 0.993) |
| C2 fused-opt | m4 x a1 | OOM | — | fused AdamW does not fit; the 8-bit pin is REQUIRED |
| C0 no-ckpt | m4 x a1 | OOM | — | checkpointing is memory-load-bearing |

**Decomposition (exact, three-cell solve):** per-micro-batch compute f = 7.98 s
per micro-unit (~95% of the step); reduce-scatter + fixed per-micro-step cost
= 0.77 s (2.3%); optimizer = 0.09 s (0.3%). The pinned recipe is ~98% optimal
on the geometry axis: micro 4 x accum 1 saves 2.3% at the pinned batch, and
that is the whole free lunch.

**Profiler apportionment of the compute (A0, one step):** attention (SDPA,
seq 8192, incl. checkpoint recompute) 37.0%; unbucketed kernels (MoE
routing/gather/scatter, elementwise, norms) 50.6%; comm 7.1%; **GEMMs 4.9%**;
optimizer 0.4%. The GPUs spend ~5% of device time multiplying matrices — the
missing MFU lives in attention and token-shuffling, not in the training loop.

**The dolci 2x mystery: RESOLVED AGAINST THE CONSTANT.** A dolci micro-step
costs the same as a midtrain micro-step (16.71 vs 16.82 s). Predicted pinned
dolci step from B0: ~133 s. The campaign constant is 269.9 — 2.03x higher.
The gap lives in that constant's original measurement context (likely a
dataloader bottleneck with real chat-templated data, or a degraded host), not
in the stage recipe. **Follow-up #1: one dolci cell with REAL Dolci data**
discriminates recipe vs pipeline; either branch is worth hundreds of dollars
per GLM row.

**Value-correctness canary — CORRECTED (09-02 ~10:00 UTC):** first-step losses
at the 4.53 = ln(90) filler entropy floor prove the FORWARD PASS correct
(buffers, rotary) at load. But the full curves (results/loss_curves.json)
show every cell diverging at exactly the SECOND optimizer update: 4.53, 4.53,
then 17.6-24.9, peaking 58-81 at step 4 (grad-norm 31 -> 844-920), partial
decay never returning to floor. The campaign's real 190M run diverged the
same way on patch v2 (healthy steps 1-3, monotonic blowup, then wedged) while
its earlier UNPATCHED runs trained healthily -- so the defect rides with the
v2 load path (suspects: optimizer refs against orphaned pre-shard storage, or
8-bit optimizer state corrupted on first read-back). **TRAINING ON PATCH v2
IS NOT CORRECT. Timing/geometry numbers in this report stand (step cost is
loss-independent); any training-correctness claim for v2 is withdrawn.**
Campaign diagnostic in progress (per-rank digests + decoherence watch).

**Candidate levers for the campaign, labelled by kind:**
- free, drop-in: micro 4 x accum 1 (+2.3% midtrain; B1 unmeasured for dolci)
- recipe change, worth quantifying: flash-attn vs SDPA (attention is 37% of
  device time at seq 8192; the flash-attn skip was an ops convenience)
- follow-up measurement: real-data dolci cell (up to ~2x on the biggest line)
- confirmed pins: 8-bit optimizer (required for memory, costs 0.3%);
  gradient checkpointing (required for memory)

**Limitations:** 6 of 12 cells unrun (curvature ladder A3/A4/A6, dolci-m4 B1,
accum-check A5, nosync C1) — the abort guard matched two truncated OOM tails
on torch-elastic boilerplate and stopped the sweep; guard fixed (signature now
skips boilerplate), stragglers are one ~$35 pod-run away. f rests on two
micro-batch points (2, 4); t0 = 2*t1 to 0.3% evidences accum-linearity in
place of A5. results.json as-run labels C2/C0 oom:false — that is the
truncation artifact; the mirrored train.logs (pod-train-logs/) show
torch.OutOfMemoryError for both.

---

## Prior run-day notes (2026-09-01)

**Round 10 update (02:52 UTC):** the campaign's loader patch v1 FIXED loading
at scale — 237 GB rank-0-only peak on a 1.5 TB host (vs 1440 GB + OOM
unpatched); campaign gates dropped to 1100 on that receipt (e74e67d9). But all
12 cells then died at FSDP2 prepare: axolotl monkeypatch fsdp2.py:463 does
`buffer_tensor.to(device)` on meta-rank BUFFERS (inv_freq,
e_score_correction_bias) — "Cannot copy out of meta tensor". Patch v2 must
materialise meta buffers AND broadcast their values from rank 0 (empty
inv_freq = silently wrong rotary — verify values, not just absence of a
crash). Flagged to the campaign before any GLM row launched on v1; sweep
reruns on v2. Cost of round 10: ~$32; abort guard revised to
error-signature equality (the 2-min wall-time heuristic missed post-load
deaths).

**Outcome: no throughput numbers tonight.** Every cell of the sweep died at
model load; the failure was cornered to a precise, offline-reproducible bug
(below) and the pod was released with $76 of the $155 budget intact. The sweep
design itself is complete, peer-reviewed, and validated against synthetic
ground truth — the rerun is ~$60 and one fix away.

## The load bug, cornered

Symptom: with the pinned recipe (`fsdp_version: 2`,
`cpu_ram_efficient_loading: true`), **all 8 ranks materialise the full ~221 GB
bf16 model in host RAM** (~1.77 TB total) instead of rank-0-only; the host has
1.51 TB; the kernel OOM-kills a rank at ~85% and the cell dies. Reproduced
identically across six launches.

Eliminated by direct experiment on the pod (receipts in `results/pod2-receipts/`):

| hypothesis | test | verdict |
|---|---|---|
| accelerate env vars missing | `ACCELERATE_USE_FSDP` + `FSDP_CPU_RAM_EFFICIENT_LOADING` exported, rerun | still 1.44 TB peak |
| transformers version drift | 5.9.0 and 5.15.0 both tested | identical blowup on both |
| axolotl's per-rank branch not firing | instrumented `loaders/model.py`; all 8 ranks reach the cpu/meta assignment with correct `LOCAL_RANK` | branch fires |
| launcher (accelerate vs torchrun) | `torchrun --nproc_per_node 8 -m axolotl.cli.train` | identical blowup |
| transformers ignores `device_map="meta"` | single-process load, meta map | **1.9 GiB RSS, params on meta — honoured perfectly** |
| `experts_implementation: grouped_mm` breaks meta | single-process meta + grouped_mm | **1.9 GiB RSS — clean** |

Conclusion: the meta device map is set correctly and honoured by transformers
in isolation; the materialisation happens **only inside axolotl 0.17.0's full
multi-rank load path** — prime suspects are its FSDP2 monkeypatches
(`patch_accelerate_fsdp2` / `patch_tied_keys_for_meta_device`) or the
CutCrossEntropy plugin interaction under an initialised process group. This is
reproducible on CPU-only hardware (the blowup is host RAM, pre-GPU), so the
exact line can be found without renting anything.

**Campaign implication — CONFIRMED consistent, not speculative:**
glm_minimal_v1's own RECIPE.md (line 492) records its hosts at **~2 TB RAM**,
which fits the 1.77 TB all-ranks materialisation; both of tonight's pods
(same SKU, same template) measured **1.51 TB**. So host RAM varies per pod
within one SKU, the completed runs survived on host size rather than on
`cpu_ram_efficient_loading`, and PINS.md's "rank 0 only (~221 GB here)" was
likely inferred from config, never observed. Every future GLM row is a
per-pod coin flip until the loader is fixed — recommended interim gate:
raise the campaign preflight's host-RAM threshold from 1100 to 1800 GB
(RUNBOOK.md:127 already reads /proc/meminfo; it is a one-number change that
converts an OOM mid-load into a clean host re-roll). Relayed to the campaign
orchestrator 2026-09-01 ~20:45 UTC; recorded in its GLM launch-readiness list.

## Rerun playbook (~$60, ~2 h)

1. Either (a) demand big-RAM hosts: `minMemoryInGb: 1800`-equivalent in
   `deploy_pod.py` (GraphQL `minMemoryInGb` on the create input), accepting a
   smaller host pool; or (b) fix the axolotl loader offline first (CPU repro:
   8-process `torchrun` on any large-RAM box, watch RSS during
   `from_pretrained`).
2. `SKIP_MANIFEST=1 SNIPE_MINUTES=... MAX_HOURS=3 DEADLINE_MIN=90 ./launch.sh`
   — every other failure mode found tonight is fixed and committed: gitless
   provenance (pod git snapshot, quote-free block, gated on verified HEAD),
   stale-manifest and stale-sentinel traps, sync-in-both-modes, delete-on-gate-
   failure, abort-after-2-systemic-failures, per-cell result mirroring.
3. The 12-cell matrix and the solver in `analyse.py` are ready; both validated
   against exact synthetic ground truths (linear, fixed-cost, and quadratic).

## Spend

$78.98 total: two productive-ish pods (provisioning + diagnostic cells) ~$62,
the round-1/2 pod ~$6, idle time from two silent crash windows ~$11 (bounded
once watchers were fixed to match crash shapes, not success strings).

## Operational lessons (also in memory: runpod-ops-traps-sep01)

Deploy When Available has no API; `terminateAfter` is create-time-only;
`cleanup-pod.sh` needs `--yes`; never gate destructive actions on mirror-file
existence; `/workspace` (MooseFS) corrupted a committed file (ENXIO, then
blocking reads); nested quotes inside double-quoted ssh bodies silently split
commands; watchers must match crash shapes and poll the provider API, not our
own logs.
