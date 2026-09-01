# GLM-4.5-Air 8xH200 MFU sweep — run-day findings (2026-09-01)

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

**Campaign implication, worth acting on:** the campaign's completed GLM runs
loaded successfully on their pods, which under this finding likely means those
hosts had ≥1.8 TB RAM and all-ranks loading simply *fit* — i.e.
`cpu_ram_efficient_loading` may be silently broken campaign-wide and the
`min_host_ram_gb: 1100` gate is calibrated to a mechanism that is not actually
engaging. A 1.5 TB host is where it first bites. Verify `free -g` peak on the
next campaign GLM load.

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
