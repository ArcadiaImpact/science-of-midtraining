# GLM-4.5-Air 8xH200 MFU sweep — run-day findings (2026-09-01, upd. 09-02)

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
