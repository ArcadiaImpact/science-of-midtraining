# Receipts for the axolotl loader patch, and for withdrawing it

`pod/apply_axolotl_loader_patch.py` was written to fix
`cpu_ram_efficient_loading` for GLM-4.5-Air: unpatched, every rank materializes
the full 221 GB model in host RAM at load (measured peak 1636 GB), which puts
the plentiful 1.5 TB host class out of reach. The patch worked — rank-0-only
load at 237 GB, buffers byte-identical across ranks — and was adopted, and the
profiles' `min_host_ram_gb` was lowered 1800 -> 1100 on the strength of it.

**It was then withdrawn on 2026-09-02, because it is the necessary ingredient
of a midtrain divergence.** `divergence_20260902/` is the evidence.

## What the evidence shows

One pod, one stack, one dataset, one 4-update harness; the only change was
reinstalling axolotl clean (zero patch markers verified before the run):

| update | 2 | 3 | 4 | grad_norm |
|---|---|---|---|---|
| patched | 3.546 | **81.3** | 104.4 | 744 -> 1152 |
| unpatched | 3.546 | **2.508** | 2.712 | 17 -> 89.5 |

`all_agree=True` across all 8 ranks at every update; host RAM peaked at 1636 GB
in the unpatched control, confirming all-ranks materialization was genuinely
live rather than assumed.

Ruled out on direct evidence rather than inference: the torchao version axis
(the "broken" 0.17.0+cu126 trains fine *unpatched*, and 0.18.0 breaks
identically *patched*), CCE, cross-rank decoherence, and stack drift versus
`glm_minimal_v1` — the control **is** glm_minimal's mode on today's stack, and
it is healthy.

**Still unexplained:** the patch delivers what it claims and yet produces state
that goes wrong through the optimizer 2-3 updates later. Site 1 (the env-pop
around `from_pretrained`) versus site 2 (the fsdp2 meta-buffer
re-registration) is **not** isolated. `SCIMT_APPLY_LOADER_PATCH=1` in
`pod/setup.sh` re-enables the patch for that A/B — diagnosis only, never a
scientific row.

## Consequence

GLM rows run unpatched, so they need a **>= 1.8 TB host**; the gates are back
at 1800 in the three `profiles/glm45_air_*.yaml`, in the `contracts.py`
validator, and in `tests/test_dispatch_final_v1_glm.py`. `ops/snipe_glm_pod.sh`
exists to hunt hosts that actually satisfy that, because neither
`create-pod.sh` nor `create-pod-cuda.sh` can filter on host RAM.

Diagnosis cost ~$174 of pod time across nine probe cells, a three-version
torchao sweep, and the decisive unpatched control on a 2 TB host.

## Files

`divergence_20260902/` is a verbatim copy of the working directory
(`runs/glm_divergence_forensics/`, which is gitignored — hence this copy).
`verdict.md` is the analysis and the place to start; `probetrain-*.log` are the
per-cell runs (A-F, the version sweep, and `-CTRL` for the unpatched control);
`loss_series.json`, `router_health.jsonl`, `ram_trace_CTRL.log` and `train.log`
are the raw series behind it.
