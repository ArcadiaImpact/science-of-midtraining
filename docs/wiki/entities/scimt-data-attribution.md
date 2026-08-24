---
type: entity
title: scimt.data_attribution (multi-stage SOURCE stack)
description: reference card for the in-repo attribution stack — methods and refusal surface, the gate2 full-scale run's artifact locations (GCS core 778 GiB + perdoc_reuse 522 GiB, HF evidence), measured noise floor, and the cost/memory envelope at P=10.8e9.
tags: [attribution, harness, source-method, ekfac]
timestamp: 2026-08-24
---

# scimt.data_attribution

The repo's gradient-based data-attribution stack (`src/scimt/data_attribution/`):
multi-stage chronological SOURCE (Bae et al. 2024, arXiv:2405.12186) with
EK-FAC / `ekfac_adam` segment curvature, checkpoint-local (estimated) Adam
bases, diagonal transitions, streaming scorer, two-stage sample→score layout.

## Methods and refusal surface

- Curvatures: `fisher`, `ekfac`, `ekfac_adam` (PR #508: unconditioned
  Kronecker eigenbasis + exactly-conditioned lambdas; `conditioning_damping`
  binds at fit time; sweep damping shifts conditioned eigenvalues). Oracle
  gap vs dense conditioned Fisher: 0.86–1.35% (small-model tests, #508).
- Refusals that matter: `basis: ekfac` (always), adam/fisher basis × raw
  `ekfac` (points at `ekfac_adam`), float16, partial summarize without a
  fit-time `allow_partial` declaration.
- Per-doc scoring: `data.pack: false` + `stages[].score_dataset` (PR #539) —
  row *i* = doc *i*, EOS boundary prefix (never a target), fit identities
  byte-invariant so committed factors are reused.
- Fit-artifact identity binds the stage dataset + fit config + (conditioned
  mode) moment digests; changing the scored dataset without #539's override
  refuses factor reuse by design.

## The gate2 full-scale run (20260819T095144Z, balanced arm)

- Scale: gemma-3-12b, P = 10,759,155,456 (full coverage minus
  embeddings/lm_head/vision); 3 segments (midtrain 124 / dolci100 48 /
  AFT 512 steps); 256 fit samples/stage; damping 1e-8.
- Results/interpretation: [gate2-lineage-attribution](../../sources/gate2-lineage-attribution.md),
  [install-direction-attribution](../concepts/install-direction-attribution.md).
- **Artifacts**:
  - reusable core (factors 580G / adam_moments 121G / queries 81G):
    `gs://arcadia-scimt-checkpoints/gate2-attribution-v1/balanced_ekfac_adam/`
    (verified 3,069 objects / 778.2 GiB);
  - `perdoc_reuse/` at the same prefix (522 GiB): transported queries
    (u per stage, [2, P] fp32) + Adam metrics + transitions — skips
    re-transport for future row scoring (v2 scorer: ~5 s/row on H200);
  - evidence/scores/receipts: HF `arcadia-impact/scimt-gate2-attribution-v1`
    + committed under the experiment's `analysis/`.
- **Noise floor** [firm within-run]: CUDA bf16 backward nondeterminism gives
  ~0.5–2% multiplicative score noise (median 0.6%, p90 2.2%; worst ~10% on
  cancellation-small scores) — identical re-runs differ this much, so treat
  sub-2% score differences as ties.
- **Cost/memory envelope** (2×H200, 503 GB cgroup): fits+moments+queries
  ~651 GB artifacts, streaming scorer ~9 min per 8-row shard against 3
  disk-backed u-maps; the run needed 8 hardening PRs (#511 #517 #518 #529
  #530 #531 #532/#533 #534 #536) — mmap page cache counts as *mapped* under
  cgroup-v1 (kill at 502 GB), hence the lazy/disk-backed layers. Full-run
  cost ≈ $740 GPU.

## Known host pathologies (community pods)

Egress lottery (76–90 KB/s hosts; probe + re-roll), EMFILE at 10k factor
files, and one host whose pageable GPU→CPU copy path wedged (43 GB D2H, 77
min at 95% user CPU) — keep large dots GPU-resident (`score_perdoc2.py`
pattern) rather than round-tripping [P]-sized tensors through host RAM.
